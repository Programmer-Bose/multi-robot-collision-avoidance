"""
Pygame replacement for live_sim.py.

Same DE-MPC closed-loop / open-loop toggle as before, but rendered with
Pygame instead of matplotlib for much higher frame rates and to serve as
the foundation for Phase 5 (mouse-based obstacle placement) and Phase 6
(start button / setup vs. running mode) -- both of which need real event
handling that matplotlib's animation API doesn't give you cleanly.

Run:
    python3 pygame_sim.py
Edit the CONFIG block below to change settings.

Controls:
    SPACE - pause/resume
    ESC / close window - quit
"""

import sys
import numpy as np
import pygame

from env import make_default_scenario
from de_mpc import DEMPCPlanner

# ----------------------------- CONFIG -----------------------------
RECEDING_HORIZON = True
SEED = 666
N_STATIC = 15
N_DYNAMIC = 8
N_TASKS = 10
CLOSED_LOOP_HORIZON = 10
OPEN_LOOP_HORIZON = 70
MAX_STEPS = 2000
OPTIMIZER = "de"
WARM_START = True

WINDOW_SIZE = 800
MARGIN = 40
FPS = 60
# --------------------------------------------------------------------

# colors (RGB)
BG = (250, 250, 250)
DEPOT = (20, 20, 20)
TASK = (30, 140, 60)
STATIC_OBS = (130, 130, 130)
DYNAMIC_OBS = (220, 90, 70)
ROBOT = (40, 90, 220)
PATH = (40, 90, 220)
TEXT = (20, 20, 20)
PANEL_BG = (255, 255, 255)


def build_env_and_planner():
    env = make_default_scenario(seed=SEED, n_static=N_STATIC, n_dynamic=N_DYNAMIC,
                                 n_tasks=N_TASKS, omega_max=np.pi)
    env.reset()
    horizon = CLOSED_LOOP_HORIZON if RECEDING_HORIZON else OPEN_LOOP_HORIZON
    planner = DEMPCPlanner(horizon=horizon, dt=env.dt, v_max=2.5,
                            omega_max=env.robot.omega_max, robot_radius=env.robot.radius,
                            seed=SEED, optimizer=OPTIMIZER, warm_start=WARM_START)
    return env, planner


class WorldToScreen:
    """Maps simulation world coordinates to pixel coordinates (y flipped)."""

    def __init__(self, world_bounds, window_size, margin):
        xmin, xmax, ymin, ymax = world_bounds
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.window_size = window_size
        self.margin = margin
        span_x = xmax - xmin
        span_y = ymax - ymin
        drawable = window_size - 2 * margin
        self.scale = drawable / max(span_x, span_y)

    def point(self, x, y):
        sx = self.margin + (x - self.xmin) * self.scale
        sy = self.margin + (self.ymax - y) * self.scale  # flip y for screen coords
        return int(sx), int(sy)

    def length(self, world_len):
        return max(1, int(world_len * self.scale))


class PygameSim:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.env, self.planner = build_env_and_planner()

        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 60))
        pygame.display.set_caption(f"DE-MPC live (Pygame) | optimizer={OPTIMIZER}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)

        self.w2s = WorldToScreen(self.env.world_bounds, WINDOW_SIZE, MARGIN)

        self.active_plan = None
        self.plan_ptr = 0
        self.finished = False
        self.collided = False
        self.paused = False
        self.step_count = 0
        self.traj_points = []

    def _get_action(self, obs):
        if RECEDING_HORIZON:
            (v0, omega0), seq, cost = self.planner.plan(
                robot_state=tuple(obs["robot_state"]), goal=tuple(obs["goal"]),
                static_obstacles=obs["static_obstacles"],
                dynamic_obstacles=obs["dynamic_obstacles"])
            return v0, omega0
        else:
            need_new_plan = (self.active_plan is None) or (self.plan_ptr >= len(self.active_plan))
            if need_new_plan:
                self.planner.prev_solution = None
                _, seq, cost = self.planner.plan(
                    robot_state=tuple(obs["robot_state"]), goal=tuple(obs["goal"]),
                    static_obstacles=obs["static_obstacles"],
                    dynamic_obstacles=obs["dynamic_obstacles"])
                self.active_plan = seq
                self.plan_ptr = 0
            v0, omega0 = self.active_plan[self.plan_ptr]
            self.plan_ptr += 1
            return v0, omega0

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
        return True

    def _simulate_step(self):
        if self.finished or self.paused:
            return
        obs = self.env.get_obs()
        v0, omega0 = self._get_action(obs)
        obs, done, info = self.env.step(v0, omega0)
        self.step_count += 1

        if info["reached_goal"] or info["skipped_task"]:
            self.active_plan = None
            self.plan_ptr = 0

        if info["collided"]:
            self.collided = True
            self.finished = True
        if done:
            self.finished = True
        if self.step_count >= MAX_STEPS:
            self.finished = True

        self.traj_points.append((self.env.robot.x, self.env.robot.y))

    def _draw(self):
        self.screen.fill(BG)
        w2s = self.w2s

        # depot
        pygame.draw.rect(self.screen, DEPOT,
                          (*[c - 6 for c in w2s.point(self.env.start[0], self.env.start[1])], 12, 12))

        # task points
        for i, tp in enumerate(self.env.task_points):
            px, py = w2s.point(tp[0], tp[1])
            pygame.draw.polygon(self.screen, TASK,
                                 [(px, py - 8), (px - 7, py + 6), (px + 7, py + 6)])
            label = self.font.render(f"T{i+1}", True, TASK)
            self.screen.blit(label, (px + 8, py - 8))

        # static obstacles
        for o in self.env.static_obstacles:
            px, py = w2s.point(o.x, o.y)
            pygame.draw.circle(self.screen, STATIC_OBS, (px, py), w2s.length(o.radius))

        # dynamic obstacles
        for o in self.env.dynamic_obstacles:
            px, py = w2s.point(o.x, o.y)
            pygame.draw.circle(self.screen, DYNAMIC_OBS, (px, py), w2s.length(o.radius))

        # trajectory
        if len(self.traj_points) > 1:
            pts = [w2s.point(x, y) for x, y in self.traj_points]
            pygame.draw.lines(self.screen, PATH, False, pts, 2)

        # robot
        r = self.env.robot
        rx, ry = w2s.point(r.x, r.y)
        pygame.draw.circle(self.screen, ROBOT, (rx, ry), w2s.length(r.radius) + 2)
        hx, hy = w2s.point(r.x + 0.3 * np.cos(r.theta), r.y + 0.3 * np.sin(r.theta))
        pygame.draw.line(self.screen, ROBOT, (rx, ry), (hx, hy), 3)

        # status panel
        pygame.draw.rect(self.screen, PANEL_BG, (0, WINDOW_SIZE, WINDOW_SIZE, 60))
        n_done = self.env.n_tasks_completed()
        n_total = self.env.n_tasks_total()
        phase = "depot" if self.env.current_task is None else f"task {self.env.active_task_orig_idx()+1}"
        status = (f"step {self.step_count} | tasks {n_done}/{n_total} | targeting {phase}"
                  f"{'  [PAUSED]' if self.paused else ''}"
                  f"{'  COLLISION!' if self.collided else '  DONE' if self.finished else ''}")
        self.screen.blit(self.font.render(status, True, TEXT), (10, WINDOW_SIZE + 10))
        req_text = f"requeued so far: {len(self.env.skipped_log)}  (SPACE=pause, ESC=quit)"
        self.screen.blit(self.font.render(req_text, True, TEXT), (10, WINDOW_SIZE + 32))

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            running = self._handle_events()
            self._simulate_step()
            self._draw()
            self.clock.tick(FPS)
        pygame.quit()


if __name__ == "__main__":
    sim = PygameSim()
    sim.run()
