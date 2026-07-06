
"""
teleop_recorder.py

Human teleoperation recorder for the DE-MPC environment.

Uses:
    env.py
    rangefinder.py
    occupancy_grid.py

Dataset format (IDENTICAL to run_episode.save_dataset()):

robot_state
goal_dist
goal_bearing
ranges
occupancy_grid
action

Controls
--------
W/S : forward / reverse
A/D : left / right
SHIFT : speed boost
SPACE : start/stop recording
ENTER : save episode
R : reset scenario
ESC : quit
"""

import os
import numpy as np
import pygame

from env import make_default_scenario
from rangefinder import simulate_rangefinder
from occupancy_grid import compute_occupancy_grid


# ---------------- CONFIG ---------------- #

SEED = 1001

N_STATIC = 20
N_DYNAMIC = 0
N_TASKS = 10

WINDOW = 850
MARGIN = 40
FPS = 60

N_RAYS = 15
MAX_SENSOR_RANGE = 5.0

SAVE_DIR = "data_traj"

MAX_LINEAR = 0.4      # instead of 2.0
MAX_ANGULAR = (2*np.pi)/6     # instead of np.pi

ACCEL = 1.0           # instead of 4.5
ANG_ACCEL = 1.0       # instead of 8.0

LIN_DAMP = 3.0        # instead of 5.5
ANG_DAMP = 3.0        # instead of 9.0

# ---------------------------------------- #


class WorldToScreen:

    def __init__(self, bounds):
        xmin, xmax, ymin, ymax = bounds
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.scale = (WINDOW - 2 * MARGIN) / max(xmax - xmin, ymax - ymin)

    def point(self, x, y):
        sx = MARGIN + (x - self.xmin) * self.scale
        sy = MARGIN + (self.ymax - y) * self.scale
        return int(sx), int(sy)

    def length(self, r):
        return max(1, int(r * self.scale))


class TeleopRecorder:

    def __init__(self):

        pygame.init()

        self.screen = pygame.display.set_mode((WINDOW, WINDOW + 80))
        pygame.display.set_caption("Human Demonstration Recorder")

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("consolas", 18)

        os.makedirs(SAVE_DIR, exist_ok=True)

        self.reset()

    def reset(self):

        self.env = make_default_scenario(
            seed=SEED,
            n_static=N_STATIC,
            n_dynamic=N_DYNAMIC,
            n_tasks=N_TASKS,
            omega_max=np.pi,
            v_max=MAX_LINEAR,
        )

        self.env.reset()

        self.mapper = WorldToScreen(self.env.world_bounds)

        self.v = 0.0
        self.omega = 0.0

        self.recording = False

        self.dataset = []

        self.traj = []

    def record_step(self):

        obs = self.env.get_obs()

        ranges = simulate_rangefinder(
            robot_state=tuple(obs["robot_state"]),
            static_obstacles=obs["static_obstacles"],
            dynamic_obstacles=obs["dynamic_obstacles"],
            world_bounds=self.env.world_bounds,
            n_rays=N_RAYS,
            max_range=MAX_SENSOR_RANGE,
        )

        occ = compute_occupancy_grid(
            robot_state=tuple(obs["robot_state"]),
            static_obstacles=obs["static_obstacles"],
            dynamic_obstacles=obs["dynamic_obstacles"],
            grid_size=21,
            resolution=0.25,
        )

        rx, ry, rt = obs["robot_state"]
        gx, gy = obs["goal"]

        goal_dist = np.hypot(gx - rx, gy - ry)

        goal_bearing = (
            (np.arctan2(gy - ry, gx - rx) - rt + np.pi)
            % (2 * np.pi)
            - np.pi
        )

        self.dataset.append(
            dict(
                robot_state=np.array([rx, ry, rt], float),
                goal_dist=goal_dist,
                goal_bearing=goal_bearing,
                ranges=ranges,
                occupancy_grid=occ,
                action=np.array([self.v, self.omega], float),
            )
        )

    def save_episode(self):

        if len(self.dataset) == 0:
            print("Nothing to save.")
            return

        idx = len(os.listdir(SAVE_DIR)) + 1

        path = os.path.join(SAVE_DIR, f"path_{idx:04d}.npz")

        np.savez(
            path,
            robot_state=np.stack([d["robot_state"] for d in self.dataset]),
            goal_dist=np.array([d["goal_dist"] for d in self.dataset]),
            goal_bearing=np.array([d["goal_bearing"] for d in self.dataset]),
            ranges=np.stack([d["ranges"] for d in self.dataset]),
            occupancy_grid=np.stack(
                [d["occupancy_grid"] for d in self.dataset]
            ),
            action=np.stack([d["action"] for d in self.dataset]),
        )

        print("Saved", path)

    def update_keyboard(self, dt):

        keys = pygame.key.get_pressed()

        boost = 1.5 if keys[pygame.K_LSHIFT] else 1.0

        target_v = 0

        if keys[pygame.K_w]:
            target_v = MAX_LINEAR * boost

        elif keys[pygame.K_s]:
            target_v = -MAX_LINEAR * boost

        target_w = 0

        if keys[pygame.K_a]:
            target_w = MAX_ANGULAR

        elif keys[pygame.K_d]:
            target_w = -MAX_ANGULAR

        if self.v < target_v:
            self.v = min(self.v + ACCEL * dt, target_v)
        else:
            self.v = max(self.v - ACCEL * dt, target_v)

        if target_v == 0:
            self.v *= np.exp(-LIN_DAMP * dt)

        if self.omega < target_w:
            self.omega = min(self.omega + ANG_ACCEL * dt, target_w)
        else:
            self.omega = max(self.omega - ANG_ACCEL * dt, target_w)

        if target_w == 0:
            self.omega *= np.exp(-ANG_DAMP * dt)

    def draw(self):

        self.screen.fill((250, 250, 250))

        m = self.mapper

        pygame.draw.rect(
            self.screen,
            (0, 0, 0),
            (*[c - 6 for c in m.point(*self.env.start[:2])], 12, 12),
        )

        for i, t in enumerate(self.env.task_points):
            px, py = m.point(*t)
            pygame.draw.circle(self.screen,(0,170,0),(px,py),7)
            label=self.font.render(f"T{i+1}",True,(0,90,0))
            self.screen.blit(label,(px+8,py-8))

        for o in self.env.static_obstacles:
            pygame.draw.circle(
                self.screen,
                (120, 120, 120),
                m.point(o.x, o.y),
                m.length(o.radius),
            )

        for o in self.env.dynamic_obstacles:
            pygame.draw.circle(
                self.screen,
                (220, 70, 70),
                m.point(o.x, o.y),
                m.length(o.radius),
            )

        if len(self.traj) > 1:
            pygame.draw.lines(
                self.screen,
                (0, 0, 255),
                False,
                [m.point(x, y) for x, y in self.traj],
                2,
            )

        r = self.env.robot

        rx, ry = m.point(r.x, r.y)

        pygame.draw.circle(
            self.screen,
            (50, 100, 255),
            (rx, ry),
            m.length(r.radius) + 2,
        )

        hx = r.x + 0.35 * np.cos(r.theta)
        hy = r.y + 0.35 * np.sin(r.theta)

        pygame.draw.line(
            self.screen,
            (0, 0, 255),
            (rx, ry),
            m.point(hx, hy),
            3,
        )

        txt = (
            f"Recording:{self.recording}   "
            f"Samples:{len(self.dataset)}   "
            f"Tasks:{self.env.n_tasks_completed()}/{self.env.n_tasks_total()}"
        )

        self.screen.blit(
            self.font.render(txt, True, (20, 20, 20)),
            (10, WINDOW + 20),
        )

        pygame.display.flip()

    def run(self):

        running = True

        while running:

            dt = self.clock.tick(FPS) / 1000.0

            for e in pygame.event.get():

                if e.type == pygame.QUIT:
                    running = False

                if e.type == pygame.KEYDOWN:

                    if e.key == pygame.K_ESCAPE:
                        running = False

                    elif e.key == pygame.K_SPACE:
                        self.recording = not self.recording
                        print("Recording:", self.recording)

                    elif e.key == pygame.K_RETURN:
                        self.save_episode()

                    elif e.key == pygame.K_r:
                        self.reset()

            self.update_keyboard(dt)

            obs, done, info = self.env.step(self.v,self.omega)

            self.traj.append((self.env.robot.x,self.env.robot.y))

            if self.recording:
                self.record_step()

            if info["collided"]:
                print(f"Collision ({info['collision_kind']})")

                self.recording=False
                self.dataset.clear()      # discard failed demonstration
                self.reset()
                continue

            if done:
                if self.recording:
                    self.save_episode()
                self.recording=False
                self.reset()
                continue

            self.traj.append((self.env.robot.x, self.env.robot.y))

            if self.recording:
                self.record_step()

            self.draw()

        pygame.quit()


if __name__ == "__main__":
    TeleopRecorder().run()
