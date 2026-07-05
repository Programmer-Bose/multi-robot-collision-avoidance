"""
Shared Pygame rendering for the robot-nav scenario. Used by:
  - RobotNavGymEnv (render_mode="human") to visualize the DRL policy acting
  - pygame_sim.py to visualize DE-MPC acting
Keeping this in one place means both always look identical, so what you see
watching the untrained/training policy is visually comparable to what you
saw watching DE-MPC.
"""

import numpy as np
import pygame

BG = (250, 250, 250)
DEPOT = (20, 20, 20)
TASK = (30, 140, 60)
STATIC_OBS = (130, 130, 130)
DYNAMIC_OBS = (220, 90, 70)
ROBOT = (40, 90, 220)
PATH = (40, 90, 220)
TEXT = (20, 20, 20)
PANEL_BG = (255, 255, 255)


class WorldToScreen:
    def __init__(self, world_bounds, window_size, margin):
        xmin, xmax, ymin, ymax = world_bounds
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        span_x, span_y = xmax - xmin, ymax - ymin
        drawable = window_size - 2 * margin
        self.scale = drawable / max(span_x, span_y)
        self.margin = margin

    def point(self, x, y):
        sx = self.margin + (x - self.xmin) * self.scale
        sy = self.margin + (self.ymax - y) * self.scale
        return int(sx), int(sy)

    def length(self, world_len):
        return max(1, int(world_len * self.scale))


class SceneRenderer:
    """
    Wraps a Pygame window + font + coordinate transform, and knows how to
    draw one frame of a SingleRobotEnv-shaped scene. Call `draw(env, traj_points,
    status_lines)` once per frame, then `flip()`.
    """

    def __init__(self, world_bounds, window_size=800, margin=40, panel_height=60,
                 caption="Robot Nav"):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((window_size, window_size + panel_height))
        pygame.display.set_caption(caption)
        self.font = pygame.font.SysFont("consolas", 16)
        self.window_size = window_size
        self.panel_height = panel_height
        self.w2s = WorldToScreen(world_bounds, window_size, margin)

    def handle_quit_events(self):
        """Returns False if the window should close (X button or ESC)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def draw(self, env, traj_points=None, status_lines=None):
        w2s = self.w2s
        self.screen.fill(BG)

        px, py = w2s.point(env.start[0], env.start[1])
        pygame.draw.rect(self.screen, DEPOT, (px - 6, py - 6, 12, 12))

        for i, tp in enumerate(env.task_points):
            px, py = w2s.point(tp[0], tp[1])
            pygame.draw.polygon(self.screen, TASK,
                                 [(px, py - 8), (px - 7, py + 6), (px + 7, py + 6)])
            self.screen.blit(self.font.render(f"T{i+1}", True, TASK), (px + 8, py - 8))

        for o in env.static_obstacles:
            px, py = w2s.point(o.x, o.y)
            pygame.draw.circle(self.screen, STATIC_OBS, (px, py), w2s.length(o.radius))

        for o in env.dynamic_obstacles:
            px, py = w2s.point(o.x, o.y)
            pygame.draw.circle(self.screen, DYNAMIC_OBS, (px, py), w2s.length(o.radius))

        if traj_points and len(traj_points) > 1:
            pts = [w2s.point(x, y) for x, y in traj_points]
            pygame.draw.lines(self.screen, PATH, False, pts, 2)

        r = env.robot
        rx, ry = w2s.point(r.x, r.y)
        pygame.draw.circle(self.screen, ROBOT, (rx, ry), w2s.length(r.radius) + 2)
        hx, hy = w2s.point(r.x + 0.3 * np.cos(r.theta), r.y + 0.3 * np.sin(r.theta))
        pygame.draw.line(self.screen, ROBOT, (rx, ry), (hx, hy), 3)

        pygame.draw.rect(self.screen, PANEL_BG,
                          (0, self.window_size, self.window_size, self.panel_height))
        for i, line in enumerate(status_lines or []):
            self.screen.blit(self.font.render(line, True, TEXT), (10, self.window_size + 8 + 22 * i))

        pygame.display.flip()

    def tick(self, clock, fps):
        clock.tick(fps)

    def close(self):
        pygame.quit()
