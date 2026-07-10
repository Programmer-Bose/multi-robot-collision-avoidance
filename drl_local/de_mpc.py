"""
de_mpc.py
---------
Differential-Evolution MPC (DE-MPC) local path-following controller for
multiple robots, replacing the DRL policy. Loads the map + per-robot global
paths via config_utils, then each robot picks its [v, w] control sequence
over a short horizon by running a vectorized DE optimizer against a cost
that rewards path progress/tracking and penalizes static/robot collisions.

Usage:
    python de_mpc.py --map map_002_robot_1.json \
                      --paths map_002_robot_1_manual_control_points.json ...
"""

import argparse
import time

import numpy as np

import config_utils as cu

# ============================================================
# MPC / DE hyperparameters
# ============================================================
HORIZON = 10                # control steps predicted per DE evaluation
POP_SIZE = 25
N_GENERATIONS = 40
F_MUT = 0.6
CR = 0.9
DT = cu.SIM_DT

W_PROGRESS = 25.0
W_LATERAL = 10.0
W_HEADING = 5.0
W_STATIC_COLL = 25.0
W_ROBOT_COLL = 25.0
W_SMOOTH = 0.5


class RobotState:
    __slots__ = ("pos", "heading", "prev_action")

    def __init__(self, pos, heading):
        self.pos = pos.astype(np.float32)
        self.heading = float(heading)
        self.prev_action = np.zeros(2, dtype=np.float32)


def rollout_batch(pop, pos0, heading0):
    """Vectorized unicycle rollout for a whole DE population.
    pop: (POP_SIZE, HORIZON, 2) actions in [-1,1]
    returns positions (POP_SIZE, HORIZON, 2), headings (POP_SIZE, HORIZON)
    """
    n = pop.shape[0]
    pos = np.tile(pos0, (n, 1)).astype(np.float32)
    heading = np.full(n, heading0, dtype=np.float32)

    out_pos = np.empty((n, HORIZON, 2), dtype=np.float32)
    out_heading = np.empty((n, HORIZON), dtype=np.float32)

    for t in range(HORIZON):
        v = pop[:, t, 0] * cu.MAX_LINEAR_VEL
        w = pop[:, t, 1] * cu.MAX_ANGULAR_VEL
        heading = cu.wrap_to_pi(heading + w * DT)
        pos = pos + np.stack([v * np.cos(heading), v * np.sin(heading)], axis=1) * DT
        pos = np.clip(pos, cu.BOUNDS_MIN, cu.BOUNDS_MAX)
        out_pos[:, t] = pos
        out_heading[:, t] = heading

    return out_pos, out_heading


def path_progress_and_lateral(traj_pos, polyline, cum_arclen, s_prev):
    """Nearest-point lookup (per pop member, last horizon step) - vectorized
    brute force against the polyline. traj_pos: (POP_SIZE, 2)"""
    diffs = traj_pos[:, None, :] - polyline[None, :, :]        # (P, N, 2)
    dists = np.hypot(diffs[..., 0], diffs[..., 1])              # (P, N)
    idx = np.argmin(dists, axis=1)
    lateral = dists[np.arange(len(idx)), idx]
    s_now = cum_arclen[idx]
    progress = np.maximum(0.0, s_now - s_prev)
    return progress, lateral, s_now


def static_obstacle_penalty(traj_pos, obstacles):
    """traj_pos: (POP_SIZE, HORIZON, 2). Returns (POP_SIZE,) penalty."""
    n, h, _ = traj_pos.shape
    flat = traj_pos.reshape(-1, 2)
    min_d = np.full(flat.shape[0], 1e9, dtype=np.float32)

    for obs in obstacles:
        if obs["type"] == "circle":
            cx, cy = obs["center"]
            d = np.hypot(flat[:, 0] - cx, flat[:, 1] - cy) - obs["radius"] - cu.ROBOT_RADIUS
        else:
            geom = obs["shape"]
            d = np.array([geom.distance(__import__("shapely.geometry", fromlist=["Point"]).Point(p))
                           - cu.ROBOT_RADIUS for p in flat], dtype=np.float32)
        min_d = np.minimum(min_d, d)

    min_d = min_d.reshape(n, h).min(axis=1)
    penalty = np.where(min_d <= 0, W_STATIC_COLL,
                        np.where(min_d < cu.COLLISION_PROXIMITY_MARGIN,
                                 W_STATIC_COLL * 0.1 * (cu.COLLISION_PROXIMITY_MARGIN - min_d), 0.0))
    return penalty


def robot_collision_penalty(traj_pos, other_positions):
    """traj_pos: (POP_SIZE, HORIZON, 2); other_positions: list of (2,) arrays
    for other robots' CURRENT positions (static within this short horizon
    is a reasonable approx for a fast local controller)."""
    if not other_positions:
        return np.zeros(traj_pos.shape[0], dtype=np.float32)
    others = np.stack(other_positions, axis=0)                  # (M, 2)
    diffs = traj_pos[:, :, None, :] - others[None, None, :, :]  # (P,H,M,2)
    d = np.hypot(diffs[..., 0], diffs[..., 1]) - 2 * cu.ROBOT_RADIUS
    min_d = d.min(axis=(1, 2))
    penalty = np.where(min_d <= 0, W_ROBOT_COLL,
                        np.where(min_d < cu.COLLISION_PROXIMITY_MARGIN,
                                 W_ROBOT_COLL * 0.1 * (cu.COLLISION_PROXIMITY_MARGIN - min_d), 0.0))
    return penalty


def evaluate_population(pop, state, polyline, cum_arclen, s_prev, obstacles, other_positions):
    traj_pos, traj_heading = rollout_batch(pop, state.pos, state.heading)

    progress, lateral, s_now = path_progress_and_lateral(
        traj_pos[:, -1, :], polyline, cum_arclen, s_prev
    )

    look_idx = np.searchsorted(cum_arclen, np.minimum(s_now + cu.PATH_LOOKAHEAD_DIST, cum_arclen[-1]))
    look_idx = np.clip(look_idx, 0, len(polyline) - 1)
    look_pts = polyline[look_idx]
    desired_heading = np.arctan2(look_pts[:, 1] - traj_pos[:, -1, 1], look_pts[:, 0] - traj_pos[:, -1, 0])
    heading_err = cu.wrap_to_pi(desired_heading - traj_heading[:, -1])

    static_pen = static_obstacle_penalty(traj_pos, obstacles)
    robot_pen = robot_collision_penalty(traj_pos, other_positions)

    smooth_pen = np.sum(np.abs(np.diff(pop[:, :, 1], axis=1, prepend=state.prev_action[1])), axis=1)

    cost = (
        -W_PROGRESS * progress
        + W_LATERAL * lateral
        + W_HEADING * (1.0 - np.cos(heading_err))
        + static_pen
        + robot_pen
        + W_SMOOTH * smooth_pen
    )
    return cost, s_now


def de_optimize(state, polyline, cum_arclen, s_prev, obstacles, other_positions, rng):
    pop = rng.uniform(-1.0, 1.0, size=(POP_SIZE, HORIZON, 2)).astype(np.float32)
    cost, _ = evaluate_population(pop, state, polyline, cum_arclen, s_prev, obstacles, other_positions)

    for _ in range(N_GENERATIONS):
        idx_a = rng.integers(0, POP_SIZE, POP_SIZE)
        idx_b = rng.integers(0, POP_SIZE, POP_SIZE)
        idx_c = rng.integers(0, POP_SIZE, POP_SIZE)

        mutant = pop[idx_a] + F_MUT * (pop[idx_b] - pop[idx_c])
        mutant = np.clip(mutant, -1.0, 1.0)

        cross_mask = rng.uniform(size=(POP_SIZE, HORIZON, 2)) < CR
        trial = np.where(cross_mask, mutant, pop)

        trial_cost, _ = evaluate_population(trial, state, polyline, cum_arclen, s_prev, obstacles, other_positions)

        improved = trial_cost < cost
        pop[improved] = trial[improved]
        cost[improved] = trial_cost[improved]

    best = int(np.argmin(cost))
    _, s_now = evaluate_population(pop[best:best + 1], state, polyline, cum_arclen, s_prev, obstacles, other_positions)
    return pop[best, 0], float(s_now[0])   # apply only first action (receding horizon)


class DEMPCController:
    """Runs one DE-MPC controller instance per active robot, sharing the
    map/obstacles. Call step() once per sim tick."""

    def __init__(self, map_json_path, path_json_paths, n_robots=None, seed=0, render=False):
        self.map_data = cu.load_map(map_json_path)
        self.obstacles = self.map_data["obstacles"]
        self.n_robots = n_robots if n_robots is not None else len(path_json_paths)
        self.rng = np.random.default_rng(seed)

        self.polylines, self.arclens = [], []
        for p in path_json_paths[: self.n_robots]:
            segs = cu.load_global_path_control_points(p)
            poly = cu.build_full_path_polyline(segs)
            self.polylines.append(poly)
            self.arclens.append(cu.path_arclength_table(poly))

        start = self.map_data["start"]
        self.states = [RobotState(start.copy(), 0.0) for _ in range(self.n_robots)]
        self.s_prev = [0.0] * self.n_robots

        self.render_enabled = render
        self._screen = None
        self._clock = None
        self._font = None
        self.step_count = 0
        if render:
            self._init_pygame()

    def _init_pygame(self):
        import pygame
        pygame.init()
        size = (cu.RENDER_SCREEN_SIZE, cu.RENDER_SCREEN_SIZE + cu.RENDER_PANEL_HEIGHT)
        self._screen = pygame.display.set_mode(size)
        pygame.display.set_caption("DE-MPC Multi-Robot")
        self._font = pygame.font.SysFont(None, 22)
        self._clock = pygame.time.Clock()

    def _world_to_screen(self, pt):
        x, y = pt
        sx = x * cu.RENDER_PIXELS_PER_UNIT
        sy = cu.RENDER_SCREEN_SIZE - y * cu.RENDER_PIXELS_PER_UNIT
        return int(sx), int(sy)

    def render(self):
        import pygame
        screen = self._screen
        screen.fill(cu.RENDER_COLOR_BG)

        for obs in self.obstacles:
            if obs["type"] == "circle":
                cx, cy = obs["center"]
                pygame.draw.circle(screen, cu.RENDER_COLOR_OBSTACLE,
                                    self._world_to_screen((cx, cy)),
                                    int(obs["radius"] * cu.RENDER_PIXELS_PER_UNIT))
            else:
                xs, ys = obs["shape"].exterior.xy
                pts = [self._world_to_screen((x, y)) for x, y in zip(xs, ys)]
                pygame.draw.polygon(screen, cu.RENDER_COLOR_OBSTACLE, pts)

        for i, state in enumerate(self.states):
            color = cu.render_robot_color(i)
            pts = [self._world_to_screen(p) for p in self.polylines[i][::4]]
            if len(pts) > 1:
                pygame.draw.lines(screen, cu.RENDER_COLOR_PATH, False, pts, 2)

            px = self._world_to_screen(state.pos)
            pygame.draw.circle(screen, color, px, int(cu.ROBOT_RADIUS * cu.RENDER_PIXELS_PER_UNIT))
            hx = px[0] + int(15 * np.cos(state.heading))
            hy = px[1] - int(15 * np.sin(state.heading))
            pygame.draw.line(screen, (0, 0, 0), px, (hx, hy), 2)

        goal = self.map_data["goal"]
        if goal is not None:
            pygame.draw.circle(screen, cu.RENDER_COLOR_GOAL, self._world_to_screen(goal), 10, 3)

        pygame.draw.rect(screen, cu.RENDER_COLOR_HUD_BG,
                          (0, cu.RENDER_SCREEN_SIZE, cu.RENDER_SCREEN_SIZE, cu.RENDER_PANEL_HEIGHT))
        hud_text = f"step {self.step_count}  n_robots={self.n_robots}  DE-MPC"
        label = self._font.render(hud_text, True, cu.RENDER_COLOR_HUD_TEXT)
        screen.blit(label, (10, cu.RENDER_SCREEN_SIZE + 18))

        pygame.event.pump()
        pygame.display.flip()
        self._clock.tick(cu.RENDER_FPS)

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None

    def step(self):
        """Advances every robot by one control action (first step of each
        robot's optimized horizon). Returns list of (pos, heading)."""
        other_positions_all = [s.pos for s in self.states]

        results = []
        for i, state in enumerate(self.states):
            others = [p for j, p in enumerate(other_positions_all) if j != i]
            action, s_now = de_optimize(
                state, self.polylines[i], self.arclens[i], self.s_prev[i],
                self.obstacles, others, self.rng
            )
            v = action[0] * cu.MAX_LINEAR_VEL
            w = action[1] * cu.MAX_ANGULAR_VEL
            state.heading = cu.wrap_to_pi(state.heading + w * DT)
            state.pos = np.clip(
                state.pos + np.array([v * np.cos(state.heading), v * np.sin(state.heading)]) * DT,
                cu.BOUNDS_MIN, cu.BOUNDS_MAX
            )
            state.prev_action = action
            self.s_prev[i] = s_now
            results.append((state.pos.copy(), state.heading))

        self.step_count += 1
        if self.render_enabled:
            self.render()

        return results

    def goal_reached(self, i):
        goal = self.map_data["goal"] if self.map_data["goal"] is not None else self.polylines[i][-1]
        return float(np.linalg.norm(goal - self.states[i].pos)) <= cu.GOAL_REACH_RADIUS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--map", type=str, required=True)
    p.add_argument("--paths", type=str, nargs="+", required=True)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true", default=False,
                    help="Show a live pygame window while running.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    controller = DEMPCController(args.map, args.paths, seed=args.seed, render=args.render)

    t0 = time.time()
    for step in range(args.steps):
        results = controller.step()
        if all(controller.goal_reached(i) for i in range(controller.n_robots)):
            print(f"All robots reached goal at step {step}")
            break
    print(f"Elapsed: {time.time() - t0:.2f}s for {step + 1} steps")
    controller.close()