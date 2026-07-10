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
from scipy.optimize import differential_evolution

import config_utils as cu

# ============================================================
# MPC / DE hyperparameters
# ============================================================
HORIZON = 6                 # control steps predicted per DE evaluation (reduced: cost fn is cheap now, but fewer params -> faster convergence too)
POPSIZE = 12                 # scipy DE popsize multiplier -> pop = popsize * n_params
N_GENERATIONS = 6           # maxiter (warm start + cheap cost fn need very few generations)
DT = cu.SIM_DT

W_PROGRESS = 25.0
W_LATERAL = 10.0
W_HEADING = 5.0
W_STATIC_COLL = 25.0
W_ROBOT_COLL = 25.0
W_SMOOTH = 0.5

# W_PROGRESS = 40.0            # stronger pull to keep moving forward on clear path
# W_LATERAL = 3.0
# W_HEADING = 2.0
# W_STATIC_COLL = 500.0
# W_ROBOT_COLL = 500.0
# W_SMOOTH = 0.2
W_REVERSE = 8.0               # penalize negative/near-zero linear velocity (discourages backing up)
W_STOP_ON_CLEAR = 3.0          # extra penalty for not moving forward when path is clear


STATIC_PROXIMITY_MARGIN = 0.25   # tighter than before -> only reacts when actually close
STATIC_PROXIMITY_WEIGHT = 2.0    # much softer shaping than the old 6.0 * clearance term


class RobotState:
    __slots__ = ("pos", "heading", "prev_action", "prev_seq")

    def __init__(self, pos, heading):
        self.pos = pos.astype(np.float32)
        self.heading = float(heading)
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_seq = np.zeros((HORIZON, 2), dtype=np.float32)   # warm-start cache


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


def obstacles_to_circles(obstacles):
    """Precompute a fast (N,3) array of (cx, cy, radius) circles from the
    map's obstacle list, ONCE, so the DE cost function never has to touch
    shapely (which was the actual bottleneck - creating a Point + calling
    .distance() per trajectory sample per generation in pure Python)."""
    circles = []
    for obs in obstacles:
        if obs["type"] == "circle":
            cx, cy = obs["center"]
            circles.append((cx, cy, obs["radius"]))
        else:
            geom = obs["shape"]
            cx, cy = geom.centroid.x, geom.centroid.y
            xs, ys = geom.exterior.xy
            r = float(np.max(np.hypot(np.array(xs) - cx, np.array(ys) - cy)))
            circles.append((cx, cy, r))
    return np.array(circles, dtype=np.float32) if circles else np.zeros((0, 3), dtype=np.float32)


def static_obstacle_penalty(traj_pos, obstacle_circles):
    """traj_pos: (POP_SIZE, HORIZON, 2). obstacle_circles: (N,3) cx,cy,r.
    Pure-numpy broadcasting, no shapely in the hot path."""
    if obstacle_circles.shape[0] == 0:
        return np.zeros(traj_pos.shape[0], dtype=np.float32)

    n, h, _ = traj_pos.shape
    flat = traj_pos.reshape(-1, 2)                                   # (n*h, 2)
    cx = obstacle_circles[:, 0][None, :]
    cy = obstacle_circles[:, 1][None, :]
    cr = obstacle_circles[:, 2][None, :]
    d = np.hypot(flat[:, 0:1] - cx, flat[:, 1:2] - cy) - cr - cu.ROBOT_RADIUS  # (n*h, N)
    min_d = d.min(axis=1).reshape(n, h).min(axis=1)                   # (n,)

    penalty = np.where(min_d <= 0, W_STATIC_COLL,
                        np.where(min_d < STATIC_PROXIMITY_MARGIN,
                                 STATIC_PROXIMITY_WEIGHT * (STATIC_PROXIMITY_MARGIN - min_d), 0.0))
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


def evaluate_population(pop, state, polyline, cum_arclen, s_prev, obstacle_circles, other_positions):
    traj_pos, traj_heading = rollout_batch(pop, state.pos, state.heading)

    progress, lateral, s_now = path_progress_and_lateral(
        traj_pos[:, -1, :], polyline, cum_arclen, s_prev
    )

    look_idx = np.searchsorted(cum_arclen, np.minimum(s_now + cu.PATH_LOOKAHEAD_DIST, cum_arclen[-1]))
    look_idx = np.clip(look_idx, 0, len(polyline) - 1)
    look_pts = polyline[look_idx]
    desired_heading = np.arctan2(look_pts[:, 1] - traj_pos[:, -1, 1], look_pts[:, 0] - traj_pos[:, -1, 0])
    heading_err = cu.wrap_to_pi(desired_heading - traj_heading[:, -1])

    static_pen = static_obstacle_penalty(traj_pos, obstacle_circles)
    robot_pen = robot_collision_penalty(traj_pos, other_positions)

    smooth_pen = np.sum(np.abs(np.diff(pop[:, :, 1], axis=1, prepend=state.prev_action[1])), axis=1)

    # --- discourage reversing / stalling on a clear path ---
    mean_v = pop[:, :, 0].mean(axis=1)                       # mean commanded linear vel over horizon
    reverse_pen = W_REVERSE * np.maximum(0.0, -mean_v)        # only penalize negative velocity
    is_clear = static_pen <= 0.0                               # no obstacle pressure this rollout
    stall_pen = np.where(is_clear, W_STOP_ON_CLEAR * np.maximum(0.0, 0.3 - mean_v), 0.0)

    cost = (
        -W_PROGRESS * progress
        + W_LATERAL * lateral
        + W_HEADING * (1.0 - np.cos(heading_err))
        + static_pen
        + robot_pen
        + W_SMOOTH * smooth_pen
        + reverse_pen
        + stall_pen
    )
    return cost, s_now


def de_optimize(state, polyline, cum_arclen, s_prev, obstacle_circles, other_positions, rng):
    """Solves for the best HORIZON-length action sequence using scipy's
    differential_evolution with a vectorized objective. Warm-starts the
    initial population from the previous timestep's solution (shifted by
    one step, receding-horizon style) so far fewer generations are needed
    once the controller is past its first call."""

    def objective(x):
        # x: (n_params, n_candidates) -> pop: (n_candidates, HORIZON, 2)
        pop = x.T.reshape(-1, HORIZON, 2).astype(np.float32)
        cost, _ = evaluate_population(pop, state, polyline, cum_arclen, s_prev, obstacle_circles, other_positions)
        return cost

    n_params = HORIZON * 2
    n_pop = POPSIZE * n_params

    # --- warm-start seed: shift last solution forward one step, repeat last action ---
    warm = np.empty((HORIZON, 2), dtype=np.float32)
    warm[:-1] = state.prev_seq[1:]
    warm[-1] = state.prev_seq[-1]

    init_pop = rng.uniform(-1.0, 1.0, size=(n_pop, n_params)).astype(np.float32)
    init_pop[0] = warm.flatten()
    # cluster a chunk of the population tightly around the warm start for fast refinement
    n_near = max(1, n_pop // 4)
    noise = rng.normal(scale=0.1, size=(n_near, n_params)).astype(np.float32)
    init_pop[1:1 + n_near] = np.clip(warm.flatten()[None, :] + noise, -1.0, 1.0)

    bounds = [(-1.0, 1.0)] * n_params
    result = differential_evolution(
        objective,
        bounds,
        popsize=POPSIZE,
        maxiter=N_GENERATIONS,
        vectorized=True,
        updating="deferred",
        polish=False,
        seed=int(rng.integers(0, 2**31 - 1)),
        init=init_pop,
    )

    best_seq = result.x.reshape(HORIZON, 2).astype(np.float32)
    state.prev_seq = best_seq
    _, s_now = evaluate_population(
        best_seq[None, :, :], state, polyline, cum_arclen, s_prev, obstacle_circles, other_positions
    )
    return best_seq[0], float(s_now[0])   # apply only first action (receding horizon)


class DEMPCController:
    """Runs one DE-MPC controller instance per active robot, sharing the
    map/obstacles. Call step() once per sim tick."""

    def __init__(self, map_json_path, path_json_paths, n_robots=None, seed=0, render=False):
        self.map_data = cu.load_map(map_json_path)
        self.obstacles = self.map_data["obstacles"]
        self.obstacle_circles = obstacles_to_circles(self.obstacles)   # precomputed ONCE
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
                self.obstacle_circles, others, self.rng
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
