"""
dde_mpc_multi.py
-----------------
Multi-robot Differential-Evolution based Model Predictive Control (DE-MPC).

Each robot follows its own pre-generated B-spline reference path (loaded from
a "manual_control_points.json" file) while sequentially visiting task points
and reaching its goal. Every robot's DE-MPC controller is stepped sequentially
(no multithreading) within each global timestep. Static obstacles come from each robot's map-config JSON (assumed
identical across robots on the same map). Dynamic obstacles are:
  (a) the OTHER robots in the swarm (always considered), and
  (b) optional externally-supplied dynamic obstacles (can be fully omitted
      by passing an empty list / zero count).

No live visualization is performed. Instead, every timestep, every robot's
state (x, y, theta), controls (v, omega), progress (s) and timestamp are
logged to a per-robot CSV file so the run can be reconstructed / plotted /
animated afterward.

Usage (bottom of file) shows a 3-robot example, but the code generalizes to
N robots: for N robots you supply N map-config JSON paths and N manual
control-point JSON paths (2N files total).
"""

import os
import csv
import json
import time
import numpy as np
from scipy.optimize import differential_evolution
from scipy.interpolate import BSpline

# ----------------------------------------------------------------------
# 0. HYPERPARAMETERS
# ----------------------------------------------------------------------

TARGET_SCALE = 12.0
BOUNDS_MIN = np.array([0.0, 0.0])
BOUNDS_MAX = np.array([TARGET_SCALE, TARGET_SCALE])

ROBOT_RADIUS = 0.15
SAFETY_MARGIN = 0.15                 # extra buffer added on top of 2*ROBOT_RADIUS for inter-robot / dynamic-obstacle checks

DT = 0.2                             # simulation / control timestep (s)
HORIZON = 7                          # MPC prediction horizon (number of control steps)
MAX_STEPS = 25                     # hard cap on total MPC steps per robot (safety net)

V_MAX = 1.2                          # max linear velocity (forward)
V_MIN = -0.6                         # max reverse velocity (negative => backward motion allowed)
OMEGA_MAX = 1.8                      # max angular velocity (rad/s)

GOAL_TOLERANCE = 0.25                # distance to final goal considered "reached"
WAYPOINT_ADVANCE_LOOKAHEAD = 0.4     # arc-length progress increment considered per step for reference window

SENSING_RANGE = 3.5                  # robots/obstacles beyond this range are ignored in the cost function


# --- Cost weights ---
W_TRACK = 6.0                        # tracking error to reference path
W_PROGRESS = 2.0                     # reward for arc-length progress (encourages forward motion along path)
W_CONTROL_EFFORT = 0.05
W_CONTROL_SMOOTH = 0.3               # penalize jerky changes in control between consecutive horizon steps
W_STATIC_COLLISION = 900.0
W_ROBOT_COLLISION = 900.0
W_DYNAMIC_OBS_COLLISION = 900.0
W_HEADING = 0.5                      # small penalty for heading misalignment with path direction

# --- Differential Evolution settings (per MPC step, per robot) ---
DE_POPSIZE_COLD = 30                 # population size used only for the very first (cold-start) step
DE_POPSIZE_WARM = 15                 # population size used for every subsequent (warm-started) step
DE_MAXITER_COLD = 40
DE_MAXITER_WARM = 12
DE_MUTATION = (0.4, 1.0)
DE_RECOMBINATION = 0.85
DE_TOL = 1e-6
WARM_JITTER_STD = 0.75              # gaussian jitter applied to shifted warm-start controls
WARM_START_FRACTION = 0.7            # fraction of warm-started population seeded from previous solution; rest is random within bounds

CSV_OUTPUT_DIR = "mpc_logs"

# ----------------------------------------------------------------------
# 1. Static map loading (obstacles) -- mirrors dde_mul_la.py conventions
# ----------------------------------------------------------------------

def load_static_obstacles(json_path):
    """Loads obstacles + scale factor from a map-config JSON (same format
    used by dde_mul_la.py's load_map_config). Returns (obstacles, scale_factor,
    orig_w, orig_h)."""
    with open(json_path, "r") as f:
        data = json.load(f)

    meta_key = "map_metadata" if "map_metadata" in data else "robot_metadata"
    orig_w, orig_h = data[meta_key]["size"]
    scale_factor = TARGET_SCALE / orig_w

    def scale_pt(pt):
        return np.array([pt[0] * scale_factor, (orig_h - pt[1]) * scale_factor])

    obstacles = []
    for obs in data["obstacles"]:
        obs_type = obs["type"]
        cx, cy = scale_pt(obs["position"])

        if obs_type == "circle":
            r_scaled = obs["radius"] * scale_factor
            obstacles.append({"type": "circle", "center": (cx, cy), "radius": r_scaled})

        elif obs_type == "square":
            h = (obs["size"] * scale_factor) / 2.0
            corners = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
            obstacles.append({"type": "polygon", "corners": corners})

        elif obs_type == "rectangle":
            hw = (obs["width"] * scale_factor) / 2.0
            hh = (obs["height"] * scale_factor) / 2.0
            corners = [(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)]
            obstacles.append({"type": "polygon", "corners": corners})

        elif obs_type == "u_shape":
            # Approximate with 3 rectangles (arms + bottom bar) as axis-aligned boxes
            h = (obs["size"] * scale_factor) / 2.0
            t = obs["thickness"] * scale_factor
            left_arm = [(cx - h, cy - h), (cx - h + t, cy - h), (cx - h + t, cy + h), (cx - h, cy + h)]
            right_arm = [(cx + h - t, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx + h - t, cy + h)]
            bottom_bar = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy - h + t), (cx - h, cy - h + t)]
            obstacles.append({"type": "polygon", "corners": left_arm})
            obstacles.append({"type": "polygon", "corners": right_arm})
            obstacles.append({"type": "polygon", "corners": bottom_bar})

    return obstacles, scale_factor


def point_to_polygon_distance(px, py, corners):
    """Approximate signed distance from point to an axis-aligned rectangle
    given its 4 corners (no shapely dependency). Returns (distance, inside)."""
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    dx = max(xmin - px, 0.0, px - xmax)
    dy = max(ymin - py, 0.0, py - ymax)
    outside_dist = np.hypot(dx, dy)

    inside = (xmin <= px <= xmax) and (ymin <= py <= ymax)
    if inside:
        inside_dist = min(px - xmin, xmax - px, py - ymin, ymax - py)
        return inside_dist, True
    return outside_dist, False


# ----------------------------------------------------------------------
# 2. Reference path: dense sampling + arc-length parameterization
# ----------------------------------------------------------------------

def make_clamped_knot_vector(n_ctrl_pts, degree):
    n_internal = n_ctrl_pts - degree - 1
    if n_internal > 0:
        internal_knots = np.linspace(0, 1, n_internal + 2)[1:-1]
    else:
        internal_knots = np.array([])
    return np.concatenate((np.zeros(degree + 1), internal_knots, np.ones(degree + 1)))


def bspline_curve(control_points, n_samples, degree=3):
    control_points = np.asarray(control_points)
    n_ctrl_pts = len(control_points)
    k = min(degree, n_ctrl_pts - 1)
    knots = make_clamped_knot_vector(n_ctrl_pts, k)
    t = np.linspace(0.0, 1.0, n_samples)
    spline_x = BSpline(knots, control_points[:, 0], k)
    spline_y = BSpline(knots, control_points[:, 1], k)
    return np.column_stack([spline_x(t), spline_y(t)])


class ReferencePath:
    """Dense, arc-length parameterized reference path built by concatenating
    every segment's B-spline curve from a manual_control_points.json file."""

    def __init__(self, control_points_json, samples_per_segment=60):
        with open(control_points_json, "r") as f:
            data = json.load(f)

        dense_points = []
        # task-point markers: arc-length at which each segment ENDS (i.e. a task/goal is reached)
        self.segment_end_s = []

        for seg in data["segments"]:
            start = np.array(seg["start_point"])
            end = np.array(seg["end_point"])
            free_pts = np.array(seg["control_points"])
            full_ctrl = np.vstack([start, free_pts, end])
            curve = bspline_curve(full_ctrl, samples_per_segment)
            if dense_points:
                curve = curve[1:]  # avoid duplicating the shared join point
            dense_points.append(curve)

        self.points = np.vstack(dense_points)

        diffs = np.diff(self.points, axis=0)
        seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
        self.cum_s = np.concatenate([[0.0], np.cumsum(seg_lens)])
        self.total_length = self.cum_s[-1]

        # recompute segment_end_s now that we know cumulative arc length,
        # by matching each segment's end_point to the nearest dense point
        running_idx = 0
        for seg in data["segments"]:
            end = np.array(seg["end_point"])
            dists = np.hypot(self.points[:, 0] - end[0], self.points[:, 1] - end[1])
            idx = int(np.argmin(dists))
            self.segment_end_s.append(self.cum_s[idx])

        self.start_point = self.points[0].copy()
        self.goal_point = self.points[-1].copy()

    def point_at_s(self, s):
        s = np.clip(s, 0.0, self.total_length)
        return np.array([np.interp(s, self.cum_s, self.points[:, 0]),
                          np.interp(s, self.cum_s, self.points[:, 1])])

    def heading_at_s(self, s, eps=0.05):
        p1 = self.point_at_s(max(s - eps, 0.0))
        p2 = self.point_at_s(min(s + eps, self.total_length))
        vec = p2 - p1
        if np.allclose(vec, 0):
            return 0.0
        return float(np.arctan2(vec[1], vec[0]))

    def closest_s(self, xy, s_guess, search_window=2.5):
        """Local search for the closest arc-length parameter near s_guess
        (cheap alternative to a global nearest-point search every step)."""
        lo = max(0.0, s_guess - search_window)
        hi = min(self.total_length, s_guess + search_window)
        mask = (self.cum_s >= lo) & (self.cum_s <= hi)
        if not np.any(mask):
            mask = slice(None)
        candidate_pts = self.points[mask]
        candidate_s = self.cum_s[mask]
        dists = np.hypot(candidate_pts[:, 0] - xy[0], candidate_pts[:, 1] - xy[1])
        best_idx = int(np.argmin(dists))
        return float(candidate_s[best_idx])


# ----------------------------------------------------------------------
# 3. Robot state / kinematics
# ----------------------------------------------------------------------

def unicycle_step(state, control, dt):
    """state = [x, y, theta]; control = [v, omega]. v may be negative
    (backward motion)."""
    x, y, theta = state
    v, omega = control
    new_theta = theta + omega * dt
    new_x = x + v * np.cos(theta) * dt
    new_y = y + v * np.sin(theta) * dt
    return np.array([new_x, new_y, new_theta])


def rollout(state0, controls_flat, horizon, dt):
    """controls_flat: flat array of length horizon*2 -> [v0, w0, v1, w1, ...]
    Returns array of shape (horizon+1, 3) of states (including initial)."""
    controls = controls_flat.reshape(horizon, 2)
    states = np.zeros((horizon + 1, 3))
    states[0] = state0
    for i in range(horizon):
        states[i + 1] = unicycle_step(states[i], controls[i], dt)
    return states, controls


# ----------------------------------------------------------------------
# 4. Dynamic obstacle container (other robots + optional external ones)
# ----------------------------------------------------------------------

class SharedSwarmState:
    """Blackboard holding each robot's latest known position and predicted
    trajectory, plus a list of externally-supplied dynamic obstacles (can be
    empty). Robots are now stepped strictly sequentially (no threading), so
    no locking is required."""

    def __init__(self, robot_ids, external_dynamic_obstacles=None):
        self.positions = {rid: None for rid in robot_ids}          # rid -> (x, y, theta)
        self.predicted_traj = {rid: None for rid in robot_ids}      # rid -> (horizon+1, 2) xy predictions from last step
        self.finished = {rid: False for rid in robot_ids}
        # external_dynamic_obstacles: list of callables t -> (x, y) OR None/[] to omit entirely
        self.external_dynamic_obstacles = external_dynamic_obstacles or []

    def update_robot(self, rid, position, predicted_xy):
        self.positions[rid] = position
        self.predicted_traj[rid] = predicted_xy

    def mark_finished(self, rid):
        self.finished[rid] = True

    def get_other_robot_predictions(self, self_rid):
        others = {}
        for rid, traj in self.predicted_traj.items():
            if rid != self_rid and traj is not None:
                others[rid] = traj.copy()
        return others

    def get_external_obstacle_positions(self, t):
        """Evaluate all external dynamic obstacle callables at time t."""
        if not self.external_dynamic_obstacles:
            return []
        return [fn(t) for fn in self.external_dynamic_obstacles]


# ----------------------------------------------------------------------
# 5. DE-MPC controller for a single robot
# ----------------------------------------------------------------------

class DE_MPC_Robot:
    def __init__(self, robot_id, ref_path: ReferencePath, static_obstacles,
                 swarm_state: SharedSwarmState, dt=DT, horizon=HORIZON):
        self.rid = robot_id
        self.ref_path = ref_path
        self.static_obstacles = static_obstacles
        self.swarm = swarm_state
        self.dt = dt
        self.horizon = horizon

        start_xy = ref_path.start_point
        start_heading = ref_path.heading_at_s(0.0)
        self.state = np.array([start_xy[0], start_xy[1], start_heading])

        self.s_progress = 0.0
        self.step_count = 0
        self.prev_solution = None   # warm-start seed (flat array, length horizon*2)
        self.finished = False
        self.log_rows = []

    # -- bounds --
    def build_bounds(self):
        return [(V_MIN, V_MAX), (-OMEGA_MAX, OMEGA_MAX)] * self.horizon

    # -- warm start population --
    def build_init_population(self, popsize, n_vars):
        if self.prev_solution is None:
            # cold start: let scipy generate the initial population itself
            return None

        # shift previous solution by one step (drop first control, repeat last)
        prev = self.prev_solution.reshape(self.horizon, 2)
        shifted = np.vstack([prev[1:], prev[-1:]])
        seed = shifted.flatten()

        n_individuals = popsize * n_vars
        n_seeded = int(round(n_individuals * WARM_START_FRACTION))
        n_random = n_individuals - n_seeded

        population = []
        for _ in range(n_seeded):
            jittered = seed + np.random.normal(0.0, WARM_JITTER_STD, size=n_vars)
            population.append(jittered)

        if n_random > 0:
            bounds = self.build_bounds()
            lows = np.array([b[0] for b in bounds])
            highs = np.array([b[1] for b in bounds])
            random_individuals = np.random.uniform(lows, highs, size=(n_random, n_vars))
            population.extend(list(random_individuals))

        return np.array(population)

    # -- cost function --
    def cost_function(self, controls_flat, t_now):
        states, controls = rollout(self.state, controls_flat, self.horizon, self.dt)
        cost = 0.0

        # gather nearby obstacles / other robots once (position doesn't change within this DE call)
        my_xy = self.state[:2]
        others = self.swarm.get_other_robot_predictions(self.rid)
        external_now = self.swarm.get_external_obstacle_positions(t_now)

        nearby_static = []
        for obs in self.static_obstacles:
            if obs["type"] == "circle":
                cx, cy = obs["center"]
            else:
                cx = float(np.mean([c[0] for c in obs["corners"]]))
                cy = float(np.mean([c[1] for c in obs["corners"]]))
            if np.hypot(cx - my_xy[0], cy - my_xy[1]) <= SENSING_RANGE + 2.0:
                nearby_static.append(obs)

        s_here = self.s_progress
        for k in range(1, self.horizon + 1):
            px, py, ptheta = states[k]

            # --- tracking error ---
            s_here = self.ref_path.closest_s(np.array([px, py]), s_here, search_window=2.5)
            ref_xy = self.ref_path.point_at_s(s_here)
            track_err = np.hypot(px - ref_xy[0], py - ref_xy[1])
            cost += W_TRACK * track_err ** 2

            # --- progress reward (negative cost) ---
            cost -= W_PROGRESS * (s_here - self.s_progress)

            # --- heading alignment ---
            ref_heading = self.ref_path.heading_at_s(s_here)
            heading_err = np.arctan2(np.sin(ptheta - ref_heading), np.cos(ptheta - ref_heading))
            cost += W_HEADING * heading_err ** 2

            # --- static obstacle collision ---
            for obs in nearby_static:
                if obs["type"] == "circle":
                    cx, cy = obs["center"]
                    r = obs["radius"]
                    d = np.hypot(px - cx, py - cy)
                    intrusion = max(r + ROBOT_RADIUS - d, 0.0)
                else:
                    d, inside = point_to_polygon_distance(px, py, obs["corners"])
                    if inside:
                        intrusion = ROBOT_RADIUS + d
                    else:
                        intrusion = max(ROBOT_RADIUS - d, 0.0)
                cost += W_STATIC_COLLISION * intrusion ** 2

            # --- inter-robot dynamic collision ---
            for other_rid, other_traj in others.items():
                if k < len(other_traj):
                    ox, oy = other_traj[k]
                else:
                    ox, oy = other_traj[-1]
                d = np.hypot(px - ox, py - oy)
                if d <= SENSING_RANGE:
                    min_safe = 2 * ROBOT_RADIUS + SAFETY_MARGIN
                    intrusion = max(min_safe - d, 0.0)
                    cost += W_ROBOT_COLLISION * intrusion ** 2

            # --- external dynamic obstacle collision ---
            for (ox, oy) in external_now:
                d = np.hypot(px - ox, py - oy)
                if d <= SENSING_RANGE:
                    min_safe = ROBOT_RADIUS + SAFETY_MARGIN
                    intrusion = max(min_safe - d, 0.0)
                    cost += W_DYNAMIC_OBS_COLLISION * intrusion ** 2

            # --- boundary penalty ---
            bx = max(BOUNDS_MIN[0] - px, 0.0) + max(px - BOUNDS_MAX[0], 0.0)
            by = max(BOUNDS_MIN[1] - py, 0.0) + max(py - BOUNDS_MAX[1], 0.0)
            cost += W_STATIC_COLLISION * (bx ** 2 + by ** 2)

        # --- control effort & smoothness ---
        cost += W_CONTROL_EFFORT * np.sum(controls[:, 0] ** 2 + controls[:, 1] ** 2)
        if self.horizon > 1:
            dctrl = np.diff(controls, axis=0)
            cost += W_CONTROL_SMOOTH * np.sum(dctrl ** 2)

        return cost

    # -- single MPC step --
    def step(self, t_now):
        if self.finished:
            return

        bounds = self.build_bounds()
        n_vars = self.horizon * 2
        is_cold = self.prev_solution is None
        popsize = DE_POPSIZE_COLD if is_cold else DE_POPSIZE_WARM
        maxiter = DE_MAXITER_COLD if is_cold else DE_MAXITER_WARM

        init_population = self.build_init_population(popsize, n_vars)
        de_kwargs = dict(
            strategy="best1bin",
            maxiter=maxiter,
            popsize=popsize,
            tol=DE_TOL,
            mutation=DE_MUTATION,
            recombination=DE_RECOMBINATION,
            seed=None,
            polish=True,
            updating="immediate",
            workers=1,
            args=(t_now,),
        )
        if init_population is not None:
            de_kwargs["init"] = init_population

        result = differential_evolution(self.cost_function, bounds, **de_kwargs)
        self.prev_solution = result.x

        states, controls = rollout(self.state, result.x, self.horizon, self.dt)
        applied_v, applied_omega = controls[0]

        # advance real state by ONE step (MPC: apply first control only)
        self.state = unicycle_step(self.state, controls[0], self.dt)
        self.s_progress = self.ref_path.closest_s(self.state[:2], self.s_progress, search_window=2.5)

        # publish predicted trajectory (xy only) for other robots to react to
        predicted_xy = states[:, :2]
        self.swarm.update_robot(self.rid, self.state.copy(), predicted_xy)

        # log this step
        self.log_rows.append({
            "step": self.step_count,
            "time": round(t_now, 4),
            "x": self.state[0], "y": self.state[1], "theta": self.state[2],
            "v": applied_v, "omega": applied_omega,
            "s_progress": self.s_progress,
            "cost": result.fun,
        })

        self.step_count += 1

        print(f"[{self.rid}] step {self.step_count:4d} | t={t_now:6.2f}s | "
              f"pos=({self.state[0]:6.3f}, {self.state[1]:6.3f}) | "
              f"theta={self.state[2]:6.3f} rad | v={applied_v:6.3f} | omega={applied_omega:6.3f} | "
              f"s={self.s_progress:6.3f}/{self.ref_path.total_length:6.3f} | cost={result.fun:8.3f}")

        # check goal
        dist_to_goal = np.hypot(self.state[0] - self.ref_path.goal_point[0],
                                 self.state[1] - self.ref_path.goal_point[1])
        if dist_to_goal <= GOAL_TOLERANCE or self.s_progress >= self.ref_path.total_length - 1e-3:
            self.finished = True
            self.swarm.mark_finished(self.rid)

    def write_csv(self, out_dir=CSV_OUTPUT_DIR):
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{self.rid}_mpc_log.csv")
        fieldnames = ["step", "time", "x", "y", "theta", "v", "omega", "s_progress", "cost"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.log_rows:
                writer.writerow(row)
        print(f"[{self.rid}] wrote {len(self.log_rows)} rows to {path}")
        return path


# ----------------------------------------------------------------------
# 6. Multi-robot orchestration (threaded, synchronized per global timestep)
# ----------------------------------------------------------------------

def run_multi_robot_de_mpc(robot_configs, external_dynamic_obstacles=None, dt=DT,
                            horizon=HORIZON, max_steps=MAX_STEPS):
    """
    robot_configs: list of dicts, one per robot, each with:
        {
          "robot_id": str,
          "map_json": path to map-config JSON (obstacles),
          "control_points_json": path to manual_control_points.json (reference path)
        }
    For N robots this requires 2*N files total (N map jsons + N control-point jsons).

    external_dynamic_obstacles: list of callables t -> (x, y), or None/[] to
        disable external dynamic obstacles entirely (only other robots are
        then treated as dynamic obstacles).
    """
    robot_ids = [cfg["robot_id"] for cfg in robot_configs]
    swarm_state = SharedSwarmState(robot_ids, external_dynamic_obstacles)

    robots = []
    for cfg in robot_configs:
        static_obstacles, _ = load_static_obstacles(cfg["map_json"])
        ref_path = ReferencePath(cfg["control_points_json"])
        robot = DE_MPC_Robot(cfg["robot_id"], ref_path, static_obstacles, swarm_state, dt=dt, horizon=horizon)
        # publish initial position/prediction before the loop starts so other
        # robots' first cost evaluation has something to react to
        swarm_state.update_robot(robot.rid, robot.state.copy(),
                                  np.repeat(robot.state[:2][None, :], horizon + 1, axis=0))
        robots.append(robot)

    print(f"Loaded {len(robots)} robot(s). Running sequential (non-threaded) DE-MPC simulation...")
    start_time = time.perf_counter()

    for global_step in range(max_steps):
        if all(r.finished for r in robots):
            print(f"All robots reached their goals at global step {global_step}.")
            break

        t_now = global_step * dt
        for robot in robots:
            if not robot.finished:
                robot.step(t_now)

        if global_step % 25 == 0:
            statuses = ", ".join(
                f"{r.rid}:s={r.s_progress:.2f}/{r.ref_path.total_length:.2f}" for r in robots
            )
            print(f"  --- global step {global_step} summary: {statuses} ---")
    else:
        print(f"Reached MAX_STEPS={max_steps} without all robots finishing.")

    end_time = time.perf_counter()
    print(f"Simulation finished in {end_time - start_time:.2f}s (wall clock).")

    csv_paths = [robot.write_csv() for robot in robots]
    return robots, csv_paths



# ----------------------------------------------------------------------
# 7. Example usage
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Example: 3 robots -> 6 files total (3 map configs + 3 control-point files).
    # Adjust paths to your actual map JSONs (the ones with "obstacles").
    robot_configs = [
        # {
        #     "robot_id": "robot_1",
        #     "map_json": "maps/map_001_robot_1.json",
        #     "control_points_json": "solves/multi/map_001_robot_1_manual_control_points.json",
        # },
        {
            "robot_id": "robot_2",
            "map_json": "maps/map_001_robot_2.json",
            "control_points_json": "solves/multi/map_001_robot_2_manual_control_points.json",
        },
    #     {
    #         "robot_id": "robot_3",
    #         "map_json": "maps/map_001_robot_3.json",
    #         "control_points_json": "solves/multi/map_001_robot_3_manual_control_points.json",
    #     },
    ]

    # Example of an OPTIONAL external dynamic obstacle (a point moving in a
    # straight line). Pass [] or None to disable external dynamic obstacles
    # entirely (only other robots will then be treated as dynamic obstacles).
    def example_moving_obstacle(t):
        return (6.0 + 0.3 * t, 6.0)

    EXTERNAL_DYNAMIC_OBSTACLES = []   # e.g. [example_moving_obstacle] to enable, [] to omit

    run_multi_robot_de_mpc(robot_configs, external_dynamic_obstacles=EXTERNAL_DYNAMIC_OBSTACLES)
