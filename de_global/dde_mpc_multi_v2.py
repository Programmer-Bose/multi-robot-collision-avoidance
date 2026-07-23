"""
dde_mpc_multi_v2.py
--------------------
Multi-robot Differential-Evolution based Model Predictive Control (DE-MPC).

CHANGES FROM ORIGINAL (dde_mpc_multi.py):

1. NO GLOBAL PATH STITCHING.
   Instead of concatenating every segment into one continuous B-spline /
   arc-length path, each robot now tracks a SEQUENCE OF SUB-GOALS (task
   points). Each segment of manual_control_points.json is treated as its
   own independent local reference path (`SubGoalSegment`). The robot only
   tracks the CURRENT segment. Once it gets within GOAL_TOLERANCE of that
   segment's end point (a task point), it advances to the next segment's
   fresh local path. This avoids long-horizon arc-length continuity errors
   across segment joins and makes "next task point" explicit and easy to
   reason about / debug.

2. HEAD-ON DEADLOCK RESOLUTION VIA RIGHT-OF-WAY PRIORITY.
   Previously, two robots meeting head-on could both push back against each
   other symmetrically in the cost function and freeze in place forever
   (a classic symmetric Nash-equilibrium deadlock in reciprocal collision
   avoidance).

   Fix: every robot broadcasts how many task points (sub-goals) it has left
   to visit, via SharedSwarmState. A robot with MORE remaining tasks gets
   right-of-way (higher priority) over a robot with fewer remaining tasks.

   When a near head-on / stalled encounter is detected (robots close,
   headings roughly opposing, both moving slowly), the LOWER-priority robot
   receives:
     - an amplified inter-robot collision weight (must clear more space)
     - a small "yield" bias encouraging it to slow down / step laterally
       off the reference path
   while the HIGHER-priority robot's cost is left basically unchanged, so
   it keeps pushing straight through. This breaks the symmetry that caused
   the freeze.
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
SAFETY_MARGIN = 0.10

DT = 0.2
HORIZON = 8
MAX_STEPS = 400

V_MAX = 1.2
V_MIN = -0.6
OMEGA_MAX = 1.8

GOAL_TOLERANCE = 0.25             # distance to a sub-goal (task point) considered "reached"
WAYPOINT_ADVANCE_LOOKAHEAD = 0.4

SENSING_RANGE = 3.5

# --- Cost weights ---
W_TRACK = 6.0
W_PROGRESS = 2.0
W_CONTROL_EFFORT = 0.05
W_CONTROL_SMOOTH = 0.3
W_STATIC_COLLISION = 900.0
W_ROBOT_COLLISION = 900.0
W_DYNAMIC_OBS_COLLISION = 900.0
W_HEADING = 0.5

# --- Right-of-way / deadlock-breaking parameters ---
DEADLOCK_DIST_THRESH = 1.2         # consider "encounter" if within this range
DEADLOCK_HEADING_DOT_THRESH = -0.4 # headings roughly opposing (cos(angle) below this)
DEADLOCK_SPEED_THRESH = 0.15       # both robots slow => candidate stall
YIELD_COLLISION_MULTIPLIER = 2.5   # extra weight on inter-robot term for the yielding robot
YIELD_BRAKE_WEIGHT = 1.5           # penalty for the yielding robot moving forward while yielding
YIELD_LATERAL_BONUS = 1.0          # small reward for yielding robot deviating off-path (creates room)

# --- Differential Evolution settings ---
DE_POPSIZE_COLD = 30
DE_POPSIZE_WARM = 15
DE_MAXITER_COLD = 40
DE_MAXITER_WARM = 12
DE_MUTATION = (0.4, 1.0)
DE_RECOMBINATION = 0.85
DE_TOL = 1e-6
WARM_JITTER_STD = 0.75
WARM_START_FRACTION = 0.7

CSV_OUTPUT_DIR = "mpc_logs"

# ----------------------------------------------------------------------
# 1. Static map loading (obstacles) -- unchanged
# ----------------------------------------------------------------------

def load_static_obstacles(json_path):
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
# 2. Reference path: PER-SEGMENT sub-goal paths (NO global stitching)
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


class SubGoalSegment:
    """A single, self-contained local reference path for ONE segment
    (start_point -> end_point / task point). Arc-length parameterization is
    LOCAL to this segment only -- it does not know about any other segment."""

    def __init__(self, seg_data, samples=60):
        start = np.array(seg_data["start_point"])
        end = np.array(seg_data["end_point"])
        free_pts = np.array(seg_data["control_points"])
        full_ctrl = np.vstack([start, free_pts, end])

        self.points = bspline_curve(full_ctrl, samples)
        diffs = np.diff(self.points, axis=0)
        seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
        self.cum_s = np.concatenate([[0.0], np.cumsum(seg_lens)])
        self.total_length = self.cum_s[-1]

        self.start_point = self.points[0].copy()
        self.goal_point = self.points[-1].copy()  # this segment's task point

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


class SubGoalPath:
    """Holds the ORDERED list of SubGoalSegment sub-goals for a robot and
    tracks which one is currently active. This REPLACES the old stitched
    ReferencePath: there is no global arc-length, only "current segment"."""

    def __init__(self, control_points_json, samples_per_segment=60):
        with open(control_points_json, "r") as f:
            data = json.load(f)

        self.segments = [SubGoalSegment(seg, samples_per_segment) for seg in data["segments"]]
        self.n_segments = len(self.segments)
        self.active_idx = 0

        self.start_point = self.segments[0].start_point.copy()
        self.goal_point = self.segments[-1].goal_point.copy()  # final overall goal

    @property
    def current(self):
        return self.segments[self.active_idx]

    def remaining_tasks(self):
        """Number of sub-goals (including current) not yet reached."""
        return self.n_segments - self.active_idx

    def is_final_segment(self):
        return self.active_idx == self.n_segments - 1

    def advance_if_reached(self, xy, tolerance=GOAL_TOLERANCE):
        """Checks if the robot is close enough to the CURRENT segment's task
        point; if so, and it's not the last segment, moves on to the next
        segment (fresh local path, s resets to 0). Returns True if it just
        advanced (a task point was reached)."""
        seg = self.current
        dist = np.hypot(xy[0] - seg.goal_point[0], xy[1] - seg.goal_point[1])
        if dist <= tolerance and not self.is_final_segment():
            self.active_idx += 1
            return True
        return False

    def reached_final_goal(self, xy, tolerance=GOAL_TOLERANCE):
        if not self.is_final_segment():
            return False
        seg = self.current
        dist = np.hypot(xy[0] - seg.goal_point[0], xy[1] - seg.goal_point[1])
        return dist <= tolerance


# ----------------------------------------------------------------------
# 3. Robot state / kinematics -- unchanged
# ----------------------------------------------------------------------

def unicycle_step(state, control, dt):
    x, y, theta = state
    v, omega = control
    new_theta = theta + omega * dt
    new_x = x + v * np.cos(theta) * dt
    new_y = y + v * np.sin(theta) * dt
    return np.array([new_x, new_y, new_theta])


def rollout(state0, controls_flat, horizon, dt):
    controls = controls_flat.reshape(horizon, 2)
    states = np.zeros((horizon + 1, 3))
    states[0] = state0
    for i in range(horizon):
        states[i + 1] = unicycle_step(states[i], controls[i], dt)
    return states, controls


# ----------------------------------------------------------------------
# 4. Shared swarm state -- now also tracks remaining-task priority
# ----------------------------------------------------------------------

class SharedSwarmState:
    """Blackboard holding each robot's latest known position, predicted
    trajectory, remaining-task count (for right-of-way priority), and any
    externally-supplied dynamic obstacles."""

    def __init__(self, robot_ids, external_dynamic_obstacles=None):
        self.positions = {rid: None for rid in robot_ids}
        self.predicted_traj = {rid: None for rid in robot_ids}
        self.finished = {rid: False for rid in robot_ids}
        self.remaining_tasks = {rid: None for rid in robot_ids}   # rid -> int
        self.external_dynamic_obstacles = external_dynamic_obstacles or []

    def update_robot(self, rid, position, predicted_xy, remaining_tasks):
        self.positions[rid] = position
        self.predicted_traj[rid] = predicted_xy
        self.remaining_tasks[rid] = remaining_tasks

    def mark_finished(self, rid):
        self.finished[rid] = True

    def get_other_robot_predictions(self, self_rid):
        others = {}
        for rid, traj in self.predicted_traj.items():
            if rid != self_rid and traj is not None:
                others[rid] = traj.copy()
        return others

    def get_other_robot_remaining_tasks(self, self_rid):
        return {rid: n for rid, n in self.remaining_tasks.items()
                if rid != self_rid and n is not None}

    def has_right_of_way(self, self_rid, other_rid):
        """More remaining tasks => higher priority => right of way.
        Ties broken deterministically by robot_id string so both robots
        agree on the same outcome without communication ambiguity."""
        my_tasks = self.remaining_tasks.get(self_rid)
        other_tasks = self.remaining_tasks.get(other_rid)
        if my_tasks is None or other_tasks is None:
            return True  # unknown -> don't yield
        if my_tasks != other_tasks:
            return my_tasks > other_tasks
        return self_rid > other_rid  # deterministic tie-break

    def get_external_obstacle_positions(self, t):
        if not self.external_dynamic_obstacles:
            return []
        return [fn(t) for fn in self.external_dynamic_obstacles]


# ----------------------------------------------------------------------
# 5. DE-MPC controller for a single robot
# ----------------------------------------------------------------------

class DE_MPC_Robot:
    def __init__(self, robot_id, sub_goal_path: SubGoalPath, static_obstacles,
                 swarm_state: SharedSwarmState, dt=DT, horizon=HORIZON):
        self.rid = robot_id
        self.path = sub_goal_path
        self.static_obstacles = static_obstacles
        self.swarm = swarm_state
        self.dt = dt
        self.horizon = horizon

        start_xy = self.path.start_point
        start_heading = self.path.current.heading_at_s(0.0)
        self.state = np.array([start_xy[0], start_xy[1], start_heading])

        self.s_progress = 0.0     # local arc-length WITHIN current segment only
        self.step_count = 0
        self.prev_solution = None
        self.finished = False
        self.log_rows = []

    def build_bounds(self):
        return [(V_MIN, V_MAX), (-OMEGA_MAX, OMEGA_MAX)] * self.horizon

    def build_init_population(self, popsize, n_vars):
        if self.prev_solution is None:
            return None

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

    def _detect_deadlock_partners(self):
        """Identify other robots that are currently in a near head-on /
        stalled encounter with this robot. Returns dict rid -> yield_flag
        (True = I must yield to them, False = they must yield to me)."""
        my_xy = self.state[:2]
        my_theta = self.state[2]
        my_heading_vec = np.array([np.cos(my_theta), np.sin(my_theta)])

        others_remaining = self.swarm.get_other_robot_remaining_tasks(self.rid)
        encounters = {}
        for other_rid in others_remaining:
            other_pos = self.swarm.positions.get(other_rid)
            if other_pos is None:
                continue
            ox, oy, otheta = other_pos
            d = np.hypot(my_xy[0] - ox, my_xy[1] - oy)
            if d > DEADLOCK_DIST_THRESH:
                continue

            other_heading_vec = np.array([np.cos(otheta), np.sin(otheta)])
            heading_dot = float(np.dot(my_heading_vec, other_heading_vec))
            if heading_dot > DEADLOCK_HEADING_DOT_THRESH:
                continue  # not roughly opposing

            i_yield = not self.swarm.has_right_of_way(self.rid, other_rid)
            encounters[other_rid] = i_yield
        return encounters

    def cost_function(self, controls_flat, t_now):
        states, controls = rollout(self.state, controls_flat, self.horizon, self.dt)
        cost = 0.0

        my_xy = self.state[:2]
        others = self.swarm.get_other_robot_predictions(self.rid)
        external_now = self.swarm.get_external_obstacle_positions(t_now)
        deadlock_partners = self._detect_deadlock_partners()
        i_must_yield_overall = any(deadlock_partners.values())

        nearby_static = []
        for obs in self.static_obstacles:
            if obs["type"] == "circle":
                cx, cy = obs["center"]
            else:
                cx = float(np.mean([c[0] for c in obs["corners"]]))
                cy = float(np.mean([c[1] for c in obs["corners"]]))
            if np.hypot(cx - my_xy[0], cy - my_xy[1]) <= SENSING_RANGE + 2.0:
                nearby_static.append(obs)

        seg = self.path.current
        s_here = self.s_progress
        for k in range(1, self.horizon + 1):
            px, py, ptheta = states[k]

            # --- tracking error (LOCAL to current segment only) ---
            s_here = seg.closest_s(np.array([px, py]), s_here, search_window=2.5)
            ref_xy = seg.point_at_s(s_here)
            track_err = np.hypot(px - ref_xy[0], py - ref_xy[1])
            cost += W_TRACK * track_err ** 2

            # --- progress reward ---
            progress_gain = (s_here - self.s_progress)
            if i_must_yield_overall:
                # yielding robot: suppress/penalize forward progress reward
                # so it prefers to brake / hang back instead of pushing on
                cost += YIELD_BRAKE_WEIGHT * max(progress_gain, 0.0) ** 2
                cost -= 0.25 * W_PROGRESS * progress_gain
            else:
                cost -= W_PROGRESS * progress_gain

            # --- heading alignment ---
            ref_heading = seg.heading_at_s(s_here)
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

            # --- inter-robot dynamic collision (with right-of-way weighting) ---
            for other_rid, other_traj in others.items():
                if k < len(other_traj):
                    ox, oy = other_traj[k]
                else:
                    ox, oy = other_traj[-1]
                d = np.hypot(px - ox, py - oy)
                if d <= SENSING_RANGE:
                    min_safe = 2 * ROBOT_RADIUS + SAFETY_MARGIN
                    intrusion = max(min_safe - d, 0.0)

                    weight = W_ROBOT_COLLISION
                    if deadlock_partners.get(other_rid) is True:
                        # I must yield to this specific robot: treat it as a
                        # much "bigger" obstacle, forcing me off the path
                        weight *= YIELD_COLLISION_MULTIPLIER
                        # small bonus for lateral deviation off the reference
                        # line (creates passing room instead of freezing)
                        lateral_dev = np.hypot(px - ref_xy[0], py - ref_xy[1])
                        cost -= YIELD_LATERAL_BONUS * min(lateral_dev, 1.0)

                    cost += weight * intrusion ** 2

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

        cost += W_CONTROL_EFFORT * np.sum(controls[:, 0] ** 2 + controls[:, 1] ** 2)
        if self.horizon > 1:
            dctrl = np.diff(controls, axis=0)
            cost += W_CONTROL_SMOOTH * np.sum(dctrl ** 2)

        return cost

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

        self.state = unicycle_step(self.state, controls[0], self.dt)
        seg = self.path.current
        self.s_progress = seg.closest_s(self.state[:2], self.s_progress, search_window=2.5)

        # --- sub-goal advancement: check if current task point reached ---
        advanced = self.path.advance_if_reached(self.state[:2])
        if advanced:
            self.s_progress = 0.0          # reset local progress for new segment
            self.prev_solution = None       # cold-start fresh segment's optimization
            print(f"[{self.rid}] reached task point -> advancing to segment "
                  f"{self.path.active_idx + 1}/{self.path.n_segments}")

        predicted_xy = states[:, :2]
        self.swarm.update_robot(self.rid, self.state.copy(), predicted_xy,
                                 self.path.remaining_tasks())

        self.log_rows.append({
            "step": self.step_count,
            "time": round(t_now, 4),
            "x": self.state[0], "y": self.state[1], "theta": self.state[2],
            "v": applied_v, "omega": applied_omega,
            "s_progress": self.s_progress,
            "active_segment": self.path.active_idx,
            "remaining_tasks": self.path.remaining_tasks(),
            "cost": result.fun,
        })

        self.step_count += 1

        print(f"[{self.rid}] step {self.step_count:4d} | t={t_now:6.2f}s | "
              f"pos=({self.state[0]:6.3f}, {self.state[1]:6.3f}) | "
              f"seg={self.path.active_idx}/{self.path.n_segments - 1} | "
              f"remaining_tasks={self.path.remaining_tasks()} | "
              f"v={applied_v:6.3f} | omega={applied_omega:6.3f} | cost={result.fun:8.3f}")

        if self.path.reached_final_goal(self.state[:2]):
            self.finished = True
            self.swarm.mark_finished(self.rid)

    def write_csv(self, out_dir=CSV_OUTPUT_DIR):
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{self.rid}_mpc_log.csv")
        fieldnames = ["step", "time", "x", "y", "theta", "v", "omega",
                      "s_progress", "active_segment", "remaining_tasks", "cost"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.log_rows:
                writer.writerow(row)
        print(f"[{self.rid}] wrote {len(self.log_rows)} rows to {path}")
        return path


# ----------------------------------------------------------------------
# 6. Multi-robot orchestration (sequential per global timestep)
# ----------------------------------------------------------------------

def run_multi_robot_de_mpc(robot_configs, external_dynamic_obstacles=None, dt=DT,
                            horizon=HORIZON, max_steps=MAX_STEPS):
    robot_ids = [cfg["robot_id"] for cfg in robot_configs]
    swarm_state = SharedSwarmState(robot_ids, external_dynamic_obstacles)

    robots = []
    for cfg in robot_configs:
        static_obstacles, _ = load_static_obstacles(cfg["map_json"])
        sub_goal_path = SubGoalPath(cfg["control_points_json"])
        robot = DE_MPC_Robot(cfg["robot_id"], sub_goal_path, static_obstacles,
                              swarm_state, dt=dt, horizon=horizon)
        swarm_state.update_robot(robot.rid, robot.state.copy(),
                                  np.repeat(robot.state[:2][None, :], horizon + 1, axis=0),
                                  robot.path.remaining_tasks())
        robots.append(robot)

    print(f"Loaded {len(robots)} robot(s). Running sequential DE-MPC simulation "
          f"with sub-goal segmentation + right-of-way deadlock resolution...")
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
                f"{r.rid}:seg={r.path.active_idx}/{r.path.n_segments - 1},"
                f"remaining={r.path.remaining_tasks()}" for r in robots
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
    robot_configs = [
        {
            "robot_id": "robot_1",
            "map_json": "maps/map_011_robot_1.json",
            "control_points_json": "solves/multi/map_011_robot_1_manual_control_points.json",
        },
        {
            "robot_id": "robot_2",
            "map_json": "maps/map_011_robot_2.json",
            "control_points_json": "solves/multi/map_011_robot_2_manual_control_points.json",
        },
        {
            "robot_id": "robot_3",
            "map_json": "maps/map_011_robot_3.json",
            "control_points_json": "solves/multi/map_011_robot_3_manual_control_points.json",
        },
    ]

    def example_moving_obstacle(t):
        return (6.0 + 0.3 * t, 6.0)

    EXTERNAL_DYNAMIC_OBSTACLES = []  # e.g. [example_moving_obstacle] to enable

    run_multi_robot_de_mpc(robot_configs, external_dynamic_obstacles=EXTERNAL_DYNAMIC_OBSTACLES)
