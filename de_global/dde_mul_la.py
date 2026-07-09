import numpy as np
from scipy.optimize import differential_evolution
from scipy.interpolate import BSpline
import shapely
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle
import time
import json
import os
import datetime
from shapely.geometry import LineString

# ----------------------------------------------------------------------
# 0. HYPERPARAMETERS (all tunable knobs live here)
# ----------------------------------------------------------------------

# --- Optimization variable mode ---
# "delta"  : DE evolves offsets (deltas) added to a straight-line baseline
#            of control points (original behavior).
# "direct" : DE evolves the absolute (x, y) coordinates of the control
#            points themselves, with bounds fixed to the arena bounds.
CONTROL_POINT_MODE = "direct"   # "delta" or "direct"

# --- B-spline shape ---
N_CONTROL_PER_SEGMENT = 6          # number of FREE (interior) B-spline control points per segment
BSPLINE_DEGREE = 3                 # cubic B-spline
N_SAMPLES_PER_SEGMENT = 50         # samples drawn along each segment's curve
N_VARS_PER_SEGMENT = N_CONTROL_PER_SEGMENT * 2

# --- Map / world scaling ---
TARGET_SCALE = 12.0                 # world is rescaled so the larger map dimension = this value
BOUNDS_MIN = np.array([0.0, 0.0])
BOUNDS_MAX = np.array([TARGET_SCALE, TARGET_SCALE])

# --- Robot / cost weights ---
ROBOT_RADIUS = 0.3
W_LENGTH = 1.0
W_COLLISION = 800.0
W_CURVATURE = 0.5
W_BOUNDARY = 800.0                  # penalty weight for leaving the [0,12]x[0,12] arena (same order as collision)
W_INTRUSION_GRADIENT = 0.05   # small continuous nudge so DE can "feel" its way out of a collision, doesn't change the 800 floor meaningfully
# --- Population-size decay across generations (per-segment) ---
# The DE run for each segment is split into POPSIZE_DECAY_STAGES stages of
# roughly equal generation count. popsize shrinks geometrically from
# DE_POPSIZE_INITIAL down to DE_POPSIZE_FINAL across the stages, carrying the
# fittest individuals of each stage forward as the next stage's warm start.
DE_POPSIZE_INITIAL = 50
DE_POPSIZE_FINAL = 50
DE_POPSIZE_DECAY_STAGES = 1         # number of shrink steps (1 = no decay, uses DE_POPSIZE_INITIAL throughout)

# --- Warm starting (population seeding across segments) ---
WARM_START_FRACTION = 0.5           # fraction of the new population seeded from previous best solution
WARM_START_NOISE_STD = 0.85         # std-dev of Gaussian jitter applied to seeded individuals

# --- Differential Evolution settings ---
DE_MAXITER = 50                     # total generations per segment, spread across the decay stages above
DE_STRATEGY = "best1bin"
DE_MUTATION = (0.4, 1.2)
DE_RECOMBINATION = 0.9
DE_TOL = 1e-8
DE_SEED_BASE = 17                   # per-segment seed = DE_SEED_BASE + segment_index

# --- Live visualization ---
LIVE_PLOT_ENABLED = True             # if True, opens a live matplotlib window updated every generation
LIVE_PLOT_PAUSE = 0.001              # seconds paused per redraw (matplotlib needs this to flush the GUI event loop)


FAN_OUT_MAGNITUDE_FRACTION = 0.5   # how far (as a fraction of segment length) the fanned bulges push out
FAN_OUT_NOISE_STD = 0.3            # small jitter on top of each fan direction


# --- Final-solution export / early termination ---
EARLY_TERMINATE_AFTER_FIRST_SEGMENT = False  # if True, stop the whole solve once segment 0 converges
CONTROL_POINTS_OUTPUT_DIR = "solves"         # combined all-segments control-point JSON is written here

# ----------------------------------------------------------------------
# 1. Global Problem State
# ----------------------------------------------------------------------

WAYPOINTS = []
OBSTACLES = []
N_SEGMENTS = 0
CURRENT_MAP_NAME = "unnamed_map"

# ----------------------------------------------------------------------
# 2. Map Loading & Coordinate Transformation
# ----------------------------------------------------------------------

def load_map_config(json_path):
    global WAYPOINTS, OBSTACLES, N_SEGMENTS, BOUNDS_MIN, BOUNDS_MAX, CURRENT_MAP_NAME

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Configuration file {json_path} not found.")

    CURRENT_MAP_NAME = os.path.splitext(os.path.basename(json_path))[0]

    with open(json_path, 'r') as f:
        data = json.load(f)

    meta_key = "map_metadata" if "map_metadata" in data else "robot_metadata"
    orig_w, orig_h = data[meta_key]["size"]
    scale_factor = TARGET_SCALE / orig_w

    def scale_pt(pt):
        return np.array([pt[0] * scale_factor, (orig_h - pt[1]) * scale_factor])

    WAYPOINTS.clear()
    WAYPOINTS.append(scale_pt(data["start_position"]))

    for task_id in data["task_sequence"]:
        WAYPOINTS.append(scale_pt(data["task_points"][str(task_id)]))

    if data.get("goal_position"):
        WAYPOINTS.append(scale_pt(data["goal_position"]))

    N_SEGMENTS = len(WAYPOINTS) - 1
    BOUNDS_MAX = np.array([TARGET_SCALE, TARGET_SCALE])

    OBSTACLES.clear()
    for obs in data["obstacles"]:
        obs_type = obs["type"]
        cx, cy = scale_pt(obs["position"])

        if obs_type == "circle":
            r_scaled = obs["radius"] * scale_factor
            OBSTACLES.append({"type": "circle", "center": (cx, cy), "radius": r_scaled})

        elif obs_type == "square":
            h = (obs["size"] * scale_factor) / 2.0
            geom = Polygon([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)])
            OBSTACLES.append({"type": "polygon", "shape": geom})

        elif obs_type == "rectangle":
            hw = (obs["width"] * scale_factor) / 2.0
            hh = (obs["height"] * scale_factor) / 2.0
            geom = Polygon([(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)])
            OBSTACLES.append({"type": "polygon", "shape": geom})

        elif obs_type == "u_shape":
            h = (obs["size"] * scale_factor) / 2.0
            t_scaled = obs["thickness"] * scale_factor
            left_arm = Polygon([(cx - h, cy - h), (cx - h + t_scaled, cy - h), (cx - h + t_scaled, cy + h), (cx - h, cy + h)])
            right_arm = Polygon([(cx + h - t_scaled, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx + h - t_scaled, cy + h)])
            bottom_bar = Polygon([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy - h + t_scaled), (cx - h, cy - h + t_scaled)])
            u_geom = unary_union([left_arm, right_arm, bottom_bar])
            OBSTACLES.append({"type": "polygon", "shape": u_geom})

# ----------------------------------------------------------------------
# 3. B-Spline Curve Construction
# ----------------------------------------------------------------------

def make_clamped_knot_vector(n_ctrl_pts, degree):
    """Open/clamped uniform knot vector so the curve interpolates the first
    and last control points (standard for path-planning B-splines)."""
    n_internal = n_ctrl_pts - degree - 1
    if n_internal > 0:
        internal_knots = np.linspace(0, 1, n_internal + 2)[1:-1]
    else:
        internal_knots = np.array([])
    knots = np.concatenate((
        np.zeros(degree + 1),
        internal_knots,
        np.ones(degree + 1)
    ))
    return knots

def bspline_curve(control_points, n_samples, degree=BSPLINE_DEGREE):
    control_points = np.asarray(control_points)
    n_ctrl_pts = len(control_points)
    k = min(degree, n_ctrl_pts - 1)  # guard against too few points
    knots = make_clamped_knot_vector(n_ctrl_pts, k)
    t = np.linspace(0.0, 1.0, n_samples)
    spline_x = BSpline(knots, control_points[:, 0], k)
    spline_y = BSpline(knots, control_points[:, 1], k)
    curve = np.column_stack([spline_x(t), spline_y(t)])
    return curve

def baseline_control_points(seg_start, seg_end):
    segment_vec = seg_end - seg_start
    if np.allclose(segment_vec, 0):
        return np.repeat(seg_start[None, :], N_CONTROL_PER_SEGMENT, axis=0)

    t_values = np.linspace(
        1.0 / (N_CONTROL_PER_SEGMENT + 1),
        N_CONTROL_PER_SEGMENT / (N_CONTROL_PER_SEGMENT + 1),
        N_CONTROL_PER_SEGMENT,
    )
    return np.array([seg_start + t * segment_vec for t in t_values])

def vars_to_control_points(flat_vars, seg_start, seg_end):
    """Maps the raw DE decision vector to the N_CONTROL_PER_SEGMENT free
    (interior) control points, according to CONTROL_POINT_MODE.

    - "delta"  : flat_vars are OFFSETS added to the straight-line baseline.
    - "direct" : flat_vars ARE the absolute (x, y) control point coordinates.
    """
    points = flat_vars.reshape(N_CONTROL_PER_SEGMENT, 2)
    if CONTROL_POINT_MODE == "delta":
        return baseline_control_points(seg_start, seg_end) + points
    elif CONTROL_POINT_MODE == "direct":
        return points
    else:
        raise ValueError(f"Unknown CONTROL_POINT_MODE: {CONTROL_POINT_MODE}")

# Backwards-compatible alias (old name used throughout the original script)
def delta_to_control_points(flat_deltas, seg_start, seg_end):
    return vars_to_control_points(flat_deltas, seg_start, seg_end)

def segment_curve_from_deltas(flat_vars, seg_start, seg_end):
    free_points = vars_to_control_points(flat_vars, seg_start, seg_end)
    control_points = np.vstack([seg_start, free_points, seg_end])
    return bspline_curve(control_points, N_SAMPLES_PER_SEGMENT)

# ----------------------------------------------------------------------
# 4. Cost Functions
# ----------------------------------------------------------------------

# def collision_penalty(curve):
#     path_line = LineString(curve)
#     swept_path = path_line.buffer(ROBOT_RADIUS)  # the actual area the robot's body sweeps

#     collided = False
#     intrusion_sum = 0.0
#     for obstacle in OBSTACLES:
#         if obstacle["type"] == "circle":
#             cx, cy = obstacle["center"]
#             r = obstacle["radius"]
#             obs_geom = Point(cx, cy).buffer(r)
#         else:
#             obs_geom = obstacle["shape"]

#         if swept_path.intersects(obs_geom):
#             collided = True
#             intrusion_sum += swept_path.intersection(obs_geom).area

#     return (1.0 if collided else 0.0) + W_INTRUSION_GRADIENT * intrusion_sum

def collision_penalty(curve):
    xs = curve[:, 0]
    ys = curve[:, 1]
    pts = shapely.points(xs, ys)

    penalty = 0.0
    for obstacle in OBSTACLES:
        if obstacle["type"] == "circle":
            cx, cy = obstacle["center"]
            r = obstacle["radius"]
            d = np.hypot(xs - cx, ys - cy)
            safe_r = r + ROBOT_RADIUS
            intrusion = np.clip(safe_r - d, 0, None)
        else:
            poly = obstacle["shape"]
            d = shapely.distance(pts, poly)
            inside = shapely.contains(poly, pts)
            boundary_d = shapely.boundary(poly)
            d_to_boundary = shapely.distance(pts, boundary_d)
            outside_intrusion = np.clip(ROBOT_RADIUS - d, 0, None)
            inside_intrusion = ROBOT_RADIUS + d_to_boundary
            intrusion = np.where(inside, inside_intrusion, outside_intrusion)
        penalty += np.sum(intrusion ** 2)
    return penalty


def boundary_penalty(curve):
    """High penalty for any curve sample that leaves the [0,12] x [0,12]
    arena, mirroring the shape of collision_penalty (squared intrusion)."""
    xs = curve[:, 0]
    ys = curve[:, 1]
    intrusion_x = np.clip(BOUNDS_MIN[0] - xs, 0, None) + np.clip(xs - BOUNDS_MAX[0], 0, None)
    intrusion_y = np.clip(BOUNDS_MIN[1] - ys, 0, None) + np.clip(ys - BOUNDS_MAX[1], 0, None)
    return np.sum(intrusion_x ** 2) + np.sum(intrusion_y ** 2)

def path_length(curve):
    diffs = np.diff(curve, axis=0)
    return np.sum(np.hypot(diffs[:, 0], diffs[:, 1]))

def curvature_penalty(curve):
    if len(curve) < 3:
        return 0.0
    v1 = curve[1:-1] - curve[:-2]
    v2 = curve[2:] - curve[1:-1]
    n1 = np.linalg.norm(v1, axis=1) + 1e-9
    n2 = np.linalg.norm(v2, axis=1) + 1e-9
    cos_angle = np.clip(np.sum(v1 * v2, axis=1) / (n1 * n2), -1.0, 1.0)
    return np.sum(np.arccos(cos_angle) ** 2)

def segment_cost_function(flat_vars, seg_start, seg_end):
    curve = segment_curve_from_deltas(flat_vars, seg_start, seg_end)
    cost = 0.0
    cost += W_LENGTH * path_length(curve)
    cost += W_COLLISION * collision_penalty(curve)
    cost += W_CURVATURE * curvature_penalty(curve)
    cost += W_BOUNDARY * boundary_penalty(curve)
    return cost

# ----------------------------------------------------------------------
# 5. Checkpoint Export (start, end, 6 control points)
# ----------------------------------------------------------------------

def export_all_segments_control_points(segment_records, filename=None):
    """Requirement: save the control points for EVERY segment's final
    (last-generation) solution into a SINGLE JSON file, rather than one file
    per checkpoint. `segment_records` is a list of dicts, one per segment,
    each containing start_point, end_point, control_points and cost."""
    os.makedirs(CONTROL_POINTS_OUTPUT_DIR, exist_ok=True)
    if filename is None:
        filename = os.path.join(CONTROL_POINTS_OUTPUT_DIR, f"{CURRENT_MAP_NAME}_control_points.json")

    payload = {
        "map_name": CURRENT_MAP_NAME,
        "control_point_mode": CONTROL_POINT_MODE,
        "num_segments": len(segment_records),
        "segments": segment_records,
    }
    with open(filename, 'w') as f:
        json.dump(payload, f, indent=4)
    print(f"Saved final control points for all segments to {filename}")
    return filename

# ----------------------------------------------------------------------
# 6. Differential Evolution
# ----------------------------------------------------------------------

def build_delta_bounds(seg_start, seg_end):
    """Bounds for the DE decision vector, dependent on CONTROL_POINT_MODE.

    - "delta"  : symmetric offset bounds around 0, scaled to segment length
                 (original behavior).
    - "direct" : each free control point coordinate is bounded by the arena
                 bounds [BOUNDS_MIN, BOUNDS_MAX] directly, since the
                 variables ARE the absolute coordinates.
    """
    if CONTROL_POINT_MODE == "delta":
        segment_length = np.linalg.norm(seg_end - seg_start)
        delta_bound = max(5, 0.8 * segment_length)
        return [(-delta_bound, delta_bound) for _ in range(N_VARS_PER_SEGMENT)]
    elif CONTROL_POINT_MODE == "direct":
        bounds = []
        for _ in range(N_CONTROL_PER_SEGMENT):
            bounds.append((BOUNDS_MIN[0], BOUNDS_MAX[0]))  # x
            bounds.append((BOUNDS_MIN[1], BOUNDS_MAX[1]))  # y
        return bounds
    else:
        raise ValueError(f"Unknown CONTROL_POINT_MODE: {CONTROL_POINT_MODE}")

def build_warm_start_population(prev_best_x, popsize, n_vars, seg_start, seg_end):
    """Warm-start methodology preserved from the original script: seed a
    fraction of the new population from the previous best solution (from the
    first segment, or from the immediately preceding segment), jittered with
    Gaussian noise; fill the remainder randomly / fanned-out."""
    n_individuals = popsize * n_vars
    n_seeded = int(n_individuals * WARM_START_FRACTION)
    n_random = n_individuals - n_seeded

    population = []
    if prev_best_x is not None:
        seed_base = np.asarray(prev_best_x, dtype=float).flatten()
        for _ in range(n_seeded):
            jittered = seed_base + np.random.normal(0.0, WARM_START_NOISE_STD, size=n_vars)
            population.append(jittered)

    if n_random > 0:
        fanned = build_fanned_out_population(n_random, n_vars, seg_start, seg_end)
        population.extend(list(fanned))

    return np.array(population)

def build_fanned_out_population(n_individuals, n_vars, seg_start, seg_end):
    """Used only when there's no previous-segment solution to warm-start
    from. Instead of unstructured per-coordinate noise, generate individuals
    whose control points are pushed out perpendicular to the segment
    baseline by varying angle/magnitude, so the initial population already
    covers 'go around left', 'go around right', 'bulge up', 'bulge down'
    etc. rather than converging on whichever direction the noise happens
    to average toward first.

    In "delta" mode the fan is expressed directly as offsets from 0.
    In "direct" mode the same fan pattern is expressed as an offset added
    to the straight-line baseline control points, so the population still
    spans absolute coordinates that are valid (and clipped to bounds).
    """
    seg_vec = seg_end - seg_start
    seg_len = np.linalg.norm(seg_vec) + 1e-9
    direction = seg_vec / seg_len
    perp = np.array([-direction[1], direction[0]])  # perpendicular to baseline

    base = baseline_control_points(seg_start, seg_end) if CONTROL_POINT_MODE == "direct" else None

    population = []
    n_fans = min(n_individuals, 16)  # number of distinct sweep directions
    for i in range(n_individuals):
        fan_idx = i % n_fans
        angle_frac = fan_idx / n_fans          # 0..1 around the circle
        sign = 1.0 if angle_frac < 0.5 else -1.0
        magnitude = FAN_OUT_MAGNITUDE_FRACTION * seg_len * (0.3 + angle_frac)

        offset = np.zeros((N_CONTROL_PER_SEGMENT, 2))
        for cp in range(N_CONTROL_PER_SEGMENT):
            # every control point in this individual pushed the same way,
            # so the whole curve bulges coherently instead of wiggling
            offset[cp] = sign * magnitude * perp

        offset += np.random.normal(0.0, FAN_OUT_NOISE_STD, size=(N_CONTROL_PER_SEGMENT, 2))

        if CONTROL_POINT_MODE == "direct":
            individual_pts = base + offset
            individual_pts[:, 0] = np.clip(individual_pts[:, 0], BOUNDS_MIN[0], BOUNDS_MAX[0])
            individual_pts[:, 1] = np.clip(individual_pts[:, 1], BOUNDS_MIN[1], BOUNDS_MAX[1])
            individual = individual_pts.flatten()
        else:
            individual = offset.flatten()

        population.append(individual)

    return np.array(population)

def build_popsize_stage_schedule():
    """Geometric decay from DE_POPSIZE_INITIAL to DE_POPSIZE_FINAL across
    DE_POPSIZE_DECAY_STAGES stages, with DE_MAXITER generations split as
    evenly as possible across those stages (remainder given to the last
    stage). Returns a list of (n_generations, popsize) tuples."""
    n_stages = max(1, DE_POPSIZE_DECAY_STAGES)

    if n_stages == 1:
        popsizes = [DE_POPSIZE_INITIAL]
    else:
        ratio = (DE_POPSIZE_FINAL / DE_POPSIZE_INITIAL) ** (1.0 / (n_stages - 1))
        popsizes = [max(4, int(round(DE_POPSIZE_INITIAL * (ratio ** i)))) for i in range(n_stages)]

    base_gens = DE_MAXITER // n_stages
    gens = [base_gens] * n_stages
    gens[-1] += DE_MAXITER - base_gens * n_stages  # remainder to last stage
    gens = [max(1, g) for g in gens]

    return list(zip(gens, popsizes))

class EarlyTerminationSignal(Exception):
    """Raised internally to unwind out of solve_sequential once the first
    segment has converged, if EARLY_TERMINATE_AFTER_FIRST_SEGMENT is set."""
    pass

class LivePlotManager:
    """Keeps a single matplotlib window open across the whole sequential
    solve and updates it every generation, showing the best-so-far curve of
    the segment currently being optimized plus the finalized curves of all
    previously solved segments."""

    def __init__(self):
        self.enabled = LIVE_PLOT_ENABLED
        self.fig = None
        self.ax = None
        self.segment_lines = {}   # seg_idx -> Line2D (updated live / frozen after done)
        self.title_artist = None

    def start(self):
        if not self.enabled:
            return
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(9, 7))

        for obstacle in OBSTACLES:
            if obstacle["type"] == "circle":
                cx, cy = obstacle["center"]
                r = obstacle["radius"]
                self.ax.add_patch(Circle((cx, cy), r, color="firebrick", alpha=0.55, zorder=2))
            else:
                poly = obstacle["shape"]
                xs, ys = poly.exterior.xy
                self.ax.add_patch(MplPolygon(list(zip(xs, ys)), closed=True, color="firebrick", alpha=0.55, zorder=2))

        for i, wp in enumerate(WAYPOINTS):
            if i == 0:
                self.ax.plot(*wp, "gs", markersize=13, zorder=4, label="Start")
            elif i == len(WAYPOINTS) - 1:
                self.ax.plot(*wp, "r*", markersize=20, zorder=4, label="Goal")
            else:
                self.ax.plot(*wp, "D", color="orange", markersize=11, zorder=4)

        self.ax.set_xlim(BOUNDS_MIN[0], BOUNDS_MAX[0])
        self.ax.set_ylim(BOUNDS_MIN[1], BOUNDS_MAX[1])
        self.ax.set_aspect("equal")
        self.ax.grid(alpha=0.3)
        self.title_artist = self.ax.set_title(f"Live optimization - starting... [mode={CONTROL_POINT_MODE}]")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(LIVE_PLOT_PAUSE)

    def update(self, seg_idx, curve, generation, cost, popsize):
        if not self.enabled:
            return
        if seg_idx not in self.segment_lines:
            (line,) = self.ax.plot(curve[:, 0], curve[:, 1], "-", color="red", linewidth=2.5, zorder=3)
            self.segment_lines[seg_idx] = line
        else:
            self.segment_lines[seg_idx].set_data(curve[:, 0], curve[:, 1])
            self.segment_lines[seg_idx].set_color("red")

        self.title_artist.set_text(
            f"[{CONTROL_POINT_MODE}] Segment {seg_idx + 1}/{N_SEGMENTS} | gen {generation} | popsize {popsize} | cost={cost:.3f}"
        )
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(LIVE_PLOT_PAUSE)

    def freeze_segment(self, seg_idx, curve, final_color):
        if not self.enabled:
            return
        line = self.segment_lines.get(seg_idx)
        if line is None:
            (line,) = self.ax.plot(curve[:, 0], curve[:, 1], "-", linewidth=2.5, zorder=3)
            self.segment_lines[seg_idx] = line
        else:
            line.set_data(curve[:, 0], curve[:, 1])
        line.set_color(final_color)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(LIVE_PLOT_PAUSE)

    def finish(self, total_cost):
        if not self.enabled:
            return
        self.title_artist.set_text(f"Sequential DE over B-Spline Control Points [{CONTROL_POINT_MODE}] (cost={total_cost:.3f})")
        self.fig.canvas.draw_idle()
        plt.ioff()

def solve_sequential():
    all_delta_vectors = []
    prev_best_x = None
    total_cost = 0.0
    stage_schedule = build_popsize_stage_schedule()
    segment_records = []  # collected final (last-generation) control points, one entry per segment

    live_plot = LivePlotManager()
    live_plot.start()
    segment_colors = plt.cm.viridis(np.linspace(0, 0.85, max(N_SEGMENTS, 1)))

    for seg in range(N_SEGMENTS):
        seg_start = WAYPOINTS[seg]
        seg_end = WAYPOINTS[seg + 1]
        bounds = build_delta_bounds(seg_start, seg_end)

        gen_counter = {"n": 0}  # cumulative generation count across this segment's stages
        stage_best_x = prev_best_x
        stage_result = None

        for stage_idx, (n_gens, popsize) in enumerate(stage_schedule):
            init_population = build_warm_start_population(stage_best_x, popsize, N_VARS_PER_SEGMENT, seg_start, seg_end)

            def gen_callback(xk, convergence):
                gen_counter["n"] += 1
                curve = segment_curve_from_deltas(xk, seg_start, seg_end)
                cost = segment_cost_function(xk, seg_start, seg_end)
                print(f"    Gen {gen_counter['n']} (popsize={popsize}): cost={cost:.4f}")

                # Requirement: live visualization of the current best path every generation
                live_plot.update(seg, curve, gen_counter["n"], cost, popsize)

            stage_result = differential_evolution(
                segment_cost_function,
                bounds,
                args=(seg_start, seg_end),
                strategy=DE_STRATEGY,
                maxiter=n_gens,
                popsize=popsize,
                tol=DE_TOL,
                mutation=DE_MUTATION,
                recombination=DE_RECOMBINATION,
                seed=DE_SEED_BASE + seg + stage_idx,
                polish=(stage_idx == len(stage_schedule) - 1),  # only polish on the final (smallest) stage
                updating="immediate",
                workers=1,
                init=init_population,
                callback=gen_callback,
            )
            stage_best_x = stage_result.x

        result = stage_result
        print(f"  Segment {seg + 1}/{N_SEGMENTS}: cost={result.fun:.4f}, nfev={result.nfev}")

        # Freeze this segment's line to its final color and record its final control points
        final_curve = segment_curve_from_deltas(result.x, seg_start, seg_end)
        live_plot.freeze_segment(seg, final_curve, segment_colors[seg])

        free_points = vars_to_control_points(result.x, seg_start, seg_end)
        segment_records.append({
            "segment_index": seg,
            "generation": gen_counter["n"],
            "cost": float(result.fun),
            "control_point_mode": CONTROL_POINT_MODE,
            "start_point": seg_start.tolist(),
            "end_point": seg_end.tolist(),
            "control_points": free_points.tolist(),  # exactly N_CONTROL_PER_SEGMENT (6) points
        })

        all_delta_vectors.append(result.x)
        prev_best_x = result.x
        total_cost += result.fun

        # Requirement: user-configurable early termination after first segment
        if seg == 0 and EARLY_TERMINATE_AFTER_FIRST_SEGMENT:
            print("  EARLY_TERMINATE_AFTER_FIRST_SEGMENT is set - stopping after segment 1.")
            break

    # Requirement: save control points for ALL segments' final generation into one JSON file
    export_all_segments_control_points(segment_records)

    live_plot.finish(total_cost)

    return all_delta_vectors, total_cost, live_plot

# ----------------------------------------------------------------------
# 7. Export and Visualization
# ----------------------------------------------------------------------

def build_full_path(delta_vectors):
    curves = []
    for seg, deltas in enumerate(delta_vectors):
        seg_start = WAYPOINTS[seg]
        seg_end = WAYPOINTS[seg + 1]
        curve = segment_curve_from_deltas(deltas, seg_start, seg_end)
        curves.append(curve)
    return curves

def export_path(curves, filename="planned_path.json"):
    path_data = {
        "metadata": {
            "map_name": CURRENT_MAP_NAME,
            "control_point_mode": CONTROL_POINT_MODE,
            "num_segments": len(curves),
            "samples_per_segment": N_SAMPLES_PER_SEGMENT
        },
        "path_segments": [curve.tolist() for curve in curves]
    }
    with open(filename, 'w') as f:
        json.dump(path_data, f, indent=4)
    print(f"Saved optimized path to {filename}")

def plot_result(delta_vectors, total_cost):
    curves = build_full_path(delta_vectors)
    fig, ax = plt.subplots(figsize=(9, 7))

    for obstacle in OBSTACLES:
        if obstacle["type"] == "circle":
            cx, cy = obstacle["center"]
            r = obstacle["radius"]
            ax.add_patch(Circle((cx, cy), r, color="firebrick", alpha=0.55, zorder=2))
        else:
            poly = obstacle["shape"]
            xs, ys = poly.exterior.xy
            ax.add_patch(MplPolygon(list(zip(xs, ys)), closed=True, color="firebrick", alpha=0.55, zorder=2))

    colors = plt.cm.viridis(np.linspace(0, 0.85, max(len(curves), 1)))
    for seg, curve in enumerate(curves):
        ax.plot(curve[:, 0], curve[:, 1], "-", color=colors[seg], linewidth=2.5, zorder=3, label=f"Segment {seg + 1}")

    for i, wp in enumerate(WAYPOINTS):
        if i == 0:
            ax.plot(*wp, "gs", markersize=13, zorder=4, label="Start")
        elif i == len(WAYPOINTS) - 1:
            ax.plot(*wp, "r*", markersize=20, zorder=4, label="Goal")
        else:
            ax.plot(*wp, "D", color="orange", markersize=11, zorder=4, label=f"Task {i}")

    ax.set_xlim(BOUNDS_MIN[0], BOUNDS_MAX[0])
    ax.set_ylim(BOUNDS_MIN[1], BOUNDS_MAX[1])
    ax.set_aspect("equal")
    ax.set_title(f"Sequential DE over B-Spline Control Points [{CONTROL_POINT_MODE}] (cost={total_cost:.3f})")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("solves", exist_ok=True)
    outpath = f"solves/de_bspline_output_{CONTROL_POINT_MODE}_{timestamp}.png"
    plt.savefig(outpath, dpi=150)
    print(f"Saved plot to {outpath}")

# ----------------------------------------------------------------------
# 8. Main Execution
# ----------------------------------------------------------------------

def run_planner(input_map_json, output_path_json=None, control_point_mode=None):
    """control_point_mode: optionally override CONTROL_POINT_MODE ("delta"
    or "direct") for this run without editing the global constant."""
    global CONTROL_POINT_MODE
    if control_point_mode is not None:
        CONTROL_POINT_MODE = control_point_mode

    print(f"Loading map configuration from: {input_map_json}")
    load_map_config(input_map_json)

    if output_path_json is None:
        os.makedirs("solves", exist_ok=True)
        output_path_json = f"solves/planned_path_{CURRENT_MAP_NAME}_{CONTROL_POINT_MODE}.json"

    print(f"Running sequential per-segment DE over B-spline control points [mode={CONTROL_POINT_MODE}]...")
    start_time = time.perf_counter()
    delta_vectors, total_cost, live_plot = solve_sequential()
    end_time = time.perf_counter()

    print(f"Total cost: {total_cost:.4f}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    curves = build_full_path(delta_vectors)
    export_path(curves, output_path_json)
    plot_result(delta_vectors, total_cost)

    if LIVE_PLOT_ENABLED:
        plt.show()  # keep the live window open after the solve finishes

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Switch mode here, or pass control_point_mode="direct" / "delta" to run_planner
    run_planner("maps/map_001_robot_1.json",
                f"solves/multi/planned_path_{timestamp}.json",
                control_point_mode=CONTROL_POINT_MODE)