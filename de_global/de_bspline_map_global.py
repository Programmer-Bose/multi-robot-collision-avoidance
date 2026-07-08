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

# ----------------------------------------------------------------------
# 0. HYPERPARAMETERS (all tunable knobs live here)
# ----------------------------------------------------------------------

# --- B-spline shape ---
N_CONTROL_PER_SEGMENT = 6          # number of FREE (interior) B-spline control points per segment
BSPLINE_DEGREE = 3                 # cubic B-spline
N_SAMPLES_PER_SEGMENT = 40         # samples drawn along each segment's curve
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

# --- Warm starting (population seeding across segments) ---
WARM_START_FRACTION = 0.5           # fraction of the new population seeded from previous best solution
WARM_START_NOISE_STD = 0.85         # std-dev of Gaussian jitter applied to seeded individuals

# --- Differential Evolution settings ---
DE_POPSIZE = 50
DE_MAXITER = 50
DE_STRATEGY = "best1bin"
DE_MUTATION = (0.4, 1.2)
DE_RECOMBINATION = 0.9
DE_TOL = 1e-8
DE_SEED_BASE = 42                   # per-segment seed = DE_SEED_BASE + segment_index

# --- Checkpointing / early termination ---
CHECKPOINT_EVERY_N_GENERATIONS = DE_MAXITER-1  # save intermediate solution state every N generations
EARLY_TERMINATE_AFTER_FIRST_SEGMENT = False  # if True, stop the whole solve once segment 0 converges
CHECKPOINT_DIR = "solves/checkpoints"

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

    orig_w, orig_h = data["map_metadata"]["size"]
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

def delta_to_control_points(flat_deltas, seg_start, seg_end):
    deltas = flat_deltas.reshape(N_CONTROL_PER_SEGMENT, 2)
    return baseline_control_points(seg_start, seg_end) + deltas

def segment_curve_from_deltas(flat_deltas, seg_start, seg_end):
    free_points = delta_to_control_points(flat_deltas, seg_start, seg_end)
    control_points = np.vstack([seg_start, free_points, seg_end])
    return bspline_curve(control_points, N_SAMPLES_PER_SEGMENT)

# ----------------------------------------------------------------------
# 4. Cost Functions
# ----------------------------------------------------------------------

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

def segment_cost_function(flat_deltas, seg_start, seg_end):
    curve = segment_curve_from_deltas(flat_deltas, seg_start, seg_end)
    cost = 0.0
    cost += W_LENGTH * path_length(curve)
    cost += W_COLLISION * collision_penalty(curve)
    cost += W_CURVATURE * curvature_penalty(curve)
    return cost

# ----------------------------------------------------------------------
# 5. Checkpoint Export (start, end, 6 control points)
# ----------------------------------------------------------------------

def export_segment_checkpoint(seg_idx, generation, flat_deltas, seg_start, seg_end, cost):
    """Requirement 1/2: save start point, end point, and the six free
    B-spline control points to a JSON file whose name is derived from the
    currently loaded map."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    free_points = delta_to_control_points(flat_deltas, seg_start, seg_end)

    checkpoint_data = {
        "map_name": CURRENT_MAP_NAME,
        "segment_index": seg_idx,
        "generation": generation,
        "cost": float(cost),
        "start_point": seg_start.tolist(),
        "end_point": seg_end.tolist(),
        "control_points": free_points.tolist(),  # exactly N_CONTROL_PER_SEGMENT (6) points
    }

    filename = os.path.join(
        CHECKPOINT_DIR,
        f"{CURRENT_MAP_NAME}_seg{seg_idx}_gen{generation}.json"
    )
    with open(filename, 'w') as f:
        json.dump(checkpoint_data, f, indent=4)
    return filename

# ----------------------------------------------------------------------
# 6. Differential Evolution
# ----------------------------------------------------------------------

def build_delta_bounds(seg_start, seg_end):
    segment_length = np.linalg.norm(seg_end - seg_start)
    delta_bound = max(5, 0.8 * segment_length)
    return [(-delta_bound, delta_bound) for _ in range(N_VARS_PER_SEGMENT)]

def build_warm_start_population(prev_best_x, popsize, n_vars):
    """Warm-start methodology preserved from the original script: seed a
    fraction of the new population from the previous best solution (from the
    first segment, or from the immediately preceding segment), jittered with
    Gaussian noise; fill the remainder randomly."""
    n_individuals = popsize * n_vars
    n_seeded = int(n_individuals * WARM_START_FRACTION)
    n_random = n_individuals - n_seeded

    population = []
    if prev_best_x is not None:
        seed_base = np.asarray(prev_best_x, dtype=float).flatten()
        for _ in range(n_seeded):
            jittered = seed_base + np.random.normal(0.0, WARM_START_NOISE_STD, size=n_vars)
            population.append(jittered)

    while len(population) < n_seeded + n_random:
        population.append(np.random.uniform(-1.0, 1.0, size=n_vars))

    return np.array(population)

class EarlyTerminationSignal(Exception):
    """Raised internally to unwind out of solve_sequential once the first
    segment has converged, if EARLY_TERMINATE_AFTER_FIRST_SEGMENT is set."""
    pass

def solve_sequential():
    all_delta_vectors = []
    prev_best_x = None
    total_cost = 0.0
    popsize = DE_POPSIZE

    for seg in range(N_SEGMENTS):
        seg_start = WAYPOINTS[seg]
        seg_end = WAYPOINTS[seg + 1]
        bounds = build_delta_bounds(seg_start, seg_end)
        init_population = build_warm_start_population(prev_best_x, popsize, N_VARS_PER_SEGMENT)

        gen_counter = {"n": 0}

        def gen_callback(xk, convergence):
            gen_counter["n"] += 1
            cost = segment_cost_function(xk, seg_start, seg_end)
            print(f"    Gen {gen_counter['n']}: cost={cost:.4f}")

            # Requirement 1: periodic checkpoint export
            if gen_counter["n"] % CHECKPOINT_EVERY_N_GENERATIONS == 0:
                path = export_segment_checkpoint(seg, gen_counter["n"], xk, seg_start, seg_end, cost)
                print(f"    Checkpoint saved: {path}")

        result = differential_evolution(
            segment_cost_function,
            bounds,
            args=(seg_start, seg_end),
            strategy=DE_STRATEGY,
            maxiter=DE_MAXITER,
            popsize=popsize,
            tol=DE_TOL,
            mutation=DE_MUTATION,
            recombination=DE_RECOMBINATION,
            seed=DE_SEED_BASE + seg,
            polish=True,
            updating="immediate",
            workers=1,
            init=init_population,
            callback=gen_callback,
        )

        print(f"  Segment {seg + 1}/{N_SEGMENTS}: cost={result.fun:.4f}, nfev={result.nfev}")

        # Final checkpoint for this segment (converged solution)
        export_segment_checkpoint(seg, gen_counter["n"], result.x, seg_start, seg_end, result.fun)

        all_delta_vectors.append(result.x)
        prev_best_x = result.x
        total_cost += result.fun

        # Requirement 3: user-configurable early termination after first segment
        if seg == 0 and EARLY_TERMINATE_AFTER_FIRST_SEGMENT:
            print("  EARLY_TERMINATE_AFTER_FIRST_SEGMENT is set - stopping after segment 1.")
            break

    return all_delta_vectors, total_cost

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
    ax.set_title(f"Sequential DE over B-Spline Control-Point Deltas (cost={total_cost:.3f})")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("solves", exist_ok=True)
    outpath = f"solves/de_bspline_output_{timestamp}.png"
    plt.savefig(outpath, dpi=150)
    print(f"Saved plot to {outpath}")

# ----------------------------------------------------------------------
# 8. Main Execution
# ----------------------------------------------------------------------

def run_planner(input_map_json, output_path_json=None):
    print(f"Loading map configuration from: {input_map_json}")
    load_map_config(input_map_json)

    if output_path_json is None:
        os.makedirs("solves", exist_ok=True)
        output_path_json = f"solves/planned_path_{CURRENT_MAP_NAME}.json"

    print("Running sequential per-segment DE over B-spline control-point deltas...")
    start_time = time.perf_counter()
    delta_vectors, total_cost = solve_sequential()
    end_time = time.perf_counter()

    print(f"Total cost: {total_cost:.4f}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    curves = build_full_path(delta_vectors)
    export_path(curves, output_path_json)
    plot_result(delta_vectors, total_cost)

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_planner("maps/env_map_config_024.json", f"solves/planned_path_{timestamp}.json")
