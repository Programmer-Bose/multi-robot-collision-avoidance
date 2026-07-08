"""
Global Path Planning via Differential Evolution over Waypoint Positions
========================================================================
WAYPOINT-BASED VERSION (Option 1)
----------------------------------
Instead of Bezier control-point deltas, each segment between two task
waypoints is subdivided into 6 free intermediate points (12 DE variables:
x,y for each point). The path within a segment is simply the straight-line
polyline through: seg_start -> p1 -> p2 -> ... -> p6 -> seg_end.

DE evolves the (x,y) offsets of these 6 points from a straight-line
baseline, exactly like the delta-based Bezier version, so the search
stays local/structured rather than searching raw absolute coordinates.

Sequential per-segment solve with warm-starting is preserved.
"""

import numpy as np
from scipy.optimize import differential_evolution
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle
import time
import json
import os
import datetime

# ----------------------------------------------------------------------
# 1. Global Problem Setup (Defaults)
# ----------------------------------------------------------------------

WAYPOINTS = []
OBSTACLES = []
N_SEGMENTS = 0

N_FREE_POINTS = 6                      # free points per segment
N_VARS_PER_SEGMENT = N_FREE_POINTS * 2  # 12 DE variables per segment
N_SAMPLES_PER_SUBSEGMENT = 8           # samples along each straight sub-edge for collision checks

BOUNDS_MIN = np.array([0.0, 0.0])
BOUNDS_MAX = np.array([12.0, 12.0])

ROBOT_RADIUS = 0.3
WARM_START_FRACTION = 0.5
WARM_START_NOISE_STD = 0.75

W_LENGTH = 1.0
W_COLLISION = 800.0
W_CURVATURE = 0.5

# ----------------------------------------------------------------------
# 2. Map Loading & Coordinate Transformation
# ----------------------------------------------------------------------

def load_map_config(json_path):
    global WAYPOINTS, OBSTACLES, N_SEGMENTS, BOUNDS_MIN, BOUNDS_MAX

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Configuration file {json_path} not found.")

    with open(json_path, 'r') as f:
        data = json.load(f)

    orig_w, orig_h = data["map_metadata"]["size"]
    target_scale = 12.0
    scale_factor = target_scale / orig_w

    def scale_pt(pt):
        return np.array([pt[0] * scale_factor, (orig_h - pt[1]) * scale_factor])

    WAYPOINTS.clear()
    WAYPOINTS.append(scale_pt(data["start_position"]))

    for task_id in data["task_sequence"]:
        WAYPOINTS.append(scale_pt(data["task_points"][str(task_id)]))

    if data.get("goal_position"):
        WAYPOINTS.append(scale_pt(data["goal_position"]))

    N_SEGMENTS = len(WAYPOINTS) - 1
    BOUNDS_MAX = np.array([target_scale, target_scale])

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
# 3. Waypoint-based Path Construction
# ----------------------------------------------------------------------

def baseline_free_points(seg_start, seg_end):
    """Straight-line baseline positions for the 6 free intermediate points."""
    segment_vec = seg_end - seg_start
    if np.allclose(segment_vec, 0):
        return np.repeat(seg_start[None, :], N_FREE_POINTS, axis=0)

    t_values = np.linspace(
        1.0 / (N_FREE_POINTS + 1),
        N_FREE_POINTS / (N_FREE_POINTS + 1),
        N_FREE_POINTS,
    )
    return np.array([seg_start + t * segment_vec for t in t_values])

def delta_to_points(flat_deltas, seg_start, seg_end):
    """Convert 12 DE variables (deltas) into 6 actual (x,y) free points."""
    deltas = flat_deltas.reshape(N_FREE_POINTS, 2)
    return baseline_free_points(seg_start, seg_end) + deltas

def densify_polyline(vertices, n_samples_per_edge):
    """Sample points densely along a polyline defined by `vertices`."""
    pts = []
    for i in range(len(vertices) - 1):
        p0, p1 = vertices[i], vertices[i + 1]
        t = np.linspace(0.0, 1.0, n_samples_per_edge, endpoint=(i == len(vertices) - 2))
        seg_pts = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
        pts.append(seg_pts)
    return np.vstack(pts)

def segment_curve_from_deltas(flat_deltas, seg_start, seg_end):
    free_points = delta_to_points(flat_deltas, seg_start, seg_end)
    vertices = np.vstack([seg_start, free_points, seg_end])
    return densify_polyline(vertices, N_SAMPLES_PER_SUBSEGMENT)

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
# 5. Differential Evolution
# ----------------------------------------------------------------------

def build_delta_bounds(seg_start, seg_end):
    segment_length = np.linalg.norm(seg_end - seg_start)
    delta_bound = max(5, 0.8 * segment_length)
    return [(-delta_bound, delta_bound) for _ in range(N_VARS_PER_SEGMENT)]

def build_warm_start_population(prev_best_x, popsize, n_vars):
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


def solve_sequential():
    all_delta_vectors = []
    prev_best_x = None
    total_cost = 0.0
    popsize = 50

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

        result = differential_evolution(
            segment_cost_function,
            bounds,
            args=(seg_start, seg_end),
            strategy="best1bin",
            maxiter=50,
            popsize=popsize,
            tol=1e-8,
            mutation=(0.4, 1.2),
            recombination=0.9,
            seed=17+seg,
            polish=True,
            updating="immediate",
            workers=1,
            init=init_population,
            callback=gen_callback,
        )

        print(f"  Segment {seg + 1}/{N_SEGMENTS}: cost={result.fun:.4f}, nfev={result.nfev}")
        all_delta_vectors.append(result.x)
        prev_best_x = result.x
        total_cost += result.fun

    return all_delta_vectors, total_cost

# ----------------------------------------------------------------------
# 6. Export and Visualization
# ----------------------------------------------------------------------

def build_full_path(delta_vectors):
    curves = []
    for seg in range(N_SEGMENTS):
        seg_start = WAYPOINTS[seg]
        seg_end = WAYPOINTS[seg + 1]
        curve = segment_curve_from_deltas(delta_vectors[seg], seg_start, seg_end)
        curves.append(curve)
    return curves

def export_path(curves, filename="planned_path.json"):
    path_data = {
        "metadata": {
            "num_segments": N_SEGMENTS,
            "free_points_per_segment": N_FREE_POINTS,
            "samples_per_subsegment": N_SAMPLES_PER_SUBSEGMENT,
            "representation": "waypoint_polyline"
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

    colors = plt.cm.viridis(np.linspace(0, 0.85, N_SEGMENTS))
    for seg, curve in enumerate(curves):
        ax.plot(curve[:, 0], curve[:, 1], "-", color=colors[seg], linewidth=2.5, zorder=3, label=f"Segment {seg + 1}")
        # mark the 6 free waypoints for this segment
        free_pts = delta_to_points(delta_vectors[seg], WAYPOINTS[seg], WAYPOINTS[seg + 1])
        ax.plot(free_pts[:, 0], free_pts[:, 1], "o", color=colors[seg], markersize=5, zorder=4, alpha=0.8)

    for i, wp in enumerate(WAYPOINTS):
        if i == 0:
            ax.plot(*wp, "gs", markersize=13, zorder=5, label="Start")
        elif i == len(WAYPOINTS) - 1:
            ax.plot(*wp, "r*", markersize=20, zorder=5, label="Goal")
        else:
            ax.plot(*wp, "D", color="orange", markersize=11, zorder=5, label=f"Task {i}")

    ax.set_xlim(BOUNDS_MIN[0], BOUNDS_MAX[0])
    ax.set_ylim(BOUNDS_MIN[1], BOUNDS_MAX[1])
    ax.set_aspect("equal")
    ax.set_title(f"Sequential DE over Waypoint Deltas (6 pts/seg, cost={total_cost:.3f})")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs("solves", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"solves/de_waypoint_output_{timestamp}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

# ----------------------------------------------------------------------
# 7. Main Execution
# ----------------------------------------------------------------------

def run_planner(input_map_json, output_path_json="planned_path.json"):
    print(f"Loading map configuration from: {input_map_json}")
    load_map_config(input_map_json)

    print(f"Running sequential per-segment DE over {N_FREE_POINTS}-point waypoint deltas "
          f"({N_VARS_PER_SEGMENT} DE variables/segment)...")
    start_time = time.perf_counter()
    delta_vectors, total_cost = solve_sequential()
    end_time = time.perf_counter()

    print(f"Total cost: {total_cost:.4f}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    curves = build_full_path(delta_vectors)
    os.makedirs(os.path.dirname(output_path_json) or ".", exist_ok=True)
    export_path(curves, output_path_json)
    plot_result(delta_vectors, total_cost)

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_planner("maps/env_map_config_006.json", f"solves/planned_path_{timestamp}.json")
