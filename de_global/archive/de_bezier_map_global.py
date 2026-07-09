import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import comb
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
# 1. Global Problem Setup (Defaults)
# ----------------------------------------------------------------------

WAYPOINTS = []
OBSTACLES = []
N_SEGMENTS = 0
N_CONTROL_PER_SEGMENT = 6
N_SAMPLES_PER_SEGMENT = 40
N_VARS_PER_SEGMENT = N_CONTROL_PER_SEGMENT * 2

BOUNDS_MIN = np.array([0.0, 0.0])
BOUNDS_MAX = np.array([12.0, 12.0])

ROBOT_RADIUS = 0.3
WARM_START_FRACTION = 0.5
WARM_START_NOISE_STD = 0.85

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
# 3. Bezier Curve Construction
# ----------------------------------------------------------------------

def bezier_curve(control_points, n_samples):
    n = len(control_points) - 1
    t = np.linspace(0.0, 1.0, n_samples)
    curve = np.zeros((n_samples, 2))
    for i, p in enumerate(control_points):
        bernstein = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        curve += np.outer(bernstein, p)
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
    return bezier_curve(control_points, N_SAMPLES_PER_SEGMENT)

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
    popsize = 100

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
            maxiter=25,
            popsize=popsize,
            tol=1e-8,
            mutation=(0.4, 1.2),
            recombination=0.9,
            seed=17 + seg,
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

    colors = plt.cm.viridis(np.linspace(0, 0.85, N_SEGMENTS))
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
    ax.set_title(f"Sequential DE over Control-Point Deltas (cost={total_cost:.3f})")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"solves/de_sequential_output_{timestamp}.png", dpi=150)
    print("Saved plot to de_sequential_output.png")

# ----------------------------------------------------------------------
# 7. Main Execution
# ----------------------------------------------------------------------

def run_planner(input_map_json, output_path_json="planned_path.json"):
    print(f"Loading map configuration from: {input_map_json}")
    load_map_config(input_map_json)
    
    print("Running sequential per-segment DE over control-point deltas...")
    start_time = time.perf_counter()
    delta_vectors, total_cost = solve_sequential()
    end_time = time.perf_counter()
    
    print(f"Total cost: {total_cost:.4f}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")
    
    curves = build_full_path(delta_vectors)
    export_path(curves, output_path_json)
    plot_result(delta_vectors, total_cost)

if __name__ == "__main__":
    # Test execution using your provided JSON template
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_planner("maps/env_map_config_029.json", f"solves/planned_path_{timestamp}.json")