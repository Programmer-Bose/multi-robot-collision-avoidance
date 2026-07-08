"""
Global Path Planning via Differential Evolution over Bezier control-point deltas
=============================================================================
SEQUENTIAL SEGMENT-WISE VERSION WITH DELTA OPTIMIZATION
------------------------------------------------------
Instead of evolving absolute control-point coordinates directly, this version
optimizes delta-x and delta-y offsets for each free control point.

For each segment, we start from a straight-line baseline between the segment
start and end waypoints, and then let DE discover how much each control point
should deviate from that baseline. This makes the search more structured and
helps the optimizer focus on local shape changes rather than raw coordinates.
"""

import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import comb
import shapely
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle
import time

# ----------------------------------------------------------------------
# 1. Problem setup: waypoint sequence + mixed obstacles
# ----------------------------------------------------------------------

WAYPOINTS = [
    np.array([0.0, 0.0]),    # start
    np.array([4.0, 6.5]),    # task point 1
    np.array([9.5, 2.0]),    # task point 2
    np.array([10.5, 5.5]), 
    np.array([8.1, 5.0]),
    np.array([6.0, -1.0]),
    np.array([0.0, 0.0]),    # start
]

ROBOT_RADIUS = 0.3

OBSTACLES = [
    {"type": "circle", "center": (2.2, 4.7), "radius": 0.5},
    {"type": "circle", "center": (6.5, 1.0), "radius": 1.5},
    {"type": "circle", "center": (2.0, 3.0), "radius": 1.0},
    {"type": "polygon", "shape": Polygon([(5.5, 4.0), (6.5, 4.0), (6.5, 6.0), (5.5, 6.0)])},
    # {"type": "polygon", "shape": Polygon([
    #     (9.0, 3.0), (11.5, 3.0), (11.5, 7.0), (10.5, 7.0),
    #     (10.5, 4.0), (10.0, 4.0), (10.0, 7.0), (9.0, 7.0),
    # ])},
    {"type": "polygon", "shape": Polygon([
        (9.0, 3.0), (13.0, 3.0), (13.0, 7.0), (12.0, 7.0),
        (12.0, 4.0), (10.0, 4.0), (10.0, 7.0), (9.0, 7.0),
    ])},
    {"type": "circle", "center": (4.5, 2.0), "radius": 0.8},
]

N_SEGMENTS = len(WAYPOINTS) - 1
N_CONTROL_PER_SEGMENT = 4
N_SAMPLES_PER_SEGMENT = 40
N_VARS_PER_SEGMENT = N_CONTROL_PER_SEGMENT * 2

BOUNDS_MIN = np.array([-1.0, -1.0])
BOUNDS_MAX = np.array([13.0, 9.0])

WARM_START_FRACTION = 0.45
WARM_START_NOISE_STD = 0.75

# ----------------------------------------------------------------------
# 2. Bezier curve construction
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
    """Create a straight-line baseline for the segment's free control points."""
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
    """Convert delta variables to actual control points."""
    deltas = flat_deltas.reshape(N_CONTROL_PER_SEGMENT, 2)
    return baseline_control_points(seg_start, seg_end) + deltas


def segment_curve_from_deltas(flat_deltas, seg_start, seg_end):
    free_points = delta_to_control_points(flat_deltas, seg_start, seg_end)
    control_points = np.vstack([seg_start, free_points, seg_end])
    return bezier_curve(control_points, N_SAMPLES_PER_SEGMENT)


# ----------------------------------------------------------------------
# 3. Collision / distance checks
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


W_LENGTH = 1.0
W_COLLISION = 800.0
W_CURVATURE = 0.5


def segment_cost_function(flat_deltas, seg_start, seg_end):
    curve = segment_curve_from_deltas(flat_deltas, seg_start, seg_end)
    cost = 0.0
    cost += W_LENGTH * path_length(curve)
    cost += W_COLLISION * collision_penalty(curve)
    cost += W_CURVATURE * curvature_penalty(curve)
    return cost


# ----------------------------------------------------------------------
# 4. Sequential DE over delta variables
# ----------------------------------------------------------------------


def build_delta_bounds(seg_start, seg_end):
    """Bounds for each delta component, proportional to segment length."""
    segment_length = np.linalg.norm(seg_end - seg_start)
    delta_bound = max(3.5, 0.5 * segment_length)
    return [(-delta_bound, delta_bound) for _ in range(N_VARS_PER_SEGMENT)]


def build_warm_start_population(prev_best_x, popsize, n_vars):
    """Seed part of the population from the previous segment's delta vector."""
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
    popsize = 20

    for seg in range(N_SEGMENTS):
        seg_start = WAYPOINTS[seg]
        seg_end = WAYPOINTS[seg + 1]
        bounds = build_delta_bounds(seg_start, seg_end)

        init_population = build_warm_start_population(prev_best_x, popsize, N_VARS_PER_SEGMENT)

        result = differential_evolution(
            segment_cost_function,
            bounds,
            args=(seg_start, seg_end),
            strategy="best1bin",
            maxiter=150,
            popsize=popsize,
            tol=1e-8,
            mutation=(0.4, 1.2),
            recombination=0.8,
            seed=17 + seg,
            polish=True,
            updating="immediate",
            workers=1,
            init=init_population,
        )

        print(
            f"  Segment {seg + 1}/{N_SEGMENTS}: cost={result.fun:.4f}, "
            f"nfev={result.nfev}, warm_start={'yes' if prev_best_x is not None else 'no (first segment)'}"
        )

        all_delta_vectors.append(result.x)
        prev_best_x = result.x
        total_cost += result.fun

    return all_delta_vectors, total_cost


# ----------------------------------------------------------------------
# 5. Reassemble full path for plotting
# ----------------------------------------------------------------------


def build_full_path(delta_vectors):
    curves = []
    for seg in range(N_SEGMENTS):
        seg_start = WAYPOINTS[seg]
        seg_end = WAYPOINTS[seg + 1]
        curve = segment_curve_from_deltas(delta_vectors[seg], seg_start, seg_end)
        curves.append(curve)
    return curves


# ----------------------------------------------------------------------
# 6. Visualization
# ----------------------------------------------------------------------


def plot_result(delta_vectors, total_cost):
    curves = build_full_path(delta_vectors)
    fig, ax = plt.subplots(figsize=(9, 7))

    for obstacle in OBSTACLES:
        if obstacle["type"] == "circle":
            cx, cy = obstacle["center"]
            r = obstacle["radius"]
            ax.add_patch(Circle((cx, cy), r, color="firebrick", alpha=0.55, zorder=2))
            ax.add_patch(Circle((cx, cy), r + ROBOT_RADIUS, color="firebrick",
                                 alpha=0.15, zorder=1))
        else:
            poly = obstacle["shape"]
            xs, ys = poly.exterior.xy
            ax.add_patch(MplPolygon(list(zip(xs, ys)), closed=True,
                                     color="firebrick", alpha=0.55, zorder=2))
            inflated = poly.buffer(ROBOT_RADIUS)
            ixs, iys = inflated.exterior.xy
            ax.add_patch(MplPolygon(list(zip(ixs, iys)), closed=True,
                                     color="firebrick", alpha=0.15, zorder=1))

    colors = plt.cm.viridis(np.linspace(0, 0.85, N_SEGMENTS))
    for seg, curve in enumerate(curves):
        ax.plot(curve[:, 0], curve[:, 1], "-", color=colors[seg], linewidth=2.5,
                zorder=3, label=f"Segment {seg + 1}")

    for i, wp in enumerate(WAYPOINTS):
        if i == 0:
            ax.plot(*wp, "gs", markersize=13, zorder=4, label="Start")
        elif i == len(WAYPOINTS) - 1:
            ax.plot(*wp, "r*", markersize=20, zorder=4, label="Goal")
        else:
            ax.plot(*wp, "D", color="orange", markersize=11, zorder=4,
                    label=f"Task {i}")

    ax.set_xlim(BOUNDS_MIN[0] - 1, BOUNDS_MAX[0] + 1)
    ax.set_ylim(BOUNDS_MIN[1] - 1, BOUNDS_MAX[1] + 1)
    ax.set_aspect("equal")
    ax.set_title(f"Sequential Segment-Wise DE over Control-Point Deltas "
                 f"(total cost={total_cost:.3f})")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("de_sequential_segment_warmstart_delta.png", dpi=150)
    print("Saved plot to de_sequential_segment_warmstart_delta.png")


if __name__ == "__main__":
    print("Running sequential per-segment DE over control-point deltas...")
    start_time = time.perf_counter()
    delta_vectors, total_cost = solve_sequential()
    end_time = time.perf_counter()
    print(f"Total cost (sum over segments): {total_cost:.4f}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")
    plot_result(delta_vectors, total_cost)
