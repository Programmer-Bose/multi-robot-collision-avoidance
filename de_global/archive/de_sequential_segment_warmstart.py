"""
Global Path Planning via Differential Evolution over Bezier Control Points
============================================================================
SEQUENTIAL SEGMENT-WISE VERSION
--------------------------------
Instead of one joint DE run over the concatenated control points of ALL
segments (dimensionality = N_SEGMENTS * N_CONTROL_PER_SEGMENT * 2, which
explodes as tasks are added), each segment is optimized by its OWN DE run,
one after another:

    segment 1 (start -> task_1)      DE run #1  (6-D search space)
    segment 2 (task_1 -> task_2)     DE run #2, seeded from run #1's result
    segment 3 (task_2 -> task_3)     DE run #3, seeded from run #2's result
    ...

Each individual DE run stays LOW-DIMENSIONAL and FIXED-SIZE regardless of
how many task points the mission has, so adding more tasks no longer blows
up the search space of any single optimization.

"Warm start from the previous segment's best solution" is implemented by
seeding a fraction of each new DE run's initial population with the
previous segment's converged control-point vector (translated to the new
segment's start waypoint), instead of drawing the whole population
uniformly at random. This gives DE a head start in regions that are known
to already look like "a good smooth curve" rather than starting from
scratch every time, which speeds up convergence.

Everything else (mixed circle / convex / non-convex polygon obstacles,
Shapely-vectorized collision checking, robot-radius inflation) is unchanged
from the joint version.
"""

import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import comb
import shapely
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle
import time

# ----------------------------------------------------------------------
# 1. Problem setup: waypoint sequence + mixed obstacles  (unchanged)
# ----------------------------------------------------------------------

WAYPOINTS = [
    np.array([0.0, 0.0]),    # start
    np.array([4.0, 6.5]),    # task point 1
    np.array([9.5, 2.0]),    # task point 2
    np.array([11, 6.8]), 
    np.array([8.1, 5.0]),
    np.array([6.0, -1.0]),
    np.array([0.0, 0.0]),    # start
]

ROBOT_RADIUS = 0.3

OBSTACLES = [
    {"type": "circle", "center": (2.2, 4.7), "radius": 0.5},
    {"type": "circle", "center": (6.5, 1.0), "radius": 1.5},
    {"type": "circle", "center": (2.0, 3.0), "radius": 1.0},
    {"type": "polygon", "shape": Polygon([(5.5, 4.0), (7.5, 4.0), (7.5, 6.0), (5.5, 6.0)])},
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
N_CONTROL_PER_SEGMENT = 6
N_SAMPLES_PER_SEGMENT = 40
N_VARS_PER_SEGMENT = N_CONTROL_PER_SEGMENT * 2   # fixed, does NOT grow with N_SEGMENTS

BOUNDS_MIN = np.array([-1.0, -1.0])
BOUNDS_MAX = np.array([13.0, 9.0])

# Warm-start controls
WARM_START_FRACTION = 0.45   # fraction of each new segment's initial population
                             # seeded from the previous segment's solution
WARM_START_NOISE_STD = 0.75  # Gaussian jitter (in map units) applied to the
                             # seeded individuals so they aren't all identical


# ----------------------------------------------------------------------
# 2. Bezier curve construction (per segment)  (unchanged)
# ----------------------------------------------------------------------

def bezier_curve(control_points, n_samples):
    n = len(control_points) - 1
    t = np.linspace(0.0, 1.0, n_samples)
    curve = np.zeros((n_samples, 2))
    for i, p in enumerate(control_points):
        bernstein = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        curve += np.outer(bernstein, p)
    return curve


def segment_curve_from_free_points(flat_free_points, seg_start, seg_end):
    free_points = flat_free_points.reshape(N_CONTROL_PER_SEGMENT, 2)
    control_points = np.vstack([seg_start, free_points, seg_end])
    return bezier_curve(control_points, N_SAMPLES_PER_SEGMENT)


# ----------------------------------------------------------------------
# 3. Collision / distance checks  (unchanged, still vectorized)
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


def segment_cost_function(flat_free_points, seg_start, seg_end):
    curve = segment_curve_from_free_points(flat_free_points, seg_start, seg_end)
    cost = 0.0
    cost += W_LENGTH * path_length(curve)
    cost += W_COLLISION * collision_penalty(curve)
    cost += W_CURVATURE * curvature_penalty(curve)
    return cost


# ----------------------------------------------------------------------
# 4. Sequential DE: one segment at a time, warm-started from the previous
#    segment's converged solution
# ----------------------------------------------------------------------

def build_warm_start_population(prev_best_x, seg_start, popsize, n_vars, bounds):
    """
    Build an initial population for the NEW segment's DE run.

    - WARM_START_FRACTION of the population is seeded from the previous
      segment's best free-control-point offsets (re-anchored onto the new
      segment's start waypoint) plus small Gaussian jitter, so DE begins
      near an already-reasonable curve shape instead of from scratch.
    - The remainder is drawn uniformly at random over the bounds for
      diversity/exploration, exactly like a fresh DE run would.
    """
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    n_individuals = popsize * n_vars   # scipy's convention: popsize * n_vars total

    n_seeded = int(n_individuals * WARM_START_FRACTION)
    n_random = n_individuals - n_seeded

    population = []

    if prev_best_x is not None:
        # Previous segment's free control points are absolute (x, y)
        # coordinates. We reuse them directly as a candidate for the new
        # segment (clipped into bounds): consecutive segments are usually
        # spatially close, so "the same rough curve shape that worked last
        # time" is a reasonable, cheap prior for where the new one should
        # bend too -- much better than starting purely from random.
        seed_base = np.clip(prev_best_x.flatten(), lower, upper)

        for _ in range(n_seeded):
            jittered = seed_base + np.random.normal(0.0, WARM_START_NOISE_STD, size=n_vars)
            jittered = np.clip(jittered, lower, upper)
            population.append(jittered)

    while len(population) < n_seeded + n_random:
        rand_ind = lower + np.random.rand(n_vars) * (upper - lower)
        population.append(rand_ind)

    population = np.array(population)
    # scipy expects population normalized to [0, 1] when passed via `init`
    # as an array of shape (n_individuals, n_vars) in PARAMETER space is
    # actually accepted directly (not normalized) -- see `init` docs: an
    # array specifies the initial population in parameter space.
    return population


def solve_sequential():
    n_vars = N_VARS_PER_SEGMENT
    bounds = [(BOUNDS_MIN[0], BOUNDS_MAX[0]) if i % 2 == 0 else (BOUNDS_MIN[1], BOUNDS_MAX[1])
              for i in range(n_vars)]

    all_free_points = []   # best flat_free_points per segment, in order
    prev_best_x = None
    total_cost = 0.0
    popsize = 20

    for seg in range(N_SEGMENTS):
        seg_start = WAYPOINTS[seg]
        seg_end = WAYPOINTS[seg + 1]

        init_population = build_warm_start_population(
            prev_best_x, seg_start, popsize, n_vars, bounds
        )

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
            seed=7 + seg,
            polish=True,
            updating="immediate",
            workers=1,
            init=init_population,
        )

        print(f"  Segment {seg + 1}/{N_SEGMENTS}: cost={result.fun:.4f}, "
              f"nfev={result.nfev}, warm_start={'yes' if prev_best_x is not None else 'no (first segment)'}")

        all_free_points.append(result.x)
        prev_best_x = result.x
        total_cost += result.fun

    flat_full = np.concatenate(all_free_points)
    return flat_full, total_cost, all_free_points


# ----------------------------------------------------------------------
# 5. Reassemble full path for plotting (same interface as joint version)
# ----------------------------------------------------------------------

def build_full_path(flat_free_points):
    pts_per_segment = N_CONTROL_PER_SEGMENT * 2
    curves = []
    for seg in range(N_SEGMENTS):
        seg_free = flat_free_points[seg * pts_per_segment:(seg + 1) * pts_per_segment]
        curve = segment_curve_from_free_points(seg_free, WAYPOINTS[seg], WAYPOINTS[seg + 1])
        curves.append(curve)
    return curves


# ----------------------------------------------------------------------
# 6. Visualization  (unchanged, just retitled)
# ----------------------------------------------------------------------

def plot_result(flat_full, total_cost):
    curves = build_full_path(flat_full)
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
    ax.set_title(f"Sequential Segment-Wise DE with Warm-Start "
                 f"(total cost={total_cost:.3f})")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("de_sequential_segment_warmstart.png", dpi=150)
    print("Saved plot to de_sequential_segment_warmstart.png")


if __name__ == "__main__":
    print("Running sequential per-segment DE with warm-start from previous segment...")
    start_time = time.perf_counter()
    flat_full, total_cost, _ = solve_sequential()
    end_time = time.perf_counter()
    print(f"Total cost (sum over segments): {total_cost:.4f}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")
    plot_result(flat_full, total_cost)
