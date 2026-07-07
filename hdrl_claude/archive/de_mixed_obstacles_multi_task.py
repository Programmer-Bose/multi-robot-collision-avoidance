"""
Global Path Planning via Differential Evolution over Bezier Control Points
============================================================================
Extended version: handles

  1. Mixed obstacle geometry: circles, convex polygons (squares/rectangles),
     and NON-convex polygons (U-shapes, L-shapes, etc.) — all via Shapely,
     so "distance to obstacle" and "point inside obstacle" are exact
     regardless of convexity.
  2. Multiple task points visited in a PRE-DEFINED sequence:
        start -> task_1 -> task_2 -> ... -> task_k -> goal
     Implemented as a chain of Bezier segments, each with its own free
     control points, all optimized jointly in a single DE run (so segments
     near a tight task point can compensate for each other).
  3. Robot footprint: the robot is treated as a disc of radius ROBOT_RADIUS.
     Collision checking inflates the safe distance to every obstacle by
     ROBOT_RADIUS (a Minkowski-sum-style approximation), so the path that
     comes out already keeps the robot's body a safe distance away —
     no extra post-processing step is needed.
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
# 1. Problem setup: waypoint sequence + mixed obstacles
# ----------------------------------------------------------------------

# Predefined sequence of points the robot MUST visit, in this order:
# start -> task points -> goal
WAYPOINTS = [
    np.array([0.0, 0.0]),    # start
    np.array([4.0, 6.5]),    # task point 1
    np.array([8.5, 2.0]),    # task point 2
    np.array([12.0, 8.0]),   # goal
    np.array([0.0, 0.0]),    # start
]

ROBOT_RADIUS = 0.3   # robot modeled as a disc; safety margin baked into collision check

# Obstacles: mix of circle / convex polygon (square) / non-convex polygon (U-shape)
OBSTACLES = [
    {"type": "circle", "center": (2.0, 3.0), "radius": 1.0},
    {"type": "polygon", "shape": Polygon([(5.5, 4.0), (7.5, 4.0), (7.5, 6.0), (5.5, 6.0)])},   # square
    {"type": "polygon", "shape": Polygon([  # U-shape (non-convex), opening upward
        (9.0, 3.0), (11.5, 3.0), (11.5, 7.0), (10.5, 7.0),
        (10.5, 4.0), (10.0, 4.0), (10.0, 7.0), (9.0, 7.0),
    ])},
    {"type": "circle", "center": (4.5, 2.0), "radius": 0.8},
]

N_SEGMENTS = len(WAYPOINTS) - 1
N_CONTROL_PER_SEGMENT = 3     # free intermediate control points per Bezier segment
N_SAMPLES_PER_SEGMENT = 40    # points sampled along each segment for cost evaluation

BOUNDS_MIN = np.array([-1.0, -1.0])
BOUNDS_MAX = np.array([13.0, 9.0])


# ----------------------------------------------------------------------
# 2. Bezier curve construction (per segment)
# ----------------------------------------------------------------------

def bezier_curve(control_points, n_samples):
    n = len(control_points) - 1
    t = np.linspace(0.0, 1.0, n_samples)
    curve = np.zeros((n_samples, 2))
    for i, p in enumerate(control_points):
        bernstein = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        curve += np.outer(bernstein, p)
    return curve


def build_full_path(flat_free_points):
    """
    flat_free_points holds N_CONTROL_PER_SEGMENT free (x,y) points for EACH
    segment, concatenated. Returns a list of per-segment sampled curves,
    each starting exactly at the previous segment's waypoint (continuity
    is automatic since consecutive segments share the same fixed endpoint).
    """
    pts_per_segment = N_CONTROL_PER_SEGMENT * 2
    curves = []
    for seg in range(N_SEGMENTS):
        seg_free = flat_free_points[seg * pts_per_segment:(seg + 1) * pts_per_segment]
        free_points = seg_free.reshape(N_CONTROL_PER_SEGMENT, 2)
        control_points = np.vstack([WAYPOINTS[seg], free_points, WAYPOINTS[seg + 1]])
        curve = bezier_curve(control_points, N_SAMPLES_PER_SEGMENT)
        curves.append(curve)
    return curves


# ----------------------------------------------------------------------
# 3. Collision / distance checks for circles AND (non-)convex polygons
# ----------------------------------------------------------------------

def collision_penalty(curve):
    """Vectorized: builds a Shapely point array for the whole curve at once,
    and uses Shapely 2.x's vectorized distance/contains ufuncs per obstacle
    instead of looping point-by-point in Python (much faster for DE)."""
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
            d = shapely.distance(pts, poly)               # 0 if inside
            inside = shapely.contains(poly, pts)
            boundary_d = shapely.boundary(poly)
            d_to_boundary = shapely.distance(pts, boundary_d)
            # outside -> ROBOT_RADIUS - d (if positive); inside -> ROBOT_RADIUS + dist-to-boundary
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


def cost_function(flat_free_points):
    curves = build_full_path(flat_free_points)
    total = 0.0
    for curve in curves:
        total += W_LENGTH * path_length(curve)
        total += W_COLLISION * collision_penalty(curve)
        total += W_CURVATURE * curvature_penalty(curve)
    return total


# ----------------------------------------------------------------------
# 4. Run Differential Evolution over ALL segments jointly
# ----------------------------------------------------------------------

def solve():
    n_vars = N_SEGMENTS * N_CONTROL_PER_SEGMENT * 2
    bounds = [(BOUNDS_MIN[0], BOUNDS_MAX[0]) if i % 2 == 0 else (BOUNDS_MIN[1], BOUNDS_MAX[1])
              for i in range(n_vars)]

    result = differential_evolution(
        cost_function,
        bounds,
        strategy="best1bin",
        maxiter=100,
        popsize=15,
        tol=1e-7,
        mutation=(0.4, 1.2),
        recombination=0.8,
        seed=7,
        polish=True,
        updating="immediate",
        workers=1,   # serial: cheap cost fn, avoids multiprocessing overhead
    )
    return result


# ----------------------------------------------------------------------
# 5. Visualization
# ----------------------------------------------------------------------

def plot_result(result):
    curves = build_full_path(result.x)
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
    ax.set_title(f"DE Path Through Mixed Convex/Non-Convex Obstacles "
                 f"(cost={result.fun:.3f})")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("de_mixed_obstacles_multi_task.png", dpi=150)
    print("Saved plot to de_mixed_obstacles_multi_task.png")


if __name__ == "__main__":
    print("Running Differential Evolution over multi-segment Bezier control points...")
    start_time = time.perf_counter()
    result = solve()
    end_time = time.perf_counter()
    print(f"Best cost: {result.fun:.4f}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")
    plot_result(result)
