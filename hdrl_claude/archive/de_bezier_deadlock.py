"""
Global Path Planning via Differential Evolution over Bezier Control Points
============================================================================

Idea:
    - Represent the global path as a Bezier curve defined by:
        [start] -> [N free control points, optimized] -> [goal]
    - Use Differential Evolution (scipy.optimize.differential_evolution) to
      search for the control-point positions that minimize a cost function:
        cost = path_length + collision_penalty + curvature_penalty
    - Static circular obstacles are avoided via a penalty term that fires
      whenever the sampled curve enters an obstacle's (radius + safety margin).

This is a "taste" / prototype script — the core loop (curve param -> cost)
is exactly the piece you'd swap for a cubic-spline representation later.
"""

import time
import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import comb
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Problem setup: start, goal, obstacles
# ----------------------------------------------------------------------

START = np.array([0.0, 5.0])
GOAL = np.array([10.0, 5.0])

# Deadlock / trap layout: a U-shaped wall of obstacles opening AWAY from the
# goal, so the straight-line and short-horizon paths get "trapped" and the
# optimizer must find the long way around through the narrow gap.
OBSTACLES = [
    (4.0, 4.0, 1.0),   # blocks the direct line, forms the base of the U
    (4.0, 5.5, 1.0),   # top arm of the U
    (4.0, 2.5, 1.0),   # bottom arm of the U
    (2.0, 6.5, 1.0),   # extends the top arm further, tightening the trap
    (2.0, 1.5, 1.0),   # extends the bottom arm further, tightening the trap
]

ROBOT_RADIUS = 0.25      # safety margin added to every obstacle radius
N_CONTROL_POINTS = 6     # more free points needed to escape the deadlock
N_SAMPLES = 60           # points sampled along the curve for cost evaluation

# Search bounds for each free control point (keep the search space sane,
# a bit larger than the start/goal bounding box)
BOUNDS_MIN = np.array([-1.0, -1.0])
BOUNDS_MAX = np.array([11.0, 11.0])


# ----------------------------------------------------------------------
# 2. Bezier curve construction
# ----------------------------------------------------------------------

def bezier_curve(control_points, n_samples=N_SAMPLES):
    """
    control_points: (M, 2) array including start and goal as the first
                     and last rows.
    Returns: (n_samples, 2) array of points sampled along the Bezier curve.
    """
    n = len(control_points) - 1
    t = np.linspace(0.0, 1.0, n_samples)
    curve = np.zeros((n_samples, 2))
    for i, p in enumerate(control_points):
        bernstein = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        curve += np.outer(bernstein, p)
    return curve


def build_control_points(flat_free_points):
    """Assemble [start, free points..., goal] into one (M,2) array."""
    free_points = flat_free_points.reshape(N_CONTROL_POINTS, 2)
    return np.vstack([START, free_points, GOAL])


# ----------------------------------------------------------------------
# 3. Cost function (this is what Differential Evolution minimizes)
# ----------------------------------------------------------------------

def path_length(curve):
    diffs = np.diff(curve, axis=0)
    return np.sum(np.hypot(diffs[:, 0], diffs[:, 1]))


def collision_penalty(curve):
    """Heavily penalize any sampled point that lands inside an obstacle
    (expanded by the robot's radius). Penalty scales with the depth of
    the intrusion so DE gets a useful gradient-like signal."""
    penalty = 0.0
    for (cx, cy, r) in OBSTACLES:
        safe_r = r + ROBOT_RADIUS
        d = np.hypot(curve[:, 0] - cx, curve[:, 1] - cy)
        intrusion = np.clip(safe_r - d, 0, None)   # >0 where inside obstacle
        penalty += np.sum(intrusion ** 2)
    return penalty


def curvature_penalty(curve):
    """Light penalty on sharp turns, to bias DE towards smoother paths."""
    if len(curve) < 3:
        return 0.0
    v1 = curve[1:-1] - curve[:-2]
    v2 = curve[2:] - curve[1:-1]
    n1 = np.linalg.norm(v1, axis=1) + 1e-9
    n2 = np.linalg.norm(v2, axis=1) + 1e-9
    cos_angle = np.clip(np.sum(v1 * v2, axis=1) / (n1 * n2), -1.0, 1.0)
    turn_angle = np.arccos(cos_angle)
    return np.sum(turn_angle ** 2)


# Weights: collision >> length > curvature
W_LENGTH = 1.0
W_COLLISION = 500.0
W_CURVATURE = 0.5


def cost_function(flat_free_points):
    control_points = build_control_points(flat_free_points)
    curve = bezier_curve(control_points)
    L = path_length(curve)
    C = collision_penalty(curve)
    K = curvature_penalty(curve)
    return W_LENGTH * L + W_COLLISION * C + W_CURVATURE * K


# ----------------------------------------------------------------------
# 4. Run Differential Evolution
# ----------------------------------------------------------------------

def solve():
    bounds = []
    for _ in range(N_CONTROL_POINTS):
        bounds.append((BOUNDS_MIN[0], BOUNDS_MAX[0]))  # x
        bounds.append((BOUNDS_MIN[1], BOUNDS_MAX[1]))  # y

    result = differential_evolution(
        cost_function,
        bounds,
        strategy="best1bin",
        maxiter=100,
        popsize=20,
        tol=1e-7,
        mutation=(0.4, 1.2),
        recombination=0.7,
        seed=42,
        polish=True,
        updating="immediate",  # faster for serial execution
        workers=1,             # serial: avoids multiprocessing startup overhead
    )
    return result


# ----------------------------------------------------------------------
# 5. Visualization
# ----------------------------------------------------------------------

def plot_result(result):
    control_points = build_control_points(result.x)
    curve = bezier_curve(control_points, n_samples=200)

    fig, ax = plt.subplots(figsize=(7, 7))

    # obstacles
    for (cx, cy, r) in OBSTACLES:
        circle = plt.Circle((cx, cy), r, color="firebrick", alpha=0.5, zorder=2)
        ax.add_patch(circle)
        safe_circle = plt.Circle((cx, cy), r + ROBOT_RADIUS, color="firebrick",
                                  alpha=0.15, linestyle="--", fill=True, zorder=1)
        ax.add_patch(safe_circle)

    # optimized path
    ax.plot(curve[:, 0], curve[:, 1], "-", color="royalblue", linewidth=2.5,
             label="Optimized Bezier path", zorder=3)

    # control polygon (dashed, shows what DE actually optimized)
    ax.plot(control_points[:, 0], control_points[:, 1], "o--", color="gray",
             alpha=0.6, label="Control points", zorder=3)

    ax.plot(*START, "gs", markersize=12, label="Start", zorder=4)
    ax.plot(*GOAL, "r*", markersize=18, label="Goal", zorder=4)

    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 12)
    ax.set_aspect("equal")
    ax.set_title(f"DE-Optimized Path Through Deadlock Trap (cost={result.fun:.3f})")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("de_bezier_deadlock.png", dpi=150)
    print("Saved plot to de_bezier_deadlock.png")


if __name__ == "__main__":
    print("Running Differential Evolution over Bezier control points...")
    start_time = time.perf_counter()
    result = solve()
    end_time = time.perf_counter()

    print(f"Best cost: {result.fun:.4f}")
    print(f"Optimized free control points:\n{result.x.reshape(N_CONTROL_POINTS, 2)}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")
    plot_result(result)
