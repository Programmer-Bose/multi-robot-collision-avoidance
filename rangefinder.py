"""
Virtual rangefinder sensor: N rays cast from the robot, evenly spaced over
360 degrees (relative to the robot's heading), each returning distance to
the nearest obstacle (static or dynamic) or world boundary along that ray,
capped at max_range.

This is the LOCAL sensing model the DRL policy will actually use at
deployment -- unlike DE's fitness function, which sees full obstacle state.
"""

import numpy as np


def _ray_circle_intersection(ox, oy, dx, dy, cx, cy, r):
    """
    Distance from ray origin (ox,oy) along direction (dx,dy) [unit vector]
    to nearest intersection with circle centered (cx,cy) radius r.
    Returns None if no intersection ahead of the ray origin.
    """
    fx, fy = ox - cx, oy - cy
    a = dx * dx + dy * dy  # == 1.0 for unit direction, kept general
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    candidates = [t for t in (t1, t2) if t > 1e-6]
    if not candidates:
        return None
    return min(candidates)


def _ray_bounds_distance(ox, oy, dx, dy, world_bounds):
    """Distance from ray origin to the world boundary along the ray direction."""
    xmin, xmax, ymin, ymax = world_bounds
    ts = []
    if dx > 1e-9:
        ts.append((xmax - ox) / dx)
    elif dx < -1e-9:
        ts.append((xmin - ox) / dx)
    if dy > 1e-9:
        ts.append((ymax - oy) / dy)
    elif dy < -1e-9:
        ts.append((ymin - oy) / dy)
    ts = [t for t in ts if t > 1e-6]
    return min(ts) if ts else np.inf


def simulate_rangefinder(robot_state, static_obstacles, dynamic_obstacles,
                          world_bounds, n_rays=15, max_range=5.0):
    """
    robot_state: (x, y, theta)
    static_obstacles: list of (x, y, radius)
    dynamic_obstacles: list of (x, y, radius, vx, vy)  -- vx,vy ignored here,
                        rangefinder only measures instantaneous geometry
    world_bounds: (xmin, xmax, ymin, ymax)
    Returns: np.array of shape (n_rays,), each entry in [0, max_range]
    """
    x, y, theta = robot_state
    angles = theta + np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    readings = np.full(n_rays, max_range, dtype=float)

    all_circles = [(ox, oy, r) for (ox, oy, r) in static_obstacles]
    all_circles += [(ox, oy, r) for (ox, oy, r, vx, vy) in dynamic_obstacles]

    for i, ang in enumerate(angles):
        dx, dy = np.cos(ang), np.sin(ang)
        best = _ray_bounds_distance(x, y, dx, dy, world_bounds)
        best = min(best, max_range)
        for (ox, oy, r) in all_circles:
            d = _ray_circle_intersection(x, y, dx, dy, ox, oy, r)
            if d is not None and d < best:
                best = d
        readings[i] = min(best, max_range)

    return readings


if __name__ == "__main__":
    # quick standalone test
    robot_state = (2.0, 2.0, 0.0)
    static_obs = [(3.0, 2.0, 0.4)]
    dyn_obs = [(2.0, 4.0, 0.25, 0.1, -0.1)]
    bounds = (0, 10, 0, 10)

    readings = simulate_rangefinder(robot_state, static_obs, dyn_obs, bounds,
                                     n_rays=15, max_range=5.0)
    for i, d in enumerate(readings):
        print(f"ray {i:2d} ({i*24:3d} deg from heading): {d:.2f}")