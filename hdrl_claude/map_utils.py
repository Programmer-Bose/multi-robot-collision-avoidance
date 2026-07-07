# map_utils.py---- generates random obstacle maps (circles + convex/non-convex polygons) and rasterizes them into an occupancy grid for CNN input.
import numpy as np
from shapely.geometry import Polygon, Point
import random

def generate_random_map(seed=None, n_obstacles=4, bounds=(0, 12),
                         circle_radius_range=(0.6, 1.2),
                         square_side_range=(1.0, 2.5),
                         u_shape_size_range=(1.5, 3.5)):
    rng = random.Random(seed)
    obstacles = []
    shapes = ["circle", "square", "u_shape"]
    for _ in range(n_obstacles):
        kind = rng.choice(shapes)
        cx, cy = rng.uniform(*bounds), rng.uniform(*bounds)
        if kind == "circle":
            r = rng.uniform(*circle_radius_range)
            obstacles.append({"type": "circle", "center": (cx, cy), "radius": r})
        elif kind == "square":
            s = rng.uniform(*square_side_range)
            obstacles.append({"type": "polygon", "shape": Polygon([
                (cx, cy), (cx + s, cy), (cx + s, cy + s), (cx, cy + s)])})
        else:
            w = rng.uniform(*u_shape_size_range)
            h = w * 1.25
            obstacles.append({"type": "polygon", "shape": Polygon([
                (cx, cy), (cx + w, cy), (cx + w, cy + h), (cx + w * 0.65, cy + h),
                (cx + w * 0.65, cy + h * 0.4), (cx + w * 0.35, cy + h * 0.4),
                (cx + w * 0.35, cy + h), (cx, cy + h)])})
    return obstacles

def rasterize_map(obstacles, bounds=(0, 12), grid_size=64):
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    xs = np.linspace(bounds[0], bounds[1], grid_size)
    ys = np.linspace(bounds[0], bounds[1], grid_size)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            for obs in obstacles:
                if obs["type"] == "circle":
                    cx, cy = obs["center"]
                    if np.hypot(x - cx, y - cy) <= obs["radius"]:
                        grid[i, j] = 1.0
                else:
                    if obs["shape"].contains(Point(x, y)):
                        grid[i, j] = 1.0
    return grid


def is_point_free(point, obstacles, margin=0.3):
    """Returns True if `point` is NOT inside any obstacle (inflated by margin)."""
    px, py = point
    for obs in obstacles:
        if obs["type"] == "circle":
            cx, cy = obs["center"]
            if np.hypot(px - cx, py - cy) <= obs["radius"] + margin:
                return False
        else:
            if obs["shape"].buffer(margin).contains(Point(px, py)):
                return False
    return True


def sample_free_point(obstacles, bounds=(0, 12), margin=0.3, max_tries=200):
    """Rejection-samples a random point guaranteed to be outside all obstacles."""
    for _ in range(max_tries):
        p = np.random.uniform(bounds[0], bounds[1], size=2)
        if is_point_free(p, obstacles, margin):
            return p
    raise RuntimeError("Could not find a free point — map may be too cluttered.")


def sample_free_task_points(obstacles, n_points, bounds=(0, 12), margin=0.3):
    """Samples n_points guaranteed to be outside all obstacles (and distinct enough)."""
    points = []
    for _ in range(n_points):
        p = sample_free_point(obstacles, bounds, margin)
        points.append(p)
    return np.array(points)

if __name__ == "__main__":
    obstacles = generate_random_map(seed=1)
    grid = rasterize_map(obstacles)
    print(grid.shape)  # (64, 64)
    
    obstacles = generate_random_map(seed=1)
    task_points = sample_free_task_points(obstacles, n_points=8)
    for tp in task_points:
        print(tp, "-> free:", is_point_free(tp, obstacles))  # should all print True