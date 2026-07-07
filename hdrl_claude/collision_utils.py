# collision_utils.py
import numpy as np
import shapely

def collision_penalty(curve, obstacles, robot_radius=0.3):
    xs, ys = curve[:, 0], curve[:, 1]
    pts = shapely.points(xs, ys)
    penalty = 0.0
    for obs in obstacles:
        if obs["type"] == "circle":
            cx, cy = obs["center"]
            d = np.hypot(xs - cx, ys - cy)
            intrusion = np.clip(obs["radius"] + robot_radius - d, 0, None)
        else:
            poly = obs["shape"]
            d = shapely.distance(pts, poly)
            inside = shapely.contains(poly, pts)
            d_bound = shapely.distance(pts, shapely.boundary(poly))
            intrusion = np.where(inside, robot_radius + d_bound,
                                  np.clip(robot_radius - d, 0, None))
        penalty += np.sum(intrusion ** 2)
    return penalty

if __name__ == "__main__":
    from map_utils import generate_random_map
    from bezier_utils import bezier_curve
    start, goal = np.array([0, 0]), np.array([10, 10])
    cps = np.array([[1, 2], [5, 2], [7 , 8]])
    curve = bezier_curve(start, cps, goal, 20)
    obstacles = generate_random_map(seed=1)
    pen = collision_penalty(curve, obstacles)
    print(pen)  # 0.0 if curve is obstacle-free