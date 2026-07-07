# bezier_utils.py --- builds a Bezier curve from [start, 3 free control points, goal] and samples points along it.
import numpy as np
from scipy.special import comb

def bezier_curve(start, control_points, goal, n_samples=50):
    """control_points: (3,2) array of free intermediate points."""
    pts = np.vstack([start, control_points, goal])
    n = len(pts) - 1
    t = np.linspace(0, 1, n_samples)
    curve = np.zeros((n_samples, 2))
    for i, p in enumerate(pts):
        bernstein = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        curve += np.outer(bernstein, p)
    return curve

if __name__ == "__main__":
    import numpy as np
    start, goal = np.array([0, 0]), np.array([10, 10])
    cps = np.array([[3, 2], [5, 5], [7, 8]])
    curve = bezier_curve(start, cps, goal, 20)
    print(curve.shape)  # (20, 2)