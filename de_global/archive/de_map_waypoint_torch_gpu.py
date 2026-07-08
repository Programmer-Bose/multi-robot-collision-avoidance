"""
GPU-Accelerated Differential Evolution over Waypoint Positions (PyTorch)
==========================================================================
Same waypoint-based representation as de_map_waypoint_top20.py (6 free
points / 12 DE variables per segment), but DE itself is reimplemented
manually in PyTorch so the ENTIRE population is evaluated in parallel as
batched tensor ops on GPU (if available). This is for benchmarking against
the SciPy CPU version.

Notes:
- Circle obstacles: fully vectorized distance formula.
- Polygon obstacles (square/rectangle/u_shape): vectorized point-in-polygon
  (crossing number) + min distance to polygon edges, batched over all
  population members and all curve sample points at once.
- DE mutation/crossover/selection are vectorized across the whole
  population (no Python-level per-individual loop).
"""

import numpy as np
import torch
import json
import os
import time
import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ----------------------------------------------------------------------
# 1. Global Problem Setup
# ----------------------------------------------------------------------

WAYPOINTS = []          # list of np.array([x,y])
CIRCLE_OBS = []         # list of (cx, cy, r)
POLY_OBS = []           # list of (N_i, 2) vertex arrays (closed order)
N_SEGMENTS = 0

N_FREE_POINTS = 6
N_VARS_PER_SEGMENT = N_FREE_POINTS * 2
N_SAMPLES_PER_SUBSEGMENT = 40

BOUNDS_MIN = np.array([0.0, 0.0])
BOUNDS_MAX = np.array([12.0, 12.0])

ROBOT_RADIUS = 0.3
WARM_START_FRACTION = 0.5
WARM_START_NOISE_STD = 0.65
WARM_START_TOP_PCT = 0.20

W_LENGTH = 1.0
W_COLLISION = 800.0
W_BOUNDARY = 800.0
W_CURVATURE = 0.5

# ----------------------------------------------------------------------
# 2. Map Loading
# ----------------------------------------------------------------------

def load_map_config(json_path):
    global WAYPOINTS, CIRCLE_OBS, POLY_OBS, N_SEGMENTS, BOUNDS_MAX

    with open(json_path, 'r') as f:
        data = json.load(f)

    orig_w, orig_h = data["map_metadata"]["size"]
    target_scale = 12.0
    scale_factor = target_scale / orig_w

    def scale_pt(pt):
        return np.array([pt[0] * scale_factor, (orig_h - pt[1]) * scale_factor])

    WAYPOINTS = [scale_pt(data["start_position"])]
    for task_id in data["task_sequence"]:
        WAYPOINTS.append(scale_pt(data["task_points"][str(task_id)]))
    if data.get("goal_position"):
        WAYPOINTS.append(scale_pt(data["goal_position"]))

    N_SEGMENTS = len(WAYPOINTS) - 1
    BOUNDS_MAX = np.array([target_scale, target_scale])

    CIRCLE_OBS = []
    POLY_OBS = []

    for obs in data["obstacles"]:
        obs_type = obs["type"]
        cx, cy = scale_pt(obs["position"])

        if obs_type == "circle":
            r_scaled = obs["radius"] * scale_factor
            CIRCLE_OBS.append((cx, cy, r_scaled))

        elif obs_type == "square":
            h = (obs["size"] * scale_factor) / 2.0
            verts = np.array([[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]])
            POLY_OBS.append(verts)

        elif obs_type == "rectangle":
            hw = (obs["width"] * scale_factor) / 2.0
            hh = (obs["height"] * scale_factor) / 2.0
            verts = np.array([[cx - hw, cy - hh], [cx + hw, cy - hh], [cx + hw, cy + hh], [cx - hw, cy + hh]])
            POLY_OBS.append(verts)

        elif obs_type == "u_shape":
            h = (obs["size"] * scale_factor) / 2.0
            t = obs["thickness"] * scale_factor
            # left arm, right arm, bottom bar as three separate rectangles
            left = np.array([[cx - h, cy - h], [cx - h + t, cy - h], [cx - h + t, cy + h], [cx - h, cy + h]])
            right = np.array([[cx + h - t, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx + h - t, cy + h]])
            bottom = np.array([[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy - h + t], [cx - h, cy - h + t]])
            POLY_OBS.extend([left, right, bottom])

    # Pre-convert obstacle data to torch tensors on DEVICE
    global CIRCLE_T, POLY_T_LIST
    if CIRCLE_OBS:
        CIRCLE_T = torch.tensor(CIRCLE_OBS, dtype=torch.float32, device=DEVICE)  # (n_circ, 3)
    else:
        CIRCLE_T = torch.empty((0, 3), dtype=torch.float32, device=DEVICE)

    POLY_T_LIST = [torch.tensor(p, dtype=torch.float32, device=DEVICE) for p in POLY_OBS]

CIRCLE_T = torch.empty((0, 3), dtype=torch.float32, device=DEVICE)
POLY_T_LIST = []

# ----------------------------------------------------------------------
# 3. Waypoint Path Construction (batched over population, torch tensors)
# ----------------------------------------------------------------------

def baseline_free_points_t(seg_start, seg_end):
    """seg_start, seg_end: (2,) torch tensors. Returns (N_FREE_POINTS, 2)."""
    segment_vec = seg_end - seg_start
    t_values = torch.linspace(1.0 / (N_FREE_POINTS + 1), N_FREE_POINTS / (N_FREE_POINTS + 1),
                               N_FREE_POINTS, device=DEVICE)
    return seg_start.unsqueeze(0) + t_values.unsqueeze(1) * segment_vec.unsqueeze(0)

def population_to_points(pop, seg_start, seg_end):
    """
    pop: (P, N_VARS_PER_SEGMENT) tensor of deltas.
    Returns: (P, N_FREE_POINTS, 2) actual free-point coordinates.
    """
    P = pop.shape[0]
    deltas = pop.view(P, N_FREE_POINTS, 2)
    baseline = baseline_free_points_t(seg_start, seg_end)  # (N_FREE_POINTS, 2)
    return baseline.unsqueeze(0) + deltas

def build_full_vertices(pop, seg_start, seg_end):
    """Returns (P, N_FREE_POINTS+2, 2): seg_start, 6 free points, seg_end."""
    P = pop.shape[0]
    free_pts = population_to_points(pop, seg_start, seg_end)  # (P, 6, 2)
    start_b = seg_start.view(1, 1, 2).expand(P, 1, 2)
    end_b = seg_end.view(1, 1, 2).expand(P, 1, 2)
    return torch.cat([start_b, free_pts, end_b], dim=1)

def densify_batch(vertices):
    """
    vertices: (P, M, 2) polyline vertices per individual.
    Returns: (P, n_total_samples, 2) densely sampled points along each polyline.
    """
    P, M, _ = vertices.shape
    n_edges = M - 1
    t = torch.linspace(0.0, 1.0, N_SAMPLES_PER_SUBSEGMENT, device=DEVICE)  # (S,)
    p0 = vertices[:, :-1, :]   # (P, n_edges, 2)
    p1 = vertices[:, 1:, :]    # (P, n_edges, 2)
    # (P, n_edges, S, 2)
    seg_pts = p0.unsqueeze(2) + t.view(1, 1, -1, 1) * (p1 - p0).unsqueeze(2)
    return seg_pts.reshape(P, n_edges * N_SAMPLES_PER_SUBSEGMENT, 2)

# ----------------------------------------------------------------------
# 4. Cost Functions (fully vectorized over population)
# ----------------------------------------------------------------------

def circle_penalty(curve):
    """curve: (P, K, 2). Returns (P,) sum-of-squared-intrusion penalty."""
    if CIRCLE_T.shape[0] == 0:
        return torch.zeros(curve.shape[0], device=DEVICE)
    cx = CIRCLE_T[:, 0].view(1, 1, -1)
    cy = CIRCLE_T[:, 1].view(1, 1, -1)
    r = CIRCLE_T[:, 2].view(1, 1, -1)
    xs = curve[:, :, 0].unsqueeze(-1)   # (P,K,1)
    ys = curve[:, :, 1].unsqueeze(-1)
    d = torch.sqrt((xs - cx) ** 2 + (ys - cy) ** 2 + 1e-12)
    safe_r = r + ROBOT_RADIUS
    intrusion = torch.clamp(safe_r - d, min=0.0)
    return (intrusion ** 2).sum(dim=(1, 2))

def point_in_polygon_batch(pts, poly):
    """
    pts: (P, K, 2), poly: (V, 2). Returns (P, K) bool inside mask via crossing number.
    """
    P, K, _ = pts.shape
    V = poly.shape[0]
    x = pts[:, :, 0].unsqueeze(-1)   # (P,K,1)
    y = pts[:, :, 1].unsqueeze(-1)
    x1 = poly[:, 0].view(1, 1, V)
    y1 = poly[:, 1].view(1, 1, V)
    x2 = poly[:, 0].roll(-1).view(1, 1, V)
    y2 = poly[:, 1].roll(-1).view(1, 1, V)

    cond = ((y1 > y) != (y2 > y))
    denom = (y2 - y1)
    denom = torch.where(denom.abs() < 1e-12, torch.full_like(denom, 1e-12), denom)
    x_intersect = x1 + (y - y1) * (x2 - x1) / denom
    cross = cond & (x < x_intersect)
    inside = (cross.sum(dim=-1) % 2) == 1
    return inside  # (P, K)

def point_to_polygon_edge_dist(pts, poly):
    """pts: (P,K,2), poly: (V,2). Returns (P,K) min distance to polygon edges."""
    V = poly.shape[0]
    a = poly                      # (V,2)
    b = poly.roll(-1, dims=0)     # (V,2)
    ab = (b - a)                  # (V,2)
    ab_len2 = (ab ** 2).sum(-1).clamp(min=1e-12)  # (V,)

    p = pts.unsqueeze(2)          # (P,K,1,2)
    a_ = a.view(1, 1, V, 2)
    ab_ = ab.view(1, 1, V, 2)
    ap = p - a_                                    # (P,K,V,2)
    t = (ap * ab_).sum(-1) / ab_len2.view(1, 1, V)  # (P,K,V)
    t = t.clamp(0.0, 1.0)
    closest = a_ + t.unsqueeze(-1) * ab_            # (P,K,V,2)
    d = torch.sqrt(((p - closest) ** 2).sum(-1) + 1e-12)  # (P,K,V)
    return d.min(dim=-1).values  # (P,K)

def polygon_penalty(curve):
    if len(POLY_T_LIST) == 0:
        return torch.zeros(curve.shape[0], device=DEVICE)
    total = torch.zeros(curve.shape[0], device=DEVICE)
    for poly in POLY_T_LIST:
        inside = point_in_polygon_batch(curve, poly)          # (P,K)
        edge_d = point_to_polygon_edge_dist(curve, poly)       # (P,K)
        outside_intrusion = torch.clamp(ROBOT_RADIUS - edge_d, min=0.0)
        inside_intrusion = ROBOT_RADIUS + edge_d
        intrusion = torch.where(inside, inside_intrusion, outside_intrusion)
        total = total + (intrusion ** 2).sum(dim=1)
    return total

def path_length_batch(curve):
    diffs = curve[:, 1:, :] - curve[:, :-1, :]
    return torch.sqrt((diffs ** 2).sum(-1) + 1e-12).sum(dim=1)

def boundary_penalty_batch(curve):
    """curve: (P, K, 2). Penalize points outside [BOUNDS_MIN, BOUNDS_MAX]."""
    low = torch.tensor(BOUNDS_MIN, dtype=torch.float32, device=DEVICE)
    high = torch.tensor(BOUNDS_MAX, dtype=torch.float32, device=DEVICE)
    below = torch.clamp(low - curve, min=0.0)   # (P,K,2)
    above = torch.clamp(curve - high, min=0.0)  # (P,K,2)
    intrusion = below + above
    return (intrusion ** 2).sum(dim=(1, 2))

def curvature_penalty_batch(curve):
    if curve.shape[1] < 3:
        return torch.zeros(curve.shape[0], device=DEVICE)
    v1 = curve[:, 1:-1, :] - curve[:, :-2, :]
    v2 = curve[:, 2:, :] - curve[:, 1:-1, :]
    n1 = torch.sqrt((v1 ** 2).sum(-1) + 1e-12) + 1e-9
    n2 = torch.sqrt((v2 ** 2).sum(-1) + 1e-12) + 1e-9
    cos_angle = ((v1 * v2).sum(-1) / (n1 * n2)).clamp(-1.0, 1.0)
    return (torch.arccos(cos_angle) ** 2).sum(dim=1)

def batch_cost(pop, seg_start, seg_end):
    vertices = build_full_vertices(pop, seg_start, seg_end)
    curve = densify_batch(vertices)
    cost = W_LENGTH * path_length_batch(curve)
    cost = cost + W_COLLISION * (circle_penalty(curve) + polygon_penalty(curve))
    cost = cost + W_CURVATURE * curvature_penalty_batch(curve)
    cost = cost + W_BOUNDARY * boundary_penalty_batch(curve)   # <-- add this
    return cost

# ----------------------------------------------------------------------
# 5. Manual Vectorized Differential Evolution (PyTorch)
# ----------------------------------------------------------------------

def de_torch(seg_start_np, seg_end_np, bounds_low, bounds_high, popsize=100, maxiter=100,
             mutation=0.8, recombination=0.9, init_population=None, verbose_prefix=""):
    seg_start = torch.tensor(seg_start_np, dtype=torch.float32, device=DEVICE)
    seg_end = torch.tensor(seg_end_np, dtype=torch.float32, device=DEVICE)
    n_vars = N_VARS_PER_SEGMENT
    low = torch.tensor(bounds_low, dtype=torch.float32, device=DEVICE)
    high = torch.tensor(bounds_high, dtype=torch.float32, device=DEVICE)

    if init_population is not None:
        pop = torch.tensor(init_population, dtype=torch.float32, device=DEVICE)
        pop = torch.clamp(pop, low, high)
    else:
        pop = low + torch.rand(popsize, n_vars, device=DEVICE) * (high - low)

    costs = batch_cost(pop, seg_start, seg_end)

    for gen in range(maxiter):
        idx = torch.arange(popsize, device=DEVICE)
        # pick 3 distinct random partners per individual (r1, r2, r3 != idx)
        r1 = torch.randint(0, popsize, (popsize,), device=DEVICE)
        r2 = torch.randint(0, popsize, (popsize,), device=DEVICE)
        r3 = torch.randint(0, popsize, (popsize,), device=DEVICE)
        clash = (r1 == idx) | (r2 == idx) | (r3 == idx) | (r1 == r2) | (r2 == r3) | (r1 == r3)
        # resample clashing indices a few times (cheap, good enough)
        for _ in range(4):
            if not clash.any():
                break
            n_clash = clash.sum()
            r1[clash] = torch.randint(0, popsize, (int(n_clash),), device=DEVICE)
            r2[clash] = torch.randint(0, popsize, (int(n_clash),), device=DEVICE)
            r3[clash] = torch.randint(0, popsize, (int(n_clash),), device=DEVICE)
            clash = (r1 == idx) | (r2 == idx) | (r3 == idx) | (r1 == r2) | (r2 == r3) | (r1 == r3)

        best_idx = torch.argmin(costs)
        best = pop[best_idx].unsqueeze(0).expand(popsize, -1)

        # best1bin-style mutation: best + F*(r1 - r2)
        mutant = best + mutation * (pop[r1] - pop[r2])
        mutant = torch.clamp(mutant, low, high)

        cross_mask = torch.rand(popsize, n_vars, device=DEVICE) < recombination
        force_idx = torch.randint(0, n_vars, (popsize,), device=DEVICE)
        cross_mask[idx, force_idx] = True

        trial = torch.where(cross_mask, mutant, pop)
        trial_costs = batch_cost(trial, seg_start, seg_end)

        improved = trial_costs < costs
        pop = torch.where(improved.unsqueeze(-1), trial, pop)
        costs = torch.where(improved, trial_costs, costs)

        best_cost = costs.min().item()
        print(f"{verbose_prefix}Gen {gen + 1}/{maxiter}: best cost={best_cost:.4f}")

    best_idx = torch.argmin(costs)
    return pop[best_idx].cpu().numpy(), costs[best_idx].item(), pop.cpu().numpy(), costs.cpu().numpy()

# ----------------------------------------------------------------------
# 6. Sequential Solve with Top-20% Warm Start
# ----------------------------------------------------------------------

def build_delta_bounds(seg_start, seg_end):
    segment_length = np.linalg.norm(seg_end - seg_start)
    delta_bound = max(5, 0.8 * segment_length)
    low = np.full(N_VARS_PER_SEGMENT, -delta_bound)
    high = np.full(N_VARS_PER_SEGMENT, delta_bound)
    return low, high

def build_warm_start_population(prev_best_pop, popsize, n_vars):
    n_seeded = int(popsize * WARM_START_FRACTION)
    n_random = popsize - n_seeded
    population = []
    if prev_best_pop is not None and len(prev_best_pop) > 0:
        for _ in range(n_seeded):
            base = prev_best_pop[np.random.randint(len(prev_best_pop))]
            population.append(base + np.random.normal(0.0, WARM_START_NOISE_STD, size=n_vars))
    while len(population) < n_seeded + n_random:
        population.append(np.random.uniform(-1.0, 1.0, size=n_vars))
    return np.array(population)

def solve_sequential(popsize=100, maxiter=50):
    all_delta_vectors = []
    prev_best_pop = None
    total_cost = 0.0

    for seg in range(N_SEGMENTS):
        seg_start = WAYPOINTS[seg]
        seg_end = WAYPOINTS[seg + 1]
        low, high = build_delta_bounds(seg_start, seg_end)
        init_population = build_warm_start_population(prev_best_pop, popsize, N_VARS_PER_SEGMENT)
        init_population = np.clip(init_population, low, high)

        print(f"Segment {seg + 1}/{N_SEGMENTS} "
              f"(warm_start={'yes' if prev_best_pop is not None else 'no (first segment)'})")

        best_x, best_cost, final_pop, final_costs = de_torch(
            seg_start, seg_end, low, high,
            popsize=popsize, maxiter=maxiter,
            init_population=init_population,
            verbose_prefix="  "
        )

        n_top = max(1, int(WARM_START_TOP_PCT * len(final_pop)))
        top_idx = np.argsort(final_costs)[:n_top]
        prev_best_pop = final_pop[top_idx]

        print(f"Segment {seg + 1}/{N_SEGMENTS} done: cost={best_cost:.4f}, elites_kept={n_top}")
        all_delta_vectors.append(best_x)
        total_cost += best_cost

    return all_delta_vectors, total_cost

# ----------------------------------------------------------------------
# 7. Export / Plot (numpy versions for final result)
# ----------------------------------------------------------------------

def baseline_free_points_np(seg_start, seg_end):
    segment_vec = seg_end - seg_start
    t_values = np.linspace(1.0 / (N_FREE_POINTS + 1), N_FREE_POINTS / (N_FREE_POINTS + 1), N_FREE_POINTS)
    return np.array([seg_start + t * segment_vec for t in t_values])

def delta_to_points_np(flat_deltas, seg_start, seg_end):
    deltas = flat_deltas.reshape(N_FREE_POINTS, 2)
    return baseline_free_points_np(seg_start, seg_end) + deltas

def densify_polyline_np(vertices, n_samples):
    pts = []
    for i in range(len(vertices) - 1):
        p0, p1 = vertices[i], vertices[i + 1]
        t = np.linspace(0.0, 1.0, n_samples, endpoint=(i == len(vertices) - 2))
        pts.append(p0[None, :] + t[:, None] * (p1 - p0)[None, :])
    return np.vstack(pts)

def build_full_path(delta_vectors):
    curves = []
    for seg in range(N_SEGMENTS):
        seg_start, seg_end = WAYPOINTS[seg], WAYPOINTS[seg + 1]
        free_pts = delta_to_points_np(delta_vectors[seg], seg_start, seg_end)
        vertices = np.vstack([seg_start, free_pts, seg_end])
        curves.append(densify_polyline_np(vertices, N_SAMPLES_PER_SUBSEGMENT))
    return curves

def export_path(curves, delta_vectors, filename="planned_path.json"):
    segments_data = []
    for seg in range(N_SEGMENTS):
        seg_start, seg_end = WAYPOINTS[seg], WAYPOINTS[seg + 1]
        free_pts = delta_to_points_np(delta_vectors[seg], seg_start, seg_end)
        segments_data.append({
            "segment_index": seg,
            "start": seg_start.tolist(),
            "end": seg_end.tolist(),
            "intermediate_points": free_pts.tolist(),
            "dense_curve": curves[seg].tolist()
        })
    path_data = {
        "metadata": {"num_segments": N_SEGMENTS, "free_points_per_segment": N_FREE_POINTS,
                     "representation": "waypoint_polyline_gpu"},
        "path_segments": segments_data
    }
    with open(filename, 'w') as f:
        json.dump(path_data, f, indent=4)
    print(f"Saved optimized path to {filename}")

def plot_result(delta_vectors, total_cost):
    curves = build_full_path(delta_vectors)
    fig, ax = plt.subplots(figsize=(9, 7))
    for (cx, cy, r) in CIRCLE_OBS:
        ax.add_patch(Circle((cx, cy), r, color="firebrick", alpha=0.55, zorder=2))
    for poly in POLY_OBS:
        ax.add_patch(MplPolygon(poly, closed=True, color="firebrick", alpha=0.55, zorder=2))

    colors = plt.cm.viridis(np.linspace(0, 0.85, N_SEGMENTS))
    for seg, curve in enumerate(curves):
        ax.plot(curve[:, 0], curve[:, 1], "-", color=colors[seg], linewidth=2.5, zorder=3)
        free_pts = delta_to_points_np(delta_vectors[seg], WAYPOINTS[seg], WAYPOINTS[seg + 1])
        ax.plot(free_pts[:, 0], free_pts[:, 1], "o", color=colors[seg], markersize=5, zorder=4)

    for i, wp in enumerate(WAYPOINTS):
        if i == 0:
            ax.plot(*wp, "gs", markersize=13, zorder=5)
        elif i == len(WAYPOINTS) - 1:
            ax.plot(*wp, "r*", markersize=20, zorder=5)
        else:
            ax.plot(*wp, "D", color="orange", markersize=11, zorder=5)

    ax.set_xlim(BOUNDS_MIN[0], BOUNDS_MAX[0])
    ax.set_ylim(BOUNDS_MIN[1], BOUNDS_MAX[1])
    ax.set_aspect("equal")
    ax.set_title(f"GPU (PyTorch) DE over Waypoint Deltas (cost={total_cost:.3f}, device={DEVICE})")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs("solves", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"solves/de_waypoint_gpu_output_{timestamp}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

# ----------------------------------------------------------------------
# 8. Main
# ----------------------------------------------------------------------

def run_planner(input_map_json, output_path_json="planned_path.json"):
    print(f"Loading map configuration from: {input_map_json}")
    load_map_config(input_map_json)

    print(f"Running GPU-vectorized sequential DE ({N_VARS_PER_SEGMENT} vars/segment, device={DEVICE})...")
    start_time = time.perf_counter()
    delta_vectors, total_cost = solve_sequential()
    end_time = time.perf_counter()

    print(f"Total cost: {total_cost:.4f}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    curves = build_full_path(delta_vectors)
    os.makedirs(os.path.dirname(output_path_json) or ".", exist_ok=True)
    export_path(curves, delta_vectors, output_path_json)
    plot_result(delta_vectors, total_cost)

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_planner("maps/env_map_config_006.json", f"solves/planned_path_gpu_{timestamp}.json")
