# inference_plot.py
import numpy as np
from zmq import device
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle

from map_utils import generate_random_map, rasterize_map
from env import GlobalPathEnv
from policy import ObstacleAwarePolicy


def run_inference(checkpoint="global_policy.pt", seed=999,
                   task_points=None, precedence=None,
                   start_pos=None, w=None, bounds=(0, 12)):

    # --- default scenario if nothing supplied ---
    if task_points is None:
        task_points = np.array([[3.0, 8.0], [7.0, 2.0], [10.0, 9.0]])
    if precedence is None:
        precedence = np.zeros((len(task_points), len(task_points)), dtype=bool)
        precedence[0, 1] = True   # task 0 must be visited before task 1
    if start_pos is None:
        start_pos = np.array([0.0, 0.0])
    if w is None:
        w = np.array([0.6, 0.2, 0.2])   # [length, energy, collision] preference

    # --- load trained policy ---
    policy = ObstacleAwarePolicy().to(device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    policy.load_state_dict(state_dict)
    policy.eval()

    # --- build environment / map ---
    obstacles = generate_random_map(seed=seed, n_obstacles=np.random.randint(6, 7))
    grid = rasterize_map(obstacles, bounds=bounds)
    env = GlobalPathEnv(bounds=bounds)
    state = env.reset(obstacles, grid, task_points, precedence, start_pos, w)

    grid_t = torch.tensor(grid).unsqueeze(0).unsqueeze(0).float().to(device)
    w_t = torch.tensor(w).unsqueeze(0).float().to(device)

    segment_curves = []
    chosen_order = []

    # --- rollout: keep predicting until every task is visited ---
    with torch.no_grad():
        while not env.done:
            pos_t = torch.tensor(state["current_pos"] / 12.0).unsqueeze(0).float().to(device)
            tasks_t = torch.tensor(state["task_points"] / 12.0).unsqueeze(0).float().to(device)
            mask_t = torch.tensor(state["available_mask"]).unsqueeze(0).to(device)

            logits, cps, cp_std, idx = policy(grid_t, pos_t, tasks_t, mask_t, w_t)
            task_idx = idx.item()
            cps_np = cps.squeeze(0).cpu().numpy()

            state, reward, done, info = env.step(task_idx, cps_np)
            segment_curves.append(info["curve"])
            chosen_order.append(task_idx)

    print("Task visiting order:", chosen_order)
    plot_result(obstacles, task_points, start_pos, segment_curves, bounds)
    return segment_curves, chosen_order


def plot_result(obstacles, task_points, start_pos, segment_curves, bounds):
    fig, ax = plt.subplots(figsize=(8, 7))

    # obstacles
    for obs in obstacles:
        if obs["type"] == "circle":
            cx, cy = obs["center"]
            ax.add_patch(Circle((cx, cy), obs["radius"], color="firebrick", alpha=0.55))
        else:
            xs, ys = obs["shape"].exterior.xy
            ax.add_patch(MplPolygon(list(zip(xs, ys)), closed=True,
                                     color="firebrick", alpha=0.55))

    # each predicted segment in a different color
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(segment_curves)))
    for i, curve in enumerate(segment_curves):
        ax.plot(curve[:, 0], curve[:, 1], "-", color=colors[i],
                 linewidth=2.5, label=f"Segment {i+1}")

    ax.plot(*start_pos, "gs", markersize=13, label="Start")
    for i, tp in enumerate(task_points):
        ax.plot(*tp, "D", color="orange", markersize=10)
        ax.annotate(f"T{i}", (tp[0] + 0.15, tp[1] + 0.15))

    ax.set_xlim(bounds[0] - 1, bounds[1] + 1)
    ax.set_ylim(bounds[0] - 1, bounds[1] + 1)
    ax.set_aspect("equal")
    ax.set_title("DRL Policy Rollout: Task Sequence + Bezier Path")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    # plt.savefig("inference_result.png", dpi=150)
    plt.show()
    # print("Saved plot to inference_result.png")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_inference(checkpoint="models\global_policy_ep200.pt", seed=42)