# run_multi_robot_inference.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle

from map_utils import generate_random_map, rasterize_map
from env import GlobalPathEnv
from policy import ObstacleAwarePolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_policy(checkpoint):
    policy = ObstacleAwarePolicy().to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def run_multi_robot_inference(checkpoint, robot_tasks, seed=10,
                               bounds=(0, 12), n_obstacles=None):
    n_robots = len(robot_tasks)
    policy = load_policy(checkpoint)

    if n_obstacles is None:
        n_obstacles = np.random.randint(4, 8)
    obstacles = generate_random_map(seed=seed, n_obstacles=np.random.randint(6, 10))
    grid = rasterize_map(obstacles, bounds=bounds)
    grid_t = torch.tensor(grid).unsqueeze(0).float().to(device)
    grid_batch_full = grid_t.unsqueeze(0).repeat(n_robots, 1, 1, 1)

    # --- find the max number of tasks across all robots, for padding ---
    max_n_tasks = max(len(r["task_points"]) for r in robot_tasks)

    envs, task_points_list, start_positions, n_real_tasks = [], [], [], []
    for r_cfg in robot_tasks:
        tp_real = np.array(r_cfg["task_points"], dtype=np.float32)
        n_real = len(tp_real)
        sp = np.array(r_cfg["start"], dtype=np.float32)

        # pad task_points with dummy far-away coords up to max_n_tasks
        pad_n = max_n_tasks - n_real
        if pad_n > 0:
            pad_coords = np.tile(tp_real[-1:], (pad_n, 1))  # repeat last point as filler
            tp_padded = np.vstack([tp_real, pad_coords])
        else:
            tp_padded = tp_real

        # pad precedence matrix to (max_n_tasks, max_n_tasks)
        precedence_real = r_cfg.get("precedence", np.zeros((n_real, n_real), dtype=bool))
        precedence_padded = np.zeros((max_n_tasks, max_n_tasks), dtype=bool)
        precedence_padded[:n_real, :n_real] = precedence_real

        w = np.array(r_cfg.get("w", [0.6, 0.2, 0.2]), dtype=np.float32)

        env = GlobalPathEnv(bounds=bounds)
        env.reset(obstacles, grid, tp_padded, precedence_padded, sp, w)
        env.n_real_tasks = n_real   # remember how many are "real" vs padding

        envs.append(env)
        task_points_list.append(tp_real)   # keep REAL (unpadded) list for plotting
        start_positions.append(sp)
        n_real_tasks.append(n_real)

    segment_curves = [[] for _ in range(n_robots)]
    chosen_orders = [[] for _ in range(n_robots)]
    active = [True] * n_robots

    with torch.no_grad():
        while any(active):
            active_idx = [i for i in range(n_robots) if active[i]]
            B = len(active_idx)

            pos_batch = torch.tensor(
                np.stack([envs[i].current_pos for i in active_idx])
            ).float().to(device)
            tasks_batch = torch.tensor(
                np.stack([envs[i].task_points for i in active_idx])
            ).float().to(device)

            # mask out padded task slots (indices >= n_real_tasks for that robot)
            mask_list = []
            for i in active_idx:
                m = envs[i]._get_state()["available_mask"].copy()
                m[envs[i].n_real_tasks:] = False   # padding slots always unavailable
                mask_list.append(m)
            mask_batch = torch.tensor(np.stack(mask_list)).to(device)

            w_batch = torch.tensor(
                np.stack([envs[i].w for i in active_idx])
            ).float().to(device)
            grid_batch_active = grid_batch_full[:B]

            logits, cps, idx = policy(grid_batch_active, pos_batch, tasks_batch,
                                       mask_batch, w_batch)

            cps_np = cps.cpu().numpy()
            idx_np = idx.cpu().numpy()

            for b, robot_i in enumerate(active_idx):
                _, _, done, info = envs[robot_i].step(idx_np[b], cps_np[b])
                segment_curves[robot_i].append(info["curve"])
                chosen_orders[robot_i].append(int(idx_np[b]))
                # done should now reflect only REAL tasks, not padding
                if np.all(envs[robot_i].visited[:envs[robot_i].n_real_tasks]):
                    active[robot_i] = False

    for r in range(n_robots):
        print(f"Robot {r}: task order {chosen_orders[r]}")

    plot_multi_robot(obstacles, task_points_list, start_positions, segment_curves, bounds)
    return segment_curves, chosen_orders


def plot_multi_robot(obstacles, task_points_list, start_positions, segment_curves, bounds):
    fig, ax = plt.subplots(figsize=(9, 8))

    for obs in obstacles:
        if obs["type"] == "circle":
            cx, cy = obs["center"]
            ax.add_patch(Circle((cx, cy), obs["radius"], color="firebrick", alpha=0.55))
        else:
            xs, ys = obs["shape"].exterior.xy
            ax.add_patch(MplPolygon(list(zip(xs, ys)), closed=True,
                                     color="firebrick", alpha=0.55))

    robot_colors = plt.cm.tab10(np.linspace(0, 1, len(segment_curves)))

    for r, curves in enumerate(segment_curves):
        for i, curve in enumerate(curves):
            label = f"Robot {r}" if i == 0 else None
            ax.plot(curve[:, 0], curve[:, 1], "-", color=robot_colors[r],
                     linewidth=2.5, label=label)
        ax.plot(*start_positions[r], "s", color=robot_colors[r], markersize=12)
        for tp in task_points_list[r]:
            ax.plot(*tp, "D", color=robot_colors[r], markersize=8, alpha=0.7)

    ax.set_xlim(bounds[0] - 1, bounds[1] + 1)
    ax.set_ylim(bounds[0] - 1, bounds[1] + 1)
    ax.set_aspect("equal")
    ax.set_title(f"Batched Multi-Robot Rollout ({len(segment_curves)} robots)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- explicit robot -> task assignment ---
    robot_tasks = [
        {"start": np.array([0.0, 0.0]),
         "task_points": np.array([[3, 8], [7, 2]]),
         "precedence": np.array([[False, True], [False, False]])},

        {"start": np.array([12.0, 0.0]),
         "task_points": np.array([[9, 9], [5, 5], [2, 10]]),
         "w": np.array([0.3, 0.3, 0.4])},

        {"start": np.array([0.0, 12.0]),
         "task_points": np.array([[6, 6], [11, 3]])},

        {"start": np.array([12.0, 12.0]),
         "task_points": np.array([[1, 1], [8, 8], [4, 3]])},
    ]

    run_multi_robot_inference(
        checkpoint="models\global_policy_20260707_114414_ep400.pt",
        robot_tasks=robot_tasks,
        seed=10,
    )

    