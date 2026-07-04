import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import make_default_scenario
from de_mpc import DEMPCPlanner
from rangefinder import simulate_rangefinder


def run_episode(seed=1, max_steps=400, horizon=10, verbose=True,
                 n_rays=15, sensor_range=5.0, record_data=False,n_static=4, n_dynamic=3, n_tasks=4):
    env = make_default_scenario(seed=seed, n_static=n_static, n_dynamic=n_dynamic, n_tasks=n_tasks)
    obs = env.reset()

    planner = DEMPCPlanner(horizon=horizon, dt=env.dt, v_max=env.robot.v_max,
                            omega_max=env.robot.omega_max,
                            robot_radius=env.robot.radius, seed=seed)

    solve_times = []
    dataset = []  # only populated if record_data=True
    for step in range(max_steps):
        t0 = time.time()
        # first solve is cold (more iterations), subsequent solves are warm-started (fewer)
        planner.maxiter = 40 if planner.prev_solution is None else 15
        (v0, omega0), seq, cost = planner.plan(
            robot_state=tuple(obs["robot_state"]),
            goal=tuple(obs["goal"]),
            static_obstacles=obs["static_obstacles"],
            dynamic_obstacles=obs["dynamic_obstacles"],
        )
        solve_times.append(time.time() - t0)

        if record_data:
            ranges = simulate_rangefinder(
                robot_state=tuple(obs["robot_state"]),
                static_obstacles=obs["static_obstacles"],
                dynamic_obstacles=obs["dynamic_obstacles"],
                world_bounds=env.world_bounds,
                n_rays=n_rays, max_range=sensor_range,
            )
            rx, ry, rtheta = obs["robot_state"]
            gx, gy = obs["goal"]
            goal_dist = float(np.hypot(gx - rx, gy - ry))
            goal_bearing = float((np.arctan2(gy - ry, gx - rx) - rtheta + np.pi) % (2 * np.pi) - np.pi)
            dataset.append({
                "step": step,
                "robot_state": np.array([rx, ry, rtheta], dtype=float),
                "goal_dist": goal_dist,
                "goal_bearing": goal_bearing,
                "ranges": ranges,               # (n_rays,)
                "action": np.array([v0, omega0], dtype=float),
            })

        obs, done, info = env.step(v0, omega0)

        if verbose and info["reached_goal"]:
            print(f"[step {step}] reached goal -> now targeting goal_idx={obs['goal_idx']}")
        if info["collided"]:
            print(f"[step {step}] COLLISION ({info['collision_kind']})")
            break
        if done:
            print(f"[step {step}] all tasks complete, returned to depot")
            break
    else:
        print("Max steps reached without finishing all tasks.")

    print(f"Total steps: {step+1} | avg solve time: {np.mean(solve_times):.3f}s "
          f"(first: {solve_times[0]:.3f}s, mean after warm-start: {np.mean(solve_times[1:]):.3f}s)")
    return env, solve_times, dataset


def save_dataset(dataset, path="demo_data.npz"):
    """Pack a list of per-step dicts (from run_episode with record_data=True)
    into flat arrays and save to a single .npz file."""
    if not dataset:
        print("Empty dataset, nothing saved.")
        return
    np.savez(
        path,
        robot_state=np.stack([d["robot_state"] for d in dataset]),
        goal_dist=np.array([d["goal_dist"] for d in dataset]),
        goal_bearing=np.array([d["goal_bearing"] for d in dataset]),
        ranges=np.stack([d["ranges"] for d in dataset]),
        action=np.stack([d["action"] for d in dataset]),
    )
    print(f"Saved {len(dataset)} timesteps to {path}")


def plot_trajectory(env, save_path="traj/episode_trajectory.png"):
    fig, ax = plt.subplots(figsize=(7, 7))
    xmin, xmax, ymin, ymax = env.world_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    traj = np.array(env.history["robot"])
    ax.plot(traj[:, 0], traj[:, 1], "b-", linewidth=1.5, label="Robot path")
    ax.plot(env.start[0], env.start[1], "ks", markersize=10, label="Depot")

    for i, tp in enumerate(env.task_points):
        ax.plot(tp[0], tp[1], "g^", markersize=10)
        ax.annotate(f"T{i+1}", (tp[0], tp[1]), textcoords="offset points",
                    xytext=(6, 6), fontsize=9, color="darkgreen")

    for o in env.static_obstacles:
        ax.add_patch(plt.Circle((o.x, o.y), o.radius, color="gray", alpha=0.6))

    # dynamic obstacle traces (faint) + final position
    dyn_hist = env.history["dyn_obs"]  # list over time of list-of-positions
    n_dyn = len(dyn_hist[0])
    for i in range(n_dyn):
        path = np.array([frame[i] for frame in dyn_hist])
        ax.plot(path[:, 0], path[:, 1], color="tomato", alpha=0.35, linewidth=1)
        ax.add_patch(plt.Circle((path[-1, 0], path[-1, 1]),
                                 env.dynamic_obstacles[i].radius, color="tomato", alpha=0.7))

    ax.plot(traj[-1, 0], traj[-1, 1], "bo", markersize=8, label="Robot (final)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"DE-MPC single-robot trajectory (seed={SEED})")
    plt.savefig(save_path, dpi=130)
    print("Saved:", save_path)

SEED = [42,23,71,254,321]
N_STATIC = 8
N_DYNAMIC = 4
N_TASKS = 5
CLOSED_LOOP_HORIZON = 10    # horizon used when RECEDING_HORIZON = True
OPEN_LOOP_HORIZON = 70      # horizon used when RECEDING_HORIZON = False (must cover a full segment)
MAX_STEPS = 1000

if __name__ == "__main__":
    env, solve_times, dataset = run_episode(seed=SEED, max_steps=MAX_STEPS, horizon=CLOSED_LOOP_HORIZON, record_data=True, n_static=N_STATIC, n_dynamic=N_DYNAMIC, n_tasks=N_TASKS)
    plot_trajectory(env)
    save_dataset(dataset, path=f"{SEED}_{N_TASKS}.npz")