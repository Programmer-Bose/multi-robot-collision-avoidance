import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from env import make_default_scenario


def plot_scenario(env, save_path="scenario_check.png"):
    fig, ax = plt.subplots(figsize=(7, 7))
    xmin, xmax, ymin, ymax = env.world_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    # start / depot
    ax.plot(env.start[0], env.start[1], "ks", markersize=10, label="Depot/Start")

    # task points, numbered in visiting order
    for i, tp in enumerate(env.task_points):
        ax.plot(tp[0], tp[1], "g^", markersize=10)
        ax.annotate(f"T{i+1}", (tp[0], tp[1]), textcoords="offset points",
                    xytext=(6, 6), fontsize=9, color="darkgreen")

    # static obstacles
    for o in env.static_obstacles:
        circ = plt.Circle((o.x, o.y), o.radius, color="gray", alpha=0.7)
        ax.add_patch(circ)

    # dynamic obstacles + velocity arrows
    for o in env.dynamic_obstacles:
        circ = plt.Circle((o.x, o.y), o.radius, color="tomato", alpha=0.7)
        ax.add_patch(circ)
        ax.arrow(o.x, o.y, o.vx, o.vy, head_width=0.12, color="darkred")

    # robot
    r = env.robot
    ax.plot(r.x, r.y, "bo", markersize=8, label="Robot")

    ax.legend(loc="upper right")
    ax.set_title("Scenario check: start, tasks, static (gray) & dynamic (red) obstacles")
    plt.savefig(save_path, dpi=130)
    print("Saved:", save_path)


if __name__ == "__main__":
    env = make_default_scenario(seed=1, n_static=4, n_dynamic=3, n_tasks=4)
    env.reset()
    plot_scenario(env)
