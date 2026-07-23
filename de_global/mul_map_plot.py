import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle

from dde_mpc_multi import ReferencePath, load_static_obstacles


PLOT_OUTPUT_DIR = "mpc_logs"


def draw_static_obstacles(ax, obstacles):
    """Draw circular and polygonal obstacles."""
    for obs in obstacles:
        if obs["type"] == "circle":
            cx, cy = obs["center"]
            r = obs["radius"]
            ax.add_patch(
                Circle(
                    (cx, cy),
                    r,
                    color="firebrick",
                    alpha=0.5,
                    zorder=1,
                )
            )
        else:
            ax.add_patch(
                MplPolygon(
                    obs["corners"],
                    closed=True,
                    color="firebrick",
                    alpha=0.5,
                    zorder=1,
                )
            )


def annotate_task_points(ax, ref_path, color, robot_id,
                         label_offset=(0.15, 0.15)):
    """Draw start, task points and goal."""

    n_tasks = len(ref_path.segment_end_s)

    # Start
    ax.plot(
        *ref_path.start_point,
        marker="s",
        markersize=10,
        color=color,
        markeredgecolor="black",
        zorder=5,
    )

    ax.annotate(
        f"{robot_id} Start",
        ref_path.start_point + label_offset,
        fontsize=8,
        color=color,
        weight="bold",
    )

    # Intermediate task points and goal
    for i, s_val in enumerate(ref_path.segment_end_s):

        pt = ref_path.point_at_s(s_val)

        is_goal = (i == n_tasks - 1)

        marker = "*" if is_goal else "D"
        marker_size = 16 if is_goal else 9

        ax.plot(
            *pt,
            marker=marker,
            markersize=marker_size,
            color=color,
            markeredgecolor="black",
            zorder=5,
        )

        label = f"{robot_id} Goal" if is_goal else f"T{i+1}"

        ax.annotate(
            label,
            pt + label_offset,
            fontsize=8,
            color=color,
            weight="bold",
        )


def plot_reference_paths(
    robot_specs,
    map_number,
    arena_size=12.0,
):
    """
    Plot only the reference B-spline paths of multiple robots.
    """

    colors = plt.cm.tab10(np.linspace(0, 1, len(robot_specs)))

    fig, ax = plt.subplots(figsize=(9, 8))

    ax.set_xlim(-1, arena_size+1)
    ax.set_ylim(-1, arena_size+1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax.set_title(f"Global Path by DE - Map {map_number:03d}")

    static_obstacles_drawn = False

    for spec, color in zip(robot_specs, colors):

        ref_path = ReferencePath(spec["control_points_json"])

        # Draw obstacles only once
        if spec.get("map_json") and not static_obstacles_drawn:
            obstacles, _ = load_static_obstacles(spec["map_json"])
            draw_static_obstacles(ax, obstacles)
            static_obstacles_drawn = True

        # -------- Reference Path (Dotted) --------
        ax.plot(
            ref_path.points[:, 0],
            ref_path.points[:, 1],
            "--",                      # dotted/dashed line
            linewidth=1.8,
            color=color,
            label=spec["robot_id"],
            zorder=2,
        )

        # Draw task points
        annotate_task_points(
            ax,
            ref_path,
            color,
            spec["robot_id"],
        )

    ax.legend(fontsize=8)
    plt.tight_layout()

    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

    save_name = f"map_{map_number:03d}_reference_paths.png"
    save_path = os.path.join(PLOT_OUTPUT_DIR, save_name)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {save_path}")


if __name__ == "__main__":

    MAP_NUMBER = "02"
    MAP_NO = 2

    robot_specs = [
        {
            "robot_id": "robot_1",
            "control_points_json":
                f"solves/multi/map_0{MAP_NUMBER}_robot_1_manual_control_points.json",
            "map_json":
                f"maps/map_0{MAP_NUMBER}_robot_1.json",
        },
        {
            "robot_id": "robot_2",
            "control_points_json":
                f"solves/multi/map_0{MAP_NUMBER}_robot_2_manual_control_points.json",
            "map_json":
                f"maps/map_0{MAP_NUMBER}_robot_2.json",
        },
        {
            "robot_id": "robot_3",
            "control_points_json":
                f"solves/multi/map_0{MAP_NUMBER}_robot_3_manual_control_points.json",
            "map_json":
                f"maps/map_0{MAP_NUMBER}_robot_3.json",
        },
    ]

    plot_reference_paths(
        robot_specs=robot_specs,
        map_number=MAP_NO,
        arena_size=12.0,
    )