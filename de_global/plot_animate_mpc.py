"""
plot_animate_mpc.py
--------------------
Reads the per-robot CSV logs produced by dde_mpc_multi.py (columns: step,
time, x, y, theta, v, omega, s_progress, cost) plus each robot's original
reference B-spline path (from its manual_control_points.json), and:

  1. Plays a LIVE animation of every robot moving along its actual
     (traversed) path, alongside its planned reference path and task-point
     markers.
  2. Saves a final static plot (PNG) once the animation finishes, showing
     the full planned path vs. the full traversed path for every robot.

Planned path = dashed line. Traversed (actual) path = solid line with a
moving marker for "robot now". Task points (segment boundaries from the
manual_control_points.json) are annotated with numbered markers.

This script performs NO optimization / simulation itself - it is purely a
post-hoc reader/plotter for CSVs + reference-path JSONs already produced by
dde_mpc_multi.py.
"""

import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle

from dde_mpc_multi import ReferencePath, load_static_obstacles

PLOT_OUTPUT_DIR = "mpc_logs"
ANIMATION_PAUSE = 0.02          # seconds paused per animation frame (matplotlib GUI flush)
FRAME_STRIDE = 1                # set >1 to skip frames and speed up the animation


def read_csv_log(csv_path):
    steps, xs, ys, thetas, vs, omegas, s_progress = [], [], [], [], [], [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            thetas.append(float(row["theta"]))
            vs.append(float(row["v"]))
            omegas.append(float(row["omega"]))
            s_progress.append(float(row["s_progress"]))
    return {
        "step": np.array(steps), "x": np.array(xs), "y": np.array(ys),
        "theta": np.array(thetas), "v": np.array(vs), "omega": np.array(omegas),
        "s": np.array(s_progress),
    }


def draw_static_obstacles(ax, obstacles):
    for obs in obstacles:
        if obs["type"] == "circle":
            cx, cy = obs["center"]
            r = obs["radius"]
            ax.add_patch(Circle((cx, cy), r, color="firebrick", alpha=0.5, zorder=1))
        else:
            ax.add_patch(MplPolygon(obs["corners"], closed=True, color="firebrick", alpha=0.5, zorder=1))


def annotate_task_points(ax, ref_path, color, robot_id, label_offset=(0.15, 0.15)):
    """Marks the start point, every intermediate segment-end (task point),
    and the final goal, with numbered annotations."""
    n_tasks = len(ref_path.segment_end_s)  # includes the final goal as last entry

    ax.plot(*ref_path.start_point, marker="s", markersize=10, color=color,
             markeredgecolor="black", zorder=5)
    ax.annotate(f"{robot_id} start", ref_path.start_point + label_offset,
                fontsize=7, color=color, weight="bold", zorder=6)

    for i, s_val in enumerate(ref_path.segment_end_s):
        pt = ref_path.point_at_s(s_val)
        is_goal = (i == n_tasks - 1)
        marker = "*" if is_goal else "D"
        msize = 16 if is_goal else 9
        ax.plot(*pt, marker=marker, markersize=msize, color=color,
                 markeredgecolor="black", zorder=5)
        label = f"{robot_id} goal" if is_goal else f"T{i + 1}"
        ax.annotate(label, pt + label_offset, fontsize=7, color=color, weight="bold", zorder=6)


def animate_and_plot(robot_specs, arena_size=12.0, save_name="mpc_final_paths.png"):
    """
    robot_specs: list of dicts, one per robot:
        {
          "robot_id": str,
          "control_points_json": path to manual_control_points.json (planned/reference path),
          "csv_log": path to the DE-MPC CSV log (actual/traversed path),
          "map_json": optional path to map-config JSON (static obstacles); omit or None to skip
        }
    """
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(robot_specs), 1)))

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_xlim(-2, arena_size+2)
    ax.set_ylim(-2, arena_size+2)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title("Live DE-MPC execution: planned vs. traversed paths")

    robots_data = []
    static_obstacles_drawn = False

    for spec, color in zip(robot_specs, colors):
        ref_path = ReferencePath(spec["control_points_json"])
        log = read_csv_log(spec["csv_log"])

        if spec.get("map_json") and not static_obstacles_drawn:
            obstacles, _ = load_static_obstacles(spec["map_json"])
            draw_static_obstacles(ax, obstacles)
            static_obstacles_drawn = True  # obstacles assumed shared/identical across robots

        # --- planned reference path: dashed line ---
        ax.plot(ref_path.points[:, 0], ref_path.points[:, 1], "--", color=color,
                 linewidth=1.5, alpha=0.85, zorder=2,
                 label=f"{spec['robot_id']} planned path")

        annotate_task_points(ax, ref_path, color, spec["robot_id"])

        # --- traversed path: solid line, built up frame by frame ---
        (traversed_line,) = ax.plot([], [], "-", color=color, linewidth=1.8, zorder=3,
                                     label=f"{spec['robot_id']} traversed path")
        (current_marker,) = ax.plot([], [], "o", color=color, markersize=11,
                                     markeredgecolor="black", zorder=4)
        (heading_arrow,) = ax.plot([], [], "-", color="black", linewidth=1.5, zorder=4)

        robots_data.append({
            "robot_id": spec["robot_id"], "log": log, "color": color,
            "traversed_line": traversed_line, "current_marker": current_marker,
            "heading_arrow": heading_arrow,
        })

    ax.legend(loc="upper left", fontsize=7, ncol=1)
    plt.tight_layout()
    plt.ion()
    plt.show(block=False)

    max_len = max(len(rd["log"]["x"]) for rd in robots_data)
    arrow_len = 0.35

    for frame in range(0, max_len, FRAME_STRIDE):
        for rd in robots_data:
            log = rd["log"]
            n = min(frame + 1, len(log["x"]))
            if n == 0:
                continue
            rd["traversed_line"].set_data(log["x"][:n], log["y"][:n])
            cx, cy = log["x"][n - 1], log["y"][n - 1]
            theta = log["theta"][n - 1]
            rd["current_marker"].set_data([cx], [cy])
            rd["heading_arrow"].set_data(
                [cx, cx + arrow_len * np.cos(theta)],
                [cy, cy + arrow_len * np.sin(theta)],
            )

        ax.set_title(f"Live DE-MPC execution: planned vs. traversed paths | frame {frame}")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(ANIMATION_PAUSE)

    plt.ioff()
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(PLOT_OUTPUT_DIR, save_name)
    plt.savefig(out_path, dpi=150)
    print(f"Saved final plot to {out_path}")
    plt.show()
    return out_path


if __name__ == "__main__":
    # Example wiring for 3 robots - adjust paths as needed. map_json is
    # optional (omit/None to skip drawing static obstacles).
    robot_specs = [
        {
            "robot_id": "robot_1",
            "control_points_json": "solves/multi/map_002_robot_1_manual_control_points.json",
            "csv_log": "mpc_logs/map2/robot_1_mpc_log.csv",
            "map_json": "maps/map_002_robot_1.json",
        },
        {
            "robot_id": "robot_2",
            "control_points_json": "solves/multi/map_002_robot_2_manual_control_points.json",
            "csv_log": "mpc_logs/map2/robot_2_mpc_log.csv",
            "map_json": "maps/map_002_robot_2.json",
        },
        {
            "robot_id": "robot_3",
            "control_points_json": "solves/multi/map_002_robot_3_manual_control_points.json",
            "csv_log": "mpc_logs/map2/robot_3_mpc_log.csv",
            "map_json": "maps/map_002_robot_3.json",
        },
    ]


    animate_and_plot(robot_specs)
