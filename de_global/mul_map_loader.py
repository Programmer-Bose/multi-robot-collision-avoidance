import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

MAPS_DIR = "maps"
SOLVES_DIR = "solves/multi/"
BSPLINE_DEGREE = 3
N_SAMPLES_PER_SEGMENT = 60

ROBOT_COLORS = ["#FFA500", "#00BFFF", "#32CD32", "#FF1493", "#9932CC", "#FFD700", "#FF4500"]


def _make_clamped_knot_vector(n_ctrl_pts, degree):
    n_internal = n_ctrl_pts - degree - 1
    if n_internal > 0:
        internal_knots = np.linspace(0, 1, n_internal + 2)[1:-1]
    else:
        internal_knots = np.array([])
    return np.concatenate((np.zeros(degree + 1), internal_knots, np.ones(degree + 1)))


def _bspline_curve(control_points, n_samples=N_SAMPLES_PER_SEGMENT, degree=BSPLINE_DEGREE):
    control_points = np.asarray(control_points)
    k = min(degree, len(control_points) - 1)
    knots = _make_clamped_knot_vector(len(control_points), k)
    t = np.linspace(0.0, 1.0, n_samples)
    spline_x = BSpline(knots, control_points[:, 0], k)
    spline_y = BSpline(knots, control_points[:, 1], k)
    return np.column_stack([spline_x(t), spline_y(t)])


def find_solved_path(map_json_path):
    """Locate the DE-optimized control-point JSON for a given robot map file
    (produced by dual_de_bspline_la_map_global.py as
    solves/{map_name}_control_points.json), and rebuild the actual curve."""
    map_name = os.path.splitext(os.path.basename(map_json_path))[0]
    candidate = os.path.join(SOLVES_DIR, f"{map_name}_manual_control_points.json")
    if not os.path.exists(candidate):
        return None

    with open(candidate, "r") as f:
        payload = json.load(f)

    curves = []
    for seg in payload["segments"]:
        start = np.array(seg["start_point"])
        end = np.array(seg["end_point"])
        free_points = np.array(seg["control_points"])
        full_ctrl = np.vstack([start, free_points, end])
        curves.append(_bspline_curve(full_ctrl))

    return np.vstack(curves)


def _scale_pt(pt, scale_factor, orig_h):
    return np.array([pt[0] * scale_factor, (orig_h - pt[1]) * scale_factor])


def _build_obstacles(obstacles, scale_factor, orig_h):
    geoms = []
    for obs in obstacles:
        obs_type = obs["type"]
        cx, cy = _scale_pt(obs["position"], scale_factor, orig_h)

        if obs_type == "circle":
            r_scaled = obs["radius"] * scale_factor
            geoms.append(Point(cx, cy).buffer(r_scaled))

        elif obs_type == "square":
            h = (obs["size"] * scale_factor) / 2.0
            geoms.append(Polygon([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]))

        elif obs_type == "rectangle":
            hw = (obs["width"] * scale_factor) / 2.0
            hh = (obs["height"] * scale_factor) / 2.0
            geoms.append(Polygon([(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)]))

        elif obs_type == "u_shape":
            h = (obs["size"] * scale_factor) / 2.0
            t = obs["thickness"] * scale_factor
            left_arm = Polygon([(cx - h, cy - h), (cx - h + t, cy - h), (cx - h + t, cy + h), (cx - h, cy + h)])
            right_arm = Polygon([(cx + h - t, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx + h - t, cy + h)])
            bottom_bar = Polygon([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy - h + t), (cx - h, cy - h + t)])
            geoms.append(unary_union([left_arm, right_arm, bottom_bar]))

    return geoms


def find_robot_files(map_number):
    """Find all solves/multi/map_{map_number:03d}_robot_*.json files, sorted by robot number."""
    pattern = os.path.join(MAPS_DIR, f"map_{map_number:03d}_robot_*.json")
    files = glob.glob(pattern)

    def robot_idx(path):
        name = os.path.splitext(os.path.basename(path))[0]
        return int(name.split("_robot_")[-1])

    files.sort(key=robot_idx)
    print(files)
    return files


def load_and_plot_map(map_number):
    files = find_robot_files(map_number)
    if not files:
        print(f"Error: no files found matching maps/map_{map_number:03d}_robot_*.json")
        return

    target_scale = 12.0
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#ffffff")

    obstacles_drawn = False

    for file_idx, json_path in enumerate(files):
        with open(json_path, "r") as f:
            data = json.load(f)

        meta_key = "map_metadata" if "map_metadata" in data else "robot_metadata"
        orig_w, orig_h = data[meta_key]["size"]
        scale_factor = target_scale / orig_w
        robot_num = data[meta_key].get("robot_number", file_idx + 1)
        color = ROBOT_COLORS[file_idx % len(ROBOT_COLORS)]

        # Obstacles are shared across robots on the same map; draw once
        if not obstacles_drawn:
            geoms = _build_obstacles(data.get("obstacles", []), scale_factor, orig_h)
            for geom in geoms:
                x, y = geom.exterior.xy
                ax.fill(x, y, color="#000000", edgecolor="#000000", zorder=2)
            obstacles_drawn = True

        start = _scale_pt(data["start_position"], scale_factor, orig_h)
        ax.scatter(start[0], start[1], color=color, marker="s", s=150,
                   edgecolors="black", zorder=4, label=f"Robot {robot_num} Start")

        goal = None
        if data.get("goal_position"):
            goal = _scale_pt(data["goal_position"], scale_factor, orig_h)
            ax.scatter(goal[0], goal[1], color=color, marker="*", s=280,
                       edgecolors="black", zorder=4, label=f"Robot {robot_num} Goal")

        task_points = data["task_points"]
        sequence = data["task_sequence"]

        ordered_path = [start]
        for rank, task_id in enumerate(sequence):
            raw_pos = task_points[str(task_id)]
            task_scaled = _scale_pt(raw_pos, scale_factor, orig_h)
            ordered_path.append(task_scaled)

            ax.scatter(task_scaled[0], task_scaled[1], color=color, marker="o", s=100,
                       edgecolors="black", zorder=3)
            ax.text(task_scaled[0] + 0.2, task_scaled[1] + 0.2, f"R{robot_num}#{rank+1}",
                    color=color, fontsize=9, fontweight="bold", zorder=5)

        if goal is not None:
            ordered_path.append(goal)

        solved_curve = find_solved_path(json_path)
        if solved_curve is not None:
            ax.plot(solved_curve[:, 0], solved_curve[:, 1], "-", color=color,
                   linewidth=2.5, zorder=1, label=f"Robot {robot_num} Path")
        else:
            path_matrix = np.array(ordered_path)
            ax.plot(path_matrix[:, 0], path_matrix[:, 1], ":", color=color, alpha=0.7,
                   linewidth=1.8, zorder=1, label=f"Robot {robot_num} (unsolved - straight)")

    ax.set_xlim(0, target_scale)
    ax.set_ylim(0, target_scale)
    ax.set_xticks(np.arange(0, target_scale + 1, 1.0))
    ax.set_yticks(np.arange(0, target_scale + 1, 1.0))
    ax.grid(True, which="both", color="#cccccc", linestyle="--", linewidth=0.7, zorder=1)
    ax.set_aspect("equal")
    ax.set_title(f"Map {map_number:03d} - {len(files)} Robot(s)", fontsize=14, fontweight="bold", pad=15)

    # De-duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(), loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Give just the map number; it will auto-find every robot file for that map
    load_and_plot_map(1)
