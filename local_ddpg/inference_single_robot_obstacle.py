"""
Single-Robot DDPG Inference with Live Animation - SEGMENTED SUBGOALS
=======================================================================

Loads a trained 7-input (path-following + static-obstacle-avoidance)
DDPG checkpoint produced by single_robot_obstacle_ddpg.py, and runs a
noise-free (explore=False) greedy rollout treating EACH SEGMENT of the
control-points JSON as an independent subgoal:

  - The robot's state (x_e, y_e, theta_e, ...) is computed only against
    the CURRENT segment's reference polyline - it has no information
    about later segments until it reaches the current segment's goal.
  - Once the current segment's end point (task point) is reached within
    GOAL_TOLERANCE, the environment's reference path is switched to the
    next segment and the robot continues from wherever it physically is
    (no repositioning / no reset).
  - Collision, leaving the arena, or straying too far off the current
    segment's path ends the whole run early.
  - The run is an overall SUCCESS only if every segment's subgoal is
    reached in sequence.

Shows a LIVE matplotlib animation of the robot moving step-by-step
(the full stitched path is drawn faintly in the background purely for
human reference - it is never given to the robot), and once the
rollout ends saves a static PNG of the run.

Usage
-----
    python inference_single_robot_obstacle.py \
        --control_points_json map_003_robot_2_manual_control_points.json \
        --obstacle_map_json map_003_robot_2.json \
        --checkpoint solves_drl_obstacle/ddpg_single_robot_obstacle_1000_XXXXXXXX.pt \
        --output_dir solves_drl_obstacle
"""

import os
import json
import argparse
import datetime

import numpy as np
import matplotlib.pyplot as plt

from local_ddpg.archive.single_robot_obstacle_ddpg import (
    ReferencePath,
    ObstacleMap,
    PathFollowObstacleEnv,
    DDPGAgent,
    make_robot_triangle,
    bspline_curve,
    N_SAMPLES_PER_SEGMENT,
    ARENA_MIN,
    ARENA_MAX,
    MAX_STEPS_PER_EPISODE,
    GOAL_TOLERANCE,
    OUTPUT_DIR,
    RENDER_PAUSE,
)


# ----------------------------------------------------------------------
# Build one independent ReferencePath per segment (no stitching)
# ----------------------------------------------------------------------

def build_segment_paths(control_points_json, map_name=None):
    """Parses the same per-segment control-point JSON used for training,
    but instead of stitching all segments into one long polyline, returns
    a LIST of independent ReferencePath objects - one per segment - each
    only aware of its own start_point -> end_point geometry."""
    with open(control_points_json, "r") as f:
        data = json.load(f)

    base_name = map_name or os.path.splitext(os.path.basename(control_points_json))[0]
    segment_paths = []

    for seg in data["segments"]:
        start = np.asarray(seg["start_point"], dtype=float)
        end = np.asarray(seg["end_point"], dtype=float)
        free_pts = np.asarray(seg["control_points"], dtype=float)
        full_ctrl = np.vstack([start, free_pts, end])
        curve = bspline_curve(full_ctrl, N_SAMPLES_PER_SEGMENT)

        seg_path = ReferencePath.__new__(ReferencePath)
        seg_path.map_name = f"{base_name}_seg{seg['segment_index']}"
        seg_path.points = curve

        diffs = np.diff(seg_path.points, axis=0)
        seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        seg_path.cum_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        seg_path.total_length = seg_path.cum_length[-1]

        headings = np.zeros(len(seg_path.points))
        headings[0] = np.arctan2(diffs[0, 1], diffs[0, 0])
        headings[-1] = np.arctan2(diffs[-1, 1], diffs[-1, 0])
        for i in range(1, len(seg_path.points) - 1):
            v = seg_path.points[i + 1] - seg_path.points[i - 1]
            headings[i] = np.arctan2(v[1], v[0])
        seg_path.headings = headings

        segment_paths.append(seg_path)

    return segment_paths


# ----------------------------------------------------------------------
# Segmented rollout with live animation
# ----------------------------------------------------------------------

def run_segmented_inference(agent, segment_paths, obstacle_map, output_dir=OUTPUT_DIR,
                             max_steps_per_segment=MAX_STEPS_PER_EPISODE, pause=RENDER_PAUSE,
                             save_png=True, animate=True, run_name="segmented_run"):
    n_segments = len(segment_paths)

    env = PathFollowObstacleEnv(segment_paths[0], obstacle_map, render_mode=None)
    state, _ = env.reset()

    # Override the jittered reset pose with the EXACT segment-0 start point
    # (no position/heading noise at inference time).
    exact_xy, exact_heading = segment_paths[0].point_at_index(0)
    env.pose = np.array([exact_xy[0], exact_xy[1], exact_heading])
    env.prev_action = np.zeros(2)
    env.steps = 0
    env.prev_arclength = segment_paths[0].arclength_at_index(0)
    env._trail_xy = [env.pose[:2].copy()]
    state, _ = env._compute_state()

    fig = ax = trail_line = robot_patch = title_artist = None
    if animate:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))

        full_points = np.vstack([sp.points for sp in segment_paths])
        ax.plot(full_points[:, 0], full_points[:, 1], "b--", linewidth=1.5,
                label="Reference path (display only)", zorder=2)
        for sp in segment_paths:
            ax.plot(*sp.goal_point, "kx", markersize=10, zorder=6)
        for c, r in zip(obstacle_map.centers, obstacle_map.radii):
            ax.add_patch(plt.Circle(c, r, color="gray", alpha=0.4, zorder=1))

        trail_line, = ax.plot([], [], "-", color="crimson", linewidth=2.5,
                               label="Actual trajectory", zorder=3)
        robot_patch = plt.Polygon(make_robot_triangle(*env.pose), closed=True,
                                   color="crimson", zorder=4)
        ax.add_patch(robot_patch)
        ax.plot(*env.pose[:2], "go", markersize=10, zorder=5, label="Start")

        ax.set_xlim(ARENA_MIN[0], ARENA_MAX[0])
        ax.set_ylim(ARENA_MIN[1], ARENA_MAX[1])
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        title_artist = ax.set_title(f"segment 1/{n_segments} | step 0")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(pause)

    trajectory = [env.pose[:2].copy()]
    segment_results = []
    overall_status = "INCOMPLETE"
    total_steps = 0
    base_name = segment_paths[0].map_name.rsplit("_seg", 1)[0]

    for seg_idx in range(n_segments):
        # Switch the robot's perceived reference path to this segment ONLY.
        # No env.reset() here - the robot keeps its current pose/velocity
        # and simply starts being scored against the new subgoal.
        env.path = segment_paths[seg_idx]
        state, info = env._compute_state()
        env.prev_arclength = env.path.arclength_at_index(info["nearest_idx"])

        seg_steps = 0
        seg_done = False
        seg_success = False
        seg_collided = False

        while not seg_done and seg_steps < max_steps_per_segment:
            action = agent.select_action(state, explore=False)
            state, reward, seg_done, info = env.step(action)
            seg_steps += 1
            total_steps += 1
            trajectory.append(env.pose[:2].copy())

            if animate:
                traj_arr = np.array(trajectory)
                trail_line.set_data(traj_arr[:, 0], traj_arr[:, 1])
                robot_patch.set_xy(make_robot_triangle(*env.pose))
                title_artist.set_text(
                    f"[{base_name}] segment {seg_idx + 1}/{n_segments} | "
                    f"step {seg_steps} (total {total_steps}) | reward={reward:.3f}"
                )
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(pause)

            if info.get("success"):
                seg_success = True
            if info.get("collided"):
                seg_collided = True

        segment_results.append({
            "segment_index": seg_idx,
            "success": seg_success,
            "collided": seg_collided,
            "steps": seg_steps,
        })
        seg_status = "SUCCESS" if seg_success else ("COLLISION" if seg_collided else "INCOMPLETE")
        print(f"Segment {seg_idx + 1}/{n_segments}: {seg_status} in {seg_steps} steps")

        if seg_collided:
            overall_status = "COLLISION"
            break
        if not seg_success:
            overall_status = "INCOMPLETE"
            break
        if seg_idx == n_segments - 1:
            overall_status = "SUCCESS"

    trajectory = np.array(trajectory)
    subgoals_reached = sum(r["success"] for r in segment_results)

    outpath = None
    if animate:
        ax.plot(*trajectory[-1], "r^", markersize=12, zorder=5, label="End")
        ax.legend(loc="upper right", fontsize=8)
        title_artist.set_text(
            f"[{base_name}] final - {overall_status} ({total_steps} steps, "
            f"{subgoals_reached}/{n_segments} subgoals reached)"
        )
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(pause)

        if save_png:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            outpath = os.path.join(output_dir, f"inference_segmented_{base_name}_{overall_status}_{timestamp}.png")
            fig.savefig(outpath, dpi=150)
            print(f"Saved final rollout plot to {outpath}")

        plt.ioff()
        plt.show()   # keep the final frame open until the user closes it

    env.close()

    print(f"\nOverall: status={overall_status} | total_steps={total_steps} | "
          f"subgoals reached={subgoals_reached}/{n_segments}")

    return {
        "trajectory": trajectory,
        "status": overall_status,
        "total_steps": total_steps,
        "segment_results": segment_results,
        "png_path": outpath,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segmented (subgoal-by-subgoal) live-animated inference for the trained "
                     "single-robot obstacle-avoidance DDPG policy"
    )
    parser.add_argument("--control_points_json", type=str, required=True,
                         help="Manual per-segment B-spline control-point JSON")
    parser.add_argument("--obstacle_map_json", type=str, required=True,
                         help="Matching obstacle-map JSON")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to a .pt checkpoint saved by single_robot_obstacle_ddpg.py")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--max_steps_per_segment", type=int, default=MAX_STEPS_PER_EPISODE,
                         help="Step budget given to the robot for EACH segment/subgoal")
    parser.add_argument("--pause", type=float, default=0.03,
                         help="Seconds to pause between animation frames (higher = slower/smoother to watch)")
    parser.add_argument("--no_save_png", action="store_true", help="Skip saving the final PNG")
    parser.add_argument("--no_animate", action="store_true", help="Run headless, no live matplotlib window")
    args = parser.parse_args()

    segment_paths = build_segment_paths(args.control_points_json)
    obstacle_map = ObstacleMap(args.obstacle_map_json)
    print(f"Loaded {len(segment_paths)} segment(s) from '{args.control_points_json}' "
          f"and {len(obstacle_map.centers)} static obstacles.")

    agent = DDPGAgent()
    agent.load(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

    run_segmented_inference(
        agent, segment_paths, obstacle_map,
        output_dir=args.output_dir,
        max_steps_per_segment=args.max_steps_per_segment,
        pause=args.pause,
        save_png=not args.no_save_png,
        animate=not args.no_animate,
    )

# python inference_single_robot_obstacle.py --control_points_json mandp/map_003_robot_2_manual_control_points.json --obstacle_map_json mandp/map_003_robot_2.json --checkpoint solves_drl_obstacle/ddpg_single_robot_obstacle_500_.pt