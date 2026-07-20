"""
DDPG Sequential Per-Segment Path-Following Inference
=====================================================

Instead of stitching an entire map's JSON into ONE long reference path and
tracking it as a single trajectory, this script walks the JSON's `segments`
list IN ORDER, treating each segment's end point as its own discrete
sub-goal:

  1. The robot starts at segment 0's start point.
  2. It runs the trained policy against segment 0's local B-spline curve
     until it reaches segment 0's end point (within GOAL_TOLERANCE).
  3. The moment that happens, the ACTIVE reference path switches to
     segment 1 - the robot's pose is NOT reset, it just keeps going from
     wherever it physically is.
  4. This repeats until the final segment's end point (the true goal) is
     reached, or the global step budget runs out.

Per your requirement: each segment gets a nominal budget of 400 steps, but
if the robot can't reach a given segment's end point in that many steps, it
is NOT force-advanced to the next segment - it just keeps trying against
that same segment until the GLOBAL step budget (400 * num_segments by
default) is exhausted.

Saves a single static PNG: full reference path (dashed blue, all segments
concatenated) vs. the actual stitched trajectory (solid crimson), with
markers at each task point showing which ones were actually reached.

Usage:
    python inference_sequential_ddpg.py \
        --control_points_json map_001_robot_2_manual_control_points.json \
        --checkpoint solves_drl/ddpg_path_following_multi_curve.pt \
        --max_steps_per_segment 400
"""

import os
import json
import argparse
import datetime

import numpy as np
import matplotlib.pyplot as plt

from ran_ddpg_path_following import (
    ReferencePath,
    PathFollowEnv,
    DDPGAgent,
    OUTPUT_DIR,
    GOAL_TOLERANCE,
    ARENA_MIN,
    ARENA_MAX,
    DT,
    MAX_LINEAR_VEL,
    MAX_ANGULAR_VEL,
    N_SAMPLES_PER_SEGMENT,
    bspline_curve,
    wrap_to_pi,
)

DEFAULT_MAX_STEPS_PER_SEGMENT = 400


# ----------------------------------------------------------------------
# Load each JSON segment as its own ReferencePath (a discrete sub-goal)
# ----------------------------------------------------------------------

def load_segment_reference_paths(control_points_json):
    with open(control_points_json, "r") as f:
        data = json.load(f)

    map_name = data.get("map_name", os.path.splitext(os.path.basename(control_points_json))[0])

    segment_paths = []
    for i, seg in enumerate(data["segments"]):
        start = np.asarray(seg["start_point"], dtype=float)
        end = np.asarray(seg["end_point"], dtype=float)
        free_pts = np.asarray(seg["control_points"], dtype=float)
        full_ctrl = np.vstack([start, free_pts, end])
        curve = bspline_curve(full_ctrl, N_SAMPLES_PER_SEGMENT)
        segment_paths.append(ReferencePath(points=curve, map_name=f"{map_name}_segment_{i}"))

    return segment_paths, map_name


# ----------------------------------------------------------------------
# Sequential rollout: swap the active sub-goal path on reach, no pose reset
# ----------------------------------------------------------------------

def run_sequential_rollout(agent, segment_paths, max_steps_per_segment=DEFAULT_MAX_STEPS_PER_SEGMENT,
                            global_max_steps=None):
    n_segments = len(segment_paths)
    if global_max_steps is None:
        global_max_steps = max_steps_per_segment * n_segments

    # start exactly at the true path start (segment 0's start point / heading)
    start_xy, start_heading = segment_paths[0].point_at_index(0)
    pose = np.array([start_xy[0], start_xy[1], start_heading])

    # a PathFollowEnv instance is reused purely as a stateless helper for
    # _compute_state() against whichever segment is currently active - we
    # do NOT call its step()/reset(), so its own internal step-count cap
    # never applies; step budget is fully controlled by this loop instead.
    env = PathFollowEnv(segment_paths[0])
    env.pose = pose.copy()
    env.prev_action = np.zeros(2)

    trajectory = [env.pose[:2].copy()]
    seg_idx = 0
    total_steps = 0
    segment_reached_step = []   # global step index at which each segment was completed
    failure_reason = None

    while total_steps < global_max_steps and seg_idx < n_segments:
        env.path = segment_paths[seg_idx]
        state, info = env._compute_state()
        action = agent.select_action(state, explore=False)

        v = float(np.clip(action[0], -MAX_LINEAR_VEL, MAX_LINEAR_VEL))
        w = float(np.clip(action[1], -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL))
        x, y, theta = env.pose
        x += v * np.cos(theta) * DT
        y += v * np.sin(theta) * DT
        theta = wrap_to_pi(theta + w * DT)
        env.pose = np.array([x, y, theta])
        env.prev_action = np.array([v, w])
        total_steps += 1
        trajectory.append(env.pose[:2].copy())

        if x < ARENA_MIN[0] or x > ARENA_MAX[0] or y < ARENA_MIN[1] or y > ARENA_MAX[1]:
            failure_reason = "left_arena"
            break

        seg_goal = segment_paths[seg_idx].goal_point
        dist_to_seg_goal = np.hypot(x - seg_goal[0], y - seg_goal[1])
        nearest_idx = env.path.nearest_index(np.array([x, y]))
        reached_segment_end = (nearest_idx >= len(env.path.points) - 2) and (dist_to_seg_goal < GOAL_TOLERANCE)

        if reached_segment_end:
            segment_reached_step.append(total_steps)
            seg_idx += 1   # advance to next sub-goal; pose carries over unchanged

    success = (seg_idx == n_segments)
    if not success and failure_reason is None:
        failure_reason = "global_step_budget_exhausted"

    return {
        "trajectory": np.array(trajectory),
        "success": success,
        "segments_completed": seg_idx,
        "n_segments": n_segments,
        "total_steps": total_steps,
        "global_max_steps": global_max_steps,
        "segment_reached_step": segment_reached_step,
        "failure_reason": failure_reason,
    }


# ----------------------------------------------------------------------
# Static comparison plot
# ----------------------------------------------------------------------

def plot_sequential_result(segment_paths, rollout, map_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    trajectory = rollout["trajectory"]

    fig, ax = plt.subplots(figsize=(9, 7))

    full_reference = np.vstack([sp.points for sp in segment_paths])
    ax.plot(full_reference[:, 0], full_reference[:, 1], "b--", linewidth=2,
            label="Reference path (all segments)", zorder=2)

    ax.plot(trajectory[:, 0], trajectory[:, 1], "-", color="crimson", linewidth=2.5,
            label="Actual path (DRL executed)", zorder=3)

    # task points: start + each segment's end point
    task_points = [segment_paths[0].points[0]] + [sp.goal_point for sp in segment_paths]
    n_reached = rollout["segments_completed"]
    for i, tp in enumerate(task_points):
        if i == 0:
            ax.plot(*tp, "go", markersize=12, zorder=5, label="Start")
        elif i <= n_reached:
            ax.plot(*tp, "D", color="limegreen", markersize=10, zorder=5,
                    label="Task point reached" if i == 1 else None)
        else:
            ax.plot(*tp, "D", color="gray", markersize=10, zorder=5,
                    label="Task point NOT reached" if i == n_reached + 1 else None)

    ax.plot(*trajectory[-1], "r^", markersize=11, zorder=5, label="Final robot position")

    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    status = "SUCCESS - all segments reached" if rollout["success"] else \
        f"INCOMPLETE - {n_reached}/{rollout['n_segments']} segments reached ({rollout['failure_reason']})"
    ax.set_title(f"Sequential per-segment rollout [{map_name}] - {status}")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(output_dir, f"sequential_sim_{map_name}_{timestamp}.png")
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved sequential rollout plot to {outpath}")
    return outpath


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential per-segment DDPG path-following inference")
    parser.add_argument("--control_points_json", type=str, required=True,
                         help="Manual per-segment B-spline control-point JSON file")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to a .pt checkpoint saved by ddpg_path_following.py")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--max_steps_per_segment", type=int, default=DEFAULT_MAX_STEPS_PER_SEGMENT,
                         help="Nominal step budget per segment; used to compute the default global budget "
                              "(max_steps_per_segment * num_segments). A segment the robot gets stuck on is "
                              "NOT force-advanced - it keeps retrying until the global budget runs out.")
    parser.add_argument("--global_max_steps", type=int, default=None,
                         help="Override the total step budget across all segments "
                              "(default: max_steps_per_segment * num_segments)")
    args = parser.parse_args()

    segment_paths, map_name = load_segment_reference_paths(args.control_points_json)
    print(f"Loaded {len(segment_paths)} segments from {args.control_points_json} (map: {map_name})")

    agent = DDPGAgent()
    agent.load(args.checkpoint)

    rollout = run_sequential_rollout(
        agent, segment_paths,
        max_steps_per_segment=args.max_steps_per_segment,
        global_max_steps=args.global_max_steps,
    )

    print(f"Segments completed: {rollout['segments_completed']}/{rollout['n_segments']}")
    print(f"Total steps used: {rollout['total_steps']}/{rollout['global_max_steps']}")
    print(f"Per-segment completion step indices: {rollout['segment_reached_step']}")
    print(f"Success: {rollout['success']} (failure_reason={rollout['failure_reason']})")

    plot_sequential_result(segment_paths, rollout, map_name, args.output_dir)

# python inference_sequential_ddpg.py --control_points_json solves/multi/map_001_robot_2_manual_control_points.json --checkpoint solves_drl/ddpg_path_following_multi_curve.pt --max_steps_per_segment 400