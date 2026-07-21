"""
Multi-Robot DDPG Batch Inference (shared model, shared obstacle map)
=======================================================================

Runs several robots at once, each following its OWN segmented reference
path (subgoal-by-subgoal, same semantics as inference_single_robot_
obstacle.py) but all sharing the SAME static obstacle map and the SAME
trained checkpoint.

Instead of looping the network once per robot, every timestep all
currently-active robots' 7-dim states are stacked into a single batch
and passed through the actor network ONCE, producing a batch of
[v, w] actions - this is the "batch inferencing" behavior requested.
Each robot is then stepped with its own action in its own environment
instance (each robot keeps its own pose, segment index, step counts).

Per robot:
  - Starts at the EXACT first point of its own path (no init jitter).
  - Only "sees" its current segment's path/goal (no look-ahead into
    later segments), exactly like the single-robot segmented script.
  - Advances to the next segment once its current subgoal is reached.
  - Drops out of the batch on SUCCESS (all segments done), COLLISION,
    or INCOMPLETE (off-path / left arena / per-segment step budget
    exhausted), and is simply skipped in later timesteps.

Usage
-----
    python inference_multi_robot_obstacle.py \
        --control_points_jsons map_003_robot_1_manual_control_points.json \
                               map_003_robot_2_manual_control_points.json \
        --obstacle_map_json map_003_robot_2.json \
        --checkpoint solves_drl_obstacle/ddpg_single_robot_obstacle_1000_XXXXXXXX.pt \
        --output_dir solves_drl_obstacle_multi
"""

import os
import json
import argparse
import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from single_robot_obstacle_ddpg2 import (
    ReferencePath,
    ObstacleMap,
    PathFollowObstacleEnv,
    DDPGAgent,
    make_robot_triangle,
    bspline_curve,
    N_SAMPLES_PER_SEGMENT,
    ARENA_MIN,
    ARENA_MAX,
    MAX_LINEAR_VEL,
    MAX_ANGULAR_VEL,
    MAX_STEPS_PER_EPISODE,
    GOAL_TOLERANCE,
    OUTPUT_DIR,
    RENDER_PAUSE,
    DEVICE,
)


# ----------------------------------------------------------------------
# Build one independent ReferencePath per segment (no stitching) -
# same helper as in inference_single_robot_obstacle.py
# ----------------------------------------------------------------------

def build_segment_paths(control_points_json, map_name=None):
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
# Batched greedy action selection (ONE forward pass for all active robots)
# ----------------------------------------------------------------------

def batch_select_actions(agent, states):
    """states: list/array of shape (n_active, STATE_DIM). Returns actions
    of shape (n_active, ACTION_DIM), noise-free (explore=False)."""
    states_t = torch.as_tensor(np.stack(states), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        actions = agent.actor(states_t).cpu().numpy()
    actions[:, 0] = np.clip(actions[:, 0], -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
    actions[:, 1] = np.clip(actions[:, 1], -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
    return actions


# ----------------------------------------------------------------------
# Per-robot bookkeeping
# ----------------------------------------------------------------------

class RobotRunState:
    def __init__(self, robot_id, segment_paths, obstacle_map, max_steps_per_segment):
        self.robot_id = robot_id
        self.segment_paths = segment_paths
        self.n_segments = len(segment_paths)
        self.max_steps_per_segment = max_steps_per_segment

        self.env = PathFollowObstacleEnv(segment_paths[0], obstacle_map, render_mode=None)
        self.env.reset()

        # Exact start point of segment 0 - no position/heading jitter.
        exact_xy, exact_heading = segment_paths[0].point_at_index(0)
        self.env.pose = np.array([exact_xy[0], exact_xy[1], exact_heading])
        self.env.prev_action = np.zeros(2)
        self.env.steps = 0
        self.env.prev_arclength = segment_paths[0].arclength_at_index(0)
        self.env._trail_xy = [self.env.pose[:2].copy()]

        self.state, _ = self.env._compute_state()
        self.seg_idx = 0
        self.seg_steps = 0
        self.total_steps = 0
        self.trajectory = [self.env.pose[:2].copy()]
        self.segment_results = []
        self.status = "ACTIVE"   # ACTIVE, SUCCESS, COLLISION, INCOMPLETE

    def advance_segment(self):
        self.seg_idx += 1
        self.env.path = self.segment_paths[self.seg_idx]
        self.state, info = self.env._compute_state()
        self.env.prev_arclength = self.env.path.arclength_at_index(info["nearest_idx"])
        self.seg_steps = 0

    def step(self, action):
        self.state, reward, seg_done, info = self.env.step(action)
        self.seg_steps += 1
        self.total_steps += 1
        self.trajectory.append(self.env.pose[:2].copy())

        if info.get("collided"):
            self.status = "COLLISION"
            self.segment_results.append({"segment_index": self.seg_idx, "success": False,
                                          "collided": True, "steps": self.seg_steps})
        elif info.get("success"):
            self.segment_results.append({"segment_index": self.seg_idx, "success": True,
                                          "collided": False, "steps": self.seg_steps})
            if self.seg_idx == self.n_segments - 1:
                self.status = "SUCCESS"
            else:
                self.advance_segment()
        elif seg_done or self.seg_steps >= self.max_steps_per_segment:
            self.status = "INCOMPLETE"
            self.segment_results.append({"segment_index": self.seg_idx, "success": False,
                                          "collided": False, "steps": self.seg_steps})

        return reward, info


# ----------------------------------------------------------------------
# Multi-robot batched rollout with live animation
# ----------------------------------------------------------------------

def run_multi_robot_inference(agent, robots, obstacle_map, output_dir=OUTPUT_DIR,
                               pause=RENDER_PAUSE, save_png=True, animate=True,
                               max_total_steps=MAX_STEPS_PER_EPISODE * 4, run_name="multi_robot_run"):
    colors = [cm.tab10(i % 10) for i in range(len(robots))]

    fig = ax = title_artist = None
    trail_lines = {}
    robot_patches = {}

    if animate:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))

        for c, r in zip(obstacle_map.centers, obstacle_map.radii):
            ax.add_patch(plt.Circle(c, r, color="gray", alpha=0.4, zorder=1))

        for robot, color in zip(robots, colors):
            full_points = np.vstack([sp.points for sp in robot.segment_paths])
            ax.plot(full_points[:, 0], full_points[:, 1], "--", color=color, linewidth=1.2,
                    alpha=0.6, zorder=2, label=f"Robot {robot.robot_id} path (display only)")
            for sp in robot.segment_paths:
                ax.plot(*sp.goal_point, "x", color=color, markersize=8, zorder=6)

            trail_line, = ax.plot([], [], "-", color=color, linewidth=2.2, zorder=3)
            trail_lines[robot.robot_id] = trail_line

            robot_patch = plt.Polygon(make_robot_triangle(*robot.env.pose), closed=True,
                                       color=color, zorder=4)
            ax.add_patch(robot_patch)
            robot_patches[robot.robot_id] = robot_patch

            ax.plot(*robot.env.pose[:2], "o", color=color, markersize=8, zorder=5)

        ax.set_xlim(ARENA_MIN[0], ARENA_MAX[0])
        ax.set_ylim(ARENA_MIN[1], ARENA_MAX[1])
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)
        title_artist = ax.set_title(f"step 0 | {len(robots)} robot(s) active")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(pause)

    global_step = 0
    while global_step < max_total_steps:
        active = [r for r in robots if r.status == "ACTIVE"]
        if not active:
            break

        states = [r.state for r in active]
        actions = batch_select_actions(agent, states)

        for robot, action in zip(active, actions):
            robot.step(action)

        global_step += 1

        if animate:
            for robot in robots:
                traj_arr = np.array(robot.trajectory)
                trail_lines[robot.robot_id].set_data(traj_arr[:, 0], traj_arr[:, 1])
                robot_patches[robot.robot_id].set_xy(make_robot_triangle(*robot.env.pose))

            n_active = sum(1 for r in robots if r.status == "ACTIVE")
            title_artist.set_text(f"step {global_step} | {n_active}/{len(robots)} robot(s) active")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(pause)

    outpath = None
    if animate:
        for robot, color in zip(robots, colors):
            marker = "r^" if robot.status != "SUCCESS" else "g^"
            ax.plot(*robot.trajectory[-1], marker, color=color, markersize=10, zorder=5)
        title_artist.set_text(f"final | step {global_step} | " +
                               ", ".join(f"R{r.robot_id}:{r.status}" for r in robots))
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(pause)

        if save_png:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            outpath = os.path.join(output_dir, f"inference_{run_name}_{timestamp}.png")
            fig.savefig(outpath, dpi=150)
            print(f"Saved final rollout plot to {outpath}")

        plt.ioff()
        plt.show()

    for robot in robots:
        robot.env.close()

    print("\n=== Final results ===")
    for robot in robots:
        subgoals_reached = sum(res["success"] for res in robot.segment_results)
        print(f"Robot {robot.robot_id}: status={robot.status} | total_steps={robot.total_steps} | "
              f"subgoals reached={subgoals_reached}/{robot.n_segments}")

    return {
        "global_steps": global_step,
        "png_path": outpath,
        "robots": [{
            "robot_id": r.robot_id, "status": r.status, "total_steps": r.total_steps,
            "trajectory": np.array(r.trajectory), "segment_results": r.segment_results,
        } for r in robots],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-robot batched inference (shared model, shared static obstacle map, "
                     "independent segmented paths per robot)"
    )
    parser.add_argument("--control_points_jsons", type=str, nargs="+", required=True,
                         help="One or more per-robot manual control-point JSON files")
    parser.add_argument("--obstacle_map_json", type=str, required=True,
                         help="Single shared obstacle-map JSON used by all robots")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to a .pt checkpoint saved by single_robot_obstacle_ddpg.py")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--max_steps_per_segment", type=int, default=MAX_STEPS_PER_EPISODE,
                         help="Step budget given to EACH robot for EACH of its segments/subgoals")
    parser.add_argument("--max_total_steps", type=int, default=MAX_STEPS_PER_EPISODE * 4,
                         help="Global safety cap on simulated timesteps for the whole multi-robot run")
    parser.add_argument("--pause", type=float, default=0.03,
                         help="Seconds to pause between animation frames")
    parser.add_argument("--no_save_png", action="store_true", help="Skip saving the final PNG")
    parser.add_argument("--no_animate", action="store_true", help="Run headless, no live matplotlib window")
    args = parser.parse_args()

    obstacle_map = ObstacleMap(args.obstacle_map_json)

    agent = DDPGAgent()
    agent.load(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

    robots = []
    for i, cp_json in enumerate(args.control_points_jsons):
        segment_paths = build_segment_paths(cp_json)
        robot = RobotRunState(i, segment_paths, obstacle_map, args.max_steps_per_segment)
        robots.append(robot)
        print(f"Robot {i}: loaded {len(segment_paths)} segment(s) from '{cp_json}'")

    print(f"{len(robots)} robot(s), {len(obstacle_map.centers)} shared static obstacles.")

    run_multi_robot_inference(
        agent, robots, obstacle_map,
        output_dir=args.output_dir,
        pause=args.pause,
        save_png=not args.no_save_png,
        animate=not args.no_animate,
        max_total_steps=args.max_total_steps,
    )

# python inference_multi_robot_obstacle.py --control_points_jsons mandp/map_001_robot_1_manual_control_points.json mandp/map_001_robot_2_manual_control_points.json --obstacle_map_json mandp/map_001_robot_2.json --checkpoint solves_drl_obstacle/ddpg_single_robot_obstacle_
