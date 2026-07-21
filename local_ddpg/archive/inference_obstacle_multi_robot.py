"""
Inference for the 7-dim Obstacle-Aware Multi-Robot DDPG Policy
================================================================

Loads ONLY the map JSON files (obstacle layout + per-segment reference
paths) and a trained 7-dim checkpoint — no training, no random curves.
Runs all robots of a map together (so they see each other as dynamic
obstacles, same as training), walks each robot through its own segments
in order, then plots the reference paths vs. actual executed paths plus
the obstacles, and saves a PNG.

Usage
-----
    python inference_obstacle_multi_robot.py \
        --map_prefix map_003 \
        --checkpoint solves_drl_obstacle/ddpg_obstacle_multi_robot_80_XXXX.pt \
        --output_dir solves_drl_obstacle
"""

import os
import argparse
import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import local_ddpg.archive.ran_ddpg_path_following as base_module
from local_ddpg.archive.ddpg_obstacle_multi_robot import (
    ObstacleAwareDDPGAgent,
    MultiRobotPathFollowEnv,
    discover_robot_maps,
    load_segment_reference_paths,
    load_map_obstacles,
    MAX_STEPS_PER_EPISODE,
    HARD_COLLISION_RADIUS,
    ROBOT_COLLISION_RADIUS,
    STATE_DIM_7,
)
base_module.GOAL_TOLERANCE = 0.2
base_module.ROBOT_MARKER_SIZE = 0.15


DEFAULT_OUTPUT_DIR = "solves_drl_obstacle"


def set_start_jitter(pos_jitter, heading_jitter):
    """PathFollowEnv.reset() (inherited by every robot env here) adds
    RANDOM, UNSEEDED jitter to the start pose every reset - this is why
    inference results vary run to run, and why a robot can spawn already
    inside an obstacle's hard-collision radius and end instantly (looking
    like it 'didn't move'). Overriding these module-level constants before
    building envs controls/removes that jitter for reproducible inference."""
    base_module.INIT_POS_JITTER = pos_jitter
    base_module.INIT_HEADING_JITTER = heading_jitter


# ----------------------------------------------------------------------
# Load a map: obstacles + per-robot per-segment reference paths
# ----------------------------------------------------------------------

def load_map(map_prefix):
    robots_info = discover_robot_maps(map_prefix)
    if not robots_info:
        raise FileNotFoundError(
            f"No robot map files found for prefix '{map_prefix}' "
            f"(expected <prefix>_robot_N.json + <prefix>_robot_N_manual_control_points.json)")

    per_robot_segments = []
    combined_obstacles = {}
    for r in robots_info:
        segs, _ = load_segment_reference_paths(r["control_points_json"])
        per_robot_segments.append(segs)
        for obs in load_map_obstacles(r["robot_json"]):
            combined_obstacles[tuple(obs["position"])] = obs

    obstacles = list(combined_obstacles.values())
    return per_robot_segments, obstacles, robots_info


# ----------------------------------------------------------------------
# Rollout: run every robot on the map together, noise-free
# ----------------------------------------------------------------------

def run_multi_robot_rollout(agent, per_robot_segments, obstacles, max_steps_per_segment=MAX_STEPS_PER_EPISODE):
    env = MultiRobotPathFollowEnv(per_robot_segments, obstacles)
    n_robots = env.n_robots
    global_max_steps = max_steps_per_segment * max(len(s) for s in per_robot_segments)

    states = env.reset()
    trajectories = [[env.envs[r].pose[:2].copy()] for r in range(n_robots)]
    step_count = 0
    all_done = False

    while not all_done and step_count < global_max_steps:
        actions = agent.select_actions_batch(states, explore=False)
        states, rewards, dones, infos, all_done = env.step(actions)
        for r in range(n_robots):
            trajectories[r].append(env.envs[r].pose[:2].copy())
        step_count += 1

    result = {
        "trajectories": [np.array(t) for t in trajectories],
        "segments_completed": list(env.seg_idx),
        "n_segments": [len(s) for s in per_robot_segments],
        "success": list(env.success_flags),
        "total_steps": step_count,
        "global_max_steps": global_max_steps,
    }
    env.close()
    return result


# ----------------------------------------------------------------------
# Plot + save
# ----------------------------------------------------------------------

def plot_result(per_robot_segments, obstacles, result, map_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    n_robots = len(per_robot_segments)
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(9, 8))

    for obs in obstacles:
        ax.add_patch(Circle(obs["position"], obs["radius"], color="black", alpha=0.35, zorder=1))

    for r in range(n_robots):
        c = colors[r % len(colors)]
        ref_points = np.vstack([sp.points for sp in per_robot_segments[r]])
        ax.plot(ref_points[:, 0], ref_points[:, 1], "--", color=c, linewidth=1.8, alpha=0.6,
                label=f"Robot {r + 1} reference path", zorder=2)

        traj = result["trajectories"][r]
        ax.plot(traj[:, 0], traj[:, 1], "-", color=c, linewidth=2.5,
                label=f"Robot {r + 1} actual path", zorder=3)
        ax.plot(*traj[0], "o", color=c, markersize=10, zorder=5)
        ax.plot(*traj[-1], "^", color=c, markersize=11, zorder=5)

        n_reached = result["segments_completed"][r]
        n_total = result["n_segments"][r]
        task_points = [per_robot_segments[r][0].points[0]] + [sp.goal_point for sp in per_robot_segments[r]]
        for i, tp in enumerate(task_points):
            if i == 0:
                continue
            marker_color = "limegreen" if i <= n_reached else "gray"
            ax.plot(*tp, "D", color=marker_color, markersize=8, zorder=6)

    status_lines = []
    for r in range(n_robots):
        status = "SUCCESS" if result["success"][r] else \
            f"{result['segments_completed'][r]}/{result['n_segments'][r]} segments"
        status_lines.append(f"Robot {r + 1}: {status}")

    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(f"Obstacle-aware multi-robot rollout [{map_name}]\n" + " | ".join(status_lines))
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(output_dir, f"inference_obstacle_{map_name}_{timestamp}.png")
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved rollout plot to {outpath}")
    return outpath


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for the 7-dim obstacle-aware multi-robot DDPG policy")
    parser.add_argument("--map_prefix", type=str, required=True,
                         help="Map prefix, e.g. map_003 or /path/to/map_003 "
                              "(expects <prefix>_robot_N.json + <prefix>_robot_N_manual_control_points.json)")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to a 7-dim checkpoint saved by ddpg_obstacle_multi_robot.py")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_steps_per_segment", type=int, default=MAX_STEPS_PER_EPISODE)
    parser.add_argument("--seed", type=int, default=0,
                         help="Random seed, for reproducible runs")
    parser.add_argument("--jitter", action="store_true",
                         help="Re-enable the training-time random start-pose jitter "
                              "(off by default for inference, so results are deterministic "
                              "and robots don't spawn already inside an obstacle)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    if args.jitter:
        print("Start-pose jitter ENABLED - results will vary between runs.")
    else:
        set_start_jitter(0.0, 0.0)
        print("Start-pose jitter DISABLED - robots start exactly at each segment's start point.")

    per_robot_segments, obstacles, robots_info = load_map(args.map_prefix)
    map_name = os.path.basename(args.map_prefix)
    print(f"Loaded map '{map_name}': {len(per_robot_segments)} robots, "
          f"{[len(s) for s in per_robot_segments]} segments each, {len(obstacles)} obstacles")

    agent = ObstacleAwareDDPGAgent()
    agent.load(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

    result = run_multi_robot_rollout(agent, per_robot_segments, obstacles,
                                      max_steps_per_segment=args.max_steps_per_segment)

    for r in range(len(per_robot_segments)):
        print(f"Robot {r + 1}: segments completed {result['segments_completed'][r]}/{result['n_segments'][r]} "
              f"| success={result['success'][r]}")
    print(f"Total steps used: {result['total_steps']}/{result['global_max_steps']}")

    plot_result(per_robot_segments, obstacles, result, map_name, args.output_dir)

# python inference_obstacle_multi_robot.py --map_prefix map_003 --checkpoint solves_drl_obstacle/ddpg_obstacle_multi_robot_80_XXXX.pt


# python inference_obstacle_multi_robot.py --map_prefix mandp/map_003 --checkpoint solves_drl_obstacle/ddpg_obstacle_multi_robot_80_XXXX.pt
