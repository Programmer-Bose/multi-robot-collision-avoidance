"""
DDPG Path-Following Inference (static plot only)
=================================================

Loads a trained DDPG checkpoint (produced by ddpg_path_following.py,
including the multi-curve-curriculum checkpoint ddpg_path_following_
multi_curve.pt) and runs the policy against a reference path. The
reference path can be either:

  - a manual per-segment control-point JSON file (real robot/map), or
  - a freshly generated random curve of one of the 6 training families
    (sine_wave, cosine_s_curve, rounded_rect, zigzag, circular_arc,
    random_bspline) - useful for sanity-checking generalization.

Saves ONE static PNG comparing:
  - Reference path (planned) : dashed blue line
  - Actual path (DRL executed): solid red line

No GIF/animation is produced.

Usage (manual map):
    python inference_ddpg_path_following.py \
        --control_points_json map_001_robot_2_manual_control_points.json \
        --checkpoint solves_drl/ddpg_path_following_multi_curve.pt \
        --max_steps 400

Usage (random curve, e.g. to sanity check generalization):
    python inference_ddpg_path_following.py \
        --curve_type zigzag --seed 42 \
        --checkpoint solves_drl/ddpg_path_following_multi_curve.pt \
        --max_steps 400
"""

import os
import argparse
import datetime

import numpy as np
import matplotlib.pyplot as plt

from ran_ddpg_path_following import (
    ReferencePath,
    PathFollowEnv,
    DDPGAgent,
    MAX_STEPS_PER_EPISODE,
    OUTPUT_DIR,
    DEFAULT_CURVE_TYPES,
    generate_random_path,
)

# ----------------------------------------------------------------------
# Rollout: run the trained policy greedily (no exploration noise)
# ----------------------------------------------------------------------

def run_inference_rollout(agent, ref_path, start_frac=0.0, max_steps=MAX_STEPS_PER_EPISODE):
    env = PathFollowEnv(ref_path)

    idx = int(start_frac * (len(ref_path.points) - 1))
    p_xy, p_heading = ref_path.point_at_index(idx)
    env.pose = np.array([p_xy[0], p_xy[1], p_heading])
    env.prev_action = np.zeros(2)
    env.steps = 0
    env.prev_arclength = ref_path.arclength_at_index(idx)
    state, _ = env._compute_state()

    poses = [env.pose.copy()]
    infos = []
    done = False

    while not done and env.steps < max_steps:
        action = agent.select_action(state, explore=False)
        state, reward, done, info = env.step(action)
        poses.append(env.pose.copy())
        infos.append(info)
        print(f"step={env.steps:4d} pose=({env.pose[0]:.3f}, {env.pose[1]:.3f}, {env.pose[2]:.3f}) "
              f"action=({action[0]:.3f}, {action[1]:.3f}) reward={reward:.3f} done={done}")

    poses = np.array(poses)
    result = {
        "poses": poses,
        "success": bool(infos[-1]["success"]) if infos else False,
        "steps_taken": env.steps,
    }
    return result


# ----------------------------------------------------------------------
# Static comparison plot
# ----------------------------------------------------------------------

def plot_static_comparison(ref_path, rollout, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    poses = rollout["poses"]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(ref_path.points[:, 0], ref_path.points[:, 1], "b--", linewidth=2,
            label="Reference path (planned)", zorder=2)
    ax.plot(poses[:, 0], poses[:, 1], "-", color="crimson", linewidth=2.5,
            label="Actual path (DRL executed)", zorder=3)
    ax.plot(*poses[0, :2], "go", markersize=11, label="Start", zorder=4)
    ax.plot(*poses[-1, :2], "r^", markersize=11, label="End (DRL)", zorder=4)
    ax.plot(*ref_path.goal_point, "b*", markersize=16, label="Goal (reference)", zorder=4)

    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    status = "SUCCESS" if rollout["success"] else "DID NOT REACH GOAL"
    ax.set_title(f"Reference vs. DRL-executed path [{ref_path.map_name}] - {status}")
    ax.legend(loc="best")
    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(output_dir, f"sim_comparison_{ref_path.map_name}_{timestamp}.png")
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved static comparison plot to {outpath}")
    return outpath


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained DDPG path-following policy and save a static comparison plot")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to a .pt checkpoint saved by ddpg_path_following.py")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--start_frac", type=float, default=0.0,
                         help="Fraction (0-1) along the reference path to start the simulated robot at")
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS_PER_EPISODE,
                         help="Maximum number of simulation steps to run before cutting off the rollout")

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--control_points_json", type=str,
                               help="Manual per-segment B-spline control-point JSON file (real map)")
    source_group.add_argument("--curve_type", type=str, choices=DEFAULT_CURVE_TYPES,
                               help="Generate a fresh random reference path of this family instead of loading a map")

    parser.add_argument("--seed", type=int, default=None,
                         help="Seed for --curve_type random path generation (ignored for --control_points_json)")
    args = parser.parse_args()

    if args.control_points_json is not None:
        ref_path = ReferencePath(control_points_json=args.control_points_json)
    else:
        rng = np.random.default_rng(args.seed)
        points = generate_random_path(args.curve_type, rng)
        ref_path = ReferencePath(points=points, map_name=f"{args.curve_type}_seed{args.seed}")

    agent = DDPGAgent()
    agent.load(args.checkpoint)

    rollout = run_inference_rollout(agent, ref_path, start_frac=args.start_frac, max_steps=args.max_steps)
    print(f"Rollout finished: steps={rollout['steps_taken']}, success={rollout['success']}")

    plot_static_comparison(ref_path, rollout, args.output_dir)

# python ran_inference_ddpg_path_following.py --curve_type sine_wave --seed 42 --checkpoint solves_drl/ddpg_path_following_multi_curve.pt --max_steps 400