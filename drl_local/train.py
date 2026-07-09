"""
train.py
--------
Main training entry point. Wires together:
    config_utils.py   - hyperparameters / curriculum stage definitions
    env.py             - MultiRobotPathEnv (variable robot count, padded)
    policy_reward.py   - ActorCritic network
    ppo_curriculum.py  - PPOAgent (rollout + update) + CurriculumManager

Usage:
    python train.py
    python train.py --no-render                 # headless (faster) training
    python train.py --iterations 500 --checkpoint-every 20
    python train.py --resume solves/checkpoints/iter_100.pt
"""

import argparse
import glob
import os
import time

import numpy as np
import torch

import config_utils as cu
from env import MultiRobotPathEnv
from ppo_curriculum import PPOAgent, RolloutBuffer, CurriculumManager


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on the multi-robot path-following task.")
    p.add_argument("--iterations", type=int, default=1000,
                    help="Number of rollout+update cycles to run.")
    p.add_argument("--render", dest="render", action="store_true", default=True,
                    help="Open a live pygame window during training (render_mode='human'). Default: on.")
    p.add_argument("--no-render", dest="render", action="store_false",
                    help="Disable rendering for faster headless training.")
    p.add_argument("--checkpoint-dir", type=str, default="solves/checkpoints",
                    help="Directory to save periodic policy checkpoints.")
    p.add_argument("--checkpoint-every", type=int, default=25,
                    help="Save a checkpoint every N iterations.")
    p.add_argument("--resume", type=str, default=None,
                    help="Path to a saved policy .pt file to resume training from.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_path_files_for_stage(stage, map_json_path=None):
    """Resolve a reusable control-point JSON for the current map.

    The repository stores solved paths under solves/multi/ as files named
    like map_002_robot_1_manual_control_points.json. We first look for a
    matching solve file for the active map and, if none is found, fall back
    to the legacy stage-based naming convention.
    """
    if map_json_path is not None:
        stem = os.path.splitext(os.path.basename(map_json_path))[0]
        patterns = [
            f"solves/**/{stem}_manual_control_points.json",
            f"solves/**/{stem}_control_points.json",
            f"solves/**/{stem}*.json",
        ]
        for pattern in patterns:
            matches = sorted(glob.glob(pattern, recursive=True))
            if matches:
                return [matches[0]] * cu.MAX_ROBOTS

    return [f"solves/{stage['name']}_robot{i}_control_points.json"
            for i in range(cu.MAX_ROBOTS)]


def make_env(curriculum, render):
    stage = curriculum.current_stage()
    map_path = curriculum.sample_map_file()
    path_files = build_path_files_for_stage(stage, map_json_path=map_path)

    env = MultiRobotPathEnv(
        map_json_path=map_path,
        path_json_paths=path_files,
        stage_cfg=stage,
        max_robots=cu.MAX_ROBOTS,
        render_mode="human" if render else None,
    )
    return env, stage


def train(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    curriculum = CurriculumManager()
    agent = PPOAgent(device=args.device)
    if args.resume is not None:
        agent.load(args.resume)
        print(f"Resumed policy weights from {args.resume}")

    env, stage = make_env(curriculum, args.render)
    obs, _ = env.reset(seed=args.seed)

    buffer = RolloutBuffer(
        n_steps=cu.PPO_CONFIG["rollout_steps"],
        max_robots=cu.MAX_ROBOTS,
        obs_dim=cu.obs_dim_for(),
        action_dim=cu.ACTION_DIM,
        device=args.device,
    )

    all_returns = []
    t_start = time.time()

    for iteration in range(1, args.iterations + 1):
        obs, stats = agent.collect_rollout(env, buffer, obs)
        losses = agent.update(buffer)

        curriculum.record_episode_results(stats["episode_successes"])
        promoted = curriculum.maybe_promote()

        if stats["episode_returns"]:
            all_returns.extend(stats["episode_returns"])
        mean_return = float(np.mean(stats["episode_returns"])) if stats["episode_returns"] else float("nan")
        elapsed = time.time() - t_start

        status = curriculum.status()
        print(
            f"[iter {iteration}/{args.iterations}] "
            f"stage={status['stage_name']} succ_rate={status['success_rate']:.2f} "
            f"({status['window_filled']}) | "
            f"policy_loss={losses['policy']:.4f} value_loss={losses['value']:.4f} "
            f"entropy={losses['entropy']:.4f} | "
            f"mean_ep_return={mean_return:.2f} | elapsed={elapsed:.1f}s"
        )

        if promoted:
            print(f"  >>> Promoted to stage '{curriculum.current_stage()['name']}' — rebuilding env for new map/robots.")
            env.close()
            env, stage = make_env(curriculum, args.render)
            obs, _ = env.reset(seed=args.seed)

        if iteration % args.checkpoint_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"iter_{iteration}.pt")
            agent.save(ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        if curriculum.is_final_stage() and status["success_rate"] >= stage["success_rate_threshold"] \
                and len(curriculum._history) >= stage["eval_window"]:
            print("Curriculum complete — final stage success threshold reached.")
            agent.save(os.path.join(args.checkpoint_dir, "final_policy.pt"))
            break

    env.close()
    agent.save(os.path.join(args.checkpoint_dir, "last_policy.pt"))
    print("Training finished. Final policy saved.")


if __name__ == "__main__":
    args = parse_args()
    train(args)
