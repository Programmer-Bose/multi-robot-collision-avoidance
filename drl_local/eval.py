"""
eval.py
-------
Evaluation / visualization for a trained PPO policy on the multi-robot
path-following task. Loads a saved policy checkpoint (from train.py) and
either:
  - runs N deterministic episodes headlessly and reports success/collision
    metrics, or
  - runs a single episode with render_mode="human" so you can watch the
    swarm behavior live.

Usage:
    python eval.py --checkpoint solves/checkpoints/final_policy.pt \
                    --map maps/stage3_multi_a.json --n-robots 3 --episodes 20

    python eval.py --checkpoint solves/checkpoints/final_policy.pt \
                    --map maps/stage3_multi_a.json --n-robots 3 --watch
"""

import argparse
import glob
import os

import numpy as np
import torch

import config_utils as cu
from env import MultiRobotPathEnv
from ppo_curriculum import PPOAgent


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained PPO policy.")
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to a saved policy .pt file (from PPOAgent.save()).")
    p.add_argument("--map", type=str, required=True,
                    help="Map JSON path (map_gen.py format).")
    p.add_argument("--path-dir", type=str, default=None,
                    help="Directory containing per-robot control-point JSONs "
                         "named '<mapname>_robot<i>_control_points.json'. "
                         "Defaults to 'solves/'.")
    p.add_argument("--n-robots", type=int, default=None,
                    help="Fixed active robot count for evaluation. If omitted, uses MAX_ROBOTS.")
    p.add_argument("--episodes", type=int, default=20,
                    help="Number of episodes to run in headless metric mode.")
    p.add_argument("--watch", action="store_true",
                    help="Run a single episode with a live pygame window instead of batch metrics.")
    p.add_argument("--deterministic", action="store_true", default=True,
                    help="Use the policy mean action (no sampling). Default: on.")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def resolve_path_files(map_path, path_dir, max_robots):
    map_name = os.path.splitext(os.path.basename(map_path))[0]
    path_dir = path_dir if path_dir is not None else "solves"

    if os.path.isfile(path_dir):
        return [path_dir] * max_robots

    path_files = []
    for i in range(max_robots):
        candidates = sorted(glob.glob(os.path.join(path_dir, f"{map_name}*robot{i}*control_points*.json")))
        if not candidates:
            raise FileNotFoundError(
                f"No control-point JSON found for robot slot {i} under '{path_dir}' "
                f"matching map '{map_name}'. Generate one with "
                f"dual_de_bspline_la_map_global.py or pygame_bspline_editor.py first."
            )
        path_files.append(candidates[0])
    return path_files


def load_agent(checkpoint, device):
    agent = PPOAgent(device=device)
    agent.load(checkpoint)
    agent.network.eval()
    return agent


def run_episode(env, agent, deterministic=True):
    obs, _ = env.reset()
    done = truncated = False
    ep_return = 0.0
    static_collisions = 0
    robot_collisions = 0
    goals_reached = 0
    steps = 0

    while not (done or truncated):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
        with torch.no_grad():
            action, _, _ = agent.network.act(obs_t, deterministic=deterministic)
        obs, reward, done, truncated, info = env.step(action.cpu().numpy())

        ep_return += reward
        steps += 1
        static_collisions += int(np.any(info["static_collision"][: env.n_active]))
        robot_collisions += int(np.any(info["robot_collision"][: env.n_active]))
        goals_reached += int(np.all(info["reached_goal"][: env.n_active]))

        if env.render_mode == "human":
            env.render()

    return {
        "return": ep_return,
        "steps": steps,
        "static_collision": static_collisions > 0,
        "robot_collision": robot_collisions > 0,
        "all_goals_reached": goals_reached > 0,
    }


def evaluate_batch(env, agent, n_episodes, deterministic=True):
    results = [run_episode(env, agent, deterministic) for _ in range(n_episodes)]
    returns = [r["return"] for r in results]
    success_rate = np.mean([r["all_goals_reached"] for r in results])
    static_collision_rate = np.mean([r["static_collision"] for r in results])
    robot_collision_rate = np.mean([r["robot_collision"] for r in results])
    mean_steps = np.mean([r["steps"] for r in results])

    print(f"Episodes: {n_episodes}")
    print(f"  Success rate (all robots reached goal): {success_rate:.2%}")
    print(f"  Static collision rate:                  {static_collision_rate:.2%}")
    print(f"  Robot-robot collision rate:              {robot_collision_rate:.2%}")
    print(f"  Mean episode return:                     {np.mean(returns):.2f} (+/- {np.std(returns):.2f})")
    print(f"  Mean episode length:                     {mean_steps:.1f} steps")
    return results


def main():
    args = parse_args()
    max_robots = cu.MAX_ROBOTS
    path_files = resolve_path_files(args.map, args.path_dir, max_robots)

    env = MultiRobotPathEnv(
        map_json_path=args.map,
        path_json_paths=path_files,
        n_robots=args.n_robots,
        max_robots=max_robots,
        render_mode="human" if args.watch else None,
    )
    env.reset(seed=args.seed)

    agent = load_agent(args.checkpoint, args.device)

    if args.watch:
        result = run_episode(env, agent, deterministic=args.deterministic)
        print("Episode result:", result)
    else:
        evaluate_batch(env, agent, args.episodes, deterministic=args.deterministic)

    env.close()


if __name__ == "__main__":
    main()
