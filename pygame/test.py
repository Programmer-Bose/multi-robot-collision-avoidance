# import numpy as np
# from robot_nav_gym_env import RobotNavGymEnv
# from ppo import PPO

# env = RobotNavGymEnv(
#     scenario_kwargs=dict(n_static=4, n_dynamic=3, n_tasks=4, omega_max=np.pi),
#     max_episode_steps=400,
# )
# agent = PPO(env, n_rays=15, grid_size=21,
#             action_low=[-0.3, -np.pi/2], action_high=[1.0, np.pi/2])
# agent.learn(total_timesteps=500_000)
##-----------------------------------------------------------------------------------------------------------------------------------
# from robot_nav_gym_env import RobotNavGymEnv
# from networks import PolicyValueNet
# import torch, numpy as np

# env = RobotNavGymEnv(
#     scenario_kwargs=dict(seed=42,n_static=4, n_dynamic=3, n_tasks=4, omega_max=np.pi),
#     render_mode="human",   # <- this is the only thing that changes
# )

# net = PolicyValueNet(n_rays=15, grid_size=21,
#                       action_low=[-0.3, -np.pi/2], action_high=[1.0, np.pi/2])
# # leave net randomly initialized to watch pre-training behavior,
# net.load_state_dict(torch.load("single_robot_policy.pt")) #to watch BC-pretrained behavior

# obs, info = env.reset(seed=1)
# for t in range(1000):
#     obs_t = {k: torch.as_tensor(np.asarray(v)).unsqueeze(0).float() for k, v in obs.items()}
#     action, _, _ = net.act(obs_t, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action.squeeze(0).detach().numpy())
#     if terminated or truncated:
#         print(f"{truncated=}, {terminated=}, {info=}")
#         break
# env.close()
##-------------------------------------------------------------------------------------------------------------------

import argparse
import os
import numpy as np
import torch

from robot_nav_gym_env import RobotNavGymEnv
from ppo import PPO


def load_checkpoint(net, path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    net.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint from {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate a single-robot policy")
    parser.add_argument("--train", action="store_true", help="Train the policy and save it")
    parser.add_argument("--load", action="store_true", help="Load a saved checkpoint and run evaluation")
    parser.add_argument("--model", default="bc_pretrained_mse.pt", help="Checkpoint path")
    parser.add_argument("--total_timesteps", type=int, default=10000)
    parser.add_argument("--n_steps_per_rollout", type=int, default=2048)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true", help="Show the environment window")
    parser.add_argument("--max_episode_steps", type=int, default=400)
    parser.add_argument("--n_static", type=int, default=4)
    parser.add_argument("--n_dynamic", type=int, default=3)
    parser.add_argument("--n_tasks", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = RobotNavGymEnv(
        scenario_kwargs=dict(
            seed=args.seed,
            n_static=args.n_static,
            n_dynamic=args.n_dynamic,
            n_tasks=args.n_tasks,
            omega_max=np.pi,
        ),
        max_episode_steps=args.max_episode_steps,
        render_mode="human" #if args.render else None,
    )

    if args.train:
        agent = PPO(
            env,
            n_rays=15,
            grid_size=21,
            action_low=[-0.3, -np.pi / 2],
            action_high=[1.0, np.pi / 2],
            lr=3e-4,
            gamma=0.99,
            lam=0.95,
            clip_eps=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            n_epochs=10,
            minibatch_size=64,
            max_grad_norm=0.5,
        )

        agent.learn(
            total_timesteps=args.total_timesteps,
            n_steps_per_rollout=args.n_steps_per_rollout,
            log_every=args.log_every,
        )

        os.makedirs(os.path.dirname(args.model) or ".", exist_ok=True)
        torch.save(
            {
                "model_state_dict": agent.net.state_dict(),
                "config": {
                    "n_rays": 15,
                    "grid_size": 21,
                    "action_low": [-0.3, -np.pi / 2],
                    "action_high": [1.0, np.pi / 2],
                },
            },
            args.model,
        )
        print(f"Saved trained model to {args.model}")

    elif args.load:
        from networks import PolicyValueNet
        net = PolicyValueNet(
            n_rays=15,
            grid_size=21,
            action_low=[-0.3, -np.pi / 2],
            action_high=[1.0, np.pi / 2],
        )
        load_checkpoint(net, args.model)

        obs, info = env.reset(seed=1)
        for t in range(1000):
            obs_t = {
                k: torch.as_tensor(np.asarray(v), dtype=torch.float32).unsqueeze(0)
                for k, v in obs.items()
            }
            action, _, _ = net.act(obs_t, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action.squeeze(0).detach().numpy())
            if terminated or truncated:
                print(f"step={t}, truncated={truncated}, terminated={terminated}, info={info}")
                break

    else:
        print("Choose --train or --load")

    env.close()

if __name__ == "__main__":
    main()