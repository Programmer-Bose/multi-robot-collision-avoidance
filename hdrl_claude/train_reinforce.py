# train_reinforce.py
import os
import glob
import numpy as np
from zmq import device
import torch
import torch.nn.functional as F
from map_utils import generate_random_map, rasterize_map
from map_utils import sample_free_task_points, sample_free_point
from env import GlobalPathEnv, build_default_precedence
from policy import ObstacleAwarePolicy
from datetime import datetime


def sample_episode(policy, env, obstacles, grid, task_points, precedence, start_pos, w):
    state = env.reset(obstacles, grid, task_points, precedence, start_pos, w)
    log_probs, rewards = [], []

    grid_t = torch.tensor(grid).unsqueeze(0).unsqueeze(0).float().to(device)
    w_t = torch.tensor(w).unsqueeze(0).float().to(device)

    while not env.done:
        pos_t = torch.tensor(state["current_pos"] / 12.0).unsqueeze(0).float().to(device)
        tasks_t = torch.tensor(state["task_points"] / 12.0).unsqueeze(0).float().to(device)
        mask_t = torch.tensor(state["available_mask"]).unsqueeze(0).to(device)

        logits, _, _, _ = policy(grid_t, pos_t, tasks_t, mask_t, w_t)  # get logits w/ argmax pass
        probs = F.softmax(logits, dim=-1)
        dist_task = torch.distributions.Categorical(probs)
        sampled_idx = dist_task.sample()
        log_prob_task = dist_task.log_prob(sampled_idx)

        # recompute cp_mean/std conditioned on the task we actually sampled
        _, cp_mean, cp_std, _ = policy(grid_t, pos_t, tasks_t, mask_t, w_t, task_idx=sampled_idx)
        # print("cp_mean:", cp_mean.squeeze(0).detach().cpu().numpy())
        cp_dist = torch.distributions.Normal(cp_mean, cp_std)
        cp_sample = cp_dist.sample()
        log_prob_cp = cp_dist.log_prob(cp_sample).sum(dim=(1, 2))

        log_prob = log_prob_task + log_prob_cp

        cps_np = cp_sample.squeeze(0).cpu().numpy()
        state, reward, done, info = env.step(sampled_idx.item(), cps_np)

        log_probs.append(log_prob)
        rewards.append(reward)

    return log_probs, rewards


def find_latest_checkpoint(checkpoint_dir="models"):
    """Finds the most recently modified checkpoint file, if any exist."""
    ckpts = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    if not ckpts:
        return None
    return max(ckpts, key=os.path.getmtime)


def train(n_episodes=200, lr=1e-4, gamma=0.99,
          checkpoint_dir="models", resume_from=None, start_episode=0):
  
    os.makedirs(checkpoint_dir, exist_ok=True)

    policy = ObstacleAwarePolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    env = GlobalPathEnv()

    # --- checkpoint resume logic ---
    if resume_from == "latest":
        resume_from = find_latest_checkpoint(checkpoint_dir)
        if resume_from is None:
            print("No checkpoint found, starting fresh.")

    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        policy.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        start_episode = checkpoint.get("episode", start_episode)

    reward_history = []
    MIN_HISTORY_FOR_BASELINE = 20
    for ep in range(start_episode, start_episode + n_episodes):
        obst = np.random.randint(6, 10)  # random number of obstacles
        obstacles = generate_random_map(seed=42, n_obstacles=6)
        grid = rasterize_map(obstacles)

        task_points = sample_free_task_points(obstacles, n_points=np.random.randint(5, 11), margin=0.3)
        start_pos = sample_free_point(obstacles, margin=0.3)
        precedence = build_default_precedence(len(task_points))
        w = np.random.dirichlet([1, 1, 1])  # random preference each episode

        log_probs, rewards = sample_episode(
        policy, env, obstacles, grid, task_points, precedence, start_pos, w)

        total_return = sum(rewards)
        reward_history.append(total_return)

        returns, G = [], 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        if len(reward_history) >= MIN_HISTORY_FOR_BASELINE:
            baseline = np.mean(reward_history[-100:])
            std = np.std(reward_history[-100:]) + 1e-6
            returns = (returns - baseline) / std
        else:
            # early episodes: just scale down raw returns instead of normalizing
            returns = returns / 1000.0

        loss = -torch.stack([lp * R for lp, R in zip(log_probs, returns)]).sum()
        opt.zero_grad()
        loss.backward()
        # print("cp_head grad norm:", policy.cp_head[0].weight.grad.norm().item())
        # print("cp_log_std grad norm:", policy.cp_log_std.grad.norm().item())
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        opt.step()

        
        if (ep + 1) % 100 == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_path = os.path.join(checkpoint_dir, f"global_policy_ep{ep+1}.pt")
            torch.save({
                "episode": ep + 1,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        print(f"episode {ep+1}/{start_episode + n_episodes} "
              f"total_reward={sum(rewards):.3f} loss={loss.item():.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = os.path.join(checkpoint_dir, f"global_policy_final_{timestamp}.pt")
    torch.save({
        "episode": start_episode + n_episodes,
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
    }, final_path)
    print(f"Saved final checkpoint: {final_path}")
    return policy


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # --- fresh training run ---
    trained_policy = train(n_episodes=200, lr=1e-4, gamma=0.99)

    # --- resume from the most recent checkpoint automatically ---
    # trained_policy = train(n_episodes=1000, lr=1e-4, gamma=0.99,
    #                         checkpoint_dir="models", resume_from="latest")

    # --- OR resume from a specific checkpoint file ---
    # trained_policy = train(n_episodes=1000, lr=1e-4, gamma=0.99,
    #                         checkpoint_dir="models",
    #                         resume_from="models/global_policy_ep500.pt")