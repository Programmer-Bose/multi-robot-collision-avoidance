"""
Minimal custom PPO, built to train PolicyValueNet on RobotNavGymEnv without
Stable Baselines3. Single-robot / no-neighbor case for now (neighbor_embeds
stays empty) -- the multi-robot rollout loop is a separate future extension,
this is deliberately scoped to get one clean, correct PPO loop working first.

Standard PPO recipe:
    1. Collect a fixed-length rollout with the current policy.
    2. Compute GAE advantages + returns.
    3. Run several epochs of minibatch clipped-surrogate updates.
"""

import numpy as np
import torch
import torch.optim as optim

from networks import PolicyValueNet


def obs_dict_to_tensor(obs, device):
    return {k: torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device).unsqueeze(0)
            for k, v in obs.items()}


def stack_obs_batch(obs_list, device):
    keys = obs_list[0].keys()
    return {k: torch.as_tensor(np.stack([o[k] for o in obs_list]), dtype=torch.float32, device=device)
            for k in keys}


class RolloutBuffer:
    def __init__(self):
        self.obs, self.actions, self.log_probs = [], [], []
        self.rewards, self.values, self.dones = [], [], []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """rewards/values/dones: lists of length T (python floats/bools). Returns (advantages, returns), each length T."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        next_nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


class PPO:
    def __init__(self, env, n_rays=15, grid_size=21, action_low=None, action_high=None,
                 lr=3e-4, gamma=0.99, lam=0.95, clip_eps=0.2, entropy_coef=0.01,
                 value_coef=0.5, n_epochs=10, minibatch_size=64, max_grad_norm=0.5,
                 device=None):
        self.env = env
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = PolicyValueNet(n_rays=n_rays, grid_size=grid_size,
                                   action_low=action_low, action_high=action_high).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm

        self.buffer = RolloutBuffer()
        self._obs, _ = self.env.reset()

    def load_pretrained(self, path, device=None):
        """Load BC-pretrained weights (see imitation_pretrain.py) before fine-tuning.
        Note: BC only supervises the actor's action distribution, so the critic
        head starts from those same loaded weights but hasn't seen value targets --
        expect the first few PPO rollouts to have noisy value loss while the
        critic catches up, even though the actor already behaves sensibly."""
        state_dict = torch.load(path, map_location=device or self.device)
        self.net.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {path}")

    def collect_rollout(self, n_steps):
        self.buffer.clear()
        ep_returns, ep_lens = [], []
        cur_ret, cur_len = 0.0, 0

        for _ in range(n_steps):
            obs_t = obs_dict_to_tensor(self._obs, self.device)
            with torch.no_grad():
                action, log_prob, value = self.net.act(obs_t)
            action_np = action.squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated

            self.buffer.add(self._obs, action_np, log_prob.item(), reward, value.item(), done)

            cur_ret += reward
            cur_len += 1
            self._obs = next_obs

            if done:
                ep_returns.append(cur_ret)
                ep_lens.append(cur_len)
                cur_ret, cur_len = 0.0, 0
                self._obs, _ = self.env.reset()

        with torch.no_grad():
            last_obs_t = obs_dict_to_tensor(self._obs, self.device)
            _, last_value = self.net.forward(last_obs_t)
            last_value = last_value.item()

        return last_value, ep_returns, ep_lens

    def update(self, last_value):
        advantages, returns = compute_gae(
            self.buffer.rewards, self.buffer.values, self.buffer.dones,
            last_value, self.gamma, self.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_batch = stack_obs_batch(self.buffer.obs, self.device)
        actions = torch.as_tensor(np.stack(self.buffer.actions), dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        n = len(self.buffer)
        idx = np.arange(n)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "n_updates": 0}

        for _ in range(self.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.minibatch_size):
                mb_idx = idx[start:start + self.minibatch_size]
                mb_idx_t = torch.as_tensor(mb_idx, dtype=torch.long, device=self.device)

                mb_obs = {k: v[mb_idx_t] for k, v in obs_batch.items()}
                mb_actions = actions[mb_idx_t]
                mb_old_log_probs = old_log_probs[mb_idx_t]
                mb_adv = advantages_t[mb_idx_t]
                mb_returns = returns_t[mb_idx_t]

                log_probs, entropy, values = self.net.evaluate(mb_obs, mb_actions)

                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = ((values - mb_returns) ** 2).mean()
                entropy_bonus = entropy.mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy_bonus.item()
                stats["n_updates"] += 1

        for k in ("policy_loss", "value_loss", "entropy"):
            stats[k] /= max(stats["n_updates"], 1)
        return stats

    def learn(self, total_timesteps, n_steps_per_rollout=2048, log_every=1):
        n_rollouts = total_timesteps // n_steps_per_rollout
        for it in range(1, n_rollouts + 1):
            last_value, ep_returns, ep_lens = self.collect_rollout(n_steps_per_rollout)
            stats = self.update(last_value)
            if it % log_every == 0:
                mean_ret = np.mean(ep_returns) if ep_returns else float("nan")
                mean_len = np.mean(ep_lens) if ep_lens else float("nan")
                print(f"[iter {it}/{n_rollouts}] episodes={len(ep_returns)} "
                      f"mean_return={mean_ret:.2f} mean_len={mean_len:.1f} "
                      f"policy_loss={stats['policy_loss']:.4f} "
                      f"value_loss={stats['value_loss']:.4f} "
                      f"entropy={stats['entropy']:.4f}")