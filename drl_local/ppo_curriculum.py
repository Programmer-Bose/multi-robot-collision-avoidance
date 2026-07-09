"""
ppo_curriculum.py
------------------
PPO algorithm (rollout collection, GAE advantage estimation, clipped
surrogate update) + curriculum manager (stage scheduling, promotion
criteria) for the variable robot-count multi-robot path-following task.

Because ActorCritic (policy_reward.py) applies the SAME shared-weight
network independently to every robot slot, the rollout buffer stores
per-robot quantities (obs, action, log_prob, value, reward, active_mask,
done) with shape (T, max_robots, ...) and PPO treats every active
(robot, timestep) pair as an independent sample when flattened for
minibatch updates. Padded/inactive slots are masked out everywhere.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

import config_utils as cu
from policy_reward import ActorCritic


# ============================================================
# 1. Rollout Buffer
# ============================================================

class RolloutBuffer:
    """Fixed-length (T steps) buffer for ONE vectorized env instance,
    storing per-robot quantities for all max_robots slots each step."""

    def __init__(self, n_steps, max_robots, obs_dim, action_dim, device="cpu"):
        self.n_steps = n_steps
        self.max_robots = max_robots
        self.device = device

        shape_obs = (n_steps, max_robots, obs_dim)
        shape_act = (n_steps, max_robots, action_dim)
        shape_flat = (n_steps, max_robots)

        self.obs = np.zeros(shape_obs, dtype=np.float32)
        self.actions = np.zeros(shape_act, dtype=np.float32)
        self.log_probs = np.zeros(shape_flat, dtype=np.float32)
        self.values = np.zeros(shape_flat, dtype=np.float32)
        self.rewards = np.zeros(shape_flat, dtype=np.float32)
        self.active_mask = np.zeros(shape_flat, dtype=np.float32)
        self.dones = np.zeros(shape_flat, dtype=np.float32)

        self.advantages = np.zeros(shape_flat, dtype=np.float32)
        self.returns = np.zeros(shape_flat, dtype=np.float32)

        self.ptr = 0

    def add(self, obs, actions, log_probs, values, rewards, active_mask, dones):
        t = self.ptr
        self.obs[t] = obs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.values[t] = values
        self.rewards[t] = rewards
        self.active_mask[t] = active_mask
        self.dones[t] = dones
        self.ptr += 1

    def full(self):
        return self.ptr >= self.n_steps

    def reset(self):
        self.ptr = 0

    def compute_gae(self, last_values, last_active_mask, gamma, lam):
        """last_values / last_active_mask: (max_robots,) bootstrap value
        and mask for the state AFTER the final stored step."""
        last_gae = np.zeros(self.max_robots, dtype=np.float32)
        next_values = last_values
        next_active = last_active_mask

        for t in reversed(range(self.n_steps)):
            mask_t = self.active_mask[t]
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * not_done - self.values[t]
            last_gae = delta + gamma * lam * not_done * last_gae
            # zero-out advantage contributions for slots inactive at this step
            self.advantages[t] = last_gae * mask_t
            next_values = self.values[t]
            next_active = mask_t

        self.returns = self.advantages + self.values

    def get_flat_batches(self, minibatch_size, rng=None):
        """Flattens (T, max_robots) -> (T*max_robots,) and filters to only
        active samples, then yields shuffled minibatches for PPO epochs."""
        rng = rng if rng is not None else np.random

        flat_active = self.active_mask.reshape(-1) > 0
        idx = np.nonzero(flat_active)[0]
        rng.shuffle(idx)

        obs_flat = self.obs.reshape(-1, self.obs.shape[-1])
        act_flat = self.actions.reshape(-1, self.actions.shape[-1])
        logp_flat = self.log_probs.reshape(-1)
        val_flat = self.values.reshape(-1)
        adv_flat = self.advantages.reshape(-1)
        ret_flat = self.returns.reshape(-1)

        # normalize advantages over the active set only
        adv_active = adv_flat[idx]
        adv_flat = adv_flat.copy()
        adv_flat[idx] = (adv_active - adv_active.mean()) / (adv_active.std() + 1e-8)

        for start in range(0, len(idx), minibatch_size):
            batch_idx = idx[start:start + minibatch_size]
            yield (
                torch.as_tensor(obs_flat[batch_idx]),
                torch.as_tensor(act_flat[batch_idx]),
                torch.as_tensor(logp_flat[batch_idx]),
                torch.as_tensor(val_flat[batch_idx]),
                torch.as_tensor(adv_flat[batch_idx]),
                torch.as_tensor(ret_flat[batch_idx]),
            )


# ============================================================
# 2. PPO Agent
# ============================================================

class PPOAgent:
    def __init__(self, network=None, config=None, device="cpu"):
        self.config = config if config is not None else cu.PPO_CONFIG
        self.device = device
        self.network = (network if network is not None else ActorCritic()).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config["learning_rate"])

    @torch.no_grad()
    def collect_rollout(self, env, buffer, obs, deterministic=False):
        """Runs env for buffer.n_steps starting from `obs` (max_robots, obs_dim),
        filling `buffer` in place. Auto-resets the env on episode end.
        Returns (final_obs, episode_stats) so the caller can continue the
        rollout seamlessly across successive collect_rollout() calls."""
        buffer.reset()
        episode_returns = []
        episode_successes = []
        running_return = 0.0

        for _ in range(buffer.n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            action_t, log_prob_t, value_t = self.network.act(obs_t, deterministic=deterministic)
            action = action_t.cpu().numpy()

            next_obs, scalar_reward, done, truncated, info = env.step(action)
            active_mask = np.zeros(buffer.max_robots, dtype=np.float32)
            active_mask[: env.n_active] = 1.0
            per_robot_reward = info["reward_per_robot"]
            per_robot_done = info["terminated_per_robot"].astype(np.float32)

            buffer.add(
                obs=obs, actions=action,
                log_probs=log_prob_t.cpu().numpy(),
                values=value_t.cpu().numpy(),
                rewards=per_robot_reward,
                active_mask=active_mask,
                dones=per_robot_done,
            )

            running_return += scalar_reward
            obs = next_obs

            if done or truncated:
                episode_returns.append(running_return)
                any_goal = bool(np.any(info["reached_goal"][: env.n_active]))
                episode_successes.append(any_goal)
                running_return = 0.0
                obs, _ = env.reset()

        # bootstrap value for GAE
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, last_values = self.network.forward(obs_t)
        last_active_mask = np.zeros(buffer.max_robots, dtype=np.float32)
        last_active_mask[: env.n_active] = 1.0
        buffer.compute_gae(
            last_values.cpu().numpy(), last_active_mask,
            gamma=self.config["gamma"], lam=self.config["gae_lambda"],
        )

        stats = {"episode_returns": episode_returns, "episode_successes": episode_successes}
        return obs, stats

    def update(self, buffer):
        cfg = self.config
        clip_eps = cfg["clip_epsilon"]
        losses = {"policy": [], "value": [], "entropy": []}

        for _ in range(cfg["n_epochs"]):
            for obs_b, act_b, old_logp_b, old_val_b, adv_b, ret_b in buffer.get_flat_batches(cfg["minibatch_size"]):
                obs_b = obs_b.to(self.device)
                act_b = act_b.to(self.device)
                old_logp_b = old_logp_b.to(self.device)
                adv_b = adv_b.to(self.device)
                ret_b = ret_b.to(self.device)

                new_logp, entropy, value = self.network.evaluate_actions(obs_b, act_b)

                ratio = torch.exp(new_logp - old_logp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(value, ret_b)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + cfg["value_coef"] * value_loss
                    + cfg["entropy_coef"] * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), cfg["max_grad_norm"])
                self.optimizer.step()

                losses["policy"].append(policy_loss.item())
                losses["value"].append(value_loss.item())
                losses["entropy"].append(-entropy_loss.item())

        return {k: float(np.mean(v)) if v else 0.0 for k, v in losses.items()}

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location=self.device))


# ============================================================
# 3. Curriculum Manager
# ============================================================

class CurriculumManager:
    """Tracks a rolling success-rate window per stage and promotes to the
    next stage once the stage's success_rate_threshold is met."""

    def __init__(self, stages=None, rng=None):
        self.stages = stages if stages is not None else cu.CURRICULUM_STAGES
        self.stage_idx = 0
        self.rng = rng if rng is not None else random.Random()
        self._history = deque(maxlen=self.current_stage()["eval_window"])

    def current_stage(self):
        return self.stages[self.stage_idx]

    def is_final_stage(self):
        return self.stage_idx == len(self.stages) - 1

    def sample_map_file(self):
        return self.rng.choice(self.current_stage()["map_files"])

    def record_episode_results(self, successes):
        """successes: bool or iterable of bools (one per finished episode
        since the last call, e.g. from PPOAgent.collect_rollout's stats)."""
        if isinstance(successes, (bool, np.bool_)):
            successes = [successes]
        self._history.extend(successes)

    def success_rate(self):
        if not self._history:
            return 0.0
        return float(np.mean(self._history))

    def maybe_promote(self):
        """Returns True if a promotion just occurred."""
        stage = self.current_stage()
        enough_data = len(self._history) >= stage["eval_window"]
        if enough_data and self.success_rate() >= stage["success_rate_threshold"] and not self.is_final_stage():
            self.stage_idx += 1
            self._history = deque(maxlen=self.current_stage()["eval_window"])
            return True
        return False

    def status(self):
        return {
            "stage_name": self.current_stage()["name"],
            "stage_idx": self.stage_idx,
            "success_rate": self.success_rate(),
            "window_filled": f"{len(self._history)}/{self.current_stage()['eval_window']}",
        }


# ============================================================
# 4. Example usage
# ============================================================
if __name__ == "__main__":
    """
    Minimal end-to-end example: build a curriculum manager + PPO agent,
    run a few rollout/update cycles against the current stage's env, and
    advance the curriculum when the success threshold is met.

    Requires real map + control-point JSON files on disk; paths below are
    placeholders matching the CURRICULUM_STAGES entries in config_utils.py.
    """
    from env import MultiRobotPathEnv

    curriculum = CurriculumManager()
    agent = PPOAgent()

    N_ITERATIONS = 100
    N_STEPS_PER_ROLLOUT = cu.PPO_CONFIG["rollout_steps"]

    for iteration in range(N_ITERATIONS):
        stage = curriculum.current_stage()
        map_path = curriculum.sample_map_file()

        # One control-point path file per possible robot slot (max_robots).
        # In practice these would be pre-generated per map via
        # dual_de_bspline_la_map_global.py or pygame_bspline_editor.py.
        path_files = [f"solves/{stage['name']}_robot{i}_control_points.json"
                      for i in range(cu.MAX_ROBOTS)]

        env = MultiRobotPathEnv(
            map_json_path=map_path,
            path_json_paths=path_files,
            stage_cfg=stage,
            max_robots=cu.MAX_ROBOTS,
            render_mode="human",   # live pygame visualization during training
        )

        obs, _ = env.reset()
        buffer = RolloutBuffer(
            n_steps=N_STEPS_PER_ROLLOUT,
            max_robots=cu.MAX_ROBOTS,
            obs_dim=cu.obs_dim_for(),
            action_dim=cu.ACTION_DIM,
        )

        obs, stats = agent.collect_rollout(env, buffer, obs)
        losses = agent.update(buffer)

        curriculum.record_episode_results(stats["episode_successes"])
        promoted = curriculum.maybe_promote()

        print(f"[iter {iteration}] {curriculum.status()} | "
              f"policy_loss={losses['policy']:.4f} value_loss={losses['value']:.4f} "
              f"mean_ep_return={np.mean(stats['episode_returns']) if stats['episode_returns'] else float('nan'):.2f}")

        if promoted:
            print(f"  >>> Promoted to stage '{curriculum.current_stage()['name']}'")

        env.close()

        if curriculum.is_final_stage() and curriculum.success_rate() >= stage["success_rate_threshold"]:
            print("Curriculum complete.")
            agent.save("solves/final_policy.pt")
            break
