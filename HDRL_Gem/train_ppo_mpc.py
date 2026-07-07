import os
import glob
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance
from bezier_env import UAVKinematicEnv

# -------------------------------------------------------------------------
# 1. Custom Rollout Buffer to Store DEMPC Expert Actions
# -------------------------------------------------------------------------
class BCOffsetRolloutBuffer(DictRolloutBuffer):
    def __init__(self, buffer_size, observation_space, action_space, *args, **kwargs):
        super().__init__(buffer_size, observation_space, action_space, *args, **kwargs)
        self.expert_actions = np.zeros((self.buffer_size, self.n_envs, *action_space.shape), dtype=np.float32)

    def reset(self) -> None:
        self.expert_actions = np.zeros((self.buffer_size, self.n_envs, *self.action_space.shape), dtype=np.float32)
        super().reset()

    def add(self, obs, action, reward, episode_start, value, log_prob, **kwargs):
        infos = kwargs.get("infos", [{}])
        for env_idx, info in enumerate(infos):
            if "expert_action" in info:
                self.expert_actions[self.pos, env_idx] = info["expert_action"]
        
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(self, batch_size=None):
        for batch in super().get(batch_size):
            batch_inds = self.generator_indices
            flat_expert = self.expert_actions[batch_inds // self.n_envs, batch_inds % self.n_envs]
            
            # Yield the standard SB3 namedtuple AND our custom expert tensor
            yield batch, torch.as_tensor(flat_expert, device=self.device)

# -------------------------------------------------------------------------
# 2. Custom PPO Class overidding Loss Function with BC Loss
# -------------------------------------------------------------------------
class PPOWithBehaviorCloning(PPO):
    def __init__(self, *args, bc_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.bc_weight = bc_weight  # Multiplier alpha for the BC loss term

    def train(self) -> None:
        """
        Custom training loop that calculates standard PPO loss combined with 
        Behavioral Cloning Mean Squared Error loss against DEMPC vectors.
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        
        clip_range = self.clip_range(self._current_progress_remaining)
        entropy_losses, pg_losses, value_losses, bc_losses, total_losses = [], [], [], [], []

        # Train for n_epochs
        for epoch in range(self.n_epochs):
            # Unpack the two objects yielded by our custom buffer
            for rollout_data, expert_actions in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                log_prob_old = rollout_data.old_log_prob
                advantages = rollout_data.advantages
                returns = rollout_data.returns

                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Evaluate current DRL policy distribution
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                
                # Standard PPO Policy Gradient Loss
                ratio = torch.exp(log_prob - log_prob_old)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value Function Loss
                value_loss = F.mse_loss(returns, values)

                # Entropy Loss
                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                # NEW: Behavioral Cloning Loss Term
                distribution = self.policy.get_distribution(rollout_data.observations)
                predicted_actions = distribution.mode() 
                bc_loss = F.mse_loss(predicted_actions, expert_actions)

                # Combined Objective Optimization
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.bc_weight * bc_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                bc_losses.append(bc_loss.item())
                total_losses.append(loss.item())

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Log metrics to TensorBoard
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/bc_loss", np.mean(bc_losses))
        self.logger.record("train/total_loss", np.mean(total_losses))
        self.logger.record("train/bc_weight_alpha", self.bc_weight)
        self.logger.record("train/explained_variance", explained_var)

# -------------------------------------------------------------------------
# 3. Model Feature Extractor & Control Callbacks
# -------------------------------------------------------------------------
class GlobalMapFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(GlobalMapFeatureExtractor, self).__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            sample_grid = torch.as_tensor(observation_space.spaces["grid"].sample()[None]).float()
            cnn_out_dim = self.cnn(sample_grid).shape[1]
            
        self.linear = nn.Sequential(
            nn.Linear(cnn_out_dim + 4, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        grid = observations["grid"].float()
        cnn_features = self.cnn(grid)
        kinematics = observations["kinematics"]
        fused = torch.cat((cnn_features, kinematics), dim=1)
        return self.linear(fused)


class BCEpisodeDecayCallback(BaseCallback):
    def __init__(self, save_freq_episodes, initial_alpha=10.0, decay_rate=0.98, save_path="./models/"):
        super().__init__(verbose=1)
        self.save_freq_episodes = save_freq_episodes
        self.save_path = save_path
        self.alpha = initial_alpha
        self.decay_rate = decay_rate
        self.episode_count = 0
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None and True in dones:
            self.episode_count += 1
            
            # Smoothly decay the teacher influence every full episode cycle
            self.alpha *= self.decay_rate
            self.model.bc_weight = max(0.0, self.alpha)
            
            if self.episode_count % self.save_freq_episodes == 0:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(self.save_path, f"ppo_bc_kinematic_ep{self.episode_count}_{timestamp}")
                self.model.save(filepath)
        return True


def train(hp):
    print("Initializing Environment...")
    env = UAVKinematicEnv(json_path="maps/env_map_config_000.json", render_mode="human")
    # env = UAVKinematicEnv(json_path="maps/env_map_config_1.json")
    
    policy_kwargs = dict(
        features_extractor_class=GlobalMapFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=hp["features_dim"]),
        net_arch=dict(pi=hp["pi_arch"], vf=hp["vf_arch"]) 
    )
    
    # Instantiate custom architectural pieces
    model = PPO(
        "MultiInputPolicy", 
        env, 
        policy_kwargs=policy_kwargs,
        learning_rate=hp["learning_rate"],
        n_steps=hp["n_steps"],
        batch_size=hp["batch_size"],
        gamma=hp["gamma"],
        verbose=1
    )
    
    # Force the agent to swap the buffer implementation
    # model.rollout_buffer = BCOffsetRolloutBuffer(
    #     hp["n_steps"], 
    #     env.observation_space, 
    #     env.action_space, 
    #     device=model.device, 
    #     n_envs=1
    # )
    
    callback = BCEpisodeDecayCallback(
        save_freq_episodes=hp["save_freq_episodes"],
        initial_alpha=hp["initial_bc_weight"],
        decay_rate=hp["bc_decay_rate"]
    )
    
    print("Starting Training with Blended Algorithm Policy...")
    model.learn(total_timesteps=hp["total_timesteps"], callback=callback)
    model.save("./models/ppo_bc_kinematic_final")

if __name__ == "__main__":
    hyperparameters = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 128,
        "gamma": 0.99,
        "total_timesteps": 10000,
        "save_freq_episodes": 2000,
        "features_dim": 256,
        "pi_arch": [256, 256],
        "vf_arch": [256, 256],
        "initial_bc_weight": 5.0,     # High weight means mimic DEMPC tightly early on
        "bc_decay_rate": 0.98         # Decay factor per finished episode
    }
    train(hyperparameters)