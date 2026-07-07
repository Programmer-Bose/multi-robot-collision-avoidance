import os
import glob
import datetime
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from bezier_env import BezierUAVEnv

class GlobalMapFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(GlobalMapFeatureExtractor, self).__init__(observation_space, features_dim)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
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


class EpisodeSaveCallback(BaseCallback):
    def __init__(self, save_freq_episodes, save_path="./models/"):
        super(EpisodeSaveCallback, self).__init__(verbose=1)
        self.save_freq_episodes = save_freq_episodes
        self.save_path = save_path
        self.episode_count = 0
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None and True in dones:
            self.episode_count += 1
            
            if self.episode_count % self.save_freq_episodes == 0:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(self.save_path, f"ppo_bezier_ep{self.episode_count}_{timestamp}")
                self.model.save(filepath)
                if self.verbose > 0:
                    print(f"\n[Callback] Saved model at episode {self.episode_count} to {filepath}")
        return True


def get_latest_model(models_dir="./models/"):
    if not os.path.exists(models_dir):
        return None
    list_of_files = glob.glob(os.path.join(models_dir, "*.zip"))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)


def train(hp):
    print("Initializing Environment...")
    env = BezierUAVEnv(json_path="maps/env_map_config_019.json")
    
    latest_model_path = get_latest_model()
    save_callback = EpisodeSaveCallback(save_freq_episodes=hp["save_freq_episodes"], save_path="./models/")
    
    if latest_model_path and hp["resume_training"]:
        print(f"Resuming training from checkpoint: {latest_model_path}")
        model = PPO.load(
            latest_model_path,
            env=env,
            learning_rate=hp["learning_rate"],
            tensorboard_log="./tensorboard_logs/"
        )
    else:
        print("Initializing New PPO Agent...")
        policy_kwargs = dict(
            features_extractor_class=GlobalMapFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=hp["features_dim"]),
            net_arch=dict(pi=hp["pi_arch"], vf=hp["vf_arch"]) 
        )
        model = PPO(
            "MultiInputPolicy", 
            env, 
            policy_kwargs=policy_kwargs,
            learning_rate=hp["learning_rate"],
            n_steps=hp["n_steps"],
            batch_size=hp["batch_size"],
            gamma=hp["gamma"],
            ent_coef=hp["ent_coef"],
            verbose=1,
            tensorboard_log="./tensorboard_logs/"
        )
    
    print("Starting Training...")
    model.learn(total_timesteps=hp["total_timesteps"], callback=save_callback, reset_num_timesteps=not hp["resume_training"])
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = f"./models/ppo_bezier_final_{timestamp}"
    model.save(final_path)
    print(f"Training Complete. Final model saved to {final_path}")


if __name__ == "__main__":
    hyperparameters = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "total_timesteps": 200000,
        "save_freq_episodes": 50000,
        "features_dim": 256,
        "pi_arch": [256, 256],
        "vf_arch": [256, 256],
        "resume_training": False  # Set to False to start fresh
    }
    
    train(hyperparameters)