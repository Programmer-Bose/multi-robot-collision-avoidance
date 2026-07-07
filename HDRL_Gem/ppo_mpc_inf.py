import matplotlib.pyplot as plt
from bezier_env import UAVKinematicEnv
from train_ppo_mpc import PPOWithBehaviorCloning

def run_inference(model_path="models/ppo_bc_kinematic_final.zip", map_path="maps/env_map_config_1.json"):
    # Initialize environment in human render mode for live playback
    env = UAVKinematicEnv(json_path=map_path, render_mode="human")
    
    # Load the custom Behavioral Cloning PPO model
    try:
        model = PPOWithBehaviorCloning.load(model_path, env=env)
    except FileNotFoundError:
        print(f"Model not found at {model_path}. Please run train_ppo_mpc.py first.")
        return

    obs, info = env.reset()
    done = False
    step_count = 0
    
    print("Running DRL Inference...")
    while not done and step_count < 1500:
        # Predict continuous kinematic action [v, theta]
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        
        if terminated:
            if reward < 0:
                print(f"Inference Terminated: Collision or out-of-bounds at step {step_count}.")
            else:
                print(f"Inference Terminated: All tasks completed successfully in {step_count} steps!")

    print("Keeping window open for inspection. Close the plot window manually to exit.")
    plt.show(block=True)
    env.close()

if __name__ == "__main__":
    run_inference()