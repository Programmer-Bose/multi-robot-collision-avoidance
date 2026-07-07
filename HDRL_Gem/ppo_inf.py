import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from bezier_env import BezierUAVEnv

def get_latest_model(models_dir="./models/"):
    """Utility to find the most recently saved model in the directory."""
    list_of_files = glob.glob(os.path.join(models_dir, "*.zip"))
    if not list_of_files:
        raise FileNotFoundError(f"No model .zip files found in {models_dir}")
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def run_inference():
    print("Loading Environment...")
    env = BezierUAVEnv(json_path="maps/env_map_config_023.json")
    
    try:
        model_path = get_latest_model()
        print(f"Loading Model: {model_path}")
        model = PPO.load(model_path)
    except FileNotFoundError as e:
        print(e)
        return

    obs, info = env.reset()
    done = False
    
    # Store trajectories for plotting
    executed_curves = []
    executed_control_points = []
    
    print("Running Simulation...")
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        
        start_pos = np.copy(env.current_pos)
        target_pos = np.copy(env.tasks[env.current_task_idx])
        
        # CORRECTED: Use Residual Logic matching the environment
        baseline_cps = np.linspace(start_pos, target_pos, num=5)[1:-1]
        max_deviation = 4.0 
        residuals = action.reshape(3, 2) * max_deviation
        actual_cps = baseline_cps + residuals
        
        # Reconstruct full control point matrix
        cps = np.vstack([start_pos, actual_cps, target_pos])
        curve = env._bezier_curve(cps, n_samples=100)
        
        executed_curves.append(curve)
        executed_control_points.append(cps)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if terminated and reward < 0:
            print("Simulation Terminated: Collision detected.")
        elif terminated and reward > 0:
            print("Simulation Terminated: All tasks completed successfully!")

    # ---------------------------------------------------------
    # Visualization using Matplotlib
    # ---------------------------------------------------------
    print("Generating Plot...")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#ffffff")

    # 1. Plot Obstacles directly from the environment's parsed Shapely geometries
    for geom in env.shapely_obstacles:
        if geom.geom_type == 'Polygon':
            x, y = geom.exterior.xy
            ax.fill(x, y, color="#000000", edgecolor="#000000", zorder=2)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                ax.fill(x, y, color="#000000", edgecolor="#000000", zorder=2)

    # 2. Plot Start, Tasks, and Goal
    ax.scatter(env.start_pos[0], env.start_pos[1], color="#FFA500", marker="s", s=150, edgecolors="black", label="Start", zorder=4)
    
    for idx, t_pos in enumerate(env.tasks):
        # Check if this is the final goal point or a task
        if hasattr(env, 'goal_pos') and np.array_equal(t_pos, env.goal_pos):
            ax.scatter(t_pos[0], t_pos[1], color="#FF0000", marker="*", s=250, edgecolors="black", label="Goal", zorder=4)
        else:
            ax.scatter(t_pos[0], t_pos[1], color="#00FFFF", marker="o", s=100, edgecolors="black", zorder=4)
            ax.text(t_pos[0] + 0.2, t_pos[1] + 0.2, f"T{idx+1}", color="#006666", fontsize=10, fontweight="bold", zorder=5)

    # 3. Plot Executed Trajectories
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(executed_curves)))
    
    for i, (curve, cps) in enumerate(zip(executed_curves, executed_control_points)):
        # Plot continuous curve
        ax.plot(curve[:, 0], curve[:, 1], "-", color=colors[i], linewidth=2.5, label=f"Path Segment {i+1}", zorder=3)
        # Plot intermediate DRL control points (dashed lines)
        ax.plot(cps[:, 0], cps[:, 1], "o--", color="gray", alpha=0.5, markersize=5, zorder=2)

    # 4. Map Configurations
    ax.set_xlim(0, env.target_scale)
    ax.set_ylim(0, env.target_scale)
    ax.set_xticks(np.arange(0, env.target_scale + 1, 1.0))
    ax.set_yticks(np.arange(0, env.target_scale + 1, 1.0))
    
    ax.grid(True, which='both', color='#cccccc', linestyle='--', linewidth=0.7, zorder=1)
    ax.set_aspect('equal')
    ax.set_title("DRL Inference: Evaluated Bezier Trajectory", fontsize=14, fontweight="bold", pad=15)
    
    # Handle duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_inference()