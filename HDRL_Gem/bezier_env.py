import os
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.strtree import STRtree
from shapely.ops import unary_union
from scipy.optimize import differential_evolution

class UAVKinematicEnv(gym.Env):
    """
    Step-by-step kinematic environment for GEN AI based UAV swarm control.
    Supports Behavioral Cloning (DEMPC expert actions) and live Matplotlib rendering.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, json_path="env_map_config.json", render_mode=None):
        super(UAVKinematicEnv, self).__init__()
        self.render_mode = render_mode
        
        # Load Map Data
        with open(json_path, 'r') as f:
            self.map_data = json.load(f)
            
        self.grid_res = self.map_data.get("grid_resolution", 64)
        self.target_scale = 12.0 # Fixed map size in meters
        
        # Restore metadata scaling
        self.orig_w, self.orig_h = self.map_data["map_metadata"]["size"]
        self.scale_factor = self.target_scale / self.orig_w
        
        # Restore procedural obstacle parsing
        self.shapely_obstacles = self._parse_obstacles(self.map_data)
        self.obstacle_tree = STRtree(self.shapely_obstacles)
        
        
        
        # Action Space: [velocity (0 to 1), heading (-1 to 1)]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        # Observation Space: Grid + Kinematics (Current Pos, Target Pos)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=1, shape=(1, self.grid_res, self.grid_res), dtype=np.uint8),
            "kinematics": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        })
        
        self.occupancy_grid = self._build_occupancy_grid()
        
        # Rendering Setup
        self.fig = None
        self.ax = None
        self.trajectory_history = []
        
    def _scale_pt(self, pt):
        # Applies the original target scale and Tkinter Y-axis flip
        return np.array([pt[0] * self.scale_factor, (self.orig_h - pt[1]) * self.scale_factor])
        
    def _parse_obstacles(self, data):
        obstacles = []
        for obs in data["obstacles"]:
            cx, cy = self._scale_pt(obs["position"])
            if obs["type"] == "circle":
                r = obs["radius"] * self.scale_factor
                obstacles.append(Point(cx, cy).buffer(r))
            elif obs["type"] == "square":
                h = (obs["size"] * self.scale_factor) / 2.0
                obstacles.append(Polygon([(cx-h, cy-h), (cx+h, cy-h), (cx+h, cy+h), (cx-h, cy+h)]))
            elif obs["type"] == "rectangle":
                hw, hh = (obs["width"] * self.scale_factor) / 2.0, (obs["height"] * self.scale_factor) / 2.0
                obstacles.append(Polygon([(cx-hw, cy-hh), (cx+hw, cy-hh), (cx+hw, cy+hh), (cx-hw, cy+hh)]))
            elif obs["type"] == "u_shape":
                h = (obs["size"] * self.scale_factor) / 2.0
                t = obs["thickness"] * self.scale_factor
                left = Polygon([(cx-h, cy-h), (cx-h+t, cy-h), (cx-h+t, cy+h), (cx-h, cy+h)])
                right = Polygon([(cx+h-t, cy-h), (cx+h, cy-h), (cx+h, cy+h), (cx+h-t, cy+h)])
                bottom = Polygon([(cx-h, cy-h), (cx+h, cy-h), (cx+h, cy-h+t), (cx-h, cy-h+t)])
                obstacles.append(unary_union([left, right, bottom]))
        return obstacles
        
    def _build_occupancy_grid(self):
        grid = np.zeros((1, self.grid_res, self.grid_res), dtype=np.uint8)
        cell_size = self.target_scale / self.grid_res
        for y in range(self.grid_res):
            for x in range(self.grid_res):
                px = x * cell_size
                py = y * cell_size
                cell_poly = Polygon([(px, py), (px+cell_size, py), (px+cell_size, py+cell_size), (px, py+cell_size)])
                hits = self.obstacle_tree.query(cell_poly)
                for idx in hits:
                    if self.shapely_obstacles[idx].intersects(cell_poly):
                        grid[0, y, x] = 1
                        break
        return grid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.start_pos = self._scale_pt(self.map_data["start_position"])
        self.tasks = [self._scale_pt(self.map_data["task_points"][str(tid)]) 
                      for tid in self.map_data["task_sequence"]]
        
        if "goal_position" in self.map_data and self.map_data["goal_position"]:
            self.tasks.append(self._scale_pt(self.map_data["goal_position"]))
            
        self.current_pos = np.copy(self.start_pos)
        self.current_task_idx = 0
        self.trajectory_history = [np.copy(self.current_pos)]
        
        # Initialize Rendering
        if self.render_mode == "human":
            if self.fig is None:
                plt.ion()
                self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.render()
            
        return self._get_obs(), {}

    def _get_obs(self):
        safe_idx = min(self.current_task_idx, len(self.tasks) - 1)
        target_pos = self.tasks[safe_idx]
        
        kin = np.array([
            self.current_pos[0] / self.target_scale,
            self.current_pos[1] / self.target_scale,
            target_pos[0] / self.target_scale,
            target_pos[1] / self.target_scale
        ], dtype=np.float32)
        
        return {
            "grid": self.occupancy_grid,
            "kinematics": kin
        }

    # def _run_dempc_solver(self, current_pos, target_pos):
    #     """
    #     Differential-Evolution Model Predictive Control (DE-MPC) planner[cite: 2].
    #     Searches over a horizon-H sequence of controls to find a path that drives 
    #     the robot toward the goal while avoiding obstacles and maintaining smoothness[cite: 2].
    #     """
    #     H = 8  # Prediction horizon (kept small for training speed)
    #     max_step = 0.5
        
    #     # Bounds for each step in the horizon: (v_min, v_max), (theta_min, theta_max)
    #     bounds = [(0.0, max_step), (-np.pi, np.pi)] * H
        
    #     # Pre-calculate cell size for fast grid lookups
    #     cell_size = self.target_scale / self.grid_res

    #     def fitness(flat_controls):
    #         controls = flat_controls.reshape((H, 2))
            
    #         # Roll a control sequence through kinematics[cite: 2]
    #         xs = np.empty(H + 1)
    #         ys = np.empty(H + 1)
    #         xs[0], ys[0] = current_pos[0], current_pos[1]
            
    #         for k in range(H):
    #             v, theta = controls[k]
    #             xs[k+1] = xs[k] + v * np.cos(theta)
    #             ys[k+1] = ys[k] + v * np.sin(theta)
                
    #         # Goal tracking: running cost + heavier terminal cost[cite: 2]
    #         dists = np.hypot(xs[1:] - target_pos[0], ys[1:] - target_pos[1])
    #         goal_cost = 3.0 * np.sum(dists) + 8.0 * dists[-1]
            
    #         # Collision penalty using fast occupancy grid lookups
    #         gx = np.clip(xs[1:] / cell_size, 0, self.grid_res - 1).astype(int)
    #         gy = np.clip(ys[1:] / cell_size, 0, self.grid_res - 1).astype(int)
    #         collision_cost = np.sum(self.occupancy_grid[0, gy, gx]) * 250.0 
            
    #         # Strict out-of-bounds penalty
    #         oob = np.sum((xs[1:] < 0) | (xs[1:] > self.target_scale) | 
    #                      (ys[1:] < 0) | (ys[1:] > self.target_scale))
    #         oob_cost = oob * 250.0
            
    #         # Smoothness: penalize large control changes[cite: 2]
    #         dtheta = np.diff(controls[:, 1])
    #         # Wrap angle differences to [-pi, pi]
    #         dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    #         smooth_cost = 0.5 * np.sum(dtheta**2) 
            
    #         return goal_cost + collision_cost + oob_cost + smooth_cost

    #     # Execute Differential Evolution search
    #     result = differential_evolution(
    #         fitness,
    #         bounds,
    #         popsize=15,       # Matches reference planner defaults[cite: 2]
    #         maxiter=20,       # Reduced slightly from 40 for DRL training speed[cite: 2]
    #         mutation=(0.4, 1.0), 
    #         recombination=0.7,
    #         tol=1e-3,
    #         updating="deferred"
    #     )
        
    #     # Only the first control of the optimized sequence is executed[cite: 2]
    #     best_controls = result.x.reshape(H, 2)
    #     v_opt, theta_opt = best_controls[0]
        
    #     return v_opt, theta_opt

    def step(self, action):
        target_pos = self.tasks[self.current_task_idx]
        
        # 1. Denormalize DRL Action
        max_step_size = 0.5 
        v_drl = action[0] * max_step_size
        theta_drl = action[1] * np.pi
        
        new_pos = self.current_pos + np.array([v_drl * np.cos(theta_drl), v_drl * np.sin(theta_drl)])
        
        # 2. Get Expert Action (DEMPC) for Behavioral Cloning
        # v_exp, theta_exp = self._run_dempc_solver(self.current_pos, target_pos)
        # new_pos = self.current_pos + np.array([v_exp * np.cos(theta_exp), v_exp * np.sin(theta_exp)])
        # expert_action = np.array([v_exp / max_step_size, theta_exp / np.pi], dtype=np.float32)
        
        # 3. Evaluate Reward & Collisions
        reward = 0.0
        terminated = False
        truncated = False
        
        cell_size = self.target_scale / self.grid_res
        grid_x = int(np.clip(new_pos[0] / cell_size, 0, self.grid_res - 1))
        grid_y = int(np.clip(new_pos[1] / cell_size, 0, self.grid_res - 1))
        
        # Out of bounds or collision
        if (new_pos[0] < 0 or new_pos[0] > self.target_scale or 
            new_pos[1] < 0 or new_pos[1] > self.target_scale or 
            self.occupancy_grid[0, grid_y, grid_x] == 1):
            
            reward -= 500.0
            terminated = True
        else:
            # Dense progress reward
            old_dist = np.linalg.norm(target_pos - self.current_pos)
            new_dist = np.linalg.norm(target_pos - new_pos)
            
            if new_dist < old_dist:
                reward += 2.0
            else:
                reward -= 5.0
                
            self.current_pos = np.copy(new_pos)
            self.trajectory_history.append(np.copy(self.current_pos))
            
            # Task reached
            if new_dist < 0.25:
                reward += 100.0
                self.current_pos = np.copy(target_pos)
                self.trajectory_history.append(np.copy(self.current_pos))
                self.current_task_idx += 1
                
                if self.current_task_idx >= len(self.tasks):
                    reward += 500.0
                    terminated = True

        if self.render_mode == "human":
            self.render()
            
        info = {}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.fig is None or self.ax is None:
            return
            
        self.ax.clear()
        self.ax.set_xlim(0, self.target_scale)
        self.ax.set_ylim(0, self.target_scale)
        self.ax.set_title("Live DRL Kinematic Navigation")
        
        # Plot Obstacles
        for geom in self.shapely_obstacles:
            if geom.geom_type == 'Polygon':
                x, y = geom.exterior.xy
                self.ax.fill(x, y, color="black", zorder=2)
                
        # Plot Tasks & Goal
        for idx, t_pos in enumerate(self.tasks):
            if idx == len(self.tasks) - 1:
                self.ax.scatter(t_pos[0], t_pos[1], c="red", marker="*", s=200, label="Goal", zorder=4)
            else:
                self.ax.scatter(t_pos[0], t_pos[1], c="cyan", s=100, zorder=4)
                self.ax.text(t_pos[0]+0.2, t_pos[1]+0.2, f"T{idx+1}", fontweight="bold")
                
        # Plot Start
        self.ax.scatter(self.start_pos[0], self.start_pos[1], c="orange", marker="s", s=150, zorder=4, label="Start")
        
        # Plot Trajectory
        if len(self.trajectory_history) > 1:
            traj = np.array(self.trajectory_history)
            self.ax.plot(traj[:, 0], traj[:, 1], c="blue", linewidth=2, zorder=3, label="UAV Path")
            
        # Plot Current Position
        self.ax.scatter(self.current_pos[0], self.current_pos[1], c="lime", s=100, edgecolors="black", zorder=5)
        
        plt.pause(0.01)

    def close(self):
        if self.fig is not None:
            plt.ioff()
            plt.close(self.fig)
#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------
# def test_environment_live():
#     env = UAVKinematicEnv(json_path="maps/env_map_config_019.json", render_mode="human")
#     obs, info = env.reset()
    
#     done = False
#     step_count = 0
    
#     print("Starting Live Simulation Test...")
#     while not done and step_count < 500:
#         safe_idx = min(env.current_task_idx, len(env.tasks) - 1)
#         target_pos = env.tasks[safe_idx]
        
#         # v_exp, theta_exp = env._run_dempc_solver(env.current_pos, target_pos)
#         # action = np.array([v_exp / 0.5, theta_exp / np.pi], dtype=np.float32)
        
#         # Check current grid coordinate before stepping
#         cell_size = env.target_scale / env.grid_res
#         gx = int(np.clip(env.current_pos[0] / cell_size, 0, env.grid_res - 1))
#         gy = int(np.clip(env.current_pos[1] / cell_size, 0, env.grid_res - 1))
        
#         obs, reward, terminated, truncated, info = env.step(action)
#         done = terminated or truncated
#         step_count += 1
        
#         if terminated:
#             if reward < 0:
#                 print(f"\n[CRASH DETECTED on Step {step_count}]")
#                 print(f" -> Current Position: {env.current_pos}")
#                 print(f" -> Grid cell checked: (x={gx}, y={gy})")
#                 print(f" -> Occupancy grid value at cell: {env.occupancy_grid[0, gy, gx]}")
#             else:
#                 print("\n[SUCCESS] All tasks completed successfully!")

#     print(f"Simulation Ended. Total Steps: {step_count}")
    
#     # Keep the window open for inspection instead of closing immediately
#     print("Keeping window open for inspection. Close the plot window manually to exit.")
#     import matplotlib.pyplot as plt
#     plt.show(block=True) 
#     env.close()
# if __name__ == "__main__":
#     test_environment_live()