import os
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from scipy.special import comb

class BezierUAVEnv(gym.Env):
    """Custom Environment for UAV Bezier Path Planning."""
    
    def __init__(self, json_path="env_map_config.json"):
        super(BezierUAVEnv, self).__init__()
        
        self.target_scale = 12.0
        self.grid_res = 64
        self.robot_radius = 0.25
        
        # Load and validate map
        self.map_data = self._load_and_validate_map(json_path)
        
        # Observation Space: Dict with 64x64 Binary Grid and Kinematics (4 values: Rx, Ry, Tx, Ty)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=1, shape=(1, self.grid_res, self.grid_res), dtype=np.uint8),
            "kinematics": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        })
        
        # Action Space: 10 continuous values (5 CPs * 2 coordinates), normalized [-1, 1]
        # self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        
        # Generate Global Context
        self.shapely_obstacles = self._parse_obstacles(self.map_data)
        self.occupancy_grid = self._generate_occupancy_grid()
        
        # Kinematic State
        self.start_pos = self._scale_pt(self.map_data["start_position"])
        self.tasks = [self._scale_pt(self.map_data["task_points"][str(tid)]) 
                      for tid in self.map_data["task_sequence"]]
                      
        # NEW: Append Goal to the end of the task sequence
        if "goal_position" in self.map_data and self.map_data["goal_position"]:
            self.goal_pos = self._scale_pt(self.map_data["goal_position"])
            self.tasks.append(self.goal_pos)
        
        self.current_pos = None
        self.current_task_idx = 0
        
    def _load_and_validate_map(self, json_path):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Map file {json_path} not found.")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        required_keys = ["map_metadata", "start_position", "task_points", "task_sequence", "obstacles"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Invalid map format: Missing key '{key}'")
                
        self.orig_w, self.orig_h = data["map_metadata"]["size"]
        self.scale_factor = self.target_scale / self.orig_w
        return data

    def _scale_pt(self, pt):
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

    def _generate_occupancy_grid(self):
        grid = np.zeros((1, self.grid_res, self.grid_res), dtype=np.uint8)
        cell_size = self.target_scale / self.grid_res
        
        for i in range(self.grid_res):
            for j in range(self.grid_res):
                # Calculate center of the grid cell
                gx = (j * cell_size) + (cell_size / 2)
                gy = (i * cell_size) + (cell_size / 2)
                pt = Point(gx, gy)
                
                # Check collision with expanded radius
                for obs in self.shapely_obstacles:
                    if obs.distance(pt) <= self.robot_radius:
                        grid[0, i, j] = 1
                        break
        return grid

    def _normalize_pos(self, pos):
        # Maps [0, 12] to [-1, 1]
        return (pos / (self.target_scale / 2.0)) - 1.0

    def _denormalize_action(self, action):
        # Maps [-1, 1] to [0, 12]
        return (action + 1.0) * (self.target_scale / 2.0)

    def _get_obs(self):
        # Cap the index to prevent IndexError on the terminal step
        safe_idx = min(self.current_task_idx, len(self.tasks) - 1)
        target_pos = self.tasks[safe_idx]
        
        kinematics = np.concatenate([
            self._normalize_pos(self.current_pos), 
            self._normalize_pos(target_pos)
        ])
        
        return {
            "grid": self.occupancy_grid,
            "kinematics": kinematics.astype(np.float32)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_pos = np.copy(self.start_pos)
        self.current_task_idx = 0
        return self._get_obs(), {}

    def _bezier_curve(self, control_points, n_samples=100):
        n = len(control_points) - 1
        t = np.linspace(0.0, 1.0, n_samples)
        curve = np.zeros((n_samples, 2))
        for i, p in enumerate(control_points):
            bernstein = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
            curve += np.outer(bernstein, p)
        return curve

    def step(self, action):
        target_pos = self.tasks[self.current_task_idx]
        
        # 1. Generate baseline straight-line control points (3 intermediate points)
        # np.linspace with num=5 gives [Start, CP1, CP2, CP3, Target]
        baseline_cps = np.linspace(self.current_pos, target_pos, num=5)[1:-1]
        
        # 2. Treat network output as residuals (deviations)
        max_deviation = 4.0 # Maximum allowed offset from the straight line in meters
        residuals = action.reshape(3, 2) * max_deviation
        
        # 3. Add offsets to the baseline straight line
        actual_cps = baseline_cps + residuals
        
        # 4. Reconstruct 5 Control Points (Start + 3 DRL Offsets + Goal)
        control_points = np.vstack([self.current_pos, actual_cps, target_pos])
        
        # 5. Sample Curve
        curve = self._bezier_curve(control_points, n_samples=100)
        
        # 6. Evaluate Reward
        reward = 0.0
        collision = False
        cell_size = self.target_scale / self.grid_res
        
        # NEW: Regularization Penalty (forces straight line if path is clear)
        action_penalty = np.sum(np.abs(action))
        reward -= action_penalty * 2.0
        
        # Path length penalty
        diffs = np.diff(curve, axis=0)
        length = np.sum(np.hypot(diffs[:, 0], diffs[:, 1]))
        reward -= length * 0.1
        
        # Collision penalty
        for pt in curve:
            # NEW: Strict out-of-bounds check
            if pt[0] < 0 or pt[0] > self.target_scale or pt[1] < 0 or pt[1] > self.target_scale:
                collision = True
                break
                
            # Map physical coordinate to grid index
            grid_x = int(np.clip(pt[0] / cell_size, 0, self.grid_res - 1))
            grid_y = int(np.clip(pt[1] / cell_size, 0, self.grid_res - 1))
            
            if self.occupancy_grid[0, grid_y, grid_x] == 1:
                collision = True
                break

        
        
        terminated = False
        truncated = False
        
        if collision:
            reward -= 500.0
            terminated = True # Episode fails on collision
        else:
            reward += 100.0 # Reward for safely reaching the task
            self.current_pos = np.copy(target_pos)
            self.current_task_idx += 1
            
            # Check if all tasks are complete
            if self.current_task_idx >= len(self.tasks):
                reward += 500.0 # Bonus for completing the whole sequence
                terminated = True

        return self._get_obs(), reward, terminated, truncated, {}