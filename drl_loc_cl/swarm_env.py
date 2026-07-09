"""
swarm_env.py
------------
Gymnasium environment for a single differential-drive robot operating inside
a shared multi-robot swarm (5 robots by default). Designed to be driven by
SB3 PPO with a SHARED policy: each robot is exposed as one "sub-environment"
instance inside a VecEnv (see swarm_vecenv.py), all instances pointing at the
same underlying SwarmWorld so robots can see and avoid each other.

Curriculum stages
------------------
stage 1 ("path_follow"):
    - Only the robot's own kinematics + goal-following reward are active.
    - Static obstacle collisions and inter-robot collisions are NOT
      penalized (robots effectively pass through obstacles/each other),
      so the policy first learns smooth, efficient goal-reaching.
stage 2 ("full"):
    - Static obstacle collision penalty and inter-robot collision penalty
      are both active. Full swarm-aware behavior is required.

Reuses the same map JSON schema / coordinate scaling convention as
dde_mul_la.py and multi_pygame_bspline_editor.py (TARGET_SCALE = 12 world
units, y flipped so +y is up).
"""

import json
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

# ----------------------------------------------------------------------
# Constants (mirrors dde_mul_la.py conventions)
# ----------------------------------------------------------------------

TARGET_SCALE = 12.0
BOUNDS_MIN = np.array([0.0, 0.0])
BOUNDS_MAX = np.array([TARGET_SCALE, TARGET_SCALE])

ROBOT_RADIUS = 0.3
MAX_LINEAR_VEL = 1.5          # world units / s
MAX_ANGULAR_VEL = 2.5         # rad / s
DT = 0.1                      # simulation timestep (s)
MAX_EPISODE_STEPS = 500

GOAL_TOLERANCE = 0.35

N_NEAREST_ROBOTS = 4           # fixed-size slots for nearest other robots
N_NEAREST_OBSTACLES = 6        # fixed-size slots for nearest obstacles
OBS_SENSE_RADIUS = 4.0         # world units; beyond this, "no detection" padding

# --- Reward shaping ---
W_PROGRESS = 5.0
W_TIME_PENALTY = 0.02
W_GOAL_BONUS = 50.0
W_COLLISION_OBSTACLE = 25.0
W_COLLISION_ROBOT = 25.0
W_BOUNDARY = 25.0
W_HEADING_ALIGN = 0.3

PYGAME_SCREEN_SIZE = 800
PIXELS_PER_UNIT = PYGAME_SCREEN_SIZE / TARGET_SCALE


# ----------------------------------------------------------------------
# Shared world: map/obstacle loading, used by every robot's env instance
# ----------------------------------------------------------------------

class SwarmWorld:
    """Holds the shared, immutable static scene (obstacles) plus the live,
    mutable state of every robot (position, heading, goal). One SwarmWorld
    is shared by reference across all N_ROBOTS SwarmEnv instances so each
    robot can observe the others."""

    def __init__(self, map_json_paths):
        """map_json_paths: list of paths, one per robot (same format as
        map_XXX_robot_N.json). Obstacles are taken from the first file
        (they are shared/identical across robots on a given map, per the
        map generator's convention)."""
        self.n_robots = len(map_json_paths)
        self.starts = []
        self.goals = []
        self.task_paths = []  # list of waypoint lists per robot (start->tasks->goal)
        self.obstacles = []

        for i, path in enumerate(map_json_paths):
            with open(path, "r") as f:
                data = json.load(f)
            meta_key = "map_metadata" if "map_metadata" in data else "robot_metadata"
            orig_w, orig_h = data[meta_key]["size"]
            scale = TARGET_SCALE / orig_w

            def scale_pt(pt, orig_h=orig_h, scale=scale):
                return np.array([pt[0] * scale, (orig_h - pt[1]) * scale])

            start = scale_pt(data["start_position"])
            goal = scale_pt(data["goal_position"]) if data.get("goal_position") else start.copy()
            waypoints = [start]
            for task_id in data["task_sequence"]:
                waypoints.append(scale_pt(data["task_points"][str(task_id)]))
            waypoints.append(goal)

            self.starts.append(start)
            self.goals.append(goal)
            self.task_paths.append(waypoints)

            if i == 0:
                self.obstacles = self._build_obstacles(data.get("obstacles", []), scale, orig_h)

        # live state, filled by reset()
        self.positions = np.zeros((self.n_robots, 2))
        self.headings = np.zeros(self.n_robots)
        self.velocities = np.zeros((self.n_robots, 2))  # [linear, angular] last applied
        self.done_flags = np.zeros(self.n_robots, dtype=bool)

    @staticmethod
    def _build_obstacles(obstacles, scale, orig_h):
        geoms = []
        for obs in obstacles:
            obs_type = obs["type"]
            cx, cy = obs["position"][0] * scale, (orig_h - obs["position"][1]) * scale

            if obs_type == "circle":
                r = obs["radius"] * scale
                geoms.append({"type": "circle", "center": (cx, cy), "radius": r})
            elif obs_type in ("square", "rectangle"):
                if obs_type == "square":
                    hw = hh = (obs["size"] * scale) / 2.0
                else:
                    hw = (obs["width"] * scale) / 2.0
                    hh = (obs["height"] * scale) / 2.0
                poly = Polygon([(cx - hw, cy - hh), (cx + hw, cy - hh),
                                 (cx + hw, cy + hh), (cx - hw, cy + hh)])
                geoms.append({"type": "polygon", "shape": poly})
            elif obs_type == "u_shape":
                h = (obs["size"] * scale) / 2.0
                t = obs["thickness"] * scale
                left = Polygon([(cx - h, cy - h), (cx - h + t, cy - h), (cx - h + t, cy + h), (cx - h, cy + h)])
                right = Polygon([(cx + h - t, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx + h - t, cy + h)])
                bottom = Polygon([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy - h + t), (cx - h, cy - h + t)])
                geoms.append({"type": "polygon", "shape": unary_union([left, right, bottom])})
        return geoms

    def reset_robot(self, idx):
        self.positions[idx] = self.starts[idx].copy()
        self.headings[idx] = np.arctan2(
            self.goals[idx][1] - self.starts[idx][1],
            self.goals[idx][0] - self.starts[idx][0],
        )
        self.velocities[idx] = 0.0
        self.done_flags[idx] = False

    def nearest_obstacle_dists(self, pos, k=N_NEAREST_OBSTACLES, sense_radius=OBS_SENSE_RADIUS):
        """Returns array of shape (k, 3): [dx, dy, dist] to the k nearest
        obstacle surface points within sense_radius, zero-padded."""
        results = []
        p = Point(pos)
        for obs in self.obstacles:
            if obs["type"] == "circle":
                cx, cy = obs["center"]
                d = np.hypot(pos[0] - cx, pos[1] - cy) - obs["radius"]
                if d <= sense_radius:
                    ang = np.arctan2(cy - pos[1], cx - pos[0])
                    nearest_x = cx - obs["radius"] * np.cos(ang)
                    nearest_y = cy - obs["radius"] * np.sin(ang)
                    results.append((nearest_x - pos[0], nearest_y - pos[1], max(d, 0.0)))
            else:
                poly = obs["shape"]
                d = poly.distance(p) if not poly.contains(p) else 0.0
                if d <= sense_radius:
                    nearest_pt = poly.exterior.interpolate(poly.exterior.project(p))
                    results.append((nearest_pt.x - pos[0], nearest_pt.y - pos[1], d))
        results.sort(key=lambda r: r[2])
        results = results[:k]
        while len(results) < k:
            results.append((0.0, 0.0, sense_radius))
        return np.array(results, dtype=np.float32)

    def collides_with_obstacle(self, pos, radius=ROBOT_RADIUS):
        p = Point(pos)
        for obs in self.obstacles:
            if obs["type"] == "circle":
                cx, cy = obs["center"]
                if np.hypot(pos[0] - cx, pos[1] - cy) <= obs["radius"] + radius:
                    return True
            else:
                if obs["shape"].distance(p) <= radius:
                    return True
        return False


# ----------------------------------------------------------------------
# Per-robot Gymnasium environment
# ----------------------------------------------------------------------

class SwarmEnv(gym.Env):
    metadata = {"render_modes": ["human", None], "render_fps": 30}

    def __init__(self, world: SwarmWorld, robot_idx: int, stage: int = 1,
                 render_mode: str = None):
        super().__init__()
        self.world = world
        self.robot_idx = robot_idx
        self.stage = stage  # 1 = path_follow only, 2 = full (obstacles + robots)
        self.render_mode = render_mode
        self.step_count = 0
        self.prev_dist_to_goal = None

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            "self_state": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            # [pos_x, pos_y, heading_sin, heading_cos, linear_vel, angular_vel]
            "goal_vector": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            # [dx, dy, dist] to current goal
            "nearby_robots": spaces.Box(low=-np.inf, high=np.inf,
                                         shape=(N_NEAREST_ROBOTS, 4), dtype=np.float32),
            # per slot: [dx, dy, rel_vx, rel_vy]
            "nearby_obstacles": spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(N_NEAREST_OBSTACLES, 3), dtype=np.float32),
            # per slot: [dx, dy, dist]
            "stage": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })

        self._pygame = None
        self._screen = None
        self._clock = None

    def set_stage(self, stage: int):
        self.stage = stage

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset_robot(self.robot_idx)
        self.step_count = 0
        goal = self.world.goals[self.robot_idx]
        pos = self.world.positions[self.robot_idx]
        self.prev_dist_to_goal = float(np.linalg.norm(goal - pos))
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        linear = float(action[0]) * MAX_LINEAR_VEL
        angular = float(action[1]) * MAX_ANGULAR_VEL

        idx = self.robot_idx
        heading = self.world.headings[idx]
        pos = self.world.positions[idx]

        new_heading = heading + angular * DT
        new_pos = pos + np.array([np.cos(new_heading), np.sin(new_heading)]) * linear * DT

        self.world.positions[idx] = new_pos
        self.world.headings[idx] = new_heading
        self.world.velocities[idx] = np.array([linear, angular])

        reward, terminated, info = self._compute_reward(new_pos, linear)
        self.step_count += 1
        truncated = self.step_count >= MAX_EPISODE_STEPS
        if terminated:
            self.world.done_flags[idx] = True

        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _compute_reward(self, pos, linear):
        idx = self.robot_idx
        goal = self.world.goals[idx]
        dist = float(np.linalg.norm(goal - pos))

        reward = W_PROGRESS * (self.prev_dist_to_goal - dist)
        reward -= W_TIME_PENALTY
        self.prev_dist_to_goal = dist

        terminated = False
        info = {}

        # Out of bounds
        if np.any(pos < BOUNDS_MIN) or np.any(pos > BOUNDS_MAX):
            reward -= W_BOUNDARY
            terminated = True
            info["termination_reason"] = "out_of_bounds"

        # Stage 2 only: static obstacle + inter-robot collisions
        if self.stage >= 2:
            if self.world.collides_with_obstacle(pos):
                reward -= W_COLLISION_OBSTACLE
                terminated = True
                info["termination_reason"] = "obstacle_collision"

            for j in range(self.world.n_robots):
                if j == idx:
                    continue
                other = self.world.positions[j]
                if np.linalg.norm(pos - other) <= 2 * ROBOT_RADIUS:
                    reward -= W_COLLISION_ROBOT
                    terminated = True
                    info["termination_reason"] = "robot_collision"
                    break

        # Goal reached
        if dist <= GOAL_TOLERANCE:
            reward += W_GOAL_BONUS
            terminated = True
            info["termination_reason"] = "goal_reached"

        return reward, terminated, info

    # ------------------------------------------------------------------
    def _get_obs(self):
        idx = self.robot_idx
        pos = self.world.positions[idx]
        heading = self.world.headings[idx]
        vel = self.world.velocities[idx]
        goal = self.world.goals[idx]

        self_state = np.array([pos[0], pos[1], np.sin(heading), np.cos(heading),
                                vel[0], vel[1]], dtype=np.float32)

        gvec = goal - pos
        goal_vector = np.array([gvec[0], gvec[1], np.linalg.norm(gvec)], dtype=np.float32)

        others = []
        for j in range(self.world.n_robots):
            if j == idx:
                continue
            rel_pos = self.world.positions[j] - pos
            rel_vel = self.world.velocities[j] - vel
            d = np.linalg.norm(rel_pos)
            others.append((d, rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1]))
        others.sort(key=lambda r: r[0])
        others = others[:N_NEAREST_ROBOTS]
        nearby_robots = np.zeros((N_NEAREST_ROBOTS, 4), dtype=np.float32)
        for i, (_, dx, dy, rvx, rvy) in enumerate(others):
            nearby_robots[i] = [dx, dy, rvx, rvy]

        nearby_obstacles = self.world.nearest_obstacle_dists(pos).astype(np.float32)

        return {
            "self_state": self_state,
            "goal_vector": goal_vector,
            "nearby_robots": nearby_robots,
            "nearby_obstacles": nearby_obstacles,
            "stage": np.array([1.0 if self.stage >= 2 else 0.0], dtype=np.float32),
        }

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return
        if self._pygame is None:
            import pygame
            self._pygame = pygame
            pygame.init()
            self._screen = pygame.display.set_mode((PYGAME_SCREEN_SIZE, PYGAME_SCREEN_SIZE))
            pygame.display.set_caption("Swarm PPO Training - live view")
            self._clock = pygame.time.Clock()

        pygame = self._pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self._pygame = None
                return

        screen = self._screen
        screen.fill((255, 255, 255))

        def w2s(p):
            return int(p[0] * PIXELS_PER_UNIT), int(PYGAME_SCREEN_SIZE - p[1] * PIXELS_PER_UNIT)

        for obs in self.world.obstacles:
            if obs["type"] == "circle":
                pygame.draw.circle(screen, (200, 100, 100), w2s(obs["center"]),
                                    int(obs["radius"] * PIXELS_PER_UNIT))
            else:
                xs, ys = obs["shape"].exterior.xy
                pts = [w2s((x, y)) for x, y in zip(xs, ys)]
                pygame.draw.polygon(screen, (200, 100, 100), pts)

        colors = [(255, 140, 0), (0, 150, 255), (50, 200, 50), (220, 30, 140), (150, 50, 220)]
        for j in range(self.world.n_robots):
            color = colors[j % len(colors)]
            pos = w2s(self.world.positions[j])
            pygame.draw.circle(screen, color, pos, int(ROBOT_RADIUS * PIXELS_PER_UNIT))
            goal_pos = w2s(self.world.goals[j])
            pygame.draw.circle(screen, color, goal_pos, 8, 2)

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None
