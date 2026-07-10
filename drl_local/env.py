"""
env.py
------
Variable-robot-count environment (1 up to MAX_ROBOTS, no hard ceiling in the
code itself) for curriculum-based multi-robot path following with PPO.

Each active robot must track its own global B-spline reference path (loaded
from a *_control_points.json produced by dual_de_bspline_la_map_global.py or
pygame_bspline_editor.py) while avoiding static obstacles from the map JSON
and other active robots. Inactive robot slots (when n_robots < MAX_ROBOTS)
are padded per config_utils.pad_robot_array and masked out of observations,
rewards, and rendering.

Supports render_mode="human" (live pygame window, used during/after training
for visual sanity-checks) and render_mode="rgb_array" (headless, returns a
frame array - useful for logging videos) as well as render_mode=None
(fastest, no rendering at all - used for bulk PPO rollouts).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import config_utils as cu


class MultiRobotPathEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": cu.RENDER_FPS}

    def __init__(self, map_json_path, path_json_paths, robot_map_json_paths, n_robots=None,
                 stage_cfg=None, max_robots=cu.MAX_ROBOTS, render_mode=None):
       
        super().__init__()

        # self._all_map_data = [cu.load_map(p) for p in map_json_paths[:max_robots]]
        # self.obstacles_per_robot = [m["obstacles"] for m in self._all_map_data]
        self.map_data = cu.load_map(map_json_path)      # obstacles shared, unchanged as before
        self.obstacles = self.map_data["obstacles"]
        assert len(robot_map_json_paths) >= max_robots, (
            "Need at least max_robots robot-map files (one start/goal source per possible robot slot)."
        )

        # NEW: per-robot start/goal, pulled from each robot's own map file
        self._robot_starts = []
        self._robot_goals = []
        for p in robot_map_json_paths[:max_robots]:      # list of per-robot map files, start/goal only
            md = cu.load_map(p)
            self._robot_starts.append(md["start"])
            self._robot_goals.append(md["goal"])

        self.max_robots = max_robots
        self.stage_cfg = stage_cfg
        self.fixed_n_robots = n_robots
        self.enable_robot_collision = (
            stage_cfg["enable_robot_collision"] if stage_cfg is not None else True
        )
        self.max_episode_steps = (
            stage_cfg["max_episode_steps"] if stage_cfg is not None else cu.MAX_EPISODE_STEPS
        )

        # Pre-load every robot's global path polyline + arclength table once.
        assert len(path_json_paths) >= max_robots, (
            "Need at least max_robots path files (one global path per possible robot slot)."
        )
        self._all_polylines = []
        self._all_arclengths = []
        for p in path_json_paths[:max_robots]:
            segments = cu.load_global_path_control_points(p)
            polyline = cu.build_full_path_polyline(segments)
            self._all_polylines.append(polyline)
            self._all_arclengths.append(cu.path_arclength_table(polyline))

        # --- Gym spaces (fixed to max_robots so PPO sees a constant shape) ---
        obs_len = cu.obs_dim_for(max_robots=self.max_robots)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.max_robots, obs_len), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.max_robots, cu.ACTION_DIM), dtype=np.float32
        )

        # --- Rendering state ---
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._pygame_screen = None
        self._pygame_font = None
        self._pygame_clock = None

        # --- Episode state (populated in reset()) ---
        self.n_active = None
        self.positions = None          # (max_robots, 2)
        self.headings = None           # (max_robots,)
        self.velocities = None         # (max_robots, 2)
        self.prev_ang_vel = None       # (max_robots,) for smoothness penalty
        self.active_mask = None        # (max_robots,)
        self.progress_s = None         # (max_robots,) arclength progress along own path
        self.done_flags = None         # (max_robots,) bool, per-robot terminal
        self.step_count = 0

    # ------------------------------------------------------------------
    # Core Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, rng=None):
        super().reset(seed=seed)
        rng = self.np_random if rng is None else rng

        if self.fixed_n_robots is not None:
            self.n_active = self.fixed_n_robots
        elif self.stage_cfg is not None:
            self.n_active = cu.sample_n_robots(self.stage_cfg, rng=rng)
        else:
            self.n_active = self.max_robots

        self.step_count = 0
        self.active_mask = np.zeros(self.max_robots, dtype=np.float32)
        self.active_mask[: self.n_active] = 1.0

        # start = self._all_map_data[i]["start"]
        self.positions = np.tile(cu.PAD_POSITION, (self.max_robots, 1)).astype(np.float32)
        self.headings = np.zeros(self.max_robots, dtype=np.float32)
        self.velocities = np.zeros((self.max_robots, 2), dtype=np.float32)
        self.prev_ang_vel = np.zeros(self.max_robots, dtype=np.float32)
        self.progress_s = np.zeros(self.max_robots, dtype=np.float32)
        self.done_flags = np.zeros(self.max_robots, dtype=bool)

        # Spawn active robots at the same nominal start with a small jitter
        # so they don't perfectly overlap on reset.
        for i in range(self.n_active):
            jitter = rng.uniform(-0.15, 0.15, size=2) if hasattr(rng, "uniform") else np.zeros(2)
            self.positions[i] = self._robot_starts[i] + jitter
            path0 = self._all_polylines[i][0]
            path1 = self._all_polylines[i][min(3, len(self._all_polylines[i]) - 1)]
            self.headings[i] = np.arctan2(*(path1 - path0)[::-1])

        obs = self._build_observation()
        info = {"n_active": self.n_active}
        if self.render_mode == "human":
            self._render_frame()
        return obs, info

    def step(self, action):
        self.step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # --- integrate dynamics for active robots only (unicycle model) ---
        for i in range(self.n_active):
            if self.done_flags[i]:
                continue
            lin_cmd, ang_cmd = action[i]
            v = lin_cmd * cu.MAX_LINEAR_VEL
            w = ang_cmd * cu.MAX_ANGULAR_VEL

            self.headings[i] = cu.wrap_to_pi(self.headings[i] + w * cu.SIM_DT)
            dx = v * np.cos(self.headings[i]) * cu.SIM_DT
            dy = v * np.sin(self.headings[i]) * cu.SIM_DT
            new_pos = cu.clip_to_bounds(self.positions[i] + np.array([dx, dy]))
            self.velocities[i] = (new_pos - self.positions[i]) / cu.SIM_DT
            self.positions[i] = new_pos

            idx, _, s_now = cu.nearest_point_on_polyline(
                self._all_polylines[i],
                self._all_arclengths[i],
                self.positions[i]
            )

            # Store current progress temporarily.
            # Do NOT overwrite self.progress_s yet.
            current_progress = s_now

            # Save for reward calculation
            if not hasattr(self, "_new_progress"):
                self._new_progress = np.zeros(self.max_robots, dtype=np.float32)

            self._new_progress[i] = current_progress

        rewards, terminated, static_hit, robot_hit, reached_goal = self._compute_rewards_and_terms(action)

        self.done_flags = np.logical_or(self.done_flags, terminated)
        truncated = self.step_count >= self.max_episode_steps
        truncated_arr = np.full(self.max_robots, truncated, dtype=bool)

        if truncated:
            rewards -= 20.0 * self.active_mask  # only active robots

        obs = self._build_observation()
        info = {
            "n_active": self.n_active,
            "static_collision": static_hit,
            "robot_collision": robot_hit,
            "reached_goal": reached_goal,
        }

        if self.render_mode == "human":
            self._render_frame()

        # Flatten per-robot reward/terminated to scalars via mask-mean, keeping
        # the option for train.py to also read the raw per-robot arrays in info.
        info["reward_per_robot"] = rewards
        info["terminated_per_robot"] = terminated
        scalar_reward = float(np.sum(rewards * self.active_mask) / max(self.n_active, 1))
        all_done = bool(np.all(self.done_flags[: self.n_active])) or truncated

        return obs, scalar_reward, all_done, truncated, info

    def close(self):
        if self._pygame_screen is not None:
            import pygame
            pygame.quit()
            self._pygame_screen = None

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _build_observation(self):
        obs_rows = []
        for i in range(self.max_robots):
            if self.active_mask[i] == 0:
                obs_rows.append(np.full(cu.obs_dim_for(self.max_robots), cu.PAD_VALUE, dtype=np.float32))
                continue

            pos = self.positions[i]
            vel = self.velocities[i]
            heading = self.headings[i]

            idx, lateral_err, s_now = cu.nearest_point_on_polyline(
                self._all_polylines[i], self._all_arclengths[i], pos
            )
            look_pt = cu.lookahead_point(
                self._all_polylines[i], self._all_arclengths[i], s_now, cu.PATH_LOOKAHEAD_DIST
            )
            look_rel = look_pt - pos
            # goal = self.map_data["goal"] if self.map_data["goal"] is not None else self._all_polylines[i][-1]
            goal = self._robot_goals[i] if self._robot_goals[i] is not None else self._all_polylines[i][-1]
            dist_to_goal = float(np.linalg.norm(goal - pos))
            total_len = self._all_arclengths[i][-1]
            progress_frac = float(s_now / total_len) if total_len > 0 else 0.0

            self_feat = np.array([
                pos[0], pos[1], vel[0], vel[1], heading,
                look_rel[0], look_rel[1], dist_to_goal, progress_frac,
            ], dtype=np.float32)

            # --- neighbor robots (relative), sorted by distance, padded ---
            neighbor_rows = []
            for j in range(self.max_robots):
                if j == i or self.active_mask[j] == 0:
                    continue
                rel = self.positions[j] - pos
                d = float(np.linalg.norm(rel))
                neighbor_rows.append((d, rel[0], rel[1], self.velocities[j][0], self.velocities[j][1]))
            neighbor_rows.sort(key=lambda r: r[0])
            n_slots = self.max_robots - 1
            neigh_arr = np.full((n_slots, cu.NEIGHBOR_OBS_DIM), cu.PAD_VALUE, dtype=np.float32)
            for k, (d, rdx, rdy, rvx, rvy) in enumerate(neighbor_rows[:n_slots]):
                neigh_arr[k] = [rdx, rdy, rvx, rvy, d, 1.0]

            # --- nearest static obstacles (relative), padded ---
            raw = cu.distance_to_nearest_static_obstacles(pos, self.obstacles, k=cu.K_NEAREST_OBSTACLES)
            obst_arr = np.full((cu.K_NEAREST_OBSTACLES, cu.OBSTACLE_OBS_DIM), cu.PAD_VALUE, dtype=np.float32)
            for k in range(raw.shape[0]):
                dx, dy, d = raw[k]
                obst_arr[k] = [dx, dy, d, 1.0]

            row = np.concatenate([self_feat, neigh_arr.flatten(), obst_arr.flatten()])
            obs_rows.append(row.astype(np.float32))

        return np.stack(obs_rows, axis=0)

    # ------------------------------------------------------------------
    # Reward / termination
    # ------------------------------------------------------------------

    def _compute_rewards_and_terms(self, action):
        w = cu.REWARD_WEIGHTS
        rewards = np.zeros(self.max_robots, dtype=np.float32)
        terminated = np.zeros(self.max_robots, dtype=bool)
        static_hit = np.zeros(self.max_robots, dtype=bool)
        robot_hit = np.zeros(self.max_robots, dtype=bool)
        reached_goal = np.zeros(self.max_robots, dtype=bool)

        for i in range(self.max_robots):
            if self.active_mask[i] == 0 or self.done_flags[i]:
                continue

            pos = self.positions[i]

            idx, lateral_err, s_now = cu.nearest_point_on_polyline(
                self._all_polylines[i],
                self._all_arclengths[i],
                pos
            )
                        

            # Look-ahead point on the reference path
            look_pt = cu.lookahead_point(
                self._all_polylines[i],
                self._all_arclengths[i],
                s_now,
                cu.PATH_LOOKAHEAD_DIST
            )

            # Desired heading toward the look-ahead point
            tangent_pt = cu.lookahead_point(
                self._all_polylines[i], self._all_arclengths[i], s_now, lookahead_dist=0.05
            )
            desired_heading = np.arctan2(
                tangent_pt[1] - pos[1],
                tangent_pt[0] - pos[0]
            )

            # Heading error (wrapped to [-pi, pi])
            heading_error = cu.wrap_to_pi(
                desired_heading - self.headings[i]
            )

            prev_s = self.progress_s[i]
            curr_s = self._new_progress[i]

            progress_delta = curr_s - prev_s   # signed: penalize backward/no progress
            

            r = 0.0
            
            if progress_delta < 0.01:
                r -= 0.5    
                
            heading_reward = np.cos(heading_error)
            proximity_factor = np.clip(1.0 - lateral_err / 1.0, 0.0, 1.0)  # 1.0 fully suppresses reward if lateral_err >= 1.0
            r += w["w_heading"] * heading_reward * proximity_factor

            r += w["w_progress"] * progress_delta

            r -= w["w_path_error"] * lateral_err
            r -= w["w_time_penalty"]

            ang_jerk = abs(action[i][1] - self.prev_ang_vel[i])
            r -= w["w_smoothness"] * ang_jerk
            self.prev_ang_vel[i] = action[i][1]

            # static obstacle collision + proximity shaping
            near_static = cu.distance_to_nearest_static_obstacles(pos, self.obstacles, k=1)
            if near_static.shape[0] > 0:
                d = near_static[0, 2] - cu.ROBOT_RADIUS
                if d <= 0:
                    r -= w["w_static_collision"]
                    static_hit[i] = True
                    terminated[i] = True
                elif d < cu.COLLISION_PROXIMITY_MARGIN:
                    r -= w["w_static_proximity"] * (cu.COLLISION_PROXIMITY_MARGIN - d)

            # inter-robot collision + proximity shaping
            if self.enable_robot_collision:
                for j in range(self.max_robots):
                    if j == i or self.active_mask[j] == 0:
                        continue
                    d = float(np.linalg.norm(self.positions[j] - pos)) - 2 * cu.ROBOT_RADIUS
                    if d <= 0:
                        r -= w["w_robot_collision"]
                        robot_hit[i] = True
                        terminated[i] = True
                    elif d < cu.COLLISION_PROXIMITY_MARGIN:
                        r -= w["w_robot_proximity"] * (cu.COLLISION_PROXIMITY_MARGIN - d)

            # goal reached
            goal = self._robot_goals[i] if self._robot_goals[i] is not None else self._all_polylines[i][-1]
            if np.linalg.norm(goal - pos) <= cu.GOAL_REACH_RADIUS:
                r += w["w_goal_bonus"]
                reached_goal[i] = True
                terminated[i] = True

            rewards[i] = r

            # print(
            #     f"Robot {i} | "
            #     f"Progress={progress_delta:.3f} | "
            #     f"Lateral={lateral_err:.3f} | "
            #     f"Improve={path_improvement:.3f} | "
            #     f"Heading={heading_reward:.3f} | "
            #     f"Reward={r:.3f}"
            # )
        # Update progress only AFTER rewards have been computed.
        self.progress_s[:] = self._new_progress[:]

        return rewards, terminated, static_hit, robot_hit, reached_goal

    # ------------------------------------------------------------------
    # Rendering (render_mode="human" live pygame window, or "rgb_array")
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame(return_array=True)
        elif self.render_mode == "human":
            self._render_frame()
            return None

    def _ensure_pygame_init(self):
        import pygame
        if self._pygame_screen is None:
            pygame.init()
            flags = 0
            size = (cu.RENDER_SCREEN_SIZE, cu.RENDER_SCREEN_SIZE + cu.RENDER_PANEL_HEIGHT)
            if self.render_mode == "human":
                self._pygame_screen = pygame.display.set_mode(size, flags)
                pygame.display.set_caption("MultiRobotPathEnv")
            else:
                self._pygame_screen = pygame.Surface(size)
            self._pygame_font = pygame.font.SysFont(None, 22)
            self._pygame_clock = pygame.time.Clock()

    def _world_to_screen(self, pt):
        x, y = pt
        sx = x * cu.RENDER_PIXELS_PER_UNIT
        sy = cu.RENDER_SCREEN_SIZE - y * cu.RENDER_PIXELS_PER_UNIT
        return int(sx), int(sy)

    def _render_frame(self, return_array=False):
        import pygame
        self._ensure_pygame_init()
        screen = self._pygame_screen
        screen.fill(cu.RENDER_COLOR_BG)

        # static obstacles
        for obs in self.obstacles:
            if obs["type"] == "circle":
                cx, cy = obs["center"]
                pygame.draw.circle(screen, cu.RENDER_COLOR_OBSTACLE,
                                    self._world_to_screen((cx, cy)),
                                    int(obs["radius"] * cu.RENDER_PIXELS_PER_UNIT))
            else:
                xs, ys = obs["shape"].exterior.xy
                pts = [self._world_to_screen((x, y)) for x, y in zip(xs, ys)]
                pygame.draw.polygon(screen, cu.RENDER_COLOR_OBSTACLE, pts)

        # each active robot's reference path + current pose
        for i in range(self.max_robots):
            if self.active_mask[i] == 0:
                continue
            color = cu.render_robot_color(i)
            pts = [self._world_to_screen(p) for p in self._all_polylines[i][::4]]
            if len(pts) > 1:
                pygame.draw.lines(screen, cu.RENDER_COLOR_PATH, False, pts, 2)

            px = self._world_to_screen(self.positions[i])
            pygame.draw.circle(screen, color, px, int(cu.ROBOT_RADIUS * cu.RENDER_PIXELS_PER_UNIT))
            hx = px[0] + int(15 * np.cos(self.headings[i]))
            hy = px[1] - int(15 * np.sin(self.headings[i]))
            pygame.draw.line(screen, (0, 0, 0), px, (hx, hy), 2)

        for i in range(self.max_robots):
            if self.active_mask[i] == 0:
                continue
            goal = self._robot_goals[i]
            if goal is not None:
                pygame.draw.circle(screen, cu.RENDER_COLOR_GOAL, self._world_to_screen(goal), 10, 3)

        # HUD
        pygame.draw.rect(screen, cu.RENDER_COLOR_HUD_BG,
                          (0, cu.RENDER_SCREEN_SIZE, cu.RENDER_SCREEN_SIZE, cu.RENDER_PANEL_HEIGHT))
        hud_text = f"step {self.step_count}/{self.max_episode_steps}  n_robots={self.n_active}"
        if self.stage_cfg is not None:
            hud_text += f"  stage={self.stage_cfg['name']}"
        label = self._pygame_font.render(hud_text, True, cu.RENDER_COLOR_HUD_TEXT)
        screen.blit(label, (10, cu.RENDER_SCREEN_SIZE + 18))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self._pygame_clock.tick(cu.RENDER_FPS)

        if return_array:
            arr = pygame.surfarray.array3d(screen)
            return np.transpose(arr, (1, 0, 2))
        return None
