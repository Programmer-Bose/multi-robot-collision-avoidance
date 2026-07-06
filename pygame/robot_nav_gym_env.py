"""
Gymnasium wrapper around env.SingleRobotEnv.

Observation is a Dict so that downstream policies can route the occupancy
grid through a CNN branch, the rangefinder/goal/velocity vector through an
MLP branch, and (later, multi-robot) neighbor embeddings through an
attention block -- without forcing a single flattened vector now that would
need to be re-split later.

Reward design (documented explicitly since this is the one piece that is a
genuine research decision, not boilerplate):
    + progress_weight * (prev_dist_to_goal - curr_dist_to_goal)   # dense shaping
    + goal_bonus on reaching a task point
    + depot_bonus on completing all tasks and returning to depot
    - collision_penalty (terminal) on collision
    - step_penalty each step (encourages efficiency, discourages stalling)

terminated vs truncated (Gym/SB3-relevant distinction):
    terminated = True on collision OR all tasks done (env's own terminal state)
    truncated  = True on hitting max_episode_steps (time limit, not a true
                 terminal state -- matters for value bootstrapping)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env import SingleRobotEnv, make_default_scenario
from rangefinder import simulate_rangefinder
from occupancy_grid import compute_occupancy_grid


class RobotNavGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenario_fn=None, scenario_kwargs=None,
                 n_rays=15, sensor_range=5.0, grid_size=21, grid_resolution=0.25,
                 max_episode_steps=600,
                 progress_weight=1.0, goal_bonus=20.0, depot_bonus=50.0,
                 collision_penalty=100.0, step_penalty=0.01,
                 skipped_task_penalty=5.0, render_mode=None, render_fps=30):
        super().__init__()

        assert render_mode is None or render_mode in ("human",), \
            f"render_mode must be None or 'human', got {render_mode!r}"
        self.render_mode = render_mode
        self.render_fps = render_fps
        self._renderer = None   # lazily created on first render() call
        self._clock = None
        self._traj_points = []
        

        # scenario_fn lets the caller pass make_default_scenario (or a custom
        # builder for the manual/interactive scenario editor in Phase 5)
        # without this wrapper needing to know how scenarios are constructed.
        self.scenario_fn = scenario_fn or make_default_scenario
        self.scenario_kwargs = scenario_kwargs or {}

        self.n_rays = n_rays
        self.sensor_range = sensor_range
        self.grid_size = grid_size
        self.grid_resolution = grid_resolution
        self.max_episode_steps = max_episode_steps

        self.progress_weight = progress_weight
        self.goal_bonus = goal_bonus
        self.depot_bonus = depot_bonus
        self.collision_penalty = collision_penalty
        self.step_penalty = step_penalty
        self.skipped_task_penalty = skipped_task_penalty

        self.env: SingleRobotEnv = None  # built in reset()
        self._step_count = 0
        self._prev_dist_to_goal = None

        # action space: (v, omega). Bounds mirror de_mpc.py's default bounds
        # (small reverse allowance + full omega range) rather than [0, v_max],
        # since the DE-MPC demonstrations the policy will imitate use that
        # same asymmetric range.
        self.action_space = spaces.Box(
            low=np.array([-0.3, -np.pi / 2], dtype=np.float32),
            high=np.array([1.0, np.pi / 2], dtype=np.float32),
            dtype=np.float32,
        )

        # world diagonal gives a real (non-infinite) upper bound for goal_dist;
        # falls back to a generous default if scenario_kwargs doesn't specify world
        world = self.scenario_kwargs.get("world", (0, 10, 0, 10))
        xmin, xmax, ymin, ymax = world
        max_dist = float(np.hypot(xmax - xmin, ymax - ymin))
        v_max = float(self.scenario_kwargs.get("v_max", 2.5))
        omega_max = float(self.scenario_kwargs.get("omega_max", np.pi))

        self.observation_space = spaces.Dict({
            "ranges": spaces.Box(low=0.0, high=sensor_range,
                                  shape=(n_rays,), dtype=np.float32),
            "occupancy_grid": spaces.Box(low=0.0, high=1.0,
                                          shape=(grid_size, grid_size), dtype=np.float32),
            "goal_dist": spaces.Box(low=0.0, high=max_dist, shape=(1,), dtype=np.float32),
            "goal_bearing": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            "velocity": spaces.Box(low=np.array([-v_max, -omega_max], dtype=np.float32),
                                    high=np.array([v_max, omega_max], dtype=np.float32),
                                    dtype=np.float32),
        })

    def _build_obs(self, obs):
        rx, ry, rtheta = obs["robot_state"]
        gx, gy = obs["goal"]
        goal_dist = float(np.hypot(gx - rx, gy - ry))
        goal_dist = min(goal_dist, self.observation_space["goal_dist"].high[0])
        goal_bearing = float((np.arctan2(gy - ry, gx - rx) - rtheta + np.pi)
                              % (2 * np.pi) - np.pi)

        ranges = simulate_rangefinder(
            robot_state=obs["robot_state"],
            static_obstacles=obs["static_obstacles"],
            dynamic_obstacles=obs["dynamic_obstacles"],
            world_bounds=self.env.world_bounds,
            n_rays=self.n_rays, max_range=self.sensor_range,
        ).astype(np.float32)

        occ_grid = compute_occupancy_grid(
            robot_state=obs["robot_state"],
            static_obstacles=obs["static_obstacles"],
            dynamic_obstacles=obs["dynamic_obstacles"],
            grid_size=self.grid_size, resolution=self.grid_resolution,
        ).astype(np.float32)

        return {
            "ranges": ranges,
            "occupancy_grid": occ_grid,
            "goal_dist": np.array([goal_dist], dtype=np.float32),
            "goal_bearing": np.array([goal_bearing], dtype=np.float32),
            "velocity": np.array(self._last_action, dtype=np.float32),
        }, goal_dist

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        kwargs = dict(self.scenario_kwargs)
        print(f"resetting env with kwargs={kwargs}")
        if seed is not None:
            kwargs["seed"] = seed
        self.env = self.scenario_fn(**kwargs)
        raw_obs = self.env.reset()

        self._step_count = 0
        self._last_action = (0.0, 0.0)
        obs, dist = self._build_obs(raw_obs)
        self._prev_dist_to_goal = dist

        info = {"n_tasks_completed": self.env.n_tasks_completed(),
                "n_tasks_total": self.env.n_tasks_total()}

        self._traj_points = [(self.env.robot.x, self.env.robot.y)]
        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        v, omega = float(action[0]), float(action[1])
        self._last_action = (v, omega)
        raw_obs, done, step_info = self.env.step(v, omega)
        self._step_count += 1

        obs, dist = self._build_obs(raw_obs)

        reward = -self.step_penalty
        reward += self.progress_weight * (self._prev_dist_to_goal - dist)
        self._prev_dist_to_goal = dist

        if step_info["reached_goal"]:
            reward += self.goal_bonus
        if step_info["skipped_task"]:
            reward -= self.skipped_task_penalty
        if step_info["collided"]:
            reward -= self.collision_penalty
        if done and not step_info["collided"] and self.env.all_tasks_done():
            reward += self.depot_bonus

        terminated = bool(done)  # collision or all_tasks_done, per env.py's own `done` flag
        truncated = bool(self._step_count >= self.max_episode_steps) and not terminated

        info = {
            "n_tasks_completed": self.env.n_tasks_completed(),
            "n_tasks_total": self.env.n_tasks_total(),
            "collided": step_info["collided"],
            "collision_kind": step_info["collision_kind"],
            "skipped_task": step_info["skipped_task"],
        }

        self._traj_points.append((self.env.robot.x, self.env.robot.y))
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return  # standard Gym convention: no-op if rendering wasn't requested

        from pygame_renderer import SceneRenderer
        import pygame

        if self._renderer is None:
            self._renderer = SceneRenderer(self.env.world_bounds,
                                            caption="RobotNavGymEnv (live policy view)")
            self._clock = pygame.time.Clock()

        if not self._renderer.handle_quit_events():
            self.close()
            return

        n_done = self.env.n_tasks_completed()
        n_total = self.env.n_tasks_total()
        phase = "depot" if self.env.current_task is None else f"task {self.env.active_task_orig_idx()+1}"
        status = [f"step {self._step_count} | tasks {n_done}/{n_total} | targeting {phase}",
                  f"last action: v={self._last_action[0]:.2f}, omega={self._last_action[1]:.2f}"]

        self._renderer.draw(self.env, self._traj_points, status)
        self._renderer.tick(self._clock, self.render_fps)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.env = None


if __name__ == "__main__":
    # smoke test: random actions for a couple episodes
    env = RobotNavGymEnv(
        scenario_kwargs=dict(seed =1,n_static=8, n_dynamic=5, n_tasks=8, omega_max=np.pi),
        max_episode_steps=200,
    )
    for ep in range(2):
        obs, info = env.reset(seed=ep)
        ep_reward = 0.0
        for t in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        print(f"episode {ep}: steps={t+1} reward={ep_reward:.2f} "
              f"tasks={info['n_tasks_completed']}/{info['n_tasks_total']} "
              f"collided={info['collided']} truncated={truncated}")
        print("  obs shapes:", {k: v.shape for k, v in obs.items()})
