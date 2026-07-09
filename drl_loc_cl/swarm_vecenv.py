"""
swarm_vecenv.py
----------------
Wraps N SwarmEnv instances (one per robot, all sharing a single SwarmWorld)
into a Stable-Baselines3-compatible VecEnv. Because all robots use the SAME
policy (parameter sharing), each robot is just another "row" in the batch
SB3 sees — this is the standard trick for training a single shared policy
across multiple homogeneous agents with off-the-shelf PPO.

Auto-reset behavior matches SB3's VecEnv convention: when a sub-env
terminates/truncates, it is reset immediately and the FINAL observation
before reset is placed in info["terminal_observation"], while the returned
obs is already the fresh post-reset observation.
"""

import glob
import os
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from gymnasium import spaces

from swarm_env import SwarmWorld, SwarmEnv, N_NEAREST_ROBOTS, N_NEAREST_OBSTACLES


def find_robot_map_files(maps_dir, map_number, n_robots):
    """Locate maps/map_{map_number:03d}_robot_*.json, sorted by robot index.
    Requires at least n_robots files to exist; if fewer are found, the
    existing robot files are cycled to fill up to n_robots (useful for
    quick experiments before every robot slot has a dedicated map file)."""
    pattern = os.path.join(maps_dir, f"map_{map_number:03d}_robot_*.json")
    files = sorted(
        glob.glob(pattern),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_robot_")[-1]),
    )
    if not files:
        raise FileNotFoundError(f"No files found matching {pattern}")
    if len(files) < n_robots:
        files = [files[i % len(files)] for i in range(n_robots)]
    else:
        files = files[:n_robots]
    return files


class SwarmVecEnv(VecEnv):
    """A VecEnv of n_robots SwarmEnv instances that all share one
    SwarmWorld, so observations of "nearby robots" reflect real, live
    positions of the other agents in the batch."""

    def __init__(self, maps_dir="maps", map_number=1, n_robots=5,
                 stage=1, render_mode=None):
        map_files = find_robot_map_files(maps_dir, map_number, n_robots)
        self.world = SwarmWorld(map_files)
        self.n_robots = n_robots
        self.stage = stage

        # Only robot 0 renders (rendering all 5 from separate calls would
        # open duplicate windows / redraw redundantly); its render() call
        # draws the full swarm since it reads shared world state anyway.
        self.envs = [
            SwarmEnv(self.world, robot_idx=i, stage=stage,
                     render_mode=(render_mode if i == 0 else None))
            for i in range(n_robots)
        ]

        obs_space = self.envs[0].observation_space
        act_space = self.envs[0].action_space
        super().__init__(n_robots, obs_space, act_space)

        self._actions = None
        self.buf_obs = {k: np.zeros((n_robots,) + s.shape, dtype=s.dtype)
                         for k, s in obs_space.spaces.items()}
        self.buf_rews = np.zeros(n_robots, dtype=np.float32)
        self.buf_dones = np.zeros(n_robots, dtype=bool)
        self.buf_infos = [{} for _ in range(n_robots)]

    # ------------------------------------------------------------------
    def set_stage(self, stage: int):
        """Advance the whole swarm's curriculum stage (called by the
        trainer's curriculum scheduler in train_ppo.py)."""
        self.stage = stage
        for env in self.envs:
            env.set_stage(stage)

    # ------------------------------------------------------------------
    def reset(self):
        for i, env in enumerate(self.envs):
            obs, _ = env.reset()
            self._write_obs(i, obs)
        return self._stacked_obs()

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(self._actions[i])
            done = terminated or truncated
            self.buf_rews[i] = reward
            self.buf_infos[i] = dict(info)

            if done:
                self.buf_infos[i]["terminal_observation"] = obs
                obs, _ = env.reset()

            self._write_obs(i, obs)
            self.buf_dones[i] = done

        return self._stacked_obs(), self.buf_rews.copy(), self.buf_dones.copy(), list(self.buf_infos)

    def close(self):
        for env in self.envs:
            env.close()

    # ------------------------------------------------------------------
    def _write_obs(self, i, obs):
        for k in self.buf_obs:
            self.buf_obs[k][i] = obs[k]

    def _stacked_obs(self):
        return {k: v.copy() for k, v in self.buf_obs.items()}

    # ------------------------------------------------------------------
    # Required abstract VecEnv plumbing (SB3 uses these for VecNormalize,
    # callbacks, attribute access, seeding, etc.)
    def get_attr(self, attr_name, indices=None):
        indices = self._get_indices(indices)
        return [getattr(self.envs[i], attr_name) for i in indices]

    def set_attr(self, attr_name, value, indices=None):
        indices = self._get_indices(indices)
        for i in indices:
            setattr(self.envs[i], attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        indices = self._get_indices(indices)
        return [getattr(self.envs[i], method_name)(*method_args, **method_kwargs) for i in indices]

    def env_is_wrapped(self, wrapper_class, indices=None):
        indices = self._get_indices(indices)
        return [False for _ in indices]

    def seed(self, seed=None):
        seeds = []
        for i, env in enumerate(self.envs):
            s = None if seed is None else seed + i
            env.reset(seed=s)
            seeds.append(s)
        return seeds

    def _get_indices(self, indices):
        if indices is None:
            indices = range(self.n_robots)
        elif isinstance(indices, int):
            indices = [indices]
        return indices
