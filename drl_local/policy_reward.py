"""
policy_reward.py
-----------------
PPO actor-critic network + reward-shaping functions for the variable
robot-count multi-robot path-following task.

Network design
--------------
Each robot's observation row (built by env.py / config_utils.obs_dim_for)
is split back into three parts:
    self_feat   : (SELF_OBS_DIM,)                       - own state/path info
    neighbors   : (max_robots-1, NEIGHBOR_OBS_DIM)       - other robots, padded
    obstacles   : (K_NEAREST_OBSTACLES, OBSTACLE_OBS_DIM) - nearest static obstacles

Neighbors and obstacles are variable-count / order-agnostic sets, so instead
of concatenating them positionally (which forces a fixed identity onto each
slot) we run a small per-entity encoder + masked mean-pool over each set
(a lightweight substitute for attention that is exchangeable/permutation
invariant and cheap). The pooled neighbor and obstacle summaries are
concatenated with the self-feature encoding and passed through the shared
trunk to produce the policy (Gaussian mean + learned log-std) and value.

The SAME network instance is applied independently to every robot slot
(shared weights across robots), so the policy generalizes to any number of
active robots from 1 up to max_robots without changing parameter count.
Padded/inactive slots are simply masked out of the loss in ppo_curriculum.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import config_utils as cu


def _mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class SetEncoder(nn.Module):
    """Per-entity MLP encoder + masked mean-pool over a variable-size,
    order-agnostic set (neighbors or obstacles)."""

    def __init__(self, entity_dim, hidden_dim, out_dim):
        super().__init__()
        self.entity_mlp = _mlp([entity_dim, hidden_dim, out_dim], output_activation=nn.ReLU)

    def forward(self, entities, active_mask):
        """
        entities    : (..., N, entity_dim)
        active_mask : (..., N)  1.0 = real entity, 0.0 = padded
        returns     : (..., out_dim) masked mean-pooled encoding
        """
        encoded = self.entity_mlp(entities)                      # (..., N, out_dim)
        mask = active_mask.unsqueeze(-1)                         # (..., N, 1)
        summed = (encoded * mask).sum(dim=-2)                    # (..., out_dim)
        count = mask.sum(dim=-2).clamp(min=1.0)                  # avoid div-by-zero when 0 active
        return summed / count


class ActorCritic(nn.Module):
    """Shared-weight per-robot actor-critic. Called once per robot slot
    (or batched over the robot dimension) with identical parameters."""

    def __init__(self,
                 self_dim=cu.SELF_OBS_DIM,
                 neighbor_dim=cu.NEIGHBOR_OBS_DIM,
                 obstacle_dim=cu.OBSTACLE_OBS_DIM,
                 n_neighbor_slots=cu.MAX_ROBOTS - 1,
                 n_obstacle_slots=cu.K_NEAREST_OBSTACLES,
                 action_dim=cu.ACTION_DIM,
                 hidden_sizes=cu.PPO_CONFIG["hidden_sizes"],
                 set_encoding_dim=64):
        super().__init__()
        self.self_dim = self_dim
        self.neighbor_dim = neighbor_dim
        self.obstacle_dim = obstacle_dim
        self.n_neighbor_slots = n_neighbor_slots
        self.n_obstacle_slots = n_obstacle_slots

        self.neighbor_encoder = SetEncoder(neighbor_dim, set_encoding_dim, set_encoding_dim)
        self.obstacle_encoder = SetEncoder(obstacle_dim, set_encoding_dim, set_encoding_dim)

        trunk_in = self_dim + set_encoding_dim + set_encoding_dim
        h1, h2 = hidden_sizes
        self.trunk = _mlp([trunk_in, h1, h2], output_activation=nn.ReLU)

        self.actor_mean = nn.Linear(h2, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)  # start fairly tight
        self.critic_head = nn.Linear(h2, 1)

    def _split_obs(self, obs_row):
        """obs_row: (..., total_obs_dim) -> self_feat, neighbor_block, obstacle_block
        each with their own active-mask extracted from the padded feature slots."""
        n_neigh_feats = self.n_neighbor_slots * self.neighbor_dim
        self_feat = obs_row[..., : self.self_dim]
        neigh_flat = obs_row[..., self.self_dim: self.self_dim + n_neigh_feats]
        obst_flat = obs_row[..., self.self_dim + n_neigh_feats:]

        neigh = neigh_flat.view(*neigh_flat.shape[:-1], self.n_neighbor_slots, self.neighbor_dim)
        obst = obst_flat.view(*obst_flat.shape[:-1], self.n_obstacle_slots, self.obstacle_dim)

        # last column of each entity row is the is_active flag (see env.py: neigh_arr[...,-1], obst_arr[...,-1])
        neigh_mask = (neigh[..., -1] > 0).float()
        obst_mask = (obst[..., -1] > 0).float()
        return self_feat, neigh, neigh_mask, obst, obst_mask

    def encode(self, obs_row):
        self_feat, neigh, neigh_mask, obst, obst_mask = self._split_obs(obs_row)
        neigh_summary = self.neighbor_encoder(neigh, neigh_mask)
        obst_summary = self.obstacle_encoder(obst, obst_mask)
        trunk_in = torch.cat([self_feat, neigh_summary, obst_summary], dim=-1)
        return self.trunk(trunk_in)

    def forward(self, obs_row):
        """Returns (dist, value) where dist is a Normal distribution over
        actions and value is the scalar critic estimate, for a batch of
        observation rows of shape (..., total_obs_dim)."""
        features = self.encode(obs_row)
        mean = torch.tanh(self.actor_mean(features))       # actions live in [-1, 1]
        log_std = self.actor_log_std.expand_as(mean)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        value = self.critic_head(features).squeeze(-1)
        return dist, value

    def act(self, obs_row, deterministic=False):
        """Convenience for rollout collection: returns (action, log_prob, value)."""
        dist, value = self.forward(obs_row)
        action = dist.mean if deterministic else dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(self, obs_row, action):
        """Used during PPO updates: returns (log_prob, entropy, value) for
        a batch of previously-taken actions under the current policy."""
        dist, value = self.forward(obs_row)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


# ============================================================
# Reward-shaping functions
# ------------------------------------------------------------
# These mirror the per-robot reward terms computed inline inside
# env.py's _compute_rewards_and_terms(), factored out here as pure
# functions so they can be unit-tested, reused by eval.py for logging
# reward breakdowns, or swapped out per curriculum stage without
# touching the environment's stepping logic.
# ============================================================

def progress_reward(prev_arclength, curr_arclength, weight=None):
    w = cu.REWARD_WEIGHTS["w_progress"] if weight is None else weight
    return w * (curr_arclength - prev_arclength)

def path_error_penalty(lateral_error, weight=None):
    w = cu.REWARD_WEIGHTS["w_path_error"] if weight is None else weight
    return -w * lateral_error

def time_penalty(weight=None):
    w = cu.REWARD_WEIGHTS["w_time_penalty"] if weight is None else weight
    return -w

def smoothness_penalty(prev_ang_vel_cmd, curr_ang_vel_cmd, weight=None):
    w = cu.REWARD_WEIGHTS["w_smoothness"] if weight is None else weight
    return -w * abs(curr_ang_vel_cmd - prev_ang_vel_cmd)

def static_collision_term(clearance, robot_radius=cu.ROBOT_RADIUS,
                           margin=cu.COLLISION_PROXIMITY_MARGIN, weights=None):
    """clearance: raw distance from robot center to nearest static obstacle
    surface (before subtracting robot radius). Returns (reward_delta, collided_bool)."""
    w = cu.REWARD_WEIGHTS if weights is None else weights
    d = clearance - robot_radius
    if d <= 0:
        return -w["w_static_collision"], True
    if d < margin:
        return -w["w_static_proximity"] * (margin - d), False
    return 0.0, False

def robot_collision_term(center_distance, robot_radius=cu.ROBOT_RADIUS,
                          margin=cu.COLLISION_PROXIMITY_MARGIN, weights=None):
    """center_distance: raw distance between two robot centers. Returns
    (reward_delta, collided_bool)."""
    w = cu.REWARD_WEIGHTS if weights is None else weights
    d = center_distance - 2 * robot_radius
    if d <= 0:
        return -w["w_robot_collision"], True
    if d < margin:
        return -w["w_robot_proximity"] * (margin - d), False
    return 0.0, False

def goal_bonus_term(dist_to_goal, reach_radius=cu.GOAL_REACH_RADIUS, weight=None):
    w = cu.REWARD_WEIGHTS["w_goal_bonus"] if weight is None else weight
    reached = dist_to_goal <= reach_radius
    return (w if reached else 0.0), reached

def total_reward_breakdown(prev_arclength, curr_arclength, lateral_error,
                            prev_ang_vel_cmd, curr_ang_vel_cmd,
                            static_clearance, neighbor_distances,
                            dist_to_goal, enable_robot_collision=True):
    """Full per-robot reward decomposition for one step, mirroring env.py's
    _compute_rewards_and_terms(). Returns (total_reward, breakdown_dict,
    static_hit, robot_hit, reached_goal)."""
    breakdown = {}
    breakdown["progress"] = progress_reward(prev_arclength, curr_arclength)
    breakdown["path_error"] = path_error_penalty(lateral_error)
    breakdown["time"] = time_penalty()
    breakdown["smoothness"] = smoothness_penalty(prev_ang_vel_cmd, curr_ang_vel_cmd)

    static_term, static_hit = static_collision_term(static_clearance)
    breakdown["static"] = static_term

    robot_term_total = 0.0
    robot_hit = False
    if enable_robot_collision:
        for d in neighbor_distances:
            term, hit = robot_collision_term(d)
            robot_term_total += term
            robot_hit = robot_hit or hit
    breakdown["robot"] = robot_term_total

    goal_term, reached_goal = goal_bonus_term(dist_to_goal)
    breakdown["goal"] = goal_term
    if reached_goal:
        static_hit = static_hit and False  # goal reach takes precedence in env.py's ordering

    total = sum(breakdown.values())
    return total, breakdown, static_hit, robot_hit, reached_goal
