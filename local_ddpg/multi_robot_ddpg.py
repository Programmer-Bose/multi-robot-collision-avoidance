"""
Multi-Robot DDPG Path-Following + Obstacle Avoidance (single-file)
====================================================================

Extends the original 5-input path-following-only DDPG
(ran_ddpg_path_following.py: state = [x_e, y_e, theta_e, prev_v, prev_w])
to a 7-input state that adds obstacle awareness:

    state = [x_e, y_e, theta_e, prev_v, prev_w, obs_dist, obs_angle]

  - obs_dist / obs_angle describe the CLOSEST obstacle to the robot at
    that instant, where "obstacle" = min-over( static circle obstacles
    from the map JSON,  every OTHER robot's current position ).
    i.e. teammate robots are just dynamic obstacles from each robot's
    own point of view - no shared/global state is required.

Supports N robots simultaneously (this script defaults to 3, per your
requirement of 3 obstacle maps + 3 paths), each with:
  - its own static-obstacle map (map_00X_robot_Y.json - circle obstacles,
    given in 800x800 pixel space and rescaled into the 12x12 arena),
  - its own reference path (map_00X_robot_Y_manual_control_points.json,
    the per-segment B-spline control-point format).

All robots share ONE policy (one Actor/Critic pair). Every simulation
step, all robots' states are stacked into a single batch and passed
through the actor in ONE forward call (true batched inference), then
the environment advances all robots' kinematics together and checks
robot-robot / robot-static collisions.

TRANSFER LEARNING
------------------
`transfer_5_to_7(old_checkpoint_path)` loads an old 5-input DDPGAgent
checkpoint (e.g. ddpg_path_following_multi_curve.pt) and constructs a
new 7-input DDPGAgent whose hidden/output layers are copied verbatim
from the old model. Only the first Linear layer's weight matrix differs
in shape (5 vs 7 input columns); the first 5 columns are copied from
the old weights and the 2 new obs_dist/obs_angle columns are freshly
(small-) initialized, so path-following behavior is preserved while the
obstacle-avoidance pathway starts learning from near-scratch.

Usage
-----
Train 3 robots from scratch:
    python multi_robot_ddpg.py \
        --obstacle_maps map_003_robot_1.json map_003_robot_2.json map_003_robot_1.json \
        --path_jsons map_003_robot_1_manual_control_points.json \
                     map_003_robot_2_manual_control_points.json \
                     map_003_robot_1_manual_control_points.json \
        --episodes 500

Transfer-learn from an existing 5-input path-following checkpoint:
    python multi_robot_ddpg.py \
        --obstacle_maps map_003_robot_1.json map_003_robot_2.json map_003_robot_1.json \
        --path_jsons map_003_robot_1_manual_control_points.json \
                     map_003_robot_2_manual_control_points.json \
                     map_003_robot_1_manual_control_points.json \
        --resume_from_5d solves_drl/ddpg_path_following_multi_curve.pt \
        --episodes 500
"""

import os
import json
import random
import argparse
import datetime
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# ----------------------------------------------------------------------
# 0. Hyperparameters
# ----------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BSPLINE_DEGREE = 3
N_SAMPLES_PER_SEGMENT = 80

DT = 0.1
MAX_LINEAR_VEL = 1.5
MAX_ANGULAR_VEL = np.pi / 2
MAX_STEPS_PER_EPISODE = 500

INIT_POS_JITTER = 0.6
INIT_HEADING_JITTER = 0.6

GOAL_TOLERANCE = 0.3
OFF_PATH_TOLERANCE = 4.0

ARENA_MIN = np.array([0.0, 0.0])
ARENA_MAX = np.array([12.0, 12.0])

# original obstacle JSONs are laid out on an 800x800 canvas -> rescale
# into the 12x12 training arena
OBSTACLE_MAP_PX = 800.0
OBSTACLE_SCALE = (ARENA_MAX[0] - ARENA_MIN[0]) / OBSTACLE_MAP_PX  # = 0.015

ROBOT_RADIUS = 0.25          # for robot-robot / robot-static collision checks
STATIC_OBSTACLE_MARGIN = 0.1  # extra clearance added on top of obstacle radius

W_PROGRESS = 2.0
W_OBSTACLE_SHAPING = 0.3      # small continuous penalty for being close to any obstacle
OBSTACLE_SENSE_RADIUS = 3.0   # obs_dist is clipped/normalized against this range

COLLISION_PENALTY_STATIC = -10.0
COLLISION_PENALTY_ROBOT = -10.0
GOAL_REWARD = 10.0

STATE_DIM_OLD = 5
STATE_DIM = 7          # [x_e, y_e, theta_e, prev_v, prev_w, obs_dist, obs_angle]
ACTION_DIM = 2
ACTOR_HIDDEN = (400, 300, 300)
CRITIC_HIDDEN = (400, 300, 300)

ACTOR_LR = 0.001
CRITIC_LR = 0.001
GAMMA = 0.9
TAU = 0.01
REPLAY_BUFFER_SIZE = 200_000
MIN_REPLAY_BEFORE_TRAINING = 2_000
BATCH_SIZE = 128
GRAD_CLIP_NORM = 5.0

OU_THETA = 0.15
OU_SIGMA = 0.3
OU_SIGMA_MIN = 0.05
OU_SIGMA_DECAY = 0.999

OUTPUT_DIR = "solves_drl_multi"


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def make_clamped_knot_vector(n_ctrl_pts, degree):
    n_internal = n_ctrl_pts - degree - 1
    if n_internal > 0:
        internal_knots = np.linspace(0, 1, n_internal + 2)[1:-1]
    else:
        internal_knots = np.array([])
    return np.concatenate((np.zeros(degree + 1), internal_knots, np.ones(degree + 1)))


def bspline_curve(control_points, n_samples, degree=BSPLINE_DEGREE):
    control_points = np.asarray(control_points)
    n_ctrl_pts = len(control_points)
    k = min(degree, n_ctrl_pts - 1)
    knots = make_clamped_knot_vector(n_ctrl_pts, k)
    t = np.linspace(0.0, 1.0, n_samples)
    spline_x = BSpline(knots, control_points[:, 0], k)
    spline_y = BSpline(knots, control_points[:, 1], k)
    return np.column_stack([spline_x(t), spline_y(t)])


# ----------------------------------------------------------------------
# 1. Reference path (per-segment control-point JSON -> stitched polyline)
# ----------------------------------------------------------------------

class ReferencePath:
    def __init__(self, control_points_json, map_name=None):
        self.map_name = map_name or os.path.splitext(os.path.basename(control_points_json))[0]
        self.points = self._load_points_from_json(control_points_json)

        diffs = np.diff(self.points, axis=0)
        seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        self.cum_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        self.total_length = self.cum_length[-1]

        headings = np.zeros(len(self.points))
        headings[0] = np.arctan2(diffs[0, 1], diffs[0, 0])
        headings[-1] = np.arctan2(diffs[-1, 1], diffs[-1, 0])
        for i in range(1, len(self.points) - 1):
            v = self.points[i + 1] - self.points[i - 1]
            headings[i] = np.arctan2(v[1], v[0])
        self.headings = headings

    @staticmethod
    def _load_points_from_json(control_points_json):
        with open(control_points_json, "r") as f:
            data = json.load(f)
        samples = []
        for seg in data["segments"]:
            start = np.asarray(seg["start_point"], dtype=float)
            end = np.asarray(seg["end_point"], dtype=float)
            free_pts = np.asarray(seg["control_points"], dtype=float)
            full_ctrl = np.vstack([start, free_pts, end])
            curve = bspline_curve(full_ctrl, N_SAMPLES_PER_SEGMENT)
            if samples:
                curve = curve[1:]
            samples.append(curve)
        return np.vstack(samples)

    def nearest_index(self, xy):
        d2 = np.sum((self.points - xy[None, :]) ** 2, axis=1)
        return int(np.argmin(d2))

    def point_at_index(self, idx):
        return self.points[idx], self.headings[idx]

    def arclength_at_index(self, idx):
        return self.cum_length[idx]

    @property
    def goal_point(self):
        return self.points[-1]


# ----------------------------------------------------------------------
# 2. Static obstacle map (rescaled circles)
# ----------------------------------------------------------------------

class ObstacleMap:
    """Loads circle obstacles from a map_00X_robot_Y.json file and rescales
    them from the original 800x800 px canvas into the 12x12 training arena."""

    def __init__(self, map_json_path):
        with open(map_json_path, "r") as f:
            data = json.load(f)
        centers = []
        radii = []
        for obs in data.get("obstacles", []):
            if obs.get("type") != "circle":
                continue

            px, py = obs["position"]

            # Convert image coordinates (origin at top-left)
            # to arena coordinates (origin at bottom-left)
            x = px * OBSTACLE_SCALE
            y = ARENA_MAX[1] - py * OBSTACLE_SCALE

            centers.append([x, y])
            radii.append(obs["radius"] * OBSTACLE_SCALE)
        self.centers = np.asarray(centers, dtype=float) if centers else np.zeros((0, 2))
        self.radii = np.asarray(radii, dtype=float) if radii else np.zeros((0,))

    def nearest_distance_and_angle(self, xy, theta):
        """Returns (surface_distance, relative_angle) to the closest
        static obstacle, or (large_value, 0.0) if there are none."""
        if len(self.centers) == 0:
            return OBSTACLE_SENSE_RADIUS, 0.0
        d = self.centers - xy[None, :]
        center_dist = np.hypot(d[:, 0], d[:, 1])
        surface_dist = center_dist - self.radii
        idx = int(np.argmin(surface_dist))
        dx, dy = d[idx]
        world_angle = np.arctan2(dy, dx)
        rel_angle = wrap_to_pi(world_angle - theta)
        return float(surface_dist[idx]), float(rel_angle)

    def min_surface_distance(self, xy):
        if len(self.centers) == 0:
            return np.inf
        d = self.centers - xy[None, :]
        center_dist = np.hypot(d[:, 0], d[:, 1])
        return float(np.min(center_dist - self.radii))


# ----------------------------------------------------------------------
# 3. Multi-robot environment (batched)
# ----------------------------------------------------------------------

class MultiRobotEnv:
    """Simulates N robots simultaneously. Each robot has its own reference
    path and its own static obstacle map. Every OTHER robot is treated as
    a dynamic (moving) obstacle from each robot's own egocentric point of
    view - there is no shared global state, only each robot's own
    7-dim observation."""

    def __init__(self, ref_paths, obstacle_maps):
        assert len(ref_paths) == len(obstacle_maps)
        self.n_robots = len(ref_paths)
        self.paths = ref_paths
        self.obstacle_maps = obstacle_maps

        self.poses = np.zeros((self.n_robots, 3))
        self.prev_actions = np.zeros((self.n_robots, 2))
        self.prev_arclength = np.zeros(self.n_robots)
        self.steps = 0
        self.done_mask = np.zeros(self.n_robots, dtype=bool)

    def reset(self):
        for i in range(self.n_robots):
            p_xy, p_heading = self.paths[i].point_at_index(0)
            jitter_xy = np.random.uniform(-INIT_POS_JITTER, INIT_POS_JITTER, size=2)
            jitter_theta = np.random.uniform(-INIT_HEADING_JITTER, INIT_HEADING_JITTER)
            self.poses[i] = [
                p_xy[0] + jitter_xy[0],
                p_xy[1] + jitter_xy[1],
                wrap_to_pi(p_heading + jitter_theta),
            ]
            self.prev_arclength[i] = self.paths[i].arclength_at_index(0)
        self.prev_actions[:] = 0.0
        self.steps = 0
        self.done_mask[:] = False
        return self._compute_states()

    def _dynamic_nearest(self, robot_idx, xy, theta):
        """Closest teammate robot's distance/angle (surface-approximated
        via ROBOT_RADIUS), or (inf, 0.0) if there is only one robot."""
        best_dist = np.inf
        best_angle = 0.0
        for j in range(self.n_robots):
            if j == robot_idx:
                continue
            other_xy = self.poses[j][:2]
            d = other_xy - xy
            center_dist = np.hypot(d[0], d[1])
            surface_dist = center_dist - 2 * ROBOT_RADIUS
            if surface_dist < best_dist:
                best_dist = surface_dist
                best_angle = wrap_to_pi(np.arctan2(d[1], d[0]) - theta)
        return best_dist, best_angle

    def _compute_states(self):
        states = np.zeros((self.n_robots, STATE_DIM), dtype=np.float32)
        infos = []
        for i in range(self.n_robots):
            x, y, theta = self.poses[i]
            idx = self.paths[i].nearest_index(np.array([x, y]))
            p_xy, p_heading = self.paths[i].point_at_index(idx)

            dx = p_xy[0] - x
            dy = p_xy[1] - y
            x_e = np.cos(theta) * dx + np.sin(theta) * dy
            y_e = -np.sin(theta) * dx + np.cos(theta) * dy
            theta_e = wrap_to_pi(p_heading - theta)

            static_dist, static_angle = self.obstacle_maps[i].nearest_distance_and_angle(
                np.array([x, y]), theta)
            dyn_dist, dyn_angle = self._dynamic_nearest(i, np.array([x, y]), theta)

            if dyn_dist < static_dist:
                obs_dist, obs_angle = dyn_dist, dyn_angle
            else:
                obs_dist, obs_angle = static_dist, static_angle
            obs_dist = float(np.clip(obs_dist, -OBSTACLE_SENSE_RADIUS, OBSTACLE_SENSE_RADIUS))

            states[i] = [x_e, y_e, theta_e, self.prev_actions[i, 0], self.prev_actions[i, 1],
                         obs_dist, obs_angle]
            infos.append({"nearest_idx": idx, "dist_to_path": np.hypot(dx, dy),
                           "static_dist": static_dist, "dyn_dist": dyn_dist})
        return states, infos

    def step(self, actions):
        """actions: (n_robots, 2) batch of [v, w]."""
        actions = np.asarray(actions, dtype=float)
        rewards = np.zeros(self.n_robots)
        collided_static = np.zeros(self.n_robots, dtype=bool)
        collided_robot = np.zeros(self.n_robots, dtype=bool)
        left_arena = np.zeros(self.n_robots, dtype=bool)
        off_path = np.zeros(self.n_robots, dtype=bool)
        reached_goal = np.zeros(self.n_robots, dtype=bool)

        for i in range(self.n_robots):
            if self.done_mask[i]:
                continue
            v = float(np.clip(actions[i, 0], -MAX_LINEAR_VEL, MAX_LINEAR_VEL))
            w = float(np.clip(actions[i, 1], -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL))
            x, y, theta = self.poses[i]
            x += v * np.cos(theta) * DT
            y += v * np.sin(theta) * DT
            theta = wrap_to_pi(theta + w * DT)
            self.poses[i] = [x, y, theta]
            self.prev_actions[i] = [v, w]

        self.steps += 1
        states, infos = self._compute_states()

        for i in range(self.n_robots):
            if self.done_mask[i]:
                continue
            x, y, theta = self.poses[i]
            info = infos[i]
            x_e, y_e, theta_e = states[i, 0], states[i, 1], states[i, 2]

            r = -(abs(x_e) + abs(y_e) + abs(theta_e))
            current_arclength = self.paths[i].arclength_at_index(info["nearest_idx"])
            progress = current_arclength - self.prev_arclength[i]
            r += W_PROGRESS * progress
            self.prev_arclength[i] = current_arclength

            # continuous obstacle-avoidance shaping: penalize being close
            nearest_any = min(info["static_dist"], info["dyn_dist"])
            if nearest_any < OBSTACLE_SENSE_RADIUS:
                r -= W_OBSTACLE_SHAPING * max(0.0, OBSTACLE_SENSE_RADIUS - nearest_any) / OBSTACLE_SENSE_RADIUS

            if info["static_dist"] < 0:
                collided_static[i] = True
                r += COLLISION_PENALTY_STATIC
            if info["dyn_dist"] < 0:
                collided_robot[i] = True
                r += COLLISION_PENALTY_ROBOT

            if x < ARENA_MIN[0] or x > ARENA_MAX[0] or y < ARENA_MIN[1] or y > ARENA_MAX[1]:
                left_arena[i] = True
                r -= 10.0
            elif info["dist_to_path"] > OFF_PATH_TOLERANCE:
                off_path[i] = True
                r -= 10.0

            dist_to_goal = np.hypot(x - self.paths[i].goal_point[0], y - self.paths[i].goal_point[1])
            if info["nearest_idx"] >= len(self.paths[i].points) - 2 and dist_to_goal < GOAL_TOLERANCE:
                reached_goal[i] = True
                r += GOAL_REWARD

            rewards[i] = r

        done = (collided_static | collided_robot | left_arena | off_path | reached_goal
                | (self.steps >= MAX_STEPS_PER_EPISODE))
        self.done_mask = self.done_mask | done

        all_done = bool(np.all(self.done_mask))
        info_out = {
            "collided_static": collided_static, "collided_robot": collided_robot,
            "left_arena": left_arena, "off_path": off_path, "reached_goal": reached_goal,
            "done_mask": self.done_mask.copy(),
        }
        return states, rewards, done, all_done, info_out


# ----------------------------------------------------------------------
# 4. Networks (7-input state)
# ----------------------------------------------------------------------

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden, max_linear, max_angular):
        super().__init__()
        h1, h2, h3 = hidden
        self.net = nn.Sequential(
            nn.Linear(state_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, h3), nn.ReLU(),
            nn.Linear(h3, action_dim), nn.Tanh(),
        )
        self.register_buffer("action_scale", torch.tensor([max_linear, max_angular], dtype=torch.float32))

    def forward(self, state):
        return self.net(state) * self.action_scale


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        super().__init__()
        h1, h2, h3 = hidden
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, h3), nn.ReLU(),
            nn.Linear(h3, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class OUNoise:
    def __init__(self, dim, theta=OU_THETA, sigma=OU_SIGMA):
        self.dim = dim
        self.theta = theta
        self.sigma = sigma
        self.state = np.zeros(dim)

    def reset(self):
        self.state = np.zeros(self.dim)

    def sample(self):
        dx = self.theta * (-self.state) + self.sigma * np.random.randn(self.dim)
        self.state = self.state + dx
        return self.state

    def decay_sigma(self):
        self.sigma = max(OU_SIGMA_MIN, self.sigma * OU_SIGMA_DECAY)


class ReplayBuffer:
    def __init__(self, capacity=REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, done = map(np.array, zip(*batch))
        return s, a, r, s2, done

    def __len__(self):
        return len(self.buffer)


# ----------------------------------------------------------------------
# 5. DDPG agent (single shared policy for all robots, batched inference)
# ----------------------------------------------------------------------

class DDPGAgent:
    def __init__(self, state_dim=STATE_DIM):
        self.state_dim = state_dim
        self.actor = Actor(state_dim, ACTION_DIM, ACTOR_HIDDEN, MAX_LINEAR_VEL, MAX_ANGULAR_VEL).to(DEVICE)
        self.actor_target = Actor(state_dim, ACTION_DIM, ACTOR_HIDDEN, MAX_LINEAR_VEL, MAX_ANGULAR_VEL).to(DEVICE)
        self.critic = Critic(state_dim, ACTION_DIM, CRITIC_HIDDEN).to(DEVICE)
        self.critic_target = Critic(state_dim, ACTION_DIM, CRITIC_HIDDEN).to(DEVICE)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.replay = ReplayBuffer()
        # one independent OU noise process PER ROBOT so exploration isn't
        # identical/correlated across robots
        self.noises = None  # set via init_noises(n_robots)

    def init_noises(self, n_robots):
        self.noises = [OUNoise(ACTION_DIM) for _ in range(n_robots)]

    def reset_noises(self):
        for n in self.noises:
            n.reset()

    def decay_noises(self):
        for n in self.noises:
            n.decay_sigma()

    def select_actions_batch(self, states, explore=True):
        """states: (n_robots, state_dim) -> actions: (n_robots, action_dim).
        A single forward pass handles ALL robots at once (true batching)."""
        state_t = torch.as_tensor(states, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            actions = self.actor(state_t).cpu().numpy()
        if explore:
            for i in range(len(actions)):
                actions[i] = actions[i] + self.noises[i].sample() * np.array([MAX_LINEAR_VEL, MAX_ANGULAR_VEL])
        actions[:, 0] = np.clip(actions[:, 0], -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
        actions[:, 1] = np.clip(actions[:, 1], -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
        return actions

    def _soft_update(self, target_net, source_net, tau):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def train_step(self):
        if len(self.replay) < max(BATCH_SIZE, MIN_REPLAY_BEFORE_TRAINING):
            return None, None

        s, a, r, s2, done = self.replay.sample(BATCH_SIZE)
        s = torch.as_tensor(s, dtype=torch.float32, device=DEVICE)
        a = torch.as_tensor(a, dtype=torch.float32, device=DEVICE)
        r = torch.as_tensor(r, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        s2 = torch.as_tensor(s2, dtype=torch.float32, device=DEVICE)
        done = torch.as_tensor(done, dtype=torch.float32, device=DEVICE).unsqueeze(1)

        with torch.no_grad():
            next_action = self.actor_target(s2)
            target_q = self.critic_target(s2, next_action)
            y = r + GAMMA * (1.0 - done) * target_q

        current_q = self.critic(s, a)
        critic_loss = nn.functional.mse_loss(current_q, y)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP_NORM)
        self.critic_optim.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP_NORM)
        self.actor_optim.step()

        self._soft_update(self.actor_target, self.actor, TAU)
        self._soft_update(self.critic_target, self.critic, TAU)
        return critic_loss.item(), actor_loss.item()

    def save(self, path):
        torch.save({
            "state_dim": self.state_dim,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])


# ----------------------------------------------------------------------
# 6. Transfer learning: 5-input checkpoint -> 7-input agent
# ----------------------------------------------------------------------

def transfer_5_to_7(old_checkpoint_path, new_col_init_std=0.01, seed=None):
    """Loads an OLD 5-input DDPGAgent checkpoint (path-following only) and
    returns a NEW 7-input DDPGAgent (path-following + obstacle avoidance)
    whose weights are warm-started from it.

    Only the very first Linear layer of the actor and critic differs in
    shape (extra 2 input columns for obs_dist / obs_angle). Every other
    layer is copied 1:1. The critic's first layer also has a shape
    difference because its input is (state_dim + action_dim); the action
    columns are copied unchanged, only the state columns are extended.
    """
    if seed is not None:
        torch.manual_seed(seed)

    old_ckpt = torch.load(old_checkpoint_path, map_location=DEVICE)
    new_agent = DDPGAgent(state_dim=STATE_DIM)

    def _copy_actor(old_sd, new_actor):
        new_sd = new_actor.state_dict()
        # first Linear layer: net.0.weight (h1, 5) -> (h1, 7); net.0.bias unchanged
        old_w = old_sd["net.0.weight"]   # (h1, 5)
        old_b = old_sd["net.0.bias"]     # (h1,)
        h1 = old_w.shape[0]
        new_w = torch.randn(h1, STATE_DIM) * new_col_init_std
        new_w[:, :STATE_DIM_OLD] = old_w
        new_sd["net.0.weight"] = new_w
        new_sd["net.0.bias"] = old_b
        # remaining layers copy verbatim
        for key in old_sd:
            if key.startswith("net.0."):
                continue
            new_sd[key] = old_sd[key]
        new_actor.load_state_dict(new_sd)

    def _copy_critic(old_sd, new_critic):
        new_sd = new_critic.state_dict()
        old_w = old_sd["net.0.weight"]   # (h1, 5 + action_dim)
        old_b = old_sd["net.0.bias"]
        h1 = old_w.shape[0]
        new_w = torch.randn(h1, STATE_DIM + ACTION_DIM) * new_col_init_std
        # old layout: [state(5), action(2)] -> copy state cols 0:5 and action cols 5:7
        new_w[:, :STATE_DIM_OLD] = old_w[:, :STATE_DIM_OLD]
        new_w[:, STATE_DIM:STATE_DIM + ACTION_DIM] = old_w[:, STATE_DIM_OLD:STATE_DIM_OLD + ACTION_DIM]
        new_sd["net.0.weight"] = new_w
        new_sd["net.0.bias"] = old_b
        for key in old_sd:
            if key.startswith("net.0."):
                continue
            new_sd[key] = old_sd[key]
        new_critic.load_state_dict(new_sd)

    _copy_actor(old_ckpt["actor"], new_agent.actor)
    _copy_actor(old_ckpt["actor_target"], new_agent.actor_target)
    _copy_critic(old_ckpt["critic"], new_agent.critic)
    _copy_critic(old_ckpt["critic_target"], new_agent.critic_target)

    print(f"Transferred weights from 5-input checkpoint '{old_checkpoint_path}' "
          f"into a new 7-input agent (obstacle-avoidance columns freshly initialized, "
          f"std={new_col_init_std}).")
    return new_agent


# ----------------------------------------------------------------------
# 7. Training loop (multi-robot, batched)
# ----------------------------------------------------------------------

def train(ref_paths, obstacle_maps, episodes=500, output_dir=OUTPUT_DIR,
          agent=None, verbose_every=10, save_every=100, run_name="multi_robot"):
    os.makedirs(output_dir, exist_ok=True)
    n_robots = len(ref_paths)
    env = MultiRobotEnv(ref_paths, obstacle_maps)

    if agent is None:
        agent = DDPGAgent(state_dim=STATE_DIM)
    agent.init_noises(n_robots)

    episode_rewards = []       # mean reward across robots, per episode
    episode_success_rate = []  # fraction of robots that reached their goal

    for ep in range(1, episodes + 1):
        states, _ = env.reset()
        agent.reset_noises()
        ep_rewards = np.zeros(n_robots)
        all_done = False
        last_info = {}

        while not all_done:
            actions = agent.select_actions_batch(states, explore=True)
            next_states, rewards, step_done, all_done, info = env.step(actions)

            for i in range(n_robots):
                if info["done_mask"][i] and not env.done_mask[i]:
                    continue
                agent.replay.push(states[i], actions[i], rewards[i], next_states[i],
                                   float(info["done_mask"][i]))

            agent.train_step()
            states = next_states
            ep_rewards += rewards
            last_info = info

        agent.decay_noises()
        episode_rewards.append(float(np.mean(ep_rewards)))
        episode_success_rate.append(float(np.mean(last_info["reached_goal"])))

        if ep % verbose_every == 0:
            recent_r = episode_rewards[-verbose_every:]
            recent_s = episode_success_rate[-verbose_every:]
            print(f"Episode {ep:5d}/{episodes} | avg reward (last {verbose_every}) = {np.mean(recent_r):.3f} "
                  f"| goal-reach rate = {np.mean(recent_s):.2f} "
                  f"| static_collisions={last_info['collided_static'].sum()} "
                  f"| robot_collisions={last_info['collided_robot'].sum()}")

        if ep % save_every == 0 or ep == episodes:
            agent.save(os.path.join(output_dir, f"{run_name}_ep{ep}.pt"))

    plot_training(episode_rewards, episode_success_rate, output_dir, run_name)
    return agent, episode_rewards, episode_success_rate


def plot_training(episode_rewards, episode_success_rate, output_dir, run_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(episode_rewards)
    axes[0].set_title("Mean reward across robots per episode")
    axes[0].set_xlabel("episode")
    axes[0].grid(alpha=0.3)

    axes[1].plot(episode_success_rate)
    axes[1].set_title("Fraction of robots reaching their goal per episode")
    axes[1].set_xlabel("episode")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(output_dir, f"training_curve_{run_name}_{timestamp}.png")
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved training curve to {outpath}")


# ----------------------------------------------------------------------
# 8. Static rollout visualization (final trajectories of all robots)
# ----------------------------------------------------------------------

def plot_multi_robot_rollout(ref_paths, obstacle_maps, agent, output_dir=OUTPUT_DIR, run_name="eval"):
    n_robots = len(ref_paths)
    env = MultiRobotEnv(ref_paths, obstacle_maps)
    agent.init_noises(n_robots)
    states, _ = env.reset()
    trajectories = [[env.poses[i][:2].copy()] for i in range(n_robots)]

    all_done = False
    while not all_done:
        actions = agent.select_actions_batch(states, explore=False)
        states, rewards, step_done, all_done, info = env.step(actions)
        for i in range(n_robots):
            trajectories[i].append(env.poses[i][:2].copy())

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_robots))
    for i in range(n_robots):
        traj = np.array(trajectories[i])
        ax.plot(ref_paths[i].points[:, 0], ref_paths[i].points[:, 1], "--", color=colors[i], alpha=0.5,
                label=f"Robot {i} reference path")
        ax.plot(traj[:, 0], traj[:, 1], "-", color=colors[i], linewidth=2,
                label=f"Robot {i} actual trajectory")
        ax.plot(*traj[0], "o", color=colors[i], markersize=8)
        ax.plot(*traj[-1], "^", color=colors[i], markersize=10)
        for c, r in zip(obstacle_maps[i].centers, obstacle_maps[i].radii):
            circle = plt.Circle(c, r, color="gray", alpha=0.3)
            ax.add_patch(circle)

    ax.set_aspect("equal")
    ax.set_xlim(ARENA_MIN[0], ARENA_MAX[0])
    ax.set_ylim(ARENA_MIN[1], ARENA_MAX[1])
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=7)
    ax.set_title("Multi-robot rollout: path-following + static/dynamic obstacle avoidance")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(output_dir, f"multi_robot_rollout_{run_name}_{timestamp}.png")
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved multi-robot rollout plot to {outpath}")
    return outpath


# ----------------------------------------------------------------------
# 9. Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-robot batched DDPG: path-following + obstacle avoidance")
    parser.add_argument("--obstacle_maps", type=str, nargs="+", required=True,
                         help="List of map_00X_robot_Y.json files, one per robot")
    parser.add_argument("--path_jsons", type=str, nargs="+", required=True,
                         help="List of *_manual_control_points.json files, one per robot "
                              "(same order/length as --obstacle_maps)")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--run_name", type=str, default="multi_robot")
    parser.add_argument("--resume_from_5d", type=str, default=None,
                         help="Path to an OLD 5-input checkpoint to transfer-learn from "
                              "(e.g. ddpg_path_following_multi_curve.pt)")
    parser.add_argument("--resume_from_7d", type=str, default=None,
                         help="Path to an existing 7-input multi-robot checkpoint to resume from directly")
    args = parser.parse_args()

    if len(args.obstacle_maps) != len(args.path_jsons):
        raise ValueError("--obstacle_maps and --path_jsons must have the same length (one pair per robot)")

    ref_paths = [ReferencePath(p, map_name=f"robot{i}_{os.path.basename(p)}")
                 for i, p in enumerate(args.path_jsons)]
    obstacle_maps = [ObstacleMap(m) for m in args.obstacle_maps]
    print(f"Loaded {len(ref_paths)} robots: "
          + ", ".join(f"[{i}] path={os.path.basename(p)} obstacles={os.path.basename(m)}"
                       for i, (p, m) in enumerate(zip(args.path_jsons, args.obstacle_maps))))

    if args.resume_from_7d is not None:
        agent = DDPGAgent(state_dim=STATE_DIM)
        agent.load(args.resume_from_7d)
        print(f"Resumed 7-input agent from {args.resume_from_7d}")
    elif args.resume_from_5d is not None:
        agent = transfer_5_to_7(args.resume_from_5d)
    else:
        agent = None  # train() will create a fresh 7-input agent

    trained_agent, rewards, success_rates = train(
        ref_paths, obstacle_maps, episodes=args.episodes, output_dir=args.output_dir,
        agent=agent, run_name=args.run_name,
    )

    plot_multi_robot_rollout(ref_paths, obstacle_maps, trained_agent,
                              output_dir=args.output_dir, run_name=args.run_name)

# Example (3 robots, same map reused twice for illustration, transfer-learned):
# python multi_robot_ddpg.py --obstacle_maps maps/map_003_robot_1.json maps/map_003_robot_2.json --path_jsons solves/multi/map_003_robot_1_manual_control_points.json solves/multi/map_003_robot_2_manual_control_points.json  --resume_from_5d solves_drl/ddpg_path_following_multi_curve_60_20260720_133403.pt --episodes 500
