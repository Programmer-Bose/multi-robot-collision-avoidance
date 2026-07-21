"""
Single-Robot DDPG Path-Following + STATIC Obstacle Avoidance
===============================================================

Same kinematics/training pipeline as ran_ddpg_path_following.py, but the
environment now also loads a static obstacle map (circle obstacles from
map_00X_robot_Y.json, rescaled from 800x800 px into the 12x12 arena) and
the state is extended from 5 -> 7 inputs:

    state = [x_e, y_e, theta_e, prev_v, prev_w, obs_dist, obs_angle]

  - obs_dist  : signed surface distance to the NEAREST static obstacle
                (negative = inside/colliding), clipped to +/- OBSTACLE_SENSE_RADIUS
  - obs_angle : bearing to that obstacle relative to the robot's heading

Training starts from SCRATCH (no checkpoint / no transfer learning) on
ONE fixed manual reference path + its matching obstacle map, so you can
verify the robot learns to track that specific path while steering
around the static obstacles before scaling up to dynamic/multi-robot.

Usage
-----
    python single_robot_obstacle_ddpg.py \
        --control_points_json map_003_robot_2_manual_control_points.json \
        --obstacle_map_json map_003_robot_2.json \
        --episodes 1000
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
MAX_STEPS_PER_EPISODE = 800

INIT_POS_JITTER = 0.6
INIT_HEADING_JITTER = 0.6

GOAL_TOLERANCE = 0.1
OFF_PATH_TOLERANCE = 4.0

ARENA_MIN = np.array([0.0, 0.0])
ARENA_MAX = np.array([12.0, 12.0])

# original obstacle JSONs are laid out on an 800x800 canvas -> rescale
# into the 12x12 training arena
OBSTACLE_MAP_PX = 800.0
OBSTACLE_SCALE = (ARENA_MAX[0] - ARENA_MIN[0]) / OBSTACLE_MAP_PX  # = 0.015

W_PROGRESS = 2.0
W_OBSTACLE_SHAPING = 0.3       # continuous penalty for being close to an obstacle
OBSTACLE_SENSE_RADIUS = 3.0    # obs_dist clip/normalization range
COLLISION_PENALTY = -10.0
GOAL_REWARD = 10.0

STATE_DIM = 7           # [x_e, y_e, theta_e, prev_v, prev_w, obs_dist, obs_angle]
ACTION_DIM = 2
ACTOR_HIDDEN = (400, 300, 300)
CRITIC_HIDDEN = (400, 300, 300)

ACTOR_LR = 0.001
CRITIC_LR = 0.001
GAMMA = 0.9
TAU = 0.01
REPLAY_BUFFER_SIZE = 100_000
MIN_REPLAY_BEFORE_TRAINING = 2_000
BATCH_SIZE = 64
GRAD_CLIP_NORM = 5.0

OU_THETA = 0.15
OU_SIGMA = 0.3
OU_SIGMA_MIN = 0.05
OU_SIGMA_DECAY = 0.999

RENDER_PAUSE = 0.001
OUTPUT_DIR = "solves_drl_obstacle"


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


def make_robot_triangle(x, y, theta, size=0.2):
    local_pts = np.array([[size, 0.0], [-size * 0.6, size * 0.6], [-size * 0.6, -size * 0.6]])
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return local_pts @ rot.T + np.array([x, y])


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
# 2. Static obstacle map (circle obstacles, rescaled into the 12x12 arena)
# ----------------------------------------------------------------------

class ObstacleMap:
    def __init__(self, map_json_path):
        with open(map_json_path, "r") as f:
            data = json.load(f)
        centers, radii = [], []
        for obs in data.get("obstacles", []):
            if obs.get("type") != "circle":
                continue
            px, py = obs["position"]
            # obstacle JSONs are in image/pixel space (origin top-left,
            # y increases DOWNWARD). The reference paths / arena are in
            # standard math space (origin bottom-left, y increases
            # UPWARD), so y must be flipped before scaling or every
            # obstacle ends up mirrored vertically relative to the path.
            arena_x = px * OBSTACLE_SCALE
            arena_y = (OBSTACLE_MAP_PX - py) * OBSTACLE_SCALE
            centers.append([arena_x, arena_y])
            radii.append(obs["radius"] * OBSTACLE_SCALE)
        self.centers = np.asarray(centers, dtype=float) if centers else np.zeros((0, 2))
        self.radii = np.asarray(radii, dtype=float) if radii else np.zeros((0,))

    def nearest_distance_and_angle(self, xy, theta):
        """Signed surface distance (negative = inside obstacle) + bearing
        to the closest obstacle. Returns (OBSTACLE_SENSE_RADIUS, 0.0) if
        there are no obstacles."""
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


# ----------------------------------------------------------------------
# 3. Path-following + static-obstacle-avoidance environment
# ----------------------------------------------------------------------

class PathFollowObstacleEnv:
    def __init__(self, ref_path: ReferencePath, obstacle_map: ObstacleMap, render_mode=None):
        self.path = ref_path
        self.obstacles = obstacle_map
        self.pose = np.zeros(3)
        self.prev_action = np.zeros(2)
        self.steps = 0
        self.prev_arclength = 0.0

        self.render_mode = render_mode
        self._fig = None
        self._ax = None
        self._path_line = None
        self._trail_line = None
        self._robot_patch = None
        self._title_artist = None
        self._trail_xy = []

    def reset(self):
        idx = 0
        p_xy, p_heading = self.path.point_at_index(idx)
        jitter_xy = np.random.uniform(-INIT_POS_JITTER, INIT_POS_JITTER, size=2)
        jitter_theta = np.random.uniform(-INIT_HEADING_JITTER, INIT_HEADING_JITTER)
        self.pose = np.array([
            p_xy[0] + jitter_xy[0], p_xy[1] + jitter_xy[1],
            wrap_to_pi(p_heading + jitter_theta),
        ])
        self.prev_action = np.zeros(2)
        self.steps = 0
        self.prev_arclength = self.path.arclength_at_index(idx)
        self._trail_xy = [self.pose[:2].copy()]

        if self.render_mode == "human":
            self._render_init()
        return self._compute_state()

    def _compute_state(self):
        x, y, theta = self.pose
        idx = self.path.nearest_index(np.array([x, y]))
        p_xy, p_heading = self.path.point_at_index(idx)

        dx = p_xy[0] - x
        dy = p_xy[1] - y
        x_e = np.cos(theta) * dx + np.sin(theta) * dy
        y_e = -np.sin(theta) * dx + np.cos(theta) * dy
        theta_e = wrap_to_pi(p_heading - theta)

        obs_dist, obs_angle = self.obstacles.nearest_distance_and_angle(np.array([x, y]), theta)
        obs_dist_clipped = float(np.clip(obs_dist, -OBSTACLE_SENSE_RADIUS, OBSTACLE_SENSE_RADIUS))

        state = np.array([x_e, y_e, theta_e, self.prev_action[0], self.prev_action[1],
                           obs_dist_clipped, obs_angle], dtype=np.float32)
        info = {"nearest_idx": idx, "dist_to_path": np.hypot(dx, dy), "obs_dist": obs_dist}
        return state, info

    def step(self, action):
        v = float(np.clip(action[0], -MAX_LINEAR_VEL, MAX_LINEAR_VEL))
        w = float(np.clip(action[1], -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL))

        x, y, theta = self.pose
        x += v * np.cos(theta) * DT
        y += v * np.sin(theta) * DT
        theta = wrap_to_pi(theta + w * DT)
        self.pose = np.array([x, y, theta])
        self.prev_action = np.array([v, w])
        self.steps += 1
        self._trail_xy.append(self.pose[:2].copy())

        state, info = self._compute_state()
        x_e, y_e, theta_e = state[0], state[1], state[2]

        reward = -(abs(x_e) + abs(y_e) + abs(theta_e))

        current_arclength = self.path.arclength_at_index(info["nearest_idx"])
        progress = current_arclength - self.prev_arclength
        reward += W_PROGRESS * progress
        self.prev_arclength = current_arclength

        # continuous obstacle-avoidance shaping (penalize getting close)
        if info["obs_dist"] < OBSTACLE_SENSE_RADIUS:
            reward -= W_OBSTACLE_SHAPING * max(0.0, OBSTACLE_SENSE_RADIUS - info["obs_dist"]) / OBSTACLE_SENSE_RADIUS

        done = False
        success = False
        collided = False
        dist_to_goal = np.hypot(x - self.path.goal_point[0], y - self.path.goal_point[1])

        if info["obs_dist"] < 0:
            done = True
            collided = True
            reward += COLLISION_PENALTY
        elif info["nearest_idx"] >= len(self.path.points) - 2 and dist_to_goal < GOAL_TOLERANCE:
            done = True
            success = True
            reward += GOAL_REWARD
        elif x < ARENA_MIN[0] or x > ARENA_MAX[0] or y < ARENA_MIN[1] or y > ARENA_MAX[1]:
            done = True
            reward -= 10.0
        elif info["dist_to_path"] > OFF_PATH_TOLERANCE:
            done = True
            reward -= 10.0
        elif self.steps >= MAX_STEPS_PER_EPISODE:
            done = True

        info["success"] = success
        info["collided"] = collided
        info["longitudinal_error"] = x_e
        info["cross_error"] = y_e
        info["theta_error"] = theta_e
        info["progress"] = progress

        if self.render_mode == "human":
            self.render(reward=reward)

        return state, reward, done, info

    # ------------------------------------------------------------------
    def _render_init(self):
        if self._fig is not None:
            plt.close(self._fig)
        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(8, 6))
        self._path_line, = self._ax.plot(self.path.points[:, 0], self.path.points[:, 1], "b--",
                                          linewidth=1.5, label="Reference path", zorder=2)
        for c, r in zip(self.obstacles.centers, self.obstacles.radii):
            self._ax.add_patch(plt.Circle(c, r, color="gray", alpha=0.4, zorder=1))
        self._trail_line, = self._ax.plot([], [], "-", color="crimson", linewidth=2.0,
                                           label="Robot trail", zorder=3)
        self._robot_patch = plt.Polygon(make_robot_triangle(*self.pose), closed=True,
                                         color="crimson", zorder=4)
        self._ax.add_patch(self._robot_patch)
        self._ax.set_xlim(ARENA_MIN[0], ARENA_MAX[0])
        self._ax.set_ylim(ARENA_MIN[1], ARENA_MAX[1])
        self._ax.set_aspect("equal")
        self._ax.grid(alpha=0.3)
        self._ax.legend(loc="upper right")
        self._title_artist = self._ax.set_title(f"[{self.path.map_name}] reset")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(RENDER_PAUSE)

    def render(self, reward=None):
        if self._fig is None:
            self._render_init()
        trail = np.array(self._trail_xy)
        self._trail_line.set_data(trail[:, 0], trail[:, 1])
        self._robot_patch.set_xy(make_robot_triangle(*self.pose))
        label = f"[{self.path.map_name}] step {self.steps}"
        if reward is not None:
            label += f" | reward={reward:.3f}"
        self._title_artist.set_text(label)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(RENDER_PAUSE)

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None


# ----------------------------------------------------------------------
# 4. Noise / replay buffer / networks / DDPG agent (7-input)
# ----------------------------------------------------------------------

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


class DDPGAgent:
    def __init__(self):
        self.actor = Actor(STATE_DIM, ACTION_DIM, ACTOR_HIDDEN, MAX_LINEAR_VEL, MAX_ANGULAR_VEL).to(DEVICE)
        self.actor_target = Actor(STATE_DIM, ACTION_DIM, ACTOR_HIDDEN, MAX_LINEAR_VEL, MAX_ANGULAR_VEL).to(DEVICE)
        self.critic = Critic(STATE_DIM, ACTION_DIM, CRITIC_HIDDEN).to(DEVICE)
        self.critic_target = Critic(STATE_DIM, ACTION_DIM, CRITIC_HIDDEN).to(DEVICE)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.replay = ReplayBuffer()
        self.noise = OUNoise(ACTION_DIM)

    def select_action(self, state, explore=True):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        if explore:
            action = action + self.noise.sample() * np.array([MAX_LINEAR_VEL, MAX_ANGULAR_VEL])
        action[0] = np.clip(action[0], -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
        action[1] = np.clip(action[1], -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
        return action

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


def greedy_eval_rollout(agent, env, max_steps=MAX_STEPS_PER_EPISODE):
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    info = {}
    while not done:
        action = agent.select_action(state, explore=False)
        state, reward, done, info = env.step(action)
        total_reward += reward
    return {"success": bool(info.get("success", False)), "collided": bool(info.get("collided", False)),
            "steps": env.steps, "reward": total_reward}


# ----------------------------------------------------------------------
# 5. Training loop (single fixed path + obstacle map, from scratch)
# ----------------------------------------------------------------------

def train(ref_path, obstacle_map, episodes=1000, output_dir=OUTPUT_DIR, verbose_every=10,
          eval_every=25, render=False, render_every=1):
    os.makedirs(output_dir, exist_ok=True)
    agent = DDPGAgent()
    env = PathFollowObstacleEnv(ref_path, obstacle_map, render_mode=None)

    episode_rewards, episode_success, episode_collision = [], [], []
    eval_log = []

    for ep in range(1, episodes + 1):
        env.render_mode = "human" if (render and (ep - 1) % render_every == 0) else None
        state, _ = env.reset()
        agent.noise.reset()
        ep_reward = 0.0
        done = False
        info = {}

        while not done:
            action = agent.select_action(state, explore=True)
            next_state, reward, done, info = env.step(action)
            agent.replay.push(state, action, reward, next_state, float(done))
            agent.train_step()
            state = next_state
            ep_reward += reward

        agent.noise.decay_sigma()
        episode_rewards.append(ep_reward)
        episode_success.append(bool(info.get("success", False)))
        episode_collision.append(bool(info.get("collided", False)))

        if ep % verbose_every == 0:
            recent_r = episode_rewards[-verbose_every:]
            recent_s = episode_success[-verbose_every:]
            recent_c = episode_collision[-verbose_every:]
            print(f"Episode {ep:5d}/{episodes} | avg reward = {np.mean(recent_r):.3f} | "
                  f"success rate = {np.mean(recent_s):.2f} | collision rate = {np.mean(recent_c):.2f} | "
                  f"noise sigma = {agent.noise.sigma:.3f}")

        if ep % eval_every == 0 or ep == episodes:
            greedy = greedy_eval_rollout(agent, env)
            eval_log.append({"episode": ep, **greedy})
            print(f"  [greedy eval] success={greedy['success']} collided={greedy['collided']} "
                  f"steps={greedy['steps']} reward={greedy['reward']:.3f}")

    env.close()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    agent.save(os.path.join(output_dir, f"ddpg_single_robot_obstacle_{episodes}_{timestamp}.pt"))
    plot_training_curve(episode_rewards, episode_success, episode_collision, eval_log, output_dir)
    return agent, episode_rewards, episode_success, episode_collision, eval_log


def plot_training_curve(episode_rewards, episode_success, episode_collision, eval_log, output_dir):
    window = 20
    smoothed = (np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
                if len(episode_rewards) >= window else episode_rewards)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(episode_rewards, alpha=0.3, label="Episode reward")
    axes[0].plot(range(window - 1, window - 1 + len(smoothed)), smoothed, linewidth=2, label=f"{window}-ep avg")
    axes[0].set_title("Training reward")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    window_sc = min(20, len(episode_success)) or 1
    success_ma = np.convolve(episode_success, np.ones(window_sc) / window_sc, mode="valid")
    collision_ma = np.convolve(episode_collision, np.ones(window_sc) / window_sc, mode="valid")
    axes[1].plot(success_ma, label="success rate (moving avg)", color="green")
    axes[1].plot(collision_ma, label="collision rate (moving avg)", color="red")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Success / collision rate")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(output_dir, f"training_curve_single_obstacle_{timestamp}.png")
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved training curve to {outpath}")


def plot_final_rollout(agent, ref_path, obstacle_map, output_dir=OUTPUT_DIR):
    env = PathFollowObstacleEnv(ref_path, obstacle_map, render_mode=None)
    state, _ = env.reset()
    trajectory = [env.pose[:2].copy()]
    done = False
    info = {}
    while not done:
        action = agent.select_action(state, explore=False)
        state, reward, done, info = env.step(action)
        trajectory.append(env.pose[:2].copy())
    trajectory = np.array(trajectory)
    env.close()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(ref_path.points[:, 0], ref_path.points[:, 1], "b--", linewidth=2, label="Reference path")
    ax.plot(trajectory[:, 0], trajectory[:, 1], "-", color="crimson", linewidth=2.5, label="Actual trajectory")
    ax.plot(*trajectory[0], "go", markersize=10, label="Start")
    ax.plot(*trajectory[-1], "r^", markersize=11, label="End")
    for c, r in zip(obstacle_map.centers, obstacle_map.radii):
        ax.add_patch(plt.Circle(c, r, color="gray", alpha=0.4))
    ax.set_aspect("equal")
    ax.set_xlim(ARENA_MIN[0], ARENA_MAX[0])
    ax.set_ylim(ARENA_MIN[1], ARENA_MAX[1])
    ax.grid(alpha=0.3)
    status = "SUCCESS" if info.get("success") else ("COLLISION" if info.get("collided") else "INCOMPLETE")
    ax.set_title(f"[{ref_path.map_name}] final rollout - {status}")
    ax.legend(loc="best")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(output_dir, f"final_rollout_{ref_path.map_name}_{timestamp}.png")
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved final rollout plot to {outpath}")
    return outpath


# ----------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-robot DDPG: path-following + static obstacle avoidance (7-input, from scratch)")
    parser.add_argument("--control_points_json", type=str, required=True,
                         help="Manual per-segment B-spline control-point JSON (e.g. map_003_robot_2_manual_control_points.json)")
    parser.add_argument("--obstacle_map_json", type=str, required=True,
                         help="Matching obstacle-map JSON (e.g. map_003_robot_2.json)")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--render", action="store_true", help="Live matplotlib rendering during training")
    parser.add_argument("--render_every", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=25)
    args = parser.parse_args()

    ref_path = ReferencePath(args.control_points_json)
    obstacle_map = ObstacleMap(args.obstacle_map_json)
    print(f"Loaded path '{ref_path.map_name}' ({len(ref_path.points)} pts, length={ref_path.total_length:.2f}) "
          f"and {len(obstacle_map.centers)} static obstacles.")

    print(f"Using device: {DEVICE}")
    agent, rewards, success, collisions, eval_log = train(
        ref_path, obstacle_map, episodes=args.episodes, output_dir=args.output_dir,
        render=args.render, render_every=args.render_every, eval_every=args.eval_every,
    )

    plot_final_rollout(agent, ref_path, obstacle_map, output_dir=args.output_dir)

# python single_robot_obstacle_ddpg.py --control_points_json map_003_robot_2_manual_control_points.json --obstacle_map_json map_003_robot_2.json --episodes 1000