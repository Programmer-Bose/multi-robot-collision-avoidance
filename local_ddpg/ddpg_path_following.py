"""
DDPG-based Path-Following Controller for a Nonholonomic Wheeled Mobile Robot
=============================================================================

Single-file implementation, phase 1 of the DRL roadmap: PATH-FOLLOWING ONLY,
no static or dynamic obstacles. Follows the technique of:

    Cheng, X.; Zhang, S.; Cheng, S.; Xia, Q.; Zhang, J.
    "Path-Following and Obstacle Avoidance Control of Nonholonomic Wheeled
    Mobile Robot Based on Deep Reinforcement Learning."
    Appl. Sci. 2022, 12, 6874.

Reference path is built directly from a manually-authored per-segment
B-spline control-point JSON file (the same format produced by
`export_all_segments_control_points` in dde_mul_la.py), e.g.
map_001_robot_2_manual_control_points.json.

Implements:
  - Unicycle kinematics (paper Eq. 1)
  - Path-following tracking error state s_pf = [x_e, y_e, theta_e, v, w]  (Eq. 3, 12)
  - Basic path-following reward r_eb = -(|x_e| + |y_e| + |theta_e|)      (Eq. 18)
  - DDPG actor/critic (Table 1 architecture: 5->400->300->300->{2,1}, ReLU/Tanh)
  - Ornstein-Uhlenbeck exploration noise
  - Experience replay + target networks with soft update tau            (Eq. 9-11)
  - Randomized initial pose near the path per training episode           (Eq. 21)

No obstacles (static or dynamic) are modeled in this phase.
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

# --- B-spline reconstruction (must match the control-point JSON format) ---
BSPLINE_DEGREE = 3
N_SAMPLES_PER_SEGMENT = 80          # denser than the DE solver since this is now our "ground truth" reference

# --- Kinematics / control ---
DT = 0.1                            # control time step (s)
MAX_LINEAR_VEL = 1.5                # m/s   (scaled-arena units per second)
MAX_ANGULAR_VEL = np.pi / 2         # rad/s
MAX_STEPS_PER_EPISODE = 800

# --- Episode initialization (paper Eq. 21: random offset around a path point) ---
INIT_POS_JITTER = 0.6               # +/- meters (world units) around chosen path point
INIT_HEADING_JITTER = 0.6           # +/- radians around path tangent heading

# --- Termination thresholds ---
GOAL_TOLERANCE = 0.3                 # reaching the final path point ends episode successfully
OFF_PATH_TOLERANCE = 2.5             # episode fails if the robot drifts this far from the path

# --- Forward-progress shaping ---
# The base path-following reward r_eb = -(|x_e|+|y_e|+|theta_e|) only
# penalizes tracking error; it never rewards actually moving forward along
# the path. Left unshaped, "stand still exactly on the path" is a valid
# (often preferred) optimum for DDPG since it guarantees ~0 error forever.
# This term rewards forward arc-length progress each step, which removes
# that degenerate solution and is what actually makes the robot move.
W_PROGRESS = 2.0

# --- Rendering ---
RENDER_PAUSE = 0.001                 # seconds paused per redraw (flushes the GUI event loop)

# --- DDPG network ---
STATE_DIM = 5                        # [x_e, y_e, theta_e, v, w]   (paper's S_pf, Eq. 12)
ACTION_DIM = 2                       # [v, w]
ACTOR_HIDDEN = (400, 300, 300)
CRITIC_HIDDEN = (400, 300, 300)

# --- DDPG training ---
ACTOR_LR = 0.001
CRITIC_LR = 0.01
GAMMA = 0.9
TAU = 0.01
REPLAY_BUFFER_SIZE = 100_000
MIN_REPLAY_BEFORE_TRAINING = 2_000
BATCH_SIZE = 256
MAX_EPISODES = 100
GRAD_CLIP_NORM = 5.0

# --- Ornstein-Uhlenbeck exploration noise ---
OU_THETA = 0.15
OU_SIGMA = 0.3
OU_SIGMA_MIN = 0.05
OU_SIGMA_DECAY = 0.999    # multiplied once per episode

OUTPUT_DIR = "solves_drl"

ARENA_MIN = np.array([0.0, 0.0])
ARENA_MAX = np.array([12.0, 12.0])

# ----------------------------------------------------------------------
# 1. Reference path: rebuild B-spline segments from manual control points
# ----------------------------------------------------------------------

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


class ReferencePath:
    """Loads a manual per-segment control-point JSON file (as produced by
    export_all_segments_control_points in dde_mul_la.py) and rebuilds the
    full sampled B-spline path, exposing nearest-point-on-path queries used
    to compute the path-following tracking error (Eq. 3)."""

    def __init__(self, control_points_json):
        with open(control_points_json, "r") as f:
            data = json.load(f)

        self.map_name = data.get("map_name", "unnamed")
        samples = []
        for seg in data["segments"]:
            start = np.asarray(seg["start_point"], dtype=float)
            end = np.asarray(seg["end_point"], dtype=float)
            free_pts = np.asarray(seg["control_points"], dtype=float)
            full_ctrl = np.vstack([start, free_pts, end])
            curve = bspline_curve(full_ctrl, N_SAMPLES_PER_SEGMENT)
            # avoid duplicating the shared point between consecutive segments
            if samples:
                curve = curve[1:]
            samples.append(curve)

        self.points = np.vstack(samples)                      # (M, 2)
        diffs = np.diff(self.points, axis=0)
        seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        self.cum_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        self.total_length = self.cum_length[-1]

        # per-sample tangent heading (central difference, clamped at ends)
        headings = np.zeros(len(self.points))
        headings[0] = np.arctan2(diffs[0, 1], diffs[0, 0])
        headings[-1] = np.arctan2(diffs[-1, 1], diffs[-1, 0])
        for i in range(1, len(self.points) - 1):
            v = self.points[i + 1] - self.points[i - 1]
            headings[i] = np.arctan2(v[1], v[0])
        self.headings = headings

    def nearest_index(self, xy):
        d2 = np.sum((self.points - xy[None, :]) ** 2, axis=1)
        return int(np.argmin(d2))

    def point_at_index(self, idx):
        return self.points[idx], self.headings[idx]

    def arclength_at_index(self, idx):
        return self.cum_length[idx]

    def random_reference_index(self, min_frac=0.0, max_frac=0.9):
        """Pick a random point along the path (used for random episode
        initialization, paper Eq. 21). max_frac < 1 keeps some remaining
        path ahead of the robot so an episode has something to follow."""
        lo = int(min_frac * (len(self.points) - 1))
        hi = max(lo + 1, int(max_frac * (len(self.points) - 1)))
        return random.randint(lo, hi)

    @property
    def goal_point(self):
        return self.points[-1]


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


ROBOT_MARKER_SIZE = 0.35


def make_robot_triangle(x, y, theta, size=ROBOT_MARKER_SIZE):
    """Vertices of a small triangle oriented along `theta`, centered at
    (x, y). Used by both live rendering and inference-time animation."""
    local_pts = np.array([
        [size, 0.0],
        [-size * 0.6, size * 0.6],
        [-size * 0.6, -size * 0.6],
    ])
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return local_pts @ rot.T + np.array([x, y])


# ----------------------------------------------------------------------
# 2. Path-following environment (Gym-style reset/step), NO obstacles
# ----------------------------------------------------------------------

class PathFollowEnv:
    """Unicycle kinematics (Eq. 1) driven by continuous action [v, w].
    State returned each step is the path-following tracking-error vector
    s_pf = [x_e, y_e, theta_e, v, w] (Eq. 3, 12); reward is the basic
    path-following reward r_eb = -(|x_e| + |y_e| + |theta_e|) (Eq. 18)
    PLUS a forward arc-length progress term (see W_PROGRESS above) that
    prevents "stand still on the path" from being a degenerate optimum.

    Call render() after step()/reset() to visualize the robot live in a
    persistent matplotlib window (standard practice for RL envs, mirroring
    e.g. Gym's render_mode="human"). Call close() when done rendering."""

    def __init__(self, ref_path: ReferencePath, render_mode=None):
        self.path = ref_path
        self.pose = np.zeros(3)   # [x, y, theta]
        self.prev_action = np.zeros(2)
        self.steps = 0
        self.prev_arclength = 0.0

        self.render_mode = render_mode   # None or "human"
        self._fig = None
        self._ax = None
        self._path_line = None
        self._trail_line = None
        self._robot_patch = None
        self._title_artist = None
        self._trail_xy = []

    def reset(self):
        # idx = self.path.random_reference_index()
        idx = 0
        p_xy, p_heading = self.path.point_at_index(idx)

        jitter_xy = np.random.uniform(-INIT_POS_JITTER, INIT_POS_JITTER, size=2)
        jitter_theta = np.random.uniform(-INIT_HEADING_JITTER, INIT_HEADING_JITTER)

        self.pose = np.array([
            p_xy[0] + jitter_xy[0],
            p_xy[1] + jitter_xy[1],
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

        # tracking error transform into the robot body frame (Eq. 3)
        x_e = np.cos(theta) * dx + np.sin(theta) * dy
        y_e = -np.sin(theta) * dx + np.cos(theta) * dy
        theta_e = wrap_to_pi(p_heading - theta)

        state = np.array([x_e, y_e, theta_e, self.prev_action[0], self.prev_action[1]], dtype=np.float32)
        info = {"nearest_idx": idx, "dist_to_path": np.hypot(dx, dy)}
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

        reward = -(abs(x_e) + abs(y_e) + abs(theta_e))   # r_eb, Eq. 18

        # forward-progress shaping: reward advancing along the path's arc
        # length, penalize regressing backward. Without this term, standing
        # still exactly on the path is a valid near-zero-error optimum and
        # the policy learns not to move at all.
        current_arclength = self.path.arclength_at_index(info["nearest_idx"])
        progress = current_arclength - self.prev_arclength
        reward += W_PROGRESS * progress
        self.prev_arclength = current_arclength

        done = False
        success = False
        dist_to_goal = np.hypot(x - self.path.goal_point[0], y - self.path.goal_point[1])
        if info["nearest_idx"] >= len(self.path.points) - 2 and dist_to_goal < GOAL_TOLERANCE:
            done = True
            success = True
            reward += 10.0   # terminal bonus for reaching the end of a finite path
        elif info["dist_to_path"] > OFF_PATH_TOLERANCE:
            done = True
            reward -= 10.0   # terminal penalty for losing the path
        elif self.steps >= MAX_STEPS_PER_EPISODE:
            done = True
        elif x < ARENA_MIN[0] or x > ARENA_MAX[0] or y < ARENA_MIN[1] or y > ARENA_MAX[1]:
            done = True
            reward -= 10.0   # terminal penalty for leaving the arena

        info["success"] = success
        info["longitudinal_error"] = x_e
        info["cross_error"] = y_e
        info["theta_error"] = theta_e
        info["progress"] = progress

        if self.render_mode == "human":
            self.render(reward=reward)

        return state, reward, done, info

    # ------------------------------------------------------------------
    # Rendering (standard RL env practice: a render() you can call any
    # time after reset()/step() to visually confirm the agent is actually
    # moving, independent of reward-curve numbers).
    # ------------------------------------------------------------------

    def _render_init(self):
        if self._fig is not None:
            plt.close(self._fig)

        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(8, 6))
        self._path_line, = self._ax.plot(
            self.path.points[:, 0], self.path.points[:, 1], "b--", linewidth=1.5,
            label="Reference path", zorder=2,
        )
        self._trail_line, = self._ax.plot([], [], "-", color="crimson", linewidth=2.0,
                                           label="Robot trail", zorder=3)
        self._robot_patch = plt.Polygon(
            make_robot_triangle(self.pose[0], self.pose[1], self.pose[2]),
            closed=True, color="crimson", zorder=4,
        )
        self._ax.add_patch(self._robot_patch)

        margin = 1.0
        self._ax.set_xlim(self.path.points[:, 0].min() - margin, self.path.points[:, 0].max() + margin)
        self._ax.set_ylim(self.path.points[:, 1].min() - margin, self.path.points[:, 1].max() + margin)
        self._ax.set_aspect("equal")
        self._ax.grid(alpha=0.3)
        self._ax.legend(loc="upper right")
        self._title_artist = self._ax.set_title("Path-Following env - reset")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(RENDER_PAUSE)

    def render(self, reward=None):
        """Draw the current robot pose + trail on top of the reference
        path. Safe to call every step (human mode) or only occasionally."""
        if self._fig is None:
            self._render_init()

        trail = np.array(self._trail_xy)
        self._trail_line.set_data(trail[:, 0], trail[:, 1])
        self._robot_patch.set_xy(make_robot_triangle(self.pose[0], self.pose[1], self.pose[2]))

        label = f"Path-Following env - step {self.steps}"
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
# 3. Ornstein-Uhlenbeck exploration noise
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


# ----------------------------------------------------------------------
# 4. Replay buffer
# ----------------------------------------------------------------------

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
# 5. Actor / Critic networks (Table 1: 5 -> 400 -> 300 -> 300 -> {2, 1})
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
        raw = self.net(state)
        return raw * self.action_scale


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


# ----------------------------------------------------------------------
# 6. DDPG agent
# ----------------------------------------------------------------------

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

        # --- Critic update (Eq. 9-10) ---
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

        # --- Actor update (Eq. 11, deterministic policy gradient) ---
        actor_loss = -self.critic(s, self.actor(s)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP_NORM)
        self.actor_optim.step()

        # --- Target network soft update (Eq. 12-13 in Algorithm 1) ---
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


# ----------------------------------------------------------------------
# 7. Training loop (Algorithm 1)
# ----------------------------------------------------------------------

def train(control_points_json, max_episodes=MAX_EPISODES, output_dir=OUTPUT_DIR, verbose_every=10,
          render=False, render_every=1, resume_from=None):
    """render: if True, opens a live matplotlib window and calls env.render()
    after every step so you can visually confirm the robot is actually
    moving during training (not just watch the reward number go up).
    render_every: only render episodes where (episode - 1) % render_every == 0,
    to avoid slowing down training too much when watching every episode."""
    os.makedirs(output_dir, exist_ok=True)

    ref_path = ReferencePath(control_points_json)
    env = PathFollowEnv(ref_path, render_mode=None)
    agent = DDPGAgent()
    if resume_from is not None:
        agent.load(resume_from)
        print(f"Resumed training from checkpoint: {resume_from}")

    episode_rewards = []
    episode_success = []

    for episode in range(1, max_episodes + 1):
        episode_should_render = render and ((episode - 1) % render_every == 0)
        env.render_mode = "human" if episode_should_render else None

        state, _ = env.reset()
        agent.noise.reset()
        episode_reward = 0.0
        done = False
        info = {}

        while not done:
            action = agent.select_action(state, explore=True)
            next_state, reward, done, info = env.step(action)
            agent.replay.push(state, action, reward, next_state, float(done))
            agent.train_step()

            state = next_state
            episode_reward += reward

        agent.noise.decay_sigma()
        episode_rewards.append(episode_reward)
        episode_success.append(bool(info.get("success", False)))

        if episode % verbose_every == 0:
            recent = episode_rewards[-verbose_every:]
            recent_success = episode_success[-verbose_every:]
            print(f"Episode {episode:4d}/{max_episodes} | "
                  f"avg reward (last {verbose_every}) = {np.mean(recent):.3f} | "
                  f"success rate = {np.mean(recent_success):.2f} | "
                  f"noise sigma = {agent.noise.sigma:.3f}")

    env.close()
    agent.save(os.path.join(output_dir, f"ddpg_path_following_{700+max_episodes}.pt"))
    plot_training_curve(episode_rewards, ref_path.map_name, output_dir)
    return agent, ref_path, episode_rewards


def plot_training_curve(episode_rewards, map_name, output_dir):
    window = 20
    if len(episode_rewards) >= window:
        smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
    else:
        smoothed = episode_rewards

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(episode_rewards, alpha=0.3, label="Episode reward")
    ax.plot(range(window - 1, window - 1 + len(smoothed)), smoothed, linewidth=2, label=f"{window}-episode moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(f"DDPG Path-Following Training Reward [{map_name}]")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(output_dir, f"training_reward_{map_name}_{timestamp}.png")
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved training reward curve to {outpath}")


# ----------------------------------------------------------------------
# 8. Evaluation / rollout visualization
# ----------------------------------------------------------------------

def evaluate(agent, ref_path, output_dir=OUTPUT_DIR, start_frac=0.0, render=False):
    env = PathFollowEnv(ref_path, render_mode="human" if render else None)

    idx = int(start_frac * (len(ref_path.points) - 1))
    p_xy, p_heading = ref_path.point_at_index(idx)
    env.pose = np.array([p_xy[0], p_xy[1], p_heading])
    env.prev_action = np.zeros(2)
    env.steps = 0
    env.prev_arclength = ref_path.arclength_at_index(idx)
    env._trail_xy = [env.pose[:2].copy()]
    if render:
        env._render_init()
    state, _ = env._compute_state()

    trajectory = [env.pose[:2].copy()]
    longitudinal_errors = []
    cross_errors = []
    theta_errors = []
    done = False

    while not done and env.steps < MAX_STEPS_PER_EPISODE:
        action = agent.select_action(state, explore=False)
        state, reward, done, info = env.step(action)
        trajectory.append(env.pose[:2].copy())
        longitudinal_errors.append(info["longitudinal_error"])
        cross_errors.append(info["cross_error"])
        theta_errors.append(info["theta_error"])

    trajectory = np.array(trajectory)
    env.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(ref_path.points[:, 0], ref_path.points[:, 1], "b--", linewidth=1.5, label="Reference path")
    axes[0].plot(trajectory[:, 0], trajectory[:, 1], "r-", linewidth=2, label="DDPG rollout")
    axes[0].plot(*trajectory[0], "go", markersize=10, label="Start")
    axes[0].set_aspect("equal")
    axes[0].set_title(f"Path-Following Rollout [{ref_path.map_name}]")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(longitudinal_errors, label="longitudinal error (x_e)")
    axes[1].plot(cross_errors, label="cross error (y_e)")
    axes[1].plot(theta_errors, label="theta error")
    axes[1].set_xlabel("step")
    axes[1].set_title("Tracking errors")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(output_dir, f"eval_rollout_{ref_path.map_name}_{timestamp}.png")
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved evaluation rollout to {outpath}")

    mse_long = float(np.mean(np.square(longitudinal_errors))) if longitudinal_errors else float("nan")
    mse_cross = float(np.mean(np.square(cross_errors))) if cross_errors else float("nan")
    print(f"MSE longitudinal error: {mse_long:.4f} | MSE cross error: {mse_cross:.4f}")
    return trajectory, mse_long, mse_cross


# ----------------------------------------------------------------------
# 9. Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPG path-following (no obstacles) - phase 1")
    parser.add_argument("--control_points_json", type=str,
                         default="solves/multi/map_001_robot_2_manual_control_points.json",
                         help="Manual per-segment B-spline control-point JSON file")
    parser.add_argument("--episodes", type=int, default=MAX_EPISODES)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--render", action="store_true",
                         help="Open a live matplotlib window during training so you can watch the robot move")
    parser.add_argument("--render_every", type=int, default=1,
                         help="Render every Nth episode during training (default: every episode)")
    parser.add_argument("--render_eval", action="store_true",
                         help="Render the final post-training evaluation rollout live")
    parser.add_argument("--resume_from", type=str, default=None,
                         help="Path to a .pt checkpoint to resume training from")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    trained_agent, path, rewards = train(
        args.control_points_json,
        max_episodes=args.episodes,
        output_dir=args.output_dir,
        render=args.render,
        render_every=args.render_every,
        resume_from=args.resume_from
    )
    evaluate(trained_agent, path, output_dir=args.output_dir, start_frac=0.0, render=args.render_eval)

    # python ddpg_path_following.py --control_points_json solves/multi/map_001_robot_2_manual_control_points.json --resume_from .\solves_drl\ddpg_path_following_700.pt --episodes 300