"""
DDPG-based Path-Following Controller for a Nonholonomic Wheeled Mobile Robot
=============================================================================

Phase 1b: PATH-FOLLOWING ONLY, no obstacles, but trained with a CURRICULUM
of procedurally-generated random reference paths instead of a single fixed
manual map, so the policy generalizes across path shapes rather than
memorizing one map.

Six random path families are generated, each confined to the 12x12 arena:
  1. sine_wave        - straight baseline + sinusoidal lateral offset
  2. cosine_s_curve   - single half-period cosine hump (S-curve)
  3. rounded_rect      - superellipse-approximated rounded-rectangle arc
  4. zigzag            - triangular/sawtooth lateral offset
  5. circular_arc      - arc of random radius/center/angular span
  6. random_bspline    - random interior control points through a B-spline
                         (same construction as dde_mul_la.py, but randomized
                         instead of DE-optimized)

Training proceeds in ROUNDS: each round samples one random path (cycling
through the 6 families), and trains on it for at least EPISODES_PER_PATH
episodes before moving to the next randomly generated path.

Still follows the base DDPG technique of:
    Cheng, X. et al. "Path-Following and Obstacle Avoidance Control of
    Nonholonomic Wheeled Mobile Robot Based on Deep Reinforcement Learning."
    Appl. Sci. 2022, 12, 6874.

Manual per-segment control-point JSON files (the DE-solver output format)
can still be loaded via ReferencePath(control_points_json=...) for later
fine-tuning / final evaluation against real maps.
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
N_SAMPLES_PER_SEGMENT = 80

# --- Kinematics / control ---
DT = 0.1
MAX_LINEAR_VEL = 1.5
MAX_ANGULAR_VEL = np.pi / 2
MAX_STEPS_PER_EPISODE = 400

# --- Episode initialization ---
# Per your requirement, episodes always start at the actual path start
# (index 0), not a random point along the path.
INIT_POS_JITTER = 0.6
INIT_HEADING_JITTER = 0.6

# --- Termination thresholds ---
GOAL_TOLERANCE = 0.3
OFF_PATH_TOLERANCE = 4.0

# --- Arena bounds (episode ends if the robot leaves this box) ---
ARENA_MIN = np.array([0.0, 0.0])
ARENA_MAX = np.array([12.0, 12.0])

# --- Forward-progress shaping ---
# The base path-following reward r_eb = -(|x_e|+|y_e|+|theta_e|) only
# penalizes tracking error; it never rewards actually moving forward along
# the path. This term rewards forward arc-length progress each step, which
# removes the "stand still" degenerate optimum.
W_PROGRESS = 2.0

# --- Rendering ---
RENDER_PAUSE = 0.001

# --- DDPG network ---
STATE_DIM = 5
ACTION_DIM = 2
ACTOR_HIDDEN = (400, 300, 300)
CRITIC_HIDDEN = (400, 300, 300)

# --- DDPG training ---
ACTOR_LR = 0.001
CRITIC_LR = 0.001   # was 0.01 - too fast relative to actor, caused critic
                    # extrapolation error that collapsed the actor to ~0 output
GAMMA = 0.9
TAU = 0.01
REPLAY_BUFFER_SIZE = 100_000
MIN_REPLAY_BEFORE_TRAINING = 2_000
BATCH_SIZE = 64
GRAD_CLIP_NORM = 5.0

# --- Ornstein-Uhlenbeck exploration noise ---
OU_THETA = 0.15
OU_SIGMA = 0.3
OU_SIGMA_MIN = 0.05
OU_SIGMA_DECAY = 0.999   # multiplied once per episode

# --- Random path curriculum ---
DEFAULT_CURVE_TYPES = [
    "sine_wave",
    "cosine_s_curve",
    "rounded_rect",
    "zigzag",
    "circular_arc",
    "random_bspline",
]
N_PATH_SAMPLES = 250          # dense samples per generated random path
EPISODES_PER_PATH = 10        # minimum episodes trained per generated path
MAX_ROUNDS = 60                # number of random-path rounds (>=10 ep each)
ARENA_MARGIN = 1.5             # keep generated paths comfortably inside the arena
PATH_GENERATION_MAX_RETRIES = 8

# --- Periodic noise-free evaluation during training ---
EVAL_EVERY_ROUNDS = 5   # run a greedy (explore=False) check every N rounds

OUTPUT_DIR = "solves_drl"


def greedy_eval_rollout(agent, ref_path, max_steps=MAX_STEPS_PER_EPISODE):
    """Runs the policy with explore=False (no OU noise) and returns whether
    it actually reaches the goal on its own. This is the number that
    reflects what inference will look like - unlike the noisy training
    success rate, which can look good purely from exploration noise
    randomly carrying the robot toward the goal while the deterministic
    policy itself has collapsed (e.g. to near-zero actions)."""
    env = PathFollowEnv(ref_path)
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    info = {}
    first_action_mag = None

    while not done:
        action = agent.select_action(state, explore=False)
        if first_action_mag is None:
            first_action_mag = float(np.hypot(action[0], action[1]))
        state, reward, done, info = env.step(action)
        total_reward += reward

    return {
        "success": bool(info.get("success", False)),
        "steps": env.steps,
        "reward": total_reward,
        "first_action_mag": first_action_mag,
    }


# ----------------------------------------------------------------------
# 1. Shared math helpers
# ----------------------------------------------------------------------

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


ROBOT_MARKER_SIZE = 0.35


def make_robot_triangle(x, y, theta, size=ROBOT_MARKER_SIZE):
    """Vertices of a small triangle oriented along `theta`, centered at
    (x, y). Used by both live rendering and inference-time plotting."""
    local_pts = np.array([
        [size, 0.0],
        [-size * 0.6, size * 0.6],
        [-size * 0.6, -size * 0.6],
    ])
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return local_pts @ rot.T + np.array([x, y])


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


def _rotate(points, angle):
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    return points @ rot.T


def _clip_to_arena(points):
    out = points.copy()
    out[:, 0] = np.clip(out[:, 0], ARENA_MIN[0] + 0.05, ARENA_MAX[0] - 0.05)
    out[:, 1] = np.clip(out[:, 1], ARENA_MIN[1] + 0.05, ARENA_MAX[1] - 0.05)
    return out


def _path_length(points):
    d = np.diff(points, axis=0)
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))


def _random_center(rng, half_extent):
    lo = ARENA_MIN + ARENA_MARGIN
    hi = ARENA_MAX - ARENA_MARGIN
    lo = np.maximum(lo, ARENA_MIN + half_extent)
    hi = np.minimum(hi, ARENA_MAX - half_extent)
    lo = np.minimum(lo, hi)  # guard against inverted bounds for large half_extent
    return rng.uniform(lo, hi)


# ----------------------------------------------------------------------
# 2. Random path generators (each returns an (M, 2) sampled polyline)
# ----------------------------------------------------------------------

def gen_sine_wave(rng, n_samples=N_PATH_SAMPLES):
    length = rng.uniform(5.0, 8.0)
    amplitude = rng.uniform(0.4, 1.6)
    n_cycles = rng.uniform(1.0, 2.5)

    s = np.linspace(0.0, length, n_samples)
    lateral = amplitude * np.sin(2 * np.pi * n_cycles * s / length)
    local_pts = np.column_stack([s, lateral])

    angle = rng.uniform(0, 2 * np.pi)
    rotated = _rotate(local_pts, angle)
    half_extent = np.array([length / 2 + amplitude, length / 2 + amplitude])
    center = _random_center(rng, half_extent)
    points = rotated - rotated.mean(axis=0) + center
    return _clip_to_arena(points)


def gen_cosine_s_curve(rng, n_samples=N_PATH_SAMPLES):
    length = rng.uniform(5.0, 8.0)
    amplitude = rng.uniform(0.6, 2.0)

    s = np.linspace(0.0, length, n_samples)
    # single half-period hump: 0 -> amplitude -> 0 (smooth S when combined
    # with forward travel)
    lateral = amplitude * (1 - np.cos(2 * np.pi * s / length)) / 2.0
    local_pts = np.column_stack([s, lateral])

    angle = rng.uniform(0, 2 * np.pi)
    rotated = _rotate(local_pts, angle)
    half_extent = np.array([length / 2 + amplitude, length / 2 + amplitude])
    center = _random_center(rng, half_extent)
    points = rotated - rotated.mean(axis=0) + center
    return _clip_to_arena(points)


def gen_rounded_rect(rng, n_samples=N_PATH_SAMPLES):
    a = rng.uniform(2.0, 4.0)   # semi-axis x
    b = rng.uniform(2.0, 4.0)   # semi-axis y
    n_exp = 8.0                  # superellipse exponent -> rounded-rectangle look

    span_frac = rng.uniform(0.5, 0.95)   # open arc, not a full closed loop
    t0 = rng.uniform(0, 2 * np.pi)
    t = np.linspace(t0, t0 + span_frac * 2 * np.pi, n_samples)

    ct, st = np.cos(t), np.sin(t)
    x = a * np.sign(ct) * (np.abs(ct) ** (2.0 / n_exp))
    y = b * np.sign(st) * (np.abs(st) ** (2.0 / n_exp))
    local_pts = np.column_stack([x, y])

    angle = rng.uniform(0, 2 * np.pi)
    rotated = _rotate(local_pts, angle)
    half_extent = np.array([a, b]) + 0.2
    center = _random_center(rng, half_extent)
    points = rotated - rotated.mean(axis=0) + center
    return _clip_to_arena(points)


def gen_zigzag(rng, n_samples=N_PATH_SAMPLES):
    length = rng.uniform(5.0, 8.0)
    amplitude = rng.uniform(0.5, 1.5)
    n_peaks = rng.integers(2, 5)

    s = np.linspace(0.0, length, n_samples)
    # smooth triangle wave via arcsin(sin(.)) so curvature stays finite
    lateral = (2 * amplitude / np.pi) * np.arcsin(np.sin(2 * np.pi * n_peaks * s / length))
    local_pts = np.column_stack([s, lateral])

    angle = rng.uniform(0, 2 * np.pi)
    rotated = _rotate(local_pts, angle)
    half_extent = np.array([length / 2 + amplitude, length / 2 + amplitude])
    center = _random_center(rng, half_extent)
    points = rotated - rotated.mean(axis=0) + center
    return _clip_to_arena(points)


def gen_circular_arc(rng, n_samples=N_PATH_SAMPLES):
    radius = rng.uniform(2.0, 4.5)
    span_deg = rng.uniform(60.0, 270.0)
    t0 = rng.uniform(0, 2 * np.pi)
    t = np.linspace(t0, t0 + np.radians(span_deg), n_samples)

    local_pts = np.column_stack([radius * np.cos(t), radius * np.sin(t)])

    half_extent = np.array([radius, radius]) + 0.2
    center = _random_center(rng, half_extent)
    points = local_pts - local_pts.mean(axis=0) + center
    return _clip_to_arena(points)


def gen_random_bspline(rng, n_samples=N_PATH_SAMPLES):
    n_free_ctrl = rng.integers(3, 6)
    lo = ARENA_MIN + ARENA_MARGIN
    hi = ARENA_MAX - ARENA_MARGIN

    start = rng.uniform(lo, hi)
    end = rng.uniform(lo, hi)
    free_pts = rng.uniform(lo, hi, size=(n_free_ctrl, 2))
    full_ctrl = np.vstack([start, free_pts, end])
    points = bspline_curve(full_ctrl, n_samples)
    return _clip_to_arena(points)


CURVE_GENERATORS = {
    "sine_wave": gen_sine_wave,
    "cosine_s_curve": gen_cosine_s_curve,
    "rounded_rect": gen_rounded_rect,
    "zigzag": gen_zigzag,
    "circular_arc": gen_circular_arc,
    "random_bspline": gen_random_bspline,
}


def generate_random_path(curve_type, rng, n_samples=N_PATH_SAMPLES, min_length=2.5):
    """Generate one instance of `curve_type`, retrying with fresh random
    parameters if the resulting path is degenerately short (can happen
    after aggressive arena clipping)."""
    generator = CURVE_GENERATORS[curve_type]
    points = generator(rng, n_samples)
    for _ in range(PATH_GENERATION_MAX_RETRIES):
        if _path_length(points) >= min_length:
            break
        points = generator(rng, n_samples)
    return points


# ----------------------------------------------------------------------
# 3. Reference path wrapper (random-curve points OR manual control-point JSON)
# ----------------------------------------------------------------------

class ReferencePath:
    """Wraps a sampled polyline reference path and exposes nearest-point
    and arc-length queries used for the path-following tracking error.

    Two ways to build one:
      - ReferencePath(points=<(M,2) array>, map_name=<str>)   [random curves]
      - ReferencePath(control_points_json=<path to .json>)     [manual maps,
        same per-segment control-point format produced by
        export_all_segments_control_points in dde_mul_la.py]
    """

    def __init__(self, control_points_json=None, points=None, map_name=None):
        if points is not None:
            self.map_name = map_name or "random_path"
            self.points = np.asarray(points, dtype=float)
        elif control_points_json is not None:
            self.map_name = map_name or os.path.splitext(os.path.basename(control_points_json))[0]
            self.points = self._load_points_from_json(control_points_json)
        else:
            raise ValueError("Provide either `points` or `control_points_json`.")

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
# 4. Path-following environment (Gym-style reset/step), NO obstacles
# ----------------------------------------------------------------------

class PathFollowEnv:
    """Unicycle kinematics driven by continuous action [v, w]. Every
    episode starts at the path's actual start point (index 0), not a
    random point along the path. Episode also ends if the robot leaves
    the 12x12 arena.

    Call render() after step()/reset() to visualize the robot live in a
    persistent matplotlib window. Call close() when done rendering."""

    def __init__(self, ref_path: ReferencePath, render_mode=None):
        self.path = ref_path
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
        idx = 0   # always start at the actual path start
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

        reward = -(abs(x_e) + abs(y_e) + abs(theta_e))

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
            reward += 10.0
        elif x < ARENA_MIN[0] or x > ARENA_MAX[0] or y < ARENA_MIN[1] or y > ARENA_MAX[1]:
            done = True
            reward -= 10.0   # terminal penalty for leaving the 12x12 arena
        elif info["dist_to_path"] > OFF_PATH_TOLERANCE:
            done = True
            reward -= 10.0
        elif self.steps >= MAX_STEPS_PER_EPISODE:
            done = True

        info["success"] = success
        info["longitudinal_error"] = x_e
        info["cross_error"] = y_e
        info["theta_error"] = theta_e
        info["progress"] = progress

        if self.render_mode == "human":
            self.render(reward=reward)

        return state, reward, done, info

    # ------------------------------------------------------------------
    # Rendering
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
        self._robot_patch.set_xy(make_robot_triangle(self.pose[0], self.pose[1], self.pose[2]))

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
# 5. Ornstein-Uhlenbeck exploration noise
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
# 6. Replay buffer
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
# 7. Actor / Critic networks
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
# 8. DDPG agent
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


# ----------------------------------------------------------------------
# 9. Training loop: curriculum over random path rounds
# ----------------------------------------------------------------------

def train(max_rounds=MAX_ROUNDS, episodes_per_path=EPISODES_PER_PATH, output_dir=OUTPUT_DIR,
          verbose_every=10, render=False, render_every=1, resume_from=None, seed=None,
          curve_types=None, eval_every_rounds=EVAL_EVERY_ROUNDS):
    """Curriculum training: each round samples ONE new random reference path
    (cycling through curve_types) and trains on it for `episodes_per_path`
    episodes (>= 10, per requirement) before moving to the next round.

    Every `eval_every_rounds` rounds, also runs a noise-free (explore=False)
    greedy rollout on that round's path and prints/logs its success - this
    is the number to trust, since the noisy training success rate can look
    good purely from exploration noise even if the deterministic policy
    itself has collapsed.

    render / render_every / render_eval work exactly as before - render_every
    counts EPISODES (across the whole run), not rounds.
    resume_from: optional path to a .pt checkpoint to continue training from.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    curve_types = curve_types or DEFAULT_CURVE_TYPES

    agent = DDPGAgent()
    if resume_from is not None:
        agent.load(resume_from)
        print(f"Resumed training from checkpoint: {resume_from}")

    episode_rewards = []
    episode_success = []
    episode_curve_types = []
    greedy_eval_log = []   # list of dicts: round_idx, curve_type, success, steps, reward, first_action_mag
    global_episode = 0

    for round_idx in range(max_rounds):
        curve_type = curve_types[round_idx % len(curve_types)]
        points = generate_random_path(curve_type, rng)
        ref_path = ReferencePath(points=points, map_name=f"{curve_type}_round{round_idx}")
        env = PathFollowEnv(ref_path, render_mode=None)

        print(f"\n=== Round {round_idx + 1}/{max_rounds} | curve_type = {curve_type} "
              f"| path length = {ref_path.total_length:.2f} ===")

        for ep_in_path in range(episodes_per_path):
            global_episode += 1
            episode_should_render = render and ((global_episode - 1) % render_every == 0)
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
            episode_curve_types.append(curve_type)

            if global_episode % verbose_every == 0:
                recent = episode_rewards[-verbose_every:]
                recent_success = episode_success[-verbose_every:]
                print(f"  Episode {global_episode:5d} (path ep {ep_in_path + 1}/{episodes_per_path}) | "
                      f"avg reward (last {verbose_every}) = {np.mean(recent):.3f} | "
                      f"success rate = {np.mean(recent_success):.2f} | "
                      f"noise sigma = {agent.noise.sigma:.3f}")

        env.close()

        if (round_idx % eval_every_rounds == 0) or (round_idx == max_rounds - 1):
            greedy = greedy_eval_rollout(agent, ref_path)
            greedy_eval_log.append({"round": round_idx, "curve_type": curve_type, **greedy})
            print(f"  [greedy eval, no noise] success={greedy['success']} | steps={greedy['steps']} | "
                  f"reward={greedy['reward']:.3f} | |first action|={greedy['first_action_mag']:.4f}")
            if greedy["first_action_mag"] < 1e-3:
                print("  WARNING: first greedy action magnitude is ~0 - actor may have collapsed to near-zero output.")

    agent.save(os.path.join(output_dir, "ddpg_path_following_multi_curve.pt"))
    plot_training_curve(episode_rewards, episode_curve_types, output_dir, greedy_eval_log)
    return agent, episode_rewards, episode_curve_types, greedy_eval_log


def plot_training_curve(episode_rewards, episode_curve_types, output_dir, greedy_eval_log=None):
    window = 20
    if len(episode_rewards) >= window:
        smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
    else:
        smoothed = episode_rewards

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episode_rewards, alpha=0.3, label="Episode reward (noisy, training)")
    ax.plot(range(window - 1, window - 1 + len(smoothed)), smoothed, linewidth=2, label=f"{window}-episode moving avg")

    # mark round boundaries where the curve type changes
    for i in range(1, len(episode_curve_types)):
        if episode_curve_types[i] != episode_curve_types[i - 1]:
            ax.axvline(i, color="gray", linewidth=0.5, alpha=0.4)

    if greedy_eval_log:
        # approximate x-position of each greedy eval as the episode index at
        # the end of that round (round_idx * episodes_per_path)
        episodes_per_path = len(episode_rewards) / max(1, (max(e["round"] for e in greedy_eval_log) + 1)) \
            if len(episode_curve_types) else EPISODES_PER_PATH
        xs = [int((e["round"] + 1) * episodes_per_path) - 1 for e in greedy_eval_log]
        ys_success = [e["reward"] for e in greedy_eval_log]
        colors = ["green" if e["success"] else "red" for e in greedy_eval_log]
        ax.scatter(xs, ys_success, c=colors, marker="D", s=60, zorder=5,
                   label="Greedy (no-noise) eval reward\n(green=reached goal, red=did not)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title("DDPG Path-Following Training Reward (multi-curve curriculum)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(output_dir, f"training_reward_multi_curve_{timestamp}.png")
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved training reward curve to {outpath}")


# ----------------------------------------------------------------------
# 10. Evaluation / rollout visualization (unchanged interface)
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
# 11. Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPG path-following trained on a random-curve curriculum (no obstacles)")
    parser.add_argument("--max_rounds", type=int, default=MAX_ROUNDS,
                         help="Number of random-path rounds to train on")
    parser.add_argument("--episodes_per_path", type=int, default=EPISODES_PER_PATH,
                         help="Episodes trained on each generated path (minimum 10 per requirement)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=None, help="Seed for the random path generator")
    parser.add_argument("--resume_from", type=str, default=None,
                         help="Path to a .pt checkpoint to resume training from")
    parser.add_argument("--render", action="store_true",
                         help="Open a live matplotlib window during training so you can watch the robot move")
    parser.add_argument("--render_every", type=int, default=1,
                         help="Render every Nth episode (counted across the whole run) during training")
    parser.add_argument("--render_eval", action="store_true",
                         help="Render the final post-training evaluation rollout live")
    parser.add_argument("--eval_every_rounds", type=int, default=EVAL_EVERY_ROUNDS,
                         help="Run a noise-free (explore=False) greedy evaluation every N rounds during training")
    parser.add_argument("--eval_control_points_json", type=str, default=None,
                         help="Optional: after training, evaluate against a real manual-map control-point JSON "
                              "(e.g. map_001_robot_2_manual_control_points.json) instead of a fresh random curve")
    args = parser.parse_args()

    if args.episodes_per_path < 10:
        print(f"Warning: episodes_per_path={args.episodes_per_path} is below the required minimum of 10; "
              f"raising it to 10.")
        args.episodes_per_path = 10

    print(f"Using device: {DEVICE}")
    trained_agent, rewards, curve_log, greedy_eval_log = train(
        max_rounds=args.max_rounds,
        episodes_per_path=args.episodes_per_path,
        output_dir=args.output_dir,
        render=args.render,
        render_every=args.render_every,
        resume_from=args.resume_from,
        seed=args.seed,
        eval_every_rounds=args.eval_every_rounds,
    )

    if args.eval_control_points_json is not None:
        eval_path = ReferencePath(control_points_json=args.eval_control_points_json)
    else:
        rng = np.random.default_rng(args.seed)
        eval_points = generate_random_path(DEFAULT_CURVE_TYPES[0], rng)
        eval_path = ReferencePath(points=eval_points, map_name=f"eval_{DEFAULT_CURVE_TYPES[0]}")

    evaluate(trained_agent, eval_path, output_dir=args.output_dir, start_frac=0.0, render=args.render_eval)


    # python ran_inference_ddpg_path_following.py --control_points_json solves/multi/map_001_robot_2_manual_control_points.json --checkpoint solves_drl/ddpg_path_following_multi_curve.pt --max_steps 400