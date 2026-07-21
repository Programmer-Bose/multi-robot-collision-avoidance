"""
DDPG Obstacle-Avoidance + Multi-Robot Extension (7-input state)
=================================================================

Phase 2: extends the 5-dim path-following-only DDPG policy trained by
ran_ddpg_path_following.py to a 7-dim state that adds an obstacle-perception
term (distance + bearing to the single nearest/most-threatening obstacle,
static circle OR another robot), following:

    Cheng, X. et al. "Path-Following and Obstacle Avoidance Control of
    Nonholonomic Wheeled Mobile Robot Based on Deep Reinforcement Learning."
    Appl. Sci. 2022, 12, 6874.

New state:
    [x_e, y_e, theta_e, prev_v, prev_w, d_obs_min, phi_obs_min]

Everything here is additive on top of ran_ddpg_path_following.py: this file
imports its helpers (bspline_curve, ReferencePath, PathFollowEnv, OUNoise,
ReplayBuffer, wrap_to_pi, etc.) and only reimplements the pieces that
actually change (state dim, networks, weight transfer, per-segment
loading, multi-robot env, training loop).

Usage
-----
Warm-start a 7-dim policy from an existing 5-dim path-following checkpoint,
then train it with a PER-SEGMENT curriculum: segment 0 of every robot in
the map is trained (all robots present at once, so they see each other as
dynamic obstacles) for `episodes_per_segment` episodes, then segment 1,
then segment 2, etc., through every map given in --maps:

    python ddpg_obstacle_multi_robot.py \
        --init_from_5dim solves_drl/ddpg_path_following_multi_curve_60_XXXX.pt \
        --maps map_003 \
        --episodes_per_segment 10 \
        --num_passes 1

Raise --episodes_per_segment (e.g. to 20) for more practice per segment.
--num_passes > 1 repeats the whole segment sweep (revisits segment 0 again
after finishing the map's last segment).

Map files expected alongside each other, named like:
    map_003_robot_1.json                       (obstacles, start/goal)
    map_003_robot_1_manual_control_points.json  (per-segment B-spline ctrl pts)
    map_003_robot_2.json
    map_003_robot_2_manual_control_points.json
"""

import os
import json
import glob
import random
import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from local_ddpg.archive.ran_ddpg_path_following import (
    DEVICE,
    bspline_curve,
    wrap_to_pi,
    make_robot_triangle,
    ReferencePath,
    PathFollowEnv,
    OUNoise,
    ReplayBuffer,
    generate_random_path,
    DEFAULT_CURVE_TYPES,
    DT,
    MAX_LINEAR_VEL,
    MAX_ANGULAR_VEL,
    MAX_STEPS_PER_EPISODE,
    GOAL_TOLERANCE,
    OFF_PATH_TOLERANCE,
    ARENA_MIN,
    ARENA_MAX,
    W_PROGRESS,
    GAMMA,
    TAU,
    ACTOR_LR,
    CRITIC_LR,
    BATCH_SIZE,
    MIN_REPLAY_BEFORE_TRAINING,
    GRAD_CLIP_NORM,
    ACTOR_HIDDEN,
    CRITIC_HIDDEN,
    OUTPUT_DIR as BASE_OUTPUT_DIR,
    EPISODES_PER_PATH,
    MAX_ROUNDS as DEFAULT_MAX_ROUNDS,
    EVAL_EVERY_ROUNDS,
)

# ----------------------------------------------------------------------
# 0. New hyperparameters for the obstacle-aware phase
# ----------------------------------------------------------------------

STATE_DIM_5 = 5          # old path-following-only state
STATE_DIM_7 = 7          # new: + d_obs_min, phi_obs_min
ACTION_DIM = 2

# Obstacle sensing / shaping
SENSE_RADIUS = 3.0        # obstacles farther than this are "not sensed";
                           # d_obs_min saturates at SENSE_RADIUS
ROBOT_COLLISION_RADIUS = 0.1   # robot footprint radius (for robot-robot d_obs)
SAFETY_MARGIN = 0.15            # extra buffer added on top of geometric radius
OBSTACLE_PENALTY_THRESHOLD = 1.0   # start penalizing once d_obs_min < this
OBSTACLE_PENALTY_WEIGHT = 3.0      # overall scale of the obstacle penalty
DIRECTIONAL_WEIGHT = 0.6           # extra weight when obstacle is ~ahead
HARD_COLLISION_RADIUS = 0.15      # d_obs_min below this ends the episode

OUTPUT_DIR = "solves_drl_obstacle"

# Curriculum mix: probability a given round is a multi-robot map round vs a
# single-robot random-curve round (keeps path-following skill from
# regressing while adding obstacle-avoidance exposure).
MULTI_ROBOT_ROUND_PROB = 0.5


# ----------------------------------------------------------------------
# 1. Obstacle perception helper: distance + bearing to nearest threat
# ----------------------------------------------------------------------

def compute_obstacle_state(pos_xy, theta, static_obstacles, other_robot_positions=None,
                            sense_radius=SENSE_RADIUS):
    """Returns (d_obs_min, phi_obs_min) — distance and robot-frame bearing to
    the single most-threatening obstacle among:
      - static_obstacles: list of dicts {"position": [x, y], "radius": r}
      - other_robot_positions: list of (x, y) for every OTHER robot

    Both static circles and other robots are reduced to "distance from robot
    center to obstacle surface" so the two obstacle types are commensurable.
    If nothing is within sense_radius, d_obs_min saturates at sense_radius
    and phi_obs_min is 0 (i.e. "no threat" -> tanh-like neutral state).
    """
    x, y = pos_xy
    best_dist = sense_radius
    best_phi = 0.0
    found = False

    candidates = []
    for obs in static_obstacles or []:
        ox, oy = obs["position"]
        r = obs.get("radius", 0.0)
        surface_dist = np.hypot(x - ox, y - oy) - r
        candidates.append((surface_dist, ox, oy))

    for (ox, oy) in other_robot_positions or []:
        surface_dist = np.hypot(x - ox, y - oy) - (ROBOT_COLLISION_RADIUS + SAFETY_MARGIN)
        candidates.append((surface_dist, ox, oy))

    for surface_dist, ox, oy in candidates:
        if surface_dist < best_dist:
            dx, dy = ox - x, oy - y
            bearing_world = np.arctan2(dy, dx)
            phi = wrap_to_pi(bearing_world - theta)
            best_dist = surface_dist
            best_phi = phi
            found = True

    d_obs_min = float(np.clip(best_dist, 0.0, sense_radius))
    phi_obs_min = float(best_phi) if found else 0.0
    return d_obs_min, phi_obs_min


def obstacle_penalty(d_obs_min, phi_obs_min):
    """Distance/direction-based penalty: grows as d_obs_min shrinks below
    OBSTACLE_PENALTY_THRESHOLD, further weighted by how directly ahead the
    obstacle is (phi_obs_min near 0 = straight ahead = worse)."""
    if d_obs_min >= OBSTACLE_PENALTY_THRESHOLD:
        return 0.0
    closeness = (OBSTACLE_PENALTY_THRESHOLD - d_obs_min) / OBSTACLE_PENALTY_THRESHOLD
    directional = 1.0 + DIRECTIONAL_WEIGHT * np.cos(phi_obs_min)  # in [1-W, 1+W]
    directional = max(directional, 0.0)
    return -OBSTACLE_PENALTY_WEIGHT * closeness * directional


# ----------------------------------------------------------------------
# 2. Map / obstacle / per-segment loading
# ----------------------------------------------------------------------

def load_map_obstacles(robot_json_path):
    with open(robot_json_path, "r") as f:
        data = json.load(f)
    obstacles = data.get("obstacles", [])
    # Normalize world coordinates: robot json is in pixel space (0-800),
    # control-point json is already in the 0-12 arena. Convert obstacles
    # from pixel space to arena space using the same 800px -> 12m scale as
    # the manual maps assume (matches size:[800,800] metadata).
    px_size = data.get("robot_metadata", {}).get("size", [800, 800])
    scale_x = (ARENA_MAX[0] - ARENA_MIN[0]) / px_size[0]
    scale_y = (ARENA_MAX[1] - ARENA_MIN[1]) / px_size[1]
    # NOTE: robot_metadata pixel coordinates use image convention (y grows
    # downward), while the manual control-point / arena coordinates use
    # y-up (matplotlib/Cartesian). Verified against map_003_robot_1:
    # start_position px [66,55] -> arena [0.99, 11.175]; 55*scale_y=0.825,
    # and 12 - 0.825 = 11.175, confirming the y-flip. Without this flip,
    # obstacles land mirrored vertically relative to the reference paths.
    scaled = []
    for obs in obstacles:
        if obs.get("type") != "circle":
            continue
        px, py = obs["position"]
        r = obs["radius"]
        arena_x = px * scale_x
        arena_y = (ARENA_MAX[1] - ARENA_MIN[1]) - (py * scale_y)
        scaled.append({
            "position": [arena_x, arena_y],
            "radius": r * (scale_x + scale_y) / 2.0,
        })
    return scaled


def load_segment_reference_paths(control_points_json):
    """Per-segment loading (NOT stitched into one polyline): one
    ReferencePath per segment, each its own local B-spline."""
    with open(control_points_json, "r") as f:
        data = json.load(f)
    map_name = data.get("map_name", os.path.splitext(os.path.basename(control_points_json))[0])

    segment_paths = []
    for i, seg in enumerate(data["segments"]):
        start = np.asarray(seg["start_point"], dtype=float)
        end = np.asarray(seg["end_point"], dtype=float)
        free_pts = np.asarray(seg["control_points"], dtype=float)
        full_ctrl = np.vstack([start, free_pts, end])
        curve = bspline_curve(full_ctrl, 80)
        segment_paths.append(ReferencePath(points=curve, map_name=f"{map_name}_segment_{i}"))
    return segment_paths, map_name


def discover_robot_maps(map_prefix):
    """Given a prefix like 'map_003', find every robot_N pair of
    (obstacles json, manual control points json) sharing that prefix."""
    ctrl_files = sorted(glob.glob(f"{map_prefix}_robot_*_manual_control_points.json"))
    robots = []
    for ctrl_path in ctrl_files:
        base = ctrl_path.replace("_manual_control_points.json", "")
        obstacles_path = base + ".json"
        if not os.path.exists(obstacles_path):
            continue
        robots.append({
            "robot_json": obstacles_path,
            "control_points_json": ctrl_path,
        })
    return robots


# ----------------------------------------------------------------------
# 3. Networks (7-dim state) + weight transfer from a 5-dim checkpoint
# ----------------------------------------------------------------------

class Actor7(nn.Module):
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


class Critic7(nn.Module):
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


def _expand_first_linear(old_linear, new_in_features, extra_input_cols=None):
    """Build a new nn.Linear with more input features than old_linear,
    copying old weights into the first `old_in_features` columns and
    initializing the new columns freshly (small random init, same scheme
    torch uses by default), so new inputs start near-zero-influence."""
    old_out, old_in = old_linear.weight.shape
    new_linear = nn.Linear(new_in_features, old_out)
    with torch.no_grad():
        new_linear.weight[:, :old_in] = old_linear.weight
        new_linear.bias[:] = old_linear.bias
        # new_linear's default init already randomized the extra columns
        # (torch initializes the whole tensor before we overwrite the first
        # old_in columns), so columns [old_in:new_in_features] are already
        # small random values from nn.Linear's default kaiming-uniform init.
    return new_linear


def build_7dim_networks_from_5dim_checkpoint(ckpt_path):
    """Loads a 5-dim checkpoint (actor/critic/actor_target/critic_target
    state_dicts, as saved by ran_ddpg_path_following.DDPGAgent.save) and
    returns freshly constructed 7-dim Actor7/Critic7 (+targets) with the
    first linear layer's first 5 input columns warm-started from the old
    weights and the 2 new columns freshly initialized."""
    ckpt = torch.load(ckpt_path, map_location="cpu")

    def build_actor():
        return Actor7(STATE_DIM_5, ACTION_DIM, ACTOR_HIDDEN, MAX_LINEAR_VEL, MAX_ANGULAR_VEL)

    def build_critic():
        return Critic7(STATE_DIM_5, ACTION_DIM, CRITIC_HIDDEN)

    networks = {}
    for name in ["actor", "actor_target"]:
        old_net = build_actor()
        old_net.load_state_dict(ckpt[name])
        new_net = Actor7(STATE_DIM_7, ACTION_DIM, ACTOR_HIDDEN, MAX_LINEAR_VEL, MAX_ANGULAR_VEL)
        new_net.net[0] = _expand_first_linear(old_net.net[0], STATE_DIM_7)
        # copy every deeper layer unchanged (shapes match since hidden dims unchanged)
        for i in [2, 4, 6]:
            new_net.net[i].load_state_dict(old_net.net[i].state_dict())
        networks[name] = new_net

    for name in ["critic", "critic_target"]:
        old_net = build_critic()
        old_net.load_state_dict(ckpt[name])
        new_net = Critic7(STATE_DIM_7, ACTION_DIM, CRITIC_HIDDEN)
        # first linear layer's input is state_dim + action_dim; only the
        # state portion grows, so we expand old_in -> old_in+2 but keep the
        # action columns (which sit at the END of the input) intact by
        # rebuilding the weight matrix manually rather than reusing the
        # generic helper (which assumes new columns are simply appended).
        old_first = old_net.net[0]
        old_out, old_in = old_first.weight.shape  # old_in = 5 + 2 = 7
        new_in = STATE_DIM_7 + ACTION_DIM          # = 9
        new_first = nn.Linear(new_in, old_out)
        with torch.no_grad():
            # old layout: [state(5) | action(2)] -> new layout: [state(7) | action(2)]
            new_first.weight[:, :STATE_DIM_5] = old_first.weight[:, :STATE_DIM_5]
            new_first.weight[:, STATE_DIM_7:STATE_DIM_7 + ACTION_DIM] = old_first.weight[:, STATE_DIM_5:STATE_DIM_5 + ACTION_DIM]
            # columns [STATE_DIM_5:STATE_DIM_7] (the 2 new obstacle inputs)
            # keep new_first's own fresh random init from construction.
            new_first.bias[:] = old_first.bias
        new_net.net[0] = new_first
        for i in [2, 4, 6]:
            new_net.net[i].load_state_dict(old_net.net[i].state_dict())
        networks[name] = new_net

    return networks


# ----------------------------------------------------------------------
# 4. DDPG agent (7-dim), otherwise identical logic to the 5-dim agent
# ----------------------------------------------------------------------

class ObstacleAwareDDPGAgent:
    def __init__(self, init_from_5dim=None):
        if init_from_5dim is not None:
            nets = build_7dim_networks_from_5dim_checkpoint(init_from_5dim)
            self.actor = nets["actor"].to(DEVICE)
            self.actor_target = nets["actor_target"].to(DEVICE)
            self.critic = nets["critic"].to(DEVICE)
            self.critic_target = nets["critic_target"].to(DEVICE)
            print(f"Warm-started 7-dim actor/critic from 5-dim checkpoint: {init_from_5dim}")
        else:
            self.actor = Actor7(STATE_DIM_7, ACTION_DIM, ACTOR_HIDDEN, MAX_LINEAR_VEL, MAX_ANGULAR_VEL).to(DEVICE)
            self.actor_target = Actor7(STATE_DIM_7, ACTION_DIM, ACTOR_HIDDEN, MAX_LINEAR_VEL, MAX_ANGULAR_VEL).to(DEVICE)
            self.critic = Critic7(STATE_DIM_7, ACTION_DIM, CRITIC_HIDDEN).to(DEVICE)
            self.critic_target = Critic7(STATE_DIM_7, ACTION_DIM, CRITIC_HIDDEN).to(DEVICE)
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

    def select_actions_batch(self, states, explore=True):
        """Batched action output for N robots in one forward pass:
        states is [N, 7] -> returns [N, 2]."""
        states_t = torch.as_tensor(states, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            actions = self.actor(states_t).cpu().numpy()
        if explore:
            for i in range(actions.shape[0]):
                actions[i] = actions[i] + self.noise.sample() * np.array([MAX_LINEAR_VEL, MAX_ANGULAR_VEL])
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
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "state_dim": STATE_DIM_7,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])


# ----------------------------------------------------------------------
# 5. Single-robot obstacle-aware env (used for the random-curve half of
#    the curriculum; no obstacles present -> neutral obstacle state)
# ----------------------------------------------------------------------

class ObstacleAwarePathFollowEnv(PathFollowEnv):
    """Wraps the 5-dim PathFollowEnv's kinematics/reward but returns a
    7-dim state and adds the obstacle penalty term. With no obstacles
    supplied, d_obs_min saturates at SENSE_RADIUS (neutral / no threat),
    matching the 5-dim behavior for backward compatibility."""

    def __init__(self, ref_path, static_obstacles=None, render_mode=None):
        super().__init__(ref_path, render_mode=render_mode)
        self.static_obstacles = static_obstacles or []

    def _compute_state_7(self, other_robot_positions=None):
        state5, info = self._compute_state()
        x, y, theta = self.pose
        d_obs_min, phi_obs_min = compute_obstacle_state(
            (x, y), theta, self.static_obstacles, other_robot_positions)
        state7 = np.concatenate([state5, [d_obs_min, phi_obs_min]]).astype(np.float32)
        info["d_obs_min"] = d_obs_min
        info["phi_obs_min"] = phi_obs_min
        return state7, info

    def reset(self):
        super().reset()
        return self._compute_state_7()

    def step(self, action, other_robot_positions=None):
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

        state7, info = self._compute_state_7(other_robot_positions)
        x_e, y_e, theta_e = state7[0], state7[1], state7[2]
        d_obs_min, phi_obs_min = state7[5], state7[6]

        reward = -(abs(x_e) + abs(y_e) + abs(theta_e))
        current_arclength = self.path.arclength_at_index(info["nearest_idx"])
        progress = current_arclength - self.prev_arclength
        reward += W_PROGRESS * progress
        self.prev_arclength = current_arclength
        reward += obstacle_penalty(d_obs_min, phi_obs_min)

        done = False
        success = False
        collided = d_obs_min < HARD_COLLISION_RADIUS
        dist_to_goal = np.hypot(x - self.path.goal_point[0], y - self.path.goal_point[1])

        if info["nearest_idx"] >= len(self.path.points) - 2 and dist_to_goal < GOAL_TOLERANCE:
            done = True
            success = True
            reward += 10.0
        elif collided:
            done = True
            reward -= 10.0
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
        info["progress"] = progress

        if self.render_mode == "human":
            self.render(reward=reward)

        return state7, reward, done, info


# ----------------------------------------------------------------------
# 6. Multi-robot environment
# ----------------------------------------------------------------------

class MultiRobotPathFollowEnv:
    """Wraps N ObstacleAwarePathFollowEnv sub-robots that share one static
    obstacle layout and see each other as dynamic obstacles. Each robot
    walks its OWN sequence of per-segment ReferencePaths independently
    (segment index can differ between robots, matching the "advance each
    robot's segment independently" requirement)."""

    def __init__(self, per_robot_segment_paths, static_obstacles, render_mode=None):
        """per_robot_segment_paths: list (len N) of lists of ReferencePath,
        one list of per-segment sub-goals per robot."""
        self.n_robots = len(per_robot_segment_paths)
        self.static_obstacles = static_obstacles
        self.segment_paths = per_robot_segment_paths
        self.seg_idx = [0] * self.n_robots
        self.envs = []
        for r in range(self.n_robots):
            env = ObstacleAwarePathFollowEnv(
                self.segment_paths[r][0], static_obstacles=static_obstacles, render_mode=None)
            self.envs.append(env)
        self.render_mode = render_mode
        self.done_flags = [False] * self.n_robots
        self.success_flags = [False] * self.n_robots

    def reset(self):
        self.seg_idx = [0] * self.n_robots
        self.done_flags = [False] * self.n_robots
        self.success_flags = [False] * self.n_robots
        states = []
        for r in range(self.n_robots):
            self.envs[r].path = self.segment_paths[r][0]
            state7, _ = self.envs[r].reset()
            states.append(state7)
        return np.array(states, dtype=np.float32)

    def _other_positions(self, exclude_idx):
        return [self.envs[r].pose[:2].copy() for r in range(self.n_robots)
                if r != exclude_idx and not self.done_flags[r]]

    def step(self, actions):
        """actions: [N, 2]. Returns states [N,7], rewards [N], dones [N],
        infos (list of dict). A robot that has already finished all its
        segments (or collided/failed) stays frozen and reports done=True."""
        states = np.zeros((self.n_robots, STATE_DIM_7), dtype=np.float32)
        rewards = np.zeros(self.n_robots, dtype=np.float32)
        infos = [{} for _ in range(self.n_robots)]

        # Snapshot positions BEFORE this tick so all robots react to the
        # same instant (order-independent), per the plan.
        snapshot_positions = [self.envs[r].pose[:2].copy() for r in range(self.n_robots)]

        for r in range(self.n_robots):
            if self.done_flags[r]:
                state7, info = self.envs[r]._compute_state_7(
                    other_robot_positions=[p for i, p in enumerate(snapshot_positions) if i != r])
                states[r] = state7
                rewards[r] = 0.0
                infos[r] = info
                continue

            other_pos = [p for i, p in enumerate(snapshot_positions) if i != r]
            state7, reward, done, info = self.envs[r].step(actions[r], other_robot_positions=other_pos)
            states[r] = state7
            rewards[r] = reward
            infos[r] = info

            if done and info.get("success", False):
                # this segment reached; advance to next sub-goal without
                # resetting pose, unless it was the robot's final segment
                completed_idx = self.seg_idx[r]
                self.seg_idx[r] += 1
                print(f"    [robot {r + 1}] reached end of segment {completed_idx} "
                      f"({self.seg_idx[r]}/{len(self.segment_paths[r])} segments done)")
                if self.seg_idx[r] >= len(self.segment_paths[r]):
                    self.done_flags[r] = True
                    self.success_flags[r] = True
                    print(f"    [robot {r + 1}] SUCCESS - all segments completed")
                else:
                    carried_pose = self.envs[r].pose.copy()
                    carried_action = self.envs[r].prev_action.copy()
                    self.envs[r].path = self.segment_paths[r][self.seg_idx[r]]
                    self.envs[r].pose = carried_pose
                    self.envs[r].prev_action = carried_action
                    self.envs[r].steps = 0
                    idx0 = self.envs[r].path.nearest_index(carried_pose[:2])
                    self.envs[r].prev_arclength = self.envs[r].path.arclength_at_index(idx0)
                    self.envs[r]._trail_xy = [carried_pose[:2].copy()]
            elif done:
                # collision / off-path / out-of-arena / step-budget: this
                # robot's episode ends (failure)
                self.done_flags[r] = True
                self.success_flags[r] = False

        all_done = all(self.done_flags)
        return states, rewards, np.array(self.done_flags), infos, all_done

    def close(self):
        for env in self.envs:
            env.close()


# ----------------------------------------------------------------------
# 7. Backward-compatible single-robot eval wrappers (5-dim sanity check)
# ----------------------------------------------------------------------

def greedy_eval_rollout_7dim(agent, ref_path, static_obstacles=None, max_steps=MAX_STEPS_PER_EPISODE):
    """Noise-free rollout used to sanity check training progress. Works for
    both obstacle (static_obstacles given) and obstacle-free (None ->
    d_obs_min saturates at SENSE_RADIUS, phi_obs_min=0, i.e. neutral,
    reproducing old 5-dim path-following behavior on the transferred
    network) cases."""
    env = ObstacleAwarePathFollowEnv(ref_path, static_obstacles=static_obstacles)
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


def evaluate_5dim_sanity(agent, curve_type="sine_wave", seed=0):
    """Runs one obstacle-free rollout to confirm the warm-started 7-dim
    network still reproduces reasonable path-following behavior before
    trusting further obstacle-avoidance training."""
    rng = np.random.default_rng(seed)
    points = generate_random_path(curve_type, rng)
    ref_path = ReferencePath(points=points, map_name=f"sanity_{curve_type}")
    result = greedy_eval_rollout_7dim(agent, ref_path, static_obstacles=None)
    print(f"[5-dim sanity check, obstacle-free] curve={curve_type} success={result['success']} "
          f"steps={result['steps']} reward={result['reward']:.3f} "
          f"|first action|={result['first_action_mag']:.4f}")
    return result


# ----------------------------------------------------------------------
# 8. Per-segment curriculum training loop
#
#    Trains ONE segment index at a time, across ALL robots of a map
#    simultaneously (so they still learn to avoid each other), for a fixed
#    number of episodes ("episodes_per_segment", tunable via CLI/arg from
#    10 up to 20+). Only once that segment's episode budget is used up does
#    training move on to the next segment index. This repeats for every
#    segment of every map, optionally for multiple full passes.
# ----------------------------------------------------------------------

def train_per_segment(maps, episodes_per_segment=EPISODES_PER_PATH, num_passes=1,
                       init_from_5dim=None, output_dir=OUTPUT_DIR, seed=None,
                       eval_every_segments=1, resume_from=None,
                       full_map_eval_every_segments=None):
    """Curriculum: for each map, for each segment index i (0, 1, 2, ...),
    run `episodes_per_segment` episodes where every robot of that map is
    reset to ITS OWN segment i's start point and must reach ITS OWN
    segment i's end point - all robots active at once so they still see
    each other as dynamic obstacles. Moves to segment i+1 only after that
    budget is exhausted. `num_passes` repeats this whole sweep (useful for
    reinforcing earlier segments after later ones have been learned).

    Maps whose robots have different segment counts are supported: a robot
    with no segment `i` is simply left out of that stage's env (skipped),
    so it doesn't block or get blocked by robots that still have segment i.
    """
    os.makedirs(output_dir, exist_ok=True)

    agent = ObstacleAwareDDPGAgent(init_from_5dim=init_from_5dim)
    if resume_from is not None:
        agent.load(resume_from)
        print(f"Resumed training from checkpoint: {resume_from}")
    elif init_from_5dim is not None:
        evaluate_5dim_sanity(agent)

    # Preload all requested maps' per-segment paths + obstacles once.
    map_data = []
    for prefix in maps:
        robots_info = discover_robot_maps(prefix)
        if not robots_info:
            print(f"WARNING: no robot map files found for prefix '{prefix}', skipping.")
            continue
        robots = []
        for r in robots_info:
            segs, _ = load_segment_reference_paths(r["control_points_json"])
            obstacles = load_map_obstacles(r["robot_json"])
            robots.append({"segments": segs, "obstacles": obstacles})
        combined_obstacles = {tuple(o["position"]): o for robot in robots for o in robot["obstacles"]}
        map_data.append({
            "map_name": prefix,
            "robots": robots,
            "obstacles": list(combined_obstacles.values()),
            "max_segments": max(len(robot["segments"]) for robot in robots),
        })

    if not map_data:
        raise ValueError("No valid maps found in `maps` - nothing to train on.")

    episode_rewards = []
    episode_stage_label = []   # e.g. "map_003_seg1_pass0"
    greedy_eval_log = []
    global_episode = 0
    stage_count = 0

    for pass_idx in range(num_passes):
        for map_entry in map_data:
            map_name = map_entry["map_name"]
            static_obstacles = map_entry["obstacles"]

            for seg_idx in range(map_entry["max_segments"]):
                # Which robots actually have this segment index?
                active_robot_ids = [r for r, robot in enumerate(map_entry["robots"])
                                     if seg_idx < len(robot["segments"])]
                if not active_robot_ids:
                    continue
                per_robot_single_segment = [[map_entry["robots"][r]["segments"][seg_idx]]
                                             for r in active_robot_ids]
                stage_label = f"{map_name}_seg{seg_idx}_pass{pass_idx}"
                stage_count += 1

                print(f"\n=== Stage {stage_count} | pass {pass_idx + 1}/{num_passes} | map = {map_name} "
                      f"| segment {seg_idx + 1}/{map_entry['max_segments']} "
                      f"| active robots = {[r + 1 for r in active_robot_ids]} "
                      f"| episodes_per_segment = {episodes_per_segment} ===")

                for ep_in_segment in range(episodes_per_segment):
                    global_episode += 1
                    env = MultiRobotPathFollowEnv(per_robot_single_segment, static_obstacles)
                    states = env.reset()
                    agent.noise.reset()
                    episode_reward = 0.0
                    all_done = False
                    step_count = 0
                    max_steps = MAX_STEPS_PER_EPISODE  # single segment -> one segment's budget

                    while not all_done and step_count < max_steps:
                        actions = agent.select_actions_batch(states, explore=True)
                        next_states, rewards, dones, infos, all_done = env.step(actions)

                        for r in range(env.n_robots):
                            agent.replay.push(states[r], actions[r], rewards[r], next_states[r], float(dones[r]))
                        agent.train_step()

                        states = next_states
                        episode_reward += float(np.sum(rewards))
                        step_count += 1

                    env.close()
                    agent.noise.decay_sigma()
                    episode_rewards.append(episode_reward)
                    episode_stage_label.append(stage_label)

                    n_success = sum(env.success_flags)
                    print(f"  Episode {global_episode:5d} (segment ep {ep_in_segment + 1}/{episodes_per_segment}) | "
                          f"reward={episode_reward:.3f} | robots reached segment end = "
                          f"{n_success}/{env.n_robots} | noise sigma={agent.noise.sigma:.3f}")

                if stage_count % eval_every_segments == 0:
                    eval_env = MultiRobotPathFollowEnv(per_robot_single_segment, static_obstacles)
                    eval_states = eval_env.reset()
                    eval_all_done = False
                    eval_steps = 0
                    while not eval_all_done and eval_steps < MAX_STEPS_PER_EPISODE:
                        eval_actions = agent.select_actions_batch(eval_states, explore=False)
                        eval_states, _, _, _, eval_all_done = eval_env.step(eval_actions)
                        eval_steps += 1
                    n_success = sum(eval_env.success_flags)
                    greedy = {"episode_idx": global_episode, "stage": stage_label,
                              "success": n_success == eval_env.n_robots, "steps": eval_steps}
                    greedy_eval_log.append(greedy)
                    eval_env.close()
                    print(f"  [greedy eval, no noise] robots succeeded = {n_success}/{eval_env.n_robots} "
                          f"| steps={eval_steps}")

    # Optional: sanity-check the fully-stitched multi-segment rollout (all
    # segments in sequence, per robot) after the per-segment curriculum, to
    # see how well the learned per-segment behavior chains together.
    full_map_eval_every_segments = full_map_eval_every_segments or eval_every_segments
    for map_entry in map_data:
        per_robot_segment_paths = [robot["segments"] for robot in map_entry["robots"]]
        eval_env = MultiRobotPathFollowEnv(per_robot_segment_paths, map_entry["obstacles"])
        eval_states = eval_env.reset()
        eval_all_done = False
        eval_steps = 0
        max_steps = MAX_STEPS_PER_EPISODE * map_entry["max_segments"]
        while not eval_all_done and eval_steps < max_steps:
            eval_actions = agent.select_actions_batch(eval_states, explore=False)
            eval_states, _, _, _, eval_all_done = eval_env.step(eval_actions)
            eval_steps += 1
        print(f"\n[full-map chained eval, no noise] map={map_entry['map_name']} | "
              f"segments completed per robot = {eval_env.seg_idx} / "
              f"{[len(s) for s in per_robot_segment_paths]} | steps={eval_steps}")
        eval_env.close()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"ddpg_obstacle_multi_robot_persegment_{timestamp}.pt")
    agent.save(save_path)
    print(f"\nSaved 7-dim checkpoint to {save_path}")
    plot_training_curve(episode_rewards, episode_stage_label, output_dir, greedy_eval_log)
    return agent, episode_rewards, episode_stage_label, greedy_eval_log


def plot_training_curve(episode_rewards, episode_stage_label, output_dir, greedy_eval_log=None):
    window = 20
    if len(episode_rewards) >= window:
        smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
    else:
        smoothed = episode_rewards

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episode_rewards, alpha=0.3, label="Episode reward (noisy, training)")
    ax.plot(range(window - 1, window - 1 + len(smoothed)), smoothed, linewidth=2, label=f"{window}-episode moving avg")

    for i in range(1, len(episode_stage_label)):
        if episode_stage_label[i] != episode_stage_label[i - 1]:
            ax.axvline(i, color="gray", linewidth=0.5, alpha=0.4)

    if greedy_eval_log:
        xs = [min(e["episode_idx"] - 1, len(episode_rewards) - 1) for e in greedy_eval_log]
        ys = [episode_rewards[x] for x in xs]
        colors = ["green" if e["success"] else "red" for e in greedy_eval_log]
        ax.scatter(xs, ys, c=colors, marker="D", s=60, zorder=5,
                   label="Greedy (no-noise) eval\n(green=all active robots succeeded, red=did not)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title("Obstacle-Aware DDPG Training Reward (per-segment curriculum)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(output_dir, f"training_reward_obstacle_{timestamp}.png")
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved training reward curve to {outpath}")


# ----------------------------------------------------------------------
# 9. Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="7-dim obstacle-aware + multi-robot DDPG training "
                                                   "(per-segment curriculum)")
    parser.add_argument("--episodes_per_segment", type=int, default=EPISODES_PER_PATH,
                         help="Number of episodes trained on EACH segment before moving to the next "
                              "segment (minimum 10; raise to 20+ for more practice per segment)")
    parser.add_argument("--num_passes", type=int, default=1,
                         help="Number of times to sweep through all segments of all maps "
                              "(pass 2 revisits segment 0 again after the last segment of pass 1, etc.)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maps", type=str, nargs="*", default=["map_003"],
                         help="Map prefixes to load (each needs "
                              "<prefix>_robot_N.json + <prefix>_robot_N_manual_control_points.json pairs)")
    parser.add_argument("--init_from_5dim", type=str, default=None,
                         help="Path to a 5-dim path-following checkpoint (from "
                              "ran_ddpg_path_following.py) to warm-start the 7-dim network from")
    parser.add_argument("--resume_from", type=str, default=None,
                         help="Path to a 7-dim checkpoint saved by THIS script, to resume training")
    parser.add_argument("--eval_every_segments", type=int, default=1,
                         help="Run a noise-free greedy eval on the just-trained segment every N segment stages")
    args = parser.parse_args()

    if args.episodes_per_segment < 10:
        print(f"Warning: episodes_per_segment={args.episodes_per_segment} is below the recommended "
              f"minimum of 10; raising it to 10.")
        args.episodes_per_segment = 10

    print(f"Using device: {DEVICE}")
    train_per_segment(
        maps=args.maps,
        episodes_per_segment=args.episodes_per_segment,
        num_passes=args.num_passes,
        init_from_5dim=args.init_from_5dim,
        output_dir=args.output_dir,
        seed=args.seed,
        eval_every_segments=args.eval_every_segments,
        resume_from=args.resume_from,
    )

    # python ddpg_obstacle_multi_robot.py --init_from_5dim solves_drl/ddpg_path_following_multi_curve_60_XXXX.pt --maps mandp/map_003 --episodes_per_segment 20 --num_passes 1
