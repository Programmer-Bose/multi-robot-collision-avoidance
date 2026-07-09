"""
config_utils.py
----------------
Central configuration (hyperparameters, environment/reward weights, curriculum
stage definitions) and shared utility functions (map loading, control-point
loading, coordinate/observation helpers, robot padding) used across:

    env.py            - variable-robot-count Gym environment
    policy_reward.py  - PPO actor-critic network + reward functions
    ppo_curriculum.py - PPO algorithm + curriculum manager
    train.py          - training entry point
    eval.py           - evaluation / visualization

Everything here mirrors the conventions already established in
map_gen.py, map_loader.py, dual_de_bspline_la_map_global.py and
pygame_bspline_editor.py, so existing map JSONs and control-point JSONs
can be consumed directly without re-generating anything.
"""

import json
import os
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

# ============================================================
# 1. WORLD / MAP CONSTANTS  (must match the map-generation pipeline)
# ============================================================

TARGET_SCALE = 12.0                 # world is scaled so larger map dim = 12.0 units
BOUNDS_MIN = np.array([0.0, 0.0])
BOUNDS_MAX = np.array([TARGET_SCALE, TARGET_SCALE])

ROBOT_RADIUS = 0.3                  # robot footprint radius, world units

# ============================================================
# 2. ROBOT / SWARM CONSTANTS
# ============================================================

MAX_ROBOTS = 5                      # experiment ceiling; env code has no hard max,
                                     # but padding/observation shapes are built to this
MIN_ROBOTS = 1

# Padding value used for inactive robot slots in observations (chosen far
# outside the arena so padded slots never look like a nearby neighbor/obstacle)
PAD_VALUE = -1.0
PAD_POSITION = np.array([-100.0, -100.0])

# ============================================================
# 3. OBSERVATION / ACTION SPACE CONSTANTS
# ============================================================

# Per-robot "self" observation: [x, y, vx, vy, heading,
#                                 lookahead_dx, lookahead_dy,   (next path point, relative)
#                                 dist_to_goal, path_progress_frac]
SELF_OBS_DIM = 9

# Per-neighbor observation (relative to ego robot): [dx, dy, vx, vy, dist, is_active]
NEIGHBOR_OBS_DIM = 6

# Per-static-obstacle observation (relative to ego robot): [dx, dy, nearest_dist, is_active]
# Only the K_NEAREST_OBSTACLES closest static obstacles are included (padded if fewer).
K_NEAREST_OBSTACLES = 5
OBSTACLE_OBS_DIM = 4

# Action: [linear_velocity_cmd, angular_velocity_cmd] normalized to [-1, 1]
ACTION_DIM = 2
MAX_LINEAR_VEL = 1.5                 # world units / s
MAX_ANGULAR_VEL = 2.0                # rad / s

def obs_dim_for(max_robots=MAX_ROBOTS, k_obstacles=K_NEAREST_OBSTACLES):
    """Total flat observation length for one ego robot:
    self + (max_robots - 1) neighbors + k nearest static obstacles."""
    n_neighbors = max_robots - 1
    return SELF_OBS_DIM + n_neighbors * NEIGHBOR_OBS_DIM + k_obstacles * OBSTACLE_OBS_DIM

# ============================================================
# 4. SIMULATION / CONTROL CONSTANTS
# ============================================================

SIM_DT = 0.1                         # seconds per env step
MAX_EPISODE_STEPS = 500              # per-episode step cap (curriculum may override)

GOAL_REACH_RADIUS = 0.25             # world units, "reached" tolerance
PATH_LOOKAHEAD_DIST = 0.5            # world units ahead along the path for steering target

# ============================================================
# 4b. RENDERING CONSTANTS  (render_mode="human" / "rgb_array" in env.py)
# ============================================================

RENDER_SCREEN_SIZE = 800                     # pygame window is SCREEN_SIZE x SCREEN_SIZE px
RENDER_PIXELS_PER_UNIT = RENDER_SCREEN_SIZE / TARGET_SCALE
RENDER_FPS = 30
RENDER_PANEL_HEIGHT = 60                     # bottom HUD strip (step count, rewards, stage)

RENDER_COLOR_BG = (255, 255, 255)
RENDER_COLOR_OBSTACLE = (200, 100, 100)
RENDER_COLOR_PATH = (180, 180, 220)
RENDER_COLOR_GOAL = (200, 30, 30)
RENDER_COLOR_HUD_BG = (235, 235, 235)
RENDER_COLOR_HUD_TEXT = (20, 20, 20)

# distinct color per robot slot (cycled if n_robots > len(list))
RENDER_ROBOT_COLORS = [
    (30, 140, 30), (30, 90, 200), (230, 140, 20),
    (160, 30, 160), (30, 180, 180), (140, 70, 20),
]

def render_robot_color(robot_idx):
    return RENDER_ROBOT_COLORS[robot_idx % len(RENDER_ROBOT_COLORS)]

# ============================================================
# 5. REWARD WEIGHTS  (shared by policy_reward.py)
# ============================================================

REWARD_WEIGHTS = {
    "w_progress": 1.0,           # reward per unit of forward progress along path
    "w_path_error": 0.5,         # penalty per unit lateral deviation from path
    "w_static_collision": 50.0,  # penalty on static-obstacle collision
    "w_robot_collision": 50.0,   # penalty on inter-robot collision
    "w_static_proximity": 2.0,   # continuous shaping term near static obstacles
    "w_robot_proximity": 2.0,    # continuous shaping term near other robots
    "w_goal_bonus": 100.0,       # one-time bonus on reaching goal
    "w_time_penalty": 0.01,      # per-step penalty to encourage speed
    "w_smoothness": 0.1,         # penalty on angular-velocity jerk
}

COLLISION_PROXIMITY_MARGIN = 0.5     # world units, shaping kicks in within this distance

# ============================================================
# 6. PPO HYPERPARAMETERS  (shared by ppo_curriculum.py)
# ============================================================

PPO_CONFIG = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "n_epochs": 10,
    "minibatch_size": 256,
    "rollout_steps": 2048,           # steps collected per PPO update (across all envs)
    "hidden_sizes": (256, 256),
}

# ============================================================
# 7. CURRICULUM STAGE DEFINITIONS  (shared by ppo_curriculum.py)
# ============================================================
# Each stage lists candidate map JSON files, the robot-count range to sample
# from during that stage, whether inter-robot collision is active, and the
# promotion criterion (mean success rate over a rolling window of episodes).

CURRICULUM_STAGES = [
    {
        "name": "stage1_single_robot_simple_path",
        "map_files": ["maps/stage1_simple_a.json", "maps/stage1_simple_b.json"],
        "n_robots_range": (1, 1),
        "enable_robot_collision": False,
        "success_rate_threshold": 0.9,
        "eval_window": 100,
        "max_episode_steps": 300,
    },
    {
        "name": "stage2_single_robot_complex_path",
        "map_files": ["maps/stage2_complex_a.json", "maps/stage2_complex_b.json"],
        "n_robots_range": (1, 1),
        "enable_robot_collision": False,
        "success_rate_threshold": 0.85,
        "eval_window": 100,
        "max_episode_steps": 400,
    },
    {
        "name": "stage3_multi_robot_low_density",
        "map_files": ["maps/stage3_multi_a.json"],
        "n_robots_range": (2, 3),
        "enable_robot_collision": True,
        "success_rate_threshold": 0.8,
        "eval_window": 150,
        "max_episode_steps": 500,
    },
    {
        "name": "stage4_multi_robot_high_density",
        "map_files": ["maps/stage4_multi_a.json"],
        "n_robots_range": (3, 5),
        "enable_robot_collision": True,
        "success_rate_threshold": 0.75,
        "eval_window": 200,
        "max_episode_steps": 500,
    },
]

# ============================================================
# 8. MAP LOADING  (same scaling convention as map_loader.py / DE script)
# ============================================================

def scale_point(pt, orig_h, scale_factor):
    """Tkinter pixel (x, y) -> world coords, with Y-axis flipped to Cartesian."""
    return np.array([pt[0] * scale_factor, (orig_h - pt[1]) * scale_factor])

def load_static_obstacles(data, orig_h, scale_factor):
    """Builds shapely geometries for all static obstacles in a map JSON,
    identical construction to load_map_config() in the DE script."""
    obstacles = []
    for obs in data["obstacles"]:
        obs_type = obs["type"]
        cx, cy = scale_point(obs["position"], orig_h, scale_factor)

        if obs_type == "circle":
            r = obs["radius"] * scale_factor
            obstacles.append({"type": "circle", "center": (cx, cy), "radius": r})

        elif obs_type == "square":
            h = (obs["size"] * scale_factor) / 2.0
            geom = Polygon([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)])
            obstacles.append({"type": "polygon", "shape": geom})

        elif obs_type == "rectangle":
            hw = (obs["width"] * scale_factor) / 2.0
            hh = (obs["height"] * scale_factor) / 2.0
            geom = Polygon([(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)])
            obstacles.append({"type": "polygon", "shape": geom})

        elif obs_type == "u_shape":
            h = (obs["size"] * scale_factor) / 2.0
            t = obs["thickness"] * scale_factor
            left_arm = Polygon([(cx - h, cy - h), (cx - h + t, cy - h), (cx - h + t, cy + h), (cx - h, cy + h)])
            right_arm = Polygon([(cx + h - t, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx + h - t, cy + h)])
            bottom_bar = Polygon([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy - h + t), (cx - h, cy - h + t)])
            obstacles.append({"type": "polygon", "shape": unary_union([left_arm, right_arm, bottom_bar])})

    return obstacles

def load_map(json_path):
    """Loads a map_gen.py-format JSON and returns (start, goal, task_points,
    task_sequence, obstacles, map_name) in world (0-TARGET_SCALE) coordinates."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Map file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    orig_w, orig_h = data["map_metadata"]["size"]
    scale_factor = TARGET_SCALE / orig_w

    start = scale_point(data["start_position"], orig_h, scale_factor)
    goal = scale_point(data["goal_position"], orig_h, scale_factor) if data.get("goal_position") else None

    task_points = {
        int(tid): scale_point(pos, orig_h, scale_factor)
        for tid, pos in data["task_points"].items()
    }
    task_sequence = data["task_sequence"]
    obstacles = load_static_obstacles(data, orig_h, scale_factor)
    map_name = os.path.splitext(os.path.basename(json_path))[0]

    return {
        "start": start,
        "goal": goal,
        "task_points": task_points,
        "task_sequence": task_sequence,
        "obstacles": obstacles,
        "map_name": map_name,
        "orig_size": (orig_w, orig_h),
        "scale_factor": scale_factor,
    }

# ============================================================
# 9. GLOBAL PATH LOADING  (control-point JSONs produced by the DE planner
#    or the manual pygame editor)
# ============================================================

def load_global_path_control_points(json_path):
    """Loads a *_control_points.json (DE planner) or *_manual_control_points_*.json
    (pygame editor) file: both share the same {segments: [{start_point,
    end_point, control_points}, ...]} schema. Returns a list of segment dicts
    with numpy arrays."""
    with open(json_path, "r") as f:
        data = json.load(f)

    segments = []
    for seg in data["segments"]:
        segments.append({
            "start_point": np.array(seg["start_point"]),
            "end_point": np.array(seg["end_point"]),
            "control_points": np.array(seg["control_points"]),
        })
    return segments

def make_clamped_knot_vector(n_ctrl_pts, degree):
    """Identical to the DE script's knot-vector construction, kept here so
    env.py can rebuild the same clamped B-spline curve from saved control
    points without importing the DE script."""
    n_internal = n_ctrl_pts - degree - 1
    if n_internal > 0:
        internal_knots = np.linspace(0, 1, n_internal + 2)[1:-1]
    else:
        internal_knots = np.array([])
    return np.concatenate((np.zeros(degree + 1), internal_knots, np.ones(degree + 1)))

def bspline_curve(control_points, n_samples, degree=3):
    """Identical formulation to dual_de_bspline_la_map_global.py /
    pygame_bspline_editor.py so global paths render identically everywhere."""
    from scipy.interpolate import BSpline
    control_points = np.asarray(control_points)
    n_ctrl_pts = len(control_points)
    k = min(degree, n_ctrl_pts - 1)
    knots = make_clamped_knot_vector(n_ctrl_pts, k)
    t = np.linspace(0.0, 1.0, n_samples)
    spline_x = BSpline(knots, control_points[:, 0], k)
    spline_y = BSpline(knots, control_points[:, 1], k)
    return np.column_stack([spline_x(t), spline_y(t)])

def build_full_path_polyline(segments, n_samples_per_segment=100):
    """Concatenates every segment's B-spline curve into one dense polyline
    (world coords) representing the full global reference path for a robot."""
    pieces = []
    for seg in segments:
        full_ctrl = np.vstack([seg["start_point"], seg["control_points"], seg["end_point"]])
        curve = bspline_curve(full_ctrl, n_samples_per_segment)
        pieces.append(curve)
    return np.vstack(pieces)

def path_arclength_table(polyline):
    """Cumulative arclength at each polyline vertex, used for progress-along-
    path calculations and lookahead-point queries."""
    diffs = np.diff(polyline, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    return cum  # cum[-1] == total path length

def nearest_point_on_polyline(polyline, cum_arclength, position):
    """Returns (nearest_idx, lateral_error, arclength_at_nearest) for a robot
    position relative to its reference polyline. O(n) brute force; fine for
    a few hundred samples per path at RL step-rate."""
    diffs = polyline - position[None, :]
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    idx = int(np.argmin(dists))
    return idx, float(dists[idx]), float(cum_arclength[idx])

def lookahead_point(polyline, cum_arclength, arclength_now, lookahead_dist):
    """Returns the polyline point at arclength_now + lookahead_dist (clamped
    to the end of the path) - used as the local steering target."""
    target_s = min(arclength_now + lookahead_dist, cum_arclength[-1])
    idx = int(np.searchsorted(cum_arclength, target_s))
    idx = min(idx, len(polyline) - 1)
    return polyline[idx]

# ============================================================
# 10. ROBOT PADDING HELPERS  (variable robot count -> fixed-size batches)
# ============================================================

def pad_robot_array(values, n_active, max_robots=MAX_ROBOTS, pad_value=PAD_VALUE):
    """Pads a (n_active, D) array up to (max_robots, D) with pad_value rows,
    plus returns an (max_robots,) active-mask (1 for real robots, 0 for pad)."""
    values = np.asarray(values, dtype=np.float32)
    n_active = int(n_active)
    D = values.shape[1] if values.ndim == 2 else 1
    padded = np.full((max_robots, D), pad_value, dtype=np.float32)
    padded[:n_active] = values.reshape(n_active, D)
    mask = np.zeros(max_robots, dtype=np.float32)
    mask[:n_active] = 1.0
    return padded, mask

def pad_obstacle_array(obstacle_feats, max_obstacles=K_NEAREST_OBSTACLES, pad_value=PAD_VALUE):
    """Pads a (k_found, D) nearest-obstacle-feature array up to
    (max_obstacles, D); returns padded array + active mask."""
    obstacle_feats = np.asarray(obstacle_feats, dtype=np.float32)
    k_found = obstacle_feats.shape[0] if obstacle_feats.ndim == 2 else 0
    D = OBSTACLE_OBS_DIM
    padded = np.full((max_obstacles, D), pad_value, dtype=np.float32)
    if k_found > 0:
        padded[:min(k_found, max_obstacles)] = obstacle_feats[:max_obstacles]
    mask = np.zeros(max_obstacles, dtype=np.float32)
    mask[:min(k_found, max_obstacles)] = 1.0
    return padded, mask

def sample_n_robots(stage_cfg, rng=None):
    """Samples an active robot count for an episode from the current
    curriculum stage's (min, max) range (inclusive)."""
    rng = rng if rng is not None else np.random
    lo, hi = stage_cfg["n_robots_range"]
    return int(rng.integers(lo, hi + 1)) if hasattr(rng, "integers") else int(rng.randint(lo, hi + 1))

# ============================================================
# 11. MISC GEOMETRY HELPERS
# ============================================================

def wrap_to_pi(angle):
    """Wraps an angle (radians) to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def clip_to_bounds(pos):
    return np.clip(pos, BOUNDS_MIN, BOUNDS_MAX)

def distance_to_nearest_static_obstacles(position, obstacles, k=K_NEAREST_OBSTACLES):
    """Returns up to k nearest static obstacles as (dx, dy, dist) feature
    rows, sorted by distance ascending. `obstacles` is the list produced by
    load_static_obstacles()."""
    from shapely.geometry import Point as ShapelyPoint
    p = ShapelyPoint(position[0], position[1])
    rows = []
    for obs in obstacles:
        if obs["type"] == "circle":
            cx, cy = obs["center"]
            d = np.hypot(position[0] - cx, position[1] - cy) - obs["radius"]
            dx, dy = cx - position[0], cy - position[1]
        else:
            geom = obs["shape"]
            d = geom.distance(p)
            nearest = geom.exterior.interpolate(geom.exterior.project(p))
            dx, dy = nearest.x - position[0], nearest.y - position[1]
        rows.append((dx, dy, d))
    rows.sort(key=lambda r: r[2])
    return np.array(rows[:k], dtype=np.float32) if rows else np.zeros((0, 3), dtype=np.float32)