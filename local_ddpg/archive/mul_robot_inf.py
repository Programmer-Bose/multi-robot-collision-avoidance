"""
Multi-Robot DDPG Inference: Sequential Segment-Goal Navigation
=================================================================

Runs a trained 7-input DDPGAgent (path-following + obstacle avoidance)
on one or more robots, but WITHOUT stitching each robot's path into a
single continuous curve to track via nearest-point. Instead, every
B-spline segment's end point is treated as its own discrete task-point
(goal). The robot chases the CURRENT goal only; once it is reached
(within `--waypoint_tolerance`), the NEXT segment's end point becomes
the new goal, and so on until the final segment's end point (the actual
path goal) is reached.

This mirrors the state definition used in training
(state = [x_e, y_e, theta_e, prev_v, prev_w, obs_dist, obs_angle]),
except x_e / y_e / theta_e are now computed relative to the current
task-point goal (position + approach heading of that segment) instead
of the nearest point on a stitched polyline.

Usage
-----
python mul_robot_inf.py --checkpoint solves_drl_multi/multi_robot_ep500.pt --obstacle_maps maps/map_003_robot_1.json maps/map_003_robot_2.json --path_jsons solves/multi/map_003_robot_1_manual_control_points.json solves/multi/map_003_robot_2_manual_control_points.json --output_dir inference_out --run_name eval1
"""

import os
import json
import argparse
import datetime

import numpy as np
import matplotlib.pyplot as plt

from local_ddpg.archive.multi_robot_ddpg import (
    ObstacleMap, DDPGAgent, wrap_to_pi,
    ARENA_MIN, ARENA_MAX, ROBOT_RADIUS,
    STATE_DIM, ACTION_DIM, MAX_LINEAR_VEL, MAX_ANGULAR_VEL, DT,
    GOAL_TOLERANCE, OBSTACLE_SENSE_RADIUS, DEVICE,
)


# ----------------------------------------------------------------------
# 1. Sequential task-point (waypoint) path -- NOT stitched
# ----------------------------------------------------------------------

class WaypointPath:
    """Reads the same per-segment control-point JSON used for training,
    but instead of stitching the segments into one dense polyline, it
    only keeps each segment's start/end point as a discrete task point.

    waypoints[0]  = spawn point (start_point of the first segment)
    waypoints[1:] = one goal per segment, in order (its end_point)
    """

    def __init__(self, control_points_json, map_name=None):
        self.map_name = map_name or os.path.splitext(os.path.basename(control_points_json))[0]
        with open(control_points_json, "r") as f:
            data = json.load(f)

        points = [np.asarray(data["segments"][0]["start_point"], dtype=float)]
        for seg in data["segments"]:
            points.append(np.asarray(seg["end_point"], dtype=float))
        self.points = np.vstack(points)  # (n_segments + 1, 2)

        # approach heading for each waypoint = direction from the
        # previous waypoint to this one
        headings = np.zeros(len(self.points))
        for i in range(1, len(self.points)):
            d = self.points[i] - self.points[i - 1]
            headings[i] = np.arctan2(d[1], d[0])
        headings[0] = headings[1] if len(headings) > 1 else 0.0
        self.headings = headings

        self.n_goals = len(self.points) - 1  # excludes spawn point

    def spawn_point(self):
        return self.points[0], self.headings[0]

    def goal(self, goal_idx):
        """goal_idx in [0, n_goals - 1] -> (xy, heading) of that task point."""
        return self.points[goal_idx + 1], self.headings[goal_idx + 1]

    @property
    def final_goal(self):
        return self.points[-1]


# ----------------------------------------------------------------------
# 2. Inference environment: chase current segment goal, then advance
# ----------------------------------------------------------------------

class SequentialGoalEnv:
    """Same kinematics / obstacle sensing as the training MultiRobotEnv,
    but the tracking target for each robot is its CURRENT task-point
    goal rather than the nearest point on a stitched path. Reaching a
    goal (within `waypoint_tolerance`) advances that robot to its next
    segment's goal."""

    def __init__(self, waypoint_paths, obstacle_maps, waypoint_tolerance=GOAL_TOLERANCE,
                 max_steps=1000, pos_jitter=0.0, heading_jitter=0.0):
        assert len(waypoint_paths) == len(obstacle_maps)
        self.n_robots = len(waypoint_paths)
        self.paths = waypoint_paths
        self.obstacle_maps = obstacle_maps
        self.waypoint_tolerance = waypoint_tolerance
        self.max_steps = max_steps
        self.pos_jitter = pos_jitter
        self.heading_jitter = heading_jitter

        self.poses = np.zeros((self.n_robots, 3))
        self.prev_actions = np.zeros((self.n_robots, 2))
        self.goal_idx = np.zeros(self.n_robots, dtype=int)
        self.finished = np.zeros(self.n_robots, dtype=bool)
        self.steps = 0

    def reset(self):
        for i in range(self.n_robots):
            xy, heading = self.paths[i].spawn_point()
            jitter_xy = np.random.uniform(-self.pos_jitter, self.pos_jitter, size=2)
            jitter_theta = np.random.uniform(-self.heading_jitter, self.heading_jitter)
            self.poses[i] = [xy[0] + jitter_xy[0], xy[1] + jitter_xy[1],
                              wrap_to_pi(heading + jitter_theta)]
            self.goal_idx[i] = 0
        self.prev_actions[:] = 0.0
        self.finished[:] = False
        self.steps = 0
        return self._compute_states()

    def _dynamic_nearest(self, robot_idx, xy, theta):
        best_dist, best_angle = np.inf, 0.0
        for j in range(self.n_robots):
            if j == robot_idx:
                continue
            d = self.poses[j][:2] - xy
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
            path = self.paths[i]
            idx = min(self.goal_idx[i], path.n_goals - 1)
            g_xy, g_heading = path.goal(idx)

            dx = g_xy[0] - x
            dy = g_xy[1] - y
            x_e = np.cos(theta) * dx + np.sin(theta) * dy
            y_e = -np.sin(theta) * dx + np.cos(theta) * dy
            theta_e = wrap_to_pi(g_heading - theta)

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
            infos.append({"dist_to_goal": np.hypot(dx, dy), "goal_idx": idx,
                           "static_dist": static_dist, "dyn_dist": dyn_dist})
        return states, infos

    def step(self, actions):
        actions = np.asarray(actions, dtype=float)
        for i in range(self.n_robots):
            if self.finished[i]:
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

        collided_static = np.zeros(self.n_robots, dtype=bool)
        collided_robot = np.zeros(self.n_robots, dtype=bool)
        left_arena = np.zeros(self.n_robots, dtype=bool)
        reached_final_goal = np.zeros(self.n_robots, dtype=bool)
        advanced_waypoint = np.zeros(self.n_robots, dtype=bool)

        for i in range(self.n_robots):
            if self.finished[i]:
                continue
            info = infos[i]
            x, y, theta = self.poses[i]
            path = self.paths[i]

            if info["static_dist"] < 0:
                collided_static[i] = True
            if info["dyn_dist"] < 0:
                collided_robot[i] = True
            if x < ARENA_MIN[0] or x > ARENA_MAX[0] or y < ARENA_MIN[1] or y > ARENA_MAX[1]:
                left_arena[i] = True

            if info["dist_to_goal"] < self.waypoint_tolerance:
                if self.goal_idx[i] >= path.n_goals - 1:
                    reached_final_goal[i] = True
                else:
                    self.goal_idx[i] += 1
                    advanced_waypoint[i] = True

            if collided_static[i] or collided_robot[i] or left_arena[i] or reached_final_goal[i]:
                self.finished[i] = True

        all_done = bool(np.all(self.finished)) or self.steps >= self.max_steps
        info_out = {
            "collided_static": collided_static, "collided_robot": collided_robot,
            "left_arena": left_arena, "reached_final_goal": reached_final_goal,
            "advanced_waypoint": advanced_waypoint, "finished": self.finished.copy(),
        }
        return states, all_done, info_out


# ----------------------------------------------------------------------
# 3. Rollout + plotting
# ----------------------------------------------------------------------

def run_inference(waypoint_paths, obstacle_maps, agent, waypoint_tolerance=GOAL_TOLERANCE,
                   max_steps=1000, pos_jitter=0.0, heading_jitter=0.0):
    n_robots = len(waypoint_paths)
    env = SequentialGoalEnv(waypoint_paths, obstacle_maps, waypoint_tolerance=waypoint_tolerance,
                             max_steps=max_steps, pos_jitter=pos_jitter, heading_jitter=heading_jitter)
    agent.init_noises(n_robots)  # required by select_actions_batch's noise list, unused with explore=False

    states, _ = env.reset()
    trajectories = [[env.poses[i][:2].copy()] for i in range(n_robots)]
    goal_reach_log = [[] for _ in range(n_robots)]  # step index at which each waypoint was reached

    all_done = False
    while not all_done:
        actions = agent.select_actions_batch(states, explore=False)
        states, all_done, info = env.step(actions)
        for i in range(n_robots):
            trajectories[i].append(env.poses[i][:2].copy())
            if info["advanced_waypoint"][i] or info["reached_final_goal"][i]:
                goal_reach_log[i].append(env.steps)

    summary = []
    for i in range(n_robots):
        summary.append({
            "robot": i,
            "steps_taken": env.steps,
            "final_goal_idx_reached": int(env.goal_idx[i]),
            "total_goals": waypoint_paths[i].n_goals,
            "reached_final_goal": bool(env.goal_idx[i] >= waypoint_paths[i].n_goals - 1
                                        and np.hypot(*(env.poses[i][:2] - waypoint_paths[i].final_goal))
                                        < waypoint_tolerance),
        })

    return trajectories, goal_reach_log, env, summary


def plot_inference_rollout(waypoint_paths, obstacle_maps, trajectories, output_dir, run_name):
    n_robots = len(waypoint_paths)
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_robots))

    for i in range(n_robots):
        traj = np.array(trajectories[i])
        wp = waypoint_paths[i].points
        ax.plot(wp[:, 0], wp[:, 1], "--", color=colors[i], alpha=0.4)
        ax.scatter(wp[1:, 0], wp[1:, 1], marker="x", color=colors[i], s=60,
                   label=f"Robot {i} task points")
        ax.plot(traj[:, 0], traj[:, 1], "-", color=colors[i], linewidth=2,
                label=f"Robot {i} trajectory")
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
    ax.set_title("Inference: sequential task-point (segment goal) navigation")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(output_dir, f"inference_rollout_{run_name}_{timestamp}.png")
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved inference rollout plot to {outpath}")
    return outpath


# ----------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential segment-goal inference for trained multi-robot DDPG")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to a trained 7-input DDPGAgent checkpoint (.pt)")
    parser.add_argument("--obstacle_maps", type=str, nargs="+", required=True)
    parser.add_argument("--path_jsons", type=str, nargs="+", required=True,
                         help="Same *_manual_control_points.json files as training, "
                              "but here each segment end-point is used as a discrete goal")
    parser.add_argument("--waypoint_tolerance", type=float, default=GOAL_TOLERANCE,
                         help="Distance within which a task point counts as reached")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--pos_jitter", type=float, default=0.0,
                         help="Random spawn position jitter (0 = deterministic start)")
    parser.add_argument("--heading_jitter", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="inference_out")
    parser.add_argument("--run_name", type=str, default="eval")
    args = parser.parse_args()

    if len(args.obstacle_maps) != len(args.path_jsons):
        raise ValueError("--obstacle_maps and --path_jsons must have the same length (one pair per robot)")

    waypoint_paths = [WaypointPath(p, map_name=f"robot{i}_{os.path.basename(p)}")
                       for i, p in enumerate(args.path_jsons)]
    obstacle_maps = [ObstacleMap(m) for m in args.obstacle_maps]
    print(f"Loaded {len(waypoint_paths)} robots: " + ", ".join(
        f"[{i}] {wp.n_goals} task points, path={os.path.basename(p)}, obstacles={os.path.basename(m)}"
        for i, (wp, p, m) in enumerate(zip(waypoint_paths, args.path_jsons, args.obstacle_maps))))

    agent = DDPGAgent(state_dim=STATE_DIM)
    agent.load(args.checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint}")

    trajectories, goal_reach_log, env, summary = run_inference(
        waypoint_paths, obstacle_maps, agent,
        waypoint_tolerance=args.waypoint_tolerance, max_steps=args.max_steps,
        pos_jitter=args.pos_jitter, heading_jitter=args.heading_jitter,
    )

    for s in summary:
        print(f"Robot {s['robot']}: reached {s['final_goal_idx_reached'] + 1}/{s['total_goals']} "
              f"task points in {s['steps_taken']} steps "
              f"({'SUCCESS' if s['reached_final_goal'] else 'INCOMPLETE'})")

    plot_inference_rollout(waypoint_paths, obstacle_maps, trajectories, args.output_dir, args.run_name)