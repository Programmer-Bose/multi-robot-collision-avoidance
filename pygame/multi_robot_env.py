"""
Multi-robot simulation environment.

Reuses the same Robot / StaticObstacle / DynamicObstacle primitives from
env.py (unicycle kinematics, bounded random-walk dynamic obstacles) but
drives N independent robots sharing one obstacle field and world.

Design choices, matching the project's decentralized framing:
    - Each robot has its OWN task queue (per-robot task_points), so the
      planned manual "assign tasks per robot from a menu" feature maps
      directly onto `task_points_per_robot` here -- one list per robot,
      set however you like (auto-generated or hand-picked).
    - Per-robot task bookkeeping (completion, stall detection, skip/retry/
      cooldown) is factored into `_RobotTaskState`, a straight port of the
      logic in env.py's SingleRobotEnv so behavior is identical per-robot;
      env.py itself is left untouched so existing single-robot scripts,
      data collection, and DE-MPC pipeline keep working exactly as before.
    - Collision checking adds robot-robot pairwise checks on top of the
      existing robot-vs-obstacle checks.
    - get_obs() returns a list of per-robot dicts, each including an
      "other_robots" field (list of (x, y, theta, radius, vx_est, vy_est)
      for every OTHER robot) -- this is what feeds the neighbor-attention
      block in networks.py; no filtering by sensing range is done here
      (that's a policy/observation-model choice, done in the Gym wrapper),
      this env just exposes full relative state.
"""

import numpy as np
from env import Robot, StaticObstacle, DynamicObstacle


class _RobotTaskState:
    """Per-robot task-queue bookkeeping. Same logic as SingleRobotEnv in env.py."""

    STALL_CHECK_INTERVAL = 20
    STALL_EPS = 0.05
    MAX_RETRIES = 3
    COOLDOWN_STEPS = 30

    def __init__(self, task_points, start_xy):
        self.task_points = [np.array(p, dtype=float) for p in task_points]
        self.n_tasks = len(self.task_points)
        self.start_xy = np.array(start_xy, dtype=float)
        self.skipped_log = []
        self.reset()

    def reset(self):
        self.completed_set = set()
        self.retries = {i: 0 for i in range(self.n_tasks)}
        self.cooldown = {i: 0 for i in range(self.n_tasks)}
        self.pending_queue = list(range(self.n_tasks))
        self.current_task = None
        self.skipped_log = []
        self._advance_task()
        self._reset_attempt_tracking(self.start_xy)

    def _reset_attempt_tracking(self, robot_xy):
        self.attempt_steps = 0
        if self.current_task is not None and robot_xy is not None:
            p = self.task_points[self.current_task]
            self.attempt_ref_dist = np.linalg.norm(robot_xy - p)
        else:
            self.attempt_ref_dist = None

    def _advance_task(self):
        tries = len(self.pending_queue)
        while tries > 0 and self.pending_queue:
            idx = self.pending_queue.pop(0)
            if self.cooldown[idx] <= 0:
                self.current_task = idx
                return
            else:
                self.pending_queue.append(idx)
            tries -= 1
        self.current_task = None

    def all_tasks_done(self, robot_xy, goal_radius):
        return len(self.completed_set) >= self.n_tasks and \
               np.linalg.norm(robot_xy - self.start_xy) < goal_radius

    def n_tasks_completed(self):
        return len(self.completed_set)

    def current_goal(self):
        if self.current_task is not None:
            return self.task_points[self.current_task]
        return self.start_xy  # idling or all done -> head to depot

    def tick(self, robot_xy, t, goal_radius):
        """Advance cooldowns, check goal arrival / stall / skip. Returns (reached, skipped)."""
        for idx in range(self.n_tasks):
            if self.cooldown[idx] > 0:
                self.cooldown[idx] -= 1
                if self.cooldown[idx] == 0 and idx not in self.completed_set \
                        and idx not in self.pending_queue and idx != self.current_task:
                    self.pending_queue.append(idx)

        if self.current_task is None and len(self.completed_set) < self.n_tasks:
            self._advance_task()
            self._reset_attempt_tracking(robot_xy)

        reached, skipped = False, False
        if self.current_task is not None:
            p = self.task_points[self.current_task]
            dist_to_goal = np.linalg.norm(robot_xy - p)

            if dist_to_goal < goal_radius:
                self.completed_set.add(self.current_task)
                reached = True
                self.current_task = None
                self._advance_task()
                self._reset_attempt_tracking(robot_xy)
            else:
                self.attempt_steps += 1
                if self.attempt_steps >= self.STALL_CHECK_INTERVAL:
                    improvement = self.attempt_ref_dist - dist_to_goal
                    if improvement < self.STALL_EPS:
                        self.retries[self.current_task] += 1
                        if self.retries[self.current_task] > self.MAX_RETRIES:
                            skipped_idx = self.current_task
                            self.cooldown[skipped_idx] = self.COOLDOWN_STEPS
                            self.retries[skipped_idx] = 0
                            self.skipped_log.append((t, skipped_idx))
                            skipped = True
                            self.current_task = None
                            self._advance_task()
                    self._reset_attempt_tracking(robot_xy)
        return reached, skipped


class MultiRobotEnv:
    def __init__(self, starts, task_points_per_robot, static_obstacles, dynamic_obstacles,
                 dt=0.1, goal_radius=0.25, world_bounds=(0, 10, 0, 10),
                 v_max=1.0, omega_max=np.pi / 2, robot_radius=0.15,
                 robot_robot_collision=True):
        """
        starts: list of (x, y, theta) or (x, y), one per robot
        task_points_per_robot: list (len = n_robots) of lists of (x, y) task points
        """
        assert len(starts) == len(task_points_per_robot), \
            "starts and task_points_per_robot must have one entry per robot"

        self.n_robots = len(starts)
        self.starts = [np.array(s, dtype=float) for s in starts]
        self.static_obstacles = static_obstacles
        self.dynamic_obstacles = dynamic_obstacles
        self.dt = dt
        self.goal_radius = goal_radius
        self.world_bounds = world_bounds
        self.robot_robot_collision = robot_robot_collision

        self.robots = [
            Robot(s[0], s[1], s[2] if len(s) > 2 else 0.0,
                  v_max=v_max, omega_max=omega_max, radius=robot_radius)
            for s in self.starts
        ]
        self.task_states = [
            _RobotTaskState(task_points_per_robot[i], self.starts[i][:2])
            for i in range(self.n_robots)
        ]

        self.t = 0
        self.history = {"robots": [], "dyn_obs": []}
        self.reset()

    def reset(self):
        self.t = 0
        for i, r in enumerate(self.robots):
            s = self.starts[i]
            r.set_state(s[0], s[1], s[2] if len(s) > 2 else 0.0)
        for ts in self.task_states:
            ts.reset()
        self.history = {"robots": [], "dyn_obs": []}
        self._log()
        return self.get_obs()

    def get_obs(self):
        """Returns a list of per-robot observation dicts."""
        obs_list = []
        for i, r in enumerate(self.robots):
            others = []
            for j, r2 in enumerate(self.robots):
                if i == j:
                    continue
                others.append((r2.x, r2.y, r2.theta, r2.radius))
            obs_list.append({
                "robot_state": r.state.copy(),
                "goal": self.task_states[i].current_goal().copy(),
                "n_tasks_completed": self.task_states[i].n_tasks_completed(),
                "n_tasks_total": self.task_states[i].n_tasks,
                "static_obstacles": [(o.x, o.y, o.radius) for o in self.static_obstacles],
                "dynamic_obstacles": [(o.x, o.y, o.radius, o.vx, o.vy) for o in self.dynamic_obstacles],
                "other_robots": others,
            })
        return obs_list

    def check_collisions(self):
        """Returns list of (collided: bool, kind: str or None) per robot.
        kind in {"static", "dynamic", "robot"}."""
        results = [(False, None) for _ in range(self.n_robots)]
        positions = [np.array([r.x, r.y]) for r in self.robots]

        for i, r in enumerate(self.robots):
            p = positions[i]
            for o in self.static_obstacles:
                if np.linalg.norm(p - o.position()) < (r.radius + o.radius):
                    results[i] = (True, "static")
                    break
            if results[i][0]:
                continue
            for o in self.dynamic_obstacles:
                if np.linalg.norm(p - o.position()) < (r.radius + o.radius):
                    results[i] = (True, "dynamic")
                    break

        if self.robot_robot_collision:
            for i in range(self.n_robots):
                if results[i][0]:
                    continue
                for j in range(self.n_robots):
                    if i == j:
                        continue
                    d = np.linalg.norm(positions[i] - positions[j])
                    if d < (self.robots[i].radius + self.robots[j].radius):
                        results[i] = (True, "robot")
                        break

        return results

    def all_tasks_done(self):
        return all(
            ts.all_tasks_done(np.array([r.x, r.y]), self.goal_radius)
            for ts, r in zip(self.task_states, self.robots)
        )

    def step(self, actions):
        """
        actions: list/array of (v, omega), one per robot, in robot order.
        Returns: (obs_list, done, info)
            done: True if ALL robots finished all tasks, or ANY robot collided
            info: {"reached": [...], "skipped": [...], "collided": [...], "collision_kind": [...]}
        """
        assert len(actions) == self.n_robots

        for r, (v, omega) in zip(self.robots, actions):
            r.step(v, omega, self.dt)
        for o in self.dynamic_obstacles:
            o.step(self.dt)
        self.t += 1

        reached_list, skipped_list = [], []
        for i, r in enumerate(self.robots):
            reached, skipped = self.task_states[i].tick(
                np.array([r.x, r.y]), self.t, self.goal_radius)
            reached_list.append(reached)
            skipped_list.append(skipped)

        collisions = self.check_collisions()
        collided_list = [c[0] for c in collisions]
        collision_kind_list = [c[1] for c in collisions]

        self._log()

        done = self.all_tasks_done() or any(collided_list)
        info = {
            "reached": reached_list,
            "skipped": skipped_list,
            "collided": collided_list,
            "collision_kind": collision_kind_list,
        }
        return self.get_obs(), done, info

    def _log(self):
        self.history["robots"].append([r.state.copy() for r in self.robots])
        self.history["dyn_obs"].append([o.position().copy() for o in self.dynamic_obstacles])


def make_multi_robot_scenario(seed=0, n_robots=3, n_static=6, n_dynamic=4, n_tasks_per_robot=3,
                               world=(0, 10, 0, 10), v_max=1.0, omega_max=np.pi / 2,
                               min_start_separation=1.0):
    """
    Auto-generates a multi-robot scenario: random non-overlapping starts,
    random per-robot task lists, shared static/dynamic obstacle field.
    For manual per-robot task assignment (Phase 5 menu), build
    `task_points_per_robot` yourself and construct MultiRobotEnv directly
    instead of using this helper.
    """
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = world

    starts = []
    while len(starts) < n_robots:
        p = rng.uniform([xmin + 1, ymin + 1], [xmax - 1, ymax - 1])
        if all(np.linalg.norm(p - s[:2]) > min_start_separation for s in starts):
            starts.append(np.array([p[0], p[1], rng.uniform(-np.pi, np.pi)]))

    task_points_per_robot = []
    for _ in range(n_robots):
        pts = []
        while len(pts) < n_tasks_per_robot:
            p = rng.uniform([xmin + 1, ymin + 1], [xmax - 1, ymax - 1])
            pts.append(p)
        task_points_per_robot.append(pts)

    static_obstacles = []
    all_start_xy = [s[:2] for s in starts]
    all_task_xy = [p for pts in task_points_per_robot for p in pts]
    while len(static_obstacles) < n_static:
        p = rng.uniform([xmin + 1, ymin + 1], [xmax - 1, ymax - 1])
        r = rng.uniform(0.2, 0.4)
        if all(np.linalg.norm(p - s) > 1.2 for s in all_start_xy) and \
           all(np.linalg.norm(p - t) > 1.0 for t in all_task_xy):
            static_obstacles.append(StaticObstacle(p[0], p[1], r))

    dynamic_obstacles = []
    for _ in range(n_dynamic):
        p = rng.uniform([xmin + 1, ymin + 1], [xmax - 1, ymax - 1])
        v = rng.uniform(-0.6, 0.6, size=2)
        dynamic_obstacles.append(
            DynamicObstacle(p[0], p[1], radius=0.3, vx=v[0], vy=v[1], bounds=world, rng=rng)
        )

    env = MultiRobotEnv(
        starts=starts,
        task_points_per_robot=task_points_per_robot,
        static_obstacles=static_obstacles,
        dynamic_obstacles=dynamic_obstacles,
        world_bounds=world,
        v_max=v_max,
        omega_max=omega_max,
    )
    return env


if __name__ == "__main__":
    # smoke test: random actions, check multi-robot stepping + collision logic
    env = make_multi_robot_scenario(seed=1, n_robots=4, n_static=5, n_dynamic=3, n_tasks_per_robot=2)
    obs_list = env.reset()
    print(f"n_robots={env.n_robots}")
    for i, obs in enumerate(obs_list):
        print(f"  robot {i}: start={obs['robot_state']}, goal={obs['goal']}, "
              f"n_other_robots={len(obs['other_robots'])}")

    for step in range(100):
        actions = [(0.5, np.sin(step * 0.1 + i)) for i in range(env.n_robots)]
        obs_list, done, info = env.step(actions)
        if any(info["collided"]):
            print(f"[step {step}] collision(s): {list(zip(range(env.n_robots), info['collided'], info['collision_kind']))}")
        if any(info["reached"]):
            print(f"[step {step}] reached: {info['reached']}")
        if done:
            print(f"[step {step}] episode done (all tasks or a collision)")
            break
    else:
        print("Ran 100 steps without finishing.")

    for i, ts in enumerate(env.task_states):
        print(f"robot {i}: completed {ts.n_tasks_completed()}/{ts.n_tasks}, skipped_log={ts.skipped_log}")
