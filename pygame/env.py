"""
Single-robot simulation environment for DE-MPC path planning.

Robot model: unicycle kinematics
    x_{t+1}     = x_t + v_t * cos(theta_t) * dt
    y_{t+1}     = y_t + v_t * sin(theta_t) * dt
    theta_{t+1} = theta_t + omega_t * dt

Action = (v, omega): linear velocity, angular velocity. Both are bounded.

Obstacles:
    - static:  fixed (x, y, radius), never move.
    - dynamic: (x, y, radius, vx, vy). Motion model is pluggable — default is
      a bounded random-walk in velocity (i.e. NOT simple constant-velocity),
      so that the planner's constant-velocity prediction is genuinely
      approximate, as it would be for a real "randomly moving" obstacle.

Tasks: an ordered list of (x, y) goal points the robot must visit in sequence.
A goal counts as "reached" when the robot is within `goal_radius`. After the
last task is reached, the final goal becomes the depot (start position).
"""

import numpy as np


class Robot:
    def __init__(self, x, y, theta, v_max=1.0, omega_max=np.pi / 2, radius=0.15):
        self.x = x
        self.y = y
        self.theta = theta
        self.v_max = v_max
        self.omega_max = omega_max
        self.radius = radius

    @property
    def state(self):
        return np.array([self.x, self.y, self.theta], dtype=float)

    def set_state(self, x, y, theta):
        self.x, self.y, self.theta = x, y, theta

    def step(self, v, omega, dt):
        v = np.clip(v, -self.v_max, self.v_max)
        omega = np.clip(omega, -self.omega_max, self.omega_max)
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += omega * dt
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
        return self.state


class StaticObstacle:
    def __init__(self, x, y, radius):
        self.x, self.y, self.radius = x, y, radius

    def position(self):
        return np.array([self.x, self.y])


class DynamicObstacle:
    """
    Bounded random-walk-in-velocity obstacle: at each real env step, velocity
    is perturbed by Gaussian noise and clipped to v_max, then position is
    integrated. This is intentionally NOT constant-velocity, so it stresses
    the planner's (simpler) prediction model.
    """

    def __init__(self, x, y, radius, vx, vy, v_max=0.6, noise_std=0.15,
                 bounds=None, rng=None):
        self.x, self.y, self.radius = x, y, radius
        self.vx, self.vy = vx, vy
        self.v_max = v_max
        self.noise_std = noise_std
        self.bounds = bounds  # (xmin, xmax, ymin, ymax) or None
        self.rng = rng if rng is not None else np.random.default_rng()

    def position(self):
        return np.array([self.x, self.y])

    def velocity(self):
        return np.array([self.vx, self.vy])

    def step(self, dt):
        self.vx += self.rng.normal(0, self.noise_std)
        self.vy += self.rng.normal(0, self.noise_std)
        speed = np.hypot(self.vx, self.vy)
        if speed > self.v_max:
            self.vx *= self.v_max / speed
            self.vy *= self.v_max / speed
        self.x += self.vx * dt
        self.y += self.vy * dt
        if self.bounds is not None:
            xmin, xmax, ymin, ymax = self.bounds
            if self.x < xmin or self.x > xmax:
                self.vx *= -1
                self.x = np.clip(self.x, xmin, xmax)
            if self.y < ymin or self.y > ymax:
                self.vy *= -1
                self.y = np.clip(self.y, ymin, ymax)


class SingleRobotEnv:
    # --- new tunables (add to __init__ signature or set as defaults) ---
    STALL_CHECK_INTERVAL = 20   # steps between progress checks
    STALL_EPS = 0.05            # min distance-to-goal improvement required per interval
    MAX_RETRIES = 3              # retries allowed before a task is skipped
    COOLDOWN_STEPS = 30          # steps a skipped task waits before being retry-eligible

    def __init__(self, start, task_points, static_obstacles, dynamic_obstacles,
                 dt=0.1, goal_radius=0.25, world_bounds=(0, 10, 0, 10),
                 v_max=1.0, omega_max=np.pi / 2, robot_radius=0.15):
        self.start = np.array(start, dtype=float)
        self.task_points = [np.array(p, dtype=float) for p in task_points]
        self.static_obstacles = static_obstacles
        self.dynamic_obstacles = dynamic_obstacles
        self.dt = dt
        self.goal_radius = goal_radius
        self.world_bounds = world_bounds

        self.robot = Robot(start[0], start[1], start[2] if len(start) > 2 else 0.0,
                            v_max=v_max, omega_max=omega_max, radius=robot_radius)

        self.n_tasks = len(self.task_points)
        self.history = {"robot": [], "goal_idx": [], "dyn_obs": []}
        self.skipped_log = []
        self.reset()

    def reset(self):
        self.robot.set_state(self.start[0], self.start[1],
                              self.start[2] if len(self.start) > 2 else 0.0)
        self.t = 0
        self.history = {"robot": [], "goal_idx": [], "dyn_obs": []}
        self.skipped_log = []

        # task bookkeeping
        self.completed_set = set()
        self.retries = {i: 0 for i in range(self.n_tasks)}
        self.cooldown = {i: 0 for i in range(self.n_tasks)}
        self.pending_queue = list(range(self.n_tasks))
        self.current_task = None
        self._advance_task()
        self._reset_attempt_tracking()

        self._log()
        return self.get_obs()

    def _reset_attempt_tracking(self):
        self.attempt_steps = 0
        if self.current_task is not None:
            p = self.task_points[self.current_task]
            self.attempt_ref_dist = np.linalg.norm(
                np.array([self.robot.x, self.robot.y]) - p)
        else:
            self.attempt_ref_dist = None

    def _advance_task(self):
        """Pick the next pending, non-cooldown task. Sets self.current_task
        to None if all remaining pending tasks are on cooldown (robot idles
        at depot briefly) or if there are none left (all completed)."""
        tries = len(self.pending_queue)
        while tries > 0 and self.pending_queue:
            idx = self.pending_queue.pop(0)
            if self.cooldown[idx] <= 0:
                self.current_task = idx
                return
            else:
                self.pending_queue.append(idx)
            tries -= 1
        self.current_task = None  # nothing retry-eligible right now

    def all_tasks_done(self):
        # done only once every task has actually been reached (not merely skipped)
        return len(self.completed_set) >= self.n_tasks and \
               np.linalg.norm(np.array([self.robot.x, self.robot.y]) - self.start[:2]) < self.goal_radius

    def n_tasks_completed(self):
        return len(self.completed_set)

    def n_tasks_total(self):
        return self.n_tasks

    def active_task_orig_idx(self):
        return self.current_task

    def current_goal(self):
        if self.current_task is not None:
            return self.task_points[self.current_task]
        if len(self.completed_set) >= self.n_tasks:
            return self.start[:2]      # all done -> head to depot
        return self.start[:2]          # idling: no eligible task right now, hover near depot

    def get_obs(self):
        return {
            "robot_state": self.robot.state.copy(),
            "goal": self.current_goal().copy(),
            "goal_idx": len(self.completed_set),  # kept for backward compat with plotting
            "static_obstacles": [(o.x, o.y, o.radius) for o in self.static_obstacles],
            "dynamic_obstacles": [(o.x, o.y, o.radius, o.vx, o.vy) for o in self.dynamic_obstacles],
        }

    def check_collision(self):
        p = np.array([self.robot.x, self.robot.y])
        for o in self.static_obstacles:
            if np.linalg.norm(p - o.position()) < (self.robot.radius + o.radius):
                return True, "static"
        for o in self.dynamic_obstacles:
            if np.linalg.norm(p - o.position()) < (self.robot.radius + o.radius):
                return True, "dynamic"
        return False, None

    def step(self, v, omega):
        self.robot.step(v, omega, self.dt)
        for o in self.dynamic_obstacles:
            o.step(self.dt)
        self.t += 1

        # tick cooldowns for skipped tasks; requeue once ready
        for idx in range(self.n_tasks):
            if self.cooldown[idx] > 0:
                self.cooldown[idx] -= 1
                if self.cooldown[idx] == 0 and idx not in self.completed_set \
                        and idx not in self.pending_queue and idx != self.current_task:
                    self.pending_queue.append(idx)

        # if robot was idling (current_task is None) and something just became
        # eligible, try to pick it up
        if self.current_task is None and len(self.completed_set) < self.n_tasks:
            self._advance_task()
            self._reset_attempt_tracking()

        reached = False
        skipped = False

        if self.current_task is not None:
            p = self.task_points[self.current_task]
            dist_to_goal = np.linalg.norm(np.array([self.robot.x, self.robot.y]) - p)

            if dist_to_goal < self.goal_radius:
                self.completed_set.add(self.current_task)
                reached = True
                self.current_task = None
                self._advance_task()
                self._reset_attempt_tracking()
            else:
                self.attempt_steps += 1
                if self.attempt_steps >= self.STALL_CHECK_INTERVAL:
                    improvement = self.attempt_ref_dist - dist_to_goal
                    if improvement < self.STALL_EPS:
                        self.retries[self.current_task] += 1
                        if self.retries[self.current_task] > self.MAX_RETRIES:
                            # skip it: cooldown, requeue later, move on
                            skipped_idx = self.current_task
                            self.cooldown[skipped_idx] = self.COOLDOWN_STEPS
                            self.retries[skipped_idx] = 0
                            self.skipped_log.append((self.t, skipped_idx))
                            skipped = True
                            self.current_task = None
                            self._advance_task()
                    self._reset_attempt_tracking()

        collided, kind = self.check_collision()
        self._log()
        done = self.all_tasks_done() or collided
        info = {"reached_goal": reached, "collided": collided,
                "collision_kind": kind, "skipped_task": skipped}
        return self.get_obs(), done, info

    def _log(self):
        self.history["robot"].append(self.robot.state.copy())
        self.history["goal_idx"].append(len(self.completed_set))
        self.history["dyn_obs"].append([o.position().copy() for o in self.dynamic_obstacles])

def order_tasks_nearest_neighbor(start_xy, task_points):
    """Order tasks by angle around the task centroid, then start at the nearest point."""
    if not task_points:
        return []

    start = np.array(start_xy, dtype=float)
    pts = [np.array(p, dtype=float) for p in task_points]

    centroid = np.mean(pts, axis=0)
    angles = [np.arctan2(p[1] - centroid[1], p[0] - centroid[0]) for p in pts]

    ordered = [p for _, p in sorted(zip(angles, pts), key=lambda x: x[0])]

    # rotate so the first visited point is the one nearest the start
    dists = [np.linalg.norm(start - p) for p in ordered]
    first_idx = int(np.argmin(dists))
    # print(f"Task ordering: start={start} -> nearest task={ordered[first_idx]} (idx={first_idx})")
    return ordered[first_idx:] + ordered[:first_idx]

def make_default_scenario(seed=0, n_static=4, n_dynamic=3, n_tasks=4,
                           world=(0, 10, 0, 10), v_max=1.0, omega_max=np.pi / 2):
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = world

    #random start
    start = np.array([rng.uniform(xmin + 1, xmax - 1), rng.uniform(ymin + 1, ymax - 1), 0.0])
    print(f"Start: {start}")

    task_points = []
    while len(task_points) < n_tasks:
        p = rng.uniform([xmin + 1, ymin + 1], [xmax - 1, ymax - 1])
        if np.linalg.norm(p - start[:2]) > 1.0:
            task_points.append(p)

    static_obstacles = []
    while len(static_obstacles) < n_static:
        p = rng.uniform([xmin + 1, ymin + 1], [xmax - 1, ymax - 1])
        r = rng.uniform(0.2, 0.4)
        if np.linalg.norm(p - start[:2]) > 1.2 and all(
            np.linalg.norm(p - tp) > 1.0 for tp in task_points
        ):
            static_obstacles.append(StaticObstacle(p[0], p[1], r))

    dynamic_obstacles = []
    for _ in range(n_dynamic):
        p = rng.uniform([xmin + 1, ymin + 1], [xmax - 1, ymax - 1])
        v = rng.uniform(-0.4, 0.4, size=2)
        dynamic_obstacles.append(
            DynamicObstacle(p[0], p[1], radius=0.3, vx=v[0], vy=v[1],
                             bounds=world, rng=rng)
        )

    task_points = order_tasks_nearest_neighbor(start[:2], task_points)
    env = SingleRobotEnv(
        start=start,
        task_points=task_points,
        static_obstacles=static_obstacles,
        dynamic_obstacles=dynamic_obstacles,
        world_bounds=world,
        v_max=v_max,
        omega_max=omega_max,
    )
    return env


if __name__ == "__main__":
    # quick smoke test: random actions, check the environment runs & logs sanely
    env = make_default_scenario(seed=1)
    obs = env.reset()
    print("Start:", obs["robot_state"], "First goal:", obs["goal"])
    for step in range(50):
        obs, done, info = env.step(v=0.5, omega=0.1)
        if info["collided"]:
            print(f"Collision at step {step} ({info['collision_kind']})")
            break
        if info["reached_goal"]:
            print(f"Reached goal {obs['goal_idx']-1} at step {step}")
        if done:
            print("All tasks done at step", step)
            break
    print("Final robot state:", obs["robot_state"])