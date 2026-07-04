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
        # self.noise_std = noise_std
        self.bounds = bounds  # (xmin, xmax, ymin, ymax) or None
        self.rng = rng if rng is not None else np.random.default_rng()

    def position(self):
        return np.array([self.x, self.y])

    # def velocity(self):
    #     return np.array([self.vx, self.vy])

    def step(self, dt):
        # self.vx += self.rng.normal(0, self.noise_std)
        # self.vy += self.rng.normal(0, self.noise_std)
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
    def __init__(self, start, task_points, static_obstacles, dynamic_obstacles,
                 dt=0.1, goal_radius=0.15, world_bounds=(0, 15, 0, 15),
                 v_max=1.0, omega_max=np.pi / 2, robot_radius=0.15,
                 task_timeout=150, max_task_retries=8):
        self.start = np.array(start, dtype=float)
        self.task_points = [np.array(p, dtype=float) for p in task_points]
        self.static_obstacles = static_obstacles
        self.dynamic_obstacles = dynamic_obstacles
        self.dt = dt
        self.goal_radius = goal_radius
        self.world_bounds = world_bounds

        self.robot = Robot(start[0], start[1], start[2] if len(start) > 2 else 0.0,
                            v_max=v_max, omega_max=omega_max, radius=robot_radius)

        # Task queue with skip-and-requeue support: a task that can't be
        # reached within `task_timeout` steps (e.g. a dynamic obstacle is
        # sitting on/near it) is pushed to the back of the queue and
        # revisited later, instead of being silently counted as "done" if
        # the robot merely grazes past it while avoiding something. Depot
        # is only targeted once every task has been genuinely visited.
        self.task_timeout = task_timeout
        self.max_task_retries = max_task_retries
        self.pending_tasks = [(i, p, 0) for i, p in enumerate(self.task_points)]  # (orig_idx, point, retries)
        self.current_task = None  # (orig_idx, point, retries) or None -> depot phase
        self.steps_on_current = 0
        self.visited_log = []    # orig_idx in actual completion order
        self.skipped_log = []    # (orig_idx, step) each time a task is skipped-and-requeued
        self.abandoned_log = []  # orig_idx forced-accepted after exceeding max_task_retries
        self.depot_reached = False
        self.t = 0
        self.history = {"robot": [], "active_task_idx": [], "dyn_obs": []}

    def reset(self):
        self.robot.set_state(self.start[0], self.start[1],
                              self.start[2] if len(self.start) > 2 else 0.0)
        self.pending_tasks = [(i, p, 0) for i, p in enumerate(self.task_points)]
        self.current_task = None
        self.steps_on_current = 0
        self.visited_log = []
        self.skipped_log = []
        self.abandoned_log = []
        self.depot_reached = False
        self.t = 0
        self.history = {"robot": [], "active_task_idx": [], "dyn_obs": []}
        self._advance_task()
        self._log()
        return self.get_obs()

    def _advance_task(self):
        """Pop the next pending task as current target; if none left, target depot."""
        if self.pending_tasks:
            self.current_task = self.pending_tasks.pop(0)
            self.steps_on_current = 0
        else:
            self.current_task = None  # signals depot phase

    def current_goal(self):
        if self.current_task is not None:
            return self.current_task[1]
        return self.start[:2]  # depot phase (or staying at depot once reached)

    def all_tasks_done(self):
        return self.current_task is None and len(self.pending_tasks) == 0 and self.depot_reached

    def active_task_orig_idx(self):
        """Original task index currently being targeted, or None if in depot phase."""
        return self.current_task[0] if self.current_task is not None else None

    def n_tasks_total(self):
        return len(self.task_points)

    def n_tasks_completed(self):
        return len(self.visited_log)

    def get_obs(self):
        return {
            "robot_state": self.robot.state.copy(),
            "goal": self.current_goal().copy(),
            "active_task_idx": self.active_task_orig_idx(),
            "n_tasks_completed": self.n_tasks_completed(),
            "n_tasks_total": self.n_tasks_total(),
            "depot_reached": self.depot_reached,
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
        self.steps_on_current += 1

        goal = self.current_goal()
        dist_to_goal = np.linalg.norm(np.array([self.robot.x, self.robot.y]) - goal)
        reached = False
        skipped = False

        if self.current_task is not None:
            orig_idx, point, retries = self.current_task
            # print(f"Step {self.t}: targeting task {orig_idx} at {point}, distance to goal: {dist_to_goal:.3f},retries: {retries}")
            if dist_to_goal < self.goal_radius:
                # genuinely arrived: only counts once the robot is actually
                # within radius of ITS current active task, not some other
                # point it happens to pass near while avoiding an obstacle.
                self.visited_log.append(orig_idx)
                reached = True
                self._advance_task()
            elif self.steps_on_current >= self.task_timeout:
                # couldn't reach it in time (e.g. an obstacle is blocking
                # approach) -> requeue it for a later attempt instead of
                # silently marking it done.
                skipped = True
                self.skipped_log.append((orig_idx, self.t))
                if retries + 1 >= self.max_task_retries:
                    # give up on this one after too many attempts, so the
                    # episode can't loop forever; log it as abandoned.
                    # print(repr(f"Step {self.t}: abandoning task {orig_idx} after {retries+1} retries"))
                    self.abandoned_log.append(orig_idx)
                else:
                    self.pending_tasks.append((orig_idx, point, retries + 1))
                self._advance_task()
        else:
            # depot phase: only reached once all tasks are visited/abandoned
            if dist_to_goal < self.goal_radius:
                self.depot_reached = True
                reached = True

        collided, kind = self.check_collision()
        self._log()
        done = self.all_tasks_done() or collided
        info = {"reached_goal": reached, "skipped_task": skipped,
                "collided": collided, "collision_kind": kind}
        return self.get_obs(), done, info

    def _log(self):
        self.history["robot"].append(self.robot.state.copy())
        self.history["active_task_idx"].append(self.active_task_orig_idx())
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
    return ordered[first_idx:] + ordered[:first_idx]

def make_default_scenario(seed=0, n_static=4, n_dynamic=3, n_tasks=4,
                           world=(0, 10, 0, 10), omega_max=np.pi/2):
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = world

    start = np.array([0.5, 0.5, 0.0])

    task_points = []
    while len(task_points) < n_tasks:
        p = rng.uniform([xmin + 1, ymin + 1], [xmax - 1, ymax - 1])
        if np.linalg.norm(p - start[:2]) > 1.0:
            task_points.append(p)

    static_obstacles = []
    while len(static_obstacles) < n_static:
        p = rng.uniform([xmin + 1, ymin + 1], [xmax - 1, ymax - 1])
        r = rng.uniform(0.3, 0.6)
        if np.linalg.norm(p - start[:2]) > 1.2 and all(
            np.linalg.norm(p - tp) > 1.0 for tp in task_points
        ):
            static_obstacles.append(StaticObstacle(p[0], p[1], r))

    dynamic_obstacles = []
    for _ in range(n_dynamic):
        p = rng.uniform([xmin + 1, ymin + 1], [xmax - 1, ymax - 1])
        v = rng.uniform(-0.4, 0.4, size=2)
        #add velocity if it is zero
        if np.linalg.norm(v) < 0.1:
            v = rng.uniform(0.2, 0.4, size=2)
        dynamic_obstacles.append(
            DynamicObstacle(p[0], p[1], radius=0.25, vx=v[0], vy=v[1],v_max=1.2,
                             bounds=world, rng=rng)
        )

    task_points = order_tasks_nearest_neighbor(start[:2], task_points)
    env = SingleRobotEnv(
        start=start,
        task_points=task_points,
        static_obstacles=static_obstacles,
        dynamic_obstacles=dynamic_obstacles,
        omega_max=omega_max,
        max_task_retries=15,
        world_bounds=world,
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
            print(f"Reached task {obs['n_tasks_completed']-1} at step {step}")
        if info["skipped_task"]:
            print(f"Skipped/requeued a task at step {step} (timeout)")
        if done:
            print("All tasks done at step", step)
            break
    print("Final robot state:", obs["robot_state"])
