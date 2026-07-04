"""
Differential-Evolution Model Predictive Control (DE-MPC) planner.

At each real timestep, DE searches over a horizon-H sequence of controls
(v_0, omega_0, ..., v_{H-1}, omega_{H-1}) that:
  - drives the robot toward its current goal,
  - avoids static obstacles (fixed distance-to-center penalty each step),
  - avoids dynamic obstacles (predicted forward via constant-velocity
    extrapolation of their CURRENT measured velocity -- note the true
    dynamic obstacles in env.py do a random-walk, so this prediction is
    intentionally an approximation, not ground truth for future steps),
  - is kinematically smooth (penalizes large control changes).

Only the first control of the optimized sequence is executed (standard
receding-horizon / MPC convention). The full optimized sequence is kept
and used to warm-start the next DE call (shifted by one step).
"""

import numpy as np
from scipy.optimize import differential_evolution



def rollout(x0, y0, theta0, controls, dt):
    """Roll a control sequence through unicycle kinematics. controls: (H,2) array of (v, omega)."""
    H = controls.shape[0]
    xs = np.empty(H + 1)
    ys = np.empty(H + 1)
    thetas = np.empty(H + 1)
    xs[0], ys[0], thetas[0] = x0, y0, theta0
    for k in range(H):
        v, omega = controls[k]
        xs[k + 1] = xs[k] + v * np.cos(thetas[k]) * dt
        ys[k + 1] = ys[k] + v * np.sin(thetas[k]) * dt
        thetas[k + 1] = thetas[k] + omega * dt
    return xs, ys, thetas


def predict_dynamic_obstacles(dynamic_obstacles, H, dt):
    """
    Constant-velocity forward prediction for each dynamic obstacle.
    dynamic_obstacles: list of (x, y, radius, vx, vy)
    Returns: list of (radius, pred_xs[H+1], pred_ys[H+1])
    """
    preds = []
    for (x, y, r, vx, vy) in dynamic_obstacles:
        ks = np.arange(H + 1)
        pred_xs = x + vx * ks * dt
        pred_ys = y + vy * ks * dt
        preds.append((r, pred_xs, pred_ys))
    return preds


class DEMPCPlanner:
    def __init__(self, horizon=10, dt=0.1, v_max=1.0, omega_max=np.pi / 2,
                 robot_radius=0.15, d_safe_static=0.3, d_safe_dynamic=0.4,
                 w_goal=3.0, w_terminal=8.0, w_collision=250.0, w_smooth=0.5,
                 w_heading=2.0, w_reverse=1.0,
                 popsize=15, maxiter=40, optimizer="de", warm_start=True, seed=None):
        self.H = horizon
        self.dt = dt
        self.v_max = v_max
        self.omega_max = omega_max
        self.robot_radius = robot_radius
        self.d_safe_static = d_safe_static
        self.d_safe_dynamic = d_safe_dynamic
        self.w_goal = w_goal
        self.w_terminal = w_terminal
        self.w_collision = w_collision
        self.w_smooth = w_smooth
        self.w_heading = w_heading
        self.w_reverse = w_reverse
        self.popsize = popsize
        self.maxiter = maxiter
        self.optimizer = optimizer
        self.warm_start = warm_start
        self.recent_positions = []   # list of (x, y), capped length
        self.recent_cap = 50         # how many real steps of history to remember
        self.w_explore = 15.0        # tune this
        self.rng = np.random.default_rng(seed)

        # self.bounds = [(0.0, v_max), (-omega_max, omega_max)] * self.H
        self.bounds = [(-0.3 * v_max, v_max), (-omega_max, omega_max)] * self.H
        self.prev_solution = None  # (H,2) array, warm start

    def update_history(self, x, y):
        self.recent_positions.append((x, y))
        if len(self.recent_positions) > self.recent_cap:
            self.recent_positions.pop(0)

    def _fitness(self, flat_controls, x0, y0, theta0, goal, static_obs, dyn_preds):
        controls = flat_controls.reshape(self.H, 2)
        xs, ys, thetas = rollout(x0, y0, theta0, controls, self.dt)

        # goal tracking: running cost + heavier terminal cost
        dists = np.hypot(xs[1:] - goal[0], ys[1:] - goal[1])
        goal_cost = self.w_goal * np.sum(dists ** 2) + self.w_terminal * dists[-1] ** 2

        # static obstacle penalty (distance from robot center to obstacle center)
        collision_cost = 0.0
        for (ox, oy, orad) in static_obs:
            d = np.hypot(xs[1:] - ox, ys[1:] - oy)
            margin = self.d_safe_static + orad + self.robot_radius
            viol = np.maximum(0.0, margin - d)
            collision_cost += np.sum(viol ** 2)

        # dynamic obstacle penalty (against predicted trajectory, same horizon index k)
        for (orad, pred_xs, pred_ys) in dyn_preds:
            d = np.hypot(xs[1:] - pred_xs[1:], ys[1:] - pred_ys[1:])
            margin = self.d_safe_dynamic + orad + self.robot_radius
            viol = np.maximum(0.0, margin - d)
            collision_cost += np.sum(viol ** 2)

        # smoothness: penalize control jumps
        dv = np.diff(controls[:, 0])
        domega = np.diff(controls[:, 1])
        smooth_cost = np.sum(dv ** 2) + np.sum(domega ** 2)

        # discourage reverse motion by default; DE will still use it when
        # the collision penalty makes forward/turning motion much worse
        reverse_cost = np.sum(np.maximum(0.0, -controls[:, 0]) ** 2)

        # heading penalty: prefer final heading that faces the goal
        dx = goal[0] - xs[-1]
        dy = goal[1] - ys[-1]
        desired_heading = np.arctan2(dy, dx)
        # wrap angle difference to [-pi, pi]
        heading_error = (thetas[-1] - desired_heading + np.pi) % (2 * np.pi) - np.pi
        heading_cost = heading_error ** 2

        return (goal_cost + self.w_collision * collision_cost
            + self.w_smooth * smooth_cost
            + self.w_heading * heading_cost
            + self.w_reverse * reverse_cost)

    def _reverse_threat(self, robot_state, dynamic_obstacles,
                     threat_radius=1.2, closing_speed_thresh=0.05):
        """True if a dynamic obstacle is close AND moving toward the robot."""
        rx, ry, _ = robot_state
        for (ox, oy, orad, vx, vy) in dynamic_obstacles:
            dx, dy = rx - ox, ry - oy
            dist = np.hypot(dx, dy)
            if dist < threat_radius + orad:
                dir_to_robot = np.array([dx, dy]) / (dist + 1e-6)
                closing_speed = np.dot([vx, vy], dir_to_robot)  # + = obstacle heading toward robot
                if closing_speed > closing_speed_thresh:
                    return True
        return False

    def _init_population(self, bounds):
        n_params = self.H * 2
        if self.prev_solution is None:
            return None

        shifted = np.vstack([self.prev_solution[1:], self.prev_solution[-1:]])
        base = shifted.flatten()

        pop = np.empty((self.popsize, n_params))
        pop[0] = base
        noise_scale = np.array([0.15, 0.3] * self.H)
        for i in range(1, self.popsize):
            cand = base + self.rng.normal(0, 1, size=n_params) * noise_scale
            for j, (lo, hi) in enumerate(bounds):   # <-- use passed-in `bounds`
                cand[j] = np.clip(cand[j], lo, hi)
            pop[i] = cand
        return pop

    def plan(self, robot_state, goal, static_obstacles, dynamic_obstacles):
        x0, y0, theta0 = robot_state
        dyn_preds = predict_dynamic_obstacles(dynamic_obstacles, self.H, self.dt)
        allow_reverse = self._reverse_threat(robot_state, dynamic_obstacles)
        v_lo = -self.v_max if allow_reverse else 0.0
        bounds = [(v_lo, self.v_max), (-self.omega_max, self.omega_max)] * self.H
        # print(f"[planner] using optimizer={self.optimizer}")

        if self.optimizer == "de":
            controls, fun = self._plan_de(bounds, x0, y0, theta0, goal, static_obstacles, dyn_preds)
        elif self.optimizer == "cem":
            controls, fun = self._plan_cem(bounds, x0, y0, theta0, goal, static_obstacles, dyn_preds)
        elif self.optimizer == "random_shoot":
            controls, fun = self._plan_random_shoot(bounds, x0, y0, theta0, goal, static_obstacles, dyn_preds)
        else:
            raise ValueError(self.optimizer)

        self.prev_solution = controls if self.warm_start else None
        return controls[0], controls, fun
    def _plan_de(self, bounds, x0, y0, theta0, goal, static_obstacles, dyn_preds):
        init = self._init_population(bounds)
        result = differential_evolution(
            self._fitness,
            bounds=bounds,
            args=(x0, y0, theta0, goal, static_obstacles, dyn_preds),
            popsize=self.popsize,
            maxiter=self.maxiter,
            init="latinhypercube" if init is None else init,
            mutation=(0.4, 1.0),
            recombination=0.7,
            seed=self.rng.integers(0, 1_000_000),
            polish=False,
            tol=1e-4,
            updating="deferred",
        )
        controls = result.x.reshape(self.H, 2)
        return controls, result.fun

    def _plan_cem(self, bounds, x0, y0, theta0, goal, static_obs, dyn_preds,
              n_samples=200, n_elite=20, n_iters=8, init_std_scale=0.5):
        n_params = self.H * 2
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        if self.prev_solution is not None:
            shifted = np.vstack([self.prev_solution[1:], self.prev_solution[-1:]])
            mean = shifted.flatten()
        else:
            mean = (lo + hi) / 2

        std = (hi - lo) * init_std_scale

        for _ in range(n_iters):
            samples = self.rng.normal(mean, std, size=(n_samples, n_params))
            samples = np.clip(samples, lo, hi)
            costs = np.array([
                self._fitness(s, x0, y0, theta0, goal, static_obs, dyn_preds)
                for s in samples
            ])
            elite_idx = np.argsort(costs)[:n_elite]
            elite = samples[elite_idx]
            mean = elite.mean(axis=0)
            std = elite.std(axis=0) + 1e-3   # avoid collapse

        best_cost = self._fitness(mean, x0, y0, theta0, goal, static_obs, dyn_preds)
        return mean.reshape(self.H, 2), best_cost

    def _plan_random_shoot(self, bounds, x0, y0, theta0, goal, static_obs, dyn_preds,
                            n_samples=2000):
        n_params = self.H * 2
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        samples = self.rng.uniform(lo, hi, size=(n_samples, n_params))
        costs = np.array([
            self._fitness(s, x0, y0, theta0, goal, static_obs, dyn_preds)
            for s in samples
        ])
        best = np.argmin(costs)
        return samples[best].reshape(self.H, 2), costs[best]

if __name__ == "__main__":
    # standalone quick test: one DE-MPC solve on a simple static scene
    planner = DEMPCPlanner(horizon=10, dt=0.1, maxiter=30, popsize=15, seed=0)
    robot_state = (0.5, 0.5, 0.0)
    goal = (3.0, 3.0)
    static_obs = [(1.5, 1.5, 0.4)]
    dyn_obs = [(2.0, 2.5, 0.25, -0.1, -0.1)]

    (v0, omega0), seq, cost = planner.plan(robot_state, goal, static_obs, dyn_obs)
    print("First action:", v0, omega0, "| fitness:", cost)
    print("Planned sequence (first 3 steps):\n", seq[:3])
