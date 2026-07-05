"""
Live simulation with a receding-horizon ON/OFF toggle.

RECEDING_HORIZON = True   -> replans with DE every single step (short horizon).
                             This is the normal DE-MPC closed-loop behavior.

RECEDING_HORIZON = False  -> solves DE ONCE per task segment with a long
                             horizon (enough to reach the goal), then blindly
                             executes that whole fixed control sequence with
                             NO replanning until the goal is reached or the
                             plan runs out. Dynamic obstacles keep moving
                             unpredictably in the meantime, so this isolates
                             what receding-horizon replanning actually buys you.

Run:
    python3 live_sim.py
Edit the CONFIG block below to change settings.
"""

from turtle import mode

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from env import make_default_scenario
from de_mpc import DEMPCPlanner
from narrow_passage_scenario import make_narrow_passage_scenario

# ----------------------------- CONFIG -----------------------------
RECEDING_HORIZON = True     # <-- TOGGLE THIS: True = replan every step, False = single-shot DE
SEED = 1
N_STATIC = 10
N_DYNAMIC = 5
N_TASKS = 15
CLOSED_LOOP_HORIZON = 10    # horizon used when RECEDING_HORIZON = True
OPEN_LOOP_HORIZON = 70      # horizon used when RECEDING_HORIZON = False (must cover a full segment)
MAX_STEPS = 2000
OPTIMIZER = "de" 
WARM_START = True
# --------------------------------------------------------------------


def build_env_and_planner():
    env = make_default_scenario(seed=SEED, n_static=N_STATIC, n_dynamic=N_DYNAMIC, n_tasks=N_TASKS, omega_max=np.pi)
    env.reset()
    # env = make_narrow_passage_scenario(gap_width=0.5, robot_radius=0.15, seed=SEED)
    # env.reset()
    horizon = CLOSED_LOOP_HORIZON if RECEDING_HORIZON else OPEN_LOOP_HORIZON
    planner = DEMPCPlanner(horizon=horizon, dt=env.dt, v_max=2.5,
                            omega_max=env.robot.omega_max, robot_radius=env.robot.radius,
                            seed=SEED, optimizer=OPTIMIZER, warm_start=WARM_START)
    return env, planner


class LiveSim:
    def __init__(self):
        self.env, self.planner = build_env_and_planner()
        self.active_plan = None   # only used when RECEDING_HORIZON = False
        self.plan_ptr = 0
        self.finished = False
        self.collided = False
        self.step_count = 0

        # ---- figure setup ----
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        xmin, xmax, ymin, ymax = self.env.world_bounds
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_aspect("equal")
        mode = "RECEDING HORIZON (replan every step)" if RECEDING_HORIZON else "OPEN-LOOP (single DE solve per segment)"
        self.ax.set_title(f"DE-MPC live | {mode} | optimizer={OPTIMIZER}")

        self.ax.plot(self.env.start[0], self.env.start[1], "ks", markersize=10, label="Depot")
        for i, tp in enumerate(self.env.task_points):
            self.ax.plot(tp[0], tp[1], "g^", markersize=10)
            self.ax.annotate(f"T{i+1}", (tp[0], tp[1]), textcoords="offset points",
                              xytext=(6, 6), fontsize=9, color="darkgreen")
        for o in self.env.static_obstacles:
            self.ax.add_patch(plt.Circle((o.x, o.y), o.radius, color="gray", alpha=0.6))

        self.dyn_patches = [
            plt.Circle((o.x, o.y), o.radius, color="tomato", alpha=0.8)
            for o in self.env.dynamic_obstacles
        ]
        for p in self.dyn_patches:
            self.ax.add_patch(p)

        self.robot_dot, = self.ax.plot([], [], "bo", markersize=9, label="Robot")
        self.robot_heading, = self.ax.plot([], [], "b-", linewidth=2)
        self.path_line, = self.ax.plot([], [], "b-", linewidth=1, alpha=0.5)
        self.status_text = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes,
                                         va="top", fontsize=9,
                                         bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        self.ax.legend(loc="upper right", fontsize=8)

        self.traj_x, self.traj_y = [], []

    def _get_action(self, obs):
        if RECEDING_HORIZON:
            (v0, omega0), seq, cost = self.planner.plan(
                robot_state=tuple(obs["robot_state"]), goal=tuple(obs["goal"]),
                static_obstacles=obs["static_obstacles"],
                dynamic_obstacles=obs["dynamic_obstacles"])
            return v0, omega0
        else:
            need_new_plan = (self.active_plan is None) or (self.plan_ptr >= len(self.active_plan))
            if need_new_plan:
                self.planner.prev_solution = None  # no warm-start across segments (fresh long solve)
                _, seq, cost = self.planner.plan(
                    robot_state=tuple(obs["robot_state"]), goal=tuple(obs["goal"]),
                    static_obstacles=obs["static_obstacles"],
                    dynamic_obstacles=obs["dynamic_obstacles"])
                self.active_plan = seq
                self.plan_ptr = 0
                print(f"[open-loop] solved new {OPEN_LOOP_HORIZON}-step plan for goal {obs['goal']}, cost={cost:.1f}")
            v0, omega0 = self.active_plan[self.plan_ptr]
            self.plan_ptr += 1
            return v0, omega0

    def update(self, frame):
        if self.finished:
            return self._artists()

        obs = self.env.get_obs()
        v0, omega0 = self._get_action(obs)
        obs, done, info = self.env.step(v0, omega0)
        self.step_count += 1

        if info["reached_goal"] or info["skipped_task"]:
            self.active_plan = None  # force fresh solve for next segment (open-loop mode)
            self.plan_ptr = 0

        if info["collided"]:
            self.collided = True
            self.finished = True
            print(f"[step {self.step_count}] COLLISION ({info['collision_kind']})")

        if done:
            self.finished = True
            if not self.collided:
                print(f"[step {self.step_count}] all tasks complete, returned to depot")

        if self.step_count >= MAX_STEPS:
            self.finished = True

        # ---- update artists ----
        r = self.env.robot
        self.traj_x.append(r.x)
        self.traj_y.append(r.y)
        self.robot_dot.set_data([r.x], [r.y])
        hx = r.x + 0.3 * np.cos(r.theta)
        hy = r.y + 0.3 * np.sin(r.theta)
        self.robot_heading.set_data([r.x, hx], [r.y, hy])
        self.path_line.set_data(self.traj_x, self.traj_y)

        for patch, o in zip(self.dyn_patches, self.env.dynamic_obstacles):
            patch.center = (o.x, o.y)

        n_done = self.env.n_tasks_completed()
        n_total = self.env.n_tasks_total()
        phase = "depot" if self.env.current_task is None else f"task {self.env.active_task_orig_idx()+1}"
        status = f"step {self.step_count} | tasks {n_done}/{n_total} | targeting {phase}"
        if self.env.skipped_log:
            status += f"\nrequeued so far: {len(self.env.skipped_log)}"
        if self.collided:
            status += "\nCOLLISION!"
        elif self.finished:
            status += "\nDONE"
        self.status_text.set_text(status)

        return self._artists()

    def _artists(self):
        return [self.robot_dot, self.robot_heading, self.path_line, self.status_text] + self.dyn_patches

    def run(self):
        ani = animation.FuncAnimation(self.fig, self.update, frames=MAX_STEPS,
                                       interval=30, blit=False, repeat=False)
        plt.show()
        return ani


if __name__ == "__main__":
    sim = LiveSim()
    ani = sim.run()
