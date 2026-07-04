"""
Manually drive the robot with the keyboard, watch it live, and record the
episode in the same data format as run_episode.py's DE-generated demos.

Controls:
    W / S       -> increase / decrease linear velocity v
    A / D       -> increase / decrease angular velocity omega
    X           -> zero both v and omega (stop)
    R           -> reset episode (new scenario, same seed unless changed)
    Q           -> quit and save recorded data

Run:
    python3 manual_control.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from env import make_default_scenario
from rangefinder import simulate_rangefinder


def save_dataset(dataset, path="demo_data.npz"):
    """Pack a list of per-step dicts into flat arrays and save to .npz."""
    if not dataset:
        print("Empty dataset, nothing saved.")
        return
    np.savez(
        path,
        robot_state=np.stack([d["robot_state"] for d in dataset]),
        goal_dist=np.array([d["goal_dist"] for d in dataset]),
        goal_bearing=np.array([d["goal_bearing"] for d in dataset]),
        ranges=np.stack([d["ranges"] for d in dataset]),
        action=np.stack([d["action"] for d in dataset]),
    )
    print(f"Saved {len(dataset)} timesteps to {path}")

# ----------------------------- CONFIG -----------------------------
SEED = 1
N_STATIC = 4
N_DYNAMIC = 3
N_TASKS = 4
N_RAYS = 15
SENSOR_RANGE = 5.0
V_STEP = 0.05        # how much W/S changes v per keypress
OMEGA_STEP = 0.01    # how much A/D changes omega per keypress
MAX_STEPS = 2000
SAVE_PATH = "manual_demo.npz"
# --------------------------------------------------------------------


class ManualControlSim:
    def __init__(self):
        self.env = make_default_scenario(seed=SEED, n_static=N_STATIC,
                                          n_dynamic=N_DYNAMIC, n_tasks=N_TASKS)
        self.obs = self.env.reset()
        self.v = 0.0
        self.omega = 0.0
        self.step_count = 0
        self.finished = False
        self.dataset = []

        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        xmin, xmax, ymin, ymax = self.env.world_bounds
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_aspect("equal")
        self.ax.set_title("Manual control | W/S=v  A/D=omega  X=stop  R=reset  Q=quit+save")

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

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def on_key(self, event):
        if event.key == "w":
            self.v = min(self.v + V_STEP, self.env.robot.v_max)
        elif event.key == "s":
            self.v = max(self.v - V_STEP, -self.env.robot.v_max)
        elif event.key == "a":
            self.omega = min(self.omega + OMEGA_STEP, self.env.robot.omega_max)
        elif event.key == "d":
            self.omega = max(self.omega - OMEGA_STEP, -self.env.robot.omega_max)
        elif event.key == "x":
            self.v, self.omega = 0.0, 0.0
        elif event.key == "r":
            self.obs = self.env.reset()
            self.v, self.omega = 0.0, 0.0
            self.step_count = 0
            self.finished = False
            self.traj_x, self.traj_y = [], []
            print("Episode reset.")
        elif event.key == "q":
            print("Quitting and saving...")
            save_dataset(self.dataset, path=SAVE_PATH)
            plt.close(self.fig)

    def _record_step(self):
        ranges = simulate_rangefinder(
            robot_state=tuple(self.obs["robot_state"]),
            static_obstacles=self.obs["static_obstacles"],
            dynamic_obstacles=self.obs["dynamic_obstacles"],
            world_bounds=self.env.world_bounds,
            n_rays=N_RAYS, max_range=SENSOR_RANGE,
        )
        rx, ry, rtheta = self.obs["robot_state"]
        gx, gy = self.obs["goal"]
        goal_dist = float(np.hypot(gx - rx, gy - ry))
        goal_bearing = float((np.arctan2(gy - ry, gx - rx) - rtheta + np.pi) % (2 * np.pi) - np.pi)
        self.dataset.append({
            "step": self.step_count,
            "robot_state": np.array([rx, ry, rtheta], dtype=float),
            "goal_dist": goal_dist,
            "goal_bearing": goal_bearing,
            "ranges": ranges,
            "action": np.array([self.v, self.omega], dtype=float),
        })

    def update(self, frame):
        if self.finished:
            return self._artists()

        self._record_step()
        self.obs, done, info = self.env.step(self.v, self.omega)
        self.step_count += 1

        if info["collided"]:
            print(f"[step {self.step_count}] COLLISION ({info['collision_kind']})")
            self.finished = True
        if done and not info["collided"]:
            print(f"[step {self.step_count}] all tasks complete, returned to depot")
            self.finished = True
        if self.step_count >= MAX_STEPS:
            self.finished = True

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

        status = (f"step {self.step_count} | v={self.v:.2f} omega={self.omega:.2f}\n"
                  f"tasks: goal_idx={self.env.goal_idx}/{len(self.env.goal_sequence)}")
        if self.finished:
            status += "\nFINISHED (press R to reset, Q to quit+save)"
        self.status_text.set_text(status)

        return self._artists()

    def _artists(self):
        return [self.robot_dot, self.robot_heading, self.path_line, self.status_text] + self.dyn_patches

    def run(self):
        ani = animation.FuncAnimation(self.fig, self.update, interval=50,
                                       blit=False, cache_frame_data=False)
        plt.show()
        return ani


if __name__ == "__main__":
    sim = ManualControlSim()
    ani = sim.run()