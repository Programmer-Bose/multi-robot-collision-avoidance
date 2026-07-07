# env.py---- Gym-style environment: at each step the agent sees the map, remaining tasks (with precedence mask), 
# and current position, then outputs a task choice + 3 Bezier control points; reward penalizes length, energy, and collisions.
import numpy as np
from bezier_utils import bezier_curve
from collision_utils import collision_penalty


def build_default_precedence(n_tasks):
    precedence = np.zeros((n_tasks, n_tasks), dtype=bool)
    for i in range(max(0, n_tasks - 1)):
        precedence[i, i + 1] = True
    return precedence


class GlobalPathEnv:
    def __init__(self, robot_radius=0.3, bounds=(0, 12)):
        self.robot_radius = robot_radius
        self.bounds = bounds

    def reset(self, obstacles, grid, task_points, precedence, start_pos, w):
        """
        obstacles: list of obstacle dicts (for collision checking)
        grid: (64,64) rasterized map for CNN input
        task_points: (N,2) array of task coordinates
        precedence: (N,N) bool matrix; precedence[i,j]=True means task i must
                    be visited before task j becomes available
        start_pos: (2,) robot start position
        w: (3,) preference vector [w_length, w_energy, w_collision]
        """
        self.obstacles = obstacles
        self.grid = grid
        self.task_points = np.asarray(task_points, dtype=float)
        self.n_tasks = len(self.task_points)
        self.precedence = self._normalize_precedence(precedence)
        self.visited = np.zeros(self.n_tasks, dtype=bool)
        self.current_pos = np.asarray(start_pos, dtype=float).copy()
        self.w = w
        self.done = False
        return self._get_state()

    def _normalize_precedence(self, precedence):
        if precedence is None:
            return build_default_precedence(self.n_tasks)

        precedence = np.array(precedence, dtype=bool)
        if precedence.shape != (self.n_tasks, self.n_tasks):
            new_precedence = np.zeros((self.n_tasks, self.n_tasks), dtype=bool)
            size = min(self.n_tasks, precedence.shape[0], precedence.shape[1])
            if size > 0:
                new_precedence[:size, :size] = precedence[:size, :size]
            return new_precedence
        return precedence

    def _get_state(self):
        # tasks whose ALL precedence-required tasks are already visited
        available = np.array([
            (not self.visited[j]) and
            all(self.visited[i] for i in range(self.n_tasks) if self.precedence[i, j])
            for j in range(self.n_tasks)
        ])
        return {
            "grid": self.grid,
            "current_pos": self.current_pos,
            "task_points": self.task_points,
            "visited": self.visited.copy(),
            "available_mask": available,
            "w": self.w,
        }

    def step(self, task_idx, control_points):
        """control_points: (3,2) array predicted by the policy for this segment."""
        goal = self.task_points[task_idx]
        curve = bezier_curve(self.current_pos, control_points, goal)

        length = np.sum(np.hypot(*np.diff(curve, axis=0).T))
        energy = length  # simple proxy; replace with a real energy model later
        coll = collision_penalty(curve, self.obstacles, self.robot_radius)

        reward_vec = np.array([-length, -energy, -800.0 * coll])
        reward = float(np.dot(self.w, reward_vec))

        self.visited[task_idx] = True
        self.current_pos = goal
        self.done = bool(np.all(self.visited))

        info = {"curve": curve, "reward_vec": reward_vec}
        return self._get_state(), reward, self.done, info
    
if __name__ == "__main__":
    import numpy as np
    from map_utils import generate_random_map, rasterize_map
    from env import GlobalPathEnv

    obstacles = generate_random_map(seed=1)
    grid = rasterize_map(obstacles)
    task_points = np.array([[3, 8], [7, 2], [10, 9]])
    precedence = np.zeros((3, 3), dtype=bool)
    precedence[0, 1] = True  # task 0 must precede task 1

    env = GlobalPathEnv()
    state = env.reset(obstacles, grid, task_points, precedence,
                    start_pos=np.array([0.0, 0.0]), w=np.array([0.6, 0.2, 0.2]))
    print(state["available_mask"])  # [True, False, True] -> task1 blocked until task0 done

    cps = np.array([[2, 3], [4, 5], [3, 7]])
    next_state, reward, done, info = env.step(0, cps)
    print(reward, done)