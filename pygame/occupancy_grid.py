"""
Egocentric binary occupancy grid.

Grid is centered on the robot and rotated so that "forward" (robot heading)
always points along +row direction (i.e. heading-aligned, not world-aligned).
Each cell is `resolution` meters square. A cell is 1 if any obstacle
(static or dynamic) overlaps it, else 0.

This mirrors the local, onboard-only sensing model used by rangefinder.py --
no obstacle identity or velocity is encoded, just occupied/free geometry,
so it is consistent with the teacher-student (privileged DE-MPC / onboard-only
DRL) split described in the project design.
"""

import numpy as np


def compute_occupancy_grid(robot_state, static_obstacles, dynamic_obstacles,
                            grid_size=21, resolution=0.25):
    """
    robot_state: (x, y, theta)
    static_obstacles: list of (x, y, radius)
    dynamic_obstacles: list of (x, y, radius, vx, vy)
    Returns: (grid_size, grid_size) float32 array, 1.0 = occupied, 0.0 = free.

    Grid layout: cell [i, j] corresponds to the local-frame offset
        forward = (i - center) * resolution
        lateral = (j - center) * resolution
    where forward is along the robot's heading and lateral is 90 deg left
    of heading (standard robotics body frame: x-forward, y-left).
    """
    x, y, theta = robot_state
    center = grid_size // 2
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    cos_t, sin_t = np.cos(theta), np.sin(theta)

    circles = [(ox, oy, r) for (ox, oy, r) in static_obstacles]
    circles += [(ox, oy, r) for (ox, oy, r, vx, vy) in dynamic_obstacles]

    if not circles:
        return grid

    # cell centers in local (forward, lateral) frame
    offsets = (np.arange(grid_size) - center) * resolution  # shape (grid_size,)
    fwd_grid, lat_grid = np.meshgrid(offsets, offsets, indexing="ij")  # (G,G) each

    # local -> world:  world = robot_pos + R(theta) @ [forward, lateral]
    # R(theta) maps body x-forward,y-left to world using heading theta
    world_x = x + fwd_grid * cos_t - lat_grid * sin_t
    world_y = y + fwd_grid * sin_t + lat_grid * cos_t

    half_cell = resolution / 2.0
    # a cell is occupied if the obstacle circle comes within half-cell-diagonal
    # of the cell center (cheap conservative approximation to true overlap)
    cell_reach = half_cell * np.sqrt(2)

    for (ox, oy, r) in circles:
        d = np.hypot(world_x - ox, world_y - oy)
        grid[d <= (r + cell_reach)] = 1.0

    return grid


if __name__ == "__main__":
    # quick standalone sanity check
    robot_state = (2.0, 2.0, 0.0)
    static_obs = [(3.0, 2.0, 0.4)]
    dyn_obs = [(2.0, 3.5, 0.25, 0.1, -0.1)]
    grid = compute_occupancy_grid(robot_state, static_obs, dyn_obs,
                                   grid_size=21, resolution=0.25)
    print("Occupied cells:", int(grid.sum()), "/", grid.size)
    print(grid.astype(int))
