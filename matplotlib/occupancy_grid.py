import numpy as np

def compute_occupancy_grid(robot_state, static_obstacles, dynamic_obstacles,
                            grid_size=21, resolution=0.25):
    """
    Returns a (grid_size, grid_size) binary array, egocentric and
    heading-aligned: robot at center, facing 'up' (+row direction).
    1 = occupied, 0 = free.
    """
    x, y, theta = robot_state
    half = grid_size // 2
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)

    all_obs = [(ox, oy, r) for (ox, oy, r) in static_obstacles]
    all_obs += [(ox, oy, r) for (ox, oy, r, vx, vy) in dynamic_obstacles]

    cos_t, sin_t = np.cos(-theta), np.sin(-theta)  # rotate world->robot frame

    for (ox, oy, r) in all_obs:
        dx, dy = ox - x, oy - y
        # rotate into robot-centric, heading-aligned frame
        rx = dx * cos_t - dy * sin_t
        ry = dx * sin_t + dy * cos_t

        # convert to grid cell range covered by the obstacle's radius
        cx = rx / resolution
        cy = ry / resolution
        cr = r / resolution

        row_c = half + int(round(cy))   # forward = +row
        col_c = half + int(round(cx))   # right   = +col

        rad_cells = int(np.ceil(cr)) + 1
        for dr in range(-rad_cells, rad_cells + 1):
            for dc in range(-rad_cells, rad_cells + 1):
                row, col = row_c + dr, col_c + dc
                if 0 <= row < grid_size and 0 <= col < grid_size:
                    # distance from cell center to obstacle center, in cell units
                    dist = np.hypot((col - col_c), (row - row_c))
                    if dist <= cr:
                        grid[row, col] = 1

    return grid