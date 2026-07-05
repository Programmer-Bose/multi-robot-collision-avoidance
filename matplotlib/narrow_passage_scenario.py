"""
narrow_passage_scenario.py

Deterministic scenario: a narrow corridor formed by two long walls
(each built from overlapping static circular obstacles), with the robot
forced to travel through the corridor to reach the goal. Useful for
testing minimum-passable-gap behavior and deadlock recovery under
sustained confinement, as opposed to a brief single-obstacle pinch point.
"""

import numpy as np
from env import SingleRobotEnv, StaticObstacle, DynamicObstacle


def _build_wall(x_center_line, y_fixed, wall_length, obstacle_radius, overlap_ratio=0.5):
    """
    Builds a wall of overlapping circles running along the x-axis at a
    fixed y, spanning from x_center_line - wall_length/2 to
    x_center_line + wall_length/2. Two such walls at different y values
    (separated by the gap) form a horizontal corridor the robot must
    travel through.
    """
    spacing = 2 * obstacle_radius * overlap_ratio
    n = int(np.ceil(wall_length / spacing)) + 1
    xs = np.linspace(x_center_line - wall_length / 2.0,
                      x_center_line + wall_length / 2.0, n)
    return [StaticObstacle(x, y_fixed, obstacle_radius) for x in xs]

def make_narrow_passage_scenario(gap_width=0.5, robot_radius=0.15,
                                  corridor_x=5.0, obstacle_radius=0.3,
                                  wall_length=6.0, overlap_ratio=0.5,
                                  world=(0, 10, 0, 10),
                                  v_max=1.0, omega_max=np.pi / 2,
                                  add_dynamic_obstacle=False,
                                  dynamic_in_corridor=False, seed=0):
    """
    Two long walls (each a row of overlapping circles) facing each other
    across a vertical gap at x = corridor_x, separated by exactly
    `gap_width` (edge-to-edge). Robot must travel down the corridor
    lengthwise-ish to pass through, not just dodge a single obstacle.

    gap_width: edge-to-edge distance between the two wall surfaces (m).
               Compare against 2*robot_radius (hard physical limit) and
               2*(d_safe_static + robot_radius) (planner's comfortable limit).
    wall_length: total length of each wall (m), centered on the corridor midline.
    obstacle_radius: radius of each circle making up the wall.
    overlap_ratio: how tightly circles overlap along the wall (<=0.5 recommended
                   to guarantee no gaps between segments; higher risks leaks).
    """
    xmin, xmax, ymin, ymax = world
    rng = np.random.default_rng(seed)

    mid_y = (ymin + ymax) / 2.0
    center_gap = gap_width + 2 * obstacle_radius

    top_wall_x = corridor_x
    top_wall_y = mid_y + center_gap / 2.0
    bot_wall_y = mid_y - center_gap / 2.0

    top_wall = _build_wall(corridor_x, top_wall_y, wall_length,
                            obstacle_radius, overlap_ratio)
    bot_wall = _build_wall(corridor_x, bot_wall_y, wall_length,
                            obstacle_radius, overlap_ratio)
    # deadend = StaticObstacle(corridor_x+3.0, mid_y, obstacle_radius)

    static_obstacles = top_wall + bot_wall #+ [deadend]

    start = np.array([1.0, 5.0, 0.0])
    task_points = [np.array([corridor_x, mid_y])]

    dynamic_obstacles = []
    if add_dynamic_obstacle:
        if dynamic_in_corridor:
            dy = rng.choice([-1, 1]) * 0.3
            dynamic_obstacles.append(
                DynamicObstacle(corridor_x, mid_y, radius=0.15,
                                 vx=0.0, vy=dy, bounds=world, rng=rng)
            )
        else:
            dynamic_obstacles.append(
                DynamicObstacle(corridor_x - 2.0, ymax - 1.0, radius=0.25,
                                 vx=0.2, vy=-0.1, bounds=world, rng=rng)
            )

    env = SingleRobotEnv(
        start=start,
        task_points=task_points,
        static_obstacles=static_obstacles,
        dynamic_obstacles=dynamic_obstacles,
        world_bounds=world,
        v_max=v_max,
        omega_max=omega_max,
        robot_radius=robot_radius,
    )

    hard_min = 2 * robot_radius
    print(f"[narrow_passage] gap_width={gap_width:.3f} m | wall_length={wall_length} m | "
          f"segments/wall={len(top_wall)} | hard physical minimum={hard_min:.3f} m | "
          f"{'PASSABLE' if gap_width > hard_min else 'IMPOSSIBLE'} for this robot size")

    return env


if __name__ == "__main__":
    for gw in [0.9, 0.6, 0.4, 0.32, 0.25]:
        make_narrow_passage_scenario(gap_width=gw, robot_radius=0.15)