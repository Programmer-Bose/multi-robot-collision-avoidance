"""
Plot every curve in the initial (cold-start) DE population, for every
segment of a map, without running any optimization.

Each waypoint-to-waypoint segment is treated as a cold start (no
previous-segment warm start), so its initial population is generated purely
by build_fanned_out_population(). All curves for all segments are drawn
together on one map.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle

import dual_de_bspline_la_map_global as m

POPSIZE = 50          # population size to visualize per segment (matches DE_POPSIZE_INITIAL by default)


def plot_initial_population(map_json, popsize=POPSIZE, control_point_mode=None):
    if control_point_mode is not None:
        m.CONTROL_POINT_MODE = control_point_mode

    m.load_map_config(map_json)

    fig, ax = plt.subplots(figsize=(9, 7))

    # Obstacles
    for obstacle in m.OBSTACLES:
        if obstacle["type"] == "circle":
            cx, cy = obstacle["center"]
            r = obstacle["radius"]
            ax.add_patch(Circle((cx, cy), r, color="firebrick", alpha=0.55, zorder=2))
        else:
            poly = obstacle["shape"]
            xs, ys = poly.exterior.xy
            ax.add_patch(MplPolygon(list(zip(xs, ys)), closed=True, color="firebrick", alpha=0.55, zorder=2))

    # Waypoints
    for i, wp in enumerate(m.WAYPOINTS):
        if i == 0:
            ax.plot(*wp, "gs", markersize=13, zorder=5, label="Start")
        elif i == len(m.WAYPOINTS) - 1:
            ax.plot(*wp, "r*", markersize=20, zorder=5, label="Goal")
        else:
            ax.plot(*wp, "D", color="orange", markersize=11, zorder=5, label=f"Task {i}")

    seg_colors = plt.cm.viridis(np.linspace(0, 0.85, max(m.N_SEGMENTS, 1)))

    for seg in range(m.N_SEGMENTS):
        seg_start = m.WAYPOINTS[seg]
        seg_end = m.WAYPOINTS[seg + 1]

        n_individuals = popsize * m.N_VARS_PER_SEGMENT
        population = m.build_fanned_out_population(
            n_individuals, m.N_VARS_PER_SEGMENT, seg_start, seg_end
        )

        for individual in population:
            curve = m.segment_curve_from_deltas(individual, seg_start, seg_end)
            ax.plot(curve[:, 0], curve[:, 1], "-", color=seg_colors[seg], alpha=0.25, linewidth=1.0, zorder=3)

    ax.set_xlim(m.BOUNDS_MIN[0], m.BOUNDS_MAX[0])
    ax.set_ylim(m.BOUNDS_MIN[1], m.BOUNDS_MAX[1])
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(f"Initial (cold-start) population per segment [mode={m.CONTROL_POINT_MODE}, popsize={popsize}]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_initial_population("maps/env_map_config_024.json")
