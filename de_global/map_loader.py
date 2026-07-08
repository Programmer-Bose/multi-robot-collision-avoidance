import json
import os
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

def load_and_plot_map(json_path="env_map_config.json"):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Please generate a map first.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Scaling Factors: 800x800 pixels -> 12x12 units
    orig_w, orig_h = data["map_metadata"]["size"]
    target_scale = 12.0
    scale_factor = target_scale / orig_w  # 12 / 800

    # Helper function to scale and invert Y-axis (Tkinter to standard Cartesian)
    def scale_pt(pt):
        x_scaled = pt[0] * scale_factor
        y_scaled = (orig_h - pt[1]) * scale_factor  # Invert Y
        return np.array([x_scaled, y_scaled])

    # 1. Initialize Matplotlib Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#ffffff")  # White background
    fig.patch.set_facecolor("#ffffff")

    # 2. Parse and Plot Obstacles using Shapely
    shapely_obstacles = []

    for obs in data["obstacles"]:
        obs_type = obs["type"]
        cx, cy = scale_pt(obs["position"])
        
        if obs_type == "circle":
            r_scaled = obs["radius"] * scale_factor
            geom = Point(cx, cy).buffer(r_scaled)
            shapely_obstacles.append(geom)

        elif obs_type == "square":
            size_scaled = obs["size"] * scale_factor
            h = size_scaled / 2.0
            geom = Polygon([
                (cx - h, cy - h), (cx + h, cy - h), 
                (cx + h, cy + h), (cx - h, cy + h)
            ])
            shapely_obstacles.append(geom)

        elif obs_type == "rectangle":
            w_scaled = obs["width"] * scale_factor
            h_scaled = obs["height"] * scale_factor
            hw, hh = w_scaled / 2.0, h_scaled / 2.0
            geom = Polygon([
                (cx - hw, cy - hh), (cx + hw, cy - hh), 
                (cx + hw, cy + hh), (cx - hw, cy + hh)
            ])
            shapely_obstacles.append(geom)

        elif obs_type == "u_shape":
            size_scaled = obs["size"] * scale_factor
            t_scaled = obs["thickness"] * scale_factor
            h = size_scaled / 2.0
            
            # Reconstruct the 3 parts of the U-Shape (taking Y-inversion into account)
            # Left arm
            left_arm = Polygon([
                (cx - h, cy - h), (cx - h + t_scaled, cy - h),
                (cx - h + t_scaled, cy + h), (cx - h, cy + h)
            ])
            # Right arm
            right_arm = Polygon([
                (cx + h - t_scaled, cy - h), (cx + h, cy - h),
                (cx + h, cy + h), (cx + h - t_scaled, cy + h)
            ])
            # Bottom connecting bar
            bottom_bar = Polygon([
                (cx - h, cy - h), (cx + h, cy - h),
                (cx + h, cy - h + t_scaled), (cx - h, cy - h + t_scaled)
            ])
            
            # Combine them into a single U-shaped geometry
            u_geom = unary_union([left_arm, right_arm, bottom_bar])
            shapely_obstacles.append(u_geom)

    # Draw all obstacles in solid black
    for geom in shapely_obstacles:
        x, y = geom.exterior.xy
        ax.fill(x, y, color="#000000", edgecolor="#000000", zorder=2)

    # 3. Plot Kinematic & Task Elements
    start = scale_pt(data["start_position"])
    ax.scatter(start[0], start[1], color="#FFA500", marker="s", s=150, edgecolors="black", label="Start", zorder=4)

    if data.get("goal_position"):
        goal = scale_pt(data["goal_position"])
        ax.scatter(goal[0], goal[1], color="#FF0000", marker="*", s=250, edgecolors="black", label="Goal", zorder=4)

    # Parse task positions mapping
    task_points = data["task_points"]
    sequence = data["task_sequence"]

    # Map positions to sequence order
    ordered_path = [start]  # To visualize the order pathway if desired
    
    for rank, task_id in enumerate(sequence):
        # JSON keys become strings when saved, so cast task_id to string
        raw_pos = task_points[str(task_id)]
        task_scaled = scale_pt(raw_pos)
        ordered_path.append(task_scaled)

        # Plot cyan task node
        ax.scatter(task_scaled[0], task_scaled[1], color="#00FFFF", marker="o", s=100, edgecolors="black", zorder=3)
        # Overlay sequence rank text (e.g., "1st", "2nd" or just its execution step index)
        ax.text(task_scaled[0] + 0.2, task_scaled[1] + 0.2, f"#{rank+1} (T{task_id})", 
                color="#006666", fontsize=10, fontweight="bold", zorder=5)

    if data.get("goal_position"):
        ordered_path.append(scale_pt(data["goal_position"]))

    # Optional: Draw a subtle dashed pathway line indicating the sequence order
    path_matrix = np.array(ordered_path)
    ax.plot(path_matrix[:, 0], path_matrix[:, 1], ":", color="#7f8c8d", alpha=0.6, label="Sequence Track", zorder=1)

    # 4. Grid, Limits and Scaling Configurations
    ax.set_xlim(0, target_scale)
    ax.set_ylim(0, target_scale)
    ax.set_xticks(np.arange(0, target_scale + 1, 1.0))
    ax.set_yticks(np.arange(0, target_scale + 1, 1.0))
    
    # Grid customization
    ax.grid(True, which='both', color='#cccccc', linestyle='--', linewidth=0.7, zorder=1)
    ax.set_aspect('equal')
    
    ax.set_title(f"Scaled Environment Map (0 to {int(target_scale)})", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    load_and_plot_map("maps/env_map_config_011.json")