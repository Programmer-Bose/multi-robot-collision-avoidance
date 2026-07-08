import tkinter as tk
import json
import os

class MapGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("GEN AI UAV Swarm - Map Generator")
        self.root.geometry("1000x800")
        self.root.configure(bg="#121212")

        # Map Configurations
        self.map_width = 800
        self.map_height = 800
        
        # Data Storage
        self.start_pos = None
        self.goal_pos = None
        self.tasks = []
        self.obstacles = []

        self.setup_ui()

    def setup_ui(self):
        # Tools Panel (Left)
        self.tool_frame = tk.Frame(self.root, width=200, bg="#1e1e1e", padx=10, pady=10)
        self.tool_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(self.tool_frame, text="Select Mode", fg="#00FFFF", bg="#1e1e1e", font=("Arial", 14, "bold")).pack(pady=10)

        self.mode = tk.StringVar(value="Start")
        modes = ["Start (Orange)", "Goal (Red)", "Task (Cyan)", "Square", "Rectangle", "Circle", "U-Shape"]
        
        for m in modes:
            tk.Radiobutton(self.tool_frame, text=m, variable=self.mode, value=m.split()[0], 
                           fg="#ffffff", bg="#1e1e1e", selectcolor="#333333", font=("Arial", 12)).pack(anchor=tk.W, pady=5)

        # Task Sequence Input
        tk.Label(self.tool_frame, text="Task Sequence (e.g. 2,1,3):", fg="#ffffff", bg="#1e1e1e", font=("Arial", 10)).pack(pady=(20, 5), anchor=tk.W)
        self.seq_entry = tk.Entry(self.tool_frame, bg="#333333", fg="#ffffff", insertbackground="#00FFFF", font=("Arial", 12))
        self.seq_entry.pack(fill=tk.X)

        tk.Button(self.tool_frame, text="Clear Map", command=self.clear_map, bg="#333333", fg="#ffffff").pack(pady=(30, 10), fill=tk.X)
        tk.Button(self.tool_frame, text="Save JSON", command=self.save_to_json, bg="#FFA500", fg="#000000", font=("Arial", 12, "bold")).pack(pady=10, fill=tk.X)

        # Canvas (Right)
        self.canvas = tk.Canvas(self.root, width=self.map_width, height=self.map_height, bg="#0a0a0a", highlightthickness=1, highlightbackground="#00FFFF")
        self.canvas.pack(side=tk.RIGHT, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def on_canvas_click(self, event):
        x, y = event.x, event.y
        current_mode = self.mode.get()

        if current_mode == "Start":
            self.start_pos = [x, y]
            self.draw_point(x, y, "#FFA500", "start")
        elif current_mode == "Goal":
            self.goal_pos = [x, y]
            self.draw_point(x, y, "#FF0000", "goal")
        elif current_mode == "Task":
            task_id = len(self.tasks) + 1
            self.tasks.append([x, y])
            self.draw_point(x, y, "#00FFFF", "task")
            # Draw the task number
            self.canvas.create_text(x + 12, y - 12, text=str(task_id), fill="#00FFFF", font=("Arial", 12, "bold"), tags="task")
        elif current_mode == "Square":
            w, h = 100, 50
            self.obstacles.append({"type": "rectangle", "position": [x, y], "width": w, "height": h})
            self.canvas.create_rectangle(x - w/2, y - h/2, x + w/2, y + h/2, fill="#333333", outline="#00FFFF", tags="obstacle")
        elif current_mode == "Rectangle":
            w, h = 50, 100
            self.obstacles.append({"type": "rectangle", "position": [x, y], "width": w, "height": h})
            self.canvas.create_rectangle(x - w/2, y - h/2, x + w/2, y + h/2, fill="#333333", outline="#00FFFF", tags="obstacle")
        elif current_mode == "Circle":
            radius = 40
            self.obstacles.append({"type": "circle", "position": [x, y], "radius": radius})
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="#333333", outline="#00FFFF", tags="obstacle")
        elif current_mode == "U-Shape":
            size = 120
            thickness = 20
            self.obstacles.append({"type": "u_shape", "position": [x, y], "size": size, "thickness": thickness})
            # Draw U-shape using 3 rectangles
            self.canvas.create_rectangle(x - size/2, y - size/2, x - size/2 + thickness, y + size/2, fill="#333333", outline="#00FFFF", tags="obstacle") # Left
            self.canvas.create_rectangle(x + size/2 - thickness, y - size/2, x + size/2, y + size/2, fill="#333333", outline="#00FFFF", tags="obstacle") # Right
            self.canvas.create_rectangle(x - size/2, y + size/2 - thickness, x + size/2, y + size/2, fill="#333333", outline="#00FFFF", tags="obstacle") # Bottom

    def draw_point(self, x, y, color, tag):
        if tag in ["start", "goal"]:
            self.canvas.delete(tag)
        self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill=color, tags=tag)

    def clear_map(self):
        self.canvas.delete("all")
        self.start_pos = None
        self.goal_pos = None
        self.tasks = []
        self.obstacles = []
        self.seq_entry.delete(0, tk.END)

    def save_to_json(self):
        if not self.start_pos:
            print("Error: Please set a Start position before saving.")
            return

        # Determine task sequence
        seq_str = self.seq_entry.get().strip()
        task_sequence = []
        
        if not seq_str:
            # Default to ascending order if empty
            task_sequence = list(range(1, len(self.tasks) + 1))
        else:
            try:
                # Parse the comma-separated string
                task_sequence = [int(s.strip()) for s in seq_str.split(",")]
            except ValueError:
                print("Warning: Invalid sequence format entered. Defaulting to ascending order.")
                task_sequence = list(range(1, len(self.tasks) + 1))

        map_data = {
            "map_metadata": {
                "size": [self.map_width, self.map_height],
                "num_obstacles": len(self.obstacles),
                "num_tasks": len(self.tasks)
            },
            "start_position": self.start_pos,
            "goal_position": self.goal_pos,
            "task_points": {i+1: pos for i, pos in enumerate(self.tasks)},
            "task_sequence": task_sequence,
            "obstacles": self.obstacles
        }

        filename = "maps/env_map_config_{:03d}.json".format(len(self.obstacles))
        with open(filename, 'w') as f:
            json.dump(map_data, f, indent=4)
        print(f"Map successfully saved to {os.path.abspath(filename)}")
        print(f"Active Task Sequence: {task_sequence}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MapGenerator(root)
    root.mainloop()