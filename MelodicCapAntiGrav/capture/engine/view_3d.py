import numpy as np

class Skeleton3DView:
    def __init__(self, canvas_width, canvas_height):
        self.width = canvas_width
        self.height = canvas_height
        self.scale = 200
        self.offset_x = canvas_width // 2
        self.offset_y = canvas_height // 2 + 100
        
        # Skeleton connections (MediaPipe indices)
        self.connections = [
            (11, 12), (11, 13), (13, 15), # Left Arm
            (12, 14), (14, 16),           # Right Arm
            (11, 23), (12, 24), (23, 24), # Torso
            (23, 25), (25, 27),           # Left Leg
            (24, 26), (26, 28)            # Right Leg
        ]

    def project_3d_to_2d(self, points_3d):
        """
        Calculates projection with auto-centering and adaptive scaling.
        """
        projected = {}
        if not points_3d: return projected
        
        # 1. Filter Outliers for Centering
        vals = []
        for p in points_3d.values():
            if abs(p[0]) < 10 and abs(p[1]) < 10: # Sanity check for metric data
                vals.append(p)
        
        if not vals: vals = list(points_3d.values())
        
        avg_x = np.mean([p[0] for p in vals])
        avg_z = np.mean([p[2] for p in vals])
        
        # Dynamic Scaling: If data is normalized (0-1), scale it up. If metric, use 400.
        is_normalized = all(0 <= abs(p[0]) <= 1 and 0 <= abs(p[1]) <= 1 for p in vals[:5])
        s = 600 if is_normalized else 350
        
        for idx, pt in points_3d.items():
            # pt is [X, Y_Forward, Z_Up] in Blender space
            # 1. Center at Hip (origin)
            x = pt[0] - avg_x
            y_fwd = pt[1] - avg_z # Using avg_z for Y-depth center
            z_up = pt[2] 
            
            # 2. Simple rotation for perspective (rotate around Z-Up)
            ang = np.radians(20)
            rx = x * np.cos(ang) + y_fwd * np.sin(ang)
            
            # 3. Project to 2D
            x_2d = rx * s + self.offset_x
            y_2d = -z_up * s + self.offset_y # Z-Up is UI vertical
            
            projected[idx] = (int(x_2d), int(y_2d))
            
        return projected

    def draw_skeleton(self, canvas, projected_points):
        """
        canvas: Tkinter canvas object
        """
        canvas.delete("skeleton")
        
        # Draw connections
        for start_idx, end_idx in self.connections:
            if start_idx in projected_points and end_idx in projected_points:
                p1 = projected_points[start_idx]
                p2 = projected_points[end_idx]
                canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="#3a86ff", width=3, tags="skeleton")
                
        # Draw joints
        for idx, pt in projected_points.items():
            r = 4
            canvas.create_oval(pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r, fill="#ff4d4d", outline="white", tags="skeleton")
