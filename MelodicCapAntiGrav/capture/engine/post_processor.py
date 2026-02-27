import json
import numpy as np
import os
import cv2
from engine.triangulation_engine import TriangulationEngine
from scipy import signal

class ScientificRefiner:
    def __init__(self, calibration_path):
        self.engine = TriangulationEngine(calibration_path)
        self.cutoff_hz = 6.0  # Professional standard for jitter removal
        self.fps = 30.0
        self.filter_order = 4
        self.image_width = self.engine.image_size[0]
        self.image_height = self.engine.image_size[1]
        
    def refine_take(self, input_path):
        """
        Processes a raw JSON take, performing high-quality triangulation
        with temporal coherence (smoothing).
        """
        if not os.path.exists(input_path):
            return None
            
        with open(input_path, 'r') as f:
            data = json.load(f)
            
        frames = data.get("frames", [])
        if not frames: return None
        
        print(f"Refining {len(frames)} frames for {input_path}")
        
        # 1. Temporal Smoothing on 2D landmarks (Gaussian-ish)
        # We collect all 2D points into a time-series for smoothing
        refined_frames = []
        
        # Mapping for result
        for i, frame in enumerate(frames):
            lms_a = frame.get("raw_2d", {}).get("cam_a")
            lms_b = frame.get("raw_2d", {}).get("cam_b")
            hands_a = frame.get("hands_2d", {}).get("cam_a")
            hands_b = frame.get("hands_2d", {}).get("cam_b")
            
            if lms_a and lms_b:
                points_3d = self._triangulate_raw(lms_a, lms_b)
                if points_3d:
                    # Apply floor offset if calibrated
                    for idx in points_3d:
                        points_3d[idx] = self.engine.apply_floor_offset(points_3d[idx])
                    frame["landmarks_3d"] = points_3d
            
            # Refine Hands (Finger-Fidelity)
            if hands_a and hands_b:
                hands_3d = self._triangulate_hands_raw(hands_a, hands_b)
            elif hands_a:
                # Single-Camera Fallback for Close-ups (Simple 2D->3D mapping)
                hands_3d = self._estimate_hands_single(hands_a)
            elif hands_b:
                hands_3d = self._estimate_hands_single(hands_b)
            else:
                hands_3d = None
                
            if hands_3d:
                # Apply floor offset to hands too
                for hand in hands_3d:
                    for i in range(len(hand)):
                        hand[i] = self.engine.apply_floor_offset(hand[i])
                frame["hands_3d"] = hands_3d
            
            refined_frames.append(frame)
            
        # 2. Apply Professional Butterworth Smoothing
        refined_frames = self._apply_butterworth_filter(refined_frames)
        
        # 3. Save result
        output_path = input_path.replace(".json", "_refined.json")
        data["frames"] = refined_frames
        data["refined"] = True
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        return output_path

    def _triangulate_raw(self, lms_dict_a, lms_dict_b):
        """Helper to triangulate from raw dictionary data"""
        pts_a = []
        pts_b = []
        indices = []
        
        # MediaPipe has 33 landmarks
        for i in range(33):
            s_i = str(i)
            if s_i in lms_dict_a and s_i in lms_dict_b:
                # [x, y, z, visibility]
                # CRITICAL: Convert normalized (0-1) to pixel coordinates!
                x_a = lms_dict_a[s_i][0] * self.image_width
                y_a = lms_dict_a[s_i][1] * self.image_height
                x_b = lms_dict_b[s_i][0] * self.image_width
                y_b = lms_dict_b[s_i][1] * self.image_height
                
                pts_a.append([x_a, y_a])
                pts_b.append([x_b, y_b])
                indices.append(i)
        
        if len(pts_a) < 4: return None
        
        # Convert to numpy for OpenCV triangulation
        pts_a = np.array(pts_a).reshape(-1, 1, 2)
        pts_b = np.array(pts_b).reshape(-1, 1, 2)
        
        # Use existing engine logic
        res_3d = self.engine.triangulate_points(pts_a, pts_b)
        
        # Map back to dict
        points_3d = {}
        for row_idx, landmark_idx in enumerate(indices):
            points_3d[str(landmark_idx)] = res_3d[row_idx].tolist()
            
        return points_3d

    def _triangulate_hands_raw(self, hands_a, hands_b):
        """Helper to triangulate multiple hands from raw dictionary data"""
        if not hands_a or not hands_b: return None
        
        hands_3d = []
        for h_a, h_b in zip(hands_a, hands_b):
            pts_a, pts_b = [], []
            for i in range(21):
                s_i = str(i)
                if s_i in h_a and s_i in h_b:
                    # CRITICAL: Convert normalized to pixels!
                    x_a = h_a[s_i][0] * self.image_width
                    y_a = h_a[s_i][1] * self.image_height
                    x_b = h_b[s_i][0] * self.image_width
                    y_b = h_b[s_i][1] * self.image_height
                    
                    pts_a.append([x_a, y_a])
                    pts_b.append([x_b, y_b])
            
            if len(pts_a) < 5: continue
            
            pts_a = np.array(pts_a).reshape(-1, 1, 2)
            pts_b = np.array(pts_b).reshape(-1, 1, 2)
            
            h_3d_res = self.engine.triangulate_points(pts_a, pts_b)
            hands_3d.append(h_3d_res.tolist())
            
        return hands_3d

    def _apply_butterworth_filter(self, frames):
        """Applies a zero-lag Butterworth low-pass filter to the 3D landmarks"""
        if len(frames) < 15: return frames # Not enough depth for filter
        
        # 1. Gather all points into a numpy array [frames, landmarks, xyz]
        n_frames = len(frames)
        # Use first valid frame to determine landmark count
        lms_keys = []
        for f in frames:
            if f.get("landmarks_3d"):
                lms_keys = list(f["landmarks_3d"].keys())
                break
        
        if not lms_keys: return frames
        
        # Create data tensor
        data = np.zeros((n_frames, len(lms_keys), 3))
        for i, frame in enumerate(frames):
            lms = frame.get("landmarks_3d", {})
            for j, key in enumerate(lms_keys):
                if key in lms:
                    data[i, j, :] = lms[key]
                else:
                    # Keep previous or zero (simple gap fill)
                    data[i, j, :] = data[i-1, j, :] if i > 0 else 0

        # 2. Design filter
        nyquist = self.fps / 2
        norm_cutoff = self.cutoff_hz / nyquist
        b, a = signal.butter(self.filter_order, norm_cutoff, btype='low', analog=False)
        
        # 3. Apply zero-lag filter (forward and backward)
        filtered_data = signal.filtfilt(b, a, data, axis=0)
        
        # 4. Write back to frames
        for i, frame in enumerate(frames):
            if "landmarks_3d" not in frame: frame["landmarks_3d"] = {}
            for j, key in enumerate(lms_keys):
                frame["landmarks_3d"][key] = filtered_data[i, j, :].tolist()
                
        return frames

    def _estimate_hands_single(self, hands_dict):
        """Estimate 3D hand positions from a single camera (normalized coords)"""
        hands_3d = []
        for h_dict in hands_dict:
            pts = []
            for i in range(21):
                s_i = str(i)
                # Convert normalized to "meter-like" scale for Blender
                # Simple projection: x, y, -x/2 for depth (heuristic)
                pt = [h_dict[s_i][0] * 2 - 1, h_dict[s_i][1] * 2 - 1, -h_dict[s_i][2]]
                pts.append(pt)
            hands_3d.append(pts)
        return hands_3d
