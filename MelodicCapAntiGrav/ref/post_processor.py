import json
import numpy as np
import os
from scipy import signal

# Import from same package
from engine.triangulation_engine import TriangulationEngine


class ScientificRefiner:
    """
    Post-processes raw motion capture takes with:
    1. Proper stereo triangulation (with undistortion)
    2. Butterworth temporal smoothing
    3. Floor alignment
    4. Gap filling
    """
    
    def __init__(self, calibration_path):
        self.engine = TriangulationEngine(calibration_path)
        self.cutoff_hz = 6.0  # Professional standard for jitter removal
        self.fps = 30.0
        self.filter_order = 4
        
        # Get image size from calibration
        self.image_width = self.engine.image_size[0]
        self.image_height = self.engine.image_size[1]

    def refine_take(self, input_path):
        """
        Process a raw JSON take with high-quality triangulation and smoothing.
        
        Args:
            input_path: Path to raw take JSON file
            
        Returns:
            Path to refined JSON file, or None on error
        """
        if not os.path.exists(input_path):
            print(f"Error: File not found: {input_path}")
            return None
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        frames = data.get("frames", [])
        if not frames:
            print("Error: No frames in take")
            return None
        
        print(f"Refining {len(frames)} frames from {os.path.basename(input_path)}")
        
        # Reset Kalman filters for fresh take
        self.engine.reset_filters()
        
        refined_frames = []
        
        for i, frame in enumerate(frames):
            # Get raw 2D landmarks (normalized 0-1)
            raw_2d = frame.get("raw_2d", {})
            lms_a = raw_2d.get("cam_a")
            lms_b = raw_2d.get("cam_b")
            
            hands_2d = frame.get("hands_2d", {})
            hands_a = hands_2d.get("cam_a")
            hands_b = hands_2d.get("cam_b")
            
            # Re-triangulate body landmarks with proper undistortion
            if lms_a and lms_b:
                points_3d = self._triangulate_landmarks(lms_a, lms_b)
                if points_3d:
                    frame["landmarks_3d"] = points_3d
            
            # Re-triangulate hands
            hands_3d = None
            if hands_a and hands_b:
                hands_3d = self._triangulate_hands(hands_a, hands_b)
            elif hands_a:
                hands_3d = self._estimate_hands_single(hands_a)
            elif hands_b:
                hands_3d = self._estimate_hands_single(hands_b)
            
            if hands_3d:
                frame["hands_3d"] = hands_3d
            
            refined_frames.append(frame)
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(frames)} frames...")
        
        # Apply Butterworth temporal smoothing
        print("Applying Butterworth filter...")
        refined_frames = self._apply_butterworth_filter(refined_frames)
        
        # Apply foot locking (optional enhancement)
        # refined_frames = self._apply_foot_locking(refined_frames)
        
        # Save refined result
        output_path = input_path.replace(".json", "_refined.json")
        data["frames"] = refined_frames
        data["refined"] = True
        data["refinement_settings"] = {
            "butterworth_cutoff_hz": self.cutoff_hz,
            "butterworth_order": self.filter_order,
            "undistortion_applied": True
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved refined take to {output_path}")
        return output_path

    def _triangulate_landmarks(self, lms_dict_a, lms_dict_b):
        """
        Triangulate body landmarks from raw 2D dictionaries.
        
        Args:
            lms_dict_a, lms_dict_b: Dicts of {idx: [x, y, z, visibility]} (normalized)
            
        Returns:
            Dict of {idx: [x, y, z]} in Blender space
        """
        pts_a = []
        pts_b = []
        indices = []
        
        # MediaPipe has 33 pose landmarks
        for i in range(33):
            s_i = str(i)
            if s_i in lms_dict_a and s_i in lms_dict_b:
                # Check visibility (index 3 in the array)
                vis_a = lms_dict_a[s_i][3] if len(lms_dict_a[s_i]) > 3 else 1.0
                vis_b = lms_dict_b[s_i][3] if len(lms_dict_b[s_i]) > 3 else 1.0
                
                if vis_a < 0.3 or vis_b < 0.3:
                    continue
                
                # CRITICAL: Convert normalized (0-1) to pixel coordinates!
                x_a = lms_dict_a[s_i][0] * self.image_width
                y_a = lms_dict_a[s_i][1] * self.image_height
                x_b = lms_dict_b[s_i][0] * self.image_width
                y_b = lms_dict_b[s_i][1] * self.image_height
                
                pts_a.append([x_a, y_a])
                pts_b.append([x_b, y_b])
                indices.append(i)
        
        if len(pts_a) < 4:
            return None
        
        # Convert to numpy arrays
        pts_a = np.array(pts_a, dtype=np.float32)
        pts_b = np.array(pts_b, dtype=np.float32)
        
        # Triangulate with proper undistortion
        result_3d = self.engine.triangulate_points(pts_a, pts_b)
        
        if result_3d is None:
            return None
        
        # Map back to dictionary with string keys (for JSON)
        points_3d = {}
        for row_idx, landmark_idx in enumerate(indices):
            pt = result_3d[row_idx]
            # Apply floor offset
            pt = self.engine.apply_floor_offset(pt)
            points_3d[str(landmark_idx)] = pt.tolist()
        
        return points_3d

    def _triangulate_hands(self, hands_a, hands_b):
        """
        Triangulate hand landmarks from raw 2D dictionaries.
        
        Args:
            hands_a, hands_b: Lists of dicts, each dict is {idx: [x, y, z]}
            
        Returns:
            List of 21x3 arrays (one per hand)
        """
        if not hands_a or not hands_b:
            return None
        
        hands_3d = []
        
        for h_a, h_b in zip(hands_a, hands_b):
            pts_a = []
            pts_b = []
            
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
            
            if len(pts_a) < 5:
                continue
            
            pts_a = np.array(pts_a, dtype=np.float32)
            pts_b = np.array(pts_b, dtype=np.float32)
            
            h_3d = self.engine.triangulate_points(pts_a, pts_b)
            
            if h_3d is not None:
                # Apply floor offset to each point
                for j in range(len(h_3d)):
                    h_3d[j] = self.engine.apply_floor_offset(h_3d[j])
                hands_3d.append(h_3d.tolist())
        
        return hands_3d if hands_3d else None

    def _estimate_hands_single(self, hands_dict):
        """
        Estimate 3D hand positions from single camera (fallback for close-ups).
        Uses MediaPipe's z-depth estimation.
        
        Args:
            hands_dict: List of dicts with {idx: [x, y, z]}
            
        Returns:
            List of 21x3 arrays
        """
        if not hands_dict:
            return None
        
        hands_3d = []
        
        for h_dict in hands_dict:
            pts = []
            for i in range(21):
                s_i = str(i)
                if s_i not in h_dict:
                    pts.append([0, 0, 0])
                    continue
                
                # MediaPipe hand z is relative depth in normalized coords
                # Scale to approximate real-world coordinates
                x = (h_dict[s_i][0] * 2 - 1) * 0.3  # ~30cm hand width
                y = (h_dict[s_i][1] * 2 - 1) * 0.3
                z = -h_dict[s_i][2] * 0.3  # Depth
                
                pts.append([x, y, z])
            
            hands_3d.append(pts)
        
        return hands_3d

    def _apply_butterworth_filter(self, frames):
        """
        Apply zero-lag Butterworth low-pass filter to 3D landmarks.
        
        Uses forward-backward filtering (filtfilt) for zero phase delay.
        """
        if len(frames) < 15:
            print("  Skipping filter: not enough frames")
            return frames
        
        # Find landmark keys from first valid frame
        lms_keys = []
        for f in frames:
            if f.get("landmarks_3d"):
                lms_keys = list(f["landmarks_3d"].keys())
                break
        
        if not lms_keys:
            return frames
        
        n_frames = len(frames)
        n_landmarks = len(lms_keys)
        
        # Create data tensor [frames, landmarks, xyz]
        data = np.full((n_frames, n_landmarks, 3), np.nan)
        
        for i, frame in enumerate(frames):
            lms = frame.get("landmarks_3d", {})
            for j, key in enumerate(lms_keys):
                if key in lms:
                    data[i, j, :] = lms[key]
        
        # Fill gaps with linear interpolation before filtering
        data = self._fill_gaps(data)
        
        # Design Butterworth filter
        nyquist = self.fps / 2
        norm_cutoff = min(self.cutoff_hz / nyquist, 0.99)
        b, a = signal.butter(self.filter_order, norm_cutoff, btype='low', analog=False)
        
        # Apply zero-lag filter (forward and backward)
        # Handle each landmark and axis separately to deal with NaN
        filtered_data = np.copy(data)
        
        for j in range(n_landmarks):
            for k in range(3):
                col = data[:, j, k]
                valid_mask = ~np.isnan(col)
                
                if np.sum(valid_mask) > self.filter_order * 3:
                    # Enough data to filter
                    filtered_data[valid_mask, j, k] = signal.filtfilt(
                        b, a, col[valid_mask]
                    )
        
        # Write back to frames
        for i, frame in enumerate(frames):
            if "landmarks_3d" not in frame:
                frame["landmarks_3d"] = {}
            
            for j, key in enumerate(lms_keys):
                if not np.isnan(filtered_data[i, j, 0]):
                    frame["landmarks_3d"][key] = filtered_data[i, j, :].tolist()
        
        return frames

    def _fill_gaps(self, data, max_gap=10):
        """
        Fill short gaps in data using linear interpolation.
        
        Args:
            data: [frames, landmarks, xyz] array with NaN for missing
            max_gap: Maximum gap length to fill
            
        Returns:
            Array with short gaps filled
        """
        n_frames, n_landmarks, _ = data.shape
        
        for j in range(n_landmarks):
            for k in range(3):
                col = data[:, j, k]
                
                # Find valid indices
                valid_idx = np.where(~np.isnan(col))[0]
                
                if len(valid_idx) < 2:
                    continue
                
                # Interpolate
                for i in range(len(valid_idx) - 1):
                    start = valid_idx[i]
                    end = valid_idx[i + 1]
                    gap = end - start - 1
                    
                    if 0 < gap <= max_gap:
                        # Linear interpolation
                        for g in range(1, gap + 1):
                            t = g / (gap + 1)
                            data[start + g, j, k] = (
                                col[start] * (1 - t) + col[end] * t
                            )
        
        return data

    def _apply_foot_locking(self, frames, velocity_threshold=0.05):
        """
        Apply foot locking to reduce sliding during ground contact.
        
        Args:
            frames: List of frame dicts
            velocity_threshold: m/frame - below this = stationary
            
        Returns:
            Frames with foot locking applied
        """
        # Ankle indices in MediaPipe
        LEFT_ANKLE = "27"
        RIGHT_ANKLE = "28"
        
        for side, idx in [("left", LEFT_ANKLE), ("right", RIGHT_ANKLE)]:
            prev_pos = None
            locked_pos = None
            lock_count = 0
            
            for frame in frames:
                lms = frame.get("landmarks_3d", {})
                
                if idx not in lms:
                    prev_pos = None
                    locked_pos = None
                    lock_count = 0
                    continue
                
                pos = np.array(lms[idx])
                
                # Ground contact: Z close to 0
                near_ground = pos[2] < 0.1
                
                if prev_pos is not None and near_ground:
                    velocity = np.linalg.norm(pos - prev_pos)
                    
                    if velocity < velocity_threshold:
                        lock_count += 1
                        if lock_count >= 3:
                            if locked_pos is None:
                                locked_pos = pos.copy()
                                locked_pos[2] = 0  # Clamp to ground
                            lms[idx] = locked_pos.tolist()
                    else:
                        locked_pos = None
                        lock_count = 0
                else:
                    locked_pos = None
                    lock_count = 0
                
                prev_pos = pos
        
        return frames
