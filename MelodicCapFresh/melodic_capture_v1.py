"""
MelodicCap Studio v1.0 - Clean Rewrite
======================================
A simple, working motion capture recorder for Melodic Justice.

Hardware:
- Camera A (Sony ZV-1F): Index 2
- Camera B (Samsung S25 via DroidCam): Index 4

Usage:
1. Run this script
2. Press 'C' to calibrate cameras (show ChArUco board to both)
3. Press 'F' to calibrate floor (place board on ground)
4. Press 'R' to start/stop recording
5. Press 'Q' to quit

Output: JSON files ready for Blender import
"""

import cv2
import numpy as np
import mediapipe as mp
import json
import time
import os
from datetime import datetime
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """All settings in one place"""
    
    # Camera indices (change these to match your system)
    CAM_A_INDEX = 2  # Sony ZV-1F
    CAM_B_INDEX = 4  # Samsung S25 DroidCam
    
    # Resolution
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FPS = 30
    
    # Paths
    BASE_DIR = Path(r"C:\Users\ninja\Documents\MelodicCapStudio\MelodicCapFresh")
    CALIBRATION_FILE = BASE_DIR / "calibration" / "stereo_calibration.json"
    TAKES_DIR = BASE_DIR / "takes"
    
    # ChArUco board (must match your printed board!)
    # Calibration board: 63.5mm squares, 47.6mm markers
    # Floor board: 47.6mm squares, 35.6mm markers
    CHARUCO_SQUARES_X = 4
    CHARUCO_SQUARES_Y = 3
    CHARUCO_SQUARE_SIZE = 0.0635  # 63.5mm in meters (CALIBRATION BOARD)
    CHARUCO_MARKER_SIZE = 0.0476  # 47.6mm in meters
    ARUCO_DICT = cv2.aruco.DICT_4X4_50
    
    # MediaPipe settings
    MP_MODEL_COMPLEXITY = 1  # 0=lite, 1=full, 2=heavy
    MP_MIN_DETECTION_CONF = 0.5
    MP_MIN_TRACKING_CONF = 0.5
    
    # Smoothing
    KALMAN_PROCESS_NOISE = 1e-4
    KALMAN_MEASUREMENT_NOISE = 1e-2


# =============================================================================
# SIMPLE KALMAN FILTER
# =============================================================================

class SimpleKalman:
    """1D Kalman filter for smoothing"""
    
    def __init__(self, q=1e-4, r=1e-2):
        self.q = q  # Process noise
        self.r = r  # Measurement noise
        self.x = 0.0  # State estimate
        self.p = 1.0  # Error covariance
        self.initialized = False
    
    def update(self, measurement):
        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return self.x
        
        # Predict
        self.p += self.q
        
        # Update
        k = self.p / (self.p + self.r)
        self.x += k * (measurement - self.x)
        self.p *= (1 - k)
        
        return self.x
    
    def reset(self):
        self.initialized = False
        self.x = 0.0
        self.p = 1.0


# =============================================================================
# STEREO CALIBRATION
# =============================================================================

class StereoCalibration:
    """Handles stereo camera calibration and triangulation"""
    
    def __init__(self):
        self.is_calibrated = False
        
        # Calibration matrices
        self.K1 = None  # Camera A intrinsic
        self.K2 = None  # Camera B intrinsic
        self.D1 = None  # Camera A distortion
        self.D2 = None  # Camera B distortion
        self.R = None   # Rotation between cameras
        self.T = None   # Translation between cameras
        self.R1 = None  # Rectification rotation A
        self.R2 = None  # Rectification rotation B
        self.P1 = None  # Projection matrix A
        self.P2 = None  # Projection matrix B
        
        # Floor calibration
        self.floor_z_offset = 0.0
        
        # Kalman filters for each landmark (33 pose + optional hands)
        self.filters = {}
        
        # ChArUco board setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(Config.ARUCO_DICT)
        self.charuco_board = cv2.aruco.CharucoBoard(
            (Config.CHARUCO_SQUARES_X, Config.CHARUCO_SQUARES_Y),
            Config.CHARUCO_SQUARE_SIZE,
            Config.CHARUCO_MARKER_SIZE,
            self.aruco_dict
        )
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
    
    def load(self, filepath):
        """Load calibration from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.K1 = np.array(data['K1'])
            self.K2 = np.array(data['K2'])
            self.D1 = np.array(data['D1'])
            self.D2 = np.array(data['D2'])
            self.R1 = np.array(data.get('R1', np.eye(3)))
            self.R2 = np.array(data.get('R2', np.eye(3)))
            self.P1 = np.array(data['P1'])
            self.P2 = np.array(data['P2'])
            
            self.is_calibrated = True
            print(f"[OK] Loaded calibration from {filepath}")
            print(f"     Baseline: {data.get('baseline_meters', 'unknown')}m")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load calibration: {e}")
            return False
    
    def save(self, filepath):
        """Save calibration to JSON file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'image_size': [Config.FRAME_WIDTH, Config.FRAME_HEIGHT],
            'K1': self.K1.tolist(),
            'K2': self.K2.tolist(),
            'D1': self.D1.tolist(),
            'D2': self.D2.tolist(),
            'R': self.R.tolist() if self.R is not None else None,
            'T': self.T.tolist() if self.T is not None else None,
            'R1': self.R1.tolist(),
            'R2': self.R2.tolist(),
            'P1': self.P1.tolist(),
            'P2': self.P2.tolist(),
            'baseline_meters': float(np.linalg.norm(self.T)) if self.T is not None else 0
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[OK] Saved calibration to {filepath}")
    
    def detect_charuco(self, image):
        """Detect ChArUco board in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect ArUco markers
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        
        if ids is None or len(ids) < 4:
            return None, None
        
        # Interpolate ChArUco corners
        num, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, self.charuco_board
        )
        
        if num < 4:
            return None, None
        
        return charuco_corners, charuco_ids
    
    def calibrate_stereo(self, frames_a, frames_b):
        """
        Calibrate stereo cameras from synchronized frame pairs.
        
        Args:
            frames_a: List of frames from camera A
            frames_b: List of frames from camera B
        
        Returns:
            True if successful
        """
        print("\n[CALIBRATING STEREO CAMERAS]")
        
        all_corners_a = []
        all_corners_b = []
        all_ids_a = []
        all_ids_b = []
        
        for i, (fa, fb) in enumerate(zip(frames_a, frames_b)):
            ca, ia = self.detect_charuco(fa)
            cb, ib = self.detect_charuco(fb)
            
            if ca is not None and cb is not None:
                all_corners_a.append(ca)
                all_ids_a.append(ia)
                all_corners_b.append(cb)
                all_ids_b.append(ib)
                print(f"  Frame {i+1}: OK ({len(ca)} corners)")
            else:
                print(f"  Frame {i+1}: Board not detected in both cameras")
        
        if len(all_corners_a) < 10:
            print(f"[ERROR] Not enough valid frames ({len(all_corners_a)}). Need at least 10.")
            return False
        
        img_size = (Config.FRAME_WIDTH, Config.FRAME_HEIGHT)
        
        # Calibrate camera A
        print("\n  Calibrating Camera A...")
        ret_a, K1, D1, _, _ = cv2.aruco.calibrateCameraCharuco(
            all_corners_a, all_ids_a, self.charuco_board, img_size, None, None
        )
        print(f"    RMS Error: {ret_a:.4f}")
        
        # Calibrate camera B
        print("  Calibrating Camera B...")
        ret_b, K2, D2, _, _ = cv2.aruco.calibrateCameraCharuco(
            all_corners_b, all_ids_b, self.charuco_board, img_size, None, None
        )
        print(f"    RMS Error: {ret_b:.4f}")
        
        # Prepare object points for stereo calibration
        obj_points = []
        img_points_a = []
        img_points_b = []
        
        for ca, ia, cb, ib in zip(all_corners_a, all_ids_a, all_corners_b, all_ids_b):
            # Find common corner IDs
            ids_a_set = set(ia.flatten())
            ids_b_set = set(ib.flatten())
            common = ids_a_set & ids_b_set
            
            if len(common) < 4:
                continue
            
            obj_pts = []
            img_a = []
            img_b = []
            
            for cid in sorted(common):
                # Get 3D position of this corner on the board
                obj_pt = self.charuco_board.getChessboardCorners()[cid]
                obj_pts.append(obj_pt)
                
                # Get 2D position in each image
                idx_a = np.where(ia.flatten() == cid)[0][0]
                idx_b = np.where(ib.flatten() == cid)[0][0]
                img_a.append(ca[idx_a].flatten())
                img_b.append(cb[idx_b].flatten())
            
            obj_points.append(np.array(obj_pts, dtype=np.float32))
            img_points_a.append(np.array(img_a, dtype=np.float32))
            img_points_b.append(np.array(img_b, dtype=np.float32))
        
        # Stereo calibration
        print("\n  Running stereo calibration...")
        flags = cv2.CALIB_FIX_INTRINSIC
        
        ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            obj_points, img_points_a, img_points_b,
            K1, D1, K2, D2, img_size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
            flags=flags
        )
        print(f"    Stereo RMS Error: {ret_stereo:.4f}")
        
        # Stereo rectification
        print("  Computing rectification...")
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K1, D1, K2, D2, img_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        
        # Store results
        self.K1, self.D1 = K1, D1
        self.K2, self.D2 = K2, D2
        self.R, self.T = R, T
        self.R1, self.R2 = R1, R2
        self.P1, self.P2 = P1, P2
        self.is_calibrated = True
        
        baseline = np.linalg.norm(T)
        print(f"\n[OK] Stereo calibration complete!")
        print(f"     Baseline: {baseline:.3f}m")
        
        return True
    
    def triangulate(self, pts_a, pts_b, apply_smoothing=True):
        """
        Triangulate 3D points from stereo correspondences.
        
        Args:
            pts_a: Nx2 array of pixel coordinates from camera A
            pts_b: Nx2 array of pixel coordinates from camera B
            apply_smoothing: Whether to apply Kalman filtering
        
        Returns:
            Nx3 array in Blender space (X=right, Y=forward, Z=up)
        """
        if not self.is_calibrated:
            return None
        
        pts_a = np.array(pts_a, dtype=np.float32).reshape(-1, 1, 2)
        pts_b = np.array(pts_b, dtype=np.float32).reshape(-1, 1, 2)
        
        # Undistort and rectify
        pts_a_rect = cv2.undistortPoints(pts_a, self.K1, self.D1, R=self.R1, P=self.P1)
        pts_b_rect = cv2.undistortPoints(pts_b, self.K2, self.D2, R=self.R2, P=self.P2)
        
        # Triangulate
        points_4d = cv2.triangulatePoints(
            self.P1, self.P2,
            pts_a_rect.reshape(-1, 2).T,
            pts_b_rect.reshape(-1, 2).T
        )
        
        # Convert from homogeneous
        cv_points = (points_4d[:3] / points_4d[3]).T
        
        # Convert OpenCV to Blender coordinates
        # OpenCV: X=right, Y=down, Z=forward
        # Blender: X=right, Y=forward, Z=up
        blender_points = np.zeros_like(cv_points)
        blender_points[:, 0] = cv_points[:, 0]      # X stays X
        blender_points[:, 1] = cv_points[:, 2]      # Y = CV's Z (depth/forward)
        blender_points[:, 2] = -cv_points[:, 1]     # Z = -CV's Y (up)
        
        # Apply floor offset
        blender_points[:, 2] += self.floor_z_offset
        
        return blender_points
    
    def triangulate_pose(self, landmarks_a, landmarks_b, smooth=True):
        """
        Triangulate MediaPipe pose landmarks.
        
        Returns:
            Dict of {landmark_idx: [x, y, z]}
        """
        if not self.is_calibrated:
            return None
        
        w, h = Config.FRAME_WIDTH, Config.FRAME_HEIGHT
        points_3d = {}
        
        for i in range(33):
            lm_a = landmarks_a.landmark[i]
            lm_b = landmarks_b.landmark[i]
            
            # Skip low visibility
            if lm_a.visibility < 0.5 or lm_b.visibility < 0.5:
                continue
            
            # Convert normalized to pixels
            pt_a = [lm_a.x * w, lm_a.y * h]
            pt_b = [lm_b.x * w, lm_b.y * h]
            
            # Triangulate single point
            result = self.triangulate([pt_a], [pt_b], apply_smoothing=False)
            
            if result is None:
                continue
            
            pt_3d = result[0].tolist()
            
            # Apply Kalman smoothing
            if smooth:
                if i not in self.filters:
                    self.filters[i] = [
                        SimpleKalman(Config.KALMAN_PROCESS_NOISE, Config.KALMAN_MEASUREMENT_NOISE)
                        for _ in range(3)
                    ]
                
                pt_3d = [
                    self.filters[i][0].update(pt_3d[0]),
                    self.filters[i][1].update(pt_3d[1]),
                    self.filters[i][2].update(pt_3d[2])
                ]
            
            points_3d[i] = pt_3d
        
        return points_3d
    
    def calibrate_floor(self, frame_a, frame_b):
        """
        Calibrate floor plane from ChArUco board on ground.
        
        Returns:
            (success, message)
        """
        corners_a, ids_a = self.detect_charuco(frame_a)
        corners_b, ids_b = self.detect_charuco(frame_b)
        
        if corners_a is None or corners_b is None:
            return False, "Board not detected in both cameras"
        
        # Find common corners
        common_ids = set(ids_a.flatten()) & set(ids_b.flatten())
        if len(common_ids) < 4:
            return False, f"Not enough common corners ({len(common_ids)})"
        
        # Build point arrays
        pts_a = []
        pts_b = []
        
        ids_a_flat = ids_a.flatten()
        ids_b_flat = ids_b.flatten()
        
        for cid in sorted(common_ids):
            idx_a = np.where(ids_a_flat == cid)[0][0]
            idx_b = np.where(ids_b_flat == cid)[0][0]
            pts_a.append(corners_a[idx_a].flatten())
            pts_b.append(corners_b[idx_b].flatten())
        
        # Triangulate floor points
        floor_points = self.triangulate(pts_a, pts_b, apply_smoothing=False)
        
        if floor_points is None:
            return False, "Triangulation failed"
        
        # Fit plane using SVD
        centroid = np.mean(floor_points, axis=0)
        centered = floor_points - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2]
        
        # Make sure normal points up (positive Z)
        if normal[2] < 0:
            normal = -normal
        
        # Calculate Z offset to put floor at Z=0
        self.floor_z_offset = -centroid[2]
        
        return True, f"Floor set (offset: {self.floor_z_offset:.3f}m)"
    
    def reset_filters(self):
        """Reset all Kalman filters"""
        for idx in self.filters:
            for f in self.filters[idx]:
                f.reset()


# =============================================================================
# MOTION CAPTURE RECORDER
# =============================================================================

class MocapRecorder:
    """Records motion capture data to JSON"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.frames = []
        self.is_recording = False
        self.start_time = 0
        self.take_name = ""
    
    def start(self):
        """Start a new recording"""
        self.frames = []
        self.is_recording = True
        self.start_time = time.time()
        self.take_name = f"take_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\n[RECORDING] Started: {self.take_name}")
    
    def add_frame(self, points_3d, landmarks_a=None, landmarks_b=None):
        """Add a frame to the recording"""
        if not self.is_recording:
            return
        
        timestamp = time.time() - self.start_time
        
        frame_data = {
            "timestamp": timestamp,
            "landmarks_3d": {str(k): v for k, v in points_3d.items()}
        }
        
        # Store raw 2D for potential re-processing
        if landmarks_a is not None:
            frame_data["raw_2d_a"] = {
                str(i): [lm.x, lm.y, lm.z, lm.visibility]
                for i, lm in enumerate(landmarks_a.landmark)
            }
        
        if landmarks_b is not None:
            frame_data["raw_2d_b"] = {
                str(i): [lm.x, lm.y, lm.z, lm.visibility]
                for i, lm in enumerate(landmarks_b.landmark)
            }
        
        self.frames.append(frame_data)
    
    def stop(self):
        """Stop recording and save to file"""
        if not self.is_recording:
            return None
        
        self.is_recording = False
        duration = time.time() - self.start_time
        fps = len(self.frames) / duration if duration > 0 else 0
        
        # Build output data
        data = {
            "take_name": self.take_name,
            "duration": duration,
            "fps": fps,
            "frame_count": len(self.frames),
            "created": datetime.now().isoformat(),
            "frames": self.frames
        }
        
        # Save
        filepath = self.output_dir / f"{self.take_name}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n[SAVED] {filepath}")
        print(f"        {len(self.frames)} frames, {fps:.1f} fps, {duration:.1f}s")
        
        return str(filepath)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class MelodicCapApp:
    """Main motion capture application"""
    
    def __init__(self):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=Config.MP_MODEL_COMPLEXITY,
            enable_segmentation=False,
            min_detection_confidence=Config.MP_MIN_DETECTION_CONF,
            min_tracking_confidence=Config.MP_MIN_TRACKING_CONF
        )
        
        # Initialize cameras
        self.cap_a = None
        self.cap_b = None
        
        # Initialize systems
        self.calibration = StereoCalibration()
        self.recorder = MocapRecorder(Config.TAKES_DIR)
        
        # Calibration frame collection
        self.cal_frames_a = []
        self.cal_frames_b = []
        self.collecting_cal_frames = False
        
        # State
        self.running = True
    
    def init_cameras(self):
        """Initialize both cameras"""
        print("\n[INITIALIZING CAMERAS]")
        
        # Camera A
        print(f"  Opening Camera A (index {Config.CAM_A_INDEX})...")
        self.cap_a = cv2.VideoCapture(Config.CAM_A_INDEX, cv2.CAP_DSHOW)
        if not self.cap_a.isOpened():
            self.cap_a = cv2.VideoCapture(Config.CAM_A_INDEX)
        
        if self.cap_a.isOpened():
            self.cap_a.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
            self.cap_a.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
            self.cap_a.set(cv2.CAP_PROP_FPS, Config.FPS)
            print(f"    [OK] Camera A opened")
        else:
            print(f"    [ERROR] Failed to open Camera A")
            return False
        
        # Camera B
        print(f"  Opening Camera B (index {Config.CAM_B_INDEX})...")
        self.cap_b = cv2.VideoCapture(Config.CAM_B_INDEX, cv2.CAP_DSHOW)
        if not self.cap_b.isOpened():
            self.cap_b = cv2.VideoCapture(Config.CAM_B_INDEX)
        
        if self.cap_b.isOpened():
            self.cap_b.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
            self.cap_b.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
            self.cap_b.set(cv2.CAP_PROP_FPS, Config.FPS)
            print(f"    [OK] Camera B opened")
        else:
            print(f"    [ERROR] Failed to open Camera B")
            return False
        
        return True
    
    def load_existing_calibration(self):
        """Try to load existing calibration"""
        if Config.CALIBRATION_FILE.exists():
            return self.calibration.load(Config.CALIBRATION_FILE)
        return False
    
    def process_frame(self, frame):
        """Run MediaPipe pose detection on a frame"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        return results
    
    def draw_pose(self, frame, results):
        """Draw pose landmarks on frame"""
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_draw.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                )
            )
        return frame
    
    def draw_status(self, frame, text, color=(0, 255, 0)):
        """Draw status text on frame"""
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, color, 2)
    
    def run(self):
        """Main loop"""
        print("\n" + "="*60)
        print("MELODICCAP STUDIO v1.0")
        print("="*60)
        
        # Create directories
        Config.BASE_DIR.mkdir(parents=True, exist_ok=True)
        (Config.BASE_DIR / "calibration").mkdir(exist_ok=True)
        Config.TAKES_DIR.mkdir(exist_ok=True)
        
        # Initialize cameras
        if not self.init_cameras():
            print("\n[FATAL] Could not initialize cameras. Exiting.")
            return
        
        # Try to load existing calibration
        self.load_existing_calibration()
        
        print("\n[CONTROLS]")
        print("  C = Collect calibration frames (hold board steady)")
        print("  S = Run stereo calibration (after collecting frames)")
        print("  F = Calibrate floor (place board on ground)")
        print("  R = Start/Stop recording")
        print("  Q = Quit")
        print()
        
        while self.running:
            # Capture frames
            ret_a, frame_a = self.cap_a.read()
            ret_b, frame_b = self.cap_b.read()
            
            if not ret_a or not ret_b:
                print("[WARNING] Frame capture failed")
                continue
            
            # Process with MediaPipe
            results_a = self.process_frame(frame_a)
            results_b = self.process_frame(frame_b)
            
            # Draw poses
            display_a = self.draw_pose(frame_a.copy(), results_a)
            display_b = self.draw_pose(frame_b.copy(), results_b)
            
            # Status text
            if self.collecting_cal_frames:
                self.draw_status(display_a, f"CALIBRATING: {len(self.cal_frames_a)} frames", (0, 255, 255))
                self.draw_status(display_b, f"CALIBRATING: {len(self.cal_frames_b)} frames", (0, 255, 255))
            elif self.recorder.is_recording:
                self.draw_status(display_a, f"RECORDING: {len(self.recorder.frames)} frames", (0, 0, 255))
                self.draw_status(display_b, f"RECORDING: {len(self.recorder.frames)} frames", (0, 0, 255))
            elif self.calibration.is_calibrated:
                self.draw_status(display_a, "READY (Press R to record)")
                self.draw_status(display_b, "READY (Press R to record)")
            else:
                self.draw_status(display_a, "NOT CALIBRATED (Press C)", (0, 165, 255))
                self.draw_status(display_b, "NOT CALIBRATED (Press C)", (0, 165, 255))
            
            # If calibrated and both poses detected, do triangulation
            if (self.calibration.is_calibrated and 
                results_a.pose_landmarks and 
                results_b.pose_landmarks):
                
                points_3d = self.calibration.triangulate_pose(
                    results_a.pose_landmarks,
                    results_b.pose_landmarks,
                    smooth=True
                )
                
                if points_3d and self.recorder.is_recording:
                    self.recorder.add_frame(
                        points_3d,
                        results_a.pose_landmarks,
                        results_b.pose_landmarks
                    )
            
            # Collecting calibration frames
            if self.collecting_cal_frames:
                # Check if board visible in both
                ca, _ = self.calibration.detect_charuco(frame_a)
                cb, _ = self.calibration.detect_charuco(frame_b)
                
                if ca is not None and cb is not None:
                    # Visual feedback
                    cv2.aruco.drawDetectedCornersCharuco(display_a, ca)
                    cv2.aruco.drawDetectedCornersCharuco(display_b, cb)
                    
                    # Collect every 10th frame when board is visible
                    if len(self.cal_frames_a) == 0 or time.time() - getattr(self, '_last_cal_time', 0) > 0.5:
                        self.cal_frames_a.append(frame_a.copy())
                        self.cal_frames_b.append(frame_b.copy())
                        self._last_cal_time = time.time()
                        print(f"  Captured frame {len(self.cal_frames_a)}")
            
            # Display
            combined = np.hstack([
                cv2.resize(display_a, (640, 360)),
                cv2.resize(display_b, (640, 360))
            ])
            cv2.imshow("MelodicCap Studio", combined)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n[QUIT]")
                self.running = False
            
            elif key == ord('c'):
                if not self.collecting_cal_frames:
                    print("\n[CALIBRATION] Starting frame collection...")
                    print("  Move the ChArUco board around. Collecting every 0.5s when visible.")
                    print("  Press 'S' when you have 20+ frames to run calibration.")
                    self.cal_frames_a = []
                    self.cal_frames_b = []
                    self.collecting_cal_frames = True
                else:
                    print("\n[CALIBRATION] Stopped collecting.")
                    self.collecting_cal_frames = False
            
            elif key == ord('s'):
                if len(self.cal_frames_a) >= 10:
                    print(f"\n[CALIBRATION] Running with {len(self.cal_frames_a)} frames...")
                    self.collecting_cal_frames = False
                    
                    if self.calibration.calibrate_stereo(self.cal_frames_a, self.cal_frames_b):
                        self.calibration.save(Config.CALIBRATION_FILE)
                    
                    self.cal_frames_a = []
                    self.cal_frames_b = []
                else:
                    print(f"\n[ERROR] Need at least 10 frames (have {len(self.cal_frames_a)})")
            
            elif key == ord('f'):
                if not self.calibration.is_calibrated:
                    print("\n[ERROR] Calibrate cameras first (press C)")
                else:
                    print("\n[FLOOR] Calibrating...")
                    success, msg = self.calibration.calibrate_floor(frame_a, frame_b)
                    if success:
                        print(f"  [OK] {msg}")
                    else:
                        print(f"  [ERROR] {msg}")
            
            elif key == ord('r'):
                if not self.calibration.is_calibrated:
                    print("\n[ERROR] Calibrate cameras first (press C)")
                elif self.recorder.is_recording:
                    self.recorder.stop()
                else:
                    self.calibration.reset_filters()
                    self.recorder.start()
        
        # Cleanup
        if self.recorder.is_recording:
            self.recorder.stop()
        
        self.cap_a.release()
        self.cap_b.release()
        cv2.destroyAllWindows()
        self.pose.close()
        
        print("\n[DONE]")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    app = MelodicCapApp()
    app.run()