"""
MelodicCap Studio v2.0 - With Visual Calibration Feedback
==========================================================
Improvements over v1:
- Visual feedback showing when BOTH cameras detect the board
- Only captures calibration frames when both cameras see board
- Calibration quality warnings
- 3-2-1 countdown before recording
- Live skeleton preview during recording

Controls:
  C = Start/stop calibration frame collection
  S = Run stereo calibration
  F = Calibrate floor
  R = Start recording (with countdown)
  Q = Quit
"""

import cv2
import numpy as np
import mediapipe as mp
import json
import time
import os
from datetime import datetime
from pathlib import Path
import threading


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """All settings in one place - EDIT THESE FOR YOUR SETUP"""
    
    # Camera indices (from detect_cameras.py)
    CAM_A_INDEX = 2  # Sony ZV-1F
    CAM_B_INDEX = 4  # Samsung S25 DroidCam
    
    # Resolution
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    TARGET_FPS = 30
    
    # Paths
    BASE_DIR = Path(r"C:\Users\ninja\Documents\MelodicCapStudio\MelodicCapFresh")
    CALIBRATION_FILE = BASE_DIR / "calibration" / "stereo_calibration.json"
    TAKES_DIR = BASE_DIR / "takes"
    
    # ChArUco board - YOUR CALIBRATION BOARD (63.5mm squares)
    CHARUCO_SQUARES_X = 4
    CHARUCO_SQUARES_Y = 3
    CHARUCO_SQUARE_SIZE = 0.0635  # 63.5mm in meters
    CHARUCO_MARKER_SIZE = 0.0476  # 47.6mm in meters (75% of square)
    ARUCO_DICT = cv2.aruco.DICT_4X4_50
    
    # Calibration quality thresholds
    MAX_INTRINSIC_RMS = 1.0    # Per-camera RMS should be below this
    MAX_STEREO_RMS = 2.0       # Stereo RMS should be below this
    MIN_CALIBRATION_FRAMES = 15
    MAX_CALIBRATION_FRAMES = 50
    
    # Recording
    COUNTDOWN_SECONDS = 3
    
    # Smoothing
    KALMAN_PROCESS_NOISE = 1e-4
    KALMAN_MEASUREMENT_NOISE = 1e-2


# =============================================================================
# SIMPLE KALMAN FILTER
# =============================================================================

class SimpleKalman:
    def __init__(self, q=1e-4, r=1e-2):
        self.q = q
        self.r = r
        self.x = 0.0
        self.p = 1.0
        self.initialized = False
    
    def update(self, measurement):
        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return self.x
        self.p += self.q
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
    def __init__(self):
        self.is_calibrated = False
        self.K1 = self.K2 = None
        self.D1 = self.D2 = None
        self.R1 = self.R2 = None
        self.P1 = self.P2 = None
        self.floor_z_offset = 0.0
        self.filters = {}
        
        # Calibration quality metrics
        self.rms_camera_a = 0
        self.rms_camera_b = 0
        self.rms_stereo = 0
        self.baseline = 0
        
        # ChArUco setup with LENIENT detection parameters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(Config.ARUCO_DICT)
        self.charuco_board = cv2.aruco.CharucoBoard(
            (Config.CHARUCO_SQUARES_X, Config.CHARUCO_SQUARES_Y),
            Config.CHARUCO_SQUARE_SIZE,
            Config.CHARUCO_MARKER_SIZE,
            self.aruco_dict
        )
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        # More lenient detection for lower quality cameras (DroidCam)
        self.detector_params.adaptiveThreshWinSizeMin = 3
        self.detector_params.adaptiveThreshWinSizeMax = 30
        self.detector_params.adaptiveThreshWinSizeStep = 5
        self.detector_params.minMarkerPerimeterRate = 0.02  # Smaller markers OK
        self.detector_params.maxMarkerPerimeterRate = 4.0
        self.detector_params.polygonalApproxAccuracyRate = 0.05  # More lenient
        self.detector_params.minCornerDistanceRate = 0.05
        self.detector_params.minMarkerDistanceRate = 0.05
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
    
    def detect_charuco(self, image):
        """Detect ChArUco board, returns (corners, ids, annotated_image)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        display = image.copy()
        
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        if ids is None or len(ids) < 4:
            return None, None, display
        
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(display, corners, ids)
        
        # Interpolate ChArUco corners
        num, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, self.charuco_board
        )
        
        if num < 4:
            return None, None, display
        
        # Draw ChArUco corners
        cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids, 
                                              cornerColor=(0, 255, 0))
        
        return charuco_corners, charuco_ids, display
    
    def load(self, filepath):
        """Load calibration from JSON"""
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
            
            self.rms_camera_a = data.get('rms_camera_a', 0)
            self.rms_camera_b = data.get('rms_camera_b', 0)
            self.rms_stereo = data.get('rms_stereo', 0)
            self.baseline = data.get('baseline_meters', 0)
            
            self.is_calibrated = True
            return True, f"Loaded (Stereo RMS: {self.rms_stereo:.2f}, Baseline: {self.baseline:.2f}m)"
        except Exception as e:
            return False, str(e)
    
    def save(self, filepath):
        """Save calibration to JSON"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'image_size': [Config.FRAME_WIDTH, Config.FRAME_HEIGHT],
            'K1': self.K1.tolist(), 'K2': self.K2.tolist(),
            'D1': self.D1.tolist(), 'D2': self.D2.tolist(),
            'R1': self.R1.tolist(), 'R2': self.R2.tolist(),
            'P1': self.P1.tolist(), 'P2': self.P2.tolist(),
            'rms_camera_a': self.rms_camera_a,
            'rms_camera_b': self.rms_camera_b,
            'rms_stereo': self.rms_stereo,
            'baseline_meters': self.baseline
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def calibrate_stereo(self, frames_a, frames_b):
        """
        Calibrate stereo cameras.
        Returns (success, message)
        """
        print("\n" + "="*50)
        print("STEREO CALIBRATION")
        print("="*50)
        
        all_corners_a, all_corners_b = [], []
        all_ids_a, all_ids_b = [], []
        valid_count = 0
        
        for i, (fa, fb) in enumerate(zip(frames_a, frames_b)):
            ca, ia, _ = self.detect_charuco(fa)
            cb, ib, _ = self.detect_charuco(fb)
            
            if ca is not None and cb is not None:
                # Find common corners
                common = set(ia.flatten()) & set(ib.flatten())
                if len(common) >= 4:
                    all_corners_a.append(ca)
                    all_ids_a.append(ia)
                    all_corners_b.append(cb)
                    all_ids_b.append(ib)
                    valid_count += 1
        
        print(f"Valid frame pairs: {valid_count}/{len(frames_a)}")
        
        if valid_count < Config.MIN_CALIBRATION_FRAMES:
            return False, f"Need {Config.MIN_CALIBRATION_FRAMES}+ valid frames (got {valid_count})"
        
        img_size = (Config.FRAME_WIDTH, Config.FRAME_HEIGHT)
        
        # Calibrate Camera A
        print("\nCalibrating Camera A...")
        ret_a, K1, D1, _, _ = cv2.aruco.calibrateCameraCharuco(
            all_corners_a, all_ids_a, self.charuco_board, img_size, None, None
        )
        self.rms_camera_a = ret_a
        print(f"  RMS: {ret_a:.4f} {'✓' if ret_a < Config.MAX_INTRINSIC_RMS else '✗ HIGH'}")
        
        # Calibrate Camera B
        print("Calibrating Camera B...")
        ret_b, K2, D2, _, _ = cv2.aruco.calibrateCameraCharuco(
            all_corners_b, all_ids_b, self.charuco_board, img_size, None, None
        )
        self.rms_camera_b = ret_b
        print(f"  RMS: {ret_b:.4f} {'✓' if ret_b < Config.MAX_INTRINSIC_RMS else '✗ HIGH'}")
        
        # Prepare for stereo calibration
        obj_points, img_points_a, img_points_b = [], [], []
        
        for ca, ia, cb, ib in zip(all_corners_a, all_ids_a, all_corners_b, all_ids_b):
            common = set(ia.flatten()) & set(ib.flatten())
            if len(common) < 4:
                continue
            
            obj_pts, img_a, img_b = [], [], []
            for cid in sorted(common):
                obj_pts.append(self.charuco_board.getChessboardCorners()[cid])
                idx_a = np.where(ia.flatten() == cid)[0][0]
                idx_b = np.where(ib.flatten() == cid)[0][0]
                img_a.append(ca[idx_a].flatten())
                img_b.append(cb[idx_b].flatten())
            
            obj_points.append(np.array(obj_pts, dtype=np.float32))
            img_points_a.append(np.array(img_a, dtype=np.float32))
            img_points_b.append(np.array(img_b, dtype=np.float32))
        
        # Stereo calibration
        print("\nRunning stereo calibration...")
        ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            obj_points, img_points_a, img_points_b,
            K1, D1, K2, D2, img_size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
            flags=cv2.CALIB_FIX_INTRINSIC
        )
        self.rms_stereo = ret_stereo
        self.baseline = float(np.linalg.norm(T))
        print(f"  Stereo RMS: {ret_stereo:.4f} {'✓' if ret_stereo < Config.MAX_STEREO_RMS else '✗ HIGH'}")
        print(f"  Baseline: {self.baseline:.3f}m")
        
        # Stereo rectification
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K1, D1, K2, D2, img_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        
        # Store results
        self.K1, self.D1, self.K2, self.D2 = K1, D1, K2, D2
        self.R1, self.R2, self.P1, self.P2 = R1, R2, P1, P2
        self.is_calibrated = True
        
        # Quality assessment
        if ret_stereo > Config.MAX_STEREO_RMS:
            return True, f"WARNING: Stereo RMS {ret_stereo:.2f} is high (should be <{Config.MAX_STEREO_RMS})"
        
        return True, f"OK (Stereo RMS: {ret_stereo:.2f}, Baseline: {self.baseline:.2f}m)"
    
    def calibrate_floor(self, frame_a, frame_b):
        """Calibrate floor from board on ground"""
        ca, ia, _ = self.detect_charuco(frame_a)
        cb, ib, _ = self.detect_charuco(frame_b)
        
        # Detailed feedback
        cam_a_ok = ca is not None and len(ca) >= 4
        cam_b_ok = cb is not None and len(cb) >= 4
        
        if not cam_a_ok and not cam_b_ok:
            return False, "Board not detected in either camera"
        elif not cam_a_ok:
            return False, "Board not detected in Camera A (left)"
        elif not cam_b_ok:
            return False, "Board not detected in Camera B (right/DroidCam)"
        
        common = set(ia.flatten()) & set(ib.flatten())
        if len(common) < 4:
            return False, f"Only {len(common)} common corners (need 4+). Move board to better position."
        
        pts_a, pts_b = [], []
        for cid in sorted(common):
            idx_a = np.where(ia.flatten() == cid)[0][0]
            idx_b = np.where(ib.flatten() == cid)[0][0]
            pts_a.append(ca[idx_a].flatten())
            pts_b.append(cb[idx_b].flatten())
        
        floor_points = self.triangulate(pts_a, pts_b)
        if floor_points is None:
            return False, "Triangulation failed - recalibrate cameras first"
        
        centroid = np.mean(floor_points, axis=0)
        self.floor_z_offset = -centroid[2]
        
        return True, f"Floor set! Z-offset: {self.floor_z_offset:.3f}m"
    
    def triangulate(self, pts_a, pts_b):
        """Triangulate 3D points from stereo correspondences"""
        if not self.is_calibrated:
            return None
        
        pts_a = np.array(pts_a, dtype=np.float32).reshape(-1, 1, 2)
        pts_b = np.array(pts_b, dtype=np.float32).reshape(-1, 1, 2)
        
        pts_a_rect = cv2.undistortPoints(pts_a, self.K1, self.D1, R=self.R1, P=self.P1)
        pts_b_rect = cv2.undistortPoints(pts_b, self.K2, self.D2, R=self.R2, P=self.P2)
        
        points_4d = cv2.triangulatePoints(self.P1, self.P2, 
                                          pts_a_rect.reshape(-1, 2).T, 
                                          pts_b_rect.reshape(-1, 2).T)
        cv_points = (points_4d[:3] / points_4d[3]).T
        
        # OpenCV to Blender: (x, y, z) -> (x, z, -y)
        blender_points = np.zeros_like(cv_points)
        blender_points[:, 0] = cv_points[:, 0]
        blender_points[:, 1] = cv_points[:, 2]
        blender_points[:, 2] = -cv_points[:, 1] + self.floor_z_offset
        
        return blender_points
    
    def triangulate_pose(self, landmarks_a, landmarks_b, smooth=True):
        """Triangulate MediaPipe pose landmarks"""
        if not self.is_calibrated:
            return None
        
        w, h = Config.FRAME_WIDTH, Config.FRAME_HEIGHT
        points_3d = {}
        
        for i in range(33):
            lm_a = landmarks_a.landmark[i]
            lm_b = landmarks_b.landmark[i]
            
            if lm_a.visibility < 0.5 or lm_b.visibility < 0.5:
                continue
            
            pt_a = [lm_a.x * w, lm_a.y * h]
            pt_b = [lm_b.x * w, lm_b.y * h]
            
            result = self.triangulate([pt_a], [pt_b])
            if result is None:
                continue
            
            pt_3d = result[0].tolist()
            
            if smooth:
                if i not in self.filters:
                    self.filters[i] = [SimpleKalman() for _ in range(3)]
                pt_3d = [self.filters[i][j].update(pt_3d[j]) for j in range(3)]
            
            points_3d[i] = pt_3d
        
        return points_3d
    
    def reset_filters(self):
        for idx in self.filters:
            for f in self.filters[idx]:
                f.reset()


# =============================================================================
# RECORDER
# =============================================================================

class MocapRecorder:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames = []
        self.is_recording = False
        self.start_time = 0
        self.take_name = ""
    
    def start(self):
        self.frames = []
        self.is_recording = True
        self.start_time = time.time()
        self.take_name = f"take_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def add_frame(self, points_3d, landmarks_a=None, landmarks_b=None):
        if not self.is_recording or not points_3d:
            return
        
        frame_data = {
            "timestamp": time.time() - self.start_time,
            "landmarks_3d": {str(k): v for k, v in points_3d.items()}
        }
        
        if landmarks_a:
            frame_data["raw_2d_a"] = {
                str(i): [lm.x, lm.y, lm.z, lm.visibility]
                for i, lm in enumerate(landmarks_a.landmark)
            }
        if landmarks_b:
            frame_data["raw_2d_b"] = {
                str(i): [lm.x, lm.y, lm.z, lm.visibility]
                for i, lm in enumerate(landmarks_b.landmark)
            }
        
        self.frames.append(frame_data)
    
    def stop(self):
        if not self.is_recording:
            return None
        
        self.is_recording = False
        duration = time.time() - self.start_time
        fps = len(self.frames) / duration if duration > 0 else 0
        
        # Quality check
        quality_ok, quality_msg = self._check_quality()
        
        data = {
            "take_name": self.take_name,
            "duration": duration,
            "fps": fps,
            "frame_count": len(self.frames),
            "quality_check": quality_msg,
            "created": datetime.now().isoformat(),
            "frames": self.frames
        }
        
        filepath = self.output_dir / f"{self.take_name}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(filepath), fps, quality_msg
    
    def _check_quality(self):
        """Check if recorded data looks reasonable"""
        if len(self.frames) < 10:
            return False, "Too few frames"
        
        # Check shoulder width across frames
        shoulder_widths = []
        for frame in self.frames[:50]:  # Sample first 50 frames
            lms = frame.get('landmarks_3d', {})
            if '11' in lms and '12' in lms:
                l = np.array(lms['11'])
                r = np.array(lms['12'])
                width = np.linalg.norm(l - r)
                shoulder_widths.append(width)
        
        if not shoulder_widths:
            return False, "No shoulder data"
        
        avg_width = np.mean(shoulder_widths)
        
        # Shoulder width should be 0.3-0.5m for adults
        if avg_width < 0.2:
            return False, f"Shoulders too narrow ({avg_width:.2f}m) - calibration may be wrong"
        elif avg_width > 0.7:
            return False, f"Shoulders too wide ({avg_width:.2f}m) - calibration may be wrong"
        
        return True, f"OK (shoulder width: {avg_width:.2f}m)"


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class MelodicCapApp:
    def __init__(self):
        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Cameras
        self.cap_a = None
        self.cap_b = None
        
        # Systems
        self.calibration = StereoCalibration()
        self.recorder = MocapRecorder(Config.TAKES_DIR)
        
        # Calibration collection
        self.cal_frames_a = []
        self.cal_frames_b = []
        self.collecting_cal = False
        self.last_cal_capture = 0
        
        # Floor calibration state
        self.waiting_for_floor = False
        
        # Recording state
        self.countdown_active = False
        self.countdown_start = 0
        
        # State
        self.running = True
        self.status_msg = ""
        self.status_color = (255, 255, 255)
    
    def init_cameras(self):
        print("\n[CAMERAS]")
        
        self.cap_a = cv2.VideoCapture(Config.CAM_A_INDEX, cv2.CAP_DSHOW)
        if not self.cap_a.isOpened():
            self.cap_a = cv2.VideoCapture(Config.CAM_A_INDEX)
        
        if self.cap_a.isOpened():
            self.cap_a.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
            self.cap_a.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
            self.cap_a.set(cv2.CAP_PROP_FPS, Config.TARGET_FPS)
            print(f"  Camera A (index {Config.CAM_A_INDEX}): OK")
        else:
            print(f"  Camera A (index {Config.CAM_A_INDEX}): FAILED")
            return False
        
        self.cap_b = cv2.VideoCapture(Config.CAM_B_INDEX, cv2.CAP_DSHOW)
        if not self.cap_b.isOpened():
            self.cap_b = cv2.VideoCapture(Config.CAM_B_INDEX)
        
        if self.cap_b.isOpened():
            self.cap_b.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
            self.cap_b.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
            self.cap_b.set(cv2.CAP_PROP_FPS, Config.TARGET_FPS)
            print(f"  Camera B (index {Config.CAM_B_INDEX}): OK")
        else:
            print(f"  Camera B (index {Config.CAM_B_INDEX}): FAILED")
            return False
        
        return True
    
    def set_status(self, msg, color=(255, 255, 255)):
        self.status_msg = msg
        self.status_color = color
        print(f"[STATUS] {msg}")
    
    def draw_ui(self, frame, camera_name, board_detected, pose_results=None):
        """Draw UI overlay on frame"""
        h, w = frame.shape[:2]
        
        # Camera label with background
        cv2.rectangle(frame, (0, 0), (120, 40), (0, 0, 0), -1)
        cv2.putText(frame, camera_name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (255, 255, 255), 2)
        
        # VERY VISIBLE board detection indicator during calibration
        if self.collecting_cal or self.waiting_for_floor:
            border_thickness = 15
            if board_detected:
                # BRIGHT GREEN border + text
                cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), border_thickness)
                cv2.rectangle(frame, (w//2-100, 50), (w//2+100, 90), (0, 100, 0), -1)
                cv2.putText(frame, "BOARD OK", (w//2-70, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)
            else:
                # BRIGHT RED border + text
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), border_thickness)
                cv2.rectangle(frame, (w//2-100, 50), (w//2+100, 90), (100, 0, 0), -1)
                cv2.putText(frame, "NO BOARD", (w//2-70, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)
        
        # Draw pose if detected
        if pose_results and pose_results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_draw.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                )
            )
        
        return frame
    
    def draw_main_status(self, combined):
        """Draw status bar at bottom of combined view"""
        h, w = combined.shape[:2]
        
        # Status bar background
        cv2.rectangle(combined, (0, h-40), (w, h), (40, 40, 40), -1)
        
        # Status message
        cv2.putText(combined, self.status_msg, (10, h-12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, self.status_color, 2)
        
        # Frame count during calibration
        if self.collecting_cal:
            count_text = f"Frames: {len(self.cal_frames_a)}/{Config.MAX_CALIBRATION_FRAMES}"
            cv2.putText(combined, count_text, (w-200, h-12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)
        
        # Frame count during recording
        if self.recorder.is_recording:
            elapsed = time.time() - self.recorder.start_time
            rec_text = f"REC: {len(self.recorder.frames)} frames ({elapsed:.1f}s)"
            cv2.putText(combined, rec_text, (w-300, h-12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
        
        # Countdown
        if self.countdown_active:
            remaining = Config.COUNTDOWN_SECONDS - (time.time() - self.countdown_start)
            if remaining > 0:
                # Big countdown number in center
                count = int(remaining) + 1
                text = str(count)
                font_scale = 5
                thickness = 10
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                       font_scale, thickness)
                x = (w - text_w) // 2
                y = (h + text_h) // 2
                cv2.putText(combined, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (0, 255, 255), thickness)
        
        return combined
    
    def run(self):
        print("\n" + "="*60)
        print("MELODICCAP STUDIO v2.0")
        print("="*60)
        
        # Setup
        Config.BASE_DIR.mkdir(parents=True, exist_ok=True)
        (Config.BASE_DIR / "calibration").mkdir(exist_ok=True)
        Config.TAKES_DIR.mkdir(exist_ok=True)
        
        if not self.init_cameras():
            print("\n[FATAL] Camera initialization failed")
            return
        
        # Load existing calibration
        if Config.CALIBRATION_FILE.exists():
            success, msg = self.calibration.load(Config.CALIBRATION_FILE)
            if success:
                self.set_status(f"Calibration loaded: {msg}", (0, 255, 0))
            else:
                self.set_status("No calibration - press C to calibrate", (0, 165, 255))
        else:
            self.set_status("No calibration - press C to calibrate", (0, 165, 255))
        
        print("\n[CONTROLS]")
        print("  C = Start/stop calibration")
        print("  S = Run stereo calibration")
        print("  F = Calibrate floor")
        print("  R = Record (with countdown)")
        print("  Q = Quit")
        
        while self.running:
            ret_a, frame_a = self.cap_a.read()
            ret_b, frame_b = self.cap_b.read()
            
            if not ret_a or not ret_b:
                continue
            
            # Detect board in both frames
            corners_a, ids_a, display_a = self.calibration.detect_charuco(frame_a)
            corners_b, ids_b, display_b = self.calibration.detect_charuco(frame_b)
            
            board_a = corners_a is not None
            board_b = corners_b is not None
            both_boards = board_a and board_b
            
            # Process poses
            rgb_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB)
            rgb_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2RGB)
            results_a = self.pose.process(rgb_a)
            results_b = self.pose.process(rgb_b)
            
            # Draw UI on frames
            display_a = self.draw_ui(display_a, "CAM A", board_a, results_a)
            display_b = self.draw_ui(display_b, "CAM B", board_b, results_b)
            
            # Calibration frame collection
            if self.collecting_cal and both_boards:
                now = time.time()
                if now - self.last_cal_capture > 0.5:  # Every 0.5 seconds
                    self.cal_frames_a.append(frame_a.copy())
                    self.cal_frames_b.append(frame_b.copy())
                    self.last_cal_capture = now
                    self.set_status(f"Captured frame {len(self.cal_frames_a)} (both cameras)", (0, 255, 0))
                    
                    if len(self.cal_frames_a) >= Config.MAX_CALIBRATION_FRAMES:
                        self.collecting_cal = False
                        self.set_status(f"Max frames reached. Press S to calibrate.", (0, 255, 255))
            
            # Countdown handling
            if self.countdown_active:
                elapsed = time.time() - self.countdown_start
                if elapsed >= Config.COUNTDOWN_SECONDS:
                    self.countdown_active = False
                    self.calibration.reset_filters()
                    self.recorder.start()
                    self.set_status("RECORDING - Press R to stop", (0, 0, 255))
            
            # Recording
            if self.recorder.is_recording:
                if self.calibration.is_calibrated and results_a.pose_landmarks and results_b.pose_landmarks:
                    points_3d = self.calibration.triangulate_pose(
                        results_a.pose_landmarks, results_b.pose_landmarks
                    )
                    self.recorder.add_frame(points_3d, results_a.pose_landmarks, results_b.pose_landmarks)
            
            # Auto floor calibration when both cameras see board
            if self.waiting_for_floor and both_boards:
                success, msg = self.calibration.calibrate_floor(frame_a, frame_b)
                if success:
                    self.waiting_for_floor = False
                    self.set_status(f"Floor: {msg}", (0, 255, 0))
                else:
                    # Show what's wrong
                    self.set_status(f"Floor: {msg}", (0, 165, 255))
            
            # Combine frames for display
            h = 400  # Display height
            w_a = int(display_a.shape[1] * h / display_a.shape[0])
            w_b = int(display_b.shape[1] * h / display_b.shape[0])
            
            resized_a = cv2.resize(display_a, (w_a, h))
            resized_b = cv2.resize(display_b, (w_b, h))
            
            combined = np.hstack([resized_a, resized_b])
            combined = self.draw_main_status(combined)
            
            cv2.imshow("MelodicCap Studio v2", combined)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                self.running = False
            
            elif key == ord('c'):
                if not self.collecting_cal:
                    self.cal_frames_a = []
                    self.cal_frames_b = []
                    self.collecting_cal = True
                    self.set_status("CALIBRATING - Show board to BOTH cameras", (0, 255, 255))
                else:
                    self.collecting_cal = False
                    self.set_status(f"Stopped. {len(self.cal_frames_a)} frames. Press S to calibrate.", (255, 255, 255))
            
            elif key == ord('s'):
                if len(self.cal_frames_a) >= Config.MIN_CALIBRATION_FRAMES:
                    self.collecting_cal = False
                    self.set_status("Running calibration...", (0, 255, 255))
                    cv2.waitKey(100)
                    
                    success, msg = self.calibration.calibrate_stereo(
                        self.cal_frames_a, self.cal_frames_b
                    )
                    
                    if success:
                        self.calibration.save(Config.CALIBRATION_FILE)
                        color = (0, 255, 0) if self.calibration.rms_stereo < Config.MAX_STEREO_RMS else (0, 165, 255)
                        self.set_status(f"Calibration: {msg}", color)
                    else:
                        self.set_status(f"Calibration FAILED: {msg}", (0, 0, 255))
                    
                    self.cal_frames_a = []
                    self.cal_frames_b = []
                else:
                    self.set_status(f"Need {Config.MIN_CALIBRATION_FRAMES}+ frames (have {len(self.cal_frames_a)})", (0, 0, 255))
            
            elif key == ord('f'):
                if not self.calibration.is_calibrated:
                    self.set_status("Calibrate cameras first (C then S)", (0, 0, 255))
                else:
                    # Toggle floor calibration mode
                    self.waiting_for_floor = not self.waiting_for_floor
                    if self.waiting_for_floor:
                        self.set_status("FLOOR MODE: Show board to BOTH cameras (press F to cancel)", (0, 255, 255))
                    else:
                        self.set_status("Floor calibration cancelled", (255, 255, 255))
            
            elif key == ord('r'):
                if self.recorder.is_recording:
                    filepath, fps, quality = self.recorder.stop()
                    self.set_status(f"Saved: {fps:.1f}fps, {quality}", (0, 255, 0))
                elif not self.calibration.is_calibrated:
                    self.set_status("Calibrate cameras first (C then S)", (0, 0, 255))
                elif self.calibration.rms_stereo > Config.MAX_STEREO_RMS:
                    self.set_status(f"WARNING: Poor calibration (RMS {self.calibration.rms_stereo:.1f}). Recording anyway...", (0, 165, 255))
                    self.countdown_active = True
                    self.countdown_start = time.time()
                else:
                    self.countdown_active = True
                    self.countdown_start = time.time()
                    self.set_status("GET READY!", (0, 255, 255))
        
        # Cleanup
        if self.recorder.is_recording:
            self.recorder.stop()
        
        self.cap_a.release()
        self.cap_b.release()
        cv2.destroyAllWindows()
        self.pose.close()
        
        print("\n[DONE]")


if __name__ == "__main__":
    app = MelodicCapApp()
    app.run()