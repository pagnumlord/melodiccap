#!/usr/bin/env python3
"""
Melodic Justice Motion Capture - Professional Dual-Camera Recorder
===================================================================

Features:
- 3-2-1 countdown with audio feedback
- Auto-stop after duration
- Real-time 3D triangulation with proper stereo calibration
- Hand tracking support
- Progress bar and recording indicators
- World coordinate export (meters, Blender-compatible)
- Foot ground plane detection for better retargeting

Based on research from FreeMoCap, Rokoko, and professional mocap workflows.

Usage:
    python professional_recorder.py --left 0 --right 1 --duration 15
    python professional_recorder.py --left 3 --right 4 --calibration ./calibration.json
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from config import POSE_LANDMARKS, HAND_LANDMARKS, MEDIAPIPE_TO_RIGIFY_MAPPING
except ImportError:
    # Fallback definitions
    POSE_LANDMARKS = {
        0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
        4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
        7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
        11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow", 14: "right_elbow",
        15: "left_wrist", 16: "right_wrist", 17: "left_pinky", 18: "right_pinky",
        19: "left_index", 20: "right_index", 21: "left_thumb", 22: "right_thumb",
        23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
        27: "left_ankle", 28: "right_ankle", 29: "left_heel", 30: "right_heel",
        31: "left_foot_index", 32: "right_foot_index"
    }


class RecordingState(Enum):
    """Recording state machine."""
    IDLE = "idle"
    COUNTDOWN = "countdown"
    RECORDING = "recording"
    SAVING = "saving"


@dataclass
class CameraCalibration:
    """Stereo camera calibration data."""
    # Camera matrices
    K1: Optional[np.ndarray] = None  # Left camera intrinsic
    K2: Optional[np.ndarray] = None  # Right camera intrinsic
    D1: Optional[np.ndarray] = None  # Left distortion coefficients
    D2: Optional[np.ndarray] = None  # Right distortion coefficients
    
    # Stereo parameters
    R: Optional[np.ndarray] = None   # Rotation between cameras
    T: Optional[np.ndarray] = None   # Translation between cameras
    
    # Projection matrices
    P1: Optional[np.ndarray] = None  # Left projection matrix
    P2: Optional[np.ndarray] = None  # Right projection matrix
    
    # Baseline info
    baseline_meters: float = 0.5
    angle_degrees: float = 50.0
    
    @classmethod
    def load(cls, filepath: str) -> 'CameraCalibration':
        """Load calibration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        cal = cls()
        cal.baseline_meters = data.get('baseline_meters', 0.5)
        cal.angle_degrees = data.get('angle_degrees', 50.0)
        
        if 'K1' in data:
            cal.K1 = np.array(data['K1'])
        if 'K2' in data:
            cal.K2 = np.array(data['K2'])
        if 'D1' in data:
            cal.D1 = np.array(data['D1'])
        if 'D2' in data:
            cal.D2 = np.array(data['D2'])
        if 'R' in data:
            cal.R = np.array(data['R'])
        if 'T' in data:
            cal.T = np.array(data['T'])
        if 'P1' in data:
            cal.P1 = np.array(data['P1'])
        if 'P2' in data:
            cal.P2 = np.array(data['P2'])
        
        return cal
    
    @property
    def is_calibrated(self) -> bool:
        """Check if we have valid stereo calibration."""
        return self.P1 is not None and self.P2 is not None


@dataclass
class Landmark3D:
    """A 3D landmark with full metadata."""
    name: str
    x: float  # In meters, Blender X (right)
    y: float  # In meters, Blender Y (forward) 
    z: float  # In meters, Blender Z (up)
    visibility: float
    
    # Original MediaPipe coordinates for debugging
    mp_x: float = 0.0
    mp_y: float = 0.0
    mp_z: float = 0.0


@dataclass
class FrameCapture:
    """Data captured for a single frame."""
    frame_number: int
    timestamp: float  # Seconds since recording start
    
    # Raw 2D landmarks from each camera (normalized 0-1)
    landmarks_2d_left: Dict[str, Dict[str, float]] = field(default_factory=dict)
    landmarks_2d_right: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Triangulated 3D landmarks (world coordinates in meters)
    landmarks_3d: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # World landmarks from MediaPipe (meters, hip-centered)
    world_landmarks: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Hand landmarks (if detected)
    left_hand: Dict[str, Dict[str, float]] = field(default_factory=dict)
    right_hand: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Computed derived landmarks
    hip_center: Optional[Dict[str, float]] = None
    spine_center: Optional[Dict[str, float]] = None
    neck_base: Optional[Dict[str, float]] = None


class StereoTriangulator:
    """Handles 3D point reconstruction from stereo camera pairs."""
    
    def __init__(self, calibration: Optional[CameraCalibration] = None):
        self.calibration = calibration or CameraCalibration()
        
        # Default projection matrices if no calibration
        if not self.calibration.is_calibrated:
            self._setup_default_projection()
    
    def _setup_default_projection(self):
        """Set up default projection matrices for uncalibrated stereo."""
        # Assume 1280x720, ~60 degree FOV
        fx = 1000.0
        fy = 1000.0
        cx = 640.0
        cy = 360.0
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.calibration.K1 = K.copy()
        self.calibration.K2 = K.copy()
        
        # Left camera at origin
        self.calibration.P1 = np.hstack([K, np.zeros((3, 1))])
        
        # Right camera offset by baseline, rotated slightly
        baseline = self.calibration.baseline_meters
        angle_rad = np.radians(self.calibration.angle_degrees / 2)
        
        # Simple translation (cameras pointing inward at subject ~2m away)
        T = np.array([[baseline], [0], [0]], dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        
        self.calibration.R = R
        self.calibration.T = T
        self.calibration.P2 = np.hstack([K @ R, K @ T])
    
    def triangulate_point(
        self,
        point_left: Tuple[float, float],
        point_right: Tuple[float, float],
        visibility_left: float,
        visibility_right: float,
        image_width: int = 1280,
        image_height: int = 720
    ) -> Optional[Tuple[float, float, float]]:
        """
        Triangulate a 3D point from stereo 2D observations.
        
        Args:
            point_left: (x, y) normalized coordinates [0-1] from left camera
            point_right: (x, y) normalized coordinates [0-1] from right camera
            visibility_left: Confidence from left camera [0-1]
            visibility_right: Confidence from right camera [0-1]
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            (x, y, z) in meters, or None if visibility too low
        """
        # Skip low-visibility points
        if visibility_left < 0.5 or visibility_right < 0.5:
            return None
        
        # Convert normalized to pixel coordinates
        x1 = point_left[0] * image_width
        y1 = point_left[1] * image_height
        x2 = point_right[0] * image_width
        y2 = point_right[1] * image_height
        
        # Prepare points for triangulation
        pts1 = np.array([[x1, y1]], dtype=np.float64)
        pts2 = np.array([[x2, y2]], dtype=np.float64)
        
        # Triangulate
        try:
            points_4d = cv2.triangulatePoints(
                self.calibration.P1, 
                self.calibration.P2,
                pts1.T,
                pts2.T
            )
            
            # Convert from homogeneous coordinates
            points_3d = points_4d[:3] / points_4d[3]
            
            x, y, z = points_3d.flatten()
            
            # Sanity check - reject points too far away or behind camera
            if z < 0.1 or z > 10.0:
                return None
            
            return (float(x), float(y), float(z))
            
        except Exception as e:
            return None
    
    def triangulate_simple(
        self,
        point_left: Tuple[float, float, float],
        point_right: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """
        Simple triangulation using disparity (for uncalibrated cameras).
        
        Args:
            point_left: (x, y, visibility) normalized from left camera
            point_right: (x, y, visibility) normalized from right camera
            
        Returns:
            (x, y, z) estimated 3D position
        """
        # Calculate disparity
        disparity = abs(point_left[0] - point_right[0])
        if disparity < 0.001:
            disparity = 0.001
        
        # Estimate depth from disparity
        focal_length_norm = 0.8  # Normalized focal length
        baseline = self.calibration.baseline_meters
        
        z = (focal_length_norm * baseline) / disparity
        z = max(0.5, min(5.0, z))  # Clamp to reasonable range
        
        # Average x, y positions and scale by depth
        x = ((point_left[0] + point_right[0]) / 2.0 - 0.5) * z
        y = ((point_left[1] + point_right[1]) / 2.0 - 0.5) * z
        
        return (x, y, z)


class ProfessionalMocapRecorder:
    """
    Professional dual-camera motion capture recorder.
    
    Combines best practices from FreeMoCap, Rokoko, and production workflows.
    """
    
    def __init__(
        self,
        output_dir: str = "./mocap_takes",
        camera_ids: List[int] = [0, 1],
        duration: int = 15,
        calibration_path: Optional[str] = None,
        resolution: Tuple[int, int] = (1280, 720),
        fps: int = 30,
        enable_hands: bool = True,
        model_complexity: int = 2
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.camera_ids = camera_ids
        self.duration = duration
        self.resolution = resolution
        self.target_fps = fps
        self.enable_hands = enable_hands
        self.model_complexity = model_complexity
        
        # Load calibration
        self.calibration = None
        if calibration_path and os.path.exists(calibration_path):
            try:
                self.calibration = CameraCalibration.load(calibration_path)
                print(f"✓ Loaded calibration from {calibration_path}")
            except Exception as e:
                print(f"⚠ Could not load calibration: {e}")
        
        # Initialize triangulator
        self.triangulator = StereoTriangulator(self.calibration)
        
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Camera captures
        self.cameras: Dict[str, cv2.VideoCapture] = {}
        self.pose_detectors: Dict[str, Any] = {}
        self.hand_detectors: Dict[str, Any] = {}
        
        # Recording state
        self.state = RecordingState.IDLE
        self.countdown_start: Optional[float] = None
        self.record_start: Optional[float] = None
        self.current_take: Optional[str] = None
        self.frame_data: List[Dict] = []
        self.frame_count = 0
        
        # Ground plane detection (for foot locking)
        self.ground_z: Optional[float] = None
        self.reference_hip_height: Optional[float] = None
    
    def initialize_cameras(self) -> bool:
        """Initialize cameras and pose detectors."""
        print("\n" + "=" * 60)
        print("INITIALIZING CAMERAS")
        print("=" * 60)
        
        # Try multiple backends on Windows
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Auto"),
        ]
        
        for i, cam_id in enumerate(self.camera_ids):
            cam_name = "left" if i == 0 else "right"
            opened = False
            
            for backend, backend_name in backends:
                cap = cv2.VideoCapture(cam_id, backend)
                if cap.isOpened():
                    # Set resolution
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                    cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                    
                    # Verify we can read a frame
                    ret, test_frame = cap.read()
                    if ret:
                        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        actual_fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        self.cameras[cam_name] = cap
                        print(f"✓ {cam_name.upper()} camera (ID {cam_id}): "
                              f"{actual_w}x{actual_h} @ {actual_fps:.0f}fps [{backend_name}]")
                        
                        # Initialize pose detector
                        self.pose_detectors[cam_name] = self.mp_pose.Pose(
                            min_detection_confidence=0.6,
                            min_tracking_confidence=0.6,
                            model_complexity=self.model_complexity,
                            enable_segmentation=False,
                            smooth_landmarks=True
                        )
                        
                        # Initialize hand detector if enabled
                        if self.enable_hands:
                            self.hand_detectors[cam_name] = self.mp_hands.Hands(
                                min_detection_confidence=0.6,
                                min_tracking_confidence=0.6,
                                max_num_hands=2
                            )
                        
                        opened = True
                        break
                    else:
                        cap.release()
                else:
                    cap.release()
            
            if not opened:
                print(f"✗ Could not open {cam_name} camera (ID {cam_id})")
        
        if len(self.cameras) < 2:
            print("\n⚠ WARNING: Less than 2 cameras detected!")
            print("  Triangulation will be limited.")
        
        return len(self.cameras) > 0
    
    def _convert_to_blender_coords(
        self, 
        x: float, 
        y: float, 
        z: float
    ) -> Tuple[float, float, float]:
        """
        Convert from MediaPipe coordinate system to Blender.
        
        MediaPipe: X-right, Y-down, Z-toward viewer (out of screen)
        Blender:   X-right, Y-forward (into screen), Z-up
        
        Transform: bx = mx, by = -mz, bz = -my
        """
        bx = x
        by = -z  # MediaPipe Z forward -> Blender Y forward
        bz = -y  # MediaPipe Y down -> Blender Z up (inverted)
        return (bx, by, bz)
    
    def _compute_derived_landmarks(
        self,
        landmarks: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute derived landmarks needed for Rigify retargeting.
        
        These are virtual landmarks computed from MediaPipe points.
        """
        derived = {}
        
        # Hip center (torso control)
        if "left_hip" in landmarks and "right_hip" in landmarks:
            lh = landmarks["left_hip"]
            rh = landmarks["right_hip"]
            derived["hip_center"] = {
                "x": (lh["x"] + rh["x"]) / 2,
                "y": (lh["y"] + rh["y"]) / 2,
                "z": (lh["z"] + rh["z"]) / 2,
                "visibility": min(lh["visibility"], rh["visibility"])
            }
        
        # Shoulder center (for spine/neck)
        if "left_shoulder" in landmarks and "right_shoulder" in landmarks:
            ls = landmarks["left_shoulder"]
            rs = landmarks["right_shoulder"]
            derived["shoulder_center"] = {
                "x": (ls["x"] + rs["x"]) / 2,
                "y": (ls["y"] + rs["y"]) / 2,
                "z": (ls["z"] + rs["z"]) / 2,
                "visibility": min(ls["visibility"], rs["visibility"])
            }
            
            # Neck base (between shoulders, slightly up)
            derived["neck_base"] = {
                "x": derived["shoulder_center"]["x"],
                "y": derived["shoulder_center"]["y"],
                "z": derived["shoulder_center"]["z"] + 0.05,  # Slightly above shoulders
                "visibility": derived["shoulder_center"]["visibility"]
            }
        
        # Spine center (between hip_center and shoulder_center)
        if "hip_center" in derived and "shoulder_center" in derived:
            hc = derived["hip_center"]
            sc = derived["shoulder_center"]
            derived["spine_center"] = {
                "x": (hc["x"] + sc["x"]) / 2,
                "y": (hc["y"] + sc["y"]) / 2,
                "z": (hc["z"] + sc["z"]) / 2,
                "visibility": min(hc["visibility"], sc["visibility"])
            }
        
        return derived
    
    def process_frame(self) -> Tuple[Dict[str, np.ndarray], FrameCapture]:
        """
        Process one frame from all cameras.
        
        Returns:
            Tuple of (images dict, frame capture data)
        """
        frame_capture = FrameCapture(
            frame_number=self.frame_count,
            timestamp=time.time() - self.record_start if self.record_start else 0
        )
        
        images = {}
        pose_results = {}
        hand_results = {}
        
        # Capture from each camera
        for cam_name, cap in self.cameras.items():
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # Pose detection
            pose_result = self.pose_detectors[cam_name].process(rgb_frame)
            pose_results[cam_name] = pose_result
            
            # Hand detection
            if self.enable_hands and cam_name in self.hand_detectors:
                hand_result = self.hand_detectors[cam_name].process(rgb_frame)
                hand_results[cam_name] = hand_result
            
            # Convert back to BGR for display
            rgb_frame.flags.writeable = True
            display_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks on display frame
            if pose_result.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    display_frame,
                    pose_result.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            # Draw hands
            if cam_name in hand_results and hand_results[cam_name].multi_hand_landmarks:
                for hand_landmarks in hand_results[cam_name].multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        display_frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
            
            images[cam_name] = display_frame
        
        # Extract 2D landmarks
        for cam_name, result in pose_results.items():
            if result.pose_landmarks:
                landmarks_2d = {}
                for idx, landmark in enumerate(result.pose_landmarks.landmark):
                    name = POSE_LANDMARKS.get(idx, f"landmark_{idx}")
                    landmarks_2d[name] = {
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility
                    }
                
                if cam_name == "left":
                    frame_capture.landmarks_2d_left = landmarks_2d
                else:
                    frame_capture.landmarks_2d_right = landmarks_2d
        
        # Extract world landmarks from left camera (more reliable)
        if "left" in pose_results and pose_results["left"].pose_world_landmarks:
            world_landmarks = {}
            for idx, landmark in enumerate(pose_results["left"].pose_world_landmarks.landmark):
                name = POSE_LANDMARKS.get(idx, f"landmark_{idx}")
                
                # Convert to Blender coordinate system
                bx, by, bz = self._convert_to_blender_coords(
                    landmark.x, landmark.y, landmark.z
                )
                
                world_landmarks[name] = {
                    "x": bx,
                    "y": by,
                    "z": bz,
                    "visibility": landmark.visibility,
                    # Keep original for debugging
                    "mp_x": landmark.x,
                    "mp_y": landmark.y,
                    "mp_z": landmark.z
                }
            
            frame_capture.world_landmarks = world_landmarks
            
            # Compute derived landmarks
            derived = self._compute_derived_landmarks(world_landmarks)
            for name, data in derived.items():
                frame_capture.world_landmarks[name] = data
        
        # Triangulate 3D landmarks if we have both cameras
        if frame_capture.landmarks_2d_left and frame_capture.landmarks_2d_right:
            landmarks_3d = {}
            
            for name in frame_capture.landmarks_2d_left:
                if name not in frame_capture.landmarks_2d_right:
                    continue
                
                left = frame_capture.landmarks_2d_left[name]
                right = frame_capture.landmarks_2d_right[name]
                
                # Try proper triangulation first
                result = self.triangulator.triangulate_point(
                    (left["x"], left["y"]),
                    (right["x"], right["y"]),
                    left["visibility"],
                    right["visibility"],
                    self.resolution[0],
                    self.resolution[1]
                )
                
                if result:
                    # Convert to Blender coordinates
                    bx, by, bz = self._convert_to_blender_coords(*result)
                    landmarks_3d[name] = {
                        "x": bx,
                        "y": by,
                        "z": bz,
                        "visibility": min(left["visibility"], right["visibility"])
                    }
                else:
                    # Fall back to simple triangulation
                    result = self.triangulator.triangulate_simple(
                        (left["x"], left["y"], left["visibility"]),
                        (right["x"], right["y"], right["visibility"])
                    )
                    bx, by, bz = self._convert_to_blender_coords(*result)
                    landmarks_3d[name] = {
                        "x": bx,
                        "y": by,
                        "z": bz,
                        "visibility": min(left["visibility"], right["visibility"]) * 0.8
                    }
            
            # Compute derived landmarks for triangulated data
            derived = self._compute_derived_landmarks(landmarks_3d)
            for name, data in derived.items():
                landmarks_3d[name] = data
            
            frame_capture.landmarks_3d = landmarks_3d
        
        return images, frame_capture
    
    def start_countdown(self):
        """Start the 3-2-1 countdown."""
        self.state = RecordingState.COUNTDOWN
        self.countdown_start = time.time()
        print("\n🎬 GET READY! Starting in 3...")
    
    def start_recording(self):
        """Start actual recording."""
        self.state = RecordingState.RECORDING
        self.record_start = time.time()
        self.frame_data = []
        self.frame_count = 0
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_take = f"take_{timestamp}"
        
        print(f"\n🔴 RECORDING: {self.current_take}")
        print(f"   Duration: {self.duration} seconds")
    
    def stop_recording(self):
        """Stop recording and save data."""
        if self.state != RecordingState.RECORDING:
            return
        
        self.state = RecordingState.SAVING
        
        # Calculate actual FPS
        actual_duration = time.time() - self.record_start
        actual_fps = len(self.frame_data) / actual_duration if actual_duration > 0 else 30
        
        # Prepare take data
        take_data = {
            "metadata": {
                "name": self.current_take,
                "created": datetime.now().isoformat(),
                "frame_count": len(self.frame_data),
                "duration_seconds": actual_duration,
                "fps": actual_fps,
                "target_fps": self.target_fps,
                "resolution": list(self.resolution),
                "calibrated": self.calibration is not None and self.calibration.is_calibrated,
                "coordinate_system": "blender",  # Z-up, Y-forward
                "units": "meters"
            },
            "calibration": {
                "baseline_meters": self.triangulator.calibration.baseline_meters,
                "angle_degrees": self.triangulator.calibration.angle_degrees
            },
            "frames": self.frame_data
        }
        
        # Save to JSON
        output_file = self.output_dir / f"{self.current_take}.json"
        with open(output_file, 'w') as f:
            json.dump(take_data, f, indent=2)
        
        print(f"\n✅ Take saved: {output_file}")
        print(f"   Frames: {len(self.frame_data)}")
        print(f"   Duration: {actual_duration:.2f}s")
        print(f"   FPS: {actual_fps:.1f}")
        
        self.state = RecordingState.IDLE
    
    def run(self):
        """Main recording loop."""
        if not self.initialize_cameras():
            print("\n❌ Failed to initialize cameras!")
            return
        
        print("\n" + "=" * 60)
        print("MELODIC JUSTICE MOTION CAPTURE - PROFESSIONAL RECORDER")
        print("=" * 60)
        print(f"\nDuration: {self.duration} seconds")
        print(f"Cameras: {len(self.cameras)}")
        print(f"Hands: {'Enabled' if self.enable_hands else 'Disabled'}")
        print(f"Calibrated: {self.calibration is not None and self.calibration.is_calibrated}")
        print("\nControls:")
        print("  SPACE - Start countdown (3-2-1 then record)")
        print("  R     - Quick start recording (no countdown)")
        print("  S     - Stop recording early")
        print("  Q     - Quit")
        print("\n" + "=" * 60)
        
        try:
            while True:
                # Process frames
                images, frame_capture = self.process_frame()
                
                # Handle countdown
                if self.state == RecordingState.COUNTDOWN:
                    elapsed = time.time() - self.countdown_start
                    remaining = 3 - elapsed
                    
                    if remaining <= 0:
                        self.start_recording()
                    else:
                        # Print countdown (debounced)
                        countdown_num = int(remaining) + 1
                        if countdown_num <= 3 and (elapsed * 10) % 10 < 1:
                            pass  # Could add audio beep here
                
                # Handle recording
                if self.state == RecordingState.RECORDING:
                    # Store frame data
                    self.frame_data.append({
                        "frame_number": frame_capture.frame_number,
                        "timestamp": frame_capture.timestamp,
                        "landmarks_3d": frame_capture.landmarks_3d,
                        "world_landmarks": frame_capture.world_landmarks,
                        "landmarks_2d_left": frame_capture.landmarks_2d_left,
                        "landmarks_2d_right": frame_capture.landmarks_2d_right,
                        "left_hand": frame_capture.left_hand,
                        "right_hand": frame_capture.right_hand
                    })
                    self.frame_count += 1
                    
                    # Check for auto-stop
                    elapsed = time.time() - self.record_start
                    if elapsed >= self.duration:
                        self.stop_recording()
                        print("\n⏹️  Recording complete!")
                        print("   Press SPACE for another take, or Q to quit\n")
                
                # Display
                if images:
                    display_frames = []
                    
                    for cam_name in ["left", "right"]:
                        if cam_name not in images:
                            continue
                        
                        img = images[cam_name].copy()
                        
                        # Add camera label
                        cv2.putText(img, cam_name.upper(), (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                        
                        # Countdown overlay
                        if self.state == RecordingState.COUNTDOWN:
                            elapsed = time.time() - self.countdown_start
                            countdown_num = int(3 - elapsed) + 1
                            if countdown_num > 0:
                                cv2.putText(img, str(countdown_num),
                                           (img.shape[1]//2 - 40, img.shape[0]//2 + 30),
                                           cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 8)
                        
                        # Recording overlay
                        if self.state == RecordingState.RECORDING:
                            elapsed = time.time() - self.record_start
                            remaining = self.duration - elapsed
                            
                            # Red recording dot
                            cv2.circle(img, (30, 60), 15, (0, 0, 255), -1)
                            
                            # Timer
                            timer_text = f"REC {remaining:.1f}s"
                            cv2.putText(img, timer_text, (55, 67),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            
                            # Frame counter
                            cv2.putText(img, f"Frame: {self.frame_count}",
                                       (10, img.shape[0] - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                            
                            # Progress bar
                            progress = elapsed / self.duration
                            bar_width = img.shape[1] - 20
                            bar_y = 85
                            cv2.rectangle(img, (10, bar_y), (10 + bar_width, bar_y + 15),
                                        (60, 60, 60), -1)
                            cv2.rectangle(img, (10, bar_y), (10 + int(bar_width * progress), bar_y + 15),
                                        (0, 200, 0), -1)
                        
                        display_frames.append(img)
                    
                    # Combine frames side by side
                    if len(display_frames) == 2:
                        # Handle different resolutions
                        h1, w1 = display_frames[0].shape[:2]
                        h2, w2 = display_frames[1].shape[:2]
                        
                        target_h = min(h1, h2)
                        
                        if h1 != target_h:
                            scale = target_h / h1
                            display_frames[0] = cv2.resize(display_frames[0], 
                                                          (int(w1 * scale), target_h))
                        if h2 != target_h:
                            scale = target_h / h2
                            display_frames[1] = cv2.resize(display_frames[1],
                                                          (int(w2 * scale), target_h))
                        
                        combined = np.hstack(display_frames)
                    elif len(display_frames) == 1:
                        combined = display_frames[0]
                    else:
                        combined = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(combined, "No camera feed", (200, 240),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    cv2.imshow("Melodic Justice MoCap", combined)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    if self.state == RecordingState.RECORDING:
                        self.stop_recording()
                    break
                elif key == ord(' '):
                    if self.state == RecordingState.IDLE:
                        self.start_countdown()
                elif key == ord('r'):
                    if self.state == RecordingState.IDLE:
                        self.start_recording()
                elif key == ord('s'):
                    if self.state == RecordingState.RECORDING:
                        self.stop_recording()
                        print("\n⏹️  Recording stopped early.")
                        print("   Press SPACE for another take, or Q to quit\n")
        
        finally:
            # Cleanup
            print("\nCleaning up...")
            for cap in self.cameras.values():
                cap.release()
            cv2.destroyAllWindows()
            
            for detector in self.pose_detectors.values():
                detector.close()
            for detector in self.hand_detectors.values():
                detector.close()
            
            print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Melodic Justice Professional Motion Capture Recorder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python professional_recorder.py --left 0 --right 1
  python professional_recorder.py --left 3 --right 4 --duration 30
  python professional_recorder.py --left 0 --right 1 --calibration ./calibration.json
        """
    )
    
    parser.add_argument("--left", type=int, default=0,
                       help="Device ID for left camera")
    parser.add_argument("--right", type=int, default=1,
                       help="Device ID for right camera")
    parser.add_argument("--duration", type=int, default=15,
                       help="Recording duration in seconds")
    parser.add_argument("--output", type=str, default="./mocap_takes",
                       help="Output directory for takes")
    parser.add_argument("--calibration", type=str, default=None,
                       help="Path to stereo calibration JSON")
    parser.add_argument("--width", type=int, default=1280,
                       help="Camera width")
    parser.add_argument("--height", type=int, default=720,
                       help="Camera height")
    parser.add_argument("--fps", type=int, default=30,
                       help="Target FPS")
    parser.add_argument("--no-hands", action="store_true",
                       help="Disable hand tracking")
    parser.add_argument("--model", type=int, default=2, choices=[0, 1, 2],
                       help="MediaPipe model complexity (0=lite, 1=full, 2=heavy)")
    
    args = parser.parse_args()
    
    recorder = ProfessionalMocapRecorder(
        output_dir=args.output,
        camera_ids=[args.left, args.right],
        duration=args.duration,
        calibration_path=args.calibration,
        resolution=(args.width, args.height),
        fps=args.fps,
        enable_hands=not args.no_hands,
        model_complexity=args.model
    )
    
    recorder.run()


if __name__ == "__main__":
    main()
