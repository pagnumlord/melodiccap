"""
Melodic Justice Motion Capture - Dual Camera Capture
====================================================
Dual camera recording with MediaPipe pose detection and 3D triangulation.
"""

import cv2
import numpy as np
import mediapipe as mp
import json
import os
import time
import threading
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from queue import Queue
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from config import (
        MocapConfig, RecordingConfig, POSE_LANDMARKS, HAND_LANDMARKS,
        MEDIAPIPE_TO_RIGIFY_MAPPING
    )
except ImportError:
    print("Warning: Could not import config, using defaults")
    POSE_LANDMARKS = {i: f"landmark_{i}" for i in range(33)}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Landmark3D:
    """A single 3D landmark point."""
    x: float
    y: float
    z: float
    visibility: float = 1.0
    name: str = ""


@dataclass
class FrameData:
    """Data for a single captured frame."""
    frame_number: int
    timestamp: float
    
    # 2D landmarks (normalized 0-1)
    landmarks_2d_left: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    landmarks_2d_right: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    
    # 3D triangulated landmarks (world coordinates)
    landmarks_3d: Dict[str, Landmark3D] = field(default_factory=dict)
    
    # Hand landmarks
    left_hand_landmarks: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    right_hand_landmarks: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)


@dataclass
class TakeData:
    """Data for a complete motion capture take."""
    name: str
    timestamp: str
    fps: int
    frame_count: int
    duration_seconds: float
    
    # Calibration used
    calibration_file: str = ""
    
    # Frame data
    frames: List[Dict] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# STEREO TRIANGULATION
# ============================================================================

class StereoTriangulator:
    """Handles 3D triangulation from stereo camera pairs."""
    
    def __init__(self, calibration_path: str = None):
        self.calibration_loaded = False
        self.P1 = None  # Projection matrix camera 1
        self.P2 = None  # Projection matrix camera 2
        self.Q = None   # Disparity-to-depth mapping matrix
        
        if calibration_path and os.path.exists(calibration_path):
            self.load_calibration(calibration_path)
    
    def load_calibration(self, filepath: str):
        """Load stereo calibration from JSON file."""
        with open(filepath, 'r') as f:
            calib = json.load(f)
        
        # Extract matrices
        cam_matrix_l = np.array(calib['camera_left']['camera_matrix'])
        cam_matrix_r = np.array(calib['camera_right']['camera_matrix'])
        
        stereo = calib['stereo']
        R = np.array(stereo['R'])
        T = np.array(stereo['T']).reshape(3, 1)
        
        # Build projection matrices
        # P1 = K1 [I | 0]
        self.P1 = cam_matrix_l @ np.hstack([np.eye(3), np.zeros((3, 1))])
        
        # P2 = K2 [R | T]
        self.P2 = cam_matrix_r @ np.hstack([R, T])
        
        # Store Q matrix for depth computation
        self.Q = np.array(stereo['Q'])
        
        self.calibration_loaded = True
        print(f"Loaded stereo calibration from: {filepath}")
    
    def triangulate_point(
        self, 
        point_left: Tuple[float, float],
        point_right: Tuple[float, float],
        image_size: Tuple[int, int] = (1280, 720)
    ) -> np.ndarray:
        """
        Triangulate a single 3D point from stereo correspondences.
        
        Args:
            point_left: (x, y) in left image (normalized 0-1)
            point_right: (x, y) in right image (normalized 0-1)
            image_size: Image dimensions for denormalization
            
        Returns:
            3D point as numpy array [x, y, z]
        """
        if not self.calibration_loaded:
            # Fallback: simple depth estimation without calibration
            return self._estimate_depth_simple(point_left, point_right)
        
        # Denormalize to pixel coordinates
        pt_l = np.array([
            point_left[0] * image_size[0],
            point_left[1] * image_size[1]
        ]).reshape(2, 1)
        
        pt_r = np.array([
            point_right[0] * image_size[0],
            point_right[1] * image_size[1]
        ]).reshape(2, 1)
        
        # Triangulate
        point_4d = cv2.triangulatePoints(self.P1, self.P2, pt_l, pt_r)
        
        # Convert from homogeneous coordinates
        point_3d = point_4d[:3] / point_4d[3]
        
        return point_3d.flatten()
    
    def _estimate_depth_simple(
        self, 
        point_left: Tuple[float, float],
        point_right: Tuple[float, float],
        baseline: float = 0.5,
        focal_length: float = 1000
    ) -> np.ndarray:
        """
        Simple depth estimation without full calibration.
        Uses basic stereo geometry assumptions.
        """
        # Disparity (difference in x coordinates)
        disparity = abs(point_left[0] - point_right[0])
        
        if disparity < 0.001:
            disparity = 0.001  # Avoid division by zero
        
        # Approximate depth using Z = (f * b) / d
        depth = (focal_length * baseline) / (disparity * 1280)
        
        # Use left camera as reference for x, y
        x = (point_left[0] - 0.5) * depth
        y = (point_left[1] - 0.5) * depth
        z = depth
        
        return np.array([x, y, z])
    
    def triangulate_landmarks(
        self,
        landmarks_left: Dict[str, Tuple[float, float, float]],
        landmarks_right: Dict[str, Tuple[float, float, float]],
        image_size: Tuple[int, int] = (1280, 720)
    ) -> Dict[str, Landmark3D]:
        """
        Triangulate all matching landmarks between two views.
        
        Returns:
            Dictionary of 3D landmarks
        """
        landmarks_3d = {}
        
        # Find common landmarks
        common_keys = set(landmarks_left.keys()) & set(landmarks_right.keys())
        
        for key in common_keys:
            lm_l = landmarks_left[key]
            lm_r = landmarks_right[key]
            
            # Check visibility threshold
            if lm_l[2] < 0.5 or lm_r[2] < 0.5:
                continue
            
            # Triangulate
            point_3d = self.triangulate_point(
                (lm_l[0], lm_l[1]),
                (lm_r[0], lm_r[1]),
                image_size
            )
            
            # Average visibility
            visibility = (lm_l[2] + lm_r[2]) / 2
            
            landmarks_3d[key] = Landmark3D(
                x=float(point_3d[0]),
                y=float(point_3d[1]),
                z=float(point_3d[2]),
                visibility=visibility,
                name=key
            )
        
        return landmarks_3d


# ============================================================================
# MEDIAPIPE POSE DETECTOR
# ============================================================================

class PoseDetector:
    """MediaPipe-based pose and hand detection."""
    
    def __init__(
        self,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        enable_hands: bool = True
    ):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize pose detector
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            enable_segmentation=False
        )
        
        # Initialize hand detector
        self.hands = None
        if enable_hands:
            self.hands = self.mp_hands.Hands(
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Dict, Dict, Dict, np.ndarray]:
        """
        Process a frame for pose and hand detection.
        
        Returns:
            Tuple of (pose_landmarks, left_hand, right_hand, annotated_frame)
        """
        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        
        # Detect pose
        pose_results = self.pose.process(rgb)
        
        # Detect hands
        hand_results = None
        if self.hands:
            hand_results = self.hands.process(rgb)
        
        rgb.flags.writeable = True
        annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # Extract pose landmarks
        pose_landmarks = {}
        if pose_results.pose_landmarks:
            for idx, lm in enumerate(pose_results.pose_landmarks.landmark):
                name = POSE_LANDMARKS.get(idx, f"landmark_{idx}")
                pose_landmarks[name] = (lm.x, lm.y, lm.visibility)
            
            # Draw pose
            self.mp_drawing.draw_landmarks(
                annotated,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        
        # Extract hand landmarks
        left_hand = {}
        right_hand = {}
        
        if hand_results and hand_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                # Determine handedness
                handedness = hand_results.multi_handedness[idx].classification[0].label
                
                landmarks = {}
                for i, lm in enumerate(hand_landmarks.landmark):
                    landmarks[f"hand_{i}"] = (lm.x, lm.y, lm.visibility if hasattr(lm, 'visibility') else 1.0)
                
                if handedness == "Left":
                    left_hand = landmarks
                else:
                    right_hand = landmarks
                
                # Draw hand
                self.mp_drawing.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        
        return pose_landmarks, left_hand, right_hand, annotated
    
    def close(self):
        """Release resources."""
        self.pose.close()
        if self.hands:
            self.hands.close()


# ============================================================================
# DUAL CAMERA RECORDER
# ============================================================================

class DualCameraRecorder:
    """Records synchronized dual-camera motion capture."""
    
    def __init__(
        self,
        output_dir: str = "./mocap_data",
        calibration_path: str = None,
        config: RecordingConfig = None
    ):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.config = config or RecordingConfig()
        
        # Initialize triangulator
        self.triangulator = StereoTriangulator(calibration_path)
        
        # Initialize pose detectors (one per camera for thread safety)
        self.detector_left = PoseDetector(
            model_complexity=self.config.model_complexity,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            enable_hands=self.config.enable_hands
        )
        self.detector_right = PoseDetector(
            model_complexity=self.config.model_complexity,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            enable_hands=self.config.enable_hands
        )
        
        # Recording state
        self.is_recording = False
        self.take_data: Optional[TakeData] = None
        self.frame_count = 0
        self.start_time = 0
        
        # Cameras
        self.cap_left = None
        self.cap_right = None
        self.image_size = (1280, 720)
    
    def _open_camera(self, device_id: int, name: str) -> Optional[cv2.VideoCapture]:
        """Open camera with fallback backends."""
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_ANY, "Auto"),
        ]
        
        for backend, backend_name in backends:
            try:
                cap = cv2.VideoCapture(device_id, backend)
                if cap.isOpened():
                    # Set resolution
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_size[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_size[1])
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    # Verify resolution
                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    print(f"Opened {name} camera (ID: {device_id}) via {backend_name}")
                    print(f"  Resolution: {actual_w}x{actual_h}")
                    
                    return cap
                cap.release()
            except Exception as e:
                print(f"Failed to open with {backend_name}: {e}")
        
        return None
    
    def detect_cameras(self) -> List[Tuple[int, str]]:
        """Detect available cameras."""
        cameras = []
        
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to get camera name
                name = f"Camera {i}"
                try:
                    # Get backend name
                    backend = cap.getBackendName()
                    name = f"Camera {i} ({backend})"
                except:
                    pass
                cameras.append((i, name))
                cap.release()
        
        return cameras
    
    def start_preview(
        self,
        camera_left_id: int = 0,
        camera_right_id: int = 1
    ):
        """Start preview mode to check camera setup."""
        self.cap_left = self._open_camera(camera_left_id, "LEFT")
        self.cap_right = self._open_camera(camera_right_id, "RIGHT")
        
        if not self.cap_left or not self.cap_right:
            print("Failed to open both cameras!")
            return
        
        print("\nPREVIEW MODE")
        print("=" * 40)
        print("Controls:")
        print("  R - Start/Stop recording")
        print("  Q - Quit")
        print("=" * 40)
        
        while True:
            ret_l, frame_l = self.cap_left.read()
            ret_r, frame_r = self.cap_right.read()
            
            if not ret_l or not ret_r:
                continue
            
            # Process frames
            pose_l, hand_l_l, hand_r_l, annotated_l = self.detector_left.process_frame(frame_l)
            pose_r, hand_l_r, hand_r_r, annotated_r = self.detector_right.process_frame(frame_r)
            
            # Add status overlay
            status = "RECORDING" if self.is_recording else "PREVIEW"
            color = (0, 0, 255) if self.is_recording else (0, 255, 0)
            
            cv2.putText(annotated_l, f"LEFT - {status}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(annotated_r, f"RIGHT - {status}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            if self.is_recording:
                cv2.putText(annotated_l, f"Frame: {self.frame_count}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Record frame
                self._record_frame(pose_l, pose_r, hand_l_l, hand_r_l)
            
            # Show landmarks count
            cv2.putText(annotated_l, f"Pose: {len(pose_l)} pts", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_r, f"Pose: {len(pose_r)} pts", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Resize for display if needed
            scale = 0.7
            display_l = cv2.resize(annotated_l, None, fx=scale, fy=scale)
            display_r = cv2.resize(annotated_r, None, fx=scale, fy=scale)
            
            # Show combined view
            combined = np.hstack([display_l, display_r])
            cv2.imshow("Dual Camera Motion Capture", combined)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):
                if self.is_recording:
                    self._stop_recording()
                else:
                    self._start_recording()
            
            elif key == ord('q'):
                if self.is_recording:
                    self._stop_recording()
                break
        
        self.cap_left.release()
        self.cap_right.release()
        cv2.destroyAllWindows()
        
        self.detector_left.close()
        self.detector_right.close()
    
    def _start_recording(self):
        """Start a new recording take."""
        self.is_recording = True
        self.frame_count = 0
        self.start_time = time.time()
        
        take_name = datetime.now().strftime("take_%Y%m%d_%H%M%S")
        
        self.take_data = TakeData(
            name=take_name,
            timestamp=datetime.now().isoformat(),
            fps=30,
            frame_count=0,
            duration_seconds=0,
            frames=[]
        )
        
        print(f"\n>>> Recording started: {take_name}")
    
    def _stop_recording(self):
        """Stop recording and save the take."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        duration = time.time() - self.start_time
        self.take_data.frame_count = self.frame_count
        self.take_data.duration_seconds = duration
        
        # Save take
        output_path = os.path.join(
            self.output_dir, 
            f"{self.take_data.name}.json"
        )
        
        # Convert to serializable format
        take_dict = asdict(self.take_data)
        
        with open(output_path, 'w') as f:
            json.dump(take_dict, f, indent=2, default=self._json_serializer)
        
        print(f"\n>>> Recording saved: {output_path}")
        print(f"    Frames: {self.frame_count}")
        print(f"    Duration: {duration:.2f}s")
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for dataclasses."""
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)
    
    def _record_frame(
        self,
        pose_left: Dict,
        pose_right: Dict,
        hand_left: Dict,
        hand_right: Dict
    ):
        """Record a single frame of motion data."""
        timestamp = time.time() - self.start_time
        
        # Triangulate 3D landmarks
        landmarks_3d = self.triangulator.triangulate_landmarks(
            pose_left,
            pose_right,
            self.image_size
        )
        
        # Convert to dict for storage
        landmarks_3d_dict = {}
        for key, lm in landmarks_3d.items():
            landmarks_3d_dict[key] = {
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility
            }
        
        frame_data = {
            'frame_number': self.frame_count,
            'timestamp': timestamp,
            'landmarks_2d_left': pose_left,
            'landmarks_2d_right': pose_right,
            'landmarks_3d': landmarks_3d_dict,
            'left_hand': hand_left,
            'right_hand': hand_right
        }
        
        self.take_data.frames.append(frame_data)
        self.frame_count += 1


# ============================================================================
# TAKE MANAGER
# ============================================================================

class TakeManager:
    """Manages saved motion capture takes."""
    
    def __init__(self, data_dir: str = "./mocap_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def list_takes(self) -> List[str]:
        """List all available takes."""
        takes = []
        for f in os.listdir(self.data_dir):
            if f.endswith('.json') and f.startswith('take_'):
                takes.append(f[:-5])  # Remove .json extension
        return sorted(takes)
    
    def load_take(self, take_name: str) -> Optional[TakeData]:
        """Load a take from disk."""
        filepath = os.path.join(self.data_dir, f"{take_name}.json")
        
        if not os.path.exists(filepath):
            print(f"Take not found: {take_name}")
            return None
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return TakeData(**data)
    
    def export_for_blender(self, take_name: str, output_path: str = None) -> str:
        """
        Export a take in Blender-compatible format.
        
        This creates a simplified JSON file optimized for the Blender addon.
        """
        take = self.load_take(take_name)
        if not take:
            return None
        
        if output_path is None:
            output_path = os.path.join(self.data_dir, f"{take_name}_blender.json")
        
        # Build Blender-compatible export
        export_data = {
            'version': '1.0',
            'format': 'melodic_justice_mocap',
            'name': take.name,
            'fps': take.fps,
            'frame_count': take.frame_count,
            'duration': take.duration_seconds,
            
            # Coordinate system info
            'coordinate_system': {
                'source': 'mediapipe',  # Y-up, Z-forward
                'target': 'blender',     # Z-up, -Y-forward
                'transform_applied': True
            },
            
            # Frame data with transformed coordinates
            'frames': []
        }
        
        for frame in take.frames:
            transformed_landmarks = {}
            
            # Transform each 3D landmark to Blender coordinates
            for name, lm in frame.get('landmarks_3d', {}).items():
                # MediaPipe: X-right, Y-down, Z-forward
                # Blender:   X-right, Y-forward, Z-up
                transformed_landmarks[name] = {
                    'x': lm['x'],
                    'y': -lm['z'],  # Z becomes -Y
                    'z': -lm['y'],  # -Y becomes Z
                    'visibility': lm['visibility']
                }
            
            export_data['frames'].append({
                'frame': frame['frame_number'],
                'time': frame['timestamp'],
                'landmarks': transformed_landmarks
            })
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Exported for Blender: {output_path}")
        return output_path


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for dual camera capture."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dual Camera Motion Capture")
    parser.add_argument('--preview', action='store_true',
                       help="Start preview/recording mode")
    parser.add_argument('--detect-cameras', action='store_true',
                       help="Detect available cameras")
    parser.add_argument('--left-camera', type=int, default=0,
                       help="Left camera device ID")
    parser.add_argument('--right-camera', type=int, default=1,
                       help="Right camera device ID")
    parser.add_argument('--calibration', type=str, default=None,
                       help="Path to stereo calibration JSON")
    parser.add_argument('--output-dir', type=str, default="./mocap_data",
                       help="Output directory for takes")
    parser.add_argument('--list-takes', action='store_true',
                       help="List saved takes")
    parser.add_argument('--export', type=str, default=None,
                       help="Export a take for Blender")
    
    args = parser.parse_args()
    
    if args.detect_cameras:
        recorder = DualCameraRecorder()
        cameras = recorder.detect_cameras()
        print("\nDetected Cameras:")
        for cam_id, cam_name in cameras:
            print(f"  [{cam_id}] {cam_name}")
    
    elif args.preview:
        recorder = DualCameraRecorder(
            output_dir=args.output_dir,
            calibration_path=args.calibration
        )
        recorder.start_preview(args.left_camera, args.right_camera)
    
    elif args.list_takes:
        manager = TakeManager(args.output_dir)
        takes = manager.list_takes()
        print("\nSaved Takes:")
        for take in takes:
            print(f"  - {take}")
    
    elif args.export:
        manager = TakeManager(args.output_dir)
        manager.export_for_blender(args.export)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
