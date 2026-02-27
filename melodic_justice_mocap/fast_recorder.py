#!/usr/bin/env python3
"""
Melodic Justice MoCap - FAST Recorder
=====================================

Optimized for high FPS by:
- Using lighter MediaPipe model
- Disabling hand tracking by default
- Processing pose on ONE camera only (using world landmarks)
- Optional: second camera for stereo verification

This should give 15-25+ FPS instead of 4 FPS.

Usage:
    python fast_recorder.py --left 2 --right 4 --duration 15
    python fast_recorder.py --left 2 --right 4 --duration 15 --stereo
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import argparse
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# MediaPipe landmark names
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


class FastMocapRecorder:
    """
    High-performance motion capture recorder.
    
    Key optimization: Uses MediaPipe's world_landmarks which gives 3D positions
    in meters directly, without needing stereo triangulation for basic capture.
    """
    
    def __init__(
        self,
        output_dir: str = "./mocap_takes",
        left_camera_id: int = 0,
        right_camera_id: int = 1,
        duration: int = 15,
        use_stereo: bool = False,
        model_complexity: int = 1,  # 0=lite, 1=full, 2=heavy
        target_fps: int = 30
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.left_id = left_camera_id
        self.right_id = right_camera_id
        self.duration = duration
        self.use_stereo = use_stereo
        self.model_complexity = model_complexity
        self.target_fps = target_fps
        
        # MediaPipe setup - SINGLE detector for speed
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Only one pose detector (major speed improvement!)
        self.pose_detector = None
        
        # Cameras
        self.cap_left = None
        self.cap_right = None
        
        # Recording state
        self.is_recording = False
        self.is_countdown = False
        self.countdown_start = None
        self.record_start = None
        self.frame_data = []
        self.frame_count = 0
        self.current_take = None
    
    def initialize(self) -> bool:
        """Initialize cameras and pose detector."""
        print("\n" + "="*60)
        print("INITIALIZING FAST RECORDER")
        print("="*60)
        
        # Initialize pose detector with chosen complexity
        print(f"Model complexity: {self.model_complexity} ", end="")
        print(["(lite/fastest)", "(full/balanced)", "(heavy/accurate)"][self.model_complexity])
        
        self.pose_detector = self.mp_pose.Pose(
            model_complexity=self.model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=False,
            smooth_landmarks=True
        )
        
        # Initialize left camera (primary - for pose detection)
        backends = [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_ANY, "Auto")]
        
        for backend, name in backends:
            self.cap_left = cv2.VideoCapture(self.left_id, backend)
            if self.cap_left.isOpened():
                self.cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap_left.set(cv2.CAP_PROP_FPS, self.target_fps)
                self.cap_left.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
                ret, _ = self.cap_left.read()
                if ret:
                    w = int(self.cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(self.cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"✓ Left camera (ID {self.left_id}): {w}x{h} [{name}]")
                    break
        
        if not self.cap_left or not self.cap_left.isOpened():
            print(f"✗ Failed to open left camera (ID {self.left_id})")
            return False
        
        # Initialize right camera (secondary - just for display/stereo)
        if self.use_stereo:
            for backend, name in backends:
                self.cap_right = cv2.VideoCapture(self.right_id, backend)
                if self.cap_right.isOpened():
                    self.cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.cap_right.set(cv2.CAP_PROP_FPS, self.target_fps)
                    self.cap_right.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, _ = self.cap_right.read()
                    if ret:
                        print(f"✓ Right camera (ID {self.right_id}): opened [{name}]")
                        break
            
            if not self.cap_right or not self.cap_right.isOpened():
                print(f"⚠ Right camera not available, continuing with single camera")
                self.use_stereo = False
        else:
            # Still open right camera for display, but don't process
            for backend, name in backends:
                self.cap_right = cv2.VideoCapture(self.right_id, backend)
                if self.cap_right.isOpened():
                    self.cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.cap_right.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    print(f"✓ Right camera (ID {self.right_id}): display only [{name}]")
                    break
        
        print("="*60 + "\n")
        return True
    
    def convert_to_blender(self, x: float, y: float, z: float):
        """
        Convert MediaPipe world coordinates to Blender.
        
        MediaPipe: X-right, Y-down, Z-toward camera
        Blender:   X-right, Y-forward, Z-up
        """
        return (x, -z, -y)
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """Process a single frame and extract pose landmarks."""
        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        
        # Process with MediaPipe
        results = self.pose_detector.process(rgb)
        
        frame_data = {
            "landmarks_3d": {},
            "landmarks_2d": {},
            "visibility": {}
        }
        
        if results.pose_world_landmarks:
            # World landmarks are in meters, centered at hip
            for idx, landmark in enumerate(results.pose_world_landmarks.landmark):
                name = POSE_LANDMARKS.get(idx, f"landmark_{idx}")
                
                # Convert to Blender coordinates
                bx, by, bz = self.convert_to_blender(
                    landmark.x, landmark.y, landmark.z
                )
                
                frame_data["landmarks_3d"][name] = {
                    "x": bx,
                    "y": by,
                    "z": bz,
                    "visibility": landmark.visibility
                }
        
        if results.pose_landmarks:
            # 2D landmarks (normalized screen coordinates)
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                name = POSE_LANDMARKS.get(idx, f"landmark_{idx}")
                frame_data["landmarks_2d"][name] = {
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility
                }
        
        # Compute derived landmarks
        if "left_hip" in frame_data["landmarks_3d"] and "right_hip" in frame_data["landmarks_3d"]:
            lh = frame_data["landmarks_3d"]["left_hip"]
            rh = frame_data["landmarks_3d"]["right_hip"]
            frame_data["landmarks_3d"]["hip_center"] = {
                "x": (lh["x"] + rh["x"]) / 2,
                "y": (lh["y"] + rh["y"]) / 2,
                "z": (lh["z"] + rh["z"]) / 2,
                "visibility": min(lh["visibility"], rh["visibility"])
            }
        
        if "left_shoulder" in frame_data["landmarks_3d"] and "right_shoulder" in frame_data["landmarks_3d"]:
            ls = frame_data["landmarks_3d"]["left_shoulder"]
            rs = frame_data["landmarks_3d"]["right_shoulder"]
            frame_data["landmarks_3d"]["shoulder_center"] = {
                "x": (ls["x"] + rs["x"]) / 2,
                "y": (ls["y"] + rs["y"]) / 2,
                "z": (ls["z"] + rs["z"]) / 2,
                "visibility": min(ls["visibility"], rs["visibility"])
            }
            
            # Neck base
            frame_data["landmarks_3d"]["neck_base"] = {
                "x": frame_data["landmarks_3d"]["shoulder_center"]["x"],
                "y": frame_data["landmarks_3d"]["shoulder_center"]["y"],
                "z": frame_data["landmarks_3d"]["shoulder_center"]["z"] + 0.05,
                "visibility": frame_data["landmarks_3d"]["shoulder_center"]["visibility"]
            }
        
        # Spine center
        if "hip_center" in frame_data["landmarks_3d"] and "shoulder_center" in frame_data["landmarks_3d"]:
            hc = frame_data["landmarks_3d"]["hip_center"]
            sc = frame_data["landmarks_3d"]["shoulder_center"]
            frame_data["landmarks_3d"]["spine_center"] = {
                "x": (hc["x"] + sc["x"]) / 2,
                "y": (hc["y"] + sc["y"]) / 2,
                "z": (hc["z"] + sc["z"]) / 2,
                "visibility": min(hc["visibility"], sc["visibility"])
            }
        
        return frame_data, results
    
    def draw_landmarks(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw pose landmarks on frame."""
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return frame
    
    def start_countdown(self):
        """Start 3-2-1 countdown."""
        self.is_countdown = True
        self.countdown_start = time.time()
        print("\n🎬 GET READY! 3...")
    
    def start_recording(self):
        """Start recording."""
        self.is_recording = True
        self.is_countdown = False
        self.record_start = time.time()
        self.frame_data = []
        self.frame_count = 0
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_take = f"take_{timestamp}"
        
        print(f"\n🔴 RECORDING: {self.current_take}")
        print(f"   Duration: {self.duration} seconds")
    
    def stop_recording(self):
        """Stop recording and save."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        actual_duration = time.time() - self.record_start
        actual_fps = len(self.frame_data) / actual_duration if actual_duration > 0 else 0
        
        # Save take
        take_data = {
            "metadata": {
                "name": self.current_take,
                "created": datetime.now().isoformat(),
                "frame_count": len(self.frame_data),
                "duration_seconds": actual_duration,
                "fps": actual_fps,
                "model_complexity": self.model_complexity,
                "coordinate_system": "blender",
                "units": "meters"
            },
            "frames": self.frame_data
        }
        
        output_file = self.output_dir / f"{self.current_take}.json"
        with open(output_file, 'w') as f:
            json.dump(take_data, f)
        
        print(f"\n✅ Take saved: {output_file}")
        print(f"   Frames: {len(self.frame_data)}")
        print(f"   Duration: {actual_duration:.2f}s")
        print(f"   FPS: {actual_fps:.1f}")
    
    def run(self):
        """Main recording loop."""
        if not self.initialize():
            print("Failed to initialize!")
            return
        
        print("="*60)
        print("FAST MOTION CAPTURE RECORDER")
        print("="*60)
        print(f"Duration: {self.duration}s")
        print(f"Model: {['Lite', 'Full', 'Heavy'][self.model_complexity]}")
        print(f"Stereo: {'Yes' if self.use_stereo else 'No (single camera 3D)'}")
        print("\nControls:")
        print("  SPACE - Start countdown")
        print("  R     - Record immediately")
        print("  S     - Stop early")
        print("  Q     - Quit")
        print("="*60 + "\n")
        
        fps_counter = []
        last_time = time.time()
        
        try:
            while True:
                loop_start = time.time()
                
                # Read frames
                ret_l, frame_l = self.cap_left.read()
                if not ret_l:
                    continue
                
                frame_r = None
                if self.cap_right and self.cap_right.isOpened():
                    ret_r, frame_r = self.cap_right.read()
                
                # Process pose (only on left camera for speed!)
                frame_result, results = self.process_frame(frame_l)
                
                # Draw landmarks
                display_l = self.draw_landmarks(frame_l.copy(), results)
                
                # Calculate FPS
                current_time = time.time()
                fps_counter.append(1.0 / (current_time - last_time) if current_time > last_time else 30)
                last_time = current_time
                if len(fps_counter) > 30:
                    fps_counter.pop(0)
                current_fps = sum(fps_counter) / len(fps_counter)
                
                # Handle countdown
                if self.is_countdown:
                    elapsed = time.time() - self.countdown_start
                    remaining = 3 - elapsed
                    if remaining <= 0:
                        self.start_recording()
                    else:
                        countdown_num = int(remaining) + 1
                        cv2.putText(display_l, str(countdown_num), 
                                   (display_l.shape[1]//2 - 50, display_l.shape[0]//2 + 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10)
                
                # Handle recording
                if self.is_recording:
                    elapsed = time.time() - self.record_start
                    remaining = self.duration - elapsed
                    
                    # Store frame
                    self.frame_data.append({
                        "frame_number": self.frame_count,
                        "timestamp": elapsed,
                        "landmarks_3d": frame_result["landmarks_3d"],
                        "landmarks_2d": frame_result["landmarks_2d"]
                    })
                    self.frame_count += 1
                    
                    # Check auto-stop
                    if remaining <= 0:
                        self.stop_recording()
                        print("\n⏹️  Recording complete!")
                        print("   Press SPACE for another take, or Q to quit\n")
                    else:
                        # Recording overlay
                        cv2.circle(display_l, (30, 30), 15, (0, 0, 255), -1)
                        cv2.putText(display_l, f"REC {remaining:.1f}s", (55, 38),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                        # Progress bar
                        progress = elapsed / self.duration
                        bar_w = display_l.shape[1] - 40
                        cv2.rectangle(display_l, (20, 55), (20 + bar_w, 70), (60, 60, 60), -1)
                        cv2.rectangle(display_l, (20, 55), (20 + int(bar_w * progress), 70), (0, 200, 0), -1)
                
                # FPS display
                fps_color = (0, 255, 0) if current_fps >= 15 else (0, 255, 255) if current_fps >= 10 else (0, 0, 255)
                cv2.putText(display_l, f"FPS: {current_fps:.1f}", (display_l.shape[1] - 120, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
                
                # Camera labels
                cv2.putText(display_l, "LEFT (Processing)", (10, display_l.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Combine displays
                if frame_r is not None:
                    cv2.putText(frame_r, "RIGHT (Reference)", (10, frame_r.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    # Match heights
                    h1, h2 = display_l.shape[0], frame_r.shape[0]
                    if h1 != h2:
                        scale = h1 / h2
                        frame_r = cv2.resize(frame_r, (int(frame_r.shape[1] * scale), h1))
                    
                    combined = np.hstack([display_l, frame_r])
                else:
                    combined = display_l
                
                cv2.imshow("Fast MoCap Recorder", combined)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    if self.is_recording:
                        self.stop_recording()
                    break
                elif key == ord(' ') and not self.is_recording and not self.is_countdown:
                    self.start_countdown()
                elif key == ord('r') and not self.is_recording:
                    self.start_recording()
                elif key == ord('s') and self.is_recording:
                    self.stop_recording()
                    print("\n⏹️  Stopped early.")
        
        finally:
            print("\nCleaning up...")
            if self.cap_left:
                self.cap_left.release()
            if self.cap_right:
                self.cap_right.release()
            if self.pose_detector:
                self.pose_detector.close()
            cv2.destroyAllWindows()
            print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Fast Motion Capture Recorder (optimized for high FPS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Speed Tips:
  --model 0    Fastest (lite model, ~20-30 FPS)
  --model 1    Balanced (full model, ~15-20 FPS)  [default]
  --model 2    Accurate (heavy model, ~8-12 FPS)

Examples:
  python fast_recorder.py --left 2 --right 4 --duration 15
  python fast_recorder.py --left 2 --right 4 --model 0 --duration 30
        """
    )
    
    parser.add_argument("--left", type=int, default=0, help="Left camera ID")
    parser.add_argument("--right", type=int, default=1, help="Right camera ID")
    parser.add_argument("--duration", type=int, default=15, help="Recording duration")
    parser.add_argument("--model", type=int, default=1, choices=[0, 1, 2],
                       help="Model complexity: 0=lite, 1=full, 2=heavy")
    parser.add_argument("--stereo", action="store_true", 
                       help="Enable stereo processing (slower)")
    parser.add_argument("--output", type=str, default="./mocap_takes",
                       help="Output directory")
    
    args = parser.parse_args()
    
    recorder = FastMocapRecorder(
        output_dir=args.output,
        left_camera_id=args.left,
        right_camera_id=args.right,
        duration=args.duration,
        use_stereo=args.stereo,
        model_complexity=args.model
    )
    
    recorder.run()


if __name__ == "__main__":
    main()
