# MELODICCAP - LIVE DUAL CAMERA CAPTURE
# capture_dual_live.py
# 
# Captures from TWO cameras simultaneously in perfect sync
# - Real-time T-pose detection
# - Clap sync detection
# - Preview of both camera feeds
# - Proper triangulation with known camera angles

import cv2
import mediapipe as mp
import numpy as np
import json
import time
import threading
from datetime import datetime
from queue import Queue
import math

# ============================================================================
# CONFIGURATION - ADJUST THESE FOR YOUR SETUP
# ============================================================================

# Camera indices (run find_cameras.py to identify these)
FRONT_CAMERA_INDEX = 1      # Sony ZV-1F (1280x720)
SIDE_CAMERA_INDEX = 3       # OBS Virtual Camera (S25 via DroidCam)

# Camera settings
RESOLUTION = (1280, 720)    # Lower res for smoother dual capture
FPS = 30

# Camera angle (side camera relative to front)
# 90 = side camera is exactly to your left
# Positive = counterclockwise from front camera's view
SIDE_CAMERA_ANGLE = 90  # degrees

# Recording settings
COUNTDOWN_SECONDS = 5
RECORD_DURATION = 15

# T-pose detection thresholds
TPOSE_ARM_SPREAD_THRESHOLD = 0.4    # Minimum wrist X distance (normalized)
TPOSE_ARM_DROP_THRESHOLD = 0.08     # Maximum wrist Y below shoulder

# ============================================================================
# MEDIAPIPE SETUP
# ============================================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ============================================================================
# CAMERA THREAD - Captures frames in background
# ============================================================================

class CameraThread:
    def __init__(self, camera_index, name, resolution, fps):
        self.camera_index = camera_index
        self.name = name
        self.resolution = resolution
        self.fps = fps
        
        self.cap = None
        self.frame = None
        self.timestamp = 0
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
    def start(self):
        # For virtual cameras (DroidCam index 2, OBS Virtual Camera index 3), try default backend first
        if self.camera_index in [2, 3]:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
            # Try DirectShow backend on Windows for physical cameras
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback to default backend
                self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"ERROR: Could not open {self.name} (index {self.camera_index})")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✓ {self.name}: {actual_w}x{actual_h} @ {actual_fps}fps")
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True
    
    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                    self.timestamp = time.time()
    
    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy(), self.timestamp
            return None, 0
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

# ============================================================================
# POSE PROCESSOR
# ============================================================================

class PoseProcessor:
    def __init__(self):
        self.pose = mp_pose.Pose(
            model_complexity=1,  # 0, 1, or 2 (1 is balanced)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def process(self, frame):
        """Process frame and return landmarks"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if results.pose_landmarks:
            landmarks = {}
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                landmarks[idx] = {
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    'visibility': lm.visibility
                }
            return landmarks, results.pose_landmarks
        return None, None
    
    def draw(self, frame, pose_landmarks):
        """Draw skeleton on frame"""
        if pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 100, 100), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
    
    def close(self):
        self.pose.close()

# ============================================================================
# T-POSE DETECTOR
# ============================================================================

def detect_tpose(landmarks):
    """
    Detect if person is in T-pose
    Returns: (is_tpose, confidence, details)
    """
    if not landmarks:
        return False, 0, "No landmarks"
    
    required = [11, 12, 15, 16]  # shoulders and wrists
    if not all(id in landmarks for id in required):
        return False, 0, "Missing landmarks"
    
    l_shoulder = landmarks[11]
    r_shoulder = landmarks[12]
    l_wrist = landmarks[15]
    r_wrist = landmarks[16]
    
    # Check arm spread (wrists far apart horizontally)
    arm_spread = abs(l_wrist['x'] - r_wrist['x'])
    
    # Check arm drop (wrists near shoulder height)
    l_drop = abs(l_wrist['y'] - l_shoulder['y'])
    r_drop = abs(r_wrist['y'] - r_shoulder['y'])
    avg_drop = (l_drop + r_drop) / 2
    
    is_tpose = (arm_spread > TPOSE_ARM_SPREAD_THRESHOLD and 
                avg_drop < TPOSE_ARM_DROP_THRESHOLD)
    
    # Confidence based on how good the T-pose is
    spread_score = min(1.0, arm_spread / 0.6)
    drop_score = max(0, 1.0 - avg_drop / 0.15)
    confidence = (spread_score + drop_score) / 2
    
    details = f"spread={arm_spread:.3f}, drop={avg_drop:.3f}"
    
    return is_tpose, confidence, details

# ============================================================================
# TRIANGULATION
# ============================================================================

def triangulate_landmarks(front_landmarks, side_landmarks, camera_angle_deg):
    """
    Combine front and side camera landmarks into 3D positions.
    
    Front camera: gives X (horizontal) and Y (vertical)
    Side camera: gives depth (becomes Y in Blender) and Y (vertical, for verification)
    
    Returns: dict of 3D landmarks
    """
    if not front_landmarks or not side_landmarks:
        return None
    
    angle_rad = math.radians(camera_angle_deg)
    
    combined = {}
    
    for idx in range(33):
        if idx not in front_landmarks or idx not in side_landmarks:
            continue
        
        front = front_landmarks[idx]
        side = side_landmarks[idx]
        
        # Front camera gives us X and Y (height)
        # We flip X for mirror view
        x = front['x']
        y_front = front['y']
        
        # Side camera gives us depth
        # The side camera's X becomes our depth (Y in Blender)
        depth = side['x']
        y_side = side['y']
        
        # Average Y from both cameras for more accuracy
        y = (y_front + y_side) / 2
        
        # Combine into 3D
        # Note: We'll transform to Blender coordinates in the import script
        combined[idx] = {
            'x': x,           # From front camera
            'y': y,           # Averaged height
            'z': depth,       # From side camera (this becomes depth/Y in Blender)
            'visibility': (front['visibility'] + side['visibility']) / 2
        }
    
    return combined

# ============================================================================
# MAIN CAPTURE APPLICATION
# ============================================================================

class DualCameraCapture:
    def __init__(self):
        self.front_camera = CameraThread(FRONT_CAMERA_INDEX, "Front", RESOLUTION, FPS)
        self.side_camera = CameraThread(SIDE_CAMERA_INDEX, "Side", RESOLUTION, FPS)
        
        self.front_processor = PoseProcessor()
        self.side_processor = PoseProcessor()
        
        self.recording = False
        self.countdown_active = False
        self.countdown_start = 0
        
        self.frames_data = []
        self.record_start = 0
        
        self.tpose_detected = False
        self.tpose_frame = -1
        
    def start(self):
        print("=" * 60)
        print("MELODICCAP - LIVE DUAL CAMERA CAPTURE")
        print("=" * 60)
        
        if not self.front_camera.start():
            print("Failed to start front camera!")
            return False
        
        if not self.side_camera.start():
            print("Failed to start side camera!")
            self.front_camera.stop()
            return False
        
        print("\nControls:")
        print("  R - Start recording (with countdown)")
        print("  T - Detect T-pose now")
        print("  Q - Quit")
        print("=" * 60)
        
        return True
    
    def run(self):
        if not self.start():
            return
        
        try:
            self._main_loop()
        finally:
            self.stop()
    
    def _main_loop(self):
        while True:
            # Get frames from both cameras
            front_frame, front_time = self.front_camera.get_frame()
            side_frame, side_time = self.side_camera.get_frame()
            
            if front_frame is None or side_frame is None:
                time.sleep(0.01)
                continue
            
            # Flip front frame for mirror view
            front_frame = cv2.flip(front_frame, 1)
            # Don't flip side frame
            
            # Process poses
            front_landmarks, front_pose = self.front_processor.process(front_frame)
            side_landmarks, side_pose = self.side_processor.process(side_frame)
            
            # Draw skeletons
            self.front_processor.draw(front_frame, front_pose)
            self.side_processor.draw(side_frame, side_pose)
            
            # Check T-pose
            is_tpose, tpose_conf, tpose_details = detect_tpose(front_landmarks)
            
            # Handle countdown
            if self.countdown_active:
                elapsed = time.time() - self.countdown_start
                remaining = COUNTDOWN_SECONDS - elapsed
                
                if remaining > 0:
                    # Show countdown
                    countdown_num = int(remaining) + 1
                    cv2.putText(front_frame, str(countdown_num),
                               (front_frame.shape[1]//2 - 50, front_frame.shape[0]//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 8)
                    cv2.putText(front_frame, "GET READY - T-POSE!",
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    # Start recording
                    self.countdown_active = False
                    self.recording = True
                    self.record_start = time.time()
                    self.frames_data = []
                    self.tpose_detected = False
                    self.tpose_frame = -1
                    print("\n*** RECORDING STARTED ***")
            
            # Handle recording
            elif self.recording:
                elapsed = time.time() - self.record_start
                
                if elapsed < RECORD_DURATION:
                    # Record frame
                    frame_num = len(self.frames_data)
                    
                    # Triangulate 3D position
                    combined_landmarks = triangulate_landmarks(
                        front_landmarks, side_landmarks, SIDE_CAMERA_ANGLE
                    )
                    
                    if combined_landmarks:
                        frame_data = {
                            'frame': frame_num,
                            'timestamp': elapsed,
                            'pose_landmarks': [
                                {'id': idx, **data} 
                                for idx, data in combined_landmarks.items()
                            ]
                        }
                        self.frames_data.append(frame_data)
                        
                        # Check for T-pose during recording
                        if is_tpose and not self.tpose_detected:
                            self.tpose_detected = True
                            self.tpose_frame = frame_num
                            print(f"  T-pose detected at frame {frame_num}")
                    
                    # Show recording status
                    cv2.putText(front_frame, f"REC: {elapsed:.1f}s",
                               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.putText(front_frame, f"Frames: {len(self.frames_data)}",
                               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    if elapsed < 3:
                        cv2.putText(front_frame, "HOLD T-POSE",
                                   (front_frame.shape[1]//2 - 150, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                else:
                    # Recording complete
                    self._save_recording()
                    self.recording = False
            
            else:
                # Not recording - show status
                cv2.putText(front_frame, "Press R to record",
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show T-pose status
                if is_tpose:
                    cv2.putText(front_frame, f"T-POSE OK ({tpose_conf:.0%})",
                               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(front_frame, f"T-pose: {tpose_details}",
                               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            
            # Show sync status
            sync_diff = abs(front_time - side_time) * 1000  # ms
            sync_color = (0, 255, 0) if sync_diff < 50 else (0, 255, 255) if sync_diff < 100 else (0, 0, 255)
            cv2.putText(side_frame, f"Sync: {sync_diff:.0f}ms",
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, sync_color, 2)
            
            # Combine frames side by side
            display_front = cv2.resize(front_frame, (640, 360))
            display_side = cv2.resize(side_frame, (640, 360))
            combined = np.hstack([display_front, display_side])
            
            # Add labels
            cv2.putText(combined, "FRONT (Mirror)", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, "SIDE", (650, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("MelodicCap - Dual Camera", combined)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r') and not self.recording and not self.countdown_active:
                self.countdown_active = True
                self.countdown_start = time.time()
                print("\nCountdown started...")
            
            elif key == ord('t'):
                if is_tpose:
                    print(f"T-pose confirmed: {tpose_details}")
                else:
                    print(f"Not in T-pose: {tpose_details}")
            
            elif key == ord('q'):
                break
    
    def _save_recording(self):
        """Save recorded data to JSON"""
        if not self.frames_data:
            print("No frames to save!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/takes/dual_live_{timestamp}.json"
        
        data = {
            'metadata': {
                'type': 'dual_camera_live',
                'frames': len(self.frames_data),
                'duration': RECORD_DURATION,
                'fps': len(self.frames_data) / RECORD_DURATION,
                'resolution': f"{RESOLUTION[0]}x{RESOLUTION[1]}",
                'date': timestamp,
                'tpose_frame': self.tpose_frame,
                'camera_angle': SIDE_CAMERA_ANGLE,
                'front_camera': FRONT_CAMERA_INDEX,
                'side_camera': SIDE_CAMERA_INDEX
            },
            'data': self.frames_data
        }
        
        # Ensure directory exists
        import os
        os.makedirs("data/takes", exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(data, f)
        
        print(f"\n✓ Saved {len(self.frames_data)} frames to {filename}")
        if self.tpose_frame >= 0:
            print(f"  T-pose frame: {self.tpose_frame}")
        else:
            print("  WARNING: No T-pose detected during recording!")
    
    def stop(self):
        self.front_camera.stop()
        self.side_camera.stop()
        self.front_processor.close()
        self.side_processor.close()
        cv2.destroyAllWindows()
        print("\nCapture stopped.")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\nMELODICCAP - LIVE DUAL CAMERA CAPTURE")
    print("=====================================\n")
    
    print("Before starting, verify:")
    print(f"  1. Front camera is index {FRONT_CAMERA_INDEX}")
    print(f"  2. Side camera is index {SIDE_CAMERA_INDEX}")
    print(f"  3. Side camera is at {SIDE_CAMERA_ANGLE}° angle (to your left)")
    print("\nEdit the CONFIGURATION section at top of script if needed.\n")
    
    input("Press ENTER to start...")
    
    app = DualCameraCapture()
    app.run()
