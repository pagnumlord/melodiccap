# MELODICCAP DUAL CAMERA LIVE CAPTURE V2
# Improved T-pose detection with visual feedback
# Shows arm span in real-time so you know when T-pose is good

import cv2
import mediapipe as mp
import numpy as np
import json
import time
from datetime import datetime
import threading
import queue

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

class DualCameraCapture:
    def __init__(self):
        # Camera indices - adjust if needed
        self.front_cam_index = 1  # Sony ZV-1F
        self.side_cam_index = 2   # Samsung S25 via DroidCam/OBS
        
        self.front_cap = None
        self.side_cap = None
        
        self.recording = False
        self.countdown_active = False
        self.frames_data = []
        
        self.record_duration = 15
        self.countdown_duration = 5
        
        # T-pose tracking
        self.best_tpose_frame = 0
        self.best_arm_span = 0
        self.current_arm_span = 0
        self.tpose_quality = 0  # 0-100%
        
        # Target arm span (normalized) for good T-pose
        # Arms fully extended = about 0.45 normalized width
        self.target_arm_span = 0.42
        
    def init_cameras(self):
        """Initialize both cameras"""
        print("Initializing cameras...")
        
        # Try to find cameras
        self.front_cap = cv2.VideoCapture(self.front_cam_index, cv2.CAP_DSHOW)
        if not self.front_cap.isOpened():
            print(f"  Front camera {self.front_cam_index} failed, trying alternatives...")
            for i in [0, 1, 2, 3]:
                self.front_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if self.front_cap.isOpened():
                    self.front_cam_index = i
                    print(f"  Found front camera at index {i}")
                    break
        
        self.side_cap = cv2.VideoCapture(self.side_cam_index, cv2.CAP_DSHOW)
        if not self.side_cap.isOpened():
            print(f"  Side camera {self.side_cam_index} failed, trying alternatives...")
            for i in [0, 1, 2, 3, 4]:
                if i == self.front_cam_index:
                    continue
                self.side_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if self.side_cap.isOpened():
                    self.side_cam_index = i
                    print(f"  Found side camera at index {i}")
                    break
        
        if not self.front_cap.isOpened() or not self.side_cap.isOpened():
            print("ERROR: Could not open both cameras!")
            return False
        
        # Set resolution
        for cap in [self.front_cap, self.side_cap]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
        
        f_w = int(self.front_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        f_h = int(self.front_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        s_w = int(self.side_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        s_h = int(self.side_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"  Front camera: {f_w}x{f_h}")
        print(f"  Side camera: {s_w}x{s_h}")
        
        return True
    
    def calculate_arm_span(self, landmarks):
        """Calculate normalized arm span from landmarks"""
        if not landmarks:
            return 0
        
        # Get wrist positions
        l_wrist = landmarks[15]
        r_wrist = landmarks[16]
        
        # Check visibility
        if l_wrist.visibility < 0.5 or r_wrist.visibility < 0.5:
            return 0
        
        # Calculate horizontal span
        span = abs(l_wrist.x - r_wrist.x)
        return span
    
    def calculate_tpose_quality(self, landmarks):
        """Calculate how close to perfect T-pose (0-100%)"""
        if not landmarks:
            return 0
        
        # Check arm positions
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        l_wrist = landmarks[15]
        r_wrist = landmarks[16]
        
        # Arms should be horizontal (wrist Y ≈ shoulder Y)
        l_arm_horizontal = 1 - min(1, abs(l_wrist.y - l_shoulder.y) * 5)
        r_arm_horizontal = 1 - min(1, abs(r_wrist.y - r_shoulder.y) * 5)
        
        # Arms should be extended (wrist X far from shoulder X)
        l_arm_extended = min(1, abs(l_wrist.x - l_shoulder.x) * 4)
        r_arm_extended = min(1, abs(r_wrist.x - r_shoulder.x) * 4)
        
        # Combine scores
        quality = (l_arm_horizontal + r_arm_horizontal + l_arm_extended + r_arm_extended) / 4
        return int(quality * 100)
    
    def draw_tpose_guide(self, frame, landmarks, quality):
        """Draw T-pose guide overlay"""
        h, w = frame.shape[:2]
        
        # Draw arm span bar at top
        bar_y = 30
        bar_width = 300
        bar_x = (w - bar_width) // 2
        
        # Background
        cv2.rectangle(frame, (bar_x - 5, bar_y - 20), (bar_x + bar_width + 5, bar_y + 25), (40, 40, 40), -1)
        
        # Target zone (green)
        target_start = int(bar_width * 0.85)
        cv2.rectangle(frame, (bar_x + target_start, bar_y), (bar_x + bar_width, bar_y + 15), (0, 100, 0), -1)
        
        # Current span indicator
        span_ratio = min(1, self.current_arm_span / self.target_arm_span)
        span_x = int(bar_x + bar_width * span_ratio)
        
        # Color based on quality
        if quality > 80:
            color = (0, 255, 0)  # Green
        elif quality > 50:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 100, 255)  # Orange
        
        cv2.rectangle(frame, (bar_x, bar_y), (span_x, bar_y + 15), color, -1)
        
        # Labels
        cv2.putText(frame, f"ARM SPAN: {quality}%", (bar_x, bar_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # T-pose instructions
        if quality < 80:
            cv2.putText(frame, "Extend arms horizontally!", (w//2 - 120, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "GOOD T-POSE!", (w//2 - 80, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame
    
    def run(self):
        """Main capture loop"""
        if not self.init_cameras():
            return
        
        # Initialize MediaPipe
        holistic_front = mp_holistic.Holistic(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        holistic_side = mp_holistic.Holistic(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("\n" + "=" * 60)
        print("MELODICCAP DUAL CAMERA CAPTURE V2")
        print("=" * 60)
        print("Controls:")
        print("  R - Start recording (5s countdown)")
        print("  Q - Quit")
        print("")
        print("T-POSE GUIDE:")
        print("  - Watch the ARM SPAN bar at top")
        print("  - Extend arms until bar reaches green zone")
        print("  - Best T-pose frame is auto-detected")
        print("=" * 60)
        
        start_time = None
        countdown_start = None
        
        while True:
            # Read both cameras synchronously
            ret_front, frame_front = self.front_cap.read()
            ret_side, frame_side = self.side_cap.read()
            
            if not ret_front or not ret_side:
                print("Camera read failed!")
                break
            
            # Flip front for mirror view
            frame_front = cv2.flip(frame_front, 1)
            
            # Process with MediaPipe
            rgb_front = cv2.cvtColor(frame_front, cv2.COLOR_BGR2RGB)
            rgb_side = cv2.cvtColor(frame_side, cv2.COLOR_BGR2RGB)
            
            results_front = holistic_front.process(rgb_front)
            results_side = holistic_side.process(rgb_side)
            
            # Draw pose on front frame
            if results_front.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame_front, results_front.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 100, 100), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
                
                # Calculate T-pose quality
                self.current_arm_span = self.calculate_arm_span(results_front.pose_landmarks.landmark)
                self.tpose_quality = self.calculate_tpose_quality(results_front.pose_landmarks.landmark)
            
            # Draw T-pose guide (always visible when not recording)
            if not self.recording:
                frame_front = self.draw_tpose_guide(frame_front, 
                    results_front.pose_landmarks.landmark if results_front.pose_landmarks else None,
                    self.tpose_quality)
            
            # Draw pose on side frame
            if results_side.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame_side, results_side.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(100, 255, 100), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
            
            # Countdown logic
            if self.countdown_active:
                elapsed = time.time() - countdown_start
                remaining = self.countdown_duration - elapsed
                
                if remaining > 0:
                    # Show countdown
                    cv2.putText(frame_front, str(int(remaining) + 1),
                                (frame_front.shape[1]//2 - 50, frame_front.shape[0]//2 + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10)
                    cv2.putText(frame_front, "GET INTO T-POSE!",
                                (frame_front.shape[1]//2 - 180, frame_front.shape[0]//2 - 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
                else:
                    # Start recording
                    self.countdown_active = False
                    self.recording = True
                    start_time = time.time()
                    self.frames_data = []
                    self.best_tpose_frame = 0
                    self.best_arm_span = 0
                    print("\n>>> RECORDING STARTED <<<")
            
            # Recording logic
            elif self.recording:
                elapsed = time.time() - start_time
                
                if elapsed < self.record_duration:
                    frame_num = len(self.frames_data)
                    
                    # Collect frame data
                    frame_data = {
                        'frame': frame_num,
                        'timestamp': elapsed,
                        'pose_landmarks': None
                    }
                    
                    if results_front.pose_landmarks and results_side.pose_landmarks:
                        combined_landmarks = []
                        
                        for i, (lm_f, lm_s) in enumerate(zip(
                            results_front.pose_landmarks.landmark,
                            results_side.pose_landmarks.landmark
                        )):
                            # Average Y (height) from both, X from front, Z (depth) from side's X
                            avg_y = (lm_f.y + lm_s.y) / 2
                            
                            combined_landmarks.append({
                                'id': i,
                                'x': lm_f.x,           # Horizontal from front
                                'y': avg_y,            # Height (averaged)
                                'z': lm_s.x,           # Depth from side camera's X
                                'visibility': min(lm_f.visibility, lm_s.visibility)
                            })
                        
                        frame_data['pose_landmarks'] = combined_landmarks
                        
                        # Track best T-pose frame (first 5 seconds)
                        if elapsed < 5:
                            current_span = self.calculate_arm_span(results_front.pose_landmarks.landmark)
                            if current_span > self.best_arm_span:
                                self.best_arm_span = current_span
                                self.best_tpose_frame = frame_num
                                print(f"  Better T-pose at frame {frame_num}: span={current_span:.3f}")
                    
                    self.frames_data.append(frame_data)
                    
                    # Recording UI
                    cv2.putText(frame_front, f"REC {elapsed:.1f}s",
                                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    
                    # T-pose reminder for first 3 seconds
                    if elapsed < 3:
                        cv2.putText(frame_front, "HOLD T-POSE!",
                                    (frame_front.shape[1]//2 - 120, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
                        cv2.putText(frame_front, f"Best span: {self.best_arm_span:.3f}",
                                    (frame_front.shape[1]//2 - 100, 140),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                else:
                    # Save recording
                    self.recording = False
                    self.save_recording()
            
            else:
                # Idle UI
                cv2.putText(frame_front, "Press R to record",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Combine views
            frame_side_resized = cv2.resize(frame_side, (frame_front.shape[1]//3, frame_front.shape[0]//3))
            frame_front[10:10+frame_side_resized.shape[0], -10-frame_side_resized.shape[1]:-10] = frame_side_resized
            
            # Show
            cv2.imshow("MelodicCap Dual Capture V2", frame_front)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r') and not self.recording and not self.countdown_active:
                self.countdown_active = True
                countdown_start = time.time()
                print("\nCountdown started - get into T-pose!")
            elif key == ord('q'):
                break
        
        # Cleanup
        self.front_cap.release()
        self.side_cap.release()
        cv2.destroyAllWindows()
        holistic_front.close()
        holistic_side.close()
    
    def save_recording(self):
        """Save recording to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/takes/dual_live_{timestamp}.json"
        
        # Calculate actual FPS
        if len(self.frames_data) > 1:
            actual_duration = self.frames_data[-1]['timestamp']
            actual_fps = len(self.frames_data) / actual_duration if actual_duration > 0 else 30
        else:
            actual_fps = 30
        
        data = {
            'metadata': {
                'type': 'dual_camera_live',
                'version': 2,
                'frames': len(self.frames_data),
                'duration': self.record_duration,
                'fps': actual_fps,
                'date': timestamp,
                'tpose_frame': self.best_tpose_frame,
                'tpose_arm_span': self.best_arm_span,
                'cameras': {
                    'front': self.front_cam_index,
                    'side': self.side_cam_index
                }
            },
            'data': self.frames_data
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f)
        
        print(f"\n{'='*60}")
        print(f"Recording saved!")
        print(f"  File: {filename}")
        print(f"  Frames: {len(self.frames_data)}")
        print(f"  FPS: {actual_fps:.1f}")
        print(f"  Best T-pose: frame {self.best_tpose_frame} (span: {self.best_arm_span:.3f})")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    import os
    os.makedirs("data/takes", exist_ok=True)
    
    capture = DualCameraCapture()
    capture.run()
