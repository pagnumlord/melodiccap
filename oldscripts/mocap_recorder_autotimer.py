"""
Dual-Camera Motion Capture Recorder with AUTO-TIMER
Press SPACE to start, automatically stops after set duration

New features:
- 3-2-1 countdown before recording starts
- Auto-stops after duration (default 15 seconds)
- Shows timer during recording

Usage:
    python mocap_recorder_autotimer.py --camera_0 3 --camera_1 4 --duration 15
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
import time


class AutoTimerMocapRecorder:
    """Records motion capture with automatic stop after duration"""
    
    def __init__(self, output_dir="./mocap_takes", camera_ids=[0, 1], duration=15):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Camera setup
        self.camera_ids = camera_ids
        self.cameras = {}
        self.pose_detectors = {}
        
        # Recording state
        self.is_recording = False
        self.is_countdown = False
        self.countdown_start = None
        self.record_start = None
        self.record_duration = duration  # seconds
        self.current_take = None
        self.frame_data = []
        
        # Camera calibration
        self.calibration = {
            'camera_0': {
                'position': np.array([0.0, 0.0, 0.0]),
                'rotation': np.array([0.0, 0.0, 0.0]),
                'distance_between_cameras': 1.0
            },
            'camera_1': {
                'position': np.array([1.0, 0.0, 0.0]),
                'rotation': np.array([0.0, 0.0, 0.0]),
                'distance_between_cameras': 1.0
            }
        }
        
    def initialize_cameras(self):
        """Initialize both cameras and pose detectors"""
        print("Initializing cameras...")
        
        for cam_id in self.camera_ids:
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                print(f"Warning: Camera {cam_id} could not be opened!")
                continue
                
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.cameras[f'camera_{cam_id}'] = cap
            
            # Initialize pose detector
            self.pose_detectors[f'camera_{cam_id}'] = self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1,
                enable_segmentation=False
            )
            
        print(f"Initialized {len(self.cameras)} cameras")
        return len(self.cameras) > 0
        
    def triangulate_3d_point(self, point_cam0, point_cam1):
        """Triangulate 3D position from two 2D points"""
        if point_cam0[2] < 0.5 or point_cam1[2] < 0.5:
            return None
            
        baseline = self.calibration['camera_0']['distance_between_cameras']
        disparity = abs(point_cam0[0] - point_cam1[0])
        
        if disparity < 0.001:
            disparity = 0.001
            
        focal_length_estimate = 1000
        z = (focal_length_estimate * baseline) / (disparity * 1280)
        
        x = ((point_cam0[0] + point_cam1[0]) / 2.0 - 0.5) * z * 2.0
        y = ((point_cam0[1] + point_cam1[1]) / 2.0 - 0.5) * z * 2.0
        
        return (x, y, z)
        
    def process_frame(self, timestamp):
        """Process one frame from both cameras"""
        frame_dict = {
            'timestamp': timestamp,
            'camera_0': None,
            'camera_1': None,
            'landmarks_2d': {},
            'landmarks_3d': {}
        }
        
        images = {}
        results = {}
        
        # Capture and process from both cameras
        for cam_name, cap in self.cameras.items():
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Convert to RGB for MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Process with MediaPipe
            pose_results = self.pose_detectors[cam_name].process(image)
            
            # Convert back for display
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks
            if pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
            images[cam_name] = image
            results[cam_name] = pose_results
            
        # Extract 2D landmarks
        for cam_name, pose_results in results.items():
            if pose_results.pose_landmarks:
                landmarks_2d = []
                for landmark in pose_results.pose_landmarks.landmark:
                    landmarks_2d.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                frame_dict['landmarks_2d'][cam_name] = landmarks_2d
                
        # Triangulate 3D positions
        if 'camera_0' in frame_dict['landmarks_2d'] and 'camera_1' in frame_dict['landmarks_2d']:
            landmarks_3d = []
            lm_cam0 = frame_dict['landmarks_2d']['camera_0']
            lm_cam1 = frame_dict['landmarks_2d']['camera_1']
            
            for i in range(min(len(lm_cam0), len(lm_cam1))):
                point_cam0 = (lm_cam0[i]['x'], lm_cam0[i]['y'], lm_cam0[i]['visibility'])
                point_cam1 = (lm_cam1[i]['x'], lm_cam1[i]['y'], lm_cam1[i]['visibility'])
                
                pos_3d = self.triangulate_3d_point(point_cam0, point_cam1)
                
                if pos_3d:
                    landmarks_3d.append({
                        'index': i,
                        'x': pos_3d[0],
                        'y': pos_3d[1], 
                        'z': pos_3d[2],
                        'visibility': min(lm_cam0[i]['visibility'], lm_cam1[i]['visibility'])
                    })
                else:
                    landmarks_3d.append({
                        'index': i,
                        'x': lm_cam0[i]['x'],
                        'y': lm_cam0[i]['y'],
                        'z': lm_cam0[i]['z'],
                        'visibility': lm_cam0[i]['visibility']
                    })
                    
            frame_dict['landmarks_3d'] = landmarks_3d
            
        return images, frame_dict
        
    def start_countdown(self):
        """Start 3-2-1 countdown before recording"""
        self.is_countdown = True
        self.countdown_start = time.time()
        print("\nCountdown started! Get into A-pose...")
        
    def start_recording(self, take_name=None):
        """Start recording after countdown"""
        if take_name is None:
            take_name = f"take_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        self.current_take = take_name
        self.frame_data = []
        self.is_recording = True
        self.is_countdown = False
        self.record_start = time.time()
        
        print(f"\n🔴 RECORDING: {take_name}")
        print(f"   Duration: {self.record_duration} seconds")
        
    def stop_take(self):
        """Stop recording and save data"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        # Save the take
        take_file = self.output_dir / f"{self.current_take}.json"
        
        take_data = {
            'take_name': self.current_take,
            'timestamp': datetime.now().isoformat(),
            'num_frames': len(self.frame_data),
            'fps': 30,
            'duration': self.record_duration,
            'calibration': {
                'camera_0': {
                    'position': self.calibration['camera_0']['position'].tolist(),
                    'rotation': self.calibration['camera_0']['rotation'].tolist()
                },
                'camera_1': {
                    'position': self.calibration['camera_1']['position'].tolist(),
                    'rotation': self.calibration['camera_1']['rotation'].tolist()
                }
            },
            'frames': self.frame_data
        }
        
        with open(take_file, 'w') as f:
            json.dump(take_data, f, indent=2)
            
        print(f"\n✅ Take saved: {take_file}")
        print(f"   Total frames: {len(self.frame_data)}")
        print(f"   Actual duration: {len(self.frame_data) / 30.0:.2f} seconds")
        
    def run(self):
        """Main recording loop with auto-timer"""
        if not self.initialize_cameras():
            print("Failed to initialize cameras!")
            return
            
        print("\n" + "="*70)
        print("DUAL CAMERA MOTION CAPTURE - AUTO TIMER MODE")
        print("="*70)
        print(f"Recording Duration: {self.record_duration} seconds")
        print("\nControls:")
        print("  SPACE - Start countdown (3-2-1 then auto-record)")
        print("  Q     - Quit")
        print("\nWorkflow:")
        print("  1. Get into A-pose")
        print("  2. Press SPACE")
        print("  3. Countdown: 3... 2... 1...")
        print("  4. Recording starts automatically")
        print(f"  5. After {self.record_duration} seconds, stops & saves automatically")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        try:
            while True:
                current_time = time.time() - start_time
                
                # Process frames
                images, frame_dict = self.process_frame(current_time)
                
                # Check countdown
                if self.is_countdown:
                    elapsed = time.time() - self.countdown_start
                    remaining = 3 - elapsed
                    
                    if remaining <= 0:
                        # Start recording!
                        self.start_recording()
                    
                # Check auto-stop
                if self.is_recording:
                    elapsed = time.time() - self.record_start
                    remaining = self.record_duration - elapsed
                    
                    # Store frame data
                    self.frame_data.append(frame_dict)
                    
                    if remaining <= 0:
                        # Auto-stop!
                        self.stop_take()
                        print("\n⏹️  Recording complete!")
                        print("   Press SPACE for another take, or Q to quit\n")
                    
                # Display feeds
                if images:
                    display_images = []
                    for cam_name in sorted(images.keys()):
                        img = images[cam_name]
                        
                        # Add countdown overlay
                        if self.is_countdown:
                            countdown_time = 3 - (time.time() - self.countdown_start)
                            if countdown_time > 0:
                                countdown_text = str(int(countdown_time) + 1)
                                cv2.putText(img, countdown_text, (img.shape[1]//2 - 50, img.shape[0]//2), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10)
                        
                        # Add recording indicator and timer
                        if self.is_recording:
                            elapsed = time.time() - self.record_start
                            remaining = self.record_duration - elapsed
                            
                            # Red dot
                            cv2.circle(img, (30, 30), 20, (0, 0, 255), -1)
                            
                            # Timer
                            timer_text = f"REC {remaining:.1f}s"
                            cv2.putText(img, timer_text, (60, 40), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                            
                            # Progress bar
                            progress = elapsed / self.record_duration
                            bar_width = img.shape[1] - 40
                            bar_height = 20
                            bar_x = 20
                            bar_y = 60
                            
                            cv2.rectangle(img, (bar_x, bar_y), 
                                        (bar_x + bar_width, bar_y + bar_height),
                                        (100, 100, 100), -1)
                            cv2.rectangle(img, (bar_x, bar_y), 
                                        (bar_x + int(bar_width * progress), bar_y + bar_height),
                                        (0, 255, 0), -1)
                        
                        display_images.append(img)
                    
                    # Stack images with resolution handling
                    if len(display_images) == 2:
                        target_height = min(img.shape[0] for img in display_images)
                        resized_images = []
                        
                        for img in display_images:
                            if img.shape[0] != target_height:
                                aspect_ratio = img.shape[1] / img.shape[0]
                                new_width = int(target_height * aspect_ratio)
                                img = cv2.resize(img, (new_width, target_height))
                            resized_images.append(img)
                        
                        combined = np.hstack(resized_images)
                    else:
                        combined = display_images[0]
                        
                    cv2.imshow('Dual Camera Mocap - Auto Timer', combined)
                    
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    if self.is_recording:
                        self.stop_take()
                    break
                elif key == ord(' '):
                    if not self.is_recording and not self.is_countdown:
                        self.start_countdown()
                    
        finally:
            # Cleanup
            for cap in self.cameras.values():
                cap.release()
            cv2.destroyAllWindows()
            
            for detector in self.pose_detectors.values():
                detector.close()
                

def main():
    parser = argparse.ArgumentParser(description='Dual Camera Motion Capture with Auto-Timer')
    parser.add_argument('--output_dir', type=str, default='./mocap_takes',
                       help='Directory to save recorded takes')
    parser.add_argument('--camera_0', type=int, default=0,
                       help='Device ID for first camera')
    parser.add_argument('--camera_1', type=int, default=1,
                       help='Device ID for second camera')
    parser.add_argument('--duration', type=int, default=15,
                       help='Recording duration in seconds (default: 15)')
    
    args = parser.parse_args()
    
    recorder = AutoTimerMocapRecorder(
        output_dir=args.output_dir,
        camera_ids=[args.camera_0, args.camera_1],
        duration=args.duration
    )
    
    recorder.run()
    

if __name__ == "__main__":
    main()
