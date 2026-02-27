"""
Dual-Camera Motion Capture Recorder for JaxRigify
Uses MediaPipe Pose for body tracking from two camera angles

Requirements:
    pip install opencv-python mediapipe numpy --break-system-packages

Usage:
    python mocap_recorder.py --output_dir ./mocap_takes
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
import time


class DualCameraMocapRecorder:
    """Records motion capture from two cameras with synchronized timestamping"""
    
    def __init__(self, output_dir="./mocap_takes", camera_ids=[0, 1]):
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
        self.current_take = None
        self.frame_data = []
        
        # Camera calibration data (will be computed during calibration)
        self.calibration = {
            'camera_0': {
                'position': np.array([0.0, 0.0, 0.0]),  # Left camera position
                'rotation': np.array([0.0, 0.0, 0.0]),  # Euler angles
                'distance_between_cameras': 1.0  # meters
            },
            'camera_1': {
                'position': np.array([1.0, 0.0, 0.0]),  # Right camera position  
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
            
            # Initialize pose detector for this camera
            self.pose_detectors[f'camera_{cam_id}'] = self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1,
                enable_segmentation=False
            )
            
        print(f"Initialized {len(self.cameras)} cameras")
        return len(self.cameras) > 0
        
    def triangulate_3d_point(self, point_cam0, point_cam1):
        """
        Triangulate 3D position from two 2D points
        This is a simplified triangulation - for production use OpenCV's triangulatePoints
        
        Args:
            point_cam0: (x, y, visibility) from camera 0
            point_cam1: (x, y, visibility) from camera 1
            
        Returns:
            (x, y, z) 3D coordinates
        """
        if point_cam0[2] < 0.5 or point_cam1[2] < 0.5:  # Low visibility
            return None
            
        # Simple triangulation assuming cameras are parallel and distance apart
        # For better results, implement proper camera calibration and triangulation
        baseline = self.calibration['camera_0']['distance_between_cameras']
        
        # Simplified depth estimation (disparity-based)
        disparity = abs(point_cam0[0] - point_cam1[0])
        
        if disparity < 0.001:  # Avoid division by zero
            disparity = 0.001
            
        # Estimate depth (Z coordinate)
        focal_length_estimate = 1000  # pixels (rough estimate)
        z = (focal_length_estimate * baseline) / (disparity * 1280)  # normalized to image width
        
        # Calculate X and Y in 3D space
        # Average the 2D positions and project to 3D
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
                        'z': landmark.z,  # MediaPipe's depth estimate
                        'visibility': landmark.visibility
                    })
                frame_dict['landmarks_2d'][cam_name] = landmarks_2d
                
        # Triangulate 3D positions if we have data from both cameras
        if 'camera_0' in frame_dict['landmarks_2d'] and 'camera_1' in frame_dict['landmarks_2d']:
            landmarks_3d = []
            lm_cam0 = frame_dict['landmarks_2d']['camera_0']
            lm_cam1 = frame_dict['landmarks_2d']['camera_1']
            
            # MediaPipe has 33 pose landmarks
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
                    # Use camera_0 data as fallback
                    landmarks_3d.append({
                        'index': i,
                        'x': lm_cam0[i]['x'],
                        'y': lm_cam0[i]['y'],
                        'z': lm_cam0[i]['z'],
                        'visibility': lm_cam0[i]['visibility']
                    })
                    
            frame_dict['landmarks_3d'] = landmarks_3d
            
        return images, frame_dict
        
    def start_take(self, take_name=None):
        """Start recording a new take"""
        if take_name is None:
            take_name = f"take_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        self.current_take = take_name
        self.frame_data = []
        self.is_recording = True
        
        print(f"Recording started: {take_name}")
        
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
            'fps': 30,  # Nominal FPS
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
            
        print(f"Take saved: {take_file}")
        print(f"Total frames: {len(self.frame_data)}")
        
        # Also export to BVH format for Blender
        self.export_to_bvh(take_file.with_suffix('.bvh'))
        
    def export_to_bvh(self, output_path):
        """
        Export recorded data to BVH format for Blender import
        BVH is a standard motion capture format that Blender can import directly
        """
        print(f"Exporting to BVH: {output_path}")
        
        if not self.frame_data:
            print("No frame data to export!")
            return
            
        # MediaPipe landmark indices mapping to skeleton
        # We'll create a simplified skeleton structure
        landmark_names = [
            "Hips",           # 24 (midpoint of hips)
            "Spine",          # 23/24 average
            "Chest",          # 11/12 average  
            "Neck",           # 11/12 midpoint higher
            "Head",           # 0 (nose)
            "LeftShoulder",   # 11
            "LeftArm",        # 13
            "LeftForeArm",    # 15
            "LeftHand",       # 17
            "RightShoulder",  # 12
            "RightArm",       # 14
            "RightForeArm",   # 16
            "RightHand",      # 18
            "LeftUpLeg",      # 23
            "LeftLeg",        # 25
            "LeftFoot",       # 27
            "RightUpLeg",     # 24
            "RightLeg",       # 26
            "RightFoot"       # 28
        ]
        
        with open(output_path, 'w') as f:
            # Write BVH header
            f.write("HIERARCHY\n")
            f.write("ROOT Hips\n")
            f.write("{\n")
            f.write("    OFFSET 0.0 0.0 0.0\n")
            f.write("    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")
            
            # Simplified skeleton hierarchy
            # In production, build full hierarchy
            
            f.write("    JOINT Spine\n")
            f.write("    {\n")
            f.write("        OFFSET 0.0 0.1 0.0\n")
            f.write("        CHANNELS 3 Zrotation Xrotation Yrotation\n")
            f.write("        JOINT Chest\n")
            f.write("        {\n")
            f.write("            OFFSET 0.0 0.15 0.0\n")
            f.write("            CHANNELS 3 Zrotation Xrotation Yrotation\n")
            f.write("            End Site\n")
            f.write("            {\n")
            f.write("                OFFSET 0.0 0.1 0.0\n")
            f.write("            }\n")
            f.write("        }\n")
            f.write("    }\n")
            f.write("}\n")
            
            # Write motion data
            f.write(f"MOTION\n")
            f.write(f"Frames: {len(self.frame_data)}\n")
            f.write(f"Frame Time: {1.0/30.0:.6f}\n")
            
            # Write frame data (simplified - just positions for now)
            for frame in self.frame_data:
                if 'landmarks_3d' in frame and frame['landmarks_3d']:
                    # Get hip position (approximate as landmark 24)
                    hips = frame['landmarks_3d'][24] if len(frame['landmarks_3d']) > 24 else frame['landmarks_3d'][0]
                    f.write(f"{hips['x']:.6f} {hips['y']:.6f} {hips['z']:.6f} ")
                    f.write("0.0 0.0 0.0 ")  # Rotations (would need to calculate)
                    f.write("0.0 0.0 0.0 ")  # Spine rotations
                    f.write("0.0 0.0 0.0")   # Chest rotations
                    f.write("\n")
                else:
                    # Write zeros if no data
                    f.write("0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n")
                    
        print("BVH export complete")
        
    def run(self):
        """Main recording loop"""
        if not self.initialize_cameras():
            print("Failed to initialize cameras!")
            return
            
        print("\n=== Dual Camera Motion Capture Recorder ===")
        print("Controls:")
        print("  SPACE - Start/Stop recording")
        print("  Q     - Quit")
        print("  C     - Calibrate cameras (TODO)")
        print("\n")
        
        start_time = time.time()
        
        try:
            while True:
                current_time = time.time() - start_time
                
                # Process frames from both cameras
                images, frame_dict = self.process_frame(current_time)
                
                # Store frame data if recording
                if self.is_recording:
                    self.frame_data.append(frame_dict)
                    
                # Display both camera feeds
                if images:
                    # Stack images horizontally for display
                    display_images = []
                    for cam_name in sorted(images.keys()):
                        img = images[cam_name]
                        
                        # Add recording indicator
                        if self.is_recording:
                            cv2.circle(img, (30, 30), 20, (0, 0, 255), -1)
                            cv2.putText(img, "REC", (60, 40), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                        display_images.append(img)
                    
                    if len(display_images) == 2:
                        # Resize images to same height for display (handles different camera resolutions)
                        target_height = min(img.shape[0] for img in display_images)
                        resized_images = []
                        
                        for img in display_images:
                            if img.shape[0] != target_height:
                                # Calculate new width to maintain aspect ratio
                                aspect_ratio = img.shape[1] / img.shape[0]
                                new_width = int(target_height * aspect_ratio)
                                img = cv2.resize(img, (new_width, target_height))
                            resized_images.append(img)
                        
                        combined = np.hstack(resized_images)
                    else:
                        combined = display_images[0]
                        
                    cv2.imshow('Dual Camera Mocap', combined)
                    
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    if self.is_recording:
                        self.stop_take()
                    break
                elif key == ord(' '):  # Space bar
                    if self.is_recording:
                        self.stop_take()
                    else:
                        self.start_take()
                elif key == ord('c'):
                    print("Camera calibration not yet implemented")
                    
        finally:
            # Cleanup
            for cap in self.cameras.values():
                cap.release()
            cv2.destroyAllWindows()
            
            for detector in self.pose_detectors.values():
                detector.close()
                

def main():
    parser = argparse.ArgumentParser(description='Dual Camera Motion Capture Recorder')
    parser.add_argument('--output_dir', type=str, default='./mocap_takes',
                       help='Directory to save recorded takes')
    parser.add_argument('--camera_0', type=int, default=0,
                       help='Device ID for first camera')
    parser.add_argument('--camera_1', type=int, default=1,
                       help='Device ID for second camera')
    
    args = parser.parse_args()
    
    recorder = DualCameraMocapRecorder(
        output_dir=args.output_dir,
        camera_ids=[args.camera_0, args.camera_1]
    )
    
    recorder.run()
    

if __name__ == "__main__":
    main()
