#!/usr/bin/env python3
"""
MelodicCap Dual Camera Capture - FIXED VERSION
Properly triangulates 3D from two 90° cameras

Front camera (Sony ZV-1F) → X and Z (height)
Side camera (Samsung S25)  → Y (depth) 

Usage:
    python dual_camera_capture_fixed.py front.mp4 side.mp4 output_name
"""

import cv2
import mediapipe as mp
import json
import sys
import numpy as np
from datetime import datetime
from pathlib import Path

mp_holistic = mp.solutions.holistic

def find_sync_frame(video_path, max_frames=300):
    """Find the frame where hands clap (biggest motion spike)"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    prev_frame = None
    motion_scores = []
    
    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray)
            motion = np.mean(diff)
            motion_scores.append((frame_count, motion))
        
        prev_frame = gray
        frame_count += 1
    
    cap.release()
    
    if not motion_scores:
        return 0, fps
    
    # Find the biggest motion spike (likely the clap)
    best_frame, best_score = max(motion_scores, key=lambda x: x[1])
    print(f"  Video: {video_path}")
    print(f"  FPS: {fps}")
    print(f"  Sync frame: {best_frame} (motion score: {best_score:.2f})")
    
    return best_frame, fps


def extract_landmarks(video_path, start_frame=0):
    """Extract MediaPipe holistic landmarks from video starting at sync frame"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Skip to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    holistic = mp_holistic.Holistic(
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=True
    )
    
    frames_data = []
    frame_num = 0
    
    print(f"  Processing from frame {start_frame}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)
        
        frame_data = {
            'frame': frame_num,
            'pose_landmarks': None,
            'face_landmarks': None,
            'left_hand_landmarks': None,
            'right_hand_landmarks': None
        }
        
        if results.pose_landmarks:
            frame_data['pose_landmarks'] = [
                {'id': idx, 'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
                for idx, lm in enumerate(results.pose_landmarks.landmark)
            ]
        
        if results.face_landmarks:
            frame_data['face_landmarks'] = [
                {'id': idx, 'x': lm.x, 'y': lm.y, 'z': lm.z}
                for idx, lm in enumerate(results.face_landmarks.landmark)
            ]
        
        if results.left_hand_landmarks:
            frame_data['left_hand_landmarks'] = [
                {'id': idx, 'x': lm.x, 'y': lm.y, 'z': lm.z}
                for idx, lm in enumerate(results.left_hand_landmarks.landmark)
            ]
        
        if results.right_hand_landmarks:
            frame_data['right_hand_landmarks'] = [
                {'id': idx, 'x': lm.x, 'y': lm.y, 'z': lm.z}
                for idx, lm in enumerate(results.right_hand_landmarks.landmark)
            ]
        
        frames_data.append(frame_data)
        frame_num += 1
        
        if frame_num % 30 == 0:
            print(f"    Frame {frame_num}...")
    
    cap.release()
    holistic.close()
    
    print(f"  Extracted {len(frames_data)} frames at {fps} fps")
    return frames_data, fps


def triangulate_3d(front_data, side_data):
    """
    Combine front and side camera data into true 3D coordinates.
    
    Front camera (facing you) provides:
        - X position (horizontal, left-right)
        - Y position (vertical, used for HEIGHT in Blender Z)
    
    Side camera (90° from front) provides:
        - X position (which is actually DEPTH from front camera's perspective)
        - Y position (vertical, can verify height)
    
    Output coordinate system (for Blender):
        - x: left-right (from front camera X)
        - y: forward-back / depth (from side camera X)
        - z: up-down / height (from front camera Y, flipped)
    """
    combined_data = []
    
    # Use the shorter of the two
    num_frames = min(len(front_data), len(side_data))
    
    for i in range(num_frames):
        front_frame = front_data[i]
        side_frame = side_data[i]
        
        frame_data = {
            'frame': i,
            'timestamp': i / 30.0,  # Approximate
            'pose_landmarks': None,
            'face_landmarks': None,
            'left_hand_landmarks': None,
            'right_hand_landmarks': None
        }
        
        # Triangulate pose landmarks
        if front_frame.get('pose_landmarks') and side_frame.get('pose_landmarks'):
            front_pose = {lm['id']: lm for lm in front_frame['pose_landmarks']}
            side_pose = {lm['id']: lm for lm in side_frame['pose_landmarks']}
            
            triangulated_pose = []
            for id in range(33):
                if id in front_pose and id in side_pose:
                    front_lm = front_pose[id]
                    side_lm = side_pose[id]
                    
                    # CORRECT TRIANGULATION:
                    # X from front camera (horizontal position)
                    # Y (depth) from side camera's X (side camera's horizontal IS our depth)
                    # Z (height) from front camera's Y (we'll flip this in Blender import)
                    
                    triangulated_pose.append({
                        'id': id,
                        'x': front_lm['x'],                    # Horizontal from front
                        'y': front_lm['y'],                    # Height from front (top=0, bottom=1)
                        'z': 1.0 - side_lm['x'],               # Depth from side (flip so closer=smaller)
                        'visibility': min(front_lm.get('visibility', 1), 
                                         side_lm.get('visibility', 1))
                    })
            
            if triangulated_pose:
                frame_data['pose_landmarks'] = triangulated_pose
        
        # For face and hands, just use front camera data (more reliable for these)
        if front_frame.get('face_landmarks'):
            frame_data['face_landmarks'] = front_frame['face_landmarks']
        
        if front_frame.get('left_hand_landmarks'):
            frame_data['left_hand_landmarks'] = front_frame['left_hand_landmarks']
        
        if front_frame.get('right_hand_landmarks'):
            frame_data['right_hand_landmarks'] = front_frame['right_hand_landmarks']
        
        combined_data.append(frame_data)
    
    return combined_data


def main():
    if len(sys.argv) < 4:
        print("Usage: python dual_camera_capture_fixed.py front.mp4 side.mp4 output_name")
        print("\nExample:")
        print("  python dual_camera_capture_fixed.py front.mp4 side.mp4 take01")
        sys.exit(1)
    
    front_video = Path(sys.argv[1])
    side_video = Path(sys.argv[2])
    output_name = sys.argv[3]
    
    if not front_video.exists():
        print(f"ERROR: Front video not found: {front_video}")
        sys.exit(1)
    
    if not side_video.exists():
        print(f"ERROR: Side video not found: {side_video}")
        sys.exit(1)
    
    print("=" * 70)
    print("MELODICCAP DUAL CAMERA PROCESSING (FIXED)")
    print("=" * 70)
    
    # Step 1: Find sync points
    print("\n[1/4] Finding sync points...")
    front_sync, front_fps = find_sync_frame(front_video)
    side_sync, side_fps = find_sync_frame(side_video)
    
    # Check framerate match
    if abs(front_fps - side_fps) > 1:
        print(f"\n⚠️  WARNING: Framerate mismatch!")
        print(f"    Front: {front_fps:.2f} fps")
        print(f"    Side:  {side_fps:.2f} fps")
        print("    Results may have sync drift over long recordings.")
    
    # Step 2: Extract landmarks from front camera
    print("\n[2/4] Processing front camera...")
    front_data, _ = extract_landmarks(front_video, front_sync)
    
    # Step 3: Extract landmarks from side camera
    print("\n[3/4] Processing side camera...")
    side_data, _ = extract_landmarks(side_video, side_sync)
    
    # Step 4: Triangulate
    print("\n[4/4] Triangulating 3D positions...")
    combined_data = triangulate_3d(front_data, side_data)
    print(f"  Combined {len(combined_data)} frames")
    
    # Save output
    output_dir = Path("data/takes")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"dual_{output_name}_{timestamp}.json"
    
    # Count valid data
    pose_frames = sum(1 for f in combined_data if f.get('pose_landmarks'))
    face_frames = sum(1 for f in combined_data if f.get('face_landmarks'))
    
    output_data = {
        'metadata': {
            'type': 'dual_camera',
            'frames': len(combined_data),
            'fps': front_fps,
            'duration': len(combined_data) / front_fps,
            'date': timestamp,
            'front_video': str(front_video),
            'side_video': str(side_video),
            'front_sync_frame': front_sync,
            'side_sync_frame': side_sync,
            'pose_frames': pose_frames,
            'face_frames': face_frames,
            'coordinate_system': {
                'x': 'horizontal (left-right from front camera)',
                'y': 'height (top-bottom from front camera, 0=top, 1=bottom)',
                'z': 'depth (from side camera, 0=far, 1=near)'
            }
        },
        'data': combined_data
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f)
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print(f"  Output: {output_file}")
    print(f"  Frames: {len(combined_data)}")
    print(f"  Duration: {len(combined_data)/front_fps:.1f} seconds")
    print(f"  Pose data: {pose_frames} frames")
    print("=" * 70)
    print("\nNext: Import in Blender using MelodicCap Master (Auto-Calibrate)")


if __name__ == "__main__":
    main()
