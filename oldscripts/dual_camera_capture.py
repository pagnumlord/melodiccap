# MELODICCAP DUAL CAMERA SYSTEM
# Process two video files (front + side) into single mocap JSON
# 
# Workflow:
# 1. Record front video (Sony ZV-1F)
# 2. Record side video (Samsung S25) - 90° angle
# 3. Clap hands at start for sync
# 4. Run this script to process both videos
# 5. Import resulting JSON into Blender

import cv2
import mediapipe as mp
import json
import numpy as np
from datetime import datetime
import os

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# === CONFIGURATION ===
OUTPUT_DIR = "data/takes"
SYNC_SEARCH_FRAMES = 90  # Search first 3 seconds (at 30fps) for sync point

def find_sync_frame(video_path, method='motion'):
    """
    Find the sync frame (clap) in a video.
    Returns the frame number where the clap/sync event occurs.
    
    Methods:
    - 'motion': Detect sudden motion spike (hand clap)
    - 'audio': Detect audio spike (requires audio track)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video FPS: {fps}")
    
    prev_frame = None
    motion_scores = []
    
    frame_num = 0
    while frame_num < SYNC_SEARCH_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if prev_frame is not None:
            # Calculate frame difference (motion)
            diff = cv2.absdiff(prev_frame, gray)
            motion_score = np.sum(diff) / diff.size
            motion_scores.append((frame_num, motion_score))
        
        prev_frame = gray
        frame_num += 1
    
    cap.release()
    
    if not motion_scores:
        return 0
    
    # Find frame with maximum motion (the clap)
    max_motion = max(motion_scores, key=lambda x: x[1])
    sync_frame = max_motion[0]
    
    print(f"  Sync frame detected: {sync_frame} (motion score: {max_motion[1]:.2f})")
    return sync_frame

def process_video(video_path, holistic, start_frame=0):
    """
    Process a single video file and extract landmarks.
    Returns list of frame data starting from start_frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Skip to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames_data = []
    frame_num = 0
    
    print(f"  Processing from frame {start_frame}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Don't flip - we handle orientation in triangulation
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)
        
        frame_data = {
            'frame': frame_num,
            'timestamp': frame_num / fps,
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
        
        # Progress indicator
        if frame_num % 30 == 0:
            print(f"    Frame {frame_num}...")
    
    cap.release()
    return frames_data, fps

def triangulate_landmarks(front_data, side_data):
    """
    Combine front and side camera data into true 3D coordinates.
    
    Front camera (ZV-1F): Captures X (left/right) and Z (up/down)
    Side camera (S25): Captures Y (depth) and Z (up/down)
    
    We use Z from front camera (more reliable for height)
    """
    combined_data = []
    
    # Use shorter of the two recordings
    num_frames = min(len(front_data), len(side_data))
    
    for i in range(num_frames):
        front_frame = front_data[i]
        side_frame = side_data[i]
        
        frame_data = {
            'frame': i,
            'timestamp': front_frame['timestamp'],
            'pose_landmarks': None,
            'face_landmarks': None,
            'left_hand_landmarks': None,
            'right_hand_landmarks': None
        }
        
        # Triangulate pose landmarks
        if front_frame['pose_landmarks'] and side_frame['pose_landmarks']:
            combined_landmarks = []
            
            for j in range(33):
                front_lm = front_frame['pose_landmarks'][j]
                side_lm = side_frame['pose_landmarks'][j]
                
                # Front camera gives us X (horizontal) and Y (which becomes Z in Blender)
                # Side camera gives us X (which becomes Y/depth in Blender) and Y (height verification)
                
                combined_landmarks.append({
                    'id': j,
                    'x': front_lm['x'],           # X from front view
                    'y': front_lm['y'],           # Y (height) from front view
                    'z': side_lm['x'],            # Depth from side view's X
                    'visibility': min(front_lm['visibility'], side_lm['visibility'])
                })
            
            frame_data['pose_landmarks'] = combined_landmarks
        
        # For face and hands, just use front camera (better angle)
        frame_data['face_landmarks'] = front_frame.get('face_landmarks')
        frame_data['left_hand_landmarks'] = front_frame.get('left_hand_landmarks')
        frame_data['right_hand_landmarks'] = front_frame.get('right_hand_landmarks')
        
        combined_data.append(frame_data)
    
    return combined_data

def process_dual_camera(front_video_path, side_video_path, output_name=None):
    """
    Main function to process dual camera footage.
    
    Args:
        front_video_path: Path to front camera video (Sony ZV-1F)
        side_video_path: Path to side camera video (Samsung S25)
        output_name: Optional custom name for output file
    """
    print("=" * 70)
    print("MELODICCAP DUAL CAMERA PROCESSING")
    print("=" * 70)
    
    # Verify files exist
    if not os.path.exists(front_video_path):
        print(f"ERROR: Front video not found: {front_video_path}")
        return None
    if not os.path.exists(side_video_path):
        print(f"ERROR: Side video not found: {side_video_path}")
        return None
    
    # Find sync points
    print("\n[1/4] Finding sync points...")
    print(f"  Front video: {front_video_path}")
    front_sync = find_sync_frame(front_video_path)
    
    print(f"  Side video: {side_video_path}")
    side_sync = find_sync_frame(side_video_path)
    
    # Initialize MediaPipe
    print("\n[2/4] Processing front camera...")
    holistic = mp_holistic.Holistic(
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=True
    )
    
    front_data, front_fps = process_video(front_video_path, holistic, front_sync)
    print(f"  Extracted {len(front_data)} frames at {front_fps} fps")
    
    print("\n[3/4] Processing side camera...")
    side_data, side_fps = process_video(side_video_path, holistic, side_sync)
    print(f"  Extracted {len(side_data)} frames at {side_fps} fps")
    
    holistic.close()
    
    # Check framerate match
    if abs(front_fps - side_fps) > 1:
        print(f"\nWARNING: Framerate mismatch! Front: {front_fps}, Side: {side_fps}")
        print("Results may be misaligned. Ensure both cameras are set to 30fps.")
    
    # Triangulate
    print("\n[4/4] Triangulating 3D positions...")
    combined_data = triangulate_landmarks(front_data, side_data)
    print(f"  Combined {len(combined_data)} frames")
    
    # Save output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_name:
        filename = f"{OUTPUT_DIR}/dual_{output_name}_{timestamp}.json"
    else:
        filename = f"{OUTPUT_DIR}/dual_{timestamp}.json"
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output = {
        'metadata': {
            'type': 'dual_camera',
            'frames': len(combined_data),
            'duration': len(combined_data) / front_fps,
            'fps': front_fps,
            'front_video': os.path.basename(front_video_path),
            'side_video': os.path.basename(side_video_path),
            'front_sync_frame': front_sync,
            'side_sync_frame': side_sync,
            'date': timestamp
        },
        'data': combined_data
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f)
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print(f"  Output: {filename}")
    print(f"  Frames: {len(combined_data)}")
    print(f"  Duration: {len(combined_data) / front_fps:.1f} seconds")
    print("=" * 70)
    
    return filename


# === COMMAND LINE INTERFACE ===
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python dual_camera_capture.py <front_video> <side_video> [output_name]")
        print("")
        print("Example:")
        print("  python dual_camera_capture.py front.mp4 side.mp4 test_take")
        print("")
        print("The script will:")
        print("  1. Find the sync point (clap) in both videos")
        print("  2. Process landmarks from both angles")
        print("  3. Triangulate true 3D positions")
        print("  4. Save combined JSON for Blender import")
        sys.exit(1)
    
    front_video = sys.argv[1]
    side_video = sys.argv[2]
    output_name = sys.argv[3] if len(sys.argv) > 3 else None
    
    process_dual_camera(front_video, side_video, output_name)