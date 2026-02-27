# Save as: capture_with_tpose.py in MelodicCapStudio folder

import cv2
import mediapipe as mp
import json
import time
from datetime import datetime

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def capture_with_tpose():
    cap = cv2.VideoCapture(0)
    
    # Try to set to 30fps
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    pose = mp_pose.Pose(
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    print("=" * 60)
    print("MelodicCap Studio - T-Pose Calibration")
    print("=" * 60)
    print("Instructions:")
    print("1. Stand in T-pose (arms out to sides)")
    print("2. Press 'r' to start recording")
    print("3. Hold T-pose for 2 seconds")
    print("4. Then perform your action")
    print("5. Recording will auto-stop after 10 seconds")
    print("Press 'q' to quit")
    print("=" * 60)
    
    recording = False
    frames_data = []
    start_time = None
    record_duration = 10  # seconds
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        
        # Process
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        # Draw skeleton
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
        
        # Recording logic
        if recording:
            elapsed = time.time() - start_time
            
            if elapsed < record_duration:
                # Add frame data
                if results.pose_landmarks:
                    landmark_data = []
                    for idx, landmark in enumerate(results.pose_landmarks.landmark):
                        landmark_data.append({
                            'id': idx,
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z,
                            'visibility': landmark.visibility
                        })
                    
                    frames_data.append({
                        'frame': len(frames_data),
                        'timestamp': elapsed,
                        'landmarks': landmark_data
                    })
                
                # Visual feedback
                cv2.putText(frame, f"RECORDING: {elapsed:.1f}s", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 0, 255), 2)
            else:
                # Auto-stop after duration
                recording = False
                
                # Save
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"data/takes/capture_{timestamp}.json"
                
                data = {
                    'metadata': {
                        'frames': len(frames_data),
                        'duration': record_duration,
                        'fps': len(frames_data) / record_duration,
                        'date': timestamp
                    },
                    'data': frames_data
                }
                
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"\nRecording complete! Captured {len(frames_data)} frames")
                print(f"Data saved to: {filename}")
                frames_data = []
        else:
            cv2.putText(frame, "Press 'r' to record", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
        
        cv2.imshow('MelodicCap - Motion Capture', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r') and not recording:
            recording = True
            start_time = time.time()
            frames_data = []
            print("\nRecording started!")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    capture_with_tpose()