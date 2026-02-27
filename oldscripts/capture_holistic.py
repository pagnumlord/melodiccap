import cv2
import mediapipe as mp
import json
import time
from datetime import datetime

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def capture_holistic():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    holistic = mp_holistic.Holistic(
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=True
    )
    
    print("=" * 60)
    print("MelodicCap Studio - Holistic Capture")
    print("=" * 60)
    print(f"Camera: {actual_w}x{actual_h} @ {actual_fps}fps")
    print("=" * 60)
    print("1. Press 'r' to start 5-second countdown")
    print("2. Get into T-pose during countdown")
    print("3. Hold T-pose for first 2 seconds of recording")
    print("4. Then perform your action")
    print("5. Recording auto-stops after 10 seconds")
    print("6. Press 'q' to quit")
    print("=" * 60)
    
    recording = False
    countdown_active = False
    frames_data = []
    start_time = None
    countdown_start = None
    record_duration = 10
    countdown_duration = 5
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)
        
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80,256,121), thickness=1)
            )
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255,100,100), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=2)
            )
        
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )
        
        tracking = []
        if results.pose_landmarks: tracking.append("BODY")
        if results.face_landmarks: tracking.append("FACE")
        if results.left_hand_landmarks: tracking.append("L-HAND")
        if results.right_hand_landmarks: tracking.append("R-HAND")
        
        # Countdown logic
        if countdown_active:
            elapsed = time.time() - countdown_start
            remaining = countdown_duration - elapsed
            
            if remaining > 0:
                # Show big countdown number
                countdown_num = int(remaining) + 1
                cv2.putText(frame, str(countdown_num), 
                           (frame.shape[1]//2 - 80, frame.shape[0]//2 + 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 255), 15)
                cv2.putText(frame, "GET READY - T-POSE!", 
                           (frame.shape[1]//2 - 250, frame.shape[0]//2 - 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            else:
                # Countdown finished, start recording
                countdown_active = False
                recording = True
                start_time = time.time()
                frames_data = []
                print("\nRECORDING NOW!")
        
        # Recording logic
        elif recording:
            elapsed = time.time() - start_time
            
            if elapsed < record_duration:
                frame_data = {
                    'frame': len(frames_data),
                    'timestamp': elapsed,
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
                
                # Show recording indicator
                cv2.putText(frame, f"REC: {elapsed:.1f}s", 
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(frame, f"Tracking: {', '.join(tracking)}", 
                           (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Show T-pose reminder for first 2 seconds
                if elapsed < 2:
                    cv2.putText(frame, "HOLD T-POSE", 
                               (frame.shape[1]//2 - 150, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            else:
                recording = False
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"data/takes/holistic_{timestamp}.json"
                
                pose_frames = sum(1 for f in frames_data if f['pose_landmarks'])
                face_frames = sum(1 for f in frames_data if f['face_landmarks'])
                lhand_frames = sum(1 for f in frames_data if f['left_hand_landmarks'])
                rhand_frames = sum(1 for f in frames_data if f['right_hand_landmarks'])
                
                data = {
                    'metadata': {
                        'type': 'holistic',
                        'frames': len(frames_data),
                        'duration': record_duration,
                        'fps': len(frames_data) / record_duration,
                        'resolution': f"{actual_w}x{actual_h}",
                        'date': timestamp,
                        'pose_frames': pose_frames,
                        'face_frames': face_frames,
                        'left_hand_frames': lhand_frames,
                        'right_hand_frames': rhand_frames
                    },
                    'data': frames_data
                }
                
                with open(filename, 'w') as f:
                    json.dump(data, f)
                
                print(f"\nRecording complete!")
                print(f"  Frames: {len(frames_data)}")
                print(f"  Body: {pose_frames}, Face: {face_frames}")
                print(f"  L-Hand: {lhand_frames}, R-Hand: {rhand_frames}")
                print(f"  Saved: {filename}")
                frames_data = []
        else:
            cv2.putText(frame, "Press 'r' to record (5s countdown)", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f"Tracking: {', '.join(tracking) if tracking else 'None'}", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('MelodicCap - Holistic', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r') and not recording and not countdown_active:
            countdown_active = True
            countdown_start = time.time()
            print("\nCountdown started! Get into position...")
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    holistic.close()

if __name__ == "__main__":
    capture_holistic()