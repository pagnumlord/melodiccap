import json
import os
import time
from datetime import datetime

class TakeRecorder:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.frames = []
        self.is_recording = False
        self.start_time = 0
        self.take_name = ""

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def start(self):
        self.frames = []
        self.is_recording = True
        self.start_time = time.time()
        self.take_name = f"take_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Started recording: {self.take_name}")

    def add_frame(self, points_3d, lms_2d_a=None, lms_2d_b=None, hands_a=None, hands_b=None):
        if not self.is_recording:
            return
        
        frame_data = {
            "timestamp": time.time() - self.start_time,
            "landmarks_3d": {},
            "raw_2d": {
                "cam_a": self._serialize_landmarks(lms_2d_a),
                "cam_b": self._serialize_landmarks(lms_2d_b)
            },
            "hands_2d": {
                "cam_a": self._serialize_hands(hands_a),
                "cam_b": self._serialize_hands(hands_b)
            }
        }
        
        # Convert numpy floats to standard python floats for JSON
        if points_3d:
            for idx, pt in points_3d.items():
                frame_data["landmarks_3d"][str(idx)] = [float(c) for c in pt]
            
        self.frames.append(frame_data)

    def _serialize_landmarks(self, landmarks):
        """Converts MediaPipe landmarks to a serializable list/dict"""
        if not landmarks: return None
        res = {}
        for i, lm in enumerate(landmarks.landmark):
            res[str(i)] = [lm.x, lm.y, lm.z, lm.visibility]
        return res

    def _serialize_hands(self, hands):
        """Converts MediaPipe multi_hand_landmarks to a serializable list"""
        if not hands: return None
        all_hands = []
        for hand_landmarks in hands:
            h_res = {}
            for i, lm in enumerate(hand_landmarks.landmark):
                h_res[str(i)] = [lm.x, lm.y, lm.z]
            all_hands.append(h_res)
        return all_hands

    def stop(self):
        if not self.is_recording:
            return None
            
        self.is_recording = False
        duration = time.time() - self.start_time
        fps = len(self.frames) / duration if duration > 0 else 0
        
        take_data = {
            "take_name": self.take_name,
            "fps": fps,
            "frame_count": len(self.frames),
            "duration": duration,
            "frames": self.frames
        }
        
        file_path = os.path.join(self.output_dir, f"{self.take_name}.json")
        with open(file_path, 'w') as f:
            json.dump(take_data, f, indent=2)
            
        print(f"Saved take to {file_path} ({len(self.frames)} frames at {fps:.2f} fps)")
        return file_path
