import cv2
import threading
import time
import mediapipe as mp
import numpy as np

class CameraThread(threading.Thread):
    def __init__(self, camera_id, label_widget, mp_pose=None):
        super().__init__()
        self.camera_id = camera_id
        self.label_widget = label_widget
        self.mp_pose = mp_pose
        self.running = True
        self.cap = None
        self.latest_frame = None
        self.landmarks = None
        
        # Initialize MediaPipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        
        # Calibration & Performance Optimization
        self.calibration_mode = False
        self.board_detected_section = None
        self.last_detection_time = 0
        self.detection_interval = 0.1 # 10 FPS for ChArUco is plenty
        
        # Initialize Board Detector
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        board = cv2.aruco.CharucoBoard((4, 3), 0.040, 0.030, dictionary)
        self.charuco_detector = cv2.aruco.CharucoDetector(board)

    def run(self):
        # Try DirectShow first (standard for Windows capture)
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            # Fallback to Media Foundation (better for some Sony cameras)
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_MSMF)
            
        if not self.cap.isOpened():
            # Final fallback to default
            self.cap = cv2.VideoCapture(self.camera_id)

        if self.cap.isOpened():
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            except Exception as e:
                print(f"Warning: Could not set camera properties for Cam {self.camera_id}: {e}")
            
            print(f"Camera {self.camera_id} opened successfully.")
        else:
            print(f"Failed to open Camera {self.camera_id}")
            self.running = False
            return
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process MediaPipe
            pose_results = self.pose.process(frame_rgb)
            hand_results = self.hands.process(frame_rgb)
            
            # 2. Process ChArUco (Throttled for Performance)
            if self.calibration_mode and (time.time() - self.last_detection_time > self.detection_interval):
                self.last_detection_time = time.time()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _, _ = self.charuco_detector.detectBoard(gray)
                
                if corners is not None and len(corners) > 4:
                    h, w = frame.shape[:2]
                    center = np.mean(corners, axis=0).flatten()
                    sec_x = int(center[0] / (w / 3))
                    sec_y = int(center[1] / (h / 3))
                    self.board_detected_section = sec_y * 3 + sec_x
                else:
                    self.board_detected_section = None
            
            # 3. Draw on frame for UI
            frame_annotated = frame_rgb.copy()
            if pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame_annotated, 
                    pose_results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(58, 134, 255), thickness=2, circle_radius=2)
                )
            
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame_annotated, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 77, 77), thickness=2, circle_radius=2)
                    )
            
            self.latest_frame = frame_annotated
            # Store both 3D world landmarks (for fallback) and normalized 2D landmarks (for triangulation)
            # Ensure multi_hand_landmarks is captured for Finger-Fidelity
            self.landmarks = {
                "pose": pose_results.pose_landmarks,
                "hands": hand_results.multi_hand_landmarks,
                "pose_world": pose_results.pose_world_landmarks
            }
        
        if self.cap:
            self.cap.release()

    def stop(self):
        self.running = False
