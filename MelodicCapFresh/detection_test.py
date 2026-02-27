"""
Quick Board Detection Test
==========================
Just tests if the board can be detected - no calibration.
Uses exact same code as working diagnostic.
"""

import cv2
import numpy as np

print("="*50)
print("BOARD DETECTION TEST")
print("="*50)

# ChArUco setup (same as diagnostic)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard((4, 3), 0.0635, 0.0476, aruco_dict)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)

# Open cameras
print("\nOpening cameras...")
cap_a = cv2.VideoCapture(2, cv2.CAP_DSHOW)  # Sony
cap_b = cv2.VideoCapture(0, cv2.CAP_ANY)    # DroidCam

if not cap_a.isOpened():
    print("ERROR: Camera A (Sony) failed!")
if not cap_b.isOpened():
    print("ERROR: Camera B (DroidCam) failed!")

print("Press Q to quit")
print("Press G to toggle Camera A flip")
print("Press H to toggle Camera B flip")

flip_a = True  # Start with flips on since cameras are mirrored
flip_b = True

while True:
    ret_a, frame_a = cap_a.read()
    ret_b, frame_b = cap_b.read()
    
    if not ret_a or not ret_b:
        continue
    
    # Apply flips
    if flip_a:
        frame_a = cv2.flip(frame_a, 1)
    if flip_b:
        frame_b = cv2.flip(frame_b, 1)
    
    # Detect on Camera A
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    corners_a, ids_a, rejected_a = detector.detectMarkers(gray_a)
    
    display_a = frame_a.copy()
    markers_a = len(ids_a) if ids_a is not None else 0
    
    if ids_a is not None and len(ids_a) > 0:
        cv2.aruco.drawDetectedMarkers(display_a, corners_a, ids_a)
        num, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            corners_a, ids_a, gray_a, board
        )
        if num and num > 0:
            cv2.aruco.drawDetectedCornersCharuco(display_a, ch_corners, ch_ids)
    
    # Detect on Camera B
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    corners_b, ids_b, rejected_b = detector.detectMarkers(gray_b)
    
    display_b = frame_b.copy()
    markers_b = len(ids_b) if ids_b is not None else 0
    
    if ids_b is not None and len(ids_b) > 0:
        cv2.aruco.drawDetectedMarkers(display_b, corners_b, ids_b)
        num, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            corners_b, ids_b, gray_b, board
        )
        if num and num > 0:
            cv2.aruco.drawDetectedCornersCharuco(display_b, ch_corners, ch_ids)
    
    # Draw info
    cv2.putText(display_a, f"CAM A - Markers: {markers_a} - Flip: {'ON' if flip_a else 'OFF'}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(display_b, f"CAM B - Markers: {markers_b} - Flip: {'ON' if flip_b else 'OFF'}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Resize and show
    display_a = cv2.resize(display_a, (640, 360))
    display_b = cv2.resize(display_b, (640, 360))
    combined = np.hstack([display_a, display_b])
    
    cv2.imshow("Board Detection Test - Q to quit", combined)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('g'):
        flip_a = not flip_a
        print(f"Camera A flip: {'ON' if flip_a else 'OFF'}")
    elif key == ord('h'):
        flip_b = not flip_b
        print(f"Camera B flip: {'ON' if flip_b else 'OFF'}")

cap_a.release()
cap_b.release()
cv2.destroyAllWindows()
print("\n[DONE]")
