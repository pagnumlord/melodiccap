"""
ArUco Dictionary Debug
======================
Tries ALL ArUco dictionaries to find which one your board uses.
Also saves a frame for inspection.
"""

import cv2
import numpy as np
import os

print("="*50)
print("ARUCO DICTIONARY DEBUG")
print("="*50)

# All ArUco dictionaries to try
DICTIONARIES = [
    (cv2.aruco.DICT_4X4_50, "DICT_4X4_50"),
    (cv2.aruco.DICT_4X4_100, "DICT_4X4_100"),
    (cv2.aruco.DICT_4X4_250, "DICT_4X4_250"),
    (cv2.aruco.DICT_4X4_1000, "DICT_4X4_1000"),
    (cv2.aruco.DICT_5X5_50, "DICT_5X5_50"),
    (cv2.aruco.DICT_5X5_100, "DICT_5X5_100"),
    (cv2.aruco.DICT_5X5_250, "DICT_5X5_250"),
    (cv2.aruco.DICT_6X6_50, "DICT_6X6_50"),
    (cv2.aruco.DICT_6X6_100, "DICT_6X6_100"),
    (cv2.aruco.DICT_6X6_250, "DICT_6X6_250"),
    (cv2.aruco.DICT_7X7_50, "DICT_7X7_50"),
    (cv2.aruco.DICT_7X7_100, "DICT_7X7_100"),
    (cv2.aruco.DICT_ARUCO_ORIGINAL, "DICT_ARUCO_ORIGINAL"),
]

# Open camera (Sony at index 2 with DSHOW)
print("\nOpening Camera A (Sony, index 2)...")
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Trying without DSHOW...")
    cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("ERROR: Could not open camera!")
    exit(1)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Camera opened. Hold the board in view...")
print("Press SPACE to capture and test all dictionaries")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Show live view
    display = frame.copy()
    cv2.putText(display, "Press SPACE to test detection", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(display, "Hold board steady in view", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("ArUco Debug - SPACE to test, Q to quit", display)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    
    elif key == ord(' '):
        print("\n" + "="*50)
        print("TESTING ALL DICTIONARIES")
        print("="*50)
        
        # Save frame
        cv2.imwrite("debug_frame.png", frame)
        print(f"Saved frame as debug_frame.png")
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("debug_frame_gray.png", gray)
        print(f"Saved grayscale as debug_frame_gray.png")
        
        # Also try with and without flip
        frame_flipped = cv2.flip(frame, 1)
        gray_flipped = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2GRAY)
        
        print("\n--- Testing WITHOUT flip ---")
        best_dict = None
        best_count = 0
        
        for dict_id, dict_name in DICTIONARIES:
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            
            corners, ids, rejected = detector.detectMarkers(gray)
            count = len(ids) if ids is not None else 0
            
            if count > 0:
                print(f"  {dict_name}: {count} markers detected! IDs: {ids.flatten().tolist()}")
                if count > best_count:
                    best_count = count
                    best_dict = dict_name
            else:
                print(f"  {dict_name}: 0 markers")
        
        print("\n--- Testing WITH flip ---")
        best_dict_flip = None
        best_count_flip = 0
        
        for dict_id, dict_name in DICTIONARIES:
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            
            corners, ids, rejected = detector.detectMarkers(gray_flipped)
            count = len(ids) if ids is not None else 0
            
            if count > 0:
                print(f"  {dict_name}: {count} markers detected! IDs: {ids.flatten().tolist()}")
                if count > best_count_flip:
                    best_count_flip = count
                    best_dict_flip = dict_name
        
        print("\n" + "="*50)
        print("RESULTS")
        print("="*50)
        
        if best_count > 0:
            print(f"WITHOUT FLIP: Best = {best_dict} ({best_count} markers)")
        else:
            print("WITHOUT FLIP: No markers detected with any dictionary!")
        
        if best_count_flip > 0:
            print(f"WITH FLIP: Best = {best_dict_flip} ({best_count_flip} markers)")
        else:
            print("WITH FLIP: No markers detected with any dictionary!")
        
        if best_count == 0 and best_count_flip == 0:
            print("\n!!! NO MARKERS DETECTED !!!")
            print("Possible issues:")
            print("  1. Board not in frame")
            print("  2. Poor lighting/contrast")
            print("  3. Board is not ArUco-based")
            print("  4. Camera resolution too low")
            print(f"\nCheck debug_frame.png in: {os.getcwd()}")
        
        print("\nPress SPACE to test again or Q to quit")

cap.release()
cv2.destroyAllWindows()
print("\n[DONE]")
