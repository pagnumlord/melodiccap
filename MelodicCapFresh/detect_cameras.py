"""
Camera Detection Utility
========================
Run this to find your camera indices.
"""

import cv2
import sys

def detect_cameras():
    print("\n" + "="*50)
    print("CAMERA DETECTION")
    print("="*50 + "\n")
    
    found = []
    
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            # Get some info
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"  Camera {i}: AVAILABLE ({w}x{h} @ {fps:.0f}fps)")
            found.append(i)
            cap.release()
        else:
            print(f"  Camera {i}: not available")
    
    print("\n" + "-"*50)
    
    if len(found) >= 2:
        print(f"\nFound {len(found)} cameras: {found}")
        print("\nRecommended settings for melodic_capture_v1.py:")
        print(f"  CAM_A_INDEX = {found[0]}  # First camera")
        print(f"  CAM_B_INDEX = {found[1]}  # Second camera")
    elif len(found) == 1:
        print(f"\nOnly found 1 camera (index {found[0]})")
        print("Make sure both cameras are connected!")
    else:
        print("\nNo cameras found!")
        print("Check your connections and drivers.")
    
    print("\n" + "="*50)
    
    # Preview option
    if found:
        print("\nWould you like to preview cameras? (y/n)")
        choice = input("> ").strip().lower()
        
        if choice == 'y':
            preview_cameras(found)


def preview_cameras(indices):
    """Show preview of all detected cameras"""
    caps = []
    
    for i in indices:
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            caps.append((i, cap))
    
    print("\nPreviewing cameras. Press 'Q' to quit.\n")
    
    while True:
        frames = []
        for idx, cap in caps:
            ret, frame = cap.read()
            if ret:
                # Add label
                cv2.putText(frame, f"Camera {idx}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Resize for display
                frame = cv2.resize(frame, (640, 360))
                frames.append(frame)
        
        if frames:
            # Stack horizontally
            import numpy as np
            combined = np.hstack(frames) if len(frames) > 1 else frames[0]
            cv2.imshow("Camera Preview (Q to quit)", combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    for _, cap in caps:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_cameras()
