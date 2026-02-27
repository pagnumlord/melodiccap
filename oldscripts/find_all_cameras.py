# MELODICCAP - FIND ALL CAMERAS
# find_all_cameras.py
# Run this to see what cameras Windows can see

import cv2
import sys

def find_cameras():
    print("=" * 60)
    print("MELODICCAP - CAMERA DETECTION")
    print("=" * 60)
    print("\nSearching for cameras...\n")
    
    found_cameras = []
    
    # Check indices 0-10
    for i in range(10):
        # Try DirectShow backend (Windows)
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        
        if cap.isOpened():
            # Get info
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Try to get backend name
            backend = cap.getBackendName()
            
            # Read a test frame
            ret, frame = cap.read()
            frame_ok = "✓" if ret else "✗"
            
            print(f"  Camera {i}: {w}x{h} @ {fps}fps [{backend}] Frame:{frame_ok}")
            found_cameras.append({
                'index': i,
                'width': w,
                'height': h,
                'fps': fps,
                'backend': backend,
                'working': ret
            })
            
            cap.release()
        else:
            # Also try without DirectShow
            cap2 = cv2.VideoCapture(i)
            if cap2.isOpened():
                w = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap2.get(cv2.CAP_PROP_FPS)
                ret, frame = cap2.read()
                frame_ok = "✓" if ret else "✗"
                
                print(f"  Camera {i}: {w}x{h} @ {fps}fps [default] Frame:{frame_ok}")
                found_cameras.append({
                    'index': i,
                    'width': w,
                    'height': h,
                    'fps': fps,
                    'backend': 'default',
                    'working': ret
                })
                cap2.release()
    
    print("\n" + "=" * 60)
    
    if not found_cameras:
        print("No cameras found!")
        print("\nTroubleshooting:")
        print("  1. Is your ZV-1F connected via USB and in webcam mode?")
        print("  2. Is your S25 connected and set to 'Webcam' USB mode?")
        print("  3. Try disconnecting and reconnecting")
    else:
        print(f"Found {len(found_cameras)} camera(s)")
        print("\nFor dual camera capture, you need 2 working cameras.")
        print("\nTypical setup:")
        print("  - Lower index = built-in/older device")
        print("  - Higher index = newer/USB device")
        
        if len(found_cameras) >= 2:
            print("\n✓ You have enough cameras for dual capture!")
            print(f"\nSuggested configuration:")
            print(f"  FRONT_CAMERA_INDEX = {found_cameras[0]['index']}  (face this one)")
            print(f"  SIDE_CAMERA_INDEX = {found_cameras[1]['index']}  (put to your left)")
    
    print("=" * 60)
    
    return found_cameras

def preview_camera(index):
    """Preview a specific camera to verify which is which"""
    print(f"\nPreviewing camera {index}... Press Q to close, N for next camera")
    
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        print(f"Could not open camera {index}")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add label
        cv2.putText(frame, f"Camera {index}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press Q to close, N for next", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow(f"Camera {index} Preview", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            cap.release()
            cv2.destroyAllWindows()
            return index + 1  # Return next index to preview
    
    cap.release()
    cv2.destroyAllWindows()
    return None

if __name__ == "__main__":
    cameras = find_cameras()
    
    if cameras:
        print("\nWould you like to preview cameras to identify them?")
        response = input("Enter camera index to preview (or 'q' to quit): ")
        
        while response.lower() != 'q':
            try:
                idx = int(response)
                next_idx = preview_camera(idx)
                if next_idx is not None:
                    response = str(next_idx)
                else:
                    response = input("\nEnter camera index to preview (or 'q' to quit): ")
            except ValueError:
                response = input("Enter camera index to preview (or 'q' to quit): ")
    
    print("\nDone!")
