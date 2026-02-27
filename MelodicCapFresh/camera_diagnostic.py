"""
MelodicCap Camera Diagnostic v2
================================
Shows ALL cameras at once so you can identify:
- Which cameras are which
- Which are mirrored
- Which detect the ChArUco board
- Best camera indices for MelodicCap

Press number keys (1-9) to toggle flip on that camera
Press Q to quit and see summary
"""

import cv2
import numpy as np
import time

# ChArUco setup (same as MelodicCap)
CHARUCO_COLS = 4
CHARUCO_ROWS = 3
CHARUCO_SQUARE_M = 0.0635
CHARUCO_MARKER_M = 0.0476

def create_detector():
    """Create ChArUco detector"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_COLS, CHARUCO_ROWS),
        CHARUCO_SQUARE_M,
        CHARUCO_MARKER_M,
        aruco_dict
    )
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    return detector, board

def detect_charuco(frame, detector, board):
    """Detect ChArUco and return corner count"""
    if frame is None:
        return 0, 0, frame
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    
    marker_count = len(ids) if ids is not None else 0
    corner_count = 0
    
    if ids is not None and len(ids) >= 2:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        num, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )
        corner_count = num if num else 0
        if num and num > 0:
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
    
    return marker_count, corner_count, frame

def main():
    print("\n" + "="*60)
    print("MELODICCAP CAMERA DIAGNOSTIC v2")
    print("="*60)
    print("\nScanning for cameras...")
    
    # Find all cameras
    cameras = {}
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cameras[i] = {
                'cap': cap,
                'width': w,
                'height': h,
                'flipped': False,
                'name': f"Camera {i}"
            }
            print(f"  [{i}] Found: {w}x{h}")
        else:
            cap.release()
    
    if len(cameras) == 0:
        print("\nNo cameras found!")
        return
    
    print(f"\nFound {len(cameras)} cameras: {list(cameras.keys())}")
    print("\n" + "="*60)
    print("LIVE VIEW - All Cameras")
    print("="*60)
    print("CONTROLS:")
    print("  Number keys (0-9) = Toggle horizontal flip on that camera")
    print("  Q = Quit and show summary")
    print("  Hold up your RIGHT hand to check for mirroring")
    print("  Show ChArUco board to test detection")
    print("="*60 + "\n")
    
    detector, board = create_detector()
    
    # Track stats
    for idx in cameras:
        cameras[idx]['markers_detected'] = 0
        cameras[idx]['corners_detected'] = 0
        cameras[idx]['frames'] = 0
    
    while True:
        frames = []
        
        for idx, cam in cameras.items():
            ret, frame = cam['cap'].read()
            
            if not ret or frame is None:
                # Create blank frame
                frame = np.zeros((360, 480, 3), dtype=np.uint8)
                cv2.putText(frame, f"Camera {idx}: No Signal", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Apply flip if enabled
                if cam['flipped']:
                    frame = cv2.flip(frame, 1)
                
                # Detect ChArUco
                markers, corners, frame = detect_charuco(frame.copy(), detector, board)
                
                # Update stats
                cam['frames'] += 1
                if markers > 0:
                    cam['markers_detected'] += 1
                if corners > 0:
                    cam['corners_detected'] += 1
                
                # Resize for display
                frame = cv2.resize(frame, (480, 270))
                h, w = frame.shape[:2]
                
                # Draw info overlay
                # Background bar
                cv2.rectangle(frame, (0, 0), (w, 65), (0, 0, 0), -1)
                
                # Camera index (big)
                cv2.putText(frame, f"[{idx}]", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                
                # Flip status
                flip_text = "FLIPPED" if cam['flipped'] else "normal"
                flip_color = (0, 255, 0) if cam['flipped'] else (150, 150, 150)
                cv2.putText(frame, flip_text, (70, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, flip_color, 2)
                
                # Resolution
                cv2.putText(frame, f"{cam['width']}x{cam['height']}", (180, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Detection status
                if corners > 0:
                    det_text = f"BOARD: {corners} corners"
                    det_color = (0, 255, 0)
                elif markers > 0:
                    det_text = f"Markers: {markers}"
                    det_color = (0, 165, 255)
                else:
                    det_text = "No board"
                    det_color = (100, 100, 100)
                
                cv2.putText(frame, det_text, (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, det_color, 2)
                
                # Draw center line for mirror check
                cv2.line(frame, (w//2, 65), (w//2, h), (50, 50, 50), 1)
                cv2.putText(frame, "L", (5, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
                cv2.putText(frame, "R", (w-15, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            
            frames.append((idx, frame))
        
        # Arrange frames in grid
        # Determine grid size
        n = len(frames)
        if n <= 2:
            cols = 2
        elif n <= 4:
            cols = 2
        elif n <= 6:
            cols = 3
        else:
            cols = 4
        
        rows = (n + cols - 1) // cols
        
        # Create grid
        cell_h, cell_w = 270, 480
        grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
        
        for i, (idx, frame) in enumerate(frames):
            r, c = i // cols, i % cols
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            
            # Ensure frame fits
            fh, fw = frame.shape[:2]
            grid[y1:y1+fh, x1:x1+fw] = frame
        
        # Add instructions at bottom
        inst_bar = np.zeros((40, grid.shape[1], 3), dtype=np.uint8)
        cv2.putText(inst_bar, "Press camera NUMBER to toggle flip | Q=Quit | Hold RIGHT hand up to check mirror | Show board to test detection",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        display = np.vstack([grid, inst_bar])
        
        cv2.imshow("MelodicCap Camera Diagnostic", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        # Number keys toggle flip
        if ord('0') <= key <= ord('9'):
            cam_idx = key - ord('0')
            if cam_idx in cameras:
                cameras[cam_idx]['flipped'] = not cameras[cam_idx]['flipped']
                state = "ON" if cameras[cam_idx]['flipped'] else "OFF"
                print(f"Camera {cam_idx} flip: {state}")
    
    # Cleanup
    for cam in cameras.values():
        cam['cap'].release()
    cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "="*60)
    print("CAMERA SUMMARY")
    print("="*60)
    
    for idx, cam in cameras.items():
        frames = cam['frames']
        det_rate = cam['corners_detected'] / max(frames, 1) * 100
        
        print(f"\n[{idx}] {cam['width']}x{cam['height']}")
        print(f"    Flip needed: {'YES' if cam['flipped'] else 'NO'}")
        print(f"    Board detection rate: {det_rate:.0f}%")
        
        # Quality assessment
        if det_rate > 50:
            print(f"    Quality: GOOD - suitable for MelodicCap")
        elif det_rate > 20:
            print(f"    Quality: OK - may work")
        else:
            print(f"    Quality: POOR - check camera/lighting")
    
    # Suggest best cameras
    good_cams = [idx for idx, cam in cameras.items() 
                 if cam['corners_detected'] / max(cam['frames'], 1) > 0.3]
    
    if len(good_cams) >= 2:
        print(f"\n" + "="*60)
        print("RECOMMENDED SETTINGS FOR melodic_capture_v3.py:")
        print("="*60)
        print(f"    CAM_A_INDEX = {good_cams[0]}")
        print(f"    CAM_B_INDEX = {good_cams[1]}")
        
        flip_a = cameras[good_cams[0]]['flipped']
        flip_b = cameras[good_cams[1]]['flipped']
        print(f"    FLIP_CAM_A_HORIZONTAL = {flip_a}")
        print(f"    FLIP_CAM_B_HORIZONTAL = {flip_b}")
    
    print("\n[DONE]")


if __name__ == "__main__":
    main()