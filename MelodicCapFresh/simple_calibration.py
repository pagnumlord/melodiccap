"""
MelodicCap Simple Calibration
=============================
Simplified version focusing ONLY on calibration.
Uses same detection code as the working diagnostic.

Controls:
  C = Collect calibration frames
  S = Run calibration
  F = Floor calibration
  G = Toggle Camera A flip
  H = Toggle Camera B flip
  Q = Quit
"""

import cv2
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION - EDIT THESE
# =============================================================================

CAM_A_INDEX = 2          # Sony ZV-1F
CAM_B_INDEX = 0          # DroidCam (direct, not OBS)

# Flip settings - both cameras are mirrored
FLIP_CAM_A = True
FLIP_CAM_B = True

# ChArUco board (your 63.5mm calibration board)
CHARUCO_COLS = 4
CHARUCO_ROWS = 3
CHARUCO_SQUARE_M = 0.0635   # 63.5mm
CHARUCO_MARKER_M = 0.0476   # 47.6mm

# Output path
OUTPUT_DIR = Path(r"C:\Users\ninja\Documents\MelodicCapStudio\MelodicCapFresh\calibration")

# =============================================================================
# SETUP
# =============================================================================

# Create ArUco detector (same as working diagnostic)
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


def detect_board(frame):
    """Detect ChArUco board - returns (corners, ids, num_markers, display_frame)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    display = frame.copy()
    
    # Detect ArUco markers
    corners, ids, rejected = detector.detectMarkers(gray)
    
    num_markers = len(ids) if ids is not None else 0
    
    if ids is None or len(ids) < 2:
        return None, None, num_markers, display
    
    # Draw detected markers
    cv2.aruco.drawDetectedMarkers(display, corners, ids)
    
    # Interpolate ChArUco corners
    num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    
    if num_corners is None or num_corners < 4:
        return None, None, num_markers, display
    
    # Draw ChArUco corners
    cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids,
                                          cornerColor=(0, 255, 0))
    
    return charuco_corners, charuco_ids, num_markers, display


def run_calibration(frames_a, frames_b):
    """Run stereo calibration on collected frames"""
    print("\n" + "="*50)
    print("RUNNING STEREO CALIBRATION")
    print("="*50)
    
    # Collect valid frame pairs
    all_corners_a = []
    all_ids_a = []
    all_corners_b = []
    all_ids_b = []
    valid_pairs = []
    
    for i, (fa, fb) in enumerate(zip(frames_a, frames_b)):
        ca, ia, _, _ = detect_board(fa)
        cb, ib, _, _ = detect_board(fb)
        
        if ca is None or cb is None:
            continue
        
        # Find common corners
        common = set(ia.flatten()) & set(ib.flatten())
        if len(common) < 4:
            continue
        
        all_corners_a.append(ca)
        all_ids_a.append(ia)
        all_corners_b.append(cb)
        all_ids_b.append(ib)
        valid_pairs.append((ca, ia, cb, ib, common))
    
    print(f"Valid frame pairs: {len(valid_pairs)}/{len(frames_a)}")
    
    if len(valid_pairs) < 10:
        return None, f"Only {len(valid_pairs)} valid pairs. Need 10+."
    
    # Get image size from first frame
    img_size = (frames_a[0].shape[1], frames_a[0].shape[0])
    
    # Calibrate Camera A
    print("\nCalibrating Camera A...")
    ret_a, K1, D1, _, _ = cv2.aruco.calibrateCameraCharuco(
        all_corners_a, all_ids_a, board, img_size, None, None
    )
    print(f"  RMS: {ret_a:.4f}")
    
    # Calibrate Camera B
    print("Calibrating Camera B...")
    ret_b, K2, D2, _, _ = cv2.aruco.calibrateCameraCharuco(
        all_corners_b, all_ids_b, board, img_size, None, None
    )
    print(f"  RMS: {ret_b:.4f}")
    
    # Prepare stereo calibration
    obj_points = []
    img_points_a = []
    img_points_b = []
    board_corners = board.getChessboardCorners()
    
    for ca, ia, cb, ib, common in valid_pairs:
        obj_pts = []
        pts_a = []
        pts_b = []
        
        ia_flat = ia.flatten()
        ib_flat = ib.flatten()
        
        for cid in sorted(common):
            obj_pts.append(board_corners[cid])
            idx_a = np.where(ia_flat == cid)[0][0]
            idx_b = np.where(ib_flat == cid)[0][0]
            pts_a.append(ca[idx_a].flatten())
            pts_b.append(cb[idx_b].flatten())
        
        obj_points.append(np.array(obj_pts, dtype=np.float32))
        img_points_a.append(np.array(pts_a, dtype=np.float32))
        img_points_b.append(np.array(pts_b, dtype=np.float32))
    
    # Stereo calibration
    print("\nRunning stereo calibration...")
    ret_stereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_a, img_points_b,
        K1, D1, K2, D2, img_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    
    baseline = float(np.linalg.norm(T))
    print(f"  Stereo RMS: {ret_stereo:.4f}")
    print(f"  Baseline: {baseline:.3f}m")
    
    if ret_stereo > 5.0:
        return None, f"Stereo RMS {ret_stereo:.1f} too high!"
    
    # Rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    
    # Save calibration
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / "stereo_calibration.json"
    
    data = {
        'timestamp': datetime.now().isoformat(),
        'image_size': list(img_size),
        'K1': K1.tolist(), 'D1': D1.tolist(),
        'K2': K2.tolist(), 'D2': D2.tolist(),
        'R1': R1.tolist(), 'R2': R2.tolist(),
        'P1': P1.tolist(), 'P2': P2.tolist(),
        'R': R.tolist(), 'T': T.tolist(),
        'rms_intrinsic_a': ret_a,
        'rms_intrinsic_b': ret_b,
        'rms_stereo': ret_stereo,
        'baseline_meters': baseline,
        'floor_z_offset': 0.0,
        'floor_calibrated': False,
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return data, f"OK! CamA:{ret_a:.2f} CamB:{ret_b:.2f} Stereo:{ret_stereo:.2f} Baseline:{baseline:.2f}m"


def main():
    global FLIP_CAM_A, FLIP_CAM_B
    
    print("\n" + "="*50)
    print("MELODICCAP SIMPLE CALIBRATION")
    print("="*50)
    
    # Open cameras
    print("\n[CAMERAS]")
    
    # Camera A - Sony with DSHOW
    print(f"  Opening Camera A (index {CAM_A_INDEX})...")
    cap_a = cv2.VideoCapture(CAM_A_INDEX, cv2.CAP_DSHOW)
    if cap_a.isOpened():
        cap_a.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap_a.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        w = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"    OK ({w}x{h})")
    else:
        print("    FAILED!")
        return
    
    # Camera B - DroidCam with CAP_ANY
    print(f"  Opening Camera B (index {CAM_B_INDEX})...")
    cap_b = cv2.VideoCapture(CAM_B_INDEX, cv2.CAP_ANY)
    if cap_b.isOpened():
        cap_b.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap_b.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        w = int(cap_b.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_b.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"    OK ({w}x{h})")
    else:
        print("    FAILED!")
        return
    
    print("\n[CONTROLS]")
    print("  C = Collect calibration frames")
    print("  S = Run stereo calibration")
    print("  G = Toggle Camera A flip")
    print("  H = Toggle Camera B flip")
    print("  Q = Quit")
    print(f"\n[FLIPS] A: {'ON' if FLIP_CAM_A else 'OFF'}, B: {'ON' if FLIP_CAM_B else 'OFF'}")
    
    # State
    cal_frames_a = []
    cal_frames_b = []
    collecting = False
    last_capture = 0
    status = "Press C to start calibration"
    
    while True:
        ret_a, frame_a = cap_a.read()
        ret_b, frame_b = cap_b.read()
        
        if not ret_a or not ret_b:
            continue
        
        # Apply flips
        if FLIP_CAM_A:
            frame_a = cv2.flip(frame_a, 1)
        if FLIP_CAM_B:
            frame_b = cv2.flip(frame_b, 1)
        
        # Detect boards
        corners_a, ids_a, markers_a, display_a = detect_board(frame_a)
        corners_b, ids_b, markers_b, display_b = detect_board(frame_b)
        
        board_a = corners_a is not None
        board_b = corners_b is not None
        
        # Draw status on frames
        h_a, w_a = display_a.shape[:2]
        h_b, w_b = display_b.shape[:2]
        
        # Camera labels
        cv2.rectangle(display_a, (0, 0), (200, 35), (0, 0, 0), -1)
        cv2.putText(display_a, f"CAM A ({markers_a}m)", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.rectangle(display_b, (0, 0), (200, 35), (0, 0, 0), -1)
        cv2.putText(display_b, f"CAM B ({markers_b}m)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Board detection borders
        if collecting:
            if board_a:
                cv2.rectangle(display_a, (0, 0), (w_a-1, h_a-1), (0, 255, 0), 10)
                cv2.putText(display_a, f"OK: {len(ids_a)} corners", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.rectangle(display_a, (0, 0), (w_a-1, h_a-1), (0, 0, 255), 10)
                cv2.putText(display_a, "NO BOARD", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if board_b:
                cv2.rectangle(display_b, (0, 0), (w_b-1, h_b-1), (0, 255, 0), 10)
                cv2.putText(display_b, f"OK: {len(ids_b)} corners", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.rectangle(display_b, (0, 0), (w_b-1, h_b-1), (0, 0, 255), 10)
                cv2.putText(display_b, "NO BOARD", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Capture frames when both boards visible
        if collecting and board_a and board_b:
            now = time.time()
            if now - last_capture > 0.5:
                common = set(ids_a.flatten()) & set(ids_b.flatten())
                if len(common) >= 4:
                    cal_frames_a.append(frame_a.copy())
                    cal_frames_b.append(frame_b.copy())
                    last_capture = now
                    status = f"Captured {len(cal_frames_a)} frames ({len(common)} common)"
                    print(f"[CAPTURE] {status}")
        
        # Resize and combine for display
        scale = 400 / display_a.shape[0]
        disp_a = cv2.resize(display_a, None, fx=scale, fy=scale)
        disp_b = cv2.resize(display_b, None, fx=scale, fy=scale)
        combined = np.hstack([disp_a, disp_b])
        
        # Status bar
        bar_h = 40
        bar = np.zeros((bar_h, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(bar, f"Frames: {len(cal_frames_a)}", (combined.shape[1]-150, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        combined = np.vstack([combined, bar])
        
        cv2.imshow("MelodicCap Calibration", combined)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('c'):
            if collecting:
                collecting = False
                status = f"Stopped. {len(cal_frames_a)} frames. Press S to calibrate."
            else:
                cal_frames_a = []
                cal_frames_b = []
                collecting = True
                status = "COLLECTING - Show board to BOTH cameras"
        
        elif key == ord('s'):
            if len(cal_frames_a) < 10:
                status = f"Need 10+ frames (have {len(cal_frames_a)})"
            else:
                collecting = False
                status = "Running calibration..."
                cv2.waitKey(100)
                
                result, msg = run_calibration(cal_frames_a, cal_frames_b)
                if result:
                    status = f"SUCCESS! {msg}"
                    print(f"\n[SUCCESS] {msg}")
                else:
                    status = f"FAILED: {msg}"
                    print(f"\n[FAILED] {msg}")
        
        elif key == ord('g'):
            FLIP_CAM_A = not FLIP_CAM_A
            status = f"Camera A flip: {'ON' if FLIP_CAM_A else 'OFF'}"
        
        elif key == ord('h'):
            FLIP_CAM_B = not FLIP_CAM_B
            status = f"Camera B flip: {'ON' if FLIP_CAM_B else 'OFF'}"
    
    cap_a.release()
    cap_b.release()
    cv2.destroyAllWindows()
    print("\n[DONE]")


if __name__ == "__main__":
    main()
