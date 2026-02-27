"""
MelodicCap Calibration v4 - FIXED
==================================
NO FLIP during detection - ArUco markers break when flipped!

The cameras may appear mirrored, but calibration works correctly.
Flip is only applied AFTER calibration for display/recording.

Controls:
  C = Collect calibration frames
  S = Run calibration
  F = Floor calibration  
  Q = Quit
"""

import cv2
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

CAM_A_INDEX = 2          # Sony ZV-1F (use DSHOW)
CAM_B_INDEX = 0          # DroidCam (use CAP_ANY)

# ChArUco board specs (your 63.5mm board)
CHARUCO_COLS = 4
CHARUCO_ROWS = 3
CHARUCO_SQUARE_M = 0.0635   # 63.5mm
CHARUCO_MARKER_M = 0.0476   # 47.6mm

# Output
OUTPUT_DIR = Path(r"C:\Users\ninja\Documents\MelodicCapStudio\MelodicCapFresh\calibration")

# =============================================================================
# DETECTOR SETUP
# =============================================================================

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
    """Detect ChArUco board - NO FLIP applied here!"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    display = frame.copy()
    
    corners, ids, rejected = detector.detectMarkers(gray)
    num_markers = len(ids) if ids is not None else 0
    
    if ids is None or len(ids) < 2:
        return None, None, num_markers, display
    
    cv2.aruco.drawDetectedMarkers(display, corners, ids)
    
    num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    
    if num_corners is None or num_corners < 4:
        return None, None, num_markers, display
    
    cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids,
                                          cornerColor=(0, 255, 0))
    
    return charuco_corners, charuco_ids, num_markers, display


def run_stereo_calibration(frames_a, frames_b):
    """Run stereo calibration"""
    print("\n" + "="*50)
    print("STEREO CALIBRATION")
    print("="*50)
    
    all_corners_a, all_ids_a = [], []
    all_corners_b, all_ids_b = [], []
    valid_pairs = []
    
    for i, (fa, fb) in enumerate(zip(frames_a, frames_b)):
        ca, ia, _, _ = detect_board(fa)
        cb, ib, _, _ = detect_board(fb)
        
        if ca is None or cb is None:
            continue
        
        common = set(ia.flatten()) & set(ib.flatten())
        if len(common) < 4:
            continue
        
        all_corners_a.append(ca)
        all_ids_a.append(ia)
        all_corners_b.append(cb)
        all_ids_b.append(ib)
        valid_pairs.append((ca, ia, cb, ib, common))
    
    print(f"Valid pairs: {len(valid_pairs)}/{len(frames_a)}")
    
    if len(valid_pairs) < 10:
        return None, f"Only {len(valid_pairs)} valid pairs (need 10+)"
    
    img_size = (frames_a[0].shape[1], frames_a[0].shape[0])
    
    # Calibrate Camera A
    print("\nCalibrating Camera A...")
    ret_a, K1, D1, _, _ = cv2.aruco.calibrateCameraCharuco(
        all_corners_a, all_ids_a, board, img_size, None, None
    )
    print(f"  RMS: {ret_a:.4f} {'✓' if ret_a < 1.0 else '!'}")
    
    # Calibrate Camera B
    print("Calibrating Camera B...")
    ret_b, K2, D2, _, _ = cv2.aruco.calibrateCameraCharuco(
        all_corners_b, all_ids_b, board, img_size, None, None
    )
    print(f"  RMS: {ret_b:.4f} {'✓' if ret_b < 1.0 else '!'}")
    
    # Build stereo correspondences
    obj_points, img_points_a, img_points_b = [], [], []
    board_corners = board.getChessboardCorners()
    
    for ca, ia, cb, ib, common in valid_pairs:
        obj_pts, pts_a, pts_b = [], [], []
        ia_flat, ib_flat = ia.flatten(), ib.flatten()
        
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
    print("\nStereo calibration...")
    ret_stereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_a, img_points_b,
        K1, D1, K2, D2, img_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    
    baseline = float(np.linalg.norm(T))
    print(f"  Stereo RMS: {ret_stereo:.4f} {'✓' if ret_stereo < 2.0 else '!'}")
    print(f"  Baseline: {baseline:.3f}m")
    
    if ret_stereo > 10.0:
        return None, f"Stereo RMS {ret_stereo:.1f} too high!"
    
    # Rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / "stereo_calibration.json"
    
    data = {
        'version': '4.0',
        'timestamp': datetime.now().isoformat(),
        'image_size': list(img_size),
        'K1': K1.tolist(), 'D1': D1.tolist(),
        'K2': K2.tolist(), 'D2': D2.tolist(),
        'R1': R1.tolist(), 'R2': R2.tolist(),
        'P1': P1.tolist(), 'P2': P2.tolist(),
        'R': R.tolist(), 'T': T.tolist(),
        'rms_cam_a': ret_a,
        'rms_cam_b': ret_b,
        'rms_stereo': ret_stereo,
        'baseline_meters': baseline,
        'floor_z_offset': 0.0,
        'floor_calibrated': False,
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved to: {filepath}")
    
    return data, f"CamA:{ret_a:.2f} CamB:{ret_b:.2f} Stereo:{ret_stereo:.2f} Base:{baseline:.2f}m"


def calibrate_floor(frame_a, frame_b, calib_data):
    """Calibrate floor from board on ground"""
    ca, ia, _, _ = detect_board(frame_a)
    cb, ib, _, _ = detect_board(frame_b)
    
    if ca is None:
        return False, "Board not detected in Camera A"
    if cb is None:
        return False, "Board not detected in Camera B"
    
    common = set(ia.flatten()) & set(ib.flatten())
    if len(common) < 4:
        return False, f"Only {len(common)} common corners"
    
    # Triangulate floor points
    K1 = np.array(calib_data['K1'])
    D1 = np.array(calib_data['D1'])
    K2 = np.array(calib_data['K2'])
    D2 = np.array(calib_data['D2'])
    R1 = np.array(calib_data['R1'])
    R2 = np.array(calib_data['R2'])
    P1 = np.array(calib_data['P1'])
    P2 = np.array(calib_data['P2'])
    
    pts_a, pts_b = [], []
    ia_flat, ib_flat = ia.flatten(), ib.flatten()
    
    for cid in sorted(common):
        idx_a = np.where(ia_flat == cid)[0][0]
        idx_b = np.where(ib_flat == cid)[0][0]
        pts_a.append(ca[idx_a].flatten())
        pts_b.append(cb[idx_b].flatten())
    
    pts_a = np.array(pts_a, dtype=np.float32).reshape(-1, 1, 2)
    pts_b = np.array(pts_b, dtype=np.float32).reshape(-1, 1, 2)
    
    pts_a_rect = cv2.undistortPoints(pts_a, K1, D1, R=R1, P=P1)
    pts_b_rect = cv2.undistortPoints(pts_b, K2, D2, R=R2, P=P2)
    
    points_4d = cv2.triangulatePoints(P1, P2, 
                                       pts_a_rect.reshape(-1, 2).T,
                                       pts_b_rect.reshape(-1, 2).T)
    points_3d = (points_4d[:3] / points_4d[3]).T
    
    # Convert to Blender coords and get floor offset
    # OpenCV: X=right, Y=down, Z=forward
    # Blender: X=right, Y=forward, Z=up
    blender_z = -points_3d[:, 1]  # -Y_cv = Z_blender
    floor_offset = -np.mean(blender_z)
    
    # Update calibration file
    calib_data['floor_z_offset'] = float(floor_offset)
    calib_data['floor_calibrated'] = True
    
    filepath = OUTPUT_DIR / "stereo_calibration.json"
    with open(filepath, 'w') as f:
        json.dump(calib_data, f, indent=2)
    
    return True, f"Floor offset: {floor_offset:.3f}m"


def main():
    print("\n" + "="*50)
    print("MELODICCAP CALIBRATION v4")
    print("="*50)
    print("\n⚠️  NO FLIP during calibration - ArUco requires this!")
    print("    Cameras may look mirrored - that's OK!")
    
    # Open cameras
    print("\n[CAMERAS]")
    
    print(f"  Camera A (index {CAM_A_INDEX})...")
    cap_a = cv2.VideoCapture(CAM_A_INDEX, cv2.CAP_DSHOW)
    if cap_a.isOpened():
        cap_a.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap_a.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        w, h = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"    OK ({w}x{h})")
    else:
        print("    FAILED!")
        return
    
    print(f"  Camera B (index {CAM_B_INDEX})...")
    cap_b = cv2.VideoCapture(CAM_B_INDEX, cv2.CAP_ANY)
    if cap_b.isOpened():
        cap_b.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap_b.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        w, h = int(cap_b.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_b.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"    OK ({w}x{h})")
    else:
        print("    FAILED!")
        return
    
    print("\n[CONTROLS]")
    print("  C = Collect calibration frames")
    print("  S = Run stereo calibration")
    print("  F = Floor calibration (lay board flat)")
    print("  Q = Quit")
    
    # State
    cal_frames_a = []
    cal_frames_b = []
    collecting = False
    floor_mode = False
    last_capture = 0
    status = "Press C to start calibration"
    calib_data = None
    
    # Try to load existing calibration
    calib_file = OUTPUT_DIR / "stereo_calibration.json"
    if calib_file.exists():
        try:
            with open(calib_file) as f:
                calib_data = json.load(f)
            status = f"Loaded calibration (Stereo RMS: {calib_data.get('rms_stereo', '?'):.2f})"
            print(f"\n[LOADED] {status}")
        except:
            pass
    
    while True:
        ret_a, frame_a = cap_a.read()
        ret_b, frame_b = cap_b.read()
        
        if not ret_a or not ret_b:
            continue
        
        # NO FLIP! Detect boards on original frames
        corners_a, ids_a, markers_a, display_a = detect_board(frame_a)
        corners_b, ids_b, markers_b, display_b = detect_board(frame_b)
        
        board_a = corners_a is not None
        board_b = corners_b is not None
        
        # Draw camera labels
        h_a, w_a = display_a.shape[:2]
        h_b, w_b = display_b.shape[:2]
        
        cv2.rectangle(display_a, (0, 0), (220, 35), (0, 0, 0), -1)
        cv2.putText(display_a, f"CAM A | {markers_a}m", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.rectangle(display_b, (0, 0), (220, 35), (0, 0, 0), -1)
        cv2.putText(display_b, f"CAM B | {markers_b}m", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Detection status borders
        if collecting or floor_mode:
            if board_a:
                cv2.rectangle(display_a, (0, 0), (w_a-1, h_a-1), (0, 255, 0), 10)
                cv2.putText(display_a, f"OK: {len(ids_a)} corners", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.rectangle(display_a, (0, 0), (w_a-1, h_a-1), (0, 0, 255), 10)
            
            if board_b:
                cv2.rectangle(display_b, (0, 0), (w_b-1, h_b-1), (0, 255, 0), 10)
                cv2.putText(display_b, f"OK: {len(ids_b)} corners", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.rectangle(display_b, (0, 0), (w_b-1, h_b-1), (0, 0, 255), 10)
        
        # Capture when both boards visible
        if collecting and board_a and board_b:
            now = time.time()
            if now - last_capture > 0.5:
                common = set(ids_a.flatten()) & set(ids_b.flatten())
                if len(common) >= 4:
                    cal_frames_a.append(frame_a.copy())
                    cal_frames_b.append(frame_b.copy())
                    last_capture = now
                    status = f"Captured {len(cal_frames_a)} ({len(common)} common)"
                    print(f"[CAPTURE] Frame {len(cal_frames_a)}")
        
        # Floor calibration
        if floor_mode and board_a and board_b and calib_data:
            success, msg = calibrate_floor(frame_a, frame_b, calib_data)
            if success:
                floor_mode = False
                status = f"FLOOR OK: {msg}"
                print(f"[FLOOR] {msg}")
            else:
                status = f"Floor: {msg}"
        
        # Resize and combine
        scale = 400 / display_a.shape[0]
        disp_a = cv2.resize(display_a, None, fx=scale, fy=scale)
        disp_b = cv2.resize(display_b, None, fx=scale, fy=scale)
        combined = np.hstack([disp_a, disp_b])
        
        # Status bar
        bar = np.zeros((45, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(bar, f"Frames: {len(cal_frames_a)}", (combined.shape[1]-150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        combined = np.vstack([combined, bar])
        cv2.imshow("MelodicCap Calibration v4", combined)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('c'):
            if collecting:
                collecting = False
                status = f"Stopped. {len(cal_frames_a)} frames. Press S."
            else:
                cal_frames_a = []
                cal_frames_b = []
                collecting = True
                floor_mode = False
                status = "COLLECTING - Show board to BOTH cameras"
        
        elif key == ord('s'):
            if len(cal_frames_a) < 10:
                status = f"Need 10+ frames (have {len(cal_frames_a)})"
            else:
                collecting = False
                status = "Running calibration..."
                cv2.waitKey(100)
                
                result, msg = run_stereo_calibration(cal_frames_a, cal_frames_b)
                if result:
                    calib_data = result
                    status = f"SUCCESS! {msg}"
                    print(f"\n{'='*50}")
                    print(f"CALIBRATION SUCCESS!")
                    print(f"  {msg}")
                    print(f"{'='*50}")
                else:
                    status = f"FAILED: {msg}"
                    print(f"\n[FAILED] {msg}")
        
        elif key == ord('f'):
            if calib_data is None:
                status = "Calibrate cameras first (C, then S)"
            else:
                floor_mode = not floor_mode
                if floor_mode:
                    status = "FLOOR MODE - Lay board flat on ground"
                else:
                    status = "Floor mode cancelled"
    
    cap_a.release()
    cap_b.release()
    cv2.destroyAllWindows()
    print("\n[DONE]")


if __name__ == "__main__":
    main()