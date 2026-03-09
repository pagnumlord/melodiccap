"""
MelodicCap Robust Stereo Calibration
=====================================
Uses iterative outlier rejection to get clean calibration.

This approach:
1. Collects frames
2. Does initial calibration
3. Calculates per-frame stereo reprojection error
4. Removes worst frames
5. Recalibrates
6. Repeats until stable

This is essentially RANSAC for stereo calibration.
"""

import cv2
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path

# Configuration
CAM_A_INDEX = 3          # Sony ZV-1F (MSMF)
CAM_B_INDEX = 0          # DroidCam USB

# Match these to generate_boards.py output — MEASURE printed squares with ruler!
# Option A (single sheet): COLS=5, ROWS=7, SQUARE=0.035, MARKER=0.025
# Option B (dual sheet):   COLS=10, ROWS=5, SQUARE=0.043, MARKER=0.031
CHARUCO_COLS = 5
CHARUCO_ROWS = 7
CHARUCO_SQUARE_M = 0.035
CHARUCO_MARKER_M = 0.025

BASE_DIR = Path(__file__).parent.parent  # MelodicCapStudio/
OUTPUT_FILE = BASE_DIR / "calibration" / "stereo_calibration.json"

# Setup
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
board_corners_3d = board.getChessboardCorners()


def detect_charuco(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is None or len(ids) < 2:
        return None, None, 0
    
    num, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    
    if num is None or num < 4:
        return None, None, len(ids)
    
    return charuco_corners, charuco_ids, len(ids)


def robust_stereo_calibration(frames_a, frames_b, max_iterations=5, outlier_threshold=2.0):
    """
    Iteratively calibrate, removing outliers until stable.
    """
    print("\n" + "="*60)
    print("ROBUST STEREO CALIBRATION")
    print("="*60)
    
    # Extract corners from all frames
    all_data = []
    for i, (fa, fb) in enumerate(zip(frames_a, frames_b)):
        cc_a, ci_a, _ = detect_charuco(fa)
        cc_b, ci_b, _ = detect_charuco(fb)
        
        if cc_a is None or cc_b is None:
            continue
        
        common = set(ci_a.flatten()) & set(ci_b.flatten())
        if len(common) < 4:
            continue
        
        all_data.append({
            'index': i,
            'corners_a': cc_a,
            'ids_a': ci_a,
            'corners_b': cc_b,
            'ids_b': ci_b,
            'common': common,
            'active': True
        })
    
    print(f"\nValid frame pairs: {len(all_data)}")
    
    if len(all_data) < 8:
        return None, "Not enough valid frames"
    
    img_size = (frames_a[0].shape[1], frames_a[0].shape[0])
    
    best_result = None
    best_rms = float('inf')
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # Get active frames
        active_data = [d for d in all_data if d['active']]
        print(f"Active frames: {len(active_data)}")
        
        if len(active_data) < 8:
            print("Too few frames remaining!")
            break
        
        # Collect corners for calibration
        corners_a = [d['corners_a'] for d in active_data]
        ids_a = [d['ids_a'] for d in active_data]
        corners_b = [d['corners_b'] for d in active_data]
        ids_b = [d['ids_b'] for d in active_data]
        
        # Calibrate cameras individually
        ret_a, K1, D1, rvecs_a, tvecs_a = cv2.aruco.calibrateCameraCharuco(
            corners_a, ids_a, board, img_size, None, None
        )
        
        ret_b, K2, D2, rvecs_b, tvecs_b = cv2.aruco.calibrateCameraCharuco(
            corners_b, ids_b, board, img_size, None, None
        )
        
        print(f"  Camera A RMS: {ret_a:.4f}")
        print(f"  Camera B RMS: {ret_b:.4f}")
        
        # Build stereo correspondences
        obj_points = []
        img_points_a = []
        img_points_b = []
        frame_indices = []
        
        for d in active_data:
            cc_a, ci_a = d['corners_a'], d['ids_a']
            cc_b, ci_b = d['corners_b'], d['ids_b']
            common = d['common']
            
            obj_pts = []
            pts_a = []
            pts_b = []
            
            ci_a_flat = ci_a.flatten()
            ci_b_flat = ci_b.flatten()
            
            for cid in sorted(common):
                obj_pts.append(board_corners_3d[cid])
                idx_a = np.where(ci_a_flat == cid)[0][0]
                idx_b = np.where(ci_b_flat == cid)[0][0]
                pts_a.append(cc_a[idx_a].flatten())
                pts_b.append(cc_b[idx_b].flatten())
            
            obj_points.append(np.array(obj_pts, dtype=np.float32))
            img_points_a.append(np.array(pts_a, dtype=np.float32))
            img_points_b.append(np.array(pts_b, dtype=np.float32))
            frame_indices.append(d['index'])
        
        # Stereo calibration
        ret_stereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            obj_points, img_points_a, img_points_b,
            K1, D1, K2, D2, img_size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
            flags=cv2.CALIB_FIX_INTRINSIC
        )
        
        baseline = np.linalg.norm(T)
        print(f"  Stereo RMS: {ret_stereo:.4f}")
        print(f"  Baseline: {baseline:.3f}m")
        
        # Check if this is our best result
        if ret_stereo < best_rms:
            best_rms = ret_stereo
            best_result = {
                'K1': K1, 'D1': D1, 'K2': K2, 'D2': D2,
                'R': R, 'T': T,
                'ret_a': ret_a, 'ret_b': ret_b, 'ret_stereo': ret_stereo,
                'baseline': baseline,
                'num_frames': len(active_data)
            }
        
        # If stereo RMS is good, we're done!
        if ret_stereo < 2.0:
            print(f"\n✓ Stereo RMS {ret_stereo:.4f} is acceptable!")
            break
        
        # Calculate per-frame reprojection error
        print("\n  Calculating per-frame errors...")
        
        # Get rectification matrices
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K1, D1, K2, D2, img_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        
        frame_errors = []
        for i, (obj_pts, pts_a, pts_b) in enumerate(zip(obj_points, img_points_a, img_points_b)):
            # Undistort and rectify points
            pts_a_rect = cv2.undistortPoints(pts_a.reshape(-1, 1, 2), K1, D1, R=R1, P=P1)
            pts_b_rect = cv2.undistortPoints(pts_b.reshape(-1, 1, 2), K2, D2, R=R2, P=P2)
            
            # Triangulate
            pts_4d = cv2.triangulatePoints(P1, P2, 
                                           pts_a_rect.reshape(-1, 2).T,
                                           pts_b_rect.reshape(-1, 2).T)
            pts_3d = (pts_4d[:3] / pts_4d[3]).T
            
            # Reproject to both cameras
            pts_3d_h = np.hstack([pts_3d, np.ones((pts_3d.shape[0], 1))])
            
            # Project to camera A (at origin)
            proj_a = (P1 @ pts_3d_h.T).T
            proj_a = proj_a[:, :2] / proj_a[:, 2:3]
            
            # Project to camera B
            proj_b = (P2 @ pts_3d_h.T).T
            proj_b = proj_b[:, :2] / proj_b[:, 2:3]
            
            # Calculate reprojection error
            err_a = np.mean(np.linalg.norm(pts_a_rect.reshape(-1, 2) - proj_a, axis=1))
            err_b = np.mean(np.linalg.norm(pts_b_rect.reshape(-1, 2) - proj_b, axis=1))
            
            frame_errors.append({
                'frame_idx': frame_indices[i],
                'error': (err_a + err_b) / 2,
                'err_a': err_a,
                'err_b': err_b
            })
        
        # Sort by error
        frame_errors.sort(key=lambda x: x['error'], reverse=True)
        
        # Find frames with error significantly above median
        errors = [f['error'] for f in frame_errors]
        median_error = np.median(errors)
        
        print(f"  Median frame error: {median_error:.4f}")
        print(f"  Worst frame error: {frame_errors[0]['error']:.4f}")
        
        # Remove worst frames (those > outlier_threshold * median)
        threshold = max(outlier_threshold * median_error, 1.0)
        removed = 0
        
        for fe in frame_errors:
            if fe['error'] > threshold and len(active_data) - removed > 10:
                # Mark as inactive
                for d in all_data:
                    if d['index'] == fe['frame_idx']:
                        d['active'] = False
                        removed += 1
                        print(f"    Removed frame {fe['frame_idx']} (error: {fe['error']:.4f})")
                        break
        
        if removed == 0:
            print("  No outliers found, stopping iteration")
            break
    
    return best_result, f"Best Stereo RMS: {best_rms:.4f}"


def main():
    print("\n" + "="*60)
    print("MELODICCAP ROBUST CALIBRATION")
    print("="*60)
    
    # Open cameras
    print("\nOpening cameras...")
    cap_a = cv2.VideoCapture(CAM_A_INDEX, cv2.CAP_DSHOW)
    cap_a.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap_a.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap_a.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cap_b = cv2.VideoCapture(CAM_B_INDEX, cv2.CAP_ANY)
    cap_b.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap_b.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap_b.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap_a.isOpened() or not cap_b.isOpened():
        print("ERROR: Could not open cameras!")
        return

    native_w_a = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h_a = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
    native_w_b = int(cap_b.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h_b = int(cap_b.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera A: {native_w_a}x{native_h_a}")
    print(f"Camera B: {native_w_b}x{native_h_b}")

    # Use smaller camera's resolution to preserve aspect ratio
    pixels_a = native_w_a * native_h_a
    pixels_b = native_w_b * native_h_b
    if pixels_a <= pixels_b:
        proc_w, proc_h = native_w_a, native_h_a
    else:
        proc_w, proc_h = native_w_b, native_h_b
    resize_a = (native_w_a != proc_w or native_h_a != proc_h)
    resize_b = (native_w_b != proc_w or native_h_b != proc_h)
    print(f"Processing at: {proc_w}x{proc_h}")
    if resize_a:
        print(f"  Camera A: will resize {native_w_a}x{native_h_a} -> {proc_w}x{proc_h}")
    if resize_b:
        print(f"  Camera B: will resize {native_w_b}x{native_h_b} -> {proc_w}x{proc_h}")
    
    print("\n[CONTROLS]")
    print("  Press SPACE to start/stop collecting")
    print("  Press S to run calibration")
    print("  Press Q to quit")
    
    frames_a = []
    frames_b = []
    collecting = False
    last_capture = 0
    status = "Press SPACE to start collecting"
    
    while True:
        ret_a, frame_a = cap_a.read()
        ret_b, frame_b = cap_b.read()

        if not ret_a or not ret_b:
            continue

        # Resize to common resolution if needed
        if resize_a:
            frame_a = cv2.resize(frame_a, (proc_w, proc_h))
        if resize_b:
            frame_b = cv2.resize(frame_b, (proc_w, proc_h))

        # Detect
        cc_a, ci_a, markers_a = detect_charuco(frame_a)
        cc_b, ci_b, markers_b = detect_charuco(frame_b)
        
        board_a = cc_a is not None
        board_b = cc_b is not None
        
        # Draw
        disp_a = frame_a.copy()
        disp_b = frame_b.copy()
        
        if board_a:
            gray = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
            c, i, _ = detector.detectMarkers(gray)
            cv2.aruco.drawDetectedMarkers(disp_a, c, i)
            cv2.aruco.drawDetectedCornersCharuco(disp_a, cc_a, ci_a)
        
        if board_b:
            gray = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
            c, i, _ = detector.detectMarkers(gray)
            cv2.aruco.drawDetectedMarkers(disp_b, c, i)
            cv2.aruco.drawDetectedCornersCharuco(disp_b, cc_b, ci_b)
        
        # Auto-capture
        if collecting and board_a and board_b:
            now = time.time()
            if now - last_capture > 0.5:
                common = set(ci_a.flatten()) & set(ci_b.flatten())
                if len(common) >= 4:
                    frames_a.append(frame_a.copy())
                    frames_b.append(frame_b.copy())
                    last_capture = now
                    print(f"  Frame {len(frames_a)}: {len(common)} common corners")
        
        # Draw status
        h, w = disp_a.shape[:2]
        
        color_a = (0, 255, 0) if board_a else (0, 0, 255)
        cv2.rectangle(disp_a, (0, 0), (200, 40), (0, 0, 0), -1)
        cv2.putText(disp_a, f"CAM A: {markers_a}m", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_a, 2)
        
        color_b = (0, 255, 0) if board_b else (0, 0, 255)
        cv2.rectangle(disp_b, (0, 0), (200, 40), (0, 0, 0), -1)
        cv2.putText(disp_b, f"CAM B: {markers_b}m", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_b, 2)
        
        if collecting:
            cv2.rectangle(disp_a, (0, 0), (w-1, h-1), (0, 255, 0), 5)
            cv2.rectangle(disp_b, (0, 0), (w-1, h-1), (0, 255, 0), 5)
        
        disp_a = cv2.resize(disp_a, (640, 360))
        disp_b = cv2.resize(disp_b, (640, 360))
        combined = np.hstack([disp_a, disp_b])
        
        bar = np.zeros((45, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, f"{status} | Frames: {len(frames_a)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        combined = np.vstack([combined, bar])
        
        cv2.imshow("Robust Calibration", combined)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord(' '):
            if collecting:
                collecting = False
                status = f"Collected {len(frames_a)} frames. Press S to calibrate."
            else:
                frames_a = []
                frames_b = []
                collecting = True
                status = "COLLECTING..."
        
        elif key == ord('s'):
            if len(frames_a) < 10:
                status = f"Need 10+ frames (have {len(frames_a)})"
            else:
                collecting = False
                status = "Running robust calibration..."
                cv2.destroyAllWindows()
                
                result, msg = robust_stereo_calibration(frames_a, frames_b)
                
                if result and result['ret_stereo'] < 10.0:
                    print("\n" + "="*60)
                    print("CALIBRATION RESULTS")
                    print("="*60)
                    print(f"  Camera A RMS: {result['ret_a']:.4f}")
                    print(f"  Camera B RMS: {result['ret_b']:.4f}")
                    print(f"  Stereo RMS: {result['ret_stereo']:.4f}")
                    print(f"  Baseline: {result['baseline']:.3f}m")
                    print(f"  Frames used: {result['num_frames']}")
                    
                    if result['ret_stereo'] < 3.0:
                        # Save calibration
                        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
                        
                        img_size = (frames_a[0].shape[1], frames_a[0].shape[0])
                        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                            result['K1'], result['D1'], result['K2'], result['D2'],
                            img_size, result['R'], result['T'],
                            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
                        )
                        
                        data = {
                            'version': '5.3-robust',
                            'timestamp': datetime.now().isoformat(),
                            'image_size': list(img_size),
                            'K1': result['K1'].tolist(), 'D1': result['D1'].tolist(),
                            'K2': result['K2'].tolist(), 'D2': result['D2'].tolist(),
                            'R1': R1.tolist(), 'R2': R2.tolist(),
                            'P1': P1.tolist(), 'P2': P2.tolist(),
                            'R': result['R'].tolist(), 'T': result['T'].tolist(),
                            'rms_cam_a': float(result['ret_a']),
                            'rms_cam_b': float(result['ret_b']),
                            'rms_stereo': float(result['ret_stereo']),
                            'baseline_meters': float(result['baseline']),
                            'floor_z_offset': 0.0,
                            'floor_calibrated': False,
                            'frames_used': result['num_frames'],
                        }
                        
                        with open(OUTPUT_FILE, 'w') as f:
                            json.dump(data, f, indent=2)
                        
                        print(f"\n✅ Saved to: {OUTPUT_FILE}")
                        status = f"SUCCESS! Stereo RMS: {result['ret_stereo']:.2f}"
                    else:
                        status = f"RMS {result['ret_stereo']:.1f} still high - try again"
                else:
                    status = f"FAILED: {msg}"
                
                # Reopen window
                cv2.namedWindow("Robust Calibration")
    
    cap_a.release()
    cap_b.release()
    cv2.destroyAllWindows()
    print("\n[DONE]")


if __name__ == "__main__":
    main()
