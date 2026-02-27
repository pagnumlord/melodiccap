"""
MelodicCap Calibration Deep Diagnostic
=======================================
Analyzes frame quality to find what's causing calibration failures.

This tool:
1. Verifies board dimensions
2. Captures frames and scores their quality
3. Shows per-frame reprojection error
4. Filters out bad frames automatically
5. Identifies the root cause of failures

Run this before tomorrow's test to understand your setup!
"""

import cv2
import numpy as np
import time
from datetime import datetime

# =============================================================================
# BOARD CONFIGURATION - VERIFY THIS IS CORRECT!
# =============================================================================

# Your board: 2.5 inch squares
# 2.5 inches = 63.5mm = 0.0635m
CHARUCO_SQUARE_M = 0.0635   # Square size in meters

# Marker is 75% of square size
# 63.5mm * 0.75 = 47.625mm ≈ 47.6mm
CHARUCO_MARKER_M = 0.0476   # Marker size in meters

# Board layout
CHARUCO_COLS = 4  # Squares horizontally
CHARUCO_ROWS = 3  # Squares vertically

# Cameras
CAM_A_INDEX = 2   # Sony
CAM_B_INDEX = 0   # DroidCam

print("="*70)
print("MELODICCAP CALIBRATION DEEP DIAGNOSTIC")
print("="*70)

print("\n[BOARD CONFIGURATION]")
print(f"  Squares: {CHARUCO_COLS}x{CHARUCO_ROWS}")
print(f"  Square size: {CHARUCO_SQUARE_M*1000:.1f}mm ({CHARUCO_SQUARE_M*1000/25.4:.2f} inches)")
print(f"  Marker size: {CHARUCO_MARKER_M*1000:.1f}mm ({CHARUCO_MARKER_M*1000/25.4:.2f} inches)")
print(f"  Marker/Square ratio: {CHARUCO_MARKER_M/CHARUCO_SQUARE_M*100:.1f}%")

# Verify with user
print("\n⚠️  VERIFY YOUR BOARD MATCHES THESE DIMENSIONS!")
print("   If your squares are different, update the values above.")
input("\nPress Enter to continue (or Ctrl+C to exit and fix dimensions)...")

# =============================================================================
# SETUP
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
board_corners_3d = board.getChessboardCorners()

def detect_charuco(frame):
    """Detect ChArUco and return detailed info"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is None or len(ids) < 2:
        return None, None, 0, len(rejected) if rejected else 0
    
    num, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    
    if num is None or num < 4:
        return None, None, len(ids), len(rejected) if rejected else 0
    
    return charuco_corners, charuco_ids, len(ids), len(rejected) if rejected else 0


def calculate_reprojection_error(corners, ids, K, D, rvec, tvec):
    """Calculate per-corner reprojection error"""
    obj_pts = np.array([board_corners_3d[i] for i in ids.flatten()], dtype=np.float32)
    projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
    
    errors = []
    for i, (corner, proj) in enumerate(zip(corners, projected)):
        err = np.linalg.norm(corner.flatten() - proj.flatten())
        errors.append(err)
    
    return errors


def calibrate_single_camera(corners_list, ids_list, img_size, camera_name):
    """Calibrate single camera and return per-frame errors"""
    print(f"\n  Calibrating {camera_name}...")
    
    ret, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        corners_list, ids_list, board, img_size, None, None
    )
    
    # Calculate per-frame reprojection error
    frame_errors = []
    for i, (corners, ids, rvec, tvec) in enumerate(zip(corners_list, ids_list, rvecs, tvecs)):
        errors = calculate_reprojection_error(corners, ids, K, D, rvec, tvec)
        frame_rms = np.sqrt(np.mean(np.array(errors)**2))
        frame_errors.append(frame_rms)
    
    return ret, K, D, rvecs, tvecs, frame_errors


# =============================================================================
# OPEN CAMERAS
# =============================================================================

print("\n[OPENING CAMERAS]")

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
    exit(1)

w_a = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
h_a = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
w_b = int(cap_b.get(cv2.CAP_PROP_FRAME_WIDTH))
h_b = int(cap_b.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"  Camera A (Sony): {w_a}x{h_a}")
print(f"  Camera B (DroidCam): {w_b}x{h_b}")

# =============================================================================
# LIVE QUALITY CHECK
# =============================================================================

print("\n" + "="*70)
print("PHASE 1: LIVE QUALITY CHECK")
print("="*70)
print("\nThis shows live detection quality on both cameras.")
print("Watch for:")
print("  - Consistent corner detection (6 corners on 4x3 board)")
print("  - Stable detection (not flickering)")
print("  - Similar detection in both cameras")
print("\nPress SPACE to start collecting frames, Q to quit")

while True:
    ret_a, frame_a = cap_a.read()
    ret_b, frame_b = cap_b.read()
    
    if not ret_a or not ret_b:
        continue
    
    # Detect
    cc_a, ci_a, markers_a, rej_a = detect_charuco(frame_a)
    cc_b, ci_b, markers_b, rej_b = detect_charuco(frame_b)
    
    corners_a = len(ci_a) if ci_a is not None else 0
    corners_b = len(ci_b) if ci_b is not None else 0
    
    # Draw
    disp_a = frame_a.copy()
    disp_b = frame_b.copy()
    
    if cc_a is not None:
        gray = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        c, i, _ = detector.detectMarkers(gray)
        cv2.aruco.drawDetectedMarkers(disp_a, c, i)
        cv2.aruco.drawDetectedCornersCharuco(disp_a, cc_a, ci_a)
    
    if cc_b is not None:
        gray = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
        c, i, _ = detector.detectMarkers(gray)
        cv2.aruco.drawDetectedMarkers(disp_b, c, i)
        cv2.aruco.drawDetectedCornersCharuco(disp_b, cc_b, ci_b)
    
    # Quality indicators
    h, w = disp_a.shape[:2]
    
    # Camera A status
    color_a = (0, 255, 0) if corners_a >= 4 else (0, 0, 255)
    cv2.rectangle(disp_a, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(disp_a, f"CAM A: {markers_a} markers, {corners_a} corners", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_a, 2)
    
    # Camera B status
    color_b = (0, 255, 0) if corners_b >= 4 else (0, 0, 255)
    cv2.rectangle(disp_b, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(disp_b, f"CAM B: {markers_b} markers, {corners_b} corners", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_b, 2)
    
    # Combine
    disp_a = cv2.resize(disp_a, (640, 360))
    disp_b = cv2.resize(disp_b, (640, 360))
    combined = np.hstack([disp_a, disp_b])
    
    cv2.putText(combined, "SPACE=collect, Q=quit", (10, combined.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.imshow("Quality Check", combined)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        break
    elif key == ord('q'):
        cap_a.release()
        cap_b.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

# =============================================================================
# COLLECT FRAMES WITH QUALITY SCORING
# =============================================================================

print("\n" + "="*70)
print("PHASE 2: COLLECTING FRAMES WITH QUALITY SCORING")
print("="*70)
print("\nSlowly wave board in the capture volume.")
print("I'll score each frame as it's captured.")
print("\nPress SPACE to stop collecting (aim for 15-25 good frames)")

frames_a = []
frames_b = []
frame_scores = []
last_capture = 0

cv2.namedWindow("Collecting")

while True:
    ret_a, frame_a = cap_a.read()
    ret_b, frame_b = cap_b.read()
    
    if not ret_a or not ret_b:
        continue
    
    cc_a, ci_a, markers_a, _ = detect_charuco(frame_a)
    cc_b, ci_b, markers_b, _ = detect_charuco(frame_b)
    
    corners_a = len(ci_a) if ci_a is not None else 0
    corners_b = len(ci_b) if ci_b is not None else 0
    
    both_ok = cc_a is not None and cc_b is not None
    
    # Auto-capture good frames
    now = time.time()
    if both_ok and now - last_capture > 0.5:
        common = set(ci_a.flatten()) & set(ci_b.flatten())
        if len(common) >= 4:
            # Calculate a quality score (higher = better)
            # Based on: number of corners, corner spread, common corners
            spread_a = np.std(cc_a.reshape(-1, 2), axis=0).mean()
            spread_b = np.std(cc_b.reshape(-1, 2), axis=0).mean()
            score = len(common) * min(spread_a, spread_b) / 100
            
            frames_a.append(frame_a.copy())
            frames_b.append(frame_b.copy())
            frame_scores.append({
                'corners_a': corners_a,
                'corners_b': corners_b,
                'common': len(common),
                'spread_a': spread_a,
                'spread_b': spread_b,
                'score': score
            })
            last_capture = now
            print(f"  Frame {len(frames_a)}: A={corners_a}c B={corners_b}c common={len(common)} score={score:.2f}")
    
    # Display
    disp_a = cv2.resize(frame_a, (640, 360))
    disp_b = cv2.resize(frame_b, (640, 360))
    
    color_a = (0, 255, 0) if cc_a is not None else (0, 0, 255)
    color_b = (0, 255, 0) if cc_b is not None else (0, 0, 255)
    
    cv2.rectangle(disp_a, (0, 0), (640, 40), (0, 0, 0), -1)
    cv2.putText(disp_a, f"CAM A: {corners_a} corners", (10, 28), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_a, 2)
    
    cv2.rectangle(disp_b, (0, 0), (640, 40), (0, 0, 0), -1)
    cv2.putText(disp_b, f"CAM B: {corners_b} corners", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_b, 2)
    
    combined = np.hstack([disp_a, disp_b])
    
    cv2.rectangle(combined, (0, combined.shape[0]-40), (combined.shape[1], combined.shape[0]), (0,0,0), -1)
    cv2.putText(combined, f"Frames: {len(frames_a)} | SPACE=stop collecting", 
                (10, combined.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("Collecting", combined)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') or len(frames_a) >= 30:
        break
    elif key == ord('q'):
        cap_a.release()
        cap_b.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

print(f"\nCollected {len(frames_a)} frames")

if len(frames_a) < 8:
    print("ERROR: Need at least 8 frames!")
    cap_a.release()
    cap_b.release()
    exit()

# =============================================================================
# ANALYZE FRAME QUALITY
# =============================================================================

print("\n" + "="*70)
print("PHASE 3: ANALYZING FRAME QUALITY")
print("="*70)

# First, calibrate each camera individually to get per-frame errors
corners_a_list = []
ids_a_list = []
corners_b_list = []
ids_b_list = []

for fa, fb in zip(frames_a, frames_b):
    cc_a, ci_a, _, _ = detect_charuco(fa)
    cc_b, ci_b, _, _ = detect_charuco(fb)
    
    if cc_a is not None and cc_b is not None:
        corners_a_list.append(cc_a)
        ids_a_list.append(ci_a)
        corners_b_list.append(cc_b)
        ids_b_list.append(ci_b)

img_size = (frames_a[0].shape[1], frames_a[0].shape[0])

ret_a, K_a, D_a, rvecs_a, tvecs_a, errors_a = calibrate_single_camera(
    corners_a_list, ids_a_list, img_size, "Camera A"
)
print(f"    Overall RMS: {ret_a:.4f}")

ret_b, K_b, D_b, rvecs_b, tvecs_b, errors_b = calibrate_single_camera(
    corners_b_list, ids_b_list, img_size, "Camera B"
)
print(f"    Overall RMS: {ret_b:.4f}")

# =============================================================================
# PER-FRAME ANALYSIS
# =============================================================================

print("\n" + "="*70)
print("PHASE 4: PER-FRAME REPROJECTION ERROR")
print("="*70)

print("\n  Frame | Cam A Error | Cam B Error | Status")
print("  " + "-"*50)

bad_frames_a = []
bad_frames_b = []
THRESHOLD = 1.0  # Frames with error > 1.0 are suspicious

for i, (err_a, err_b) in enumerate(zip(errors_a, errors_b)):
    status = "✓"
    if err_a > THRESHOLD:
        status = "⚠️ CamA high"
        bad_frames_a.append(i)
    if err_b > THRESHOLD:
        status = "⚠️ CamB high" if "CamA" not in status else "⚠️ BOTH high"
        bad_frames_b.append(i)
    
    print(f"  {i+1:5d} | {err_a:11.4f} | {err_b:11.4f} | {status}")

print("\n" + "="*70)
print("PHASE 5: DIAGNOSIS")
print("="*70)

print(f"\n  Camera A: {len(bad_frames_a)} frames with error > {THRESHOLD}")
print(f"  Camera B: {len(bad_frames_b)} frames with error > {THRESHOLD}")

if len(bad_frames_b) > len(bad_frames_a) * 2:
    print("\n  🔴 DIAGNOSIS: Camera B (DroidCam) has significantly more bad frames!")
    print("     Possible causes:")
    print("     1. Autofocus hunting - lock focus in DroidCam app")
    print("     2. Motion blur - move board more slowly")
    print("     3. Lower resolution - check DroidCam is sending 720p")
    print("     4. Compression artifacts - try USB instead of WiFi")
    print("     5. Auto-exposure changes - lock exposure in DroidCam app")

if len(bad_frames_a) > 3:
    print("\n  🟡 Camera A (Sony) also has some high-error frames")
    print("     Check for motion blur or focus issues")

# =============================================================================
# FILTERED CALIBRATION
# =============================================================================

print("\n" + "="*70)
print("PHASE 6: FILTERED STEREO CALIBRATION")
print("="*70)

# Remove bad frames
bad_indices = set(bad_frames_a) | set(bad_frames_b)
good_indices = [i for i in range(len(corners_a_list)) if i not in bad_indices]

print(f"\n  Using {len(good_indices)} good frames (removed {len(bad_indices)} bad frames)")

if len(good_indices) < 8:
    print("\n  ⚠️ Not enough good frames! Need to recapture with better technique.")
    print("  Tips:")
    print("    - Move board SLOWLY")
    print("    - Ensure board is sharp (not blurry) in BOTH cameras")
    print("    - Lock focus on DroidCam")
else:
    # Filter to good frames only
    good_corners_a = [corners_a_list[i] for i in good_indices]
    good_ids_a = [ids_a_list[i] for i in good_indices]
    good_corners_b = [corners_b_list[i] for i in good_indices]
    good_ids_b = [ids_b_list[i] for i in good_indices]
    
    # Recalibrate with filtered data
    print("\n  Recalibrating Camera A with filtered frames...")
    ret_a2, K1, D1, _, _ = cv2.aruco.calibrateCameraCharuco(
        good_corners_a, good_ids_a, board, img_size, None, None
    )
    print(f"    RMS: {ret_a2:.4f}")
    
    print("  Recalibrating Camera B with filtered frames...")
    ret_b2, K2, D2, _, _ = cv2.aruco.calibrateCameraCharuco(
        good_corners_b, good_ids_b, board, img_size, None, None
    )
    print(f"    RMS: {ret_b2:.4f}")
    
    # Build stereo correspondences
    obj_points = []
    img_points_a = []
    img_points_b = []
    
    for cc_a, ci_a, cc_b, ci_b in zip(good_corners_a, good_ids_a, good_corners_b, good_ids_b):
        common = set(ci_a.flatten()) & set(ci_b.flatten())
        if len(common) < 4:
            continue
        
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
    
    print(f"\n  Running stereo calibration with {len(obj_points)} frame pairs...")
    
    ret_stereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_a, img_points_b,
        K1, D1, K2, D2, img_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    
    baseline = np.linalg.norm(T)
    
    print(f"\n" + "="*70)
    print("FILTERED CALIBRATION RESULTS")
    print("="*70)
    print(f"  Camera A RMS: {ret_a2:.4f} {'✓' if ret_a2 < 1.0 else '!'}")
    print(f"  Camera B RMS: {ret_b2:.4f} {'✓' if ret_b2 < 1.0 else '!'}")
    print(f"  Stereo RMS:   {ret_stereo:.4f} {'✓' if ret_stereo < 2.0 else '!'}")
    print(f"  Baseline:     {baseline:.3f}m ({baseline*3.281:.1f} feet)")
    
    if ret_stereo < 2.0:
        print("\n  🎉 CALIBRATION SUCCESSFUL!")
        
        # Save calibration
        from pathlib import Path
        import json
        
        OUTPUT_FILE = Path(r"C:\Users\ninja\Documents\MelodicCapStudio\MelodicCapFresh\calibration\stereo_calibration.json")
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K1, D1, K2, D2, img_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        
        data = {
            'version': '5.1-filtered',
            'timestamp': datetime.now().isoformat(),
            'image_size': list(img_size),
            'K1': K1.tolist(), 'D1': D1.tolist(),
            'K2': K2.tolist(), 'D2': D2.tolist(),
            'R1': R1.tolist(), 'R2': R2.tolist(),
            'P1': P1.tolist(), 'P2': P2.tolist(),
            'R': R.tolist(), 'T': T.tolist(),
            'rms_cam_a': float(ret_a2),
            'rms_cam_b': float(ret_b2),
            'rms_stereo': float(ret_stereo),
            'baseline_meters': float(baseline),
            'floor_z_offset': 0.0,
            'floor_calibrated': False,
            'frames_used': len(good_indices),
            'frames_rejected': len(bad_indices),
        }
        
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  Saved to: {OUTPUT_FILE}")
    else:
        print("\n  ⚠️ Stereo RMS still too high!")
        print("  Recommendations:")
        print("    1. Check if cameras can both see the same area clearly")
        print("    2. Ensure board is at similar distance from both cameras")
        print("    3. Try capturing with board closer to cameras")
        print("    4. Check DroidCam settings (720p, fixed focus, fixed exposure)")

cap_a.release()
cap_b.release()
print("\n[DONE]")
