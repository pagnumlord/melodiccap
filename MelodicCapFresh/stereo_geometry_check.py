"""
Stereo Geometry Diagnostic
===========================
Checks WHY stereo calibration fails even when individual cameras are good.

The most likely issue: camera images have inconsistent geometric relationship.
This tool visualizes the corner positions to diagnose the problem.
"""

import cv2
import numpy as np
import time

# Configuration
CAM_A_INDEX = 2
CAM_B_INDEX = 0

CHARUCO_COLS = 4
CHARUCO_ROWS = 3
CHARUCO_SQUARE_M = 0.0635
CHARUCO_MARKER_M = 0.0476

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

print("="*70)
print("STEREO GEOMETRY DIAGNOSTIC")
print("="*70)

# Open cameras
print("\nOpening cameras...")
cap_a = cv2.VideoCapture(CAM_A_INDEX, cv2.CAP_DSHOW)
cap_b = cv2.VideoCapture(CAM_B_INDEX, cv2.CAP_ANY)

for cap in [cap_a, cap_b]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap_a.isOpened() or not cap_b.isOpened():
    print("ERROR: Could not open cameras!")
    exit(1)

print("OK")

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


def get_x_order(corners, ids):
    """Get corner IDs sorted by X position"""
    if corners is None or ids is None:
        return []
    
    id_x_pairs = []
    for corner, cid in zip(corners, ids):
        x = corner.flatten()[0]
        id_x_pairs.append((cid[0], x))
    
    # Sort by X position
    sorted_pairs = sorted(id_x_pairs, key=lambda p: p[1])
    return [p[0] for p in sorted_pairs]


def check_x_order_consistency(order_a, order_b):
    """Check if X orderings are consistent"""
    if len(order_a) == 0 or len(order_b) == 0:
        return "NO_DATA", None
    
    # Get common corners in order
    common = set(order_a) & set(order_b)
    if len(common) < 3:
        return "TOO_FEW", None
    
    # Filter to common only, maintaining order
    filtered_a = [c for c in order_a if c in common]
    filtered_b = [c for c in order_b if c in common]
    
    if filtered_a == filtered_b:
        return "SAME", filtered_a
    elif filtered_a == filtered_b[::-1]:
        return "REVERSED", filtered_a
    else:
        return "DIFFERENT", (filtered_a, filtered_b)


print("\n" + "="*70)
print("LIVE GEOMETRY CHECK")
print("="*70)
print("\nThis shows corner X-ordering in both cameras.")
print("For stereo to work, the orderings must be CONSISTENT.")
print("\nSAME = Good (both cameras see corners in same left-to-right order)")
print("REVERSED = Bad! (one camera's view is geometrically flipped)")
print("\nPress Q to quit and collect frames for detailed analysis")

same_count = 0
reversed_count = 0
different_count = 0

while True:
    ret_a, frame_a = cap_a.read()
    ret_b, frame_b = cap_b.read()
    
    if not ret_a or not ret_b:
        continue
    
    # Detect
    cc_a, ci_a, markers_a = detect_charuco(frame_a)
    cc_b, ci_b, markers_b = detect_charuco(frame_b)
    
    # Get X orderings
    order_a = get_x_order(cc_a, ci_a)
    order_b = get_x_order(cc_b, ci_b)
    
    consistency, data = check_x_order_consistency(order_a, order_b)
    
    # Update counts
    if consistency == "SAME":
        same_count += 1
    elif consistency == "REVERSED":
        reversed_count += 1
    elif consistency == "DIFFERENT":
        different_count += 1
    
    # Create visualization
    disp_a = frame_a.copy()
    disp_b = frame_b.copy()
    
    # Draw detections
    if cc_a is not None:
        gray = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        c, i, _ = detector.detectMarkers(gray)
        cv2.aruco.drawDetectedMarkers(disp_a, c, i)
        cv2.aruco.drawDetectedCornersCharuco(disp_a, cc_a, ci_a)
        
        # Draw corner IDs with positions
        for corner, cid in zip(cc_a, ci_a):
            x, y = corner.flatten().astype(int)
            cv2.putText(disp_a, str(cid[0]), (x+10, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    if cc_b is not None:
        gray = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
        c, i, _ = detector.detectMarkers(gray)
        cv2.aruco.drawDetectedMarkers(disp_b, c, i)
        cv2.aruco.drawDetectedCornersCharuco(disp_b, cc_b, ci_b)
        
        for corner, cid in zip(cc_b, ci_b):
            x, y = corner.flatten().astype(int)
            cv2.putText(disp_b, str(cid[0]), (x+10, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Draw order strings
    h = disp_a.shape[0]
    
    cv2.rectangle(disp_a, (0, 0), (640, 80), (0, 0, 0), -1)
    cv2.putText(disp_a, f"CAM A X-order: {order_a}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.rectangle(disp_b, (0, 0), (640, 80), (0, 0, 0), -1)
    cv2.putText(disp_b, f"CAM B X-order: {order_b}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw consistency status
    if consistency == "SAME":
        color = (0, 255, 0)
        status = "✓ SAME - Good for stereo"
    elif consistency == "REVERSED":
        color = (0, 0, 255)
        status = "✗ REVERSED - Need to flip one camera!"
    elif consistency == "DIFFERENT":
        color = (0, 165, 255)
        status = "? DIFFERENT - Unusual, check board position"
    else:
        color = (150, 150, 150)
        status = "Waiting for detection..."
    
    cv2.putText(disp_a, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(disp_b, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Resize and combine
    disp_a = cv2.resize(disp_a, (640, 360))
    disp_b = cv2.resize(disp_b, (640, 360))
    combined = np.hstack([disp_a, disp_b])
    
    # Stats bar
    total = same_count + reversed_count + different_count
    if total > 0:
        stats = f"Same: {same_count} ({same_count*100//total}%) | Reversed: {reversed_count} ({reversed_count*100//total}%) | Other: {different_count}"
    else:
        stats = "Collecting stats..."
    
    bar = np.zeros((50, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, stats, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    combined = np.vstack([combined, bar])
    
    cv2.imshow("Stereo Geometry Check - Q to quit", combined)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

print("\n" + "="*70)
print("LIVE CHECK RESULTS")
print("="*70)
print(f"  SAME (good):     {same_count}")
print(f"  REVERSED (bad):  {reversed_count}")
print(f"  OTHER:           {different_count}")

if reversed_count > same_count:
    print("\n🔴 DIAGNOSIS: X-ordering is mostly REVERSED!")
    print("   This means the cameras have opposite geometric views.")
    print("   Solution: Flip ONE camera's image horizontally.")
    print("\n   Would you like to test with Camera B flipped? (y/n)")
    
    choice = input("   > ").strip().lower()
    if choice == 'y':
        FLIP_CAM_B = True
    else:
        FLIP_CAM_B = False
elif same_count > 10:
    print("\n✓ X-ordering is mostly SAME - good!")
    FLIP_CAM_B = False
else:
    print("\n⚠️ Not enough data to determine. Defaulting to no flip.")
    FLIP_CAM_B = False

# =============================================================================
# COLLECT AND CALIBRATE WITH/WITHOUT FLIP
# =============================================================================

print("\n" + "="*70)
print("PHASE 2: COLLECTING FRAMES FOR CALIBRATION")
print("="*70)
print(f"\nCamera B flip: {'ON' if FLIP_CAM_B else 'OFF'}")
print("Press H to toggle Camera B flip during collection")
print("Press SPACE when done collecting (aim for 15-20 frames)")

frames_a = []
frames_b = []
frame_info = []
last_capture = 0

while True:
    ret_a, frame_a = cap_a.read()
    ret_b, frame_b = cap_b.read()
    
    if not ret_a or not ret_b:
        continue
    
    # Apply flip if enabled
    if FLIP_CAM_B:
        frame_b_proc = cv2.flip(frame_b, 1)
    else:
        frame_b_proc = frame_b
    
    # Detect
    cc_a, ci_a, markers_a = detect_charuco(frame_a)
    cc_b, ci_b, markers_b = detect_charuco(frame_b_proc)
    
    # Get X orderings
    order_a = get_x_order(cc_a, ci_a)
    order_b = get_x_order(cc_b, ci_b)
    
    consistency, _ = check_x_order_consistency(order_a, order_b)
    
    # Auto-capture when consistent
    now = time.time()
    if cc_a is not None and cc_b is not None and consistency == "SAME" and now - last_capture > 0.5:
        common = set(ci_a.flatten()) & set(ci_b.flatten())
        if len(common) >= 4:
            frames_a.append(frame_a.copy())
            frames_b.append(frame_b_proc.copy())  # Save processed frame
            frame_info.append({
                'order_a': order_a,
                'order_b': order_b,
                'common': len(common),
                'consistency': consistency
            })
            last_capture = now
            print(f"  Frame {len(frames_a)}: {len(common)} common, {consistency}")
    
    # Display
    disp_a = frame_a.copy()
    disp_b = frame_b_proc.copy()
    
    # Draw detections
    if cc_a is not None:
        gray = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        c, i, _ = detector.detectMarkers(gray)
        cv2.aruco.drawDetectedMarkers(disp_a, c, i)
    
    if cc_b is not None:
        gray = cv2.cvtColor(frame_b_proc, cv2.COLOR_BGR2GRAY)
        c, i, _ = detector.detectMarkers(gray)
        cv2.aruco.drawDetectedMarkers(disp_b, c, i)
    
    # Status
    color = (0, 255, 0) if consistency == "SAME" else (0, 0, 255)
    
    cv2.rectangle(disp_a, (0, 0), (400, 50), (0, 0, 0), -1)
    cv2.putText(disp_a, f"CAM A | {consistency}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.rectangle(disp_b, (0, 0), (400, 50), (0, 0, 0), -1)
    flip_status = "[FLIPPED]" if FLIP_CAM_B else ""
    cv2.putText(disp_b, f"CAM B {flip_status} | {consistency}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    disp_a = cv2.resize(disp_a, (640, 360))
    disp_b = cv2.resize(disp_b, (640, 360))
    combined = np.hstack([disp_a, disp_b])
    
    bar = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, f"Frames: {len(frames_a)} | H=toggle flip | SPACE=done", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    combined = np.vstack([combined, bar])
    
    cv2.imshow("Collecting - Only SAME ordering captured", combined)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        break
    elif key == ord('h'):
        FLIP_CAM_B = not FLIP_CAM_B
        print(f"  Camera B flip: {'ON' if FLIP_CAM_B else 'OFF'}")
    elif key == ord('q'):
        cap_a.release()
        cap_b.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

print(f"\nCollected {len(frames_a)} frames with consistent ordering")

if len(frames_a) < 8:
    print("ERROR: Need at least 8 frames!")
    cap_a.release()
    cap_b.release()
    exit()

# =============================================================================
# RUN CALIBRATION
# =============================================================================

print("\n" + "="*70)
print("PHASE 3: STEREO CALIBRATION")
print("="*70)

# Collect corners
all_corners_a, all_ids_a = [], []
all_corners_b, all_ids_b = [], []
valid_pairs = []

for fa, fb in zip(frames_a, frames_b):
    cc_a, ci_a, _ = detect_charuco(fa)
    cc_b, ci_b, _ = detect_charuco(fb)
    
    if cc_a is None or cc_b is None:
        continue
    
    common = set(ci_a.flatten()) & set(ci_b.flatten())
    if len(common) < 4:
        continue
    
    all_corners_a.append(cc_a)
    all_ids_a.append(ci_a)
    all_corners_b.append(cc_b)
    all_ids_b.append(ci_b)
    valid_pairs.append((cc_a, ci_a, cc_b, ci_b, common))

print(f"\nValid pairs: {len(valid_pairs)}")

img_size = (frames_a[0].shape[1], frames_a[0].shape[0])

# Calibrate cameras
print("\nCalibrating Camera A...")
ret_a, K1, D1, _, _ = cv2.aruco.calibrateCameraCharuco(
    all_corners_a, all_ids_a, board, img_size, None, None
)
print(f"  RMS: {ret_a:.4f}")

print("Calibrating Camera B...")
ret_b, K2, D2, _, _ = cv2.aruco.calibrateCameraCharuco(
    all_corners_b, all_ids_b, board, img_size, None, None
)
print(f"  RMS: {ret_b:.4f}")

# Build stereo correspondences
obj_points = []
img_points_a = []
img_points_b = []

for cc_a, ci_a, cc_b, ci_b, common in valid_pairs:
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

# Stereo calibration
print("\nStereo calibration...")
ret_stereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    obj_points, img_points_a, img_points_b,
    K1, D1, K2, D2, img_size,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    flags=cv2.CALIB_FIX_INTRINSIC
)

baseline = np.linalg.norm(T)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"  Camera A RMS: {ret_a:.4f} {'✓' if ret_a < 1.0 else '!'}")
print(f"  Camera B RMS: {ret_b:.4f} {'✓' if ret_b < 1.0 else '!'}")
print(f"  Stereo RMS:   {ret_stereo:.4f} {'✓' if ret_stereo < 2.0 else '!'}")
print(f"  Baseline:     {baseline:.3f}m ({baseline*3.281:.1f} feet)")
print(f"  Camera B flip: {'ON' if FLIP_CAM_B else 'OFF'}")

if ret_stereo < 3.0:
    print("\n🎉 CALIBRATION SUCCESS!")
    
    # Save
    import json
    from datetime import datetime
    from pathlib import Path
    
    OUTPUT_FILE = Path(r"C:\Users\ninja\Documents\MelodicCapStudio\MelodicCapFresh\calibration\stereo_calibration.json")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    
    data = {
        'version': '5.2-geometry-checked',
        'timestamp': datetime.now().isoformat(),
        'image_size': list(img_size),
        'K1': K1.tolist(), 'D1': D1.tolist(),
        'K2': K2.tolist(), 'D2': D2.tolist(),
        'R1': R1.tolist(), 'R2': R2.tolist(),
        'P1': P1.tolist(), 'P2': P2.tolist(),
        'R': R.tolist(), 'T': T.tolist(),
        'rms_cam_a': float(ret_a),
        'rms_cam_b': float(ret_b),
        'rms_stereo': float(ret_stereo),
        'baseline_meters': float(baseline),
        'floor_z_offset': 0.0,
        'floor_calibrated': False,
        'flip_cam_b': FLIP_CAM_B,
        'frames_used': len(valid_pairs),
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  Saved to: {OUTPUT_FILE}")
    
    if FLIP_CAM_B:
        print("\n  ⚠️ IMPORTANT: Camera B flip is ON!")
        print("     Make sure melodic_capture_v5.py also flips Camera B!")
else:
    print("\n❌ STEREO STILL FAILING!")
    print("\nPossible issues:")
    print("  1. Cameras have very different viewing angles")
    print("  2. Board orientation is inconsistent between cameras")
    print("  3. Camera positions are unusual")
    print("\nTry:")
    print("  - Position cameras closer together (2-3 feet apart)")
    print("  - Make sure both cameras are at similar height")
    print("  - Point both cameras at the same area")

cap_a.release()
cap_b.release()
print("\n[DONE]")
