"""
Stereo Calibration Debug
=========================
Step-by-step debug of stereo calibration to find why it's failing.
"""

import cv2
import numpy as np
import time

print("="*60)
print("STEREO CALIBRATION DEBUG")
print("="*60)

# Setup
CAM_A_INDEX = 2
CAM_B_INDEX = 0

CHARUCO_COLS = 4
CHARUCO_ROWS = 3
CHARUCO_SQUARE_M = 0.0635
CHARUCO_MARKER_M = 0.0476

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

# Get board's 3D corner coordinates
board_corners_3d = board.getChessboardCorners()
print(f"\nBoard has {len(board_corners_3d)} corners")
print("Corner 3D positions (in board frame):")
for i, pt in enumerate(board_corners_3d):
    print(f"  Corner {i}: [{pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f}]")

# Open cameras
print("\n" + "-"*60)
print("Opening cameras...")
cap_a = cv2.VideoCapture(CAM_A_INDEX, cv2.CAP_DSHOW)
cap_b = cv2.VideoCapture(CAM_B_INDEX, cv2.CAP_ANY)

for cap in [cap_a, cap_b]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("OK")

print("\n" + "="*60)
print("STEP 1: Capture a single frame pair with board visible")
print("="*60)
print("Hold board so BOTH cameras can see it clearly.")
print("Press SPACE to capture, Q to quit")

captured_a = None
captured_b = None

while True:
    ret_a, frame_a = cap_a.read()
    ret_b, frame_b = cap_b.read()
    
    if not ret_a or not ret_b:
        continue
    
    # Detect in both
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    
    corners_a, ids_a, _ = detector.detectMarkers(gray_a)
    corners_b, ids_b, _ = detector.detectMarkers(gray_b)
    
    markers_a = len(ids_a) if ids_a is not None else 0
    markers_b = len(ids_b) if ids_b is not None else 0
    
    # Display
    disp_a = cv2.resize(frame_a, (640, 360))
    disp_b = cv2.resize(frame_b, (640, 360))
    
    color_a = (0, 255, 0) if markers_a >= 4 else (0, 0, 255)
    color_b = (0, 255, 0) if markers_b >= 4 else (0, 0, 255)
    
    cv2.putText(disp_a, f"CAM A: {markers_a} markers", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_a, 2)
    cv2.putText(disp_b, f"CAM B: {markers_b} markers", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_b, 2)
    
    combined = np.hstack([disp_a, disp_b])
    cv2.putText(combined, "SPACE=capture, Q=quit", (10, combined.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("Capture Frame Pair", combined)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        if markers_a >= 4 and markers_b >= 4:
            captured_a = frame_a.copy()
            captured_b = frame_b.copy()
            print(f"\nCaptured! A:{markers_a} markers, B:{markers_b} markers")
            break
        else:
            print("Need at least 4 markers in BOTH cameras!")
    elif key == ord('q'):
        cap_a.release()
        cap_b.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

print("\n" + "="*60)
print("STEP 2: Detect ChArUco corners in both frames")
print("="*60)

def detect_charuco(frame, name):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    
    print(f"\n{name}:")
    print(f"  ArUco markers detected: {len(ids) if ids is not None else 0}")
    
    if ids is None or len(ids) < 2:
        print(f"  ERROR: Not enough markers!")
        return None, None
    
    print(f"  Marker IDs: {sorted(ids.flatten().tolist())}")
    
    num, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    
    if num is None or num < 4:
        print(f"  ERROR: Not enough ChArUco corners! (got {num})")
        return None, None
    
    print(f"  ChArUco corners: {num}")
    print(f"  Corner IDs: {sorted(charuco_ids.flatten().tolist())}")
    
    # Print corner pixel positions
    print(f"  Corner positions (pixels):")
    for i, (corner, cid) in enumerate(zip(charuco_corners, charuco_ids)):
        px, py = corner.flatten()
        print(f"    ID {cid[0]}: ({px:.1f}, {py:.1f})")
    
    return charuco_corners, charuco_ids

corners_a, ids_a = detect_charuco(captured_a, "Camera A")
corners_b, ids_b = detect_charuco(captured_b, "Camera B")

if corners_a is None or corners_b is None:
    print("\nERROR: Detection failed!")
    cap_a.release()
    cap_b.release()
    exit()

print("\n" + "="*60)
print("STEP 3: Find common corners")
print("="*60)

ids_a_set = set(ids_a.flatten())
ids_b_set = set(ids_b.flatten())
common_ids = ids_a_set & ids_b_set

print(f"Camera A corner IDs: {sorted(ids_a_set)}")
print(f"Camera B corner IDs: {sorted(ids_b_set)}")
print(f"Common corner IDs: {sorted(common_ids)}")
print(f"Number of common corners: {len(common_ids)}")

if len(common_ids) < 4:
    print("\nERROR: Not enough common corners for calibration!")
    cap_a.release()
    cap_b.release()
    exit()

print("\n" + "="*60)
print("STEP 4: Check corner matching")
print("="*60)

print("\nMatching corners between cameras:")
print("-"*60)
print(f"{'Corner ID':<12} {'Camera A (px)':<20} {'Camera B (px)':<20}")
print("-"*60)

ids_a_flat = ids_a.flatten()
ids_b_flat = ids_b.flatten()

for cid in sorted(common_ids):
    idx_a = np.where(ids_a_flat == cid)[0][0]
    idx_b = np.where(ids_b_flat == cid)[0][0]
    
    px_a = corners_a[idx_a].flatten()
    px_b = corners_b[idx_b].flatten()
    
    print(f"{cid:<12} ({px_a[0]:6.1f}, {px_a[1]:6.1f})    ({px_b[0]:6.1f}, {px_b[1]:6.1f})")

# Check if corner positions make geometric sense
print("\n" + "="*60)
print("STEP 5: Geometric sanity check")
print("="*60)

# Get leftmost and rightmost corners in each camera
corners_a_x = [corners_a[np.where(ids_a_flat == cid)[0][0]].flatten()[0] for cid in common_ids]
corners_b_x = [corners_b[np.where(ids_b_flat == cid)[0][0]].flatten()[0] for cid in common_ids]

print(f"\nCamera A X range: {min(corners_a_x):.1f} to {max(corners_a_x):.1f}")
print(f"Camera B X range: {min(corners_b_x):.1f} to {max(corners_b_x):.1f}")

# Check if X ordering is consistent
# If cameras are both mirrored, the ordering should be the same
# If one is mirrored and one isn't, the ordering would be reversed

sorted_by_a = sorted(common_ids, key=lambda c: corners_a[np.where(ids_a_flat == c)[0][0]].flatten()[0])
sorted_by_b = sorted(common_ids, key=lambda c: corners_b[np.where(ids_b_flat == c)[0][0]].flatten()[0])

print(f"\nCorner IDs sorted by X position in Camera A: {sorted_by_a}")
print(f"Corner IDs sorted by X position in Camera B: {sorted_by_b}")

if sorted_by_a == sorted_by_b:
    print("\n✓ X ordering is SAME in both cameras (consistent mirroring)")
elif sorted_by_a == sorted_by_b[::-1]:
    print("\n✗ X ordering is REVERSED! One camera is mirrored, one isn't!")
    print("  THIS IS THE PROBLEM! Need to flip ONE camera.")
else:
    print("\n? X ordering is different but not simply reversed")
    print("  Cameras might be at very different angles")

print("\n" + "="*60)
print("STEP 6: Try stereo calibration with this one frame pair")
print("="*60)

# Collect 10 more frame pairs for calibration
print("\nCollecting 10 frame pairs for calibration...")
print("Wave the board slowly. Press Q to skip.")

frames_a = [captured_a]
frames_b = [captured_b]

cv2.namedWindow("Collecting")

while len(frames_a) < 10:
    ret_a, frame_a = cap_a.read()
    ret_b, frame_b = cap_b.read()
    
    if not ret_a or not ret_b:
        continue
    
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    
    c_a, i_a, _ = detector.detectMarkers(gray_a)
    c_b, i_b, _ = detector.detectMarkers(gray_b)
    
    disp = np.hstack([cv2.resize(frame_a, (640, 360)), cv2.resize(frame_b, (640, 360))])
    cv2.putText(disp, f"Frames: {len(frames_a)}/10", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Collecting", disp)
    
    if i_a is not None and i_b is not None and len(i_a) >= 4 and len(i_b) >= 4:
        _, cc_a, ci_a = cv2.aruco.interpolateCornersCharuco(c_a, i_a, gray_a, board)
        _, cc_b, ci_b = cv2.aruco.interpolateCornersCharuco(c_b, i_b, gray_b, board)
        
        if cc_a is not None and cc_b is not None:
            common = set(ci_a.flatten()) & set(ci_b.flatten())
            if len(common) >= 4:
                frames_a.append(frame_a.copy())
                frames_b.append(frame_b.copy())
                print(f"  Captured frame {len(frames_a)}")
                time.sleep(0.3)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

print(f"\nCollected {len(frames_a)} frame pairs")

# Run calibration
print("\nRunning stereo calibration...")

all_corners_a = []
all_ids_a = []
all_corners_b = []
all_ids_b = []
valid_pairs = []

for fa, fb in zip(frames_a, frames_b):
    gray_a = cv2.cvtColor(fa, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY)
    
    c_a, i_a, _ = detector.detectMarkers(gray_a)
    c_b, i_b, _ = detector.detectMarkers(gray_b)
    
    if i_a is None or i_b is None:
        continue
    
    _, cc_a, ci_a = cv2.aruco.interpolateCornersCharuco(c_a, i_a, gray_a, board)
    _, cc_b, ci_b = cv2.aruco.interpolateCornersCharuco(c_b, i_b, gray_b, board)
    
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

print(f"Valid pairs: {len(valid_pairs)}")

img_size = (1280, 720)

# Calibrate Camera A
print("\nCalibrating Camera A intrinsics...")
ret_a, K1, D1, rvecs_a, tvecs_a = cv2.aruco.calibrateCameraCharuco(
    all_corners_a, all_ids_a, board, img_size, None, None
)
print(f"  RMS: {ret_a:.4f}")

# Calibrate Camera B
print("Calibrating Camera B intrinsics...")
ret_b, K2, D2, rvecs_b, tvecs_b = cv2.aruco.calibrateCameraCharuco(
    all_corners_b, all_ids_b, board, img_size, None, None
)
print(f"  RMS: {ret_b:.4f}")

# Build correspondences for stereo
print("\nBuilding stereo correspondences...")
obj_points = []
img_points_a = []
img_points_b = []
board_corners_3d = board.getChessboardCorners()

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

print(f"  Total correspondence sets: {len(obj_points)}")
print(f"  Points per set: {[len(p) for p in obj_points]}")

# Stereo calibration
print("\nRunning stereo calibration...")
try:
    ret_stereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_a, img_points_b,
        K1, D1, K2, D2, img_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    
    baseline = np.linalg.norm(T)
    
    print(f"\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Stereo RMS: {ret_stereo:.4f}")
    print(f"Baseline: {baseline:.3f}m")
    print(f"\nRotation matrix R:\n{R}")
    print(f"\nTranslation vector T:\n{T.flatten()}")
    
    if ret_stereo > 10:
        print("\n⚠️  HIGH RMS - Something is wrong!")
        print("Possible causes:")
        print("  1. Cameras have inconsistent mirroring")
        print("  2. Very different focal lengths")
        print("  3. Cameras too close or too far apart")
        print("  4. Poor corner detection quality")
    else:
        print("\n✓ Calibration looks reasonable!")
        
except Exception as e:
    print(f"\nERROR during stereo calibration: {e}")

cap_a.release()
cap_b.release()
print("\n[DONE]")
