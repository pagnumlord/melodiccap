"""
MelodicCap Studio - Motion Capture v2.2
========================================
Based on capture v8 (proven: 100% tracking, Kalman smoothing, 30-frame grace period).

v2.2 changes (from v2.1):
- Post-processing pipeline on take save:
  - Velocity-based outlier rejection (0.5m/frame threshold, catches stereo spikes)
  - Bone-length stabilization (enforces consistent skeleton proportions)
  - 3-frame temporal smoothing (reduces jitter)
  - Auto-trim collapsed frames at end of take
  - Floor clamping (feet can't go below z=0)
- Lowered outlier threshold from 5.0m to 3.0m

v2.1 changes (from v2.0):
- Iterative outlier rejection in stereo calibration (from robust_calibration.py)
- Fixed empty session log bug (fsync + write verification)
- Better resolution handling (match to smaller camera, preserve aspect ratio)
- Calibration quality validation with per-frame error reporting

v2.0 changes:
- Comprehensive debug logging to file (every session gets a log)
- Per-landmark visibility/confidence saved in take JSON (uncertainty analysis)
- Camera resolution validation on startup
- Frame timing diagnostics (inter-camera grab delay)
- Full capture metadata in take JSON
- Outlier filter: rejects landmarks > threshold from origin
- Configurable camera indices via command-line args

Controls:
  C = Collect calibration frames
  S = Run stereo calibration
  F = Floor calibration
  R = Record motion capture
  Q = Quit
"""

import cv2
import numpy as np
import mediapipe as mp
import json
import os
import time
import sys
import traceback
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

CAM_A_INDEX = 2          # Sony ZV-1F (use DSHOW)
CAM_B_INDEX = 0          # DroidCam (use CAP_ANY)

# ChArUco board (63.5mm squares)
CHARUCO_COLS = 4
CHARUCO_ROWS = 3
CHARUCO_SQUARE_M = 0.0635
CHARUCO_MARKER_M = 0.0476

# Paths - relative to this script's parent (MelodicCapStudio/)
BASE_DIR = Path(__file__).parent.parent  # MelodicCapStudio/
CALIBRATION_FILE = BASE_DIR / "calibration" / "stereo_calibration.json"
TAKES_DIR = BASE_DIR / "takes"
LOGS_DIR = BASE_DIR / "logs"

# Recording settings
COUNTDOWN_SECONDS = 10

# Grace period for tracking loss
TRACKING_GRACE_FRAMES = 30

# MediaPipe settings
MEDIAPIPE_MODEL_COMPLEXITY = 1      # 0=lite, 1=full, 2=heavy
MEDIAPIPE_MIN_DETECTION_CONF = 0.3
MEDIAPIPE_MIN_TRACKING_CONF = 0.3

# Outlier filter: reject landmarks more than this many meters from origin
# Prevents bad triangulation spikes (seen up to +/-715m in production takes)
OUTLIER_THRESHOLD_M = 3.0

# Minimum visibility score to accept a landmark from MediaPipe
MIN_VISIBILITY = 0.3

# =============================================================================
# LOGGING
# =============================================================================

_log_file = None
_log_path = None

def log_init(tag="capture"):
    """Open a session log file."""
    global _log_file, _log_path
    log_close()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_path = LOGS_DIR / f"capture_{tag}_{ts}.log"
    try:
        _log_file = open(_log_path, 'w', encoding='utf-8')
        # Verify the file handle works immediately
        _log_file.write("")
        _log_file.flush()
        os.fsync(_log_file.fileno())
    except Exception as e:
        print(f"[ERROR] Failed to open log file {_log_path}: {e}")
        _log_file = None

    log(f"Log file: {_log_path}")
    log(f"Session started: {datetime.now().isoformat()}")
    log(f"Python: {sys.version}")
    log(f"OpenCV: {cv2.__version__}")
    log(f"MediaPipe: {mp.__version__}")

def log_close():
    """Close the log file."""
    global _log_file, _log_path
    if _log_file:
        try:
            _log_file.flush()
            os.fsync(_log_file.fileno())
            _log_file.close()
        except Exception:
            pass
        _log_file = None

def log(msg, level="INFO"):
    """Log to both console and file."""
    line = f"[{level}] {msg}"
    print(line)
    if _log_file:
        try:
            _log_file.write(line + "\n")
            _log_file.flush()
        except Exception:
            pass  # Don't crash on log write failure

# =============================================================================
# SKELETON DEFINITIONS (for bone-length stabilization)
# =============================================================================

# Bone segments: (start_landmark, end_landmark, name)
# These define the rigid segments of the human skeleton
SKELETON_BONES = [
    (11, 12, 'shoulders'),      # shoulder width
    (23, 24, 'hips'),           # hip width
    (11, 13, 'l_upper_arm'),    # left upper arm
    (13, 15, 'l_forearm'),      # left forearm
    (12, 14, 'r_upper_arm'),    # right upper arm
    (14, 16, 'r_forearm'),      # right forearm
    (23, 25, 'l_thigh'),        # left thigh
    (25, 27, 'l_shin'),         # left shin
    (24, 26, 'r_thigh'),        # right thigh
    (26, 28, 'r_shin'),         # right shin
    (11, 23, 'l_torso'),        # left torso (shoulder to hip)
    (12, 24, 'r_torso'),        # right torso
]

# Landmarks that are anchored (hips define root, everything else adjusts)
HIP_LANDMARKS = [23, 24]

# =============================================================================
# POST-PROCESSING: Stabilize bone lengths, smooth, trim, clamp floor
# =============================================================================

def postprocess_take(frames, log_fn=None):
    """Post-process recorded frames to fix stereo triangulation noise.

    Steps:
    1. Velocity-based outlier rejection (per-landmark)
    2. Bone-length stabilization (enforce consistent skeleton proportions)
    3. Temporal smoothing (moving average)
    4. Trim collapsed frames at end of take
    5. Floor clamping (feet can't go below z=0)

    Modifies frames in-place and returns the (possibly trimmed) list.
    """
    if not frames or len(frames) < 5:
        return frames

    def _log(msg, level="INFO"):
        if log_fn:
            log_fn(msg, level)

    _log("=" * 50)
    _log("POST-PROCESSING TAKE")
    _log("=" * 50)

    n_original = len(frames)

    # --- Step 1: Velocity-based outlier rejection ---
    _log("\n  [Step 1] Velocity-based outlier rejection")
    # At ~13 FPS, max human movement speed is ~10 m/s -> ~0.77m per frame
    # Use 0.5m as threshold (catches spikes while allowing fast movements)
    MAX_JUMP_M = 0.5
    outlier_replacements = 0

    all_landmarks = set()
    for f in frames:
        all_landmarks.update(f.get('landmarks', {}).keys())

    prev_good = {}  # landmark_key -> [x, y, z]
    for fidx, fdata in enumerate(frames):
        lms = fdata.get('landmarks', {})
        for lm_key in list(lms.keys()):
            pos = lms[lm_key]
            if lm_key in prev_good:
                pg = prev_good[lm_key]
                dx = pos[0] - pg[0]
                dy = pos[1] - pg[1]
                dz = pos[2] - pg[2]
                dist = (dx*dx + dy*dy + dz*dz) ** 0.5
                if dist > MAX_JUMP_M:
                    # Replace with last good position
                    lms[lm_key] = list(prev_good[lm_key])
                    outlier_replacements += 1
                else:
                    prev_good[lm_key] = list(pos)
            else:
                prev_good[lm_key] = list(pos)

    _log(f"    Replaced {outlier_replacements} outlier positions (threshold: {MAX_JUMP_M}m/frame)")

    # --- Step 2: Bone-length stabilization ---
    _log("\n  [Step 2] Bone-length stabilization")

    # Measure reference bone lengths from first 10 clean frames (median)
    ref_lengths = {}
    for bone_start, bone_end, bone_name in SKELETON_BONES:
        lengths = []
        for f in frames[:min(30, len(frames))]:
            lms = f.get('landmarks', {})
            sk_s = str(bone_start)
            sk_e = str(bone_end)
            # Try both string and int keys
            p1 = lms.get(sk_s) or lms.get(bone_start)
            p2 = lms.get(sk_e) or lms.get(bone_end)
            if p1 and p2:
                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                dz = p1[2] - p2[2]
                length = (dx*dx + dy*dy + dz*dz) ** 0.5
                if length > 0.01 and length < 2.0:  # sanity bounds
                    lengths.append(length)
        if lengths:
            # Use median to be robust to any remaining outliers
            lengths.sort()
            ref_lengths[bone_name] = lengths[len(lengths)//2]

    if ref_lengths:
        _log(f"    Reference bone lengths (from first frames):")
        for name, length in ref_lengths.items():
            _log(f"      {name:15s}: {length:.3f}m")

    # Now enforce bone lengths on every frame
    # Strategy: keep hips fixed (root), adjust child landmarks to maintain
    # correct bone length while preserving direction
    corrections = 0
    for fidx, fdata in enumerate(frames):
        lms = fdata.get('landmarks', {})

        for bone_start, bone_end, bone_name in SKELETON_BONES:
            if bone_name not in ref_lengths:
                continue

            target_len = ref_lengths[bone_name]
            sk_s = str(bone_start)
            sk_e = str(bone_end)
            p1 = lms.get(sk_s) or lms.get(bone_start)
            p2 = lms.get(sk_e) or lms.get(bone_end)

            if not p1 or not p2:
                continue

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dz = p2[2] - p1[2]
            current_len = (dx*dx + dy*dy + dz*dz) ** 0.5

            if current_len < 0.001:
                continue

            # Allow 15% tolerance before correcting
            ratio = current_len / target_len
            if ratio < 0.85 or ratio > 1.15:
                # Scale the endpoint to match target length
                scale = target_len / current_len
                new_p2 = [
                    p1[0] + dx * scale,
                    p1[1] + dy * scale,
                    p1[2] + dz * scale,
                ]
                # Update the endpoint (whichever key format is used)
                key = sk_e if sk_e in lms else bone_end
                if key in lms:
                    lms[key] = new_p2
                    corrections += 1

    _log(f"    Applied {corrections} bone-length corrections")

    # --- Step 3: Temporal smoothing (3-frame moving average) ---
    _log("\n  [Step 3] Temporal smoothing (3-frame moving average)")
    if len(frames) >= 3:
        # Collect all landmark keys
        all_keys = set()
        for f in frames:
            all_keys.update(f.get('landmarks', {}).keys())

        for lm_key in all_keys:
            # Extract time series for this landmark
            positions = []
            for f in frames:
                lms = f.get('landmarks', {})
                positions.append(lms.get(lm_key))

            # Apply 3-frame moving average (skip missing)
            smoothed = [None] * len(positions)
            for i in range(len(positions)):
                if positions[i] is None:
                    smoothed[i] = None
                    continue

                # Collect neighbors
                pts = []
                for j in range(max(0, i-1), min(len(positions), i+2)):
                    if positions[j] is not None:
                        pts.append(positions[j])

                if pts:
                    avg = [
                        sum(p[0] for p in pts) / len(pts),
                        sum(p[1] for p in pts) / len(pts),
                        sum(p[2] for p in pts) / len(pts),
                    ]
                    smoothed[i] = avg

            # Write back
            for i, f in enumerate(frames):
                lms = f.get('landmarks', {})
                if smoothed[i] is not None and lm_key in lms:
                    lms[lm_key] = smoothed[i]

        _log(f"    Smoothed {len(all_keys)} landmarks across {len(frames)} frames")

    # --- Step 4: Trim collapsed frames ---
    _log("\n  [Step 4] Trim collapsed frames (skeleton height < 70% of reference)")

    # Measure reference height from first 5 frames
    ref_heights = []
    for f in frames[:5]:
        lms = f.get('landmarks', {})
        nose = lms.get('0') or lms.get(0)
        l_ankle = lms.get('27') or lms.get(27)
        r_ankle = lms.get('28') or lms.get(28)
        if nose and l_ankle and r_ankle:
            ankle_z = (l_ankle[2] + r_ankle[2]) / 2
            height = nose[2] - ankle_z
            if height > 0.5:
                ref_heights.append(height)

    if ref_heights:
        ref_height = sum(ref_heights) / len(ref_heights)
        min_height = ref_height * 0.70
        _log(f"    Reference height: {ref_height:.3f}m, minimum: {min_height:.3f}m")

        # Find last good frame (scan from end)
        trim_from = len(frames)
        for i in range(len(frames) - 1, -1, -1):
            lms = frames[i].get('landmarks', {})
            nose = lms.get('0') or lms.get(0)
            l_ankle = lms.get('27') or lms.get(27)
            r_ankle = lms.get('28') or lms.get(28)
            if nose and l_ankle and r_ankle:
                ankle_z = (l_ankle[2] + r_ankle[2]) / 2
                height = nose[2] - ankle_z
                if height >= min_height:
                    trim_from = i + 1
                    break

        if trim_from < len(frames):
            trimmed = len(frames) - trim_from
            frames[:] = frames[:trim_from]
            _log(f"    Trimmed {trimmed} collapsed frames from end")
        else:
            _log(f"    No collapsed frames detected")
    else:
        _log(f"    Could not measure reference height (missing nose/ankle)")

    # --- Step 5: Floor clamping ---
    _log("\n  [Step 5] Floor clamping (feet z >= 0)")
    floor_clamps = 0
    foot_landmarks = ['27', '28', '29', '30', '31', '32']  # ankles, heels, foot index
    for f in frames:
        lms = f.get('landmarks', {})
        for lm_key in foot_landmarks:
            if lm_key in lms and lms[lm_key][2] < 0:
                lms[lm_key][2] = 0.0
                floor_clamps += 1

    _log(f"    Clamped {floor_clamps} foot positions to floor")

    # --- Summary ---
    _log(f"\n  POST-PROCESSING COMPLETE:")
    _log(f"    Input frames:  {n_original}")
    _log(f"    Output frames: {len(frames)}")
    _log(f"    Outliers fixed: {outlier_replacements}")
    _log(f"    Bone corrections: {corrections}")
    _log(f"    Floor clamps: {floor_clamps}")

    return frames


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
board_corners_3d = board.getChessboardCorners()
BOARD_MAX_CORNERS = (CHARUCO_COLS - 1) * (CHARUCO_ROWS - 1)

# MediaPipe landmark names for logging
LANDMARK_NAMES = {
    0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow", 15: "left_wrist", 16: "right_wrist",
    17: "left_pinky", 18: "right_pinky", 19: "left_index", 20: "right_index",
    21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle", 29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index",
}

# =============================================================================
# CALIBRATION FUNCTIONS
# =============================================================================

def detect_charuco(frame):
    """Detect ChArUco corners in frame."""
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


def draw_detection(frame, charuco_corners, charuco_ids, num_markers):
    """Draw detection results on frame."""
    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(display, corners, ids)

    if charuco_corners is not None:
        cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids,
                                              cornerColor=(0, 255, 0))

    return display


def run_stereo_calibration(frames_a, frames_b, max_iterations=5, outlier_threshold=2.0):
    """Run stereo calibration with iterative outlier rejection.

    Approach (from robust_calibration.py):
    1. Extract corners from all frames
    2. Do initial calibration
    3. Calculate per-frame stereo reprojection error
    4. Remove worst frames (error > outlier_threshold * median)
    5. Recalibrate
    6. Repeat until stable or RMS < 2.0
    """
    log("=" * 50)
    log("STEREO CALIBRATION (iterative outlier rejection)")
    log("=" * 50)

    # For small boards, require more corners per frame for stronger constraints
    min_corners = max(BOARD_MAX_CORNERS - 1, 4)  # e.g. 5 out of 6 for 4x3 board
    use_joint_optimization = BOARD_MAX_CORNERS < 12

    if use_joint_optimization:
        log(f"  Small board mode: {BOARD_MAX_CORNERS} max corners, requiring {min_corners}+ per frame", "WARN")
        log(f"  Using joint intrinsic+extrinsic optimization for better results", "WARN")
        log(f"  TIP: Move board SLOWLY, hold still at each position, cover varied angles", "WARN")
    else:
        min_corners = 4

    # Phase 1: Extract corners from all frame pairs
    all_data = []
    for i, (fa, fb) in enumerate(zip(frames_a, frames_b)):
        cc_a, ci_a, _ = detect_charuco(fa)
        cc_b, ci_b, _ = detect_charuco(fb)

        if cc_a is None or cc_b is None:
            log(f"  Frame {i}: skipped (detection failed)")
            continue

        common = set(ci_a.flatten()) & set(ci_b.flatten())
        if len(common) < min_corners:
            log(f"  Frame {i}: skipped (only {len(common)} common corners, need {min_corners}+)")
            continue

        all_data.append({
            'index': i,
            'corners_a': cc_a, 'ids_a': ci_a,
            'corners_b': cc_b, 'ids_b': ci_b,
            'common': common,
            'active': True,
        })
        log(f"  Frame {i}: {len(common)} common corners")

    log(f"Valid pairs: {len(all_data)}/{len(frames_a)}")

    if len(all_data) < 8:
        return None, f"Only {len(all_data)} valid pairs (need 8+)"

    img_size = (frames_a[0].shape[1], frames_a[0].shape[0])

    best_result = None
    best_rms = float('inf')

    # Phase 2: Iterative calibration with outlier rejection
    for iteration in range(max_iterations):
        log(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

        active_data = [d for d in all_data if d['active']]
        log(f"Active frames: {len(active_data)}")

        if len(active_data) < 8:
            log("Too few frames remaining, stopping iteration")
            break

        # Individual camera calibration
        corners_a = [d['corners_a'] for d in active_data]
        ids_a = [d['ids_a'] for d in active_data]
        corners_b = [d['corners_b'] for d in active_data]
        ids_b = [d['ids_b'] for d in active_data]

        ret_a, K1, D1, _, _ = cv2.aruco.calibrateCameraCharuco(
            corners_a, ids_a, board, img_size, None, None
        )
        ret_b, K2, D2, _, _ = cv2.aruco.calibrateCameraCharuco(
            corners_b, ids_b, board, img_size, None, None
        )
        log(f"  Camera A RMS: {ret_a:.4f} (fx={K1[0,0]:.1f}, fy={K1[1,1]:.1f})")
        log(f"  Camera B RMS: {ret_b:.4f} (fx={K2[0,0]:.1f}, fy={K2[1,1]:.1f})")

        # Build stereo correspondences
        obj_points = []
        img_points_a = []
        img_points_b = []
        frame_indices = []

        for d in active_data:
            cc_a, ci_a = d['corners_a'], d['ids_a']
            cc_b, ci_b = d['corners_b'], d['ids_b']
            common = d['common']

            obj_pts, pts_a, pts_b = [], [], []
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
        # For small boards, use INTRINSIC_GUESS to jointly optimize intrinsics+extrinsics
        # since individual camera calibrations are underdetermined with few corners.
        # For large boards, fix intrinsics (they're reliable from individual calibration).
        if use_joint_optimization:
            stereo_flags = cv2.CALIB_USE_INTRINSIC_GUESS
        else:
            stereo_flags = cv2.CALIB_FIX_INTRINSIC

        ret_stereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            obj_points, img_points_a, img_points_b,
            K1, D1, K2, D2, img_size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
            flags=stereo_flags
        )

        baseline = float(np.linalg.norm(T))
        log(f"  Stereo RMS: {ret_stereo:.4f}")
        log(f"  Baseline: {baseline:.3f}m")

        # Track best result
        if ret_stereo < best_rms:
            best_rms = ret_stereo
            best_result = {
                'K1': K1.copy(), 'D1': D1.copy(),
                'K2': K2.copy(), 'D2': D2.copy(),
                'R': R.copy(), 'T': T.copy(),
                'ret_a': ret_a, 'ret_b': ret_b,
                'ret_stereo': ret_stereo,
                'baseline': baseline,
                'num_frames': len(active_data),
            }

        # If RMS is acceptable, stop iterating
        if ret_stereo < 2.0:
            log(f"  Stereo RMS {ret_stereo:.4f} is good, stopping iteration")
            break

        # Phase 3: Calculate per-frame reprojection error to find outliers
        log("  Calculating per-frame reprojection errors...")

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K1, D1, K2, D2, img_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )

        frame_errors = []
        for fi, (obj_pts, pts_a, pts_b) in enumerate(zip(obj_points, img_points_a, img_points_b)):
            pts_a_rect = cv2.undistortPoints(pts_a.reshape(-1, 1, 2), K1, D1, R=R1, P=P1)
            pts_b_rect = cv2.undistortPoints(pts_b.reshape(-1, 1, 2), K2, D2, R=R2, P=P2)

            pts_4d = cv2.triangulatePoints(P1, P2,
                                            pts_a_rect.reshape(-1, 2).T,
                                            pts_b_rect.reshape(-1, 2).T)
            pts_3d = (pts_4d[:3] / pts_4d[3]).T

            pts_3d_h = np.hstack([pts_3d, np.ones((pts_3d.shape[0], 1))])
            proj_a = (P1 @ pts_3d_h.T).T
            proj_a = proj_a[:, :2] / proj_a[:, 2:3]
            proj_b = (P2 @ pts_3d_h.T).T
            proj_b = proj_b[:, :2] / proj_b[:, 2:3]

            err_a = np.mean(np.linalg.norm(pts_a_rect.reshape(-1, 2) - proj_a, axis=1))
            err_b = np.mean(np.linalg.norm(pts_b_rect.reshape(-1, 2) - proj_b, axis=1))

            frame_errors.append({
                'frame_idx': frame_indices[fi],
                'error': (err_a + err_b) / 2,
                'err_a': err_a, 'err_b': err_b
            })

        frame_errors.sort(key=lambda x: x['error'], reverse=True)
        errors = [f['error'] for f in frame_errors]
        median_error = np.median(errors)

        log(f"  Median frame error: {median_error:.4f}")
        log(f"  Worst frame error: {frame_errors[0]['error']:.4f} (frame {frame_errors[0]['frame_idx']})")

        # Remove frames with error > threshold * median
        # For small boards, keep more frames to avoid over-pruning
        min_frames_keep = 15 if use_joint_optimization else 10
        threshold = max(outlier_threshold * median_error, 1.0)
        removed = 0

        for fe in frame_errors:
            if fe['error'] > threshold and len(active_data) - removed > min_frames_keep:
                for d in all_data:
                    if d['index'] == fe['frame_idx']:
                        d['active'] = False
                        removed += 1
                        log(f"    Removed frame {fe['frame_idx']} (error: {fe['error']:.4f})")
                        break

        if removed == 0:
            log("  No outliers found, stopping iteration")
            break

        log(f"  Removed {removed} outlier frames")

    # Phase 4: Validate and use best result
    if best_result is None:
        return None, "Calibration failed - no valid result"

    if best_rms > 3.0:
        log(f"\n  REJECTED: Best Stereo RMS {best_rms:.2f} exceeds quality threshold (3.0)", "WARN")
        return None, f"Stereo RMS {best_rms:.1f} too high (need < 3.0). Try a larger board or more varied angles."

    log(f"\n  BEST RESULT: Stereo RMS={best_rms:.4f}, {best_result['num_frames']} frames, baseline={best_result['baseline']:.3f}m")

    # Sanity check: focal lengths should be in reasonable range for the image size
    img_w = img_size[0]
    for cam_name, K in [("A", best_result['K1']), ("B", best_result['K2'])]:
        fx, fy = K[0, 0], K[1, 1]
        if fx < img_w * 0.3 or fx > img_w * 3.0 or fy < img_w * 0.3 or fy > img_w * 3.0:
            log(f"\n  REJECTED: Camera {cam_name} focal length implausible (fx={fx:.1f}, fy={fy:.1f}) for {img_w}px width", "WARN")
            return None, f"Camera {cam_name} focal length implausible (fx={fx:.0f}). Calibration data insufficient."

    # Sanity check: baseline should be physically reasonable (0.1m - 5.0m)
    baseline = best_result['baseline']
    if baseline < 0.1 or baseline > 5.0:
        log(f"\n  REJECTED: Baseline {baseline:.3f}m is implausible (expect 0.1-5.0m)", "WARN")
        return None, f"Baseline {baseline:.2f}m implausible. Check camera positions."

    # Compute rectification from best result
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        best_result['K1'], best_result['D1'],
        best_result['K2'], best_result['D2'],
        img_size, best_result['R'], best_result['T'],
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'version': 'studio-2.1-robust',
        'timestamp': datetime.now().isoformat(),
        'image_size': list(img_size),
        'K1': best_result['K1'].tolist(), 'D1': best_result['D1'].tolist(),
        'K2': best_result['K2'].tolist(), 'D2': best_result['D2'].tolist(),
        'R1': R1.tolist(), 'R2': R2.tolist(),
        'P1': P1.tolist(), 'P2': P2.tolist(),
        'R': best_result['R'].tolist(), 'T': best_result['T'].tolist(),
        'rms_cam_a': float(best_result['ret_a']),
        'rms_cam_b': float(best_result['ret_b']),
        'rms_stereo': float(best_result['ret_stereo']),
        'baseline_meters': float(best_result['baseline']),
        'floor_z_offset': 0.0,
        'floor_calibrated': False,
        'frames_used': best_result['num_frames'],
    }

    # Protect against overwriting a better existing calibration
    if CALIBRATION_FILE.exists():
        try:
            with open(CALIBRATION_FILE, 'r') as f:
                existing = json.load(f)
            existing_rms = existing.get('rms_stereo', float('inf'))
            if existing_rms < best_rms:
                log(f"\n  WARNING: Existing calibration (RMS={existing_rms:.4f}) is better than new (RMS={best_rms:.4f})", "WARN")
                log(f"  Saving new calibration anyway - old one is backed up", "WARN")
                # Backup existing calibration
                backup_path = CALIBRATION_FILE.with_suffix('.json.bak')
                with open(backup_path, 'w') as f:
                    json.dump(existing, f, indent=2)
                log(f"  Backup saved: {backup_path}")
        except (json.JSONDecodeError, IOError):
            pass  # No existing calibration or can't read it

    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(data, f, indent=2)

    log(f"Saved: {CALIBRATION_FILE}")

    return data, f"CamA:{best_result['ret_a']:.2f} CamB:{best_result['ret_b']:.2f} Stereo:{best_rms:.2f} Base:{best_result['baseline']:.2f}m ({best_result['num_frames']}f)"


def calibrate_floor(frame_a, frame_b, calib_data):
    """Calibrate floor from board lying flat on ground."""
    cc_a, ci_a, _ = detect_charuco(frame_a)
    cc_b, ci_b, _ = detect_charuco(frame_b)

    if cc_a is None:
        return False, "Board not detected in Camera A"
    if cc_b is None:
        return False, "Board not detected in Camera B"

    common = set(ci_a.flatten()) & set(ci_b.flatten())
    if len(common) < 4:
        return False, f"Only {len(common)} common corners"

    K1 = np.array(calib_data['K1'])
    D1 = np.array(calib_data['D1'])
    K2 = np.array(calib_data['K2'])
    D2 = np.array(calib_data['D2'])
    R1 = np.array(calib_data['R1'])
    R2 = np.array(calib_data['R2'])
    P1 = np.array(calib_data['P1'])
    P2 = np.array(calib_data['P2'])

    pts_a = []
    pts_b = []
    ci_a_flat = ci_a.flatten()
    ci_b_flat = ci_b.flatten()

    for cid in sorted(common):
        idx_a = np.where(ci_a_flat == cid)[0][0]
        idx_b = np.where(ci_b_flat == cid)[0][0]
        pts_a.append(cc_a[idx_a].flatten())
        pts_b.append(cc_b[idx_b].flatten())

    pts_a = np.array(pts_a, dtype=np.float32).reshape(-1, 1, 2)
    pts_b = np.array(pts_b, dtype=np.float32).reshape(-1, 1, 2)

    pts_a_rect = cv2.undistortPoints(pts_a, K1, D1, R=R1, P=P1)
    pts_b_rect = cv2.undistortPoints(pts_b, K2, D2, R=R2, P=P2)

    points_4d = cv2.triangulatePoints(P1, P2,
                                       pts_a_rect.reshape(-1, 2).T,
                                       pts_b_rect.reshape(-1, 2).T)
    points_3d = (points_4d[:3] / points_4d[3]).T

    # OpenCV -Y -> Blender Z, so floor is at -mean(-Y)
    blender_z = -points_3d[:, 1]
    floor_offset = -np.mean(blender_z)

    # Check consistency of floor points
    z_std = np.std(blender_z)
    z_range = np.max(blender_z) - np.min(blender_z)

    log(f"  Floor calibration: {len(common)} points")
    log(f"    Blender Z values: {blender_z}")
    log(f"    Mean: {np.mean(blender_z):.4f}m, Std: {z_std:.4f}m, Range: {z_range:.4f}m")
    log(f"    Floor offset: {floor_offset:.4f}m")

    if z_std > 0.05:
        log(f"    WARNING: Floor point spread is high ({z_std:.4f}m std). "
            "Board might not be flat or calibration is noisy.", "WARN")

    calib_data['floor_z_offset'] = float(floor_offset)
    calib_data['floor_calibrated'] = True
    calib_data['floor_z_std'] = float(z_std)
    calib_data['floor_z_range'] = float(z_range)

    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(calib_data, f, indent=2)

    return True, f"Floor offset: {floor_offset:.3f}m (std: {z_std:.3f}m)"


# =============================================================================
# MOTION CAPTURE FUNCTIONS
# =============================================================================

class KalmanFilter3D:
    """Kalman filter for 3D point smoothing with prediction capability."""
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        self.kf = cv2.KalmanFilter(6, 3)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        self.initialized = False
        self.frames_since_update = 0

    def update(self, measurement):
        """Update filter with measurement."""
        if measurement is None:
            return self.predict()

        if isinstance(measurement, (list, tuple)):
            m = np.array([[measurement[0]], [measurement[1]], [measurement[2]]], dtype=np.float32)
        else:
            m = np.array(measurement, dtype=np.float32).reshape(3, 1)

        if not self.initialized:
            self.kf.statePost = np.array([[m[0,0]], [m[1,0]], [m[2,0]], [0], [0], [0]], dtype=np.float32)
            self.initialized = True
            self.frames_since_update = 0
            return [float(m[0,0]), float(m[1,0]), float(m[2,0])]

        self.kf.predict()
        corrected = self.kf.correct(m)
        self.frames_since_update = 0
        return [float(corrected[0,0]), float(corrected[1,0]), float(corrected[2,0])]

    def predict(self):
        """Get prediction without measurement."""
        if not self.initialized:
            return None

        pred = self.kf.predict()
        self.frames_since_update += 1
        return [float(pred[0,0]), float(pred[1,0]), float(pred[2,0])]


def triangulate_landmarks(landmarks_a, landmarks_b, calib_data, img_size):
    """Triangulate MediaPipe landmarks to 3D Blender coordinates.

    Coordinate conversion:
        blender_x = opencv_x      (right)
        blender_y = opencv_z      (forward/depth)
        blender_z = -opencv_y + floor_offset  (up)

    Returns:
        points_3d: dict of {landmark_idx: [x, y, z]}
        visibility: dict of {landmark_idx: {'vis_a': float, 'vis_b': float, 'vis_avg': float}}
        outlier_count: number of landmarks rejected by outlier filter
    """
    K1 = np.array(calib_data['K1'])
    D1 = np.array(calib_data['D1'])
    K2 = np.array(calib_data['K2'])
    D2 = np.array(calib_data['D2'])
    R1 = np.array(calib_data['R1'])
    R2 = np.array(calib_data['R2'])
    P1 = np.array(calib_data['P1'])
    P2 = np.array(calib_data['P2'])
    floor_offset = calib_data.get('floor_z_offset', 0.0)

    w, h = img_size
    points_3d = {}
    visibility = {}
    outlier_count = 0

    for idx in range(33):
        lm_a = landmarks_a.landmark[idx]
        lm_b = landmarks_b.landmark[idx]

        vis_a = lm_a.visibility
        vis_b = lm_b.visibility
        vis_avg = (vis_a + vis_b) / 2

        if vis_a < MIN_VISIBILITY or vis_b < MIN_VISIBILITY:
            continue

        # Save visibility regardless of outlier status
        visibility[idx] = {
            'vis_a': round(float(vis_a), 3),
            'vis_b': round(float(vis_b), 3),
            'vis_avg': round(float(vis_avg), 3),
        }

        pt_a = np.array([[lm_a.x * w, lm_a.y * h]], dtype=np.float32).reshape(-1, 1, 2)
        pt_b = np.array([[lm_b.x * w, lm_b.y * h]], dtype=np.float32).reshape(-1, 1, 2)

        pt_a_rect = cv2.undistortPoints(pt_a, K1, D1, R=R1, P=P1)
        pt_b_rect = cv2.undistortPoints(pt_b, K2, D2, R=R2, P=P2)

        point_4d = cv2.triangulatePoints(P1, P2,
                                          pt_a_rect.reshape(2, 1),
                                          pt_b_rect.reshape(2, 1))
        point_3d = (point_4d[:3] / point_4d[3]).flatten()

        blender_x = point_3d[0]
        blender_y = point_3d[2]
        blender_z = -point_3d[1] + floor_offset

        # OUTLIER FILTER: reject landmarks with unreasonable coordinates.
        if abs(blender_x) > OUTLIER_THRESHOLD_M or abs(blender_y) > OUTLIER_THRESHOLD_M or abs(blender_z) > OUTLIER_THRESHOLD_M:
            outlier_count += 1
            continue

        points_3d[idx] = [float(blender_x), float(blender_y), float(blender_z)]

    return points_3d, visibility, outlier_count


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Parse optional camera index overrides
    cam_a = CAM_A_INDEX
    cam_b = CAM_B_INDEX
    if len(sys.argv) >= 3:
        cam_a = int(sys.argv[1])
        cam_b = int(sys.argv[2])
        print(f"Using camera indices from args: A={cam_a}, B={cam_b}")

    # Initialize logging
    log_init("session")

    log("=" * 60)
    log("MELODICCAP STUDIO - MOTION CAPTURE v2.2")
    log("=" * 60)

    log(f"\n  Configuration:")
    log(f"    Camera A index: {cam_a}")
    log(f"    Camera B index: {cam_b}")
    log(f"    Countdown: {COUNTDOWN_SECONDS}s")
    log(f"    Grace period: {TRACKING_GRACE_FRAMES} frames")
    log(f"    Outlier threshold: {OUTLIER_THRESHOLD_M}m")
    log(f"    Min visibility: {MIN_VISIBILITY}")
    log(f"    MediaPipe complexity: {MEDIAPIPE_MODEL_COMPLEXITY}")
    log(f"    MediaPipe detection conf: {MEDIAPIPE_MIN_DETECTION_CONF}")
    log(f"    MediaPipe tracking conf: {MEDIAPIPE_MIN_TRACKING_CONF}")
    log(f"    Base dir: {BASE_DIR}")
    log(f"    Calibration file: {CALIBRATION_FILE}")
    log(f"    Takes dir: {TAKES_DIR}")
    log(f"    ChArUco board: {CHARUCO_COLS}x{CHARUCO_ROWS} ({BOARD_MAX_CORNERS} max corners)")
    if BOARD_MAX_CORNERS < 12:
        log(f"    NOTE: Small board ({BOARD_MAX_CORNERS} corners). Calibration uses joint optimization mode.", "WARN")
        log(f"    TIPS: Move board SLOWLY, ensure {BOARD_MAX_CORNERS - 1}+ corners visible in BOTH cameras,", "WARN")
        log(f"          cover many angles (tilt, rotate), collect 30+ frames.", "WARN")

    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    # Separate pose detectors for each camera
    pose_a = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
        min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONF,
        min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONF
    )
    pose_b = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
        min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONF,
        min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONF
    )

    # Open cameras
    log("\n[CAMERAS]")

    log(f"  Opening Camera A (index {cam_a}, DSHOW)...")
    cap_a = cv2.VideoCapture(cam_a, cv2.CAP_DSHOW)
    cap_a.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap_a.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap_a.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap_a.isOpened():
        log("    FAILED to open Camera A!", "ERROR")
        log_close()
        return

    native_w_a = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h_a = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps_a = cap_a.get(cv2.CAP_PROP_FPS)
    log(f"    Native: {native_w_a}x{native_h_a} @ {actual_fps_a:.1f}fps")

    log(f"  Opening Camera B (index {cam_b}, CAP_ANY)...")
    cap_b = cv2.VideoCapture(cam_b, cv2.CAP_ANY)
    cap_b.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap_b.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap_b.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap_b.isOpened():
        log("    FAILED to open Camera B!", "ERROR")
        cap_a.release()
        log_close()
        return

    native_w_b = int(cap_b.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h_b = int(cap_b.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps_b = cap_b.get(cv2.CAP_PROP_FPS)
    log(f"    Native: {native_w_b}x{native_h_b} @ {actual_fps_b:.1f}fps")

    # Determine common processing resolution.
    # Both cameras MUST process at the same resolution for calibration/triangulation
    # to work correctly (K matrices encode focal lengths for a specific resolution).
    # Use the smaller camera's resolution to preserve aspect ratio.
    pixels_a = native_w_a * native_h_a
    pixels_b = native_w_b * native_h_b
    if pixels_a <= pixels_b:
        PROC_W, PROC_H = native_w_a, native_h_a
    else:
        PROC_W, PROC_H = native_w_b, native_h_b
    # Track whether we need to resize each camera's frames
    resize_a = (native_w_a != PROC_W or native_h_a != PROC_H)
    resize_b = (native_w_b != PROC_W or native_h_b != PROC_H)

    log(f"\n  Processing resolution: {PROC_W}x{PROC_H}")
    if resize_a:
        log(f"    Camera A: will resize {native_w_a}x{native_h_a} -> {PROC_W}x{PROC_H}")
    if resize_b:
        log(f"    Camera B: will resize {native_w_b}x{native_h_b} -> {PROC_W}x{PROC_H}")
    if not resize_a and not resize_b:
        log(f"    Both cameras match - no resizing needed")

    # Load existing calibration
    calib_data = None
    if CALIBRATION_FILE.exists():
        try:
            with open(CALIBRATION_FILE) as f:
                calib_data = json.load(f)
            log(f"\n[CALIBRATION] Loaded (v{calib_data.get('version', '?')}, "
                f"Stereo RMS: {calib_data.get('rms_stereo', '?'):.4f}, "
                f"Baseline: {calib_data.get('baseline_meters', '?'):.3f}m)")

            # Check if calibration resolution matches current processing resolution
            cal_size = calib_data.get('image_size', [0, 0])
            if cal_size[0] != PROC_W or cal_size[1] != PROC_H:
                log(f"  RESOLUTION MISMATCH: calibration was done at "
                    f"{cal_size[0]}x{cal_size[1]} but cameras are now at "
                    f"{PROC_W}x{PROC_H}", "WARN")
                log(f"  You MUST recalibrate (C then S) before recording!", "WARN")
                calib_data = None  # Invalidate old calibration

            if calib_data and calib_data.get('floor_calibrated'):
                log(f"[FLOOR] Offset: {calib_data.get('floor_z_offset', 0):.4f}m")
            elif calib_data:
                log("[FLOOR] Not calibrated - press F with board on floor", "WARN")
        except Exception as e:
            log(f"Failed to load calibration: {e}", "ERROR")

    log("\n[CONTROLS]")
    log("  C = Collect calibration frames (hold board visible to both cameras)")
    log("  S = Run stereo calibration (after collecting 15+ frames)")
    log("  F = Floor calibration (lay board flat on floor)")
    log("  R = Start/stop recording motion capture")
    log("  Q = Quit")

    # State
    cal_frames_a = []
    cal_frames_b = []
    collecting_cal = False
    floor_mode = False
    recording = False
    countdown_active = False
    countdown_start = 0
    last_capture = 0
    status = "Ready" if calib_data else "Press C to calibrate"
    recorded_frames = []
    record_start_time = 0

    # Kalman filters
    kalman_filters = {}

    # Tracking statistics
    frames_without_tracking = 0
    predicted_frames = 0
    dropped_frames = 0
    total_outliers = 0

    # Per-camera tracking stats
    cam_a_lost_count = 0
    cam_b_lost_count = 0
    both_lost_count = 0
    total_frames = 0

    # Frame timing
    frame_grab_delays = []  # Time between cam A and cam B reads

    # Per-frame visibility data
    all_visibility = []

    frame_count = 0

    try:
        while True:
            # Measure inter-camera grab delay
            t_grab_a = time.time()
            ret_a, frame_a = cap_a.read()
            t_grab_b = time.time()
            ret_b, frame_b = cap_b.read()
            t_grab_done = time.time()

            if not ret_a or not ret_b:
                continue

            # Resize to common processing resolution if needed
            if resize_a:
                frame_a = cv2.resize(frame_a, (PROC_W, PROC_H))
            if resize_b:
                frame_b = cv2.resize(frame_b, (PROC_W, PROC_H))

            frame_count += 1
            grab_delay = t_grab_b - t_grab_a

            if recording:
                frame_grab_delays.append(grab_delay)

            # Detect ChArUco (skip during recording to save CPU)
            if not recording:
                cc_a, ci_a, markers_a = detect_charuco(frame_a)
                cc_b, ci_b, markers_b = detect_charuco(frame_b)

                board_a = cc_a is not None
                board_b = cc_b is not None
            else:
                cc_a = ci_a = markers_a = None
                cc_b = ci_b = markers_b = None
                board_a = board_b = False

            # Create display frames
            if collecting_cal or floor_mode:
                display_a = draw_detection(frame_a, cc_a, ci_a, markers_a)
                display_b = draw_detection(frame_b, cc_b, ci_b, markers_b)
            else:
                display_a = frame_a.copy()
                display_b = frame_b.copy()

            # Process poses
            pose_a_valid = False
            pose_b_valid = False

            if not collecting_cal and not floor_mode:
                results_a = pose_a.process(cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB))
                results_b = pose_b.process(cv2.cvtColor(frame_b, cv2.COLOR_BGR2RGB))

                pose_a_valid = results_a.pose_landmarks is not None
                pose_b_valid = results_b.pose_landmarks is not None

                # Track per-camera statistics during recording
                if recording:
                    total_frames += 1
                    if not pose_a_valid and not pose_b_valid:
                        both_lost_count += 1
                    elif not pose_a_valid:
                        cam_a_lost_count += 1
                    elif not pose_b_valid:
                        cam_b_lost_count += 1

                # Draw skeletons
                if pose_a_valid:
                    mp_draw.draw_landmarks(display_a, results_a.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                if pose_b_valid:
                    mp_draw.draw_landmarks(display_b, results_b.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Recording logic with grace period
                if recording and calib_data:
                    both_valid = pose_a_valid and pose_b_valid

                    if both_valid:
                        frames_without_tracking = 0

                        try:
                            img_size = (frame_a.shape[1], frame_a.shape[0])
                            points_3d, vis_data, outlier_count = triangulate_landmarks(
                                results_a.pose_landmarks,
                                results_b.pose_landmarks,
                                calib_data,
                                img_size
                            )
                            total_outliers += outlier_count

                            smoothed = {}
                            for idx, pt in points_3d.items():
                                if idx not in kalman_filters:
                                    kalman_filters[idx] = KalmanFilter3D()

                                filtered = kalman_filters[idx].update(pt)
                                if filtered is not None:
                                    smoothed[idx] = filtered

                            if len(smoothed) > 0:
                                recorded_frames.append({
                                    'timestamp': time.time() - record_start_time,
                                    'landmarks': smoothed,
                                    'num_landmarks': len(smoothed),
                                    'predicted': False,
                                    'outliers_this_frame': outlier_count,
                                    'grab_delay_ms': round(grab_delay * 1000, 2),
                                })
                                all_visibility.append(vis_data)
                        except Exception as e:
                            log(f"Frame {frame_count}: {e}", "WARN")
                            log(traceback.format_exc(), "WARN")

                    else:
                        frames_without_tracking += 1

                        if frames_without_tracking <= TRACKING_GRACE_FRAMES and len(kalman_filters) > 0:
                            smoothed = {}
                            for idx, kf in kalman_filters.items():
                                if kf.initialized:
                                    pred = kf.predict()
                                    if pred is not None:
                                        smoothed[idx] = pred

                            if len(smoothed) > 5:
                                recorded_frames.append({
                                    'timestamp': time.time() - record_start_time,
                                    'landmarks': smoothed,
                                    'num_landmarks': len(smoothed),
                                    'predicted': True,
                                    'outliers_this_frame': 0,
                                    'grab_delay_ms': round(grab_delay * 1000, 2),
                                })
                                all_visibility.append({})
                                predicted_frames += 1
                            else:
                                dropped_frames += 1
                        else:
                            dropped_frames += 1

            # Draw borders for calibration modes
            h_a, w_a = display_a.shape[:2]
            h_b, w_b = display_b.shape[:2]

            if collecting_cal or floor_mode:
                color_a = (0, 255, 0) if board_a else (0, 0, 255)
                color_b = (0, 255, 0) if board_b else (0, 0, 255)
                cv2.rectangle(display_a, (0, 0), (w_a-1, h_a-1), color_a, 8)
                cv2.rectangle(display_b, (0, 0), (w_b-1, h_b-1), color_b, 8)

            # Camera labels with tracking status
            cv2.rectangle(display_a, (0, 0), (200, 40), (0, 0, 0), -1)
            if collecting_cal or floor_mode:
                cv2.putText(display_a, f"CAM A ({markers_a}m)", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                color = (0, 255, 0) if pose_a_valid else (0, 0, 255)
                cv2.putText(display_a, f"CAM A {'OK' if pose_a_valid else 'LOST'}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.rectangle(display_b, (0, 0), (200, 40), (0, 0, 0), -1)
            if collecting_cal or floor_mode:
                cv2.putText(display_b, f"CAM B ({markers_b}m)", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                color = (0, 255, 0) if pose_b_valid else (0, 0, 255)
                cv2.putText(display_b, f"CAM B {'OK' if pose_b_valid else 'LOST'}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Per-camera loss stats during recording
            if recording and total_frames > 0:
                a_loss_pct = (cam_a_lost_count / total_frames) * 100
                cv2.putText(display_a, f"Lost: {a_loss_pct:.0f}%", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

                b_loss_pct = (cam_b_lost_count / total_frames) * 100
                cv2.putText(display_b, f"Lost: {b_loss_pct:.0f}%", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            # Collect calibration frames
            # Require more corners for small boards to get better constraints
            min_cal_corners = max(BOARD_MAX_CORNERS - 1, 4)
            if collecting_cal and board_a and board_b:
                common = set(ci_a.flatten()) & set(ci_b.flatten())
                now = time.time()
                if len(common) >= min_cal_corners and now - last_capture > 0.4:
                    cal_frames_a.append(frame_a.copy())
                    cal_frames_b.append(frame_b.copy())
                    last_capture = now
                    status = f"Captured {len(cal_frames_a)} frames ({len(common)}/{BOARD_MAX_CORNERS} corners)"
                    log(f"[CAL] Frame {len(cal_frames_a)}: {len(common)} common corners")

            # Floor calibration
            if floor_mode and board_a and board_b and calib_data:
                success, msg = calibrate_floor(frame_a, frame_b, calib_data)
                if success:
                    floor_mode = False
                    status = f"Floor OK: {msg}"
                    log(f"[FLOOR] {msg}")

            # Countdown handling
            if countdown_active:
                remaining = COUNTDOWN_SECONDS - (time.time() - countdown_start)
                if remaining <= 0:
                    countdown_active = False
                    recording = True
                    record_start_time = time.time()
                    recorded_frames = []
                    kalman_filters = {}
                    frames_without_tracking = 0
                    predicted_frames = 0
                    dropped_frames = 0
                    total_outliers = 0
                    cam_a_lost_count = 0
                    cam_b_lost_count = 0
                    both_lost_count = 0
                    total_frames = 0
                    frame_grab_delays = []
                    all_visibility = []
                    status = "RECORDING..."
                    log("[REC] Recording started")

            # Resize both to same size and combine
            DISPLAY_W, DISPLAY_H = 640, 360
            disp_a = cv2.resize(display_a, (DISPLAY_W, DISPLAY_H))
            disp_b = cv2.resize(display_b, (DISPLAY_W, DISPLAY_H))
            combined = np.hstack([disp_a, disp_b])

            # Draw countdown
            if countdown_active:
                remaining = COUNTDOWN_SECONDS - (time.time() - countdown_start)
                count = max(1, int(remaining) + 1)
                h, w = combined.shape[:2]
                cv2.rectangle(combined, (w//2-80, h//2-80), (w//2+80, h//2+80), (0, 0, 0), -1)
                cv2.putText(combined, str(count), (w//2-30, h//2+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 5)
                cv2.putText(combined, "GET READY!", (w//2-70, h//2+70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Recording indicator
            if recording:
                h, w = combined.shape[:2]
                cv2.circle(combined, (w-30, 30), 15, (0, 0, 255), -1)
                elapsed = time.time() - record_start_time

                cv2.putText(combined, f"{len(recorded_frames)}f {elapsed:.1f}s", (w-160, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if predicted_frames > 0:
                    cv2.putText(combined, f"pred:{predicted_frames}", (w-160, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                if dropped_frames > 0:
                    cv2.putText(combined, f"drop:{dropped_frames}", (w-160, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)

                if frames_without_tracking > 0:
                    grace_remaining = TRACKING_GRACE_FRAMES - frames_without_tracking
                    if grace_remaining > 0:
                        cv2.putText(combined, f"GRACE: {grace_remaining}", (w//2-50, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        cv2.putText(combined, "LOST!", (w//2-30, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Status bar
            bar = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
            cv2.putText(bar, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(bar, f"Cal frames: {len(cal_frames_a)}", (combined.shape[1]-180, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            combined = np.vstack([combined, bar])
            cv2.imshow("MelodicCap Studio", combined)

            # Key handling
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('c'):
                if collecting_cal:
                    collecting_cal = False
                    status = f"Collected {len(cal_frames_a)} frames. Press S to calibrate."
                    log(f"[CAL] Collection stopped: {len(cal_frames_a)} frames")
                else:
                    cal_frames_a = []
                    cal_frames_b = []
                    collecting_cal = True
                    floor_mode = False
                    status = "CALIBRATING - Show board to BOTH cameras"
                    log("[CAL] Collection started")

            elif key == ord('s'):
                if len(cal_frames_a) < 8:
                    status = f"Need 8+ frames (have {len(cal_frames_a)})"
                else:
                    collecting_cal = False
                    status = "Running calibration..."
                    cv2.waitKey(100)

                    result, msg = run_stereo_calibration(cal_frames_a, cal_frames_b)
                    if result:
                        calib_data = result
                        status = f"SUCCESS! {msg}"
                    else:
                        status = f"FAILED: {msg}"

            elif key == ord('f'):
                if calib_data is None:
                    status = "Calibrate cameras first!"
                else:
                    floor_mode = not floor_mode
                    status = "FLOOR MODE - Lay board flat on floor" if floor_mode else "Floor mode off"
                    log(f"[FLOOR] Mode {'ON' if floor_mode else 'OFF'}")

            elif key == ord('r'):
                if calib_data is None:
                    status = "Calibrate cameras first!"
                elif recording:
                    recording = False
                    log("[REC] Recording stopped")

                    if len(recorded_frames) > 0:
                        # Post-process: stabilize bones, smooth, trim, clamp
                        recorded_frames = postprocess_take(recorded_frames, log_fn=log)

                        TAKES_DIR.mkdir(parents=True, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = TAKES_DIR / f"take_{timestamp}.json"

                        # Compute visibility summary
                        vis_summary = {}
                        for vis_frame in all_visibility:
                            for lm_idx, vis in vis_frame.items():
                                if lm_idx not in vis_summary:
                                    vis_summary[lm_idx] = {'count': 0, 'total_avg': 0}
                                vis_summary[lm_idx]['count'] += 1
                                vis_summary[lm_idx]['total_avg'] += vis['vis_avg']

                        # Average visibility per landmark
                        vis_avg_per_lm = {}
                        for lm_idx, data in vis_summary.items():
                            avg = data['total_avg'] / max(data['count'], 1)
                            name = LANDMARK_NAMES.get(lm_idx, f"lm_{lm_idx}")
                            vis_avg_per_lm[str(lm_idx)] = {
                                'name': name,
                                'avg_visibility': round(avg, 3),
                                'frames_visible': data['count'],
                                'coverage_pct': round(data['count'] / len(recorded_frames) * 100, 1),
                            }

                        # Frame timing stats
                        timing_stats = {}
                        if frame_grab_delays:
                            timing_stats = {
                                'mean_grab_delay_ms': round(np.mean(frame_grab_delays) * 1000, 2),
                                'max_grab_delay_ms': round(np.max(frame_grab_delays) * 1000, 2),
                                'std_grab_delay_ms': round(np.std(frame_grab_delays) * 1000, 2),
                            }

                        take_data = {
                            'version': 'studio-2.2',
                            'timestamp': datetime.now().isoformat(),
                            'duration_seconds': time.time() - record_start_time,
                            'frame_count': len(recorded_frames),
                            'predicted_frames': predicted_frames,
                            'dropped_frames': dropped_frames,
                            'total_outliers_filtered': total_outliers,
                            'tracking_stats': {
                                'total_frames': total_frames,
                                'cam_a_lost': cam_a_lost_count,
                                'cam_b_lost': cam_b_lost_count,
                                'both_lost': both_lost_count,
                            },
                            'calibration': {
                                'rms_stereo': calib_data.get('rms_stereo'),
                                'baseline': calib_data.get('baseline_meters'),
                                'floor_offset': calib_data.get('floor_z_offset', 0),
                            },
                            'capture_settings': {
                                'cam_a_index': cam_a,
                                'cam_b_index': cam_b,
                                'cam_a_native': f"{native_w_a}x{native_h_a}",
                                'cam_b_native': f"{native_w_b}x{native_h_b}",
                                'processing_resolution': f"{PROC_W}x{PROC_H}",
                                'mediapipe_complexity': MEDIAPIPE_MODEL_COMPLEXITY,
                                'mediapipe_detection_conf': MEDIAPIPE_MIN_DETECTION_CONF,
                                'mediapipe_tracking_conf': MEDIAPIPE_MIN_TRACKING_CONF,
                                'outlier_threshold_m': OUTLIER_THRESHOLD_M,
                                'min_visibility': MIN_VISIBILITY,
                                'grace_frames': TRACKING_GRACE_FRAMES,
                            },
                            'frame_timing': timing_stats,
                            'landmark_visibility': vis_avg_per_lm,
                            'frames': recorded_frames,
                        }

                        with open(filename, 'w') as f:
                            json.dump(take_data, f)

                        file_size_mb = filename.stat().st_size / 1024 / 1024

                        status = f"Saved: {filename.name} ({len(recorded_frames)}f)"

                        # Comprehensive save summary
                        log("\n" + "=" * 60)
                        log("TAKE SAVED")
                        log("=" * 60)
                        log(f"  File: {filename}")
                        log(f"  Size: {file_size_mb:.1f} MB")
                        log(f"  Duration: {time.time() - record_start_time:.1f}s")
                        log(f"  Recorded frames: {len(recorded_frames)}")
                        log(f"  Predicted frames: {predicted_frames}")
                        log(f"  Dropped frames: {dropped_frames}")
                        log(f"  FPS: {len(recorded_frames) / max(time.time() - record_start_time, 0.1):.1f}")
                        log(f"  Total outliers filtered: {total_outliers}")

                        log(f"\n  Camera tracking:")
                        log(f"    Total frames processed: {total_frames}")
                        log(f"    Camera A lost: {cam_a_lost_count} ({cam_a_lost_count*100//max(1,total_frames)}%)")
                        log(f"    Camera B lost: {cam_b_lost_count} ({cam_b_lost_count*100//max(1,total_frames)}%)")
                        log(f"    Both lost: {both_lost_count} ({both_lost_count*100//max(1,total_frames)}%)")

                        if timing_stats:
                            log(f"\n  Frame timing:")
                            log(f"    Mean grab delay: {timing_stats['mean_grab_delay_ms']:.2f}ms")
                            log(f"    Max grab delay: {timing_stats['max_grab_delay_ms']:.2f}ms")
                            log(f"    Std grab delay: {timing_stats['std_grab_delay_ms']:.2f}ms")

                        log(f"\n  Landmark visibility (sorted by coverage):")
                        sorted_vis = sorted(vis_avg_per_lm.items(),
                                          key=lambda x: x[1]['coverage_pct'], reverse=True)
                        for lm_key, data in sorted_vis:
                            log(f"    [{lm_key:>2s}] {data['name']:20s}: "
                                f"vis={data['avg_visibility']:.3f}  "
                                f"coverage={data['coverage_pct']:.1f}%  "
                                f"({data['frames_visible']}/{len(recorded_frames)} frames)")
                    else:
                        status = "No frames recorded"
                        log("No frames recorded")
                else:
                    countdown_active = True
                    countdown_start = time.time()
                    status = "GET READY..."
                    log("[REC] Countdown started")

    except Exception as e:
        log(f"FATAL ERROR: {e}", "ERROR")
        log(traceback.format_exc(), "ERROR")
    finally:
        cap_a.release()
        cap_b.release()
        cv2.destroyAllWindows()
        log("\n[DONE] Session ended")
        log_close()


if __name__ == "__main__":
    main()
