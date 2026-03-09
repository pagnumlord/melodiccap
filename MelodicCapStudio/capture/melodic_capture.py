"""
MelodicCap Studio - Motion Capture v3.0
========================================
Based on capture v8 (proven: 100% tracking, Kalman smoothing, 30-frame grace period).

v3.0 changes (from v2.2):
- Parallel camera grabs via ThreadedCamera (eliminates serial grab delay)
- Parallel MediaPipe processing via ThreadPoolExecutor (~2x FPS improvement)
- --lite flag for fast capture (model_complexity=0)
- --offline mode: process pre-recorded video clips with heavy model (complexity=2)
  - Flash sync: automatic frame synchronization between two video files
  - No time pressure, every frame processed, best quality
- Acceleration-based outlier detection (catches sudden velocity spikes)
- Tighter outlier thresholds: spatial 3.0→2.0m, hip velocity 6.0→4.0 m/s
- CLI argument parser (replaces positional-only cam indices)

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
import threading
import traceback
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

CAM_A_INDEX = 3          # Sony ZV-1F (MSMF, 1280x720)
CAM_B_INDEX = 0          # DroidCam USB (MSMF, 1920x1080)

# ChArUco board: 10x5 dual-sheet board (DICT_4X4_50)
# Measured: 1 and 11/16 inches = 42.86mm per square
CHARUCO_COLS = 10
CHARUCO_ROWS = 5
CHARUCO_SQUARE_M = 0.04286  # 42.86mm (measured: 1 11/16")
CHARUCO_MARKER_M = 0.0309   # 30.9mm (72% of square, proportional to 43mm->31mm design)

# Floor ChArUco board: smaller board for floor calibration (DICT_4X4_50, same as main)
# (easier to place on floor within both cameras' view)
# NOTE: Same dictionary as main board. Only have ONE board visible at a time
# during calibration to prevent marker ID collision.
FLOOR_CHARUCO_COLS = 4
FLOOR_CHARUCO_ROWS = 3
FLOOR_CHARUCO_SQUARE_M = 0.0635   # 63.5mm squares (2.5 inches, measured correct)
FLOOR_CHARUCO_MARKER_M = 0.0476   # 47.6mm markers (75% of square)

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
# Reduced from 3.0 to 2.0: indoor capture with stereo cameras at ~1-2m baseline
# should never produce valid landmarks beyond 2m from origin
OUTLIER_THRESHOLD_M = 2.0

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
    1. Velocity-based outlier rejection (per-landmark, adaptive to body part)
    2. Bone-length stabilization (enforce consistent skeleton, ALWAYS applied)
    3. Biomechanical joint-angle constraints (prevent impossible poses)
    4. Temporal smoothing (5-frame Gaussian-weighted, not simple average)
    5. Trim collapsed frames at end of take
    6. Floor clamping (feet can't go below z=0)
    7. Data quality report

    Modifies frames in-place and returns the (possibly trimmed) list.
    """
    if not frames or len(frames) < 5:
        return frames

    def _log(msg, level="INFO"):
        if log_fn:
            log_fn(msg, level)

    _log("=" * 50)
    _log("POST-PROCESSING TAKE v3.0")
    _log("=" * 50)

    n_original = len(frames)

    # Compute actual FPS from timestamps
    if len(frames) >= 2 and 'timestamp' in frames[0] and 'timestamp' in frames[-1]:
        dt = frames[-1]['timestamp'] - frames[0]['timestamp']
        actual_fps = (len(frames) - 1) / max(dt, 0.001)
    else:
        actual_fps = 13.0  # fallback

    _log(f"  Actual FPS: {actual_fps:.1f}")

    # --- Per-landmark max velocity thresholds (m/s) ---
    # Human biomechanics: hips move slowly, hands can move fast
    # Usain Bolt sprint: 12.4 m/s. Punching: ~15 m/s. Head: ~5 m/s.
    # Using generous limits to avoid false positives but catch the 27-1075 m/s spikes.
    MAX_VELOCITY_MPS = {
        # Head/face: slow
        '0': 5.0, '1': 5.0, '2': 5.0, '3': 5.0, '4': 5.0,
        '5': 5.0, '6': 5.0, '7': 5.0, '8': 5.0, '9': 5.0, '10': 5.0,
        # Shoulders: moderate
        '11': 8.0, '12': 8.0,
        # Elbows: moderate-fast
        '13': 12.0, '14': 12.0,
        # Wrists: fast (punching, gesturing)
        '15': 15.0, '16': 15.0,
        # Fingertips: fast
        '17': 15.0, '18': 15.0, '19': 15.0, '20': 15.0,
        # Hips: slow (core of body, center of mass can't move fast)
        '23': 4.0, '24': 4.0,
        # Knees: moderate
        '25': 10.0, '26': 10.0,
        # Ankles/feet: moderate
        '27': 10.0, '28': 10.0,
        '29': 10.0, '30': 10.0, '31': 10.0, '32': 10.0,
    }
    # Default for any unlisted landmark
    DEFAULT_MAX_VEL = 10.0

    # --- Step 1: Velocity-based outlier rejection ---
    _log("\n  [Step 1] Adaptive velocity-based outlier rejection")
    outlier_replacements = 0
    outlier_by_landmark = {}

    all_landmarks = set()
    for f in frames:
        all_landmarks.update(f.get('landmarks', {}).keys())

    prev_good = {}  # landmark_key -> [x, y, z]
    for fidx, fdata in enumerate(frames):
        lms = fdata.get('landmarks', {})
        for lm_key in list(lms.keys()):
            pos = lms[lm_key]
            max_vel = MAX_VELOCITY_MPS.get(lm_key, DEFAULT_MAX_VEL)
            # Convert m/s to m/frame
            max_jump = max_vel / actual_fps

            if lm_key in prev_good:
                pg = prev_good[lm_key]
                dx = pos[0] - pg[0]
                dy = pos[1] - pg[1]
                dz = pos[2] - pg[2]
                dist = (dx*dx + dy*dy + dz*dz) ** 0.5
                if dist > max_jump:
                    # Replace with last good position
                    lms[lm_key] = list(prev_good[lm_key])
                    outlier_replacements += 1
                    outlier_by_landmark[lm_key] = outlier_by_landmark.get(lm_key, 0) + 1
                else:
                    prev_good[lm_key] = list(pos)
            else:
                prev_good[lm_key] = list(pos)

    _log(f"    Replaced {outlier_replacements} outlier positions")
    if outlier_by_landmark:
        worst = sorted(outlier_by_landmark.items(), key=lambda x: -x[1])[:5]
        for lm_key, count in worst:
            name = LANDMARK_NAMES.get(int(lm_key), f"lm_{lm_key}")
            _log(f"      {name} ({lm_key}): {count} outliers")

    # --- Step 1b: Acceleration-based outlier detection ---
    _log("\n  [Step 1b] Acceleration-based outlier rejection")
    accel_replacements = 0

    # Max acceleration thresholds (m/s^2) by body part
    MAX_ACCEL = {
        '0': 80, '7': 80, '8': 80,                           # head
        '11': 100, '12': 100,                                  # shoulders
        '13': 150, '14': 150,                                  # elbows
        '15': 200, '16': 200,                                  # wrists
        '23': 50, '24': 50,                                    # hips
        '25': 100, '26': 100,                                  # knees
        '27': 100, '28': 100,                                  # ankles
    }
    DEFAULT_MAX_ACCEL = 150

    prev_vel = {}  # lm_key -> velocity vector from previous frame
    for fidx in range(1, len(frames)):
        lms = frames[fidx].get('landmarks', {})
        prev_lms = frames[fidx - 1].get('landmarks', {})

        for lm_key in list(lms.keys()):
            pos = lms.get(lm_key)
            prev_pos = prev_lms.get(lm_key)
            if not pos or not prev_pos:
                continue

            # Current velocity
            vel = [(pos[i] - prev_pos[i]) * actual_fps for i in range(3)]

            if lm_key in prev_vel:
                pv = prev_vel[lm_key]
                # Acceleration = change in velocity per second
                accel_sq = sum(((vel[i] - pv[i]) * actual_fps) ** 2 for i in range(3))
                max_a = MAX_ACCEL.get(lm_key, DEFAULT_MAX_ACCEL)
                if accel_sq > max_a * max_a:
                    # Replace with interpolation from neighbors
                    if fidx + 1 < len(frames):
                        next_lms = frames[fidx + 1].get('landmarks', {})
                        next_pos = next_lms.get(lm_key)
                        if next_pos:
                            lms[lm_key] = [(prev_pos[i] + next_pos[i]) / 2 for i in range(3)]
                        else:
                            lms[lm_key] = list(prev_pos)
                    else:
                        lms[lm_key] = list(prev_pos)
                    accel_replacements += 1
                    # Reset velocity to avoid cascade
                    vel = [(lms[lm_key][i] - prev_pos[i]) * actual_fps for i in range(3)]

            prev_vel[lm_key] = vel

    _log(f"    Replaced {accel_replacements} acceleration-spike positions")

    # --- Step 2: Bone-length stabilization (ALWAYS enforce, no tolerance) ---
    _log("\n  [Step 2] Bone-length stabilization (strict enforcement)")

    # Measure reference bone lengths from ALL frames (robust median)
    ref_lengths = {}
    for bone_start, bone_end, bone_name in SKELETON_BONES:
        lengths = []
        for f in frames:
            lms = f.get('landmarks', {})
            sk_s = str(bone_start)
            sk_e = str(bone_end)
            p1 = lms.get(sk_s) or lms.get(bone_start)
            p2 = lms.get(sk_e) or lms.get(bone_end)
            if p1 and p2:
                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                dz = p1[2] - p2[2]
                length = (dx*dx + dy*dy + dz*dz) ** 0.5
                if 0.01 < length < 2.0:  # sanity bounds
                    lengths.append(length)
        if lengths:
            lengths.sort()
            ref_lengths[bone_name] = lengths[len(lengths)//2]

    if ref_lengths:
        _log(f"    Reference bone lengths (median over all frames):")
        for name, length in ref_lengths.items():
            _log(f"      {name:15s}: {length:.3f}m")

    # Enforce bone lengths STRICTLY on every frame
    # Process bones in order: torso first, then limbs (parent before child)
    BONE_ORDER = [
        (23, 24, 'hips'),
        (11, 12, 'shoulders'),
        (11, 23, 'l_torso'),
        (12, 24, 'r_torso'),
        (11, 13, 'l_upper_arm'),
        (13, 15, 'l_forearm'),
        (12, 14, 'r_upper_arm'),
        (14, 16, 'r_forearm'),
        (23, 25, 'l_thigh'),
        (25, 27, 'l_shin'),
        (24, 26, 'r_thigh'),
        (26, 28, 'r_shin'),
    ]

    corrections = 0
    for fidx, fdata in enumerate(frames):
        lms = fdata.get('landmarks', {})

        for bone_start, bone_end, bone_name in BONE_ORDER:
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

            # ALWAYS enforce (no tolerance). Bones don't change length.
            if abs(current_len - target_len) > 0.001:
                scale = target_len / current_len
                new_p2 = [
                    p1[0] + dx * scale,
                    p1[1] + dy * scale,
                    p1[2] + dz * scale,
                ]
                key = sk_e if sk_e in lms else bone_end
                if key in lms:
                    lms[key] = new_p2
                    corrections += 1

    _log(f"    Applied {corrections} bone-length corrections")

    # --- Step 3: Biomechanical joint-angle constraints ---
    _log("\n  [Step 3] Biomechanical joint-angle constraints")
    angle_fixes = 0

    # Joint angle limits (degrees) — prevent hyperextension and impossible bends
    # Format: (parent_landmark, joint_landmark, child_landmark, min_angle, max_angle)
    JOINT_LIMITS = [
        # Elbows: can't hyperextend past ~170° or bend past ~30°
        (11, 13, 15, 25, 175, 'l_elbow'),
        (12, 14, 16, 25, 175, 'r_elbow'),
        # Knees: can't hyperextend past ~175° or overbend past ~30°
        (23, 25, 27, 25, 178, 'l_knee'),
        (24, 26, 28, 25, 178, 'r_knee'),
    ]

    import math

    def vec_angle_deg(a, b):
        """Angle between two vectors in degrees."""
        dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
        mag_a = (a[0]**2 + a[1]**2 + a[2]**2) ** 0.5
        mag_b = (b[0]**2 + b[1]**2 + b[2]**2) ** 0.5
        if mag_a < 0.001 or mag_b < 0.001:
            return 180.0
        cos_angle = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
        return math.degrees(math.acos(cos_angle))

    for fidx, fdata in enumerate(frames):
        lms = fdata.get('landmarks', {})

        for parent_idx, joint_idx, child_idx, min_ang, max_ang, name in JOINT_LIMITS:
            sp = str(parent_idx)
            sj = str(joint_idx)
            sc = str(child_idx)
            p_parent = lms.get(sp) or lms.get(parent_idx)
            p_joint = lms.get(sj) or lms.get(joint_idx)
            p_child = lms.get(sc) or lms.get(child_idx)

            if not p_parent or not p_joint or not p_child:
                continue

            # Vectors from joint to parent and child
            v_to_parent = [p_parent[i] - p_joint[i] for i in range(3)]
            v_to_child = [p_child[i] - p_joint[i] for i in range(3)]

            angle = vec_angle_deg(v_to_parent, v_to_child)

            if angle < min_ang or angle > max_ang:
                # Clamp by moving the child along its current direction
                # to the nearest valid angle
                target_angle = max(min_ang, min(max_ang, angle))
                # For simplicity, blend child position toward straight (180°)
                # or bent (min_ang) based on which limit was violated
                if angle < min_ang:
                    # Over-bent: push child slightly toward straight
                    blend = 0.3  # conservative correction
                elif angle > max_ang:
                    # Hyperextended: pull child back slightly
                    blend = 0.3

                # Move child 30% of the way toward a valid configuration
                # by interpolating toward the parent-to-joint vector extended
                child_len = (v_to_child[0]**2 + v_to_child[1]**2 + v_to_child[2]**2) ** 0.5
                if child_len > 0.001:
                    # Use previous frame's child position as reference if available
                    if fidx > 0:
                        prev_lms = frames[fidx-1].get('landmarks', {})
                        prev_child = prev_lms.get(sc) or prev_lms.get(child_idx)
                        if prev_child:
                            new_child = [
                                p_child[i] * (1 - blend) + prev_child[i] * blend
                                for i in range(3)
                            ]
                            key = sc if sc in lms else child_idx
                            if key in lms:
                                lms[key] = new_child
                                angle_fixes += 1

    _log(f"    Fixed {angle_fixes} joint-angle violations")

    # --- Step 4: Temporal smoothing (5-frame Gaussian-weighted) ---
    _log("\n  [Step 4] Temporal smoothing (5-frame Gaussian-weighted)")
    if len(frames) >= 5:
        # Gaussian weights for 5-frame window: [1, 4, 6, 4, 1] / 16
        weights = [1.0/16, 4.0/16, 6.0/16, 4.0/16, 1.0/16]

        all_keys = set()
        for f in frames:
            all_keys.update(f.get('landmarks', {}).keys())

        for lm_key in all_keys:
            positions = []
            for f in frames:
                lms = f.get('landmarks', {})
                positions.append(lms.get(lm_key))

            smoothed = [None] * len(positions)
            for i in range(len(positions)):
                if positions[i] is None:
                    smoothed[i] = None
                    continue

                # Collect up to 5-frame window
                w_sum = 0.0
                avg = [0.0, 0.0, 0.0]
                for offset in range(-2, 3):
                    j = i + offset
                    if 0 <= j < len(positions) and positions[j] is not None:
                        w = weights[offset + 2]
                        avg[0] += positions[j][0] * w
                        avg[1] += positions[j][1] * w
                        avg[2] += positions[j][2] * w
                        w_sum += w

                if w_sum > 0:
                    smoothed[i] = [avg[0]/w_sum, avg[1]/w_sum, avg[2]/w_sum]

            for i, f in enumerate(frames):
                lms = f.get('landmarks', {})
                if smoothed[i] is not None and lm_key in lms:
                    lms[lm_key] = smoothed[i]

        _log(f"    Smoothed {len(all_keys)} landmarks across {len(frames)} frames")

    # --- Step 5: Re-enforce bone lengths after smoothing ---
    # Smoothing can drift bone lengths slightly; re-enforce to keep skeleton rigid
    _log("\n  [Step 5] Re-enforce bone lengths after smoothing")
    re_corrections = 0
    for fidx, fdata in enumerate(frames):
        lms = fdata.get('landmarks', {})
        for bone_start, bone_end, bone_name in BONE_ORDER:
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
            if abs(current_len - target_len) > 0.001:
                scale = target_len / current_len
                new_p2 = [p1[0] + dx * scale, p1[1] + dy * scale, p1[2] + dz * scale]
                key = sk_e if sk_e in lms else bone_end
                if key in lms:
                    lms[key] = new_p2
                    re_corrections += 1
    _log(f"    Applied {re_corrections} post-smoothing bone corrections")

    # --- Step 6: Trim collapsed frames ---
    _log("\n  [Step 6] Trim collapsed frames (skeleton height < 70% of reference)")

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

    # --- Step 7: Floor clamping ---
    _log("\n  [Step 7] Floor clamping (feet z >= 0)")
    floor_clamps = 0
    foot_landmarks = ['27', '28', '29', '30', '31', '32']
    for f in frames:
        lms = f.get('landmarks', {})
        for lm_key in foot_landmarks:
            if lm_key in lms and lms[lm_key][2] < 0:
                lms[lm_key][2] = 0.0
                floor_clamps += 1

    _log(f"    Clamped {floor_clamps} foot positions to floor")

    # --- Step 8: Data quality report ---
    _log(f"\n  DATA QUALITY REPORT:")
    _log(f"    Input frames:  {n_original}")
    _log(f"    Output frames: {len(frames)}")
    _log(f"    Velocity outliers fixed: {outlier_replacements}")
    _log(f"    Acceleration outliers fixed: {accel_replacements}")
    _log(f"    Bone corrections: {corrections} (initial) + {re_corrections} (post-smooth)")
    _log(f"    Joint-angle fixes: {angle_fixes}")
    _log(f"    Floor clamps: {floor_clamps}")

    # Compute final bone-length consistency as quality metric
    if ref_lengths and len(frames) > 0:
        _log(f"\n    Final bone-length consistency (should all be ~0%):")
        for bone_start, bone_end, bone_name in SKELETON_BONES:
            if bone_name not in ref_lengths:
                continue
            lengths = []
            for f in frames:
                lms = f.get('landmarks', {})
                p1 = lms.get(str(bone_start))
                p2 = lms.get(str(bone_end))
                if p1 and p2:
                    dx = p1[0]-p2[0]; dy = p1[1]-p2[1]; dz = p1[2]-p2[2]
                    lengths.append((dx*dx + dy*dy + dz*dz) ** 0.5)
            if lengths:
                mean_l = sum(lengths) / len(lengths)
                if mean_l > 0:
                    cv = ((sum((l-mean_l)**2 for l in lengths) / len(lengths)) ** 0.5) / mean_l * 100
                    _log(f"      {bone_name:15s}: {mean_l:.3f}m  CV={cv:.1f}%")

    # Quality score: how many outliers per second of capture
    duration = frames[-1].get('timestamp', 0) - frames[0].get('timestamp', 0) if len(frames) > 1 else 1.0
    outlier_rate = outlier_replacements / max(duration, 0.1)
    if outlier_rate > 10:
        _log(f"\n    QUALITY: POOR ({outlier_rate:.0f} outliers/sec) — recalibrate with larger board", "WARN")
    elif outlier_rate > 3:
        _log(f"\n    QUALITY: FAIR ({outlier_rate:.0f} outliers/sec) — usable but noisy", "WARN")
    else:
        _log(f"\n    QUALITY: GOOD ({outlier_rate:.1f} outliers/sec)")

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

# Floor board (separate smaller board for floor calibration)
# Uses same DICT_4X4_50 as main board. Only show ONE board at a time during calibration.
# Create both orientations — detection tries both and picks the winner
floor_board_a = cv2.aruco.CharucoBoard(
    (FLOOR_CHARUCO_COLS, FLOOR_CHARUCO_ROWS),
    FLOOR_CHARUCO_SQUARE_M,
    FLOOR_CHARUCO_MARKER_M,
    aruco_dict
)
floor_board_b = cv2.aruco.CharucoBoard(
    (FLOOR_CHARUCO_ROWS, FLOOR_CHARUCO_COLS),  # swapped orientation
    FLOOR_CHARUCO_SQUARE_M,
    FLOOR_CHARUCO_MARKER_M,
    aruco_dict
)
FLOOR_BOARD_MAX_CORNERS = (FLOOR_CHARUCO_COLS - 1) * (FLOOR_CHARUCO_ROWS - 1)

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


def detect_charuco_floor(frame, debug=False):
    """Detect ChArUco corners using the smaller floor board.
    Both boards use DICT_4X4_50 — only show ONE board at a time.
    Tries both orientations and picks whichever finds more corners."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) < 2:
        if debug:
            log(f"    [FLOOR DBG] No ArUco markers found")
        return None, None, 0

    if debug:
        log(f"    [FLOOR DBG] Found {len(ids)} ArUco markers, IDs: {sorted(ids.flatten().tolist())}")

    # Try both orientations
    best_num = 0
    best_corners = None
    best_ids = None
    best_label = None

    for label, fb in [("4x3", floor_board_a), ("3x4", floor_board_b)]:
        num, cc, ci = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, fb)
        if num is not None and num > best_num:
            best_num = num
            best_corners = cc
            best_ids = ci
            best_label = label

    if debug:
        log(f"    [FLOOR DBG] Best orientation: {best_label}, charuco corners: {best_num}")

    if best_num < 4:
        if debug:
            log(f"    [FLOOR DBG] Need >= 4 charuco corners, only got {best_num}")
        return None, None, len(ids)

    return best_corners, best_ids, len(ids)


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

        # Limit distortion model based on available data.
        # With few corners, high-order distortion coefficients (k3) overfit to noise.
        # Total corners across all frames:
        total_corners = sum(len(d['common']) for d in active_data)
        if total_corners < 200:
            # Not enough data for 5-parameter distortion — fix k3=0
            # This prevents the k3=-17 overfitting seen with small boards
            cal_flags = cv2.CALIB_FIX_K3
            log(f"  Using 4-param distortion model (k3=0): only {total_corners} total corner observations")
        else:
            cal_flags = 0
            log(f"  Using full 5-param distortion model: {total_corners} total corner observations")

        ret_a, K1, D1, _, _ = cv2.aruco.calibrateCameraCharuco(
            corners_a, ids_a, board, img_size, None, None,
            flags=cal_flags
        )
        ret_b, K2, D2, _, _ = cv2.aruco.calibrateCameraCharuco(
            corners_b, ids_b, board, img_size, None, None,
            flags=cal_flags
        )
        log(f"  Camera A RMS: {ret_a:.4f} (fx={K1[0,0]:.1f}, fy={K1[1,1]:.1f})")
        log(f"  Camera B RMS: {ret_b:.4f} (fx={K2[0,0]:.1f}, fy={K2[1,1]:.1f})")

        # Check for extreme distortion coefficients (sign of bad calibration)
        for cam_name, D in [("A", D1), ("B", D2)]:
            d = D.flatten()
            if len(d) >= 5 and abs(d[4]) > 10.0:
                log(f"  WARNING: Camera {cam_name} k3={d[4]:.1f} is extreme — calibration may be overfitting", "WARN")

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
    """Calibrate floor from smaller board lying flat on ground."""
    log("  [FLOOR] Attempting floor calibration...")
    cc_a, ci_a, _ = detect_charuco_floor(frame_a, debug=True)
    cc_b, ci_b, _ = detect_charuco_floor(frame_b, debug=True)

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
# THREADED CAMERA (parallel frame grabs)
# =============================================================================

class ThreadedCamera:
    """Wraps a cv2.VideoCapture to grab frames in a background thread.

    Eliminates serial grab delay — both cameras are read simultaneously.
    The main loop calls read() to get the latest frame instantly.
    """
    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.ret = False
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.thread.join(timeout=2.0)
        self.cap.release()

    def get(self, prop):
        return self.cap.get(prop)

    def set(self, prop, val):
        return self.cap.set(prop, val)

    def isOpened(self):
        return self.cap.isOpened()


# =============================================================================
# OFFLINE CLIP-BASED PROCESSING
# =============================================================================

def find_flash_frame(cap, threshold=50, max_frames=300):
    """Scan for a bright flash frame to use as sync point.

    Returns the frame index with the largest brightness spike, or 0 if none found.
    """
    prev_brightness = 0
    best_delta = 0
    best_frame = 0

    for frame_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        brightness = float(np.mean(frame))
        delta = brightness - prev_brightness
        if delta > best_delta and delta > threshold:
            best_delta = delta
            best_frame = frame_idx
        prev_brightness = brightness

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind
    return best_frame if best_delta > threshold else 0


def process_offline(video_a_path, video_b_path, calib_path=None):
    """Process two pre-recorded video files for motion capture.

    Uses the heavy MediaPipe model (complexity=2) with no time pressure,
    processes every frame, and applies full post-processing pipeline.

    Args:
        video_a_path: Path to Camera A video file
        video_b_path: Path to Camera B video file
        calib_path: Optional path to calibration JSON (default: auto-detect)
    """
    log_init("offline")

    log("=" * 60)
    log("MELODICCAP STUDIO - OFFLINE PROCESSING")
    log("=" * 60)

    # Load calibration
    cal_file = Path(calib_path) if calib_path else CALIBRATION_FILE
    if not cal_file.exists():
        log(f"Calibration file not found: {cal_file}", "ERROR")
        log("Run live calibration first (C then S), then use --offline", "ERROR")
        log_close()
        return

    with open(cal_file, 'r') as f:
        calib_data = json.load(f)

    log(f"  Calibration: RMS={calib_data.get('rms_stereo', '?'):.4f}, "
        f"baseline={calib_data.get('baseline_meters', '?'):.3f}m")

    # Open video files
    cap_a = cv2.VideoCapture(video_a_path)
    cap_b = cv2.VideoCapture(video_b_path)

    if not cap_a.isOpened():
        log(f"Failed to open video A: {video_a_path}", "ERROR")
        log_close()
        return
    if not cap_b.isOpened():
        log(f"Failed to open video B: {video_b_path}", "ERROR")
        cap_a.release()
        log_close()
        return

    fps_a = cap_a.get(cv2.CAP_PROP_FPS)
    fps_b = cap_b.get(cv2.CAP_PROP_FPS)
    total_a = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
    total_b = int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT))
    w_a = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_a = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_b = int(cap_b.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_b = int(cap_b.get(cv2.CAP_PROP_FRAME_HEIGHT))

    log(f"  Video A: {video_a_path}")
    log(f"    {w_a}x{h_a} @ {fps_a:.1f}fps, {total_a} frames")
    log(f"  Video B: {video_b_path}")
    log(f"    {w_b}x{h_b} @ {fps_b:.1f}fps, {total_b} frames")

    # Determine processing resolution (match calibration)
    cal_size = calib_data.get('image_size', [w_a, h_a])
    PROC_W, PROC_H = cal_size[0], cal_size[1]
    log(f"  Processing resolution: {PROC_W}x{PROC_H} (from calibration)")

    # Flash sync: find sync point in both videos
    log("\n  [SYNC] Scanning for flash sync...")
    sync_a = find_flash_frame(cap_a)
    sync_b = find_flash_frame(cap_b)

    if sync_a > 0 or sync_b > 0:
        log(f"    Flash detected: Video A frame {sync_a}, Video B frame {sync_b}")
        cap_a.set(cv2.CAP_PROP_POS_FRAMES, sync_a)
        cap_b.set(cv2.CAP_PROP_POS_FRAMES, sync_b)
        remaining_a = total_a - sync_a
        remaining_b = total_b - sync_b
    else:
        log(f"    No flash detected — assuming simultaneous start")
        remaining_a = total_a
        remaining_b = total_b

    total_frames = min(remaining_a, remaining_b)
    log(f"    Processing {total_frames} synchronized frames")

    # Initialize MediaPipe with heavy model (no time pressure)
    mp_pose = mp.solutions.pose
    pose_a = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,  # Heavy model for best accuracy
        min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONF,
        min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONF
    )
    pose_b = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONF,
        min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONF
    )

    # Kalman filters for smoothing
    kalman_filters = {}
    recorded_frames = []
    all_visibility = []
    total_outliers = 0
    start_time = time.time()

    log(f"\n  Processing frames...")

    for frame_idx in range(total_frames):
        ret_a, frame_a = cap_a.read()
        ret_b, frame_b = cap_b.read()

        if not ret_a or not ret_b:
            break

        # Resize to calibration resolution
        if frame_a.shape[1] != PROC_W or frame_a.shape[0] != PROC_H:
            frame_a = cv2.resize(frame_a, (PROC_W, PROC_H))
        if frame_b.shape[1] != PROC_W or frame_b.shape[0] != PROC_H:
            frame_b = cv2.resize(frame_b, (PROC_W, PROC_H))

        # Process poses
        rgb_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB)
        rgb_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2RGB)
        results_a = pose_a.process(rgb_a)
        results_b = pose_b.process(rgb_b)

        if results_a.pose_landmarks and results_b.pose_landmarks:
            img_size = (PROC_W, PROC_H)
            points_3d, vis_data, outlier_count = triangulate_landmarks(
                results_a.pose_landmarks,
                results_b.pose_landmarks,
                calib_data,
                img_size
            )
            total_outliers += outlier_count

            # Kalman smoothing
            smoothed = {}
            for idx, pt in points_3d.items():
                if idx not in kalman_filters:
                    kalman_filters[idx] = KalmanFilter3D()
                filtered = kalman_filters[idx].update(pt)
                if filtered is not None:
                    smoothed[idx] = filtered

            if len(smoothed) > 0:
                timestamp = frame_idx / fps_a  # Use video A's fps for timing
                recorded_frames.append({
                    'timestamp': timestamp,
                    'landmarks': smoothed,
                    'num_landmarks': len(smoothed),
                    'predicted': False,
                    'outliers_this_frame': outlier_count,
                })
                all_visibility.append(vis_data)

        # Progress logging
        if frame_idx % 100 == 0 and frame_idx > 0:
            elapsed = time.time() - start_time
            fps_proc = frame_idx / elapsed
            eta = (total_frames - frame_idx) / fps_proc
            log(f"    Frame {frame_idx}/{total_frames} ({fps_proc:.1f} fps, ETA {eta:.0f}s)")

    processing_time = time.time() - start_time
    log(f"\n  Processing complete: {len(recorded_frames)} frames in {processing_time:.1f}s "
        f"({len(recorded_frames)/max(processing_time, 0.1):.1f} fps)")

    cap_a.release()
    cap_b.release()
    pose_a.close()
    pose_b.close()

    if len(recorded_frames) < 5:
        log("Too few frames captured — check videos and calibration", "ERROR")
        log_close()
        return

    # Post-process
    recorded_frames = postprocess_take(recorded_frames, log_fn=log)

    # Save take
    TAKES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = TAKES_DIR / f"take_offline_{timestamp}.json"

    # Visibility summary
    vis_summary = {}
    for vis_frame in all_visibility:
        for lm_idx, vis in vis_frame.items():
            if lm_idx not in vis_summary:
                vis_summary[lm_idx] = {'count': 0, 'total_avg': 0}
            vis_summary[lm_idx]['count'] += 1
            vis_summary[lm_idx]['total_avg'] += vis['vis_avg']

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

    take_data = {
        'version': 'studio-2.2-offline',
        'timestamp': datetime.now().isoformat(),
        'capture_mode': 'offline',
        'source_videos': {
            'video_a': str(video_a_path),
            'video_b': str(video_b_path),
        },
        'duration_seconds': recorded_frames[-1]['timestamp'] if recorded_frames else 0,
        'frame_count': len(recorded_frames),
        'predicted_frames': 0,
        'dropped_frames': total_frames - len(recorded_frames),
        'total_outliers_filtered': total_outliers,
        'calibration': {
            'rms_stereo': calib_data.get('rms_stereo'),
            'baseline': calib_data.get('baseline_meters'),
            'floor_offset': calib_data.get('floor_z_offset', 0),
        },
        'capture_settings': {
            'processing_resolution': f"{PROC_W}x{PROC_H}",
            'mediapipe_complexity': 2,
            'outlier_threshold_m': OUTLIER_THRESHOLD_M,
            'min_visibility': MIN_VISIBILITY,
        },
        'processing_stats': {
            'processing_time_seconds': round(processing_time, 1),
            'processing_fps': round(len(recorded_frames) / max(processing_time, 0.1), 1),
        },
        'landmark_visibility': vis_avg_per_lm,
        'frames': recorded_frames,
    }

    with open(filename, 'w') as f:
        json.dump(take_data, f)

    file_size_mb = filename.stat().st_size / 1024 / 1024

    log("\n" + "=" * 60)
    log("OFFLINE TAKE SAVED")
    log("=" * 60)
    log(f"  File: {filename}")
    log(f"  Size: {file_size_mb:.1f} MB")
    log(f"  Frames: {len(recorded_frames)}")
    log(f"  Duration: {take_data['duration_seconds']:.1f}s")
    log(f"  Processing time: {processing_time:.1f}s")
    log(f"  Outliers filtered: {total_outliers}")

    log_close()
    print(f"\nSaved: {filename}")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='MelodicCap Studio - Motion Capture')
    parser.add_argument('cam_a', nargs='?', type=int, default=CAM_A_INDEX,
                        help=f'Camera A index (default: {CAM_A_INDEX})')
    parser.add_argument('cam_b', nargs='?', type=int, default=CAM_B_INDEX,
                        help=f'Camera B index (default: {CAM_B_INDEX})')
    parser.add_argument('--lite', action='store_true',
                        help='Use lite MediaPipe model (faster, less accurate)')
    parser.add_argument('--offline', nargs=2, metavar=('VIDEO_A', 'VIDEO_B'),
                        help='Offline mode: process two video files instead of live cameras')
    parser.add_argument('--calib', type=str, default=None,
                        help='Path to calibration JSON (default: auto-detect)')

    args = parser.parse_args()
    cam_a = args.cam_a
    cam_b = args.cam_b

    # Handle offline mode
    if args.offline:
        process_offline(args.offline[0], args.offline[1], args.calib)
        return

    # Lite mode overrides model complexity
    model_complexity = 0 if args.lite else MEDIAPIPE_MODEL_COMPLEXITY

    # Initialize logging
    log_init("session")

    log("=" * 60)
    log("MELODICCAP STUDIO - MOTION CAPTURE v3.0")
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
    log(f"    Floor board: {FLOOR_CHARUCO_COLS}x{FLOOR_CHARUCO_ROWS} ({FLOOR_BOARD_MAX_CORNERS} max corners, {FLOOR_CHARUCO_SQUARE_M*1000:.1f}mm squares)")
    if BOARD_MAX_CORNERS < 12:
        log(f"    NOTE: Small board ({BOARD_MAX_CORNERS} corners). Calibration uses joint optimization mode.", "WARN")
        log(f"    TIPS: Move board SLOWLY, ensure {BOARD_MAX_CORNERS - 1}+ corners visible in BOTH cameras,", "WARN")
        log(f"          cover many angles (tilt, rotate), collect 30+ frames.", "WARN")

    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    # Separate pose detectors for each camera
    log(f"  MediaPipe model complexity: {model_complexity}" +
        (" (LITE)" if model_complexity == 0 else ""))
    pose_a = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONF,
        min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONF
    )
    pose_b = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONF,
        min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONF
    )

    # Open cameras
    log("\n[CAMERAS]")

    log(f"  Opening Camera A (index {cam_a}, MSMF)...")
    cap_a = cv2.VideoCapture(cam_a, cv2.CAP_MSMF)
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

    # Wrap cameras in threaded readers for parallel grabs during recording
    threaded_a = ThreadedCamera(cap_a)
    threaded_b = ThreadedCamera(cap_b)

    # Thread pool for parallel MediaPipe processing
    mp_executor = ThreadPoolExecutor(max_workers=2)

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
            # Read frames from threaded cameras (near-simultaneous)
            t_grab_a = time.time()
            ret_a, frame_a = threaded_a.read()
            ret_b, frame_b = threaded_b.read()
            t_grab_done = time.time()

            if not ret_a or not ret_b or frame_a is None or frame_b is None:
                continue

            # Resize to common processing resolution if needed
            if resize_a:
                frame_a = cv2.resize(frame_a, (PROC_W, PROC_H))
            if resize_b:
                frame_b = cv2.resize(frame_b, (PROC_W, PROC_H))

            frame_count += 1
            grab_delay = t_grab_done - t_grab_a

            if recording:
                frame_grab_delays.append(grab_delay)

            # Detect ChArUco (skip during recording to save CPU)
            if not recording:
                if floor_mode:
                    # Use smaller floor board for floor calibration
                    cc_a, ci_a, markers_a = detect_charuco_floor(frame_a)
                    cc_b, ci_b, markers_b = detect_charuco_floor(frame_b)
                else:
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
                # Parallel MediaPipe processing via thread pool
                rgb_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB)
                rgb_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2RGB)
                future_a = mp_executor.submit(pose_a.process, rgb_a)
                future_b = mp_executor.submit(pose_b.process, rgb_b)
                results_a = future_a.result()
                results_b = future_b.result()

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
                                # Save raw 2D landmarks for potential re-triangulation
                                raw_2d = {'cam_a': {}, 'cam_b': {}}
                                for idx in range(33):
                                    lm_a = results_a.pose_landmarks.landmark[idx]
                                    lm_b = results_b.pose_landmarks.landmark[idx]
                                    raw_2d['cam_a'][str(idx)] = [
                                        round(float(lm_a.x), 5),
                                        round(float(lm_a.y), 5),
                                        round(float(lm_a.z), 5),
                                        round(float(lm_a.visibility), 3)
                                    ]
                                    raw_2d['cam_b'][str(idx)] = [
                                        round(float(lm_b.x), 5),
                                        round(float(lm_b.y), 5),
                                        round(float(lm_b.z), 5),
                                        round(float(lm_b.visibility), 3)
                                    ]

                                recorded_frames.append({
                                    'timestamp': time.time() - record_start_time,
                                    'landmarks': smoothed,
                                    'raw_2d': raw_2d,
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
            if floor_mode and calib_data:
                if board_a and board_b:
                    success, msg = calibrate_floor(frame_a, frame_b, calib_data)
                    if success:
                        floor_mode = False
                        status = f"Floor OK: {msg}"
                        log(f"[FLOOR] {msg}")
                    else:
                        status = f"Floor FAILED: {msg}"
                        log(f"[FLOOR] Failed: {msg}", "WARN")
                else:
                    missing = []
                    if not board_a:
                        missing.append(f"A({markers_a or 0}mkr)")
                    if not board_b:
                        missing.append(f"B({markers_b or 0}mkr)")
                    status = f"FLOOR: Not enough corners in {' '.join(missing)} - need 4+ charuco corners"

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
                            'version': 'studio-3.0',
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
        mp_executor.shutdown(wait=False)
        threaded_a.release()
        threaded_b.release()
        cv2.destroyAllWindows()
        log("\n[DONE] Session ended")
        log_close()


if __name__ == "__main__":
    main()
