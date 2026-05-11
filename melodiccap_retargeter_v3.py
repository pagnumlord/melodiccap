"""
MelodicCap Retargeter v3.0
==========================
Based on v2.0, with three critical improvements:

v3.0 PIPELINE FIXES:
1. SMART REFERENCE FRAME SELECTION
   Frame 0 is often garbage (person still settling, landmarks jittering).
   Now scans first N frames and picks the best A-pose reference based on:
   - Landmark completeness (all 33 present?)
   - Shoulder/hip symmetry (balanced A-pose)
   - Upright posture (spine vertical)
   - Stillness (low inter-frame velocity = stable pose)
   Falls back to frame 0 if no good frame found.

2. BUTTERWORTH TEMPORAL SMOOTHING
   Raw MediaPipe landmarks have 3-5 pixel jitter → 2-5cm 3D noise.
   After all keyframes are inserted, applies a 2nd-order Butterworth low-pass
   filter to every FCurve. Configurable cutoff frequency (default 6 Hz).
   Implemented without scipy using the bilinear transform (runs inside Blender).
   Separate cutoff for location (6 Hz) and rotation (4 Hz) channels.

3. LANDMARK GAP INTERPOLATION
   Takes often have 15-85% missing landmarks per frame. Previously, missing
   frames were simply skipped, leaving gaps in the animation.
   Now interpolates missing landmarks from neighboring frames using cubic
   interpolation when gaps are short (< 10 frames), and marks long gaps
   for hold-last-good behavior.

All v2.0 features preserved:
- v4's proven delta-from-reference IK approach (mathematically correct)
- AntiGrav V3's V2R (Vector-to-Rotation) FK method
- Pole targets for correct elbow/knee direction (from Keemap/AntiGrav approach)
- IK target rotation for wrist/foot orientation
- Ground clamping (feet can't go through floor)
- Smart foot pinning (reduces sliding when feet should be planted)
- 4-segment spine animation via virtual midpoints
- Outlier filtering (catches MediaPipe landmark spikes)
- Torso ROTATION from hip orientation (body yaw/tilt)
- Spine TWIST distribution (shoulder twist relative to hips)
- Velocity-based foot contact detection with hysteresis state machine
- Quaternion continuity enforcement on ALL rotation channels
- Relaxed finger rest pose (natural curl, static keyframe on start frame)
- Head/neck animation from nose + ear landmarks (V2R facing direction)

For Blender 4.4+ with JaxRigify armature.

KEY DESIGN DECISIONS:
- NO mirroring: Person's LEFT = Character's LEFT (both at +X in capture/Blender coords)
  Person faces camera (-Y), character's .L bones are at +X — same side.
- NO X-axis negation (v5/v12.1 proved this double-mirrors)
- Data is already in Blender coordinates from the capture script
- IK targets use delta-from-reference (includes hip movement naturally)
- Pole targets use 3-point projection (Keemap algorithm) with delta-from-reference
- FK rest axes from ACTUAL JaxRigify bone dump (A-pose: arms hang DOWN, not T-pose)
- IK rotation rest axes from actual bone directions (hand_ik=-Z, foot_ik=+Y)

v1.5 TORSO ROTATION + SPINE TWIST:
- Torso bone now gets ROTATION keyframes in addition to location.
  Rotation is derived from hip orientation: the vector from right_hip to left_hip
  gives the body's lateral axis. Yaw (horizontal turn) and tilt are computed as
  the angular change from the reference frame's hip orientation.
  This is CRITICAL — without it the character never turns when you turn your body.
- Spine twist: The old virtual midpoint approach produced COLLINEAR points, so all
  4 spine segments got identical rotations (just the overall lean direction).
  Now we compute the twist between shoulder line and hip line, and distribute it
  across the spine segments. Lower segments follow hips, upper follow shoulders.
- Enhanced frame-by-frame debugging: hip yaw, shoulder yaw, twist angle logged.

v1.4 OUTLIER FILTERING:
- Velocity-based pre-filter on raw landmarks before animation.
  MediaPipe sometimes outputs garbage positions (landmarks jumping 30-80m in a
  single frame). The pre-filter scans all frames, tracks per-landmark velocity,
  and replaces outlier positions with the last known good position.
  Threshold: configurable max velocity (default 10 m/s) adapted to capture FPS.
  Fixes IK targets, pole targets, FK rotations, and spine all at once since
  they all derive from the same landmark data.

v1.3 CRITICAL FIXES (from Rigify property diagnostics):
- IK_parent set to 0 (root space) during import. Default IK_parent=1 makes IK
  targets follow torso via parent chain, but our delta already includes hip
  displacement → double root motion (arms fly off during walking).
  Root space (0) means IK targets are independent of torso movement.
- pole_vector enabled (True) during import. Default is False, which means the
  IK solver ignores pole target bone positions entirely.
- pole_parent set to 0 (root space) for same reason as IK_parent.
"""

bl_info = {
    "name": "MelodicCap Retargeter",
    "author": "Karsten / MelodicCap Studio",
    "version": (3, 0, 0),
    "blender": (4, 4, 0),
    "location": "View3D > Sidebar > MelodicCap",
    "description": "Import MelodicCap motion capture data to JaxRigify armature",
    "category": "Animation",
}

import bpy
import json
import os
import math
import copy
import datetime
from pathlib import Path
from mathutils import Vector, Matrix, Quaternion
from bpy.props import StringProperty, FloatProperty, BoolProperty, IntProperty
from bpy_extras.io_utils import ImportHelper

# =============================================================================
# LANDMARK DEFINITIONS (MediaPipe 33 body landmarks)
# =============================================================================

LANDMARKS = {
    0: "nose", 7: "left_ear", 8: "right_ear",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow", 15: "left_wrist", 16: "right_wrist",
    19: "left_index", 20: "right_index",
    23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
    29: "left_heel", 30: "right_heel", 31: "left_foot_index", 32: "right_foot_index",
}

# =============================================================================
# BONE MAPPING
# Person's LEFT = Character's LEFT (NO mirroring)
# Both the performer and JaxRigify have LEFT at +X in the shared coordinate space.
# (Person faces camera at -Y; character's .L bones verified at +X)
# =============================================================================

# IK targets for hand/foot position
IK_TARGETS = {
    'hand_ik.L': 15,   # Person's left wrist -> Character's left hand
    'hand_ik.R': 16,   # Person's right wrist -> Character's right hand
    'foot_ik.L': 27,   # Person's left ankle -> Character's left foot
    'foot_ik.R': 28,   # Person's right ankle -> Character's right foot
}

# FK bone chains for limb rotation (V2R: start landmark -> end landmark)
FK_CHAINS = {
    # Person's LEFT -> Character's LEFT
    'upper_arm_fk.L': (11, 13),   # left shoulder -> left elbow
    'forearm_fk.L':   (13, 15),   # left elbow -> left wrist
    'thigh_fk.L':     (23, 25),   # left hip -> left knee
    'shin_fk.L':      (25, 27),   # left knee -> left ankle
    # Person's RIGHT -> Character's RIGHT
    'upper_arm_fk.R': (12, 14),   # right shoulder -> right elbow
    'forearm_fk.R':   (14, 16),   # right elbow -> right wrist
    'thigh_fk.R':     (24, 26),   # right hip -> right knee
    'shin_fk.R':      (26, 28),   # right knee -> right ankle
}

# Pole targets for IK elbow/knee direction (3-point: root, mid, end)
# Bone names verified from JaxRigify diagnostic dump
POLE_TARGETS = {
    'upper_arm_ik_target.L': (11, 13, 15),  # Person's L shoulder→elbow→wrist
    'upper_arm_ik_target.R': (12, 14, 16),  # Person's R shoulder→elbow→wrist
    'thigh_ik_target.L': (23, 25, 27),       # Person's L hip→knee→ankle
    'thigh_ik_target.R': (24, 26, 28),       # Person's R hip→knee→ankle
}

# IK target rotation mapping (for wrist/foot orientation)
# Rest axes from actual JaxRigify bone dump:
#   hand_ik.L dir=( 0.147,-0.048,-0.988)  hand_ik.R dir=(-0.077,-0.128,-0.989)
#   foot_ik.L dir=( 0.000, 1.000, 0.000)  foot_ik.R dir=( 0.000, 1.000, 0.000)
IK_ROTATION = {
    'hand_ik.L': (15, 19, Vector(( 0.147, -0.048, -0.988))),  # L wrist→index finger
    'hand_ik.R': (16, 20, Vector((-0.077, -0.128, -0.989))),  # R wrist→index finger
    'foot_ik.L': (27, 31, Vector((0, 1, 0))),                  # L ankle→foot_index (toe dir)
    'foot_ik.R': (28, 32, Vector((0, 1, 0))),                  # R ankle→foot_index (toe dir)
}

# Fallback IK rotation: used when primary landmarks aren't available
IK_ROTATION_FALLBACK = {
    'hand_ik.L': (13, 15, Vector(( 0.147, -0.048, -0.988))),  # L forearm dir
    'hand_ik.R': (14, 16, Vector((-0.077, -0.128, -0.989))),  # R forearm dir
    'foot_ik.L': (25, 27, Vector((0, 1, 0))),                  # L shin dir
    'foot_ik.R': (26, 28, Vector((0, 1, 0))),                  # R shin dir
}

# Relaxed finger curl angles (degrees) for natural rest pose
# Applied per-joint: .01 = proximal, .02 = middle, .03 = distal
FINGER_CURL = {
    'f_index.01': 12, 'f_index.02': 15, 'f_index.03': 10,
    'f_middle.01': 14, 'f_middle.02': 17, 'f_middle.03': 12,
    'f_ring.01': 16, 'f_ring.02': 19, 'f_ring.03': 14,
    'f_pinky.01': 18, 'f_pinky.02': 21, 'f_pinky.03': 16,
    'thumb.01': 10, 'thumb.02': 8, 'thumb.03': 5,
}

# Head tracking landmarks: nose (0), left ear (7), right ear (8)
HEAD_LANDMARKS = {'nose': 0, 'left_ear': 7, 'right_ear': 8}

# Spine V2R using virtual midpoints (4-segment spine)
SPINE_CHAINS = {
    'spine_fk':      ('hip_mid', 'spine_low'),
    'spine_fk.001':  ('spine_low', 'spine_mid'),
    'spine_fk.002':  ('spine_mid', 'neck_mid'),
    'spine_fk.003':  ('neck_mid', 'shoulder_mid'),
}

# Rigify IK/FK switch bones
IK_FK_SWITCHES = {
    'upper_arm_parent.L': 'IK_FK',
    'upper_arm_parent.R': 'IK_FK',
    'thigh_parent.L': 'IK_FK',
    'thigh_parent.R': 'IK_FK',
}

# Rest axes for V2R FK rotations — from ACTUAL JaxRigify bone dump (A-pose)
# These are the bone rest directions in ARMATURE SPACE for each FK bone.
# Arms hang DOWN in A-pose (not sideways like T-pose).
BONE_REST_AXES = {
    # Arms hang down and slightly outward (A-pose, from bone dump)
    'upper_arm_fk.L': Vector(( 0.287,  0.070, -0.955)).normalized(),
    'forearm_fk.L':   Vector(( 0.468, -0.156, -0.870)).normalized(),
    'upper_arm_fk.R': Vector((-0.265,  0.071, -0.962)).normalized(),
    'forearm_fk.R':   Vector((-0.453, -0.179, -0.873)).normalized(),
    # Legs point down (nearly straight, from bone dump)
    'thigh_fk.L': Vector(( 0.070, -0.040, -0.997)).normalized(),
    'shin_fk.L':  Vector((-0.010,  0.066, -0.998)).normalized(),
    'thigh_fk.R': Vector((-0.055, -0.041, -0.998)).normalized(),
    'shin_fk.R':  Vector((-0.036,  0.066, -0.997)).normalized(),
    # Spine points up (from bone dump)
    'spine_fk':      Vector(( 0.000, -0.095,  0.995)).normalized(),
    'spine_fk.001':  Vector((-0.000, -0.010,  1.000)).normalized(),
    'spine_fk.002':  Vector((-0.000,  0.060,  0.998)).normalized(),
    'spine_fk.003':  Vector(( 0.000,  0.001,  1.000)).normalized(),
}

# =============================================================================
# LOGGING — writes to both Blender console AND a log file
# =============================================================================

_log_file = None
_log_path = None

def log_init(tag="import"):
    """Open a log file in the logs/ directory next to this addon."""
    global _log_file, _log_path
    log_close()  # Close any previous log

    # Find a writable logs directory
    # Try addon directory first, fall back to temp
    addon_dir = Path(__file__).parent.parent  # MelodicCapStudio/
    logs_dir = addon_dir / "logs"
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        logs_dir = Path(bpy.app.tempdir) / "melodiccap_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_path = logs_dir / f"melodiccap_{tag}_{ts}.log"
    _log_file = open(_log_path, 'w', encoding='utf-8')
    log(f"Log file: {_log_path}")

def log_close():
    """Close the log file."""
    global _log_file, _log_path
    if _log_file:
        _log_file.close()
        _log_file = None

def log(msg, level="INFO"):
    """Log to both Blender console and file."""
    line = f"[{level}] MelodicCap: {msg}"
    print(line)
    if _log_file:
        _log_file.write(line + "\n")
        _log_file.flush()  # Flush immediately so we never lose data

def log_get_path():
    """Return the current log file path."""
    return _log_path

# =============================================================================
# UTILITIES
# =============================================================================

def track_range(ranges, bone_name, value, value_type='loc'):
    """Track min/max range for a bone's animated values."""
    if bone_name not in ranges:
        ranges[bone_name] = {
            'min': [float('inf')] * 3,
            'max': [float('-inf')] * 3,
            'type': value_type,
        }
    r = ranges[bone_name]
    for i in range(3):
        v = value[i]
        if v < r['min'][i]:
            r['min'][i] = v
        if v > r['max'][i]:
            r['max'][i] = v

def get_lm(landmarks, idx):
    """Get landmark as Vector. Handles both string and int keys."""
    for key in [str(idx), idx]:
        if key in landmarks:
            p = landmarks[key]
            return Vector((p[0], p[1], p[2]))
    return None

def get_mid(landmarks, i1, i2):
    """Get midpoint of two landmarks."""
    p1, p2 = get_lm(landmarks, i1), get_lm(landmarks, i2)
    if p1 and p2:
        return (p1 + p2) / 2
    return None

def compute_pole_position(v_root, v_mid, v_end, offset=0.3):
    """Compute IK pole target position from a 3-joint chain.

    Uses the Keemap algorithm: project the mid-joint perpendicular to the
    root→end line and place the pole target along that perpendicular direction.
    Returns the pole position in mocap world space, or None if the limb is
    too straight to determine a bend direction.

    Args:
        v_root: Start of chain (shoulder/hip) as Vector
        v_mid: Mid-joint (elbow/knee) as Vector
        v_end: End of chain (wrist/ankle) as Vector
        offset: Distance to place pole target from mid-joint (meters)
    """
    line = v_end - v_root
    if line.length < 0.001:
        return None

    line_norm = line.normalized()
    # Project mid-joint onto the root→end line
    proj_length = (v_mid - v_root).dot(line_norm)
    proj_point = v_root + line_norm * proj_length

    # Perpendicular from projected point to actual mid-joint = pole direction
    pole_dir = v_mid - proj_point
    if pole_dir.length < 0.005:
        return None  # Limb too straight; pole direction undefined

    return v_mid + pole_dir.normalized() * offset


def compute_virtual_spine(landmarks):
    """Calculate virtual midpoints for 4-segment spine animation.

    Creates 5 virtual landmarks from hips (23,24) and shoulders (11,12):
      hip_mid -> spine_low -> spine_mid -> neck_mid -> shoulder_mid
    """
    hip_l = get_lm(landmarks, 23)
    hip_r = get_lm(landmarks, 24)
    sh_l = get_lm(landmarks, 11)
    sh_r = get_lm(landmarks, 12)

    if not all([hip_l, hip_r, sh_l, sh_r]):
        return None

    hip_mid = (hip_l + hip_r) / 2
    shoulder_mid = (sh_l + sh_r) / 2
    spine_mid = (hip_mid + shoulder_mid) / 2
    spine_low = (hip_mid + spine_mid) / 2
    neck_mid = (spine_mid + shoulder_mid) / 2

    return {
        'hip_mid': hip_mid,
        'spine_low': spine_low,
        'spine_mid': spine_mid,
        'neck_mid': neck_mid,
        'shoulder_mid': shoulder_mid,
    }


def compute_body_orientation(landmarks):
    """Compute hip and shoulder orientation from landmarks.

    Returns a dict with:
      hip_dir: Vector from right_hip to left_hip (lateral axis, +X direction)
      shoulder_dir: Vector from right_shoulder to left_shoulder
      hip_yaw: Angle of hip_dir projected onto XY plane (radians)
      shoulder_yaw: Angle of shoulder_dir projected onto XY plane (radians)
      spine_up: Direction from hip_mid to shoulder_mid (spine lean)
      hip_forward: Forward direction from hip orientation (perpendicular to hip_dir, horizontal)
    Or None if landmarks missing.
    """
    hip_l = get_lm(landmarks, 23)
    hip_r = get_lm(landmarks, 24)
    sh_l = get_lm(landmarks, 11)
    sh_r = get_lm(landmarks, 12)

    if not all([hip_l, hip_r, sh_l, sh_r]):
        return None

    hip_dir = hip_l - hip_r           # right→left, roughly +X in performer space
    shoulder_dir = sh_l - sh_r        # right→left

    hip_mid = (hip_l + hip_r) / 2
    shoulder_mid = (sh_l + sh_r) / 2
    spine_up = (shoulder_mid - hip_mid).normalized()

    # Yaw angles from XY projection (atan2 of the lateral axis)
    hip_yaw = math.atan2(hip_dir.y, hip_dir.x)
    shoulder_yaw = math.atan2(shoulder_dir.y, shoulder_dir.x)

    # Hip forward direction: perpendicular to hip_dir in XY plane, pointing forward (-Y)
    hip_dir_xy = Vector((hip_dir.x, hip_dir.y, 0))
    if hip_dir_xy.length > 0.01:
        hip_dir_xy.normalize()
        # Rotate 90° clockwise in XY: (x,y) → (y,-x) gives the forward direction
        # For a person facing -Y with left at +X: hip_dir = +X, forward = -Y = (0,-1)
        # Rotate (1,0) by -90° → (0,-1) ✓
        hip_forward = Vector((hip_dir_xy.y, -hip_dir_xy.x, 0))
    else:
        hip_forward = Vector((0, -1, 0))

    return {
        'hip_dir': hip_dir,
        'shoulder_dir': shoulder_dir,
        'hip_yaw': hip_yaw,
        'shoulder_yaw': shoulder_yaw,
        'spine_up': spine_up,
        'hip_forward': hip_forward,
        'hip_mid': hip_mid,
        'shoulder_mid': shoulder_mid,
    }


def compute_torso_rotation(cur_orient, ref_orient, world_inv_quat):
    """Compute torso rotation quaternion from hip orientation change.

    Uses the hip lateral axis (right→left) to determine body yaw (horizontal turn)
    and the spine direction to capture forward/backward lean.

    Args:
        cur_orient: Current frame body orientation dict
        ref_orient: Reference frame body orientation dict
        world_inv_quat: Inverse of armature world rotation

    Returns:
        Quaternion in armature local space, or None.
    """
    ref_hip = ref_orient['hip_dir'].copy()
    cur_hip = cur_orient['hip_dir'].copy()

    if ref_hip.length < 0.01 or cur_hip.length < 0.01:
        return None

    ref_hip.normalize()
    cur_hip.normalize()

    # Full 3D rotation from reference hip direction to current hip direction
    # This captures yaw (turning) + any roll/tilt of the hips
    hip_rot = ref_hip.rotation_difference(cur_hip)

    # Also capture spine lean: the direction from hip_mid to shoulder_mid
    ref_up = ref_orient['spine_up'].copy()
    cur_up = cur_orient['spine_up'].copy()

    if ref_up.length > 0.01 and cur_up.length > 0.01:
        ref_up.normalize()
        cur_up.normalize()
        lean_rot = ref_up.rotation_difference(cur_up)

        # Blend: hip rotation captures yaw, lean captures tilt
        # Use slerp to combine — hip yaw is primary, lean is secondary
        # hip_rot handles lateral axis rotation, lean_rot handles forward/back tilt
        combined = hip_rot @ lean_rot
        # Normalize to prevent quaternion drift
        combined.normalize()
    else:
        combined = hip_rot

    # Transform to armature local space
    local_rot = world_inv_quat @ combined @ world_inv_quat.conjugated()
    local_rot.normalize()

    return local_rot


def compute_spine_twist_rotations(cur_orient, ref_orient, world_inv_quat):
    """Compute per-segment spine rotations that include TWIST.

    The old approach computed V2R from collinear midpoints, giving identical
    rotations for all 4 spine segments (just overall lean, no twist).

    New approach:
    1. Compute the overall spine lean direction (hip_mid → shoulder_mid)
    2. Compute the twist angle between shoulder line and hip line
    3. Distribute twist across 4 segments with increasing weight toward shoulders
    4. For each segment, combine its V2R lean rotation with its share of twist

    Args:
        cur_orient: Current frame body orientation dict
        ref_orient: Reference frame body orientation dict
        world_inv_quat: Inverse of armature world rotation

    Returns:
        Dict mapping spine bone name → Quaternion, or None.
    """
    # Twist angle: difference between shoulder yaw and hip yaw
    cur_twist = cur_orient['shoulder_yaw'] - cur_orient['hip_yaw']
    ref_twist = ref_orient['shoulder_yaw'] - ref_orient['hip_yaw']
    delta_twist = cur_twist - ref_twist

    # Normalize to [-pi, pi]
    while delta_twist > math.pi:
        delta_twist -= 2 * math.pi
    while delta_twist < -math.pi:
        delta_twist += 2 * math.pi

    # Spine lean direction (for V2R base rotation)
    cur_up = cur_orient['spine_up'].copy()
    cur_up_local = world_inv_quat @ cur_up

    # Twist weights: lower spine follows hips, upper follows shoulders
    # segment 0 (spine_fk) = near hips → small twist
    # segment 3 (spine_fk.003) = near shoulders → large twist
    twist_weights = [0.1, 0.25, 0.5, 0.8]

    segment_names = ['spine_fk', 'spine_fk.001', 'spine_fk.002', 'spine_fk.003']
    result = {}

    for i, bone_name in enumerate(segment_names):
        rest_axis = BONE_REST_AXES.get(bone_name, Vector((0, 0, 1)))

        # Base rotation: V2R from rest axis to current spine lean direction
        base_quat = rest_axis.rotation_difference(cur_up_local)

        # Twist rotation around the spine's up axis (local Z after lean)
        twist_amount = delta_twist * twist_weights[i]
        twist_axis = cur_up_local.normalized()
        twist_quat = Quaternion(twist_axis, twist_amount)

        # Combine: first apply lean, then twist on top
        combined = twist_quat @ base_quat
        combined.normalize()

        result[bone_name] = combined

    return result


def ensure_quaternion_continuity(q_current, q_prev):
    """Ensure quaternion takes the shortest path from the previous frame.

    Quaternions q and -q represent the same rotation, but interpolating
    between q_prev and -q_current goes the long way around (360° spin).
    If the dot product is negative, negate to keep continuity.
    """
    if q_prev is not None and q_current.dot(q_prev) < 0:
        q_current.negate()
    return q_current


# =============================================================================
# v3.0 FEATURE 1: SMART REFERENCE FRAME SELECTION
# =============================================================================

# Essential landmarks that must be present for a valid reference frame
ESSENTIAL_LANDMARKS = [11, 12, 23, 24, 27, 28]  # shoulders, hips, ankles
# Full set for quality scoring
QUALITY_LANDMARKS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def score_reference_frame(landmarks, prev_landmarks=None, fps=10.0):
    """Score a frame's suitability as a reference (A-pose) frame.

    Criteria (higher is better, max ~100):
    1. Landmark completeness (0-30): How many of the 15 key landmarks are present
    2. Symmetry (0-25): Left/right shoulder and hip symmetry (balanced A-pose)
    3. Uprightness (0-25): How vertical the spine is (hip_mid → shoulder_mid)
    4. Stillness (0-20): Low velocity from previous frame (stable, not moving)

    Returns (score, details_dict) or (0, None) if essential landmarks missing.
    """
    details = {'completeness': 0, 'symmetry': 0, 'uprightness': 0, 'stillness': 0}

    # Check essential landmarks first
    for lm_idx in ESSENTIAL_LANDMARKS:
        if get_lm(landmarks, lm_idx) is None:
            return 0, None

    # 1. COMPLETENESS (0-30)
    present = sum(1 for lm in QUALITY_LANDMARKS if get_lm(landmarks, lm) is not None)
    details['completeness'] = (present / len(QUALITY_LANDMARKS)) * 30

    # 2. SYMMETRY (0-25)
    # Good A-pose: shoulders at same height, hips at same height,
    # arms roughly equidistant from body center
    hip_l = get_lm(landmarks, 23)
    hip_r = get_lm(landmarks, 24)
    sh_l = get_lm(landmarks, 11)
    sh_r = get_lm(landmarks, 12)

    hip_mid = (hip_l + hip_r) / 2

    # Shoulder height symmetry (Z difference)
    sh_z_diff = abs(sh_l.z - sh_r.z)
    sh_symmetry = max(0, 1.0 - sh_z_diff / 0.1)  # Perfect if < 1cm diff

    # Hip height symmetry
    hip_z_diff = abs(hip_l.z - hip_r.z)
    hip_symmetry = max(0, 1.0 - hip_z_diff / 0.05)

    # Shoulder width symmetry (distance from center)
    sh_l_dist = abs(sh_l.x - hip_mid.x)
    sh_r_dist = abs(sh_r.x - hip_mid.x)
    width_ratio = min(sh_l_dist, sh_r_dist) / max(sh_l_dist, sh_r_dist, 0.01)

    # Arm symmetry if wrists available
    wrist_sym = 1.0
    wr_l = get_lm(landmarks, 15)
    wr_r = get_lm(landmarks, 16)
    if wr_l and wr_r:
        wr_l_dist = (wr_l - hip_mid).length
        wr_r_dist = (wr_r - hip_mid).length
        wrist_sym = min(wr_l_dist, wr_r_dist) / max(wr_l_dist, wr_r_dist, 0.01)

    details['symmetry'] = ((sh_symmetry + hip_symmetry + width_ratio + wrist_sym) / 4) * 25

    # 3. UPRIGHTNESS (0-25)
    shoulder_mid = (sh_l + sh_r) / 2
    spine = shoulder_mid - hip_mid
    if spine.length > 0.01:
        # Dot product with pure up vector (0, 0, 1) — should be close to 1
        spine_norm = spine.normalized()
        uprightness = max(0, spine_norm.z)  # 1.0 when perfectly vertical
        details['uprightness'] = uprightness * 25
    else:
        details['uprightness'] = 0

    # 4. STILLNESS (0-20)
    if prev_landmarks is not None:
        total_vel = 0
        vel_count = 0
        for lm_idx in ESSENTIAL_LANDMARKS:
            cur = get_lm(landmarks, lm_idx)
            prev = get_lm(prev_landmarks, lm_idx)
            if cur and prev:
                vel = (cur - prev).length * fps
                total_vel += vel
                vel_count += 1
        if vel_count > 0:
            avg_vel = total_vel / vel_count
            # Stillness score: 1.0 if avg velocity < 0.05 m/s, 0 if > 0.5 m/s
            stillness = max(0, 1.0 - avg_vel / 0.5)
            details['stillness'] = stillness * 20
        else:
            details['stillness'] = 10  # Partial score if can't compute
    else:
        details['stillness'] = 10  # First frame gets partial score

    total = details['completeness'] + details['symmetry'] + details['uprightness'] + details['stillness']
    return total, details


def find_best_reference_frame(frames, scan_range=60, fps=10.0):
    """Scan the first N frames to find the best A-pose reference frame.

    Args:
        frames: List of frame dicts with 'landmarks' key
        scan_range: How many frames to scan (default 60 = ~6 seconds at 10fps)
        fps: Capture framerate for velocity computation

    Returns:
        (best_frame_index, best_score, details) or (0, 0, None) if no good frame found.
    """
    scan_count = min(len(frames), scan_range)

    best_idx = 0
    best_score = 0
    best_details = None

    log(f"  Scanning first {scan_count} frames for best A-pose reference...")

    prev_lms = None
    for fidx in range(scan_count):
        lms = frames[fidx].get('landmarks', {})
        score, details = score_reference_frame(lms, prev_lms, fps)
        prev_lms = lms

        if score > best_score:
            best_score = score
            best_idx = fidx
            best_details = details

    if best_details:
        log(f"  Best reference frame: {best_idx} (score {best_score:.1f}/100)")
        log(f"    Completeness: {best_details['completeness']:.1f}/30")
        log(f"    Symmetry:     {best_details['symmetry']:.1f}/25")
        log(f"    Uprightness:  {best_details['uprightness']:.1f}/25")
        log(f"    Stillness:    {best_details['stillness']:.1f}/20")
        if best_idx != 0:
            log(f"    (Frame 0 was NOT the best — frame {best_idx} is better)")
    else:
        log(f"  WARNING: No frame scored above 0. Using frame 0 as fallback.", "WARN")

    return best_idx, best_score, best_details


# =============================================================================
# v3.0 FEATURE 2: BUTTERWORTH TEMPORAL SMOOTHING
# =============================================================================

def butterworth_lowpass_coefficients(cutoff_hz, sample_rate, order=2):
    """Compute 2nd-order Butterworth low-pass filter coefficients.

    Uses the bilinear transform to convert analog prototype to digital.
    No scipy required — pure math implementation.

    Args:
        cutoff_hz: Cutoff frequency in Hz (e.g., 6.0 for motion, 4.0 for rotation)
        sample_rate: Sample rate in Hz (Blender's scene FPS, e.g., 24)
        order: Filter order (only 2 is implemented)

    Returns:
        (b, a) coefficient arrays for the difference equation:
        y[n] = b[0]*x[n] + b[1]*x[n-1] + b[2]*x[n-2] - a[1]*y[n-1] - a[2]*y[n-2]
    """
    if order != 2:
        raise ValueError("Only 2nd order Butterworth is implemented")

    # Pre-warp the cutoff frequency
    wc = math.tan(math.pi * cutoff_hz / sample_rate)
    wc2 = wc * wc
    sqrt2 = math.sqrt(2.0)

    # Bilinear transform of 2nd-order Butterworth: H(s) = 1 / (s^2 + sqrt(2)*s + 1)
    denom = 1.0 + sqrt2 * wc + wc2

    b0 = wc2 / denom
    b1 = 2.0 * wc2 / denom
    b2 = wc2 / denom
    a1 = 2.0 * (wc2 - 1.0) / denom
    a2 = (1.0 - sqrt2 * wc + wc2) / denom

    return [b0, b1, b2], [1.0, a1, a2]


def apply_butterworth_filter(values, b, a):
    """Apply IIR filter forward and backward (zero-phase, like scipy filtfilt).

    Forward-backward filtering eliminates phase distortion, which is critical
    for motion data (otherwise limbs would lag behind the body).

    Args:
        values: List of float values (one channel of keyframe data)
        b: Numerator coefficients [b0, b1, b2]
        a: Denominator coefficients [1.0, a1, a2]

    Returns:
        List of filtered float values (same length as input).
    """
    n = len(values)
    if n < 6:
        return values  # Too short to filter meaningfully

    def filter_pass(data):
        """Single-direction IIR filter pass."""
        out = [0.0] * len(data)
        # Initialize with first value to avoid transient
        out[0] = data[0]
        out[1] = data[1] if len(data) > 1 else data[0]
        for i in range(2, len(data)):
            out[i] = (b[0] * data[i] + b[1] * data[i-1] + b[2] * data[i-2]
                       - a[1] * out[i-1] - a[2] * out[i-2])
        return out

    # Forward pass
    forward = filter_pass(values)
    # Backward pass (reverse, filter, reverse)
    backward_in = list(reversed(forward))
    backward = filter_pass(backward_in)
    result = list(reversed(backward))

    return result


def smooth_fcurves(armature, loc_cutoff_hz=6.0, rot_cutoff_hz=4.0):
    """Apply Butterworth smoothing to all MelodicCap keyframed FCurves.

    Processes each FCurve independently:
    - Location channels: higher cutoff (6 Hz default) to preserve sharp motion
    - Rotation channels: lower cutoff (4 Hz default) for smoother rotations

    This runs AFTER all keyframes are inserted, as a post-processing step.

    Args:
        armature: The armature object with animation data
        loc_cutoff_hz: Cutoff frequency for location channels
        rot_cutoff_hz: Cutoff frequency for rotation channels
    """
    if not armature.animation_data or not armature.animation_data.action:
        log("  Smoothing: No animation data to smooth", "WARN")
        return

    action = armature.animation_data.action
    scene_fps = bpy.context.scene.render.fps
    smoothed_count = 0
    skipped_count = 0

    log(f"\n  Butterworth smoothing: scene={scene_fps}fps, "
        f"loc_cutoff={loc_cutoff_hz}Hz, rot_cutoff={rot_cutoff_hz}Hz")

    # Pre-compute coefficients
    b_loc, a_loc = butterworth_lowpass_coefficients(loc_cutoff_hz, scene_fps)
    b_rot, a_rot = butterworth_lowpass_coefficients(rot_cutoff_hz, scene_fps)

    for fc in action.fcurves:
        # Only smooth pose bone channels
        if 'pose.bones' not in fc.data_path:
            continue

        kf_points = fc.keyframe_points
        n = len(kf_points)
        if n < 6:
            skipped_count += 1
            continue

        # Determine channel type
        is_rotation = 'rotation' in fc.data_path
        b = b_rot if is_rotation else b_loc
        a = a_rot if is_rotation else a_loc

        # Extract values
        values = [kp.co[1] for kp in kf_points]

        # Apply zero-phase Butterworth filter
        filtered = apply_butterworth_filter(values, b, a)

        # Write back
        for i, kp in enumerate(kf_points):
            kp.co[1] = filtered[i]

        # Update handles
        fc.update()
        smoothed_count += 1

    log(f"  Smoothed {smoothed_count} FCurves, skipped {skipped_count} (too short)")


# =============================================================================
# v3.0 FEATURE 3: LANDMARK GAP INTERPOLATION
# =============================================================================

def interpolate_landmark_gaps(frames, max_gap=10):
    """Fill missing landmarks by interpolating from neighboring frames.

    For each landmark, finds gaps (sequences of frames where it's missing)
    and fills them:
    - Short gaps (≤ max_gap frames): cubic Hermite interpolation using
      the values and velocities at gap boundaries
    - Long gaps (> max_gap frames): hold the last known good position

    This runs BEFORE the outlier filter, so interpolated values also get
    velocity-checked.

    Args:
        frames: List of frame dicts (modified in-place)
        max_gap: Maximum gap length for interpolation (longer gaps use hold)

    Returns:
        Dict of stats: {total_interpolated, total_held, per_landmark_counts}
    """
    if len(frames) < 3:
        return {'total_interpolated': 0, 'total_held': 0}

    # Collect all landmark indices used across frames
    all_lm_indices = set()
    for fdata in frames:
        lms = fdata.get('landmarks', {})
        for key in lms.keys():
            try:
                all_lm_indices.add(int(key))
            except (ValueError, TypeError):
                pass

    stats = {'total_interpolated': 0, 'total_held': 0, 'per_landmark': {}}

    for lm_idx in sorted(all_lm_indices):
        # Build presence mask
        key_str = str(lm_idx)
        positions = []  # (frame_idx, Vector) for frames where landmark exists
        for fidx, fdata in enumerate(frames):
            lms = fdata.get('landmarks', {})
            pos = None
            for key in [key_str, lm_idx]:
                if key in lms:
                    p = lms[key]
                    pos = Vector((p[0], p[1], p[2]))
                    break
            positions.append(pos)

        # Find gaps
        interpolated = 0
        held = 0
        i = 0
        while i < len(positions):
            if positions[i] is not None:
                i += 1
                continue

            # Found start of a gap
            gap_start = i
            while i < len(positions) and positions[i] is None:
                i += 1
            gap_end = i  # One past the last missing frame
            gap_len = gap_end - gap_start

            # Find boundary values
            before_idx = gap_start - 1
            after_idx = gap_end

            has_before = before_idx >= 0 and positions[before_idx] is not None
            has_after = after_idx < len(positions) and positions[after_idx] is not None

            if has_before and has_after and gap_len <= max_gap:
                # Short gap with both boundaries: cubic interpolation
                p0 = positions[before_idx]
                p1 = positions[after_idx]

                # Estimate velocities at boundaries for Hermite interpolation
                # Use adjacent frames if available, otherwise zero velocity
                v0 = Vector((0, 0, 0))
                if before_idx > 0 and positions[before_idx - 1] is not None:
                    v0 = p0 - positions[before_idx - 1]
                v1 = Vector((0, 0, 0))
                if after_idx + 1 < len(positions) and positions[after_idx + 1] is not None:
                    v1 = positions[after_idx + 1] - p1

                for g in range(gap_len):
                    t = (g + 1) / (gap_len + 1)  # 0 < t < 1
                    # Cubic Hermite: h00*p0 + h10*v0 + h01*p1 + h11*v1
                    t2 = t * t
                    t3 = t2 * t
                    h00 = 2*t3 - 3*t2 + 1
                    h10 = t3 - 2*t2 + t
                    h01 = -2*t3 + 3*t2
                    h11 = t3 - t2

                    interp = p0 * h00 + v0 * (h10 * (gap_len + 1)) + p1 * h01 + v1 * (h11 * (gap_len + 1))
                    fill_idx = gap_start + g

                    # Write interpolated value back to frame data
                    lms = frames[fill_idx].get('landmarks', {})
                    actual_key = key_str if key_str in frames[0].get('landmarks', {}) else lm_idx
                    lms[actual_key] = [interp.x, interp.y, interp.z]
                    positions[fill_idx] = interp
                    interpolated += 1

            elif has_before:
                # Hold last known value
                hold_val = positions[before_idx]
                for g in range(gap_len):
                    fill_idx = gap_start + g
                    lms = frames[fill_idx].get('landmarks', {})
                    actual_key = key_str if key_str in frames[0].get('landmarks', {}) else lm_idx
                    lms[actual_key] = [hold_val.x, hold_val.y, hold_val.z]
                    positions[fill_idx] = hold_val.copy()
                    held += 1
            # If no before value, leave the gap (can't interpolate from nothing)

        if interpolated > 0 or held > 0:
            name = LANDMARKS.get(lm_idx, f"lm_{lm_idx}")
            stats['per_landmark'][lm_idx] = {'interpolated': interpolated, 'held': held, 'name': name}
            stats['total_interpolated'] += interpolated
            stats['total_held'] += held

    # Log summary
    if stats['total_interpolated'] > 0 or stats['total_held'] > 0:
        log(f"  Gap interpolation: {stats['total_interpolated']} values interpolated, "
            f"{stats['total_held']} values held across {len(stats['per_landmark'])} landmarks:")
        for lm_idx in sorted(stats['per_landmark'].keys()):
            info = stats['per_landmark'][lm_idx]
            total_frames = len(frames)
            pct = (info['interpolated'] + info['held']) / total_frames * 100
            log(f"    [{lm_idx:2d}] {info['name']:15s}: "
                f"interp={info['interpolated']}, held={info['held']} "
                f"({pct:.1f}% of frames filled)")
    else:
        log(f"  Gap interpolation: no gaps found (all landmarks present)")

    return stats


# =============================================================================
# IMPORTER
# =============================================================================

class MelodicCapImporter:

    def __init__(self, armature, take_data, settings):
        self.armature = armature
        self.take_data = take_data
        self.settings = settings

        # Reference frame data
        self.ref_hip = None
        self.ref_landmarks = None
        self.ref_orientation = None  # Hip/shoulder orientation at frame 0

        # Scaling
        self.scale = 1.0
        self.char_height = 1.87

        # IK target rest positions (world space)
        self.ik_rest_positions = {}

        # Reference pole positions (mocap world space, for delta computation)
        self.ref_pole_positions = {}

        # Foot contact state machine (velocity-based with hysteresis)
        self.foot_state = {}          # bone -> 'MOVING'|'PLANTING'|'PLANTED'
        self.foot_velocity_buf = {}   # bone -> list of recent positions (ring buffer)
        self.foot_locked_pos = {}     # bone -> locked position delta
        self.foot_locked_rot = {}     # bone -> locked rotation quaternion
        self.foot_state_frames = {}   # bone -> frames in current state

        # Quaternion continuity tracking (prevents hemisphere flips / 360° pops)
        self.prev_quats = {}          # bone_name -> previous frame's quaternion

        # Head animation reference
        self.ref_head_dir = None      # Reference head direction for V2R

        # Stats
        self.stats = {'frames': 0, 'keys': 0, 'bones': set()}

        # Per-bone range tracking for diagnostics
        self.ranges = {}  # bone_name -> {'min': Vector, 'max': Vector, 'type': 'loc'|'rot'}

    def analyze(self):
        """Analyze character and mocap data, compute scale factor."""
        log("=" * 60)
        log("MELODICCAP RETARGETER v3.0 - ANALYSIS")
        log("=" * 60)

        # --- Armature scale check ---
        arm_scale = self.armature.scale
        log(f"  Armature scale: ({arm_scale.x:.3f}, {arm_scale.y:.3f}, {arm_scale.z:.3f})")
        if abs(arm_scale.x - 1.0) > 0.01 or abs(arm_scale.y - 1.0) > 0.01 or abs(arm_scale.z - 1.0) > 0.01:
            log("  WARNING: Armature scale is not 1.0! Consider applying scale (Ctrl+A).", "WARN")

        # --- Character height from bone extents ---
        bones = self.armature.data.bones
        world = self.armature.matrix_world

        min_z, max_z = float('inf'), float('-inf')
        for bone in bones:
            h = (world @ bone.head_local).z
            t = (world @ bone.tail_local).z
            min_z = min(min_z, h, t)
            max_z = max(max_z, h, t)

        self.char_height = max_z - min_z if max_z > min_z else 1.87
        log(f"  Character height: {self.char_height:.3f}m")

        # --- IK target rest positions ---
        log(f"\n  IK target rest positions (world space):")
        for ik_bone in IK_TARGETS.keys():
            if ik_bone in bones:
                bone = bones[ik_bone]
                head_world = world @ bone.head_local
                self.ik_rest_positions[ik_bone] = head_world.copy()
                log(f"    {ik_bone}: ({head_world.x:.3f}, {head_world.y:.3f}, {head_world.z:.3f})")

        # --- IK/FK switch status ---
        log(f"\n  IK/FK switch status (BEFORE import configuration):")
        pose_bones = self.armature.pose.bones
        for switch_bone, prop_name in IK_FK_SWITCHES.items():
            if switch_bone in pose_bones:
                pb = pose_bones[switch_bone]
                props = []
                if prop_name in pb:
                    val = pb[prop_name]
                    mode = "FK" if val > 0.5 else "IK"
                    props.append(f"IK_FK={val:.1f}({mode})")
                if 'IK_parent' in pb:
                    props.append(f"IK_parent={pb['IK_parent']}")
                if 'pole_vector' in pb:
                    props.append(f"pole_vector={pb['pole_vector']}")
                if 'pole_parent' in pb:
                    props.append(f"pole_parent={pb['pole_parent']}")
                log(f"    {switch_bone}: {', '.join(props)}")

        # --- Mocap data ---
        frames = self.take_data.get('frames', [])
        if not frames:
            log("ERROR: No frames in take data!", "ERROR")
            return False

        log(f"\n  Mocap data: {len(frames)} frames, {self.take_data.get('duration_seconds', 0):.1f}s")

        # --- Calibration info ---
        calib = self.take_data.get('calibration', {})
        log(f"  Calibration: stereo RMS={calib.get('rms_stereo', 'N/A')}, "
            f"baseline={calib.get('baseline', 'N/A')}m, "
            f"floor_offset={calib.get('floor_offset', 0):.3f}m")

        # --- Smart reference frame selection (v3.0) ---
        duration = self.take_data.get('duration_seconds', len(frames) / 10.0)
        capture_fps = len(frames) / max(duration, 0.1)
        scan_range = self.settings.get('ref_scan_range', 60)

        ref_idx, ref_score, ref_details = find_best_reference_frame(
            frames, scan_range=scan_range, fps=capture_fps)
        self.ref_frame_idx = ref_idx

        self.ref_landmarks = frames[ref_idx].get('landmarks', {})
        self.ref_hip = get_mid(self.ref_landmarks, 23, 24)

        if not self.ref_hip:
            log(f"ERROR: No hip center in reference frame {ref_idx}!", "ERROR")
            return False

        log(f"\n  Reference frame {ref_idx} hip: ({self.ref_hip.x:.3f}, {self.ref_hip.y:.3f}, {self.ref_hip.z:.3f})")

        # --- Reference body orientation ---
        self.ref_orientation = compute_body_orientation(self.ref_landmarks)
        if self.ref_orientation:
            hip_yaw_deg = math.degrees(self.ref_orientation['hip_yaw'])
            sh_yaw_deg = math.degrees(self.ref_orientation['shoulder_yaw'])
            twist_deg = sh_yaw_deg - hip_yaw_deg
            fwd = self.ref_orientation['hip_forward']
            log(f"  Reference hip yaw: {hip_yaw_deg:.1f}°")
            log(f"  Reference shoulder yaw: {sh_yaw_deg:.1f}°")
            log(f"  Reference hip-shoulder twist: {twist_deg:.1f}°")
            log(f"  Reference hip forward: ({fwd.x:.3f}, {fwd.y:.3f}, {fwd.z:.3f})")
            hd = self.ref_orientation['hip_dir']
            log(f"  Reference hip lateral (R→L): ({hd.x:.3f}, {hd.y:.3f}, {hd.z:.3f})")
        else:
            log("  WARNING: Could not compute reference body orientation!", "WARN")

        # --- Person height: nose to ankle midpoint + 0.15m ---
        nose = get_lm(self.ref_landmarks, 0)
        ankle_l = get_lm(self.ref_landmarks, 27)
        ankle_r = get_lm(self.ref_landmarks, 28)

        if nose and ankle_l and ankle_r:
            ankle_mid = (ankle_l + ankle_r) / 2
            mocap_height = (nose.z - ankle_mid.z) + 0.15
            self.scale = self.char_height / mocap_height
            log(f"  Person height: {mocap_height:.3f}m")
            log(f"  Scale factor: {self.scale:.4f}")
        else:
            log("  WARNING: Could not measure person height, using scale=1.0", "WARN")

        # --- Reference landmark positions ---
        log(f"\n  Reference landmarks (relative to hip):")
        for idx in [11, 12, 15, 16, 23, 24, 27, 28]:
            pos = get_lm(self.ref_landmarks, idx)
            if pos:
                rel = pos - self.ref_hip
                name = LANDMARKS.get(idx, str(idx))
                log(f"    [{idx:2d}] {name:15s}: ({rel.x:7.3f}, {rel.y:7.3f}, {rel.z:7.3f})")

        # --- Reference pole target positions ---
        log(f"\n  Reference pole positions:")
        for pole_bone, (lm_root, lm_mid, lm_end) in POLE_TARGETS.items():
            v_root = get_lm(self.ref_landmarks, lm_root)
            v_mid = get_lm(self.ref_landmarks, lm_mid)
            v_end = get_lm(self.ref_landmarks, lm_end)
            if v_root and v_mid and v_end:
                pole_pos = compute_pole_position(v_root, v_mid, v_end)
                if pole_pos:
                    self.ref_pole_positions[pole_bone] = pole_pos
                    log(f"    {pole_bone}: ({pole_pos.x:.3f}, {pole_pos.y:.3f}, {pole_pos.z:.3f})")
                else:
                    log(f"    {pole_bone}: limb too straight, skipping")

        return True

    def prefilter_landmarks(self):
        """Pre-filter landmark data to remove outlier spikes.

        Uses ADAPTIVE per-landmark velocity limits based on human biomechanics:
        - Head/face: max 5 m/s
        - Shoulders: max 8 m/s
        - Elbows: max 12 m/s
        - Wrists: max 15 m/s (punching/gesturing)
        - Hips: max 6 m/s (core, moves slowly)
        - Knees/ankles: max 10 m/s

        Also validates bone-length consistency and flags bad frames.

        Modifies the frame landmark data in-place so all downstream consumers
        (IK targets, pole targets, FK rotations, spine) benefit automatically.
        """
        frames = self.take_data.get('frames', [])
        if len(frames) < 2:
            return

        user_max_velocity = self.settings.get('outlier_velocity', 10.0)

        # Per-landmark velocity limits (m/s) based on biomechanics
        # These are generous — real movement rarely exceeds these
        LANDMARK_MAX_VEL = {
            0: 5.0,   # nose
            1: 5.0, 2: 5.0, 3: 5.0, 4: 5.0, 5: 5.0, 6: 5.0,  # eyes
            7: 5.0, 8: 5.0,  # ears
            9: 5.0, 10: 5.0,  # mouth
            11: 8.0, 12: 8.0,  # shoulders
            13: 12.0, 14: 12.0,  # elbows
            15: 15.0, 16: 15.0,  # wrists
            17: 15.0, 18: 15.0, 19: 15.0, 20: 15.0,  # fingers
            21: 15.0, 22: 15.0,
            23: 4.0, 24: 4.0,  # hips (center of mass, can't move fast)
            25: 10.0, 26: 10.0,  # knees
            27: 10.0, 28: 10.0,  # ankles
            29: 10.0, 30: 10.0, 31: 10.0, 32: 10.0,  # feet
        }

        # Compute frame duration from capture metadata
        duration = self.take_data.get('duration_seconds', len(frames) / 10.0)
        fps = len(frames) / max(duration, 0.1)

        log(f"\n  Outlier filter: adaptive per-landmark velocity limits, "
            f"capture fps={fps:.1f}")
        log(f"    User velocity scale: {user_max_velocity:.0f} m/s")

        # Collect all landmark indices used by any mapping
        used_landmarks = set()
        for lm_idx in IK_TARGETS.values():
            used_landmarks.add(lm_idx)
        for i1, i2 in FK_CHAINS.values():
            used_landmarks.add(i1)
            used_landmarks.add(i2)
        for lm_root, lm_mid, lm_end in POLE_TARGETS.values():
            used_landmarks.add(lm_root)
            used_landmarks.add(lm_mid)
            used_landmarks.add(lm_end)
        for lm_start, lm_end, _ in IK_ROTATION.values():
            used_landmarks.add(lm_start)
            used_landmarks.add(lm_end)
        used_landmarks.update([0, 7, 8, 11, 12, 23, 24])  # spine/height/head landmarks

        # Track state per landmark
        prev_good = {}       # lm_idx -> Vector (last accepted position)
        filter_counts = {}   # lm_idx -> total filtered frames
        consecutive = {}     # lm_idx -> current consecutive hold count
        max_consecutive = {} # lm_idx -> worst consecutive hold run

        # Scale factor: if user sets velocity to 20, it doubles all limits
        vel_scale = user_max_velocity / 10.0

        for fidx, fdata in enumerate(frames):
            lms = fdata.get('landmarks', {})

            for lm_idx in used_landmarks:
                pos = get_lm(lms, lm_idx)
                if pos is None:
                    continue

                # Per-landmark velocity limit
                base_vel = LANDMARK_MAX_VEL.get(lm_idx, 10.0)
                max_jump = (base_vel * vel_scale) / fps

                # Determine the actual key type used in this dict
                key = str(lm_idx) if str(lm_idx) in lms else lm_idx

                if lm_idx in prev_good:
                    displacement = (pos - prev_good[lm_idx]).length
                    if displacement > max_jump:
                        # Outlier — replace with last good position
                        good = prev_good[lm_idx]
                        lms[key] = [good.x, good.y, good.z]
                        filter_counts[lm_idx] = filter_counts.get(lm_idx, 0) + 1
                        consecutive[lm_idx] = consecutive.get(lm_idx, 0) + 1
                        mc = max_consecutive.get(lm_idx, 0)
                        if consecutive[lm_idx] > mc:
                            max_consecutive[lm_idx] = consecutive[lm_idx]
                    else:
                        # Good frame — update baseline
                        prev_good[lm_idx] = pos
                        consecutive[lm_idx] = 0
                else:
                    # First frame — set baseline
                    prev_good[lm_idx] = pos
                    consecutive[lm_idx] = 0

        # Log results
        if filter_counts:
            total = sum(filter_counts.values())
            log(f"  Outlier filter: {total} landmark values replaced "
                f"across {len(filter_counts)} landmarks:")
            for lm_idx in sorted(filter_counts.keys()):
                name = LANDMARKS.get(lm_idx, f"landmark_{lm_idx}")
                count = filter_counts[lm_idx]
                pct = count / len(frames) * 100
                mc = max_consecutive.get(lm_idx, 0)
                base_vel = LANDMARK_MAX_VEL.get(lm_idx, 10.0)
                log(f"    [{lm_idx:2d}] {name:15s}: {count:4d} frames ({pct:5.1f}%), "
                    f"max consecutive hold={mc}, limit={base_vel*vel_scale:.0f}m/s")
                if mc > fps * 2:  # More than 2 seconds of sustained holds
                    log(f"         WARNING: {mc} consecutive holds ({mc/fps:.1f}s) — "
                        f"possible sustained tracking loss", "WARN")
        else:
            log(f"  Outlier filter: no outliers detected")

    def set_ik_fk_mode(self, use_ik=True):
        """Set IK/FK switches on the rig. 0.0=IK, 1.0=FK."""
        target_value = 0.0 if use_ik else 1.0
        pose_bones = self.armature.pose.bones

        for switch_bone, prop_name in IK_FK_SWITCHES.items():
            if switch_bone in pose_bones:
                pb = pose_bones[switch_bone]
                if prop_name in pb:
                    pb[prop_name] = target_value

        log(f"  Set {'IK' if use_ik else 'FK'} mode on all limbs")

    def apply_finger_rest_pose(self, start_frame):
        """Apply a natural relaxed finger curl pose (static, not animated per-frame).

        Sets all finger bones to a slight curl on the start frame.
        Since no finger animation data overwrites these, they persist.
        """
        pose_bones = self.armature.pose.bones
        finger_count = 0

        for bone_base, angle_deg in FINGER_CURL.items():
            for side in ['.L', '.R']:
                bone_name = bone_base + side
                if bone_name not in pose_bones:
                    continue

                pb = pose_bones[bone_name]
                pb.rotation_mode = 'QUATERNION'

                # Curl around local X axis (flex axis for fingers)
                angle_rad = math.radians(angle_deg)
                curl_quat = Quaternion(Vector((1, 0, 0)), angle_rad)

                pb.rotation_quaternion = curl_quat
                pb.keyframe_insert(data_path="rotation_quaternion", frame=start_frame)
                finger_count += 1

        log(f"  Applied relaxed finger pose to {finger_count} bones")

    def apply_animation(self):
        """Apply animation using hybrid approach:
        - Torso: delta from reference hip (root motion) + hip orientation rotation
        - IK targets: delta from reference positions (hands/feet)
        - IK rotation: wrist/foot orientation from forearm/shin direction
        - Pole targets: 3-point projection for elbow/knee bend direction
        - FK rotations: V2R with per-bone rest axes (visible in FK mode only)
        - Spine: V2R lean + twist distribution (shoulder twist relative to hips)
        """
        log("\n" + "=" * 60)
        log("APPLYING ANIMATION")
        log("=" * 60)

        frames = self.take_data.get('frames', [])
        pose_bones = self.armature.pose.bones
        start = self.settings.get('start_frame', 1)
        pin_threshold = self.settings.get('pin_threshold', 0.02)
        animate_fk = self.settings.get('animate_fk', True)
        animate_spine = self.settings.get('animate_spine', True)
        ground_clamp = self.settings.get('ground_clamp', True)
        animate_poles = self.settings.get('animate_poles', True)
        animate_ik_rot = self.settings.get('animate_ik_rot', True)

        # Compute capture FPS for velocity calculations
        duration = self.take_data.get('duration_seconds', len(frames) / 10.0)
        fps = len(frames) / max(duration, 0.1)
        log(f"  Capture FPS: {fps:.1f}")

        # Gap interpolation (v3.0) — fill missing landmarks BEFORE outlier filter
        if self.settings.get('interpolate_gaps', True):
            max_gap = self.settings.get('max_interp_gap', 10)
            interpolate_landmark_gaps(frames, max_gap=max_gap)

        # Pre-filter outlier landmarks (modifies frame data in-place)
        if self.settings.get('filter_outliers', True):
            self.prefilter_landmarks()

        # Force IK mode
        self.set_ik_fk_mode(use_ik=True)

        # Apply relaxed finger pose (static, before animation loop)
        if self.settings.get('finger_curl', True):
            self.apply_finger_rest_pose(start)

        # Configure Rigify properties for correct mocap retargeting
        log("  Configuring Rigify properties:")
        for switch_bone in IK_FK_SWITCHES:
            if switch_bone in pose_bones:
                pb = pose_bones[switch_bone]
                # IK_parent=0 (root space): CRITICAL for delta-from-reference.
                # Default IK_parent=1 makes IK targets follow torso via parent
                # chain, but our delta already includes hip displacement.
                # That causes double root motion (arms fly off during walking).
                # Root space (0) makes IK targets independent of torso.
                if 'IK_parent' in pb:
                    old = pb['IK_parent']
                    pb['IK_parent'] = 0
                    if old != 0:
                        log(f"    {switch_bone}: IK_parent {old}->0 (root space)")
                # pole_vector=True: enables pole target bones so our elbow/knee
                # direction animation actually affects the IK solver.
                # Default False means pole targets are ignored entirely.
                if 'pole_vector' in pb:
                    old = pb['pole_vector']
                    pb['pole_vector'] = True
                    if not old:
                        log(f"    {switch_bone}: pole_vector False->True")
                # pole_parent=0 (root space): same reasoning as IK_parent.
                # Prevents double-motion on pole target positions.
                if 'pole_parent' in pb:
                    old = pb['pole_parent']
                    pb['pole_parent'] = 0
                    if old != 0:
                        log(f"    {switch_bone}: pole_parent {old}->0 (root space)")

        # Set quaternion rotation mode on all FK bones
        for pb in pose_bones:
            pb.rotation_mode = 'QUATERNION'

        # Armature transforms
        world = self.armature.matrix_world
        world_inv = world.inverted()

        # Check available bones
        avail_ik = {}
        for bone, lm_idx in IK_TARGETS.items():
            if bone in pose_bones:
                avail_ik[bone] = lm_idx

        avail_fk = {}
        for bone, (i1, i2) in FK_CHAINS.items():
            if bone in pose_bones:
                avail_fk[bone] = (i1, i2)

        avail_spine = {}
        if animate_spine:
            for bone, (s, e) in SPINE_CHAINS.items():
                if bone in pose_bones:
                    avail_spine[bone] = (s, e)

        avail_poles = {}
        if animate_poles:
            for bone, (lm_root, lm_mid, lm_end) in POLE_TARGETS.items():
                if bone in pose_bones and bone in self.ref_pole_positions:
                    avail_poles[bone] = (lm_root, lm_mid, lm_end)

        avail_ik_rot = {}
        if animate_ik_rot:
            for bone, (lm_start, lm_end, rest_ax) in IK_ROTATION.items():
                if bone in pose_bones:
                    avail_ik_rot[bone] = (lm_start, lm_end, rest_ax)

        has_torso = 'torso' in pose_bones

        log(f"  Available: {len(avail_ik)} IK targets, {len(avail_poles)} pole targets, "
            f"{len(avail_fk)} FK chains, {len(avail_spine)} spine segments, "
            f"{len(avail_ik_rot)} IK rotations, torso={'yes' if has_torso else 'no'}")

        # Foot contact detection settings
        foot_plant_vel = self.settings.get('foot_plant_velocity', 0.3)
        foot_lift_vel = self.settings.get('foot_lift_velocity', 0.5)

        # Initialize foot state for each foot IK bone
        for bone_name in avail_ik:
            if 'foot' in bone_name:
                self.foot_state[bone_name] = 'MOVING'
                self.foot_velocity_buf[bone_name] = []
                self.foot_state_frames[bone_name] = 0

        # How many frames get full detailed logging (first N)
        DETAIL_FRAMES = 5

        # Extra orientation sampling for debugging (log yaw/twist at these intervals)
        ORIENT_LOG_INTERVAL = 25

        # Process each frame
        log(f"  Processing {len(frames)} frames (detailed log for first {DETAIL_FRAMES})...")

        for fidx, fdata in enumerate(frames):
            bf = start + fidx
            bpy.context.scene.frame_set(bf)
            detail = fidx < DETAIL_FRAMES  # Verbose logging for early frames

            lms = fdata.get('landmarks', {})
            hip = get_mid(lms, 23, 24)

            if not hip:
                if detail:
                    log(f"    Frame {fidx}: SKIPPED (no hip landmarks)")
                continue

            if detail:
                log(f"\n    --- Frame {fidx} (Blender frame {bf}) ---")
                log(f"    Hip center: ({hip.x:.4f}, {hip.y:.4f}, {hip.z:.4f})")
                log(f"    Hip delta from ref: ({hip.x - self.ref_hip.x:.4f}, "
                    f"{hip.y - self.ref_hip.y:.4f}, {hip.z - self.ref_hip.z:.4f})")

            # =================================================================
            # ROOT MOTION (TORSO) — location + rotation
            # =================================================================
            cur_orient = compute_body_orientation(lms)

            if has_torso:
                delta = (hip - self.ref_hip) * self.scale
                local_delta = world_inv.to_3x3() @ delta

                torso = pose_bones['torso']
                torso.location = local_delta
                torso.keyframe_insert(data_path="location", frame=bf)

                track_range(self.ranges, 'torso', local_delta, 'loc')
                self.stats['keys'] += 1
                self.stats['bones'].add('torso')

                if detail:
                    log(f"    TORSO loc: ({local_delta.x:.4f}, {local_delta.y:.4f}, {local_delta.z:.4f})")

                # TORSO ROTATION from hip orientation change
                if cur_orient and self.ref_orientation:
                    torso_rot = compute_torso_rotation(
                        cur_orient, self.ref_orientation,
                        world_inv.to_quaternion())

                    if torso_rot:
                        # Quaternion continuity
                        prev_q = self.prev_quats.get('torso_rot')
                        torso_rot = ensure_quaternion_continuity(torso_rot, prev_q)
                        self.prev_quats['torso_rot'] = torso_rot.copy()

                        torso.rotation_mode = 'QUATERNION'
                        torso.rotation_quaternion = torso_rot
                        torso.keyframe_insert(data_path="rotation_quaternion", frame=bf)

                        track_range(self.ranges, 'torso_rot',
                                    [torso_rot.w, torso_rot.x, torso_rot.y], 'rot')
                        self.stats['keys'] += 1

                        if detail:
                            hip_yaw = math.degrees(cur_orient['hip_yaw'])
                            ref_yaw = math.degrees(self.ref_orientation['hip_yaw'])
                            delta_yaw = hip_yaw - ref_yaw
                            log(f"    TORSO rot: w={torso_rot.w:.3f} ({torso_rot.x:.3f}, "
                                f"{torso_rot.y:.3f}, {torso_rot.z:.3f})  "
                                f"hip_yaw={hip_yaw:.1f}° (Δ{delta_yaw:+.1f}°)")

            # =================================================================
            # IK TARGETS (HANDS AND FEET)
            # =================================================================
            for ik_bone, lm_idx in avail_ik.items():
                pos = get_lm(lms, lm_idx)
                if not pos:
                    continue

                ref_pos = get_lm(self.ref_landmarks, lm_idx)
                if not ref_pos:
                    continue

                mocap_delta = (pos - ref_pos) * self.scale

                # Ground clamp
                clamped = False
                if ground_clamp and 'foot' in ik_bone:
                    rest_pos = self.ik_rest_positions.get(ik_bone)
                    if rest_pos:
                        world_z = rest_pos.z + mocap_delta.z
                        if world_z < 0:
                            mocap_delta.z = -rest_pos.z
                            clamped = True

                local_delta = world_inv.to_3x3() @ mocap_delta

                # Velocity-based foot contact detection with state machine
                pinned = False
                if 'foot' in ik_bone and ik_bone in self.foot_state:
                    # Track position history for velocity computation
                    buf = self.foot_velocity_buf[ik_bone]
                    buf.append(local_delta.copy())
                    if len(buf) > 3:
                        buf.pop(0)

                    # Compute velocity over the buffer window
                    if len(buf) >= 2:
                        # Use frame-to-frame displacement scaled by fps
                        frame_vel = (buf[-1] - buf[-2]).length * fps
                    else:
                        frame_vel = float('inf')

                    state = self.foot_state[ik_bone]
                    self.foot_state_frames[ik_bone] += 1

                    if state == 'MOVING':
                        if frame_vel < foot_plant_vel:
                            self.foot_state[ik_bone] = 'PLANTING'
                            self.foot_state_frames[ik_bone] = 1
                    elif state == 'PLANTING':
                        if frame_vel < foot_plant_vel:
                            if self.foot_state_frames[ik_bone] >= 2:
                                # Confirmed plant — lock position (rotation locked in IK rot section)
                                self.foot_state[ik_bone] = 'PLANTED'
                                self.foot_locked_pos[ik_bone] = local_delta.copy()
                                # Pre-lock current rotation if available from previous frame
                                prev_rot = self.prev_quats.get(ik_bone + '_rot')
                                if prev_rot:
                                    self.foot_locked_rot[ik_bone] = prev_rot.copy()
                                self.foot_state_frames[ik_bone] = 0
                        else:
                            # Velocity went back up, cancel planting
                            self.foot_state[ik_bone] = 'MOVING'
                            self.foot_state_frames[ik_bone] = 0
                    elif state == 'PLANTED':
                        if frame_vel > foot_lift_vel:
                            if self.foot_state_frames[ik_bone] >= 2:
                                # Confirmed lift — velocity sustained above threshold
                                self.foot_state[ik_bone] = 'MOVING'
                                self.foot_state_frames[ik_bone] = 0
                            # else: keep counting high-velocity frames
                        else:
                            # Low velocity — reset the lift counter (foot still planted)
                            # Don't reset to 0 here; only reset when we need to
                            # track consecutive high-velocity frames for lift confirmation
                            if self.foot_state_frames[ik_bone] > 0:
                                self.foot_state_frames[ik_bone] = 0

                    # Apply locked position when planted
                    if self.foot_state[ik_bone] == 'PLANTED' and ik_bone in self.foot_locked_pos:
                        local_delta = self.foot_locked_pos[ik_bone].copy()
                        pinned = True

                pb = pose_bones[ik_bone]
                pb.location = local_delta
                pb.keyframe_insert(data_path="location", frame=bf)

                track_range(self.ranges, ik_bone, local_delta, 'loc')
                self.stats['keys'] += 1
                self.stats['bones'].add(ik_bone)

                if detail:
                    flags = ""
                    if clamped:
                        flags += " [CLAMPED]"
                    if pinned:
                        flags += " [PINNED]"
                    lm_name = LANDMARKS.get(lm_idx, str(lm_idx))
                    log(f"    {ik_bone} loc: ({local_delta.x:.4f}, {local_delta.y:.4f}, {local_delta.z:.4f})"
                        f"  lm{lm_idx}({lm_name}){flags}")

            # =================================================================
            # IK TARGET ROTATION (with fallback and quaternion continuity)
            # =================================================================
            for ik_bone, (lm_start, lm_end, rest_ax) in avail_ik_rot.items():
                # Skip rotation for planted feet — use locked rotation
                if self.foot_state.get(ik_bone) == 'PLANTED' and ik_bone in self.foot_locked_rot:
                    quat = self.foot_locked_rot[ik_bone].copy()
                    pb = pose_bones[ik_bone]
                    pb.rotation_mode = 'QUATERNION'
                    pb.rotation_quaternion = quat
                    pb.keyframe_insert(data_path="rotation_quaternion", frame=bf)
                    track_range(self.ranges, ik_bone + '_rot', [quat.w, quat.x, quat.y], 'rot')
                    self.stats['keys'] += 1
                    if detail:
                        log(f"    {ik_bone} rot: LOCKED (planted)")
                    continue

                p1 = get_lm(lms, lm_start)
                p2 = get_lm(lms, lm_end)
                used_fallback = False

                # Fallback to forearm/shin direction if finger/toe landmarks missing
                if not p1 or not p2:
                    fb = IK_ROTATION_FALLBACK.get(ik_bone)
                    if fb:
                        p1 = get_lm(lms, fb[0])
                        p2 = get_lm(lms, fb[1])
                        rest_ax = fb[2]
                        used_fallback = True

                if not p1 or not p2:
                    continue

                limb_dir = (p2 - p1).normalized()
                dir_local = world_inv.to_quaternion() @ limb_dir
                quat = rest_ax.rotation_difference(dir_local)

                # Quaternion continuity: prevent hemisphere flips
                prev_q = self.prev_quats.get(ik_bone + '_rot')
                quat = ensure_quaternion_continuity(quat, prev_q)
                self.prev_quats[ik_bone + '_rot'] = quat.copy()

                # Store rotation for foot locking
                if 'foot' in ik_bone and self.foot_state.get(ik_bone) == 'PLANTED':
                    self.foot_locked_rot[ik_bone] = quat.copy()

                pb = pose_bones[ik_bone]
                pb.rotation_mode = 'QUATERNION'
                pb.rotation_quaternion = quat
                pb.keyframe_insert(data_path="rotation_quaternion", frame=bf)

                track_range(self.ranges, ik_bone + '_rot', [quat.w, quat.x, quat.y], 'rot')
                self.stats['keys'] += 1

                if detail:
                    fb_tag = " [FALLBACK]" if used_fallback else ""
                    log(f"    {ik_bone} rot: w={quat.w:.3f} ({quat.x:.3f}, {quat.y:.3f}, {quat.z:.3f})"
                        f"  dir_local=({dir_local.x:.3f}, {dir_local.y:.3f}, {dir_local.z:.3f}){fb_tag}")

            # =================================================================
            # POLE TARGETS
            # =================================================================
            for pole_bone, (lm_root, lm_mid, lm_end) in avail_poles.items():
                v_root = get_lm(lms, lm_root)
                v_mid = get_lm(lms, lm_mid)
                v_end = get_lm(lms, lm_end)
                if not all([v_root, v_mid, v_end]):
                    continue

                pole_pos = compute_pole_position(v_root, v_mid, v_end)
                if not pole_pos:
                    if detail:
                        log(f"    {pole_bone}: limb too straight, skipped")
                    continue

                ref_pole = self.ref_pole_positions.get(pole_bone)
                if not ref_pole:
                    continue

                pole_delta = (pole_pos - ref_pole) * self.scale
                local_delta = world_inv.to_3x3() @ pole_delta

                pb = pose_bones[pole_bone]
                pb.location = local_delta
                pb.keyframe_insert(data_path="location", frame=bf)

                track_range(self.ranges, pole_bone, local_delta, 'loc')
                self.stats['keys'] += 1
                self.stats['bones'].add(pole_bone)

                if detail:
                    log(f"    {pole_bone} loc: ({local_delta.x:.4f}, {local_delta.y:.4f}, {local_delta.z:.4f})")

            # =================================================================
            # FK ROTATIONS (V2R METHOD)
            # =================================================================
            if animate_fk:
                for fk_bone, (i1, i2) in avail_fk.items():
                    p1, p2 = get_lm(lms, i1), get_lm(lms, i2)
                    if not p1 or not p2:
                        continue

                    target_dir = (p2 - p1).normalized()
                    target_dir_local = world_inv.to_quaternion() @ target_dir

                    rest_axis = BONE_REST_AXES.get(fk_bone)
                    if not rest_axis:
                        continue

                    quat = rest_axis.rotation_difference(target_dir_local)

                    # Quaternion continuity
                    prev_q = self.prev_quats.get(fk_bone)
                    quat = ensure_quaternion_continuity(quat, prev_q)
                    self.prev_quats[fk_bone] = quat.copy()

                    pb = pose_bones[fk_bone]
                    pb.rotation_quaternion = quat
                    pb.keyframe_insert(data_path="rotation_quaternion", frame=bf)

                    track_range(self.ranges, fk_bone, [quat.w, quat.x, quat.y], 'rot')
                    self.stats['keys'] += 1
                    self.stats['bones'].add(fk_bone)

                    if detail:
                        log(f"    {fk_bone} rot: w={quat.w:.3f} ({quat.x:.3f}, {quat.y:.3f}, {quat.z:.3f})"
                            f"  dir=({target_dir_local.x:.3f}, {target_dir_local.y:.3f}, {target_dir_local.z:.3f})")

            # =================================================================
            # SPINE ANIMATION (V2R with lean + twist distribution)
            # =================================================================
            if animate_spine and avail_spine:
                if cur_orient and self.ref_orientation:
                    # New approach: lean + twist per segment
                    spine_rots = compute_spine_twist_rotations(
                        cur_orient, self.ref_orientation,
                        world_inv.to_quaternion())

                    if spine_rots:
                        for bone_name in avail_spine:
                            quat = spine_rots.get(bone_name)
                            if not quat:
                                continue

                            # Quaternion continuity
                            prev_q = self.prev_quats.get(bone_name)
                            quat = ensure_quaternion_continuity(quat, prev_q)
                            self.prev_quats[bone_name] = quat.copy()

                            pb = pose_bones[bone_name]
                            pb.rotation_quaternion = quat
                            pb.keyframe_insert(data_path="rotation_quaternion", frame=bf)

                            track_range(self.ranges, bone_name, [quat.w, quat.x, quat.y], 'rot')
                            self.stats['keys'] += 1
                            self.stats['bones'].add(bone_name)

                            if detail:
                                twist_deg = math.degrees(
                                    (cur_orient['shoulder_yaw'] - cur_orient['hip_yaw']) -
                                    (self.ref_orientation['shoulder_yaw'] - self.ref_orientation['hip_yaw']))
                                log(f"    {bone_name} rot: w={quat.w:.3f} ({quat.x:.3f}, {quat.y:.3f}, {quat.z:.3f})"
                                    f"  twist={twist_deg:+.1f}°")
                else:
                    # Fallback: old collinear midpoint approach if orientation unavailable
                    spine_pts = compute_virtual_spine(lms)
                    if spine_pts:
                        for bone_name, (start_key, end_key) in avail_spine.items():
                            s_pos = spine_pts.get(start_key)
                            e_pos = spine_pts.get(end_key)
                            if not s_pos or not e_pos:
                                continue

                            target_dir = (e_pos - s_pos).normalized()
                            target_dir_local = world_inv.to_quaternion() @ target_dir

                            rest_axis = BONE_REST_AXES.get(bone_name, Vector((0, 0, 1)))
                            quat = rest_axis.rotation_difference(target_dir_local)

                            pb = pose_bones[bone_name]
                            pb.rotation_quaternion = quat
                            pb.keyframe_insert(data_path="rotation_quaternion", frame=bf)

                            track_range(self.ranges, bone_name, [quat.w, quat.x, quat.y], 'rot')
                            self.stats['keys'] += 1
                            self.stats['bones'].add(bone_name)

                            if detail:
                                log(f"    {bone_name} rot: w={quat.w:.3f} ({quat.x:.3f}, {quat.y:.3f}, {quat.z:.3f})"
                                    f"  dir=({target_dir_local.x:.3f}, {target_dir_local.y:.3f}, {target_dir_local.z:.3f})"
                                    f"  [FALLBACK-collinear]")

            # =================================================================
            # HEAD ANIMATION (nose + ear midpoint → head facing direction)
            # =================================================================
            if 'head' in pose_bones:
                nose = get_lm(lms, 0)
                ear_l = get_lm(lms, 7)
                ear_r = get_lm(lms, 8)
                if nose and ear_l and ear_r:
                    ear_mid = (ear_l + ear_r) / 2
                    head_dir = (nose - ear_mid)
                    if head_dir.length > 0.01:
                        head_dir.normalize()
                        head_dir_local = world_inv.to_quaternion() @ head_dir

                        # Compute reference head direction on first frame
                        if self.ref_head_dir is None:
                            self.ref_head_dir = head_dir_local.copy()

                        # V2R: rotation from reference to current head direction
                        quat = self.ref_head_dir.rotation_difference(head_dir_local)

                        # Quaternion continuity
                        prev_q = self.prev_quats.get('head')
                        quat = ensure_quaternion_continuity(quat, prev_q)
                        self.prev_quats['head'] = quat.copy()

                        pb = pose_bones['head']
                        pb.rotation_mode = 'QUATERNION'
                        pb.rotation_quaternion = quat
                        pb.keyframe_insert(data_path="rotation_quaternion", frame=bf)

                        track_range(self.ranges, 'head', [quat.w, quat.x, quat.y], 'rot')
                        self.stats['keys'] += 1
                        self.stats['bones'].add('head')

                        if detail:
                            log(f"    head rot: w={quat.w:.3f} ({quat.x:.3f}, {quat.y:.3f}, {quat.z:.3f})")

            self.stats['frames'] += 1

            # Periodic orientation logging for debugging
            if fidx >= DETAIL_FRAMES and fidx % ORIENT_LOG_INTERVAL == 0:
                parts = [f"    Frame {fidx}/{len(frames)}"]
                if cur_orient and self.ref_orientation:
                    hip_yaw = math.degrees(cur_orient['hip_yaw'])
                    ref_yaw = math.degrees(self.ref_orientation['hip_yaw'])
                    sh_yaw = math.degrees(cur_orient['shoulder_yaw'])
                    ref_sh_yaw = math.degrees(self.ref_orientation['shoulder_yaw'])
                    twist = (cur_orient['shoulder_yaw'] - cur_orient['hip_yaw']) - \
                            (self.ref_orientation['shoulder_yaw'] - self.ref_orientation['hip_yaw'])
                    parts.append(f"hip_yaw={hip_yaw:.1f}°(Δ{hip_yaw-ref_yaw:+.1f}°)")
                    parts.append(f"sh_yaw={sh_yaw:.1f}°(Δ{sh_yaw-ref_sh_yaw:+.1f}°)")
                    parts.append(f"twist={math.degrees(twist):+.1f}°")
                log("  ".join(parts))

        # === POST-PROCESSING: Butterworth smoothing (v3.0) ===
        if self.settings.get('smooth_curves', True):
            loc_cutoff = self.settings.get('smooth_loc_cutoff', 6.0)
            rot_cutoff = self.settings.get('smooth_rot_cutoff', 4.0)
            smooth_fcurves(self.armature, loc_cutoff_hz=loc_cutoff, rot_cutoff_hz=rot_cutoff)

        return True

    def summary(self):
        """Print import summary with range diagnostics."""
        log("\n" + "=" * 60)
        log("IMPORT SUMMARY")
        log("=" * 60)
        log(f"  Frames processed: {self.stats['frames']}")
        log(f"  Keyframes created: {self.stats['keys']}")
        log(f"  Bones animated ({len(self.stats['bones'])}):")

        ik_bones = sorted(b for b in self.stats['bones'] if '_ik' in b)
        fk_bones = sorted(b for b in self.stats['bones'] if '_fk' in b)
        spine_bones = sorted(b for b in self.stats['bones'] if 'spine' in b)
        other_bones = sorted(b for b in self.stats['bones'] if '_ik' not in b and '_fk' not in b and 'spine' not in b)

        if other_bones:
            log(f"    Root: {other_bones}")
        if ik_bones:
            log(f"    IK: {ik_bones}")
        if fk_bones:
            log(f"    FK: {fk_bones}")
        if spine_bones:
            log(f"    Spine: {spine_bones}")

        # === RANGE DIAGNOSTICS ===
        log("\n" + "=" * 60)
        log("RANGE DIAGNOSTICS (min → max per axis)")
        log("=" * 60)
        log("  Location ranges (meters from rest position):")
        for bone_name in sorted(self.ranges.keys()):
            r = self.ranges[bone_name]
            if r['type'] != 'loc':
                continue
            mn, mx = r['min'], r['max']
            log(f"    {bone_name:35s}  X:[{mn[0]:+.3f} → {mx[0]:+.3f}]  "
                f"Y:[{mn[1]:+.3f} → {mx[1]:+.3f}]  Z:[{mn[2]:+.3f} → {mx[2]:+.3f}]")
            # Flag suspicious ranges
            for i, axis in enumerate(['X', 'Y', 'Z']):
                span = mx[i] - mn[i]
                if span > 2.0:
                    log(f"      WARNING: {axis} span = {span:.3f}m (>2m — possible outlier?)", "WARN")

        log("\n  Rotation ranges (quaternion w,x,y components):")
        for bone_name in sorted(self.ranges.keys()):
            r = self.ranges[bone_name]
            if r['type'] != 'rot':
                continue
            mn, mx = r['min'], r['max']
            log(f"    {bone_name:35s}  w:[{mn[0]:+.3f}→{mx[0]:+.3f}]  "
                f"x:[{mn[1]:+.3f}→{mx[1]:+.3f}]  y:[{mn[2]:+.3f}→{mx[2]:+.3f}]")


# =============================================================================
# OPERATORS
# =============================================================================

class MELODICCAP_OT_import(bpy.types.Operator, ImportHelper):
    """Import MelodicCap take JSON and apply to selected armature"""
    bl_idname = "melodiccap.import_take"
    bl_label = "Import Take"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})

    def execute(self, context):
        arm = context.active_object
        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature first!")
            return {'CANCELLED'}

        # Initialize file logging
        log_init("import")

        log(f"\n  File: {self.filepath}")
        log(f"  Armature: {arm.name}")
        log(f"  Blender version: {bpy.app.version_string}")

        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load JSON: {e}")
            log(f"  FAILED: {e}", "ERROR")
            log_close()
            return {'CANCELLED'}

        # === TAKE DATA VALIDATION ===
        log("\n  === TAKE DATA VALIDATION ===")
        frames = data.get('frames', [])
        version = data.get('version', 'unknown')
        capture_mode = data.get('capture_mode', 'unknown')
        duration = data.get('duration_seconds', 0)
        frame_count = data.get('frame_count', len(frames))

        log(f"  Version: {version}")
        log(f"  Capture mode: {capture_mode}")
        log(f"  Frame count (metadata): {frame_count}, actual: {len(frames)}")
        log(f"  Duration: {duration:.1f}s")

        if len(frames) == 0:
            self.report({'ERROR'}, "Take file has NO frames!")
            log("  FAILED: No frames in take data", "ERROR")
            log_close()
            return {'CANCELLED'}

        if len(frames) < 10:
            log(f"  WARNING: Very short take ({len(frames)} frames). Results may be poor.", "WARN")

        # Check frame 0 has essential landmarks
        f0_lms = frames[0].get('landmarks', {})
        essential = [11, 12, 23, 24, 27, 28]  # shoulders, hips, ankles
        missing = [str(lm) for lm in essential if str(lm) not in f0_lms and lm not in f0_lms]
        if missing:
            log(f"  WARNING: Frame 0 missing essential landmarks: {missing}", "WARN")
            log(f"  Available landmarks in frame 0: {sorted(f0_lms.keys())}", "WARN")
            if len(missing) > 3:
                self.report({'ERROR'}, f"Frame 0 missing critical landmarks: {missing}. Bad take data.")
                log_close()
                return {'CANCELLED'}

        # Check coordinate space sanity
        hip_l = f0_lms.get('23') or f0_lms.get(23)
        hip_r = f0_lms.get('24') or f0_lms.get(24)
        if hip_l and hip_r:
            hip_z = (hip_l[2] + hip_r[2]) / 2
            log(f"  Hip Z at frame 0: {hip_z:.3f}m (should be ~0.8-1.2 for standing person)")
            if hip_z < 0.2:
                log(f"  WARNING: Hip Z very low ({hip_z:.3f}m). Floor calibration may be wrong.", "WARN")
            if hip_z > 3.0:
                log(f"  WARNING: Hip Z very high ({hip_z:.3f}m). Calibration may be wrong.", "WARN")
            if hip_z < -0.5:
                log(f"  ERROR: Hip Z negative ({hip_z:.3f}m). Floor offset is wrong or coords not in Blender space.", "ERROR")

        # Calibration info
        calib = data.get('calibration', {})
        log(f"  Calibration: RMS={calib.get('rms_stereo', 'N/A')}, "
            f"baseline={calib.get('baseline', 'N/A')}, "
            f"floor_offset={calib.get('floor_offset', 'N/A')}")

        # Check for outlier frames
        total_outliers = data.get('total_outliers_filtered', 0)
        predicted = data.get('predicted_frames', 0)
        dropped = data.get('dropped_frames', 0)
        log(f"  Outliers filtered: {total_outliers}, Predicted frames: {predicted}, Dropped: {dropped}")

        # Landmark coverage summary
        vis = data.get('landmark_visibility', {})
        if vis:
            log(f"  Landmark coverage (key landmarks):")
            for lm_idx in ['11', '12', '15', '16', '23', '24', '27', '28']:
                lm_data = vis.get(lm_idx, {})
                coverage = lm_data.get('coverage_pct', 0)
                name = lm_data.get('name', f'lm_{lm_idx}')
                avg_vis = lm_data.get('avg_visibility', 0)
                log(f"    [{lm_idx:>2s}] {name:20s}: coverage={coverage:.1f}%  vis={avg_vis:.3f}")
                if coverage < 50:
                    log(f"         WARNING: Low coverage! This landmark may cause issues.", "WARN")

        log("  === END VALIDATION ===\n")

        settings = {
            'start_frame': context.scene.melodiccap_start_frame,
            'animate_fk': context.scene.melodiccap_animate_fk,
            'animate_spine': context.scene.melodiccap_animate_spine,
            'animate_poles': context.scene.melodiccap_animate_poles,
            'animate_ik_rot': context.scene.melodiccap_animate_ik_rot,
            'ground_clamp': context.scene.melodiccap_ground_clamp,
            'pin_threshold': context.scene.melodiccap_pin_threshold,
            'filter_outliers': context.scene.melodiccap_filter_outliers,
            'outlier_velocity': context.scene.melodiccap_outlier_velocity,
            'foot_plant_velocity': context.scene.melodiccap_foot_plant_velocity,
            'foot_lift_velocity': context.scene.melodiccap_foot_lift_velocity,
            'finger_curl': context.scene.melodiccap_finger_curl,
            # v3.0 settings
            'interpolate_gaps': context.scene.melodiccap_interpolate_gaps,
            'max_interp_gap': context.scene.melodiccap_max_interp_gap,
            'smooth_curves': context.scene.melodiccap_smooth_curves,
            'smooth_loc_cutoff': context.scene.melodiccap_smooth_loc_cutoff,
            'smooth_rot_cutoff': context.scene.melodiccap_smooth_rot_cutoff,
            'ref_scan_range': context.scene.melodiccap_ref_scan_range,
        }

        log(f"\n  Settings:")
        for k, v in settings.items():
            log(f"    {k}: {v}")

        imp = MelodicCapImporter(arm, data, settings)

        if not imp.analyze():
            self.report({'ERROR'}, "Analysis failed - check console")
            log_close()
            return {'CANCELLED'}

        bpy.ops.object.mode_set(mode='POSE')
        imp.apply_animation()
        imp.summary()

        log_path = log_get_path()
        log_close()

        msg = f"Imported {imp.stats['frames']} frames, {len(imp.stats['bones'])} bones"
        if log_path:
            msg += f" | Log: {log_path}"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class MELODICCAP_OT_clear(bpy.types.Operator):
    """Clear all animation data and reset pose"""
    bl_idname = "melodiccap.clear"
    bl_label = "Clear Animation"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm = context.active_object
        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature!")
            return {'CANCELLED'}

        if arm.animation_data:
            arm.animation_data_clear()

        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.pose.transforms_clear()

        log("Cleared animation and reset pose")
        self.report({'INFO'}, "Animation cleared")
        return {'FINISHED'}


class MELODICCAP_OT_set_ik_mode(bpy.types.Operator):
    """Set rig to IK mode (recommended for mocap)"""
    bl_idname = "melodiccap.set_ik_mode"
    bl_label = "Set IK Mode"

    def execute(self, context):
        arm = context.active_object
        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature!")
            return {'CANCELLED'}

        pose_bones = arm.pose.bones
        for switch_bone, prop_name in IK_FK_SWITCHES.items():
            if switch_bone in pose_bones:
                pb = pose_bones[switch_bone]
                if prop_name in pb:
                    pb[prop_name] = 0.0

        self.report({'INFO'}, "Set to IK mode (all limbs)")
        return {'FINISHED'}


class MELODICCAP_OT_set_fk_mode(bpy.types.Operator):
    """Set rig to FK mode"""
    bl_idname = "melodiccap.set_fk_mode"
    bl_label = "Set FK Mode"

    def execute(self, context):
        arm = context.active_object
        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature!")
            return {'CANCELLED'}

        pose_bones = arm.pose.bones
        for switch_bone, prop_name in IK_FK_SWITCHES.items():
            if switch_bone in pose_bones:
                pb = pose_bones[switch_bone]
                if prop_name in pb:
                    pb[prop_name] = 1.0

        self.report({'INFO'}, "Set to FK mode (all limbs)")
        return {'FINISHED'}


class MELODICCAP_OT_diagnostic(bpy.types.Operator):
    """Dump complete rig diagnostic to log file"""
    bl_idname = "melodiccap.diagnostic"
    bl_label = "Diagnostic Dump"

    def execute(self, context):
        arm = context.active_object
        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature!")
            return {'CANCELLED'}

        log_init("diagnostic")
        log("=" * 60)
        log("RIG DIAGNOSTIC DUMP")
        log("=" * 60)

        # Armature info
        log(f"  Armature: {arm.name}")
        log(f"  Location: ({arm.location.x:.4f}, {arm.location.y:.4f}, {arm.location.z:.4f})")
        log(f"  Scale: ({arm.scale.x:.4f}, {arm.scale.y:.4f}, {arm.scale.z:.4f})")
        log(f"  Rotation: ({arm.rotation_euler.x:.4f}, {arm.rotation_euler.y:.4f}, {arm.rotation_euler.z:.4f})")

        world = arm.matrix_world
        log(f"\n  World matrix:")
        for r in range(4):
            log(f"    [{world[r][0]:+.4f}  {world[r][1]:+.4f}  {world[r][2]:+.4f}  {world[r][3]:+.4f}]")

        # IK/FK switch state
        log(f"\n  IK/FK Switch Properties:")
        pose_bones = arm.pose.bones
        for switch_bone, prop_name in IK_FK_SWITCHES.items():
            if switch_bone in pose_bones:
                pb = pose_bones[switch_bone]
                props = {k: pb[k] for k in pb.keys() if not k.startswith('_')}
                log(f"    {switch_bone}: {props}")

        # All bones we care about — current pose state
        log(f"\n  Bone Pose State (current frame {context.scene.frame_current}):")
        all_bones = (
            list(IK_TARGETS.keys()) +
            list(POLE_TARGETS.keys()) +
            list(FK_CHAINS.keys()) +
            list(SPINE_CHAINS.keys()) +
            ['torso']
        )
        for bone_name in sorted(set(all_bones)):
            if bone_name not in pose_bones:
                log(f"    {bone_name:35s}  MISSING")
                continue
            pb = pose_bones[bone_name]
            loc = pb.location
            rot = pb.rotation_quaternion
            parent = pb.parent.name if pb.parent else "None"
            log(f"    {bone_name:35s}  loc=({loc.x:+.4f}, {loc.y:+.4f}, {loc.z:+.4f})  "
                f"rot=w{rot.w:+.3f}({rot.x:+.3f},{rot.y:+.3f},{rot.z:+.3f})  parent={parent}")

        # Bone rest positions
        log(f"\n  Bone Rest Positions (world space):")
        bones = arm.data.bones
        for bone_name in sorted(set(all_bones)):
            if bone_name not in bones:
                continue
            bone = bones[bone_name]
            head_world = world @ bone.head_local
            tail_world = world @ bone.tail_local
            direction = (tail_world - head_world).normalized()
            length = (tail_world - head_world).length
            log(f"    {bone_name:35s}  head=({head_world.x:+.4f},{head_world.y:+.4f},{head_world.z:+.4f})  "
                f"dir=({direction.x:+.3f},{direction.y:+.3f},{direction.z:+.3f})  len={length:.4f}")

        # Animation data
        log(f"\n  Animation Data:")
        if arm.animation_data and arm.animation_data.action:
            action = arm.animation_data.action
            log(f"    Action: {action.name}")
            log(f"    Frame range: {action.frame_range[0]:.0f} - {action.frame_range[1]:.0f}")
            log(f"    FCurves: {len(action.fcurves)}")
            # List unique bone data paths
            bone_paths = set()
            for fc in action.fcurves:
                parts = fc.data_path.split('"')
                if len(parts) >= 2:
                    bone_paths.add(parts[1])
            log(f"    Animated bones ({len(bone_paths)}): {sorted(bone_paths)}")
        else:
            log(f"    No animation data")

        path = log_get_path()
        log_close()

        self.report({'INFO'}, f"Diagnostic saved: {path}")
        return {'FINISHED'}


# =============================================================================
# PANEL
# =============================================================================

class MELODICCAP_PT_panel(bpy.types.Panel):
    bl_label = "MelodicCap Retargeter"
    bl_idname = "MELODICCAP_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'MelodicCap'

    def draw(self, context):
        layout = self.layout

        # Target armature
        box = layout.box()
        box.label(text="Target:", icon='ARMATURE_DATA')
        if context.active_object and context.active_object.type == 'ARMATURE':
            box.label(text=f"  {context.active_object.name}")
            scale = context.active_object.scale
            if abs(scale.x - 1.0) > 0.01 or abs(scale.y - 1.0) > 0.01 or abs(scale.z - 1.0) > 0.01:
                box.label(text="  Scale not 1.0!", icon='ERROR')
        else:
            box.label(text="  (Select armature)")

        # Settings
        box = layout.box()
        box.label(text="Settings:", icon='SETTINGS')
        box.prop(context.scene, "melodiccap_start_frame")
        box.separator()

        # Foot contact settings
        box.label(text="Foot Contact:")
        box.prop(context.scene, "melodiccap_foot_plant_velocity")
        box.prop(context.scene, "melodiccap_foot_lift_velocity")
        box.prop(context.scene, "melodiccap_ground_clamp")
        box.separator()

        # Finger pose
        box.prop(context.scene, "melodiccap_finger_curl")
        box.separator()

        # Outlier filtering
        box.prop(context.scene, "melodiccap_filter_outliers")
        if context.scene.melodiccap_filter_outliers:
            box.prop(context.scene, "melodiccap_outlier_velocity")
        box.separator()

        # v3.0: Reference frame
        box.label(text="Reference Frame (v3.0):")
        box.prop(context.scene, "melodiccap_ref_scan_range")
        box.separator()

        # v3.0: Gap interpolation
        box.prop(context.scene, "melodiccap_interpolate_gaps")
        if context.scene.melodiccap_interpolate_gaps:
            box.prop(context.scene, "melodiccap_max_interp_gap")
        box.separator()

        # v3.0: Smoothing
        box.prop(context.scene, "melodiccap_smooth_curves")
        if context.scene.melodiccap_smooth_curves:
            box.prop(context.scene, "melodiccap_smooth_loc_cutoff")
            box.prop(context.scene, "melodiccap_smooth_rot_cutoff")
        box.separator()

        # Animation channels
        box.label(text="Animation Channels:")
        box.prop(context.scene, "melodiccap_animate_poles")
        box.prop(context.scene, "melodiccap_animate_ik_rot")
        box.prop(context.scene, "melodiccap_animate_fk")
        box.prop(context.scene, "melodiccap_animate_spine")

        # Actions
        box = layout.box()
        box.label(text="Actions:", icon='ACTION')
        box.operator("melodiccap.import_take", icon='IMPORT')

        row = box.row(align=True)
        row.operator("melodiccap.set_ik_mode", icon='CON_KINEMATIC')
        row.operator("melodiccap.set_fk_mode", icon='BONE_DATA')

        box.operator("melodiccap.clear", icon='X')

        box.separator()
        box.operator("melodiccap.diagnostic", icon='FILE_TEXT')


# =============================================================================
# REGISTRATION
# =============================================================================

classes = [
    MELODICCAP_OT_import,
    MELODICCAP_OT_clear,
    MELODICCAP_OT_set_ik_mode,
    MELODICCAP_OT_set_fk_mode,
    MELODICCAP_OT_diagnostic,
    MELODICCAP_PT_panel,
]

def register():
    for c in classes:
        bpy.utils.register_class(c)

    bpy.types.Scene.melodiccap_start_frame = IntProperty(
        name="Start Frame",
        default=1,
        min=1,
        description="Frame to start animation"
    )
    bpy.types.Scene.melodiccap_animate_poles = BoolProperty(
        name="Pole Targets",
        default=True,
        description="Animate elbow/knee pole targets for correct bend direction (CRITICAL)"
    )
    bpy.types.Scene.melodiccap_animate_ik_rot = BoolProperty(
        name="IK Rotation",
        default=True,
        description="Animate wrist/foot orientation based on forearm/shin direction"
    )
    bpy.types.Scene.melodiccap_animate_fk = BoolProperty(
        name="FK Rotations",
        default=True,
        description="Animate FK bone rotations (invisible in IK mode; available if you switch to FK)"
    )
    bpy.types.Scene.melodiccap_animate_spine = BoolProperty(
        name="Spine Animation",
        default=True,
        description="Animate spine using virtual midpoints (4-segment V2R)"
    )
    bpy.types.Scene.melodiccap_filter_outliers = BoolProperty(
        name="Filter Outliers",
        default=True,
        description="Remove landmark spikes caused by MediaPipe tracking glitches"
    )
    bpy.types.Scene.melodiccap_outlier_velocity = FloatProperty(
        name="Max Landmark Speed (m/s)",
        default=10.0,
        min=1.0,
        max=50.0,
        description="Maximum plausible landmark velocity. Faster movement is treated as an outlier"
    )
    bpy.types.Scene.melodiccap_ground_clamp = BoolProperty(
        name="Ground Clamp Feet",
        default=True,
        description="Prevent feet from going below floor level"
    )
    bpy.types.Scene.melodiccap_pin_threshold = FloatProperty(
        name="Foot Pin Threshold (legacy)",
        default=0.02,
        min=0.0,
        max=0.2,
        description="Legacy position-delta pinning. Use foot plant velocity instead"
    )
    bpy.types.Scene.melodiccap_foot_plant_velocity = FloatProperty(
        name="Foot Plant Velocity",
        default=0.3,
        min=0.05,
        max=1.0,
        description="Foot plants when velocity drops below this (m/s). Lower = stricter"
    )
    bpy.types.Scene.melodiccap_foot_lift_velocity = FloatProperty(
        name="Foot Lift Velocity",
        default=0.5,
        min=0.1,
        max=2.0,
        description="Foot lifts when velocity exceeds this (m/s). Higher = stickier"
    )
    bpy.types.Scene.melodiccap_finger_curl = BoolProperty(
        name="Relaxed Finger Pose",
        default=True,
        description="Apply natural finger curl (static rest pose)"
    )

    # v3.0 properties
    bpy.types.Scene.melodiccap_ref_scan_range = IntProperty(
        name="Reference Scan Range",
        default=60,
        min=1,
        max=300,
        description="Number of frames to scan for best A-pose reference (v3.0)"
    )
    bpy.types.Scene.melodiccap_interpolate_gaps = BoolProperty(
        name="Interpolate Gaps",
        default=True,
        description="Fill missing landmark data using cubic interpolation (v3.0)"
    )
    bpy.types.Scene.melodiccap_max_interp_gap = IntProperty(
        name="Max Interpolation Gap",
        default=10,
        min=1,
        max=60,
        description="Gaps longer than this many frames use hold instead of interpolation"
    )
    bpy.types.Scene.melodiccap_smooth_curves = BoolProperty(
        name="Butterworth Smoothing",
        default=True,
        description="Apply low-pass filter to all keyframes to reduce jitter (v3.0)"
    )
    bpy.types.Scene.melodiccap_smooth_loc_cutoff = FloatProperty(
        name="Location Cutoff (Hz)",
        default=6.0,
        min=1.0,
        max=15.0,
        description="Butterworth cutoff frequency for position channels. Lower = smoother"
    )
    bpy.types.Scene.melodiccap_smooth_rot_cutoff = FloatProperty(
        name="Rotation Cutoff (Hz)",
        default=4.0,
        min=1.0,
        max=12.0,
        description="Butterworth cutoff frequency for rotation channels. Lower = smoother"
    )

    log("MelodicCap Retargeter v3.0 registered")

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

    props = [
        'melodiccap_start_frame', 'melodiccap_filter_outliers',
        'melodiccap_outlier_velocity', 'melodiccap_animate_poles',
        'melodiccap_animate_ik_rot', 'melodiccap_animate_fk',
        'melodiccap_animate_spine', 'melodiccap_ground_clamp',
        'melodiccap_pin_threshold',
        'melodiccap_foot_plant_velocity', 'melodiccap_foot_lift_velocity',
        'melodiccap_finger_curl',
        # v3.0
        'melodiccap_ref_scan_range', 'melodiccap_interpolate_gaps',
        'melodiccap_max_interp_gap', 'melodiccap_smooth_curves',
        'melodiccap_smooth_loc_cutoff', 'melodiccap_smooth_rot_cutoff',
    ]
    for prop in props:
        try:
            delattr(bpy.types.Scene, prop)
        except AttributeError:
            pass

if __name__ == "__main__":
    register()
