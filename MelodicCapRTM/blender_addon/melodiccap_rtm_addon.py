"""
MelodicCap RTM Blender Addon v4.4
===================================
Imports JSON motion capture data and retargets to JaxRigify armature.

Supports two JSON formats:
- "melodiccap_rtm_v1" (COCO-WholeBody indices) — from MelodicCapRTM
- Legacy/no format field (MediaPipe indices) — from MelodicCapFresh

Bone names verified against JaxRigify:
- torso, hips, chest (torso controls)
- spine_fk, spine_fk.001, spine_fk.002, spine_fk.003
- neck, head
- upper_arm_fk.L/R, forearm_fk.L/R, hand_fk.L/R
- hand_ik.L/R, upper_arm_ik_target.L/R
- thigh_fk.L/R, shin_fk.L/R, foot_fk.L/R, toe_fk.L/R
- foot_ik.L/R, thigh_ik_target.L/R
- f_index.01/02/03.L/R, f_middle, f_ring, f_pinky
- thumb.01/02/03.L/R
"""

bl_info = {
    "name": "MelodicCap RTM Importer",
    "author": "Karsten Allen",
    "version": (4, 4),
    "blender": (4, 4, 0),
    "location": "View3D > Sidebar > MelodicCap",
    "description": "Import MelodicCap RTM/Fresh JSON motion capture to JaxRigify",
    "category": "Animation",
}

import bpy
import json
import os
import math
from mathutils import Vector, Quaternion, Matrix
from bpy_extras.io_utils import ImportHelper


# =============================================================================
# LANDMARK INDICES (COCO-WholeBody format)
# =============================================================================
# These match the indices output by MelodicCapRTM (RTMW detector).
# Legacy MediaPipe takes are converted to these indices on import.

class LM:
    """COCO-WholeBody landmark indices (internal standard)."""
    # Body
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    # Feet
    LEFT_BIG_TOE = 17
    LEFT_SMALL_TOE = 18
    LEFT_HEEL = 19
    RIGHT_BIG_TOE = 20
    RIGHT_SMALL_TOE = 21
    RIGHT_HEEL = 22

    # Left hand (91-111)
    LEFT_HAND_WRIST = 91
    LEFT_THUMB_CMC = 92
    LEFT_THUMB_MCP = 93
    LEFT_THUMB_IP = 94
    LEFT_THUMB_TIP = 95
    LEFT_INDEX_MCP = 96
    LEFT_INDEX_PIP = 97
    LEFT_INDEX_DIP = 98
    LEFT_INDEX_TIP = 99
    LEFT_MIDDLE_MCP = 100
    LEFT_MIDDLE_PIP = 101
    LEFT_MIDDLE_DIP = 102
    LEFT_MIDDLE_TIP = 103
    LEFT_RING_MCP = 104
    LEFT_RING_PIP = 105
    LEFT_RING_DIP = 106
    LEFT_RING_TIP = 107
    LEFT_PINKY_MCP = 108
    LEFT_PINKY_PIP = 109
    LEFT_PINKY_DIP = 110
    LEFT_PINKY_TIP = 111

    # Right hand (112-132)
    RIGHT_HAND_WRIST = 112
    RIGHT_THUMB_CMC = 113
    RIGHT_THUMB_MCP = 114
    RIGHT_THUMB_IP = 115
    RIGHT_THUMB_TIP = 116
    RIGHT_INDEX_MCP = 117
    RIGHT_INDEX_PIP = 118
    RIGHT_INDEX_DIP = 119
    RIGHT_INDEX_TIP = 120
    RIGHT_MIDDLE_MCP = 121
    RIGHT_MIDDLE_PIP = 122
    RIGHT_MIDDLE_DIP = 123
    RIGHT_MIDDLE_TIP = 124
    RIGHT_RING_MCP = 125
    RIGHT_RING_PIP = 126
    RIGHT_RING_DIP = 127
    RIGHT_RING_TIP = 128
    RIGHT_PINKY_MCP = 129
    RIGHT_PINKY_PIP = 130
    RIGHT_PINKY_DIP = 131
    RIGHT_PINKY_TIP = 132


# =============================================================================
# MEDIAPIPE LEGACY CONVERSION
# =============================================================================

# MediaPipe index -> LM index (for importing old MelodicCapFresh takes)
_MP_TO_LM = {
    0: LM.NOSE,
    2: LM.LEFT_EYE,
    5: LM.RIGHT_EYE,
    7: LM.LEFT_EAR,
    8: LM.RIGHT_EAR,
    11: LM.LEFT_SHOULDER,
    12: LM.RIGHT_SHOULDER,
    13: LM.LEFT_ELBOW,
    14: LM.RIGHT_ELBOW,
    15: LM.LEFT_WRIST,
    16: LM.RIGHT_WRIST,
    17: LM.LEFT_PINKY_TIP,
    18: LM.RIGHT_PINKY_TIP,
    19: LM.LEFT_INDEX_TIP,
    20: LM.RIGHT_INDEX_TIP,
    21: LM.LEFT_THUMB_TIP,
    22: LM.RIGHT_THUMB_TIP,
    23: LM.LEFT_HIP,
    24: LM.RIGHT_HIP,
    25: LM.LEFT_KNEE,
    26: LM.RIGHT_KNEE,
    27: LM.LEFT_ANKLE,
    28: LM.RIGHT_ANKLE,
    29: LM.LEFT_HEEL,
    30: LM.RIGHT_HEEL,
    31: LM.LEFT_BIG_TOE,
    32: LM.RIGHT_BIG_TOE,
}


def _convert_mp_frame(landmarks_3d):
    """Convert a single frame from MediaPipe indices to LM indices."""
    converted = {}
    for mp_key, coords in landmarks_3d.items():
        mp_idx = int(mp_key)
        if mp_idx in _MP_TO_LM:
            lm_idx = _MP_TO_LM[mp_idx]
            converted[str(lm_idx)] = coords
    return converted


def _is_rtm_format(data):
    """Check if JSON data is in the new RTM format."""
    fmt = data.get('format', '')
    return fmt.startswith('melodiccap_rtm')


# =============================================================================
# BONE MAPPING (using LM indices)
# =============================================================================

# FK rotation mapping: bone_name -> (start_landmark, end_landmark)
V2R_MAPPING = {
    # Arms
    "upper_arm_fk.L": (LM.LEFT_SHOULDER, LM.LEFT_ELBOW),
    "forearm_fk.L": (LM.LEFT_ELBOW, LM.LEFT_WRIST),
    "upper_arm_fk.R": (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW),
    "forearm_fk.R": (LM.RIGHT_ELBOW, LM.RIGHT_WRIST),

    # Legs
    "thigh_fk.L": (LM.LEFT_HIP, LM.LEFT_KNEE),
    "shin_fk.L": (LM.LEFT_KNEE, LM.LEFT_ANKLE),
    "foot_fk.L": (LM.LEFT_ANKLE, LM.LEFT_BIG_TOE),
    "thigh_fk.R": (LM.RIGHT_HIP, LM.RIGHT_KNEE),
    "shin_fk.R": (LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
    "foot_fk.R": (LM.RIGHT_ANKLE, LM.RIGHT_BIG_TOE),
}

# IK position targets: bone_name -> landmark_index
IK_TARGETS = {
    "hand_ik.L": LM.LEFT_WRIST,
    "hand_ik.R": LM.RIGHT_WRIST,
    "foot_ik.L": LM.LEFT_ANKLE,
    "foot_ik.R": LM.RIGHT_ANKLE,
}

# Pole targets for IK chains
POLE_TARGETS = {
    "upper_arm_ik_target.L": (LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST),
    "upper_arm_ik_target.R": (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
    "thigh_ik_target.L": (LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE),
    "thigh_ik_target.R": (LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
}

# Finger FK mapping: bone_name -> (start_keypoint, end_keypoint)
# Only used when wholebody (133 kp) data is available
FINGER_FK_MAPPING = {
    # Left hand - Thumb
    "thumb.01.L": (LM.LEFT_THUMB_CMC, LM.LEFT_THUMB_MCP),
    "thumb.02.L": (LM.LEFT_THUMB_MCP, LM.LEFT_THUMB_IP),
    "thumb.03.L": (LM.LEFT_THUMB_IP, LM.LEFT_THUMB_TIP),
    # Left hand - Index
    "f_index.01.L": (LM.LEFT_INDEX_MCP, LM.LEFT_INDEX_PIP),
    "f_index.02.L": (LM.LEFT_INDEX_PIP, LM.LEFT_INDEX_DIP),
    "f_index.03.L": (LM.LEFT_INDEX_DIP, LM.LEFT_INDEX_TIP),
    # Left hand - Middle
    "f_middle.01.L": (LM.LEFT_MIDDLE_MCP, LM.LEFT_MIDDLE_PIP),
    "f_middle.02.L": (LM.LEFT_MIDDLE_PIP, LM.LEFT_MIDDLE_DIP),
    "f_middle.03.L": (LM.LEFT_MIDDLE_DIP, LM.LEFT_MIDDLE_TIP),
    # Left hand - Ring
    "f_ring.01.L": (LM.LEFT_RING_MCP, LM.LEFT_RING_PIP),
    "f_ring.02.L": (LM.LEFT_RING_PIP, LM.LEFT_RING_DIP),
    "f_ring.03.L": (LM.LEFT_RING_DIP, LM.LEFT_RING_TIP),
    # Left hand - Pinky
    "f_pinky.01.L": (LM.LEFT_PINKY_MCP, LM.LEFT_PINKY_PIP),
    "f_pinky.02.L": (LM.LEFT_PINKY_PIP, LM.LEFT_PINKY_DIP),
    "f_pinky.03.L": (LM.LEFT_PINKY_DIP, LM.LEFT_PINKY_TIP),

    # Right hand - Thumb
    "thumb.01.R": (LM.RIGHT_THUMB_CMC, LM.RIGHT_THUMB_MCP),
    "thumb.02.R": (LM.RIGHT_THUMB_MCP, LM.RIGHT_THUMB_IP),
    "thumb.03.R": (LM.RIGHT_THUMB_IP, LM.RIGHT_THUMB_TIP),
    # Right hand - Index
    "f_index.01.R": (LM.RIGHT_INDEX_MCP, LM.RIGHT_INDEX_PIP),
    "f_index.02.R": (LM.RIGHT_INDEX_PIP, LM.RIGHT_INDEX_DIP),
    "f_index.03.R": (LM.RIGHT_INDEX_DIP, LM.RIGHT_INDEX_TIP),
    # Right hand - Middle
    "f_middle.01.R": (LM.RIGHT_MIDDLE_MCP, LM.RIGHT_MIDDLE_PIP),
    "f_middle.02.R": (LM.RIGHT_MIDDLE_PIP, LM.RIGHT_MIDDLE_DIP),
    "f_middle.03.R": (LM.RIGHT_MIDDLE_DIP, LM.RIGHT_MIDDLE_TIP),
    # Right hand - Ring
    "f_ring.01.R": (LM.RIGHT_RING_MCP, LM.RIGHT_RING_PIP),
    "f_ring.02.R": (LM.RIGHT_RING_PIP, LM.RIGHT_RING_DIP),
    "f_ring.03.R": (LM.RIGHT_RING_DIP, LM.RIGHT_RING_TIP),
    # Right hand - Pinky
    "f_pinky.01.R": (LM.RIGHT_PINKY_MCP, LM.RIGHT_PINKY_PIP),
    "f_pinky.02.R": (LM.RIGHT_PINKY_PIP, LM.RIGHT_PINKY_DIP),
    "f_pinky.03.R": (LM.RIGHT_PINKY_DIP, LM.RIGHT_PINKY_TIP),
}

# Which FK limb bones to SKIP in IK mode (IK solver handles these)
LIMB_FK_BONES = {
    "upper_arm_fk.L", "forearm_fk.L",
    "upper_arm_fk.R", "forearm_fk.R",
    "thigh_fk.L", "shin_fk.L",
    "thigh_fk.R", "shin_fk.R",
    "foot_fk.L", "foot_fk.R",
}

# Map IK target bones to their scale key
IK_SCALE_KEY = {
    "hand_ik.L": "arm.L",
    "hand_ik.R": "arm.R",
    "foot_ik.L": "leg.L",
    "foot_ik.R": "leg.R",
}


# =============================================================================
# DIAGNOSTIC LOGGER
# =============================================================================

class DiagLog:
    """Prints diagnostic info to Blender's terminal (System Console)."""
    PREFIX = "[MelodicCap]"

    @staticmethod
    def info(msg):
        print(f"{DiagLog.PREFIX} {msg}")

    @staticmethod
    def data(label, value):
        print(f"{DiagLog.PREFIX}   {label}: {value}")

    @staticmethod
    def section(title):
        print(f"{DiagLog.PREFIX} ── {title} ──")

    @staticmethod
    def bone(name, kind, detail=""):
        extra = f" | {detail}" if detail else ""
        print(f"{DiagLog.PREFIX}   [{kind}] {name}{extra}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_landmark(landmarks_3d, idx):
    """Get a landmark as Vector, handling string keys."""
    key = str(idx)
    if key in landmarks_3d:
        return Vector(landmarks_3d[key])
    return None


def set_bone_world_position(rig, bone, world_pos):
    """
    Set a pose bone's location so its head ends up at world_pos.

    bone.location is a DELTA from the bone's rest position in parent space.
    Simply assigning armature-space coordinates would add the rest position
    on top, causing stretching.
    """
    armature_pos = rig.matrix_world.inverted() @ world_pos
    # Subtract rest-pose head position to get the correct delta
    bone.location = armature_pos - Vector(bone.bone.head_local)


def compute_midpoint(landmarks_3d, idx1, idx2):
    """Compute midpoint between two landmarks."""
    p1 = get_landmark(landmarks_3d, idx1)
    p2 = get_landmark(landmarks_3d, idx2)
    if p1 and p2:
        return (p1 + p2) / 2
    return None


def compute_virtual_spine_points(landmarks_3d):
    """Compute virtual points for spine chain."""
    hip_mid = compute_midpoint(landmarks_3d, LM.LEFT_HIP, LM.RIGHT_HIP)
    shoulder_mid = compute_midpoint(landmarks_3d, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)

    if hip_mid is None or shoulder_mid is None:
        return {}

    spine_mid = (hip_mid + shoulder_mid) / 2
    spine_low = (hip_mid + spine_mid) / 2
    chest = (spine_mid + shoulder_mid) / 2

    return {
        'hip_mid': hip_mid,
        'spine_low': spine_low,
        'spine_mid': spine_mid,
        'chest': chest,
        'shoulder_mid': shoulder_mid,
    }


def compute_pole_position(p_root, p_mid, p_end, offset=0.3):
    """Compute pole target position for IK."""
    line_dir = (p_end - p_root).normalized()
    proj_length = (p_mid - p_root).dot(line_dir)
    proj_point = p_root + line_dir * proj_length
    pole_dir = (p_mid - proj_point)
    if pole_dir.length < 0.001:
        return p_mid
    pole_dir = pole_dir.normalized()
    return p_mid + pole_dir * offset


def has_hand_data(landmarks_3d):
    """Check if wholebody hand keypoints are present."""
    for i in range(91, 112):
        if str(i) in landmarks_3d:
            return True
    return False


# =============================================================================
# PROPORTIONAL RETARGETING
# =============================================================================

def measure_rig_proportions(rig):
    """Measure bone chain lengths from the rig's rest pose."""
    props = {}

    # Spine length (hip to shoulders)
    spine_bones = ["spine_fk", "spine_fk.001", "spine_fk.002", "spine_fk.003"]
    spine_len = 0.0
    for name in spine_bones:
        bone = rig.pose.bones.get(name)
        if bone:
            spine_len += (Vector(bone.bone.tail_local) - Vector(bone.bone.head_local)).length
    props['spine'] = spine_len

    # Arm and leg chain lengths
    for side in ['.L', '.R']:
        upper = rig.pose.bones.get(f"upper_arm_fk{side}")
        forearm = rig.pose.bones.get(f"forearm_fk{side}")
        if upper and forearm:
            props[f'arm{side}'] = (
                (Vector(upper.bone.tail_local) - Vector(upper.bone.head_local)).length +
                (Vector(forearm.bone.tail_local) - Vector(forearm.bone.head_local)).length
            )

        thigh = rig.pose.bones.get(f"thigh_fk{side}")
        shin = rig.pose.bones.get(f"shin_fk{side}")
        if thigh and shin:
            props[f'leg{side}'] = (
                (Vector(thigh.bone.tail_local) - Vector(thigh.bone.head_local)).length +
                (Vector(shin.bone.tail_local) - Vector(shin.bone.head_local)).length
            )

    # Hip center rest position
    torso = rig.pose.bones.get("torso")
    if torso:
        props['hip_rest'] = Vector(torso.bone.head_local)

    return props


def measure_mocap_proportions(landmarks_3d):
    """Measure body proportions from a mocap frame."""
    props = {}

    hip_mid = compute_midpoint(landmarks_3d, LM.LEFT_HIP, LM.RIGHT_HIP)
    shoulder_mid = compute_midpoint(landmarks_3d, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)

    if hip_mid and shoulder_mid:
        props['spine'] = (shoulder_mid - hip_mid).length

    for side_label, sh, el, wr in [
        ('.L', LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST),
        ('.R', LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
    ]:
        p_sh = get_landmark(landmarks_3d, sh)
        p_el = get_landmark(landmarks_3d, el)
        p_wr = get_landmark(landmarks_3d, wr)
        if p_sh and p_el and p_wr:
            props[f'arm{side_label}'] = (p_el - p_sh).length + (p_wr - p_el).length

    for side_label, hp, kn, an in [
        ('.L', LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE),
        ('.R', LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
    ]:
        p_hp = get_landmark(landmarks_3d, hp)
        p_kn = get_landmark(landmarks_3d, kn)
        p_an = get_landmark(landmarks_3d, an)
        if p_hp and p_kn and p_an:
            props[f'leg{side_label}'] = (p_kn - p_hp).length + (p_an - p_kn).length

    if hip_mid:
        props['hip_pos'] = hip_mid

    return props


def compute_scale_factors(rig_props, mocap_props):
    """Compute per-chain scale factors: rig_length / mocap_length."""
    scales = {}

    if rig_props.get('spine', 0) > 0.01 and mocap_props.get('spine', 0) > 0.01:
        scales['global'] = rig_props['spine'] / mocap_props['spine']
    else:
        scales['global'] = 1.0

    for key in ['arm.L', 'arm.R', 'leg.L', 'leg.R']:
        if rig_props.get(key, 0) > 0.01 and mocap_props.get(key, 0) > 0.01:
            scales[key] = rig_props[key] / mocap_props[key]
        else:
            scales[key] = scales['global']

    return scales


def validate_frame0_pose(landmarks_3d, mocap_props):
    """
    Validate that frame 0 is a clean standing A-pose.
    Logs warnings for asymmetry, non-upright spine, or suspicious proportions.
    Returns a dict of quality metrics.
    """
    quality = {'warnings': [], 'ok': True}

    # Arm symmetry check
    arm_l = mocap_props.get('arm.L', 0)
    arm_r = mocap_props.get('arm.R', 0)
    if arm_l > 0.01 and arm_r > 0.01:
        arm_asym = abs(arm_l - arm_r) / max(arm_l, arm_r) * 100
        quality['arm_asymmetry_pct'] = arm_asym
        if arm_asym > 3.0:
            quality['warnings'].append(
                f"ARM ASYMMETRY: L={arm_l:.4f} R={arm_r:.4f} ({arm_asym:.1f}% diff) — "
                f"frame 0 pose is not symmetric or triangulation error on one side")
            quality['ok'] = False

    # Leg symmetry check
    leg_l = mocap_props.get('leg.L', 0)
    leg_r = mocap_props.get('leg.R', 0)
    if leg_l > 0.01 and leg_r > 0.01:
        leg_asym = abs(leg_l - leg_r) / max(leg_l, leg_r) * 100
        quality['leg_asymmetry_pct'] = leg_asym
        if leg_asym > 3.0:
            quality['warnings'].append(
                f"LEG ASYMMETRY: L={leg_l:.4f} R={leg_r:.4f} ({leg_asym:.1f}% diff)")
            quality['ok'] = False

    # Spine uprightness: check if hip-to-shoulder vector is mostly vertical
    hip_mid = compute_midpoint(landmarks_3d, LM.LEFT_HIP, LM.RIGHT_HIP)
    shoulder_mid = compute_midpoint(landmarks_3d, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)
    if hip_mid and shoulder_mid:
        spine_vec = shoulder_mid - hip_mid
        if spine_vec.length > 0.01:
            spine_up = spine_vec.normalized()
            # Angle from vertical (Z axis)
            spine_tilt = math.degrees(math.acos(max(-1, min(1, spine_up.z))))
            quality['spine_tilt_deg'] = spine_tilt
            if spine_tilt > 15:
                quality['warnings'].append(
                    f"SPINE NOT UPRIGHT: {spine_tilt:.1f}° from vertical — "
                    f"frame 0 may not be a standing pose (sitting?)")
                quality['ok'] = False

    # Hip height sanity: typical standing hip is 0.8-1.1m above ground
    hip_pos = mocap_props.get('hip_pos')
    if hip_pos:
        quality['hip_height'] = hip_pos.z
        if hip_pos.z < 0.6:
            quality['warnings'].append(
                f"HIP TOO LOW: Z={hip_pos.z:.3f}m — person may be sitting/crouching in frame 0")
            quality['ok'] = False
        elif hip_pos.z > 1.2:
            quality['warnings'].append(
                f"HIP UNUSUALLY HIGH: Z={hip_pos.z:.3f}m — check calibration")

    # Shoulder width sanity
    p_ls = get_landmark(landmarks_3d, LM.LEFT_SHOULDER)
    p_rs = get_landmark(landmarks_3d, LM.RIGHT_SHOULDER)
    if p_ls and p_rs:
        shoulder_width = (p_ls - p_rs).length
        quality['shoulder_width'] = shoulder_width
        if shoulder_width < 0.2 or shoulder_width > 0.6:
            quality['warnings'].append(
                f"SHOULDER WIDTH UNUSUAL: {shoulder_width:.3f}m (expected 0.3-0.5m)")

    # Wrist height relative to hip: in A-pose, wrists should be near hip height
    for side, wrist_idx in [("L", LM.LEFT_WRIST), ("R", LM.RIGHT_WRIST)]:
        p_wr = get_landmark(landmarks_3d, wrist_idx)
        if p_wr and hip_pos:
            wrist_hip_dz = p_wr.z - hip_pos.z
            quality[f'wrist_{side}_dz'] = wrist_hip_dz
            if abs(wrist_hip_dz) > 0.15:
                quality['warnings'].append(
                    f"WRIST {side} NOT AT HIP HEIGHT: dZ={wrist_hip_dz:+.3f}m — "
                    f"arms may not be in A-pose")

    return quality


def scale_position(pos, hip_center_mocap, scale_factor):
    """Scale a position relative to the mocap hip center."""
    offset = pos - hip_center_mocap
    return hip_center_mocap + offset * scale_factor


# =============================================================================
# TEMPORAL SMOOTHING
# =============================================================================

def smooth_frames(frames, window=5):
    """Apply centered moving average to landmark positions."""
    if window < 2 or len(frames) < window:
        return frames

    half = window // 2
    smoothed = []

    for i in range(len(frames)):
        start = max(0, i - half)
        end = min(len(frames), i + half + 1)

        avg_landmarks = {}
        count = 0
        for j in range(start, end):
            lm = frames[j].get('landmarks_3d', {})
            if not lm:
                continue
            count += 1
            for key, coords in lm.items():
                if key not in avg_landmarks:
                    avg_landmarks[key] = [0.0, 0.0, 0.0]
                avg_landmarks[key][0] += coords[0]
                avg_landmarks[key][1] += coords[1]
                avg_landmarks[key][2] += coords[2]

        if count > 0:
            for key in avg_landmarks:
                avg_landmarks[key] = [c / count for c in avg_landmarks[key]]

        new_frame = dict(frames[i])
        new_frame['landmarks_3d'] = avg_landmarks
        smoothed.append(new_frame)

    return smoothed


# =============================================================================
# BUTTERWORTH LOW-PASS FILTER
# =============================================================================

def butterworth_filter_landmarks(frames, fps, cutoff_body=4.0, cutoff_feet=2.0):
    """
    Apply a 2nd-order Butterworth low-pass filter to landmark positions.

    Feet (ankles) get a lower cutoff to kill jitter since they need to be
    stable on the ground. Everything else gets a moderate cutoff.

    Uses forward-backward filtering (filtfilt equivalent) for zero phase lag.
    """
    if len(frames) < 5:
        return frames

    FOOT_INDICES = {str(LM.LEFT_ANKLE), str(LM.RIGHT_ANKLE),
                    str(LM.LEFT_BIG_TOE), str(LM.RIGHT_BIG_TOE),
                    str(LM.LEFT_SMALL_TOE), str(LM.RIGHT_SMALL_TOE),
                    str(LM.LEFT_HEEL), str(LM.RIGHT_HEEL),
                    str(LM.LEFT_KNEE), str(LM.RIGHT_KNEE)}

    # Collect all landmark keys
    all_keys = set()
    for f in frames:
        all_keys.update(f.get('landmarks_3d', {}).keys())

    # Build time series per landmark per axis
    n = len(frames)

    for key in all_keys:
        cutoff = cutoff_feet if key in FOOT_INDICES else cutoff_body

        # Extract XYZ series
        series = [None] * n
        for i, f in enumerate(frames):
            lm = f.get('landmarks_3d', {})
            if key in lm:
                series[i] = lm[key]

        # Skip if too many gaps
        valid = [i for i in range(n) if series[i] is not None]
        if len(valid) < 5:
            continue

        for axis in range(3):
            vals = [series[i][axis] for i in valid]
            filtered = _butter_lowpass_filtfilt(vals, cutoff, fps)
            for j, i in enumerate(valid):
                series[i][axis] = filtered[j]

        # Write back
        for i in valid:
            frames[i]['landmarks_3d'][key] = series[i]

    return frames


def _butter_lowpass_filtfilt(data, cutoff, fs, order=2):
    """
    Zero-phase Butterworth filter implemented without scipy.

    Uses cascaded biquad sections with forward-backward passes.
    """
    # Compute biquad coefficients for 2nd order Butterworth
    nyq = fs / 2.0
    wc = cutoff / nyq
    if wc >= 1.0:
        return data  # cutoff above nyquist, no filtering needed

    # Bilinear transform: analog butterworth -> digital
    warp = math.tan(math.pi * wc / 2.0)
    k = warp * warp
    norm = 1.0 / (1.0 + math.sqrt(2.0) * warp + k)

    b0 = k * norm
    b1 = 2.0 * b0
    b2 = b0
    a1 = 2.0 * (k - 1.0) * norm
    a2 = (1.0 - math.sqrt(2.0) * warp + k) * norm

    # Forward pass
    y = list(data)
    n = len(y)
    if n < 3:
        return y

    # Pad edges to reduce transients
    pad = min(3 * order, n - 1)
    front_pad = [2.0 * y[0] - y[i] for i in range(pad, 0, -1)]
    back_pad = [2.0 * y[-1] - y[-(i + 2)] for i in range(pad)]
    padded = front_pad + y + back_pad

    def _apply_filter(signal):
        out = list(signal)
        for i in range(2, len(out)):
            out[i] = b0 * signal[i] + b1 * signal[i - 1] + b2 * signal[i - 2] \
                     - a1 * out[i - 1] - a2 * out[i - 2]
        return out

    # Forward
    fwd = _apply_filter(padded)
    # Backward (reverse, filter, reverse)
    fwd.reverse()
    bwd = _apply_filter(fwd)
    bwd.reverse()

    # Strip padding
    return bwd[pad:pad + n]


# =============================================================================
# FK ROTATION HELPERS
# =============================================================================

def compute_fk_rotation(bone, target_dir_armature, parent_matrix_override=None):
    """
    Compute rotation_quaternion for a pose bone to point along target_dir.

    Two modes:
    1) parent_matrix_override is None (default): uses bone.bone.matrix_local
       (rest pose) to convert target direction. Works when parent is at rest
       or hasn't been updated yet (e.g., spine chain). This was the original
       behavior and is preserved for backward compatibility.

    2) parent_matrix_override is provided: uses the parent's DEFORMED matrix
       to properly account for rotations set earlier in the same frame.
       Required for arm FK bones whose parent chain (spine) has been rotated.

       Formula: bone direction = parent.matrix @ rest_offset @ pose_rot @ Y
       So: pose_rot @ Y = (parent.matrix @ rest_offset).inv() @ target_dir

    Pass parent_matrix_override='auto' to use bone.parent.matrix (requires
    depsgraph to have been flushed since parent was last rotated).
    """
    if parent_matrix_override is not None and bone.parent:
        if parent_matrix_override == 'auto':
            parent_world = bone.parent.matrix.to_3x3()
        else:
            parent_world = parent_matrix_override.to_3x3()
        rest_offset = bone.parent.bone.matrix_local.to_3x3().inverted() @ bone.bone.matrix_local.to_3x3()
        basis = parent_world @ rest_offset
        target_local = (basis.inverted() @ target_dir_armature).normalized()
    else:
        # Original rest-based math (backward compatible)
        rest_inv = bone.bone.matrix_local.to_3x3().inverted()
        target_local = (rest_inv @ target_dir_armature).normalized()
    return Vector((0, 1, 0)).rotation_difference(target_local)


def compute_spine_fk_chain(rig, spine_points, armature_inv_33, spine_rest_dir=None):
    """
    Compute FK rotations for the full spine chain (4 bones).

    Distributes the overall spine rotation equally across all chain bones.
    Each bone independently computes its full rotation from rest to target,
    then takes a 1/N fraction so the compound effect approximates the total.

    This avoids cumulative parent-space tracking which caused a 0.34m shoulder
    drop in the previous implementation.

    Returns list of (bone, quaternion) pairs to apply.
    """
    hip_mid = spine_points.get('hip_mid')
    shoulder_mid = spine_points.get('shoulder_mid')
    if not hip_mid or not shoulder_mid:
        return []

    spine_dir = (armature_inv_33 @ (shoulder_mid - hip_mid)).normalized()
    # Subtract frame-0 forward tilt (systematic ~8° lean)
    if spine_rest_dir is not None:
        spine_dir.y -= spine_rest_dir.y
        spine_dir = spine_dir.normalized()

    chain_names = ["spine_fk", "spine_fk.001", "spine_fk.002", "spine_fk.003"]
    bones = [(name, rig.pose.bones.get(name)) for name in chain_names]
    bones = [(name, b) for name, b in bones if b is not None]

    if not bones:
        return []

    n = len(bones)
    results = []

    for _name, bone in bones:
        # What rotation would this bone need if it were the only one?
        full_rot = compute_fk_rotation(bone, spine_dir)

        # Take 1/N of that rotation so N bones compound to the full rotation
        partial_rot = Quaternion().slerp(full_rot, 1.0 / n)
        results.append((bone, partial_rot))

    return results


# =============================================================================
# OPERATORS
# =============================================================================

class MELODICCAP_OT_import_json(bpy.types.Operator, ImportHelper):
    """Import MelodicCap JSON motion capture file"""
    bl_idname = "melodiccap.import_json"
    bl_label = "Import MelodicCap JSON"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(default="*.json", options={'HIDDEN'})

    use_ik: bpy.props.BoolProperty(
        name="Use IK Targets",
        description="Apply positions to IK targets (hands/feet)",
        default=True
    )

    use_fk: bpy.props.BoolProperty(
        name="Use FK Rotations",
        description="Apply rotations to FK bones (spine, fingers; limbs skipped in IK mode)",
        default=True
    )

    use_fingers: bpy.props.BoolProperty(
        name="Use Finger Tracking",
        description="Apply finger rotations from wholebody data (experimental, noisy at low FPS)",
        default=False
    )

    ground_clamp: bpy.props.BoolProperty(
        name="Ground Clamp Feet",
        description="Prevent feet from going below Z=0",
        default=True
    )

    pin_threshold: bpy.props.FloatProperty(
        name="Foot Pin Speed (m/s)",
        description="Foot speed threshold for pinning in meters/second (0 = disabled). Normalized by FPS so it works at any frame rate.",
        default=0.15,
        min=0.0,
        max=1.0
    )

    foot_floor_height: bpy.props.FloatProperty(
        name="Floor Contact Height",
        description="Height below which feet are considered on the ground (meters)",
        default=0.08,
        min=0.0,
        max=0.3
    )

    smooth_window: bpy.props.IntProperty(
        name="Smoothing Window",
        description="Moving average window size (1 = off, 3 = light, 5 = moderate). Capture already applies Kalman filtering.",
        default=3,
        min=1,
        max=15
    )

    butterworth: bpy.props.BoolProperty(
        name="Butterworth Filter",
        description="Apply Butterworth low-pass filter (reduces foot jitter significantly)",
        default=True
    )

    butter_cutoff_body: bpy.props.FloatProperty(
        name="Body Cutoff (Hz)",
        description="Butterworth cutoff frequency for body joints",
        default=4.0,
        min=1.0,
        max=15.0
    )

    butter_cutoff_feet: bpy.props.FloatProperty(
        name="Feet Cutoff (Hz)",
        description="Butterworth cutoff frequency for feet (lower = smoother, higher preserves foot lift during walking)",
        default=3.5,
        min=0.5,
        max=8.0
    )

    hand_rotation: bpy.props.BoolProperty(
        name="Hand Rotation from Forearm",
        description="Derive natural hand orientation from forearm direction (relaxed palms-inward pose)",
        default=True
    )

    calibration_frame: bpy.props.IntProperty(
        name="Calibration Frame",
        description="Frame to use for proportion measurement (0 = first frame). Must be a clean standing A-pose",
        default=0,
        min=0
    )

    spine_chain: bpy.props.BoolProperty(
        name="Spine FK Chain",
        description="Distribute spine rotation across all 4 FK bones (better sitting/bending). Off = single bone only",
        default=False
    )

    arm_fk: bpy.props.BoolProperty(
        name="Arm FK (hybrid)",
        description="Use FK rotations for arms instead of IK positioning. "
                    "Directly encodes elbow direction from mocap data — fixes chicken-wing in sitting poses. "
                    "Legs stay on IK for foot pinning",
        default=True
    )

    def execute(self, context):
        rig = context.active_object

        if not rig or rig.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the JaxRigify armature first")
            return {'CANCELLED'}

        # Load JSON
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load JSON: {e}")
            return {'CANCELLED'}

        frames = data.get('frames', [])
        if not frames:
            self.report({'ERROR'}, "No frames in JSON file")
            return {'CANCELLED'}

        # Detect format
        is_rtm = _is_rtm_format(data)
        format_name = data.get('format', 'mediapipe_legacy')
        fps = data.get('fps', 30)

        DiagLog.section("IMPORT START")
        DiagLog.data("File", os.path.basename(self.filepath))
        DiagLog.data("Format", format_name)
        DiagLog.data("Detector", data.get('detector', 'unknown'))
        DiagLog.data("Frames", len(frames))
        DiagLog.data("FPS", fps)
        DiagLog.data("IK mode", self.use_ik)
        DiagLog.data("FK mode", self.use_fk)
        DiagLog.data("Smoothing", self.smooth_window)
        DiagLog.data("Rig", rig.name)

        # Switch to pose mode
        bpy.ops.object.mode_set(mode='POSE')

        # Set IK/FK sliders per limb
        # Hybrid mode (arm_fk=True): arms use FK, legs use IK
        ik_fk_count = 0
        hybrid_mode = self.use_ik and self.arm_fk
        ik_fk_bones = []
        for bone in rig.pose.bones:
            if "IK_FK" in bone.keys():
                if hybrid_mode:
                    # Arms to FK (1.0), legs to IK (0.0)
                    is_arm = "arm" in bone.name.lower()
                    bone["IK_FK"] = 1.0 if is_arm else 0.0
                else:
                    bone["IK_FK"] = 0.0 if self.use_ik else 1.0
                # Keyframe the IK_FK property on frame 0 so it persists in animation
                bone.keyframe_insert(data_path='["IK_FK"]', frame=0)
                ik_fk_bones.append(bone)
                ik_fk_count += 1
        DiagLog.data("IK_FK sliders found", ik_fk_count)
        if hybrid_mode:
            DiagLog.info("Hybrid mode: arms=FK, legs=IK")

        # Convert legacy frames if needed
        if not is_rtm:
            DiagLog.info("Converting legacy MediaPipe indices...")
            for i, f in enumerate(frames):
                lm = f.get('landmarks_3d', {})
                if lm:
                    frames[i]['landmarks_3d'] = _convert_mp_frame(lm)

        # Calibration frame for proportion measurement
        cal_idx = min(self.calibration_frame, len(frames) - 1)
        if cal_idx != 0:
            DiagLog.info(f"Using frame {cal_idx} for proportion calibration (instead of frame 0)")
        first_landmarks = frames[cal_idx].get('landmarks_3d', {})

        # Check hand data availability
        finger_data_available = has_hand_data(first_landmarks)
        DiagLog.data("Finger data available", finger_data_available)
        DiagLog.data("Calibration frame", cal_idx)

        # =====================
        # TEMPORAL SMOOTHING
        # =====================
        if self.smooth_window > 1:
            DiagLog.info(f"Applying {self.smooth_window}-frame moving average...")
            frames = smooth_frames(frames, self.smooth_window)

        # =====================
        # BUTTERWORTH FILTER
        # =====================
        if self.butterworth:
            DiagLog.info(f"Applying Butterworth filter (body={self.butter_cutoff_body}Hz, feet={self.butter_cutoff_feet}Hz)...")
            frames = butterworth_filter_landmarks(
                frames, fps,
                cutoff_body=self.butter_cutoff_body,
                cutoff_feet=self.butter_cutoff_feet
            )

        # =====================
        # PROPORTIONAL SCALING
        # =====================
        DiagLog.section("PROPORTIONAL RETARGETING")

        rig_props = measure_rig_proportions(rig)
        DiagLog.data("Rig spine length", f"{rig_props.get('spine', 0):.4f}")
        DiagLog.data("Rig arm.L length", f"{rig_props.get('arm.L', 0):.4f}")
        DiagLog.data("Rig arm.R length", f"{rig_props.get('arm.R', 0):.4f}")
        DiagLog.data("Rig leg.L length", f"{rig_props.get('leg.L', 0):.4f}")
        DiagLog.data("Rig leg.R length", f"{rig_props.get('leg.R', 0):.4f}")
        if rig_props.get('hip_rest'):
            DiagLog.data("Rig hip rest pos", f"{rig_props['hip_rest']}")

        mocap_props = measure_mocap_proportions(first_landmarks)
        DiagLog.data("Mocap spine length", f"{mocap_props.get('spine', 0):.4f}")
        DiagLog.data("Mocap arm.L length", f"{mocap_props.get('arm.L', 0):.4f}")
        DiagLog.data("Mocap arm.R length", f"{mocap_props.get('arm.R', 0):.4f}")
        DiagLog.data("Mocap leg.L length", f"{mocap_props.get('leg.L', 0):.4f}")
        DiagLog.data("Mocap leg.R length", f"{mocap_props.get('leg.R', 0):.4f}")
        if mocap_props.get('hip_pos'):
            DiagLog.data("Mocap hip pos (frame 0)", f"{mocap_props['hip_pos']}")

        scales = compute_scale_factors(rig_props, mocap_props)
        DiagLog.data("Scale global (spine)", f"{scales['global']:.3f}")
        DiagLog.data("Scale arm.L", f"{scales.get('arm.L', 0):.3f}")
        DiagLog.data("Scale arm.R", f"{scales.get('arm.R', 0):.3f}")
        DiagLog.data("Scale leg.L", f"{scales.get('leg.L', 0):.3f}")
        DiagLog.data("Scale leg.R", f"{scales.get('leg.R', 0):.3f}")

        global_scale = scales['global']

        # Validate calibration frame pose quality
        DiagLog.section(f"CALIBRATION FRAME {cal_idx} POSE QUALITY CHECK")
        pose_quality = validate_frame0_pose(first_landmarks, mocap_props)
        if pose_quality['ok']:
            DiagLog.info(f"Frame {cal_idx} pose: OK (clean A-pose)")
        else:
            DiagLog.info(f"Frame {cal_idx} pose: PROBLEMS DETECTED")
            for w in pose_quality['warnings']:
                DiagLog.info(f"  ⚠ {w}")
        if 'spine_tilt_deg' in pose_quality:
            DiagLog.data("Spine tilt from vertical", f"{pose_quality['spine_tilt_deg']:.1f}°")
        if 'arm_asymmetry_pct' in pose_quality:
            DiagLog.data("Arm length asymmetry", f"{pose_quality['arm_asymmetry_pct']:.1f}%")
        if 'shoulder_width' in pose_quality:
            DiagLog.data("Shoulder width", f"{pose_quality['shoulder_width']:.3f}m")
        for side in ['L', 'R']:
            if f'wrist_{side}_dz' in pose_quality:
                DiagLog.data(f"Wrist {side} vs hip dZ", f"{pose_quality[f'wrist_{side}_dz']:+.3f}m")

        # Hip height is determined by leg length, not spine length.
        # Use average leg scale for vertical positioning to prevent hunching.
        leg_scale_avg = (scales.get('leg.L', global_scale) + scales.get('leg.R', global_scale)) / 2.0
        hip_height_scale = leg_scale_avg
        DiagLog.data("Hip height scale (leg avg)", f"{hip_height_scale:.3f}")

        # Precompute armature inverse matrix (3x3 for directions, 4x4 for positions)
        armature_inv = rig.matrix_world.inverted()
        armature_inv_33 = armature_inv.to_3x3()

        # Track previous foot positions for pinning
        # raw = pre-pin position (for speed calc), final = post-pin (for output)
        prev_foot_raw = {"L": None, "R": None}
        prev_timestamp = 0
        pinned_foot_pos = {"L": None, "R": None}
        # Smooth unpin: blend from pinned to raw over N frames
        unpin_blend = {"L": 0.0, "R": 0.0}  # 0 = fully pinned, 1 = fully unpinned
        pin_blend = {"L": 0.0, "R": 0.0}    # 0 = raw, 1 = fully pinned
        UNPIN_BLEND_FRAMES = 8  # frames to blend over when unpinning
        PIN_BLEND_FRAMES = 6    # frames to blend over when pinning (slightly faster)
        # Walking-aware pinning: track hip position when foot was pinned
        # If hip XY moves too far from the pin point, force-unpin (person walked away)
        pin_hip_pos = {"L": None, "R": None}  # hip XY when foot was pinned
        pin_frame_count = {"L": 0, "R": 0}    # frames since pin started
        HIP_DRIFT_UNPIN = 0.12    # meters — if hip moves this far in XY, force unpin
        MIN_PIN_FRAMES = 4        # minimum frames to stay pinned (prevent toggling)
        PIN_SLIDE_RATE = 0.15     # how fast pin slides toward foot during walking (0-1)

        # Precompute ankle-to-ground offset from first N standing frames.
        # The mocap ankle keypoint is anatomically above the ground (~5-7cm).
        # Without this offset, unpinned feet float visibly above the floor.
        foot_z_offset = {"L": 0.0, "R": 0.0}
        OFFSET_SAMPLE_FRAMES = min(20, len(frames))
        for side_key, ankle_idx in [("L", LM.LEFT_ANKLE), ("R", LM.RIGHT_ANKLE)]:
            z_samples = []
            for i in range(OFFSET_SAMPLE_FRAMES):
                f_data = frames[i]
                lm = f_data.get('landmarks_3d', {})
                hip_c = compute_midpoint(lm, LM.LEFT_HIP, LM.RIGHT_HIP)
                ankle = get_landmark(lm, ankle_idx)
                if hip_c and ankle:
                    # Replicate the same scaling the main loop does
                    chain_k = f'leg.{side_key}'
                    cs = scales.get(chain_k, global_scale)
                    pos = scale_position(ankle, hip_c, cs)
                    pos_z = pos.z + (hip_height_scale - 1.0) * hip_c.z
                    z_samples.append(pos_z)
            if z_samples:
                foot_z_offset[side_key] = sum(z_samples) / len(z_samples)
        DiagLog.data("Foot Z offset L", f"{foot_z_offset['L']:.4f}m")
        DiagLog.data("Foot Z offset R", f"{foot_z_offset['R']:.4f}m")

        # Arm splay CLAMP: limit how far outward upper arms can splay.
        # The chicken wing error is in CAMERA space (triangulation pushes
        # elbows outward), but we can't subtract a bias in ARMATURE space
        # because the correction direction rotates with the body.
        # Instead, we CLAMP the maximum outward X component to prevent
        # extreme splay without pushing arms inward at other yaw angles.
        ARM_SPLAY_MAX = 0.10  # max |X| for upper arm direction (outward only)

        # Spine rest direction: measure frame-0 spine direction so we can
        # subtract the systematic 8° forward tilt from spine FK.
        spine_rest_dir = None
        cal_hip_mid = compute_midpoint(first_landmarks, LM.LEFT_HIP, LM.RIGHT_HIP)
        cal_sh_mid = compute_midpoint(first_landmarks, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)
        if cal_hip_mid and cal_sh_mid:
            spine_rest_dir = (armature_inv_33 @ (cal_sh_mid - cal_hip_mid)).normalized()
            DiagLog.data("Spine rest dir", f"({spine_rest_dir.x:.3f}, {spine_rest_dir.y:.3f}, {spine_rest_dir.z:.3f})")
            spine_rest_tilt_y = spine_rest_dir.y  # forward lean component (~0.15)
            DiagLog.data("Spine rest tilt Y", f"{spine_rest_tilt_y:.3f} (subtracted from spine FK)")

        # Foot diagnostics: collect per-frame data for analysis
        foot_diag = {"L": [], "R": []}
        # Arm diagnostics: collect per-frame ratios for summary
        arm_ratios = {"L": [], "R": []}
        # Arm velocity clamp: track previous hand IK positions to reject spikes
        prev_hand_pos = {"L": None, "R": None}
        ARM_MAX_SPEED = 8.0  # m/s — fast arm swing is ~6 m/s, reject above this
        ARM_MIN_RATIO = 0.55  # minimum arm reach ratio — prevents extreme chicken-wing

        DiagLog.section("PROCESSING FRAMES")
        log_every = max(1, len(frames) // 10)

        # Process each frame
        for frame_idx, frame_data in enumerate(frames):
            timestamp = frame_data.get('timestamp', 0)
            frame_num = int(timestamp * fps)
            context.scene.frame_set(frame_num)

            landmarks_3d = frame_data.get('landmarks_3d', {})
            if not landmarks_3d:
                continue

            # Compute virtual spine points
            spine_points = compute_virtual_spine_points(landmarks_3d)
            hip_center = spine_points.get('hip_mid')

            if frame_idx % log_every == 0:
                DiagLog.info(f"Frame {frame_idx}/{len(frames)} (t={timestamp:.2f}s, blender={frame_num})")
                if hip_center:
                    DiagLog.data("  hip_raw", f"({hip_center.x:.3f}, {hip_center.y:.3f}, {hip_center.z:.3f})")
                    # Show what the scaled hip will be
                    sh_z = hip_center.z * hip_height_scale
                    DiagLog.data("  hip_scaled_z", f"{sh_z:.3f} (raw_z * {hip_height_scale:.3f})")
                    # Hip drop from frame 0
                    f0_hip = mocap_props.get('hip_pos')
                    if f0_hip:
                        dz = hip_center.z - f0_hip.z
                        DiagLog.data("  hip_dz_from_f0", f"{dz:+.3f}m")

            # =====================
            # SIT / STAND DETECTION
            # =====================
            # When hip drops >15cm from frame 0, we're sitting/crouching.
            # Switch legs from IK (foot-anchored) to FK (rotation-based)
            # and disable foot pinning which makes no sense seated.
            SIT_THRESHOLD = -0.15  # meters below frame-0 hip → sit
            STAND_THRESHOLD = -0.10  # meters below frame-0 hip → stand (hysteresis)
            is_sitting = mocap_props.get('_prev_sitting', False)
            mocap_hip_f0 = mocap_props.get('hip_pos')
            if hip_center and mocap_hip_f0:
                hip_dz = hip_center.z - mocap_hip_f0.z
                # Hysteresis: must cross a different threshold to change state
                if is_sitting:
                    is_sitting = hip_dz < STAND_THRESHOLD  # must rise above -0.10 to stand
                else:
                    is_sitting = hip_dz < SIT_THRESHOLD    # must drop below -0.15 to sit

                # Update IK_FK sliders for legs when sit state changes
                if is_sitting != mocap_props.get('_prev_sitting', False):
                    for ikfk_bone in ik_fk_bones:
                        is_leg = "thigh" in ikfk_bone.name.lower()
                        if is_leg:
                            # Sitting: legs → FK (1.0). Standing: legs → IK (0.0)
                            ikfk_bone["IK_FK"] = 1.0 if is_sitting else 0.0
                            ikfk_bone.keyframe_insert(data_path='["IK_FK"]', frame=frame_num)
                    if frame_idx % log_every == 0 or is_sitting != mocap_props.get('_prev_sitting', False):
                        DiagLog.info(f"  SIT DETECT: {'SITTING' if is_sitting else 'STANDING'} (hip_dz={hip_dz:+.3f}m)")
                    mocap_props['_prev_sitting'] = is_sitting

            # =====================
            # ROOT / TORSO POSITION (scaled)
            # =====================
            torso = rig.pose.bones.get("torso")
            if torso and hip_center:
                scaled_hip = hip_center.copy()
                # Use leg-based scale for height (prevents hunching from short mocap spine)
                scaled_hip.z *= hip_height_scale
                mocap_hip_frame0 = mocap_props.get('hip_pos')
                if mocap_hip_frame0:
                    dx = hip_center.x - mocap_hip_frame0.x
                    dy = hip_center.y - mocap_hip_frame0.y
                    scaled_hip.x = mocap_hip_frame0.x * global_scale + dx * global_scale
                    scaled_hip.y = mocap_hip_frame0.y * global_scale + dy * global_scale

                set_bone_world_position(rig, torso, scaled_hip)
                torso.keyframe_insert(data_path="location")

            # =====================
            # TORSO ROTATION (yaw + pitch from shoulder/hip lines)
            # Yaw: body facing direction from averaged shoulder+hip lines
            # Pitch: forward lean from spine vector (hip_mid → shoulder_mid)
            # Spine FK chain handles RESIDUAL curvature on top of this.
            # =====================
            if torso:
                p_ls = get_landmark(landmarks_3d, LM.LEFT_SHOULDER)
                p_rs = get_landmark(landmarks_3d, LM.RIGHT_SHOULDER)
                p_lh = get_landmark(landmarks_3d, LM.LEFT_HIP)
                p_rh = get_landmark(landmarks_3d, LM.RIGHT_HIP)

                if p_ls and p_rs and p_lh and p_rh:
                    # --- YAW (Z rotation): body facing direction ---
                    shoulder_vec = (p_ls - p_rs)
                    hip_vec = (p_lh - p_rh)
                    body_right = ((shoulder_vec + hip_vec) / 2)
                    body_right.z = 0  # Project to ground plane

                    yaw_angle = 0.0
                    if body_right.length > 0.01:
                        body_right = body_right.normalized()
                        yaw_angle = math.atan2(-body_right.y, body_right.x)

                    # --- PITCH (X rotation): forward/backward lean ---
                    # Measure spine tilt: angle between hip→shoulder and vertical
                    # Subtract frame-0 tilt so standing = 0° pitch
                    pitch_angle = 0.0
                    shoulder_mid = spine_points.get('shoulder_mid')
                    hip_mid_sp = spine_points.get('hip_mid')
                    if shoulder_mid and hip_mid_sp:
                        spine_v = shoulder_mid - hip_mid_sp
                        if spine_v.length > 0.01:
                            # Pitch = angle from vertical in the sagittal plane
                            # Convert spine vector to armature space for correct plane
                            spine_arm = armature_inv_33 @ spine_v
                            # Pitch is rotation around X axis (forward lean = -Y in Blender)
                            pitch_from_vert = math.atan2(-spine_arm.y, spine_arm.z)

                            # Subtract rest pitch (frame 0 lean) so A-pose = zero
                            if 'torso_rest_pitch' not in mocap_props:
                                mocap_props['torso_rest_pitch'] = pitch_from_vert
                            pitch_angle = pitch_from_vert - mocap_props['torso_rest_pitch']

                    # Combine yaw + pitch as quaternion
                    # Order: yaw (Z) then pitch (X) — yaw first so pitch
                    # is applied in the rotated body frame
                    torso.rotation_mode = 'QUATERNION'
                    q_yaw = Quaternion((0, 0, 1), yaw_angle)
                    q_pitch = Quaternion((1, 0, 0), pitch_angle)
                    torso.rotation_quaternion = q_yaw @ q_pitch
                    torso.keyframe_insert(data_path="rotation_quaternion")

                    # Store torso yaw for head attenuation at large angles
                    mocap_props['_current_torso_yaw'] = yaw_angle

                    if frame_idx % log_every == 0:
                        DiagLog.data("  torso_yaw", f"{math.degrees(yaw_angle):.1f}°")
                        DiagLog.data("  torso_pitch", f"{math.degrees(pitch_angle):.1f}° (from vertical, rest-subtracted)")

            # =====================
            # FK ROTATIONS
            # =====================
            if self.use_fk:
                # --- Spine FK ---
                if self.spine_chain and spine_points.get('hip_mid') and spine_points.get('shoulder_mid'):
                    # Full chain: distribute rotation across all 4 spine FK bones
                    # Subtract frame-0 tilt before computing FK
                    chain_results = compute_spine_fk_chain(rig, spine_points, armature_inv_33, spine_rest_dir)
                    for bone, quat in chain_results:
                        bone.rotation_mode = 'QUATERNION'
                        bone.rotation_quaternion = quat
                        bone.keyframe_insert(data_path="rotation_quaternion")
                    if frame_idx % log_every == 0:
                        spine_dir = (armature_inv_33 @ (spine_points['shoulder_mid'] - spine_points['hip_mid'])).normalized()
                        DiagLog.data("  spine_dir", f"({spine_dir.x:.3f}, {spine_dir.y:.3f}, {spine_dir.z:.3f})")
                        DiagLog.data("  spine_chain_bones", f"{len(chain_results)}")
                else:
                    # Single bone fallback (default, proven stable)
                    spine_fk = rig.pose.bones.get("spine_fk")
                    if spine_fk and spine_points.get('hip_mid') and spine_points.get('shoulder_mid'):
                        spine_dir = (armature_inv_33 @ (spine_points['shoulder_mid'] - spine_points['hip_mid'])).normalized()
                        # Subtract frame-0 forward tilt (systematic ~8° lean)
                        if spine_rest_dir is not None:
                            spine_dir.y -= spine_rest_tilt_y
                            spine_dir = spine_dir.normalized()
                        if frame_idx % log_every == 0:
                            DiagLog.data("  spine_dir", f"({spine_dir.x:.3f}, {spine_dir.y:.3f}, {spine_dir.z:.3f})")
                        spine_fk.rotation_mode = 'QUATERNION'
                        spine_fk.rotation_quaternion = compute_fk_rotation(spine_fk, spine_dir)
                        spine_fk.keyframe_insert(data_path="rotation_quaternion")

                # --- Head/Neck FK from face keypoints ---
                # Neck: point from shoulder_mid toward ear_mid (upward direction)
                # Head: do NOT apply compute_fk_rotation — the head bone Y axis
                # points up, and ear_mid→nose is horizontal. Applying it rotates
                # the head to face the ground. Neck rotation alone captures head
                # tilt and some turn. Head yaw needs relative rotation (future).
                p_nose = get_landmark(landmarks_3d, LM.NOSE)
                p_lear = get_landmark(landmarks_3d, LM.LEFT_EAR)
                p_rear = get_landmark(landmarks_3d, LM.RIGHT_EAR)

                if p_lear and p_rear:
                    ear_mid = (p_lear + p_rear) / 2.0

                    # Head/neck confidence: attenuate when turned sideways to cameras.
                    # Ear keypoints become unreliable in profile view (one ear occluded).
                    # Full confidence below 30° torso yaw, fades to 0 at 60°.
                    torso_yaw_abs = abs(mocap_props.get('_current_torso_yaw', 0.0))
                    HEAD_YAW_FULL = math.radians(30)   # full confidence below this
                    HEAD_YAW_ZERO = math.radians(60)   # zero confidence above this
                    if torso_yaw_abs <= HEAD_YAW_FULL:
                        head_confidence = 1.0
                    elif torso_yaw_abs >= HEAD_YAW_ZERO:
                        head_confidence = 0.0
                    else:
                        t = (torso_yaw_abs - HEAD_YAW_FULL) / (HEAD_YAW_ZERO - HEAD_YAW_FULL)
                        head_confidence = 1.0 - t * t * (3 - 2 * t)  # smoothstep fade

                    neck_bone = rig.pose.bones.get("neck")
                    if neck_bone and spine_points.get('shoulder_mid'):
                        neck_dir = (armature_inv_33 @ (ear_mid - spine_points['shoulder_mid'])).normalized()
                        neck_bone.rotation_mode = 'QUATERNION'
                        neck_rot = compute_fk_rotation(neck_bone, neck_dir)
                        # Blend toward identity when confidence is low
                        if head_confidence < 1.0:
                            neck_rot = Quaternion().slerp(neck_rot, head_confidence)
                        neck_bone.rotation_quaternion = neck_rot
                        neck_bone.keyframe_insert(data_path="rotation_quaternion")

                    # Head yaw + pitch: rotate head based on ear/nose keypoints
                    # relative to shoulder orientation (head turn + nod relative to torso)
                    head_bone = rig.pose.bones.get("head")
                    if head_bone and p_nose:
                        p_ls = get_landmark(landmarks_3d, LM.LEFT_SHOULDER)
                        p_rs = get_landmark(landmarks_3d, LM.RIGHT_SHOULDER)
                        if p_ls and p_rs:
                            # Ear line angle vs shoulder line angle = head yaw
                            ear_vec = p_lear - p_rear
                            shoulder_vec = p_ls - p_rs
                            ear_angle = math.atan2(ear_vec.y, ear_vec.x)
                            shoulder_angle = math.atan2(shoulder_vec.y, shoulder_vec.x)
                            head_yaw = ear_angle - shoulder_angle
                            # Clamp to reasonable range (avoid wild rotations)
                            head_yaw = max(-0.7, min(0.7, head_yaw))
                            # Attenuate when turned sideways
                            head_yaw *= head_confidence

                            # Apply as rotation around bone's local Y axis (up)
                            # Head pitch is handled by neck FK (compute_fk_rotation
                            # on neck bone from shoulder_mid to ear_mid). Adding
                            # explicit pitch here doubles the effect and causes the
                            # head to stare at the ground due to nose-ear bias.
                            head_bone.rotation_mode = 'QUATERNION'
                            head_bone.rotation_quaternion = Quaternion((0, 1, 0), head_yaw)
                            head_bone.keyframe_insert(data_path="rotation_quaternion")

                            if frame_idx % log_every == 0:
                                DiagLog.data("  head_yaw", f"{math.degrees(head_yaw):.1f}° (ear_ang={math.degrees(ear_angle):.1f} - sh_ang={math.degrees(shoulder_angle):.1f}, conf={head_confidence:.2f})")
                                # Log head pitch for diagnostics (handled by neck FK)
                                nose_ear_vec = p_nose - ear_mid
                                head_pitch = math.degrees(math.atan2(-nose_ear_vec.z, math.sqrt(nose_ear_vec.x**2 + nose_ear_vec.y**2)))
                                DiagLog.data("  head_pitch_via_neck", f"{head_pitch:.1f}° (applied via neck FK, conf={head_confidence:.2f})")

                # --- Flush depsgraph so parent matrices are current ---
                # Spine/torso/neck/head rotations were set above. Without this
                # update, bone.parent.matrix still reflects the PREVIOUS frame,
                # making child FK rotations (arms, legs) incorrect.
                bpy.context.view_layer.update()

                # --- Limb FK ---
                # Full FK mode: apply all limb bones
                # Hybrid mode (arm_fk + use_ik): arm FK always, leg FK when sitting
                # Pure IK mode: skip all limb FK (unless sitting)
                leg_fk_bones = {
                    "thigh_fk.L", "shin_fk.L", "foot_fk.L",
                    "thigh_fk.R", "shin_fk.R", "foot_fk.R",
                }

                # Determine which FK bones to process this frame
                do_arm_fk = hybrid_mode or (self.use_fk and not self.use_ik)
                do_leg_fk = is_sitting or (self.use_fk and not self.use_ik)

                if not do_arm_fk and not do_leg_fk:
                    pass  # pure IK + standing, no FK limbs
                elif hybrid_mode or (do_arm_fk and not do_leg_fk):
                    # Hybrid: process arm FK bones in parent→child order
                    # so we can pass upper_arm's computed matrix to forearm
                    for side in ["L", "R"]:
                        ua_name = f"upper_arm_fk.{side}"
                        fa_name = f"forearm_fk.{side}"
                        ua_bone = rig.pose.bones.get(ua_name)
                        fa_bone = rig.pose.bones.get(fa_name)

                        # Compute arm FK confidence from wrist-to-shoulder ratio
                        # When hands are on lap (sitting), ratio < 0.7 and FK directions
                        # are garbage — blend toward rest pose
                        sh_idx = LM.LEFT_SHOULDER if side == "L" else LM.RIGHT_SHOULDER
                        wr_idx = LM.LEFT_WRIST if side == "L" else LM.RIGHT_WRIST
                        p_sh_raw = get_landmark(landmarks_3d, sh_idx)
                        p_wr_raw = get_landmark(landmarks_3d, wr_idx)
                        rig_arm_len = rig_props.get(f'arm.{side}', 0.53)
                        arm_fk_conf = 1.0
                        if p_sh_raw is not None and p_wr_raw is not None and rig_arm_len > 0:
                            raw_dist = (p_wr_raw - p_sh_raw).length
                            arm_ratio = raw_dist / rig_arm_len
                            if arm_ratio < 0.7:
                                # Blend: 0.7 → conf=1.0, 0.4 → conf=0.0
                                arm_fk_conf = max(0.0, (arm_ratio - 0.4) / 0.3)

                        # Upper arm first
                        ua_mapping = V2R_MAPPING.get(ua_name)
                        ua_expected_matrix = None
                        if ua_bone and ua_mapping:
                            p_start = get_landmark(landmarks_3d, ua_mapping[0])
                            p_end = get_landmark(landmarks_3d, ua_mapping[1])
                            if p_start is not None and p_end is not None:
                                target_dir = (armature_inv_33 @ (p_end - p_start)).normalized()

                                # Clamp outward arm splay (chicken wing fix)
                                # L arm: outward is +X, R arm: outward is -X
                                if side == "L" and target_dir.x > ARM_SPLAY_MAX:
                                    target_dir.x = ARM_SPLAY_MAX
                                    target_dir = target_dir.normalized()
                                elif side == "R" and target_dir.x < -ARM_SPLAY_MAX:
                                    target_dir.x = -ARM_SPLAY_MAX
                                    target_dir = target_dir.normalized()

                                ua_bone.rotation_mode = 'QUATERNION'
                                ua_rot = compute_fk_rotation(ua_bone, target_dir, 'auto')
                                # Blend toward rest pose when arm ratio is low (seated)
                                if arm_fk_conf < 1.0:
                                    ua_rot = Quaternion().slerp(ua_rot, arm_fk_conf)
                                ua_bone.rotation_quaternion = ua_rot
                                ua_bone.keyframe_insert(data_path="rotation_quaternion")

                                # Compute what upper_arm's matrix WILL be after this rotation
                                # so forearm can use it without a depsgraph update
                                if ua_bone.parent:
                                    parent_mat = ua_bone.parent.matrix
                                    rest_off = ua_bone.parent.bone.matrix_local.inverted() @ ua_bone.bone.matrix_local
                                else:
                                    parent_mat = Matrix.Identity(4)
                                    rest_off = ua_bone.bone.matrix_local
                                ua_expected_matrix = parent_mat @ rest_off @ ua_rot.to_matrix().to_4x4()

                                if frame_idx % log_every == 0:
                                    DiagLog.data(f"  arm_fk.{ua_name}",
                                        f"dir=({target_dir.x:.3f},{target_dir.y:.3f},{target_dir.z:.3f})")

                        # Forearm second (uses upper_arm's computed matrix)
                        fa_mapping = V2R_MAPPING.get(fa_name)
                        if fa_bone and fa_mapping:
                            p_start = get_landmark(landmarks_3d, fa_mapping[0])
                            p_end = get_landmark(landmarks_3d, fa_mapping[1])
                            if p_start is not None and p_end is not None:
                                target_dir = (armature_inv_33 @ (p_end - p_start)).normalized()
                                fa_bone.rotation_mode = 'QUATERNION'
                                fa_rot = compute_fk_rotation(
                                    fa_bone, target_dir, ua_expected_matrix)
                                # Blend toward rest pose when arm ratio is low (seated)
                                if arm_fk_conf < 1.0:
                                    fa_rot = Quaternion().slerp(fa_rot, arm_fk_conf)
                                fa_bone.rotation_quaternion = fa_rot
                                fa_bone.keyframe_insert(data_path="rotation_quaternion")

                                if frame_idx % log_every == 0:
                                    DiagLog.data(f"  arm_fk.{fa_name}",
                                        f"dir=({target_dir.x:.3f},{target_dir.y:.3f},{target_dir.z:.3f})")

                    # Leg FK when sitting (parent→child: thigh → shin → foot)
                    if do_leg_fk:
                        for side in ["L", "R"]:
                            leg_chain = [
                                f"thigh_fk.{side}",
                                f"shin_fk.{side}",
                                f"foot_fk.{side}",
                            ]
                            prev_expected = None
                            for bone_name in leg_chain:
                                mapping = V2R_MAPPING.get(bone_name)
                                if not mapping:
                                    continue
                                bone = rig.pose.bones.get(bone_name)
                                if not bone:
                                    continue
                                p_start = get_landmark(landmarks_3d, mapping[0])
                                p_end = get_landmark(landmarks_3d, mapping[1])
                                if p_start is None or p_end is None:
                                    continue
                                target_dir = (armature_inv_33 @ (p_end - p_start)).normalized()
                                # Use tracked parent matrix or 'auto' (depsgraph flushed)
                                parent_ovr = prev_expected if prev_expected is not None else 'auto'
                                bone.rotation_mode = 'QUATERNION'
                                rot = compute_fk_rotation(bone, target_dir, parent_ovr)
                                bone.rotation_quaternion = rot
                                bone.keyframe_insert(data_path="rotation_quaternion")
                                # Track expected matrix for next child in chain
                                if bone.parent:
                                    p_mat = prev_expected if prev_expected is not None else bone.parent.matrix
                                    rest_off = bone.parent.bone.matrix_local.inverted() @ bone.bone.matrix_local
                                else:
                                    p_mat = Matrix.Identity(4)
                                    rest_off = bone.bone.matrix_local
                                prev_expected = p_mat @ rest_off @ rot.to_matrix().to_4x4()

                            if frame_idx % log_every == 0:
                                DiagLog.data(f"  leg_fk.{side}", "sitting FK active")

                elif do_leg_fk and not do_arm_fk:
                    # Sitting with pure IK arms — only process leg FK
                    for side in ["L", "R"]:
                        leg_chain = [
                            f"thigh_fk.{side}",
                            f"shin_fk.{side}",
                            f"foot_fk.{side}",
                        ]
                        prev_expected = None
                        for bone_name in leg_chain:
                            mapping = V2R_MAPPING.get(bone_name)
                            if not mapping:
                                continue
                            bone = rig.pose.bones.get(bone_name)
                            if not bone:
                                continue
                            p_start = get_landmark(landmarks_3d, mapping[0])
                            p_end = get_landmark(landmarks_3d, mapping[1])
                            if p_start is None or p_end is None:
                                continue
                            target_dir = (armature_inv_33 @ (p_end - p_start)).normalized()
                            parent_ovr = prev_expected if prev_expected is not None else 'auto'
                            bone.rotation_mode = 'QUATERNION'
                            rot = compute_fk_rotation(bone, target_dir, parent_ovr)
                            bone.rotation_quaternion = rot
                            bone.keyframe_insert(data_path="rotation_quaternion")
                            if bone.parent:
                                p_mat = prev_expected if prev_expected is not None else bone.parent.matrix
                                rest_off = bone.parent.bone.matrix_local.inverted() @ bone.bone.matrix_local
                            else:
                                p_mat = Matrix.Identity(4)
                                rest_off = bone.bone.matrix_local
                            prev_expected = p_mat @ rest_off @ rot.to_matrix().to_4x4()
                else:
                    # Pure FK: process all limb bones (legs + arms)
                    # Process in parent→child order within each chain
                    fk_chain_order = [
                        "upper_arm_fk.L", "forearm_fk.L",
                        "upper_arm_fk.R", "forearm_fk.R",
                        "thigh_fk.L", "shin_fk.L", "foot_fk.L",
                        "thigh_fk.R", "shin_fk.R", "foot_fk.R",
                    ]
                    prev_parent_matrix = {}
                    for bone_name in fk_chain_order:
                        mapping = V2R_MAPPING.get(bone_name)
                        if not mapping:
                            continue
                        bone = rig.pose.bones.get(bone_name)
                        if not bone:
                            continue
                        p_start = get_landmark(landmarks_3d, mapping[0])
                        p_end = get_landmark(landmarks_3d, mapping[1])
                        if p_start is None or p_end is None:
                            continue

                        target_dir = (armature_inv_33 @ (p_end - p_start)).normalized()
                        # Use manually tracked parent matrix if available,
                        # otherwise 'auto' (depsgraph was flushed above)
                        parent_override = prev_parent_matrix.get(bone_name, 'auto')
                        bone.rotation_mode = 'QUATERNION'
                        rot = compute_fk_rotation(bone, target_dir, parent_override)
                        bone.rotation_quaternion = rot
                        bone.keyframe_insert(data_path="rotation_quaternion")

                        # Track this bone's expected matrix for its children
                        if bone.parent:
                            p_mat = prev_parent_matrix.get(bone_name, bone.parent.matrix)
                            rest_off = bone.parent.bone.matrix_local.inverted() @ bone.bone.matrix_local
                        else:
                            p_mat = Matrix.Identity(4)
                            rest_off = bone.bone.matrix_local
                        expected = p_mat @ rest_off @ rot.to_matrix().to_4x4()

                        # Map children: forearm's parent_override key is forearm_fk.X
                        child_map = {
                            "upper_arm_fk.L": "forearm_fk.L",
                            "upper_arm_fk.R": "forearm_fk.R",
                            "thigh_fk.L": "shin_fk.L",
                            "thigh_fk.R": "shin_fk.R",
                            "shin_fk.L": "foot_fk.L",
                            "shin_fk.R": "foot_fk.R",
                        }
                        child = child_map.get(bone_name)
                        if child:
                            prev_parent_matrix[child] = expected

                # --- Fingers (wholebody data, uses simple FK) ---
                if finger_data_available and self.use_fingers:
                    for bone_name, (start_idx, end_idx) in FINGER_FK_MAPPING.items():
                        bone = rig.pose.bones.get(bone_name)
                        if not bone:
                            continue

                        p_start = get_landmark(landmarks_3d, start_idx)
                        p_end = get_landmark(landmarks_3d, end_idx)
                        if p_start is None or p_end is None:
                            continue

                        target_dir = (armature_inv_33 @ (p_end - p_start)).normalized()

                        bone.rotation_mode = 'QUATERNION'
                        bone.rotation_quaternion = compute_fk_rotation(bone, target_dir)
                        bone.keyframe_insert(data_path="rotation_quaternion")

            # =====================
            # IK POSITIONS (proportionally scaled)
            # =====================
            if self.use_ik and hip_center:
                # Evaluate rig to get actual shoulder positions after torso+spine FK.
                # This ensures arm IK targets are relative to where the rig's shoulder
                # ACTUALLY is, avoiding mismatch from spine curvature/height scaling.
                bpy.context.view_layer.update()

                # Read rig's evaluated shoulder world positions
                upper_arm_L = rig.pose.bones.get("upper_arm_fk.L")
                upper_arm_R = rig.pose.bones.get("upper_arm_fk.R")
                rig_shoulder_L = (rig.matrix_world @ upper_arm_L.matrix).to_translation() if upper_arm_L else None
                rig_shoulder_R = (rig.matrix_world @ upper_arm_R.matrix).to_translation() if upper_arm_R else None

                # Mocap shoulder positions for arm vector computation
                p_shoulder_l = get_landmark(landmarks_3d, LM.LEFT_SHOULDER)
                p_shoulder_r = get_landmark(landmarks_3d, LM.RIGHT_SHOULDER)

                # Arm diagnostics (log every Nth frame)
                if frame_idx % log_every == 0:
                    for side_label, rig_sh, mocap_sh in [
                        ("L", rig_shoulder_L, p_shoulder_l),
                        ("R", rig_shoulder_R, p_shoulder_r),
                    ]:
                        wrist_idx = LM.LEFT_WRIST if side_label == "L" else LM.RIGHT_WRIST
                        elbow_idx = LM.LEFT_ELBOW if side_label == "L" else LM.RIGHT_ELBOW
                        p_wr = get_landmark(landmarks_3d, wrist_idx)
                        p_el = get_landmark(landmarks_3d, elbow_idx)
                        arm_len = rig_props.get(f'arm.{side_label}', 0)
                        if rig_sh and mocap_sh and p_wr:
                            arm_vec = p_wr - mocap_sh
                            arm_scale = scales.get(f'arm.{side_label}', global_scale)
                            target = rig_sh + arm_vec * arm_scale
                            dist = (target - rig_sh).length
                            ratio = dist / arm_len if arm_len > 0 else 0
                            # Log mocap raw positions for capture quality assessment
                            mocap_arm_raw = (p_wr - mocap_sh).length
                            DiagLog.data(f"  arm.{side_label}",
                                f"rig_sh=({rig_sh.x:.3f},{rig_sh.y:.3f},{rig_sh.z:.3f}) "
                                f"target_dist={dist:.4f} rig_arm_len={arm_len:.4f} "
                                f"ratio={ratio:.3f}" if arm_len > 0 else "no data")
                            DiagLog.data(f"  arm.{side_label}_mocap",
                                f"sh=({mocap_sh.x:.3f},{mocap_sh.y:.3f},{mocap_sh.z:.3f}) "
                                f"wr=({p_wr.x:.3f},{p_wr.y:.3f},{p_wr.z:.3f}) "
                                f"raw_dist={mocap_arm_raw:.4f}")
                            if p_el:
                                DiagLog.data(f"  arm.{side_label}_elbow",
                                    f"({p_el.x:.3f},{p_el.y:.3f},{p_el.z:.3f})")
                            # Warn on bad ratios
                            if arm_len > 0 and ratio < 0.7:
                                DiagLog.info(f"  !! ARM {side_label} RATIO {ratio:.3f} < 0.7 — "
                                    f"wrist target too close to shoulder. "
                                    f"Capture or retargeting error!")

                for bone_name, landmark_idx in IK_TARGETS.items():
                    # Sitting: skip foot IK targets (legs use FK)
                    if is_sitting and "foot" in bone_name:
                        continue

                    # Hybrid mode: snap hand_ik to wrist position (visual cleanup)
                    # The IK chain doesn't drive the mesh (IK_FK=1.0), but stale
                    # hand_ik positions look confusing in the viewport.
                    if hybrid_mode and "hand" in bone_name:
                        hik_bone = rig.pose.bones.get(bone_name)
                        if hik_bone:
                            side = "L" if ".L" in bone_name else "R"
                            fa_name = f"forearm_fk.{side}"
                            fa_bone = rig.pose.bones.get(fa_name)
                            if fa_bone:
                                # Snap hand_ik to forearm tail (wrist) position
                                # fa_bone.tail is in armature space after depsgraph update
                                wrist_armature = fa_bone.tail
                                wrist_world = rig.matrix_world @ Vector((wrist_armature.x, wrist_armature.y, wrist_armature.z))
                                set_bone_world_position(rig, hik_bone, wrist_world)
                                hik_bone.keyframe_insert(data_path="location")
                        continue

                    bone = rig.pose.bones.get(bone_name)
                    if not bone:
                        continue

                    pos = get_landmark(landmarks_3d, landmark_idx)
                    if pos is None:
                        continue

                    chain_key = IK_SCALE_KEY.get(bone_name, 'global')
                    chain_scale = scales.get(chain_key, global_scale)

                    if "hand" in bone_name:
                        # Arms: compute wrist target relative to the rig's ACTUAL
                        # evaluated shoulder position.
                        side = "L" if ".L" in bone_name else "R"
                        rig_sh = rig_shoulder_L if side == "L" else rig_shoulder_R
                        mocap_sh = p_shoulder_l if side == "L" else p_shoulder_r
                        if rig_sh and mocap_sh:
                            arm_vec = pos - mocap_sh
                            pos_scaled = rig_sh + arm_vec * chain_scale

                            # Arm velocity clamp: reject frame-to-frame spikes
                            # (e.g. triangulation glitches where arm teleports)
                            dt = timestamp - prev_timestamp if prev_timestamp > 0 else 0.033
                            if prev_hand_pos[side] is not None and dt > 0:
                                hand_dist = (pos_scaled - prev_hand_pos[side]).length
                                hand_speed = hand_dist / dt
                                if hand_speed > ARM_MAX_SPEED:
                                    # Reject: keep previous position
                                    pos_scaled = prev_hand_pos[side].copy()
                                    if frame_idx % log_every == 0:
                                        DiagLog.info(f"  !! ARM {side} velocity clamp f{frame_idx}: {hand_speed:.1f} m/s > {ARM_MAX_SPEED}")

                            # Arm ratio minimum clamp: prevent extreme chicken-wing
                            # When wrist is too close to shoulder (bent arms while
                            # sitting), the IK solver bends the elbow sideways.
                            # Clamp outward along the arm direction to keep elbow
                            # in a natural forward-bent pose.
                            arm_len = rig_props.get(f'arm.{side}', 0)
                            if arm_len > 0:
                                target_dist = (pos_scaled - rig_sh).length
                                ratio = target_dist / arm_len
                                if ratio < ARM_MIN_RATIO:
                                    # Push target outward along arm direction
                                    arm_dir = (pos_scaled - rig_sh)
                                    if arm_dir.length > 0.001:
                                        arm_dir = arm_dir.normalized()
                                    else:
                                        arm_dir = Vector((0, -1, 0))  # fallback: forward
                                    pos_scaled = rig_sh + arm_dir * (arm_len * ARM_MIN_RATIO)
                                    ratio = ARM_MIN_RATIO

                                arm_ratios[side].append({
                                    'frame': frame_idx,
                                    'ratio': ratio,
                                    'mocap_dist': arm_vec.length,
                                })

                            prev_hand_pos[side] = pos_scaled.copy()
                        else:
                            pos = scale_position(pos, hip_center, chain_scale)
                            pos_scaled = pos.copy()
                            pos_scaled.z += (hip_height_scale - 1.0) * hip_center.z
                            mocap_hip_frame0 = mocap_props.get('hip_pos')
                            if mocap_hip_frame0:
                                pos_scaled.x += (global_scale - 1.0) * mocap_hip_frame0.x
                                pos_scaled.y += (global_scale - 1.0) * mocap_hip_frame0.y
                    else:
                        # Legs: scale from hip_center (correct for leg reach)
                        pos = scale_position(pos, hip_center, chain_scale)

                        # Height/XY offset for leg IK targets
                        pos_scaled = pos.copy()
                        pos_scaled.z += (hip_height_scale - 1.0) * hip_center.z
                        mocap_hip_frame0 = mocap_props.get('hip_pos')
                        if mocap_hip_frame0:
                            pos_scaled.x += (global_scale - 1.0) * mocap_hip_frame0.x
                            pos_scaled.y += (global_scale - 1.0) * mocap_hip_frame0.y

                    is_foot = "foot" in bone_name
                    side = "L" if ".L" in bone_name else "R"

                    if is_foot:
                        # Apply ankle-to-ground offset so feet naturally sit near floor.
                        # The mocap ankle keypoint is ~5-7cm above ground anatomically.
                        # Without this, unpinned feet visibly float.
                        pos_scaled.z -= foot_z_offset[side]

                        # Capture pre-processing position for diagnostics
                        raw_z = pos_scaled.z
                        raw_pos = pos_scaled.copy()

                        # Ground clamp: never below floor
                        if self.ground_clamp and pos_scaled.z < 0:
                            pos_scaled.z = 0

                        # Foot pinning with smooth blend transitions.
                        # Both pin-IN and unpin are blended over multiple frames
                        # to prevent visible snapping.
                        # Walking-aware: force-unpin when hip drifts too far from
                        # where the foot was pinned (person walked away).
                        foot_speed = 0.0
                        foot_pinned = False
                        if self.pin_threshold > 0 and not is_sitting:
                            near_floor = pos_scaled.z < self.foot_floor_height
                            dt = timestamp - prev_timestamp if prev_timestamp > 0 else 0.033

                            # Check hip drift: if hip moved far from pin point, slide pin
                            hip_drift_unpin = False
                            hip_drift_slide = False
                            if pinned_foot_pos[side] is not None and hip_center and pin_hip_pos[side] is not None:
                                hip_dx = hip_center.x - pin_hip_pos[side].x
                                hip_dy = hip_center.y - pin_hip_pos[side].y
                                hip_drift = math.sqrt(hip_dx * hip_dx + hip_dy * hip_dy)
                                if hip_drift > HIP_DRIFT_UNPIN and pin_frame_count[side] >= MIN_PIN_FRAMES:
                                    # Instead of force-unpin, slide pin XY toward current foot
                                    # This keeps foot on ground but allows it to "step"
                                    hip_drift_slide = True

                            if prev_foot_raw[side] is not None and dt > 0:
                                dist = (pos_scaled - prev_foot_raw[side]).length
                                foot_speed = dist / dt

                                want_pin = (foot_speed < self.pin_threshold and near_floor)

                                if want_pin:
                                    # PINNING — foot should be on ground
                                    pin_frame_count[side] += 1
                                    if unpin_blend[side] > 0:
                                        # Mid-unpin blend — let it finish first
                                        unpin_blend[side] += 1.0 / UNPIN_BLEND_FRAMES
                                        if unpin_blend[side] < 1.0:
                                            t = unpin_blend[side]
                                            t = t * t * (3 - 2 * t)
                                            pin_pos = pinned_foot_pos[side]
                                            pos_scaled = pin_pos + (pos_scaled - pin_pos) * t
                                            foot_pinned = True
                                        else:
                                            # Unpin blend done, now start pin blend
                                            pinned_foot_pos[side] = pos_scaled.copy()
                                            pinned_foot_pos[side].z = 0
                                            pin_hip_pos[side] = hip_center.copy() if hip_center else None
                                            pin_frame_count[side] = 0
                                            unpin_blend[side] = 0.0
                                            pin_blend[side] = 1.0  # already at target
                                            pos_scaled = pinned_foot_pos[side].copy()
                                            foot_pinned = True
                                    elif pinned_foot_pos[side] is None:
                                        # New pin — start pin-IN blend
                                        pinned_foot_pos[side] = pos_scaled.copy()
                                        pinned_foot_pos[side].z = 0
                                        pin_hip_pos[side] = hip_center.copy() if hip_center else None
                                        pin_frame_count[side] = 0
                                        pin_blend[side] += 1.0 / PIN_BLEND_FRAMES
                                        if pin_blend[side] < 1.0:
                                            t = pin_blend[side]
                                            t = t * t * (3 - 2 * t)
                                            pin_pos = pinned_foot_pos[side]
                                            pos_scaled = raw_pos + (pin_pos - raw_pos) * t
                                            foot_pinned = True
                                        else:
                                            pin_blend[side] = 1.0
                                            pos_scaled = pinned_foot_pos[side].copy()
                                            foot_pinned = True
                                    elif pin_blend[side] < 1.0:
                                        # Continue pin-IN blend
                                        pin_blend[side] += 1.0 / PIN_BLEND_FRAMES
                                        if pin_blend[side] < 1.0:
                                            t = pin_blend[side]
                                            t = t * t * (3 - 2 * t)
                                            pin_pos = pinned_foot_pos[side]
                                            pos_scaled = raw_pos + (pin_pos - raw_pos) * t
                                            foot_pinned = True
                                        else:
                                            pin_blend[side] = 1.0
                                            pos_scaled = pinned_foot_pos[side].copy()
                                            foot_pinned = True
                                    else:
                                        # Fully pinned — slide XY if hip drifted (walking)
                                        if hip_drift_slide:
                                            # Slide pin toward current foot XY position
                                            pinned_foot_pos[side].x += (pos_scaled.x - pinned_foot_pos[side].x) * PIN_SLIDE_RATE
                                            pinned_foot_pos[side].y += (pos_scaled.y - pinned_foot_pos[side].y) * PIN_SLIDE_RATE
                                            pin_hip_pos[side] = hip_center.copy() if hip_center else None
                                        pos_scaled = pinned_foot_pos[side].copy()
                                        foot_pinned = True
                                else:
                                    # NOT PINNING — blend out if we were pinned
                                    pin_blend[side] = 0.0
                                    if pinned_foot_pos[side] is not None:
                                        unpin_blend[side] += 1.0 / UNPIN_BLEND_FRAMES
                                        if unpin_blend[side] < 1.0:
                                            t = unpin_blend[side]
                                            t = t * t * (3 - 2 * t)  # smoothstep
                                            pin_pos = pinned_foot_pos[side]
                                            pos_scaled = pin_pos + (pos_scaled - pin_pos) * t
                                            foot_pinned = True  # still blending
                                        else:
                                            pinned_foot_pos[side] = None
                                            pin_hip_pos[side] = None
                                            pin_frame_count[side] = 0
                                            unpin_blend[side] = 0.0

                            prev_foot_raw[side] = raw_pos.copy()

                        # Collect diagnostic data
                        foot_diag[side].append({
                            'frame': frame_idx,
                            'raw_z': raw_z,
                            'final_z': pos_scaled.z,
                            'raw_x': raw_pos.x, 'raw_y': raw_pos.y,
                            'final_x': pos_scaled.x, 'final_y': pos_scaled.y,
                            'speed': foot_speed,
                            'pinned': foot_pinned,
                            'clamped': raw_z < 0,
                            'blend_t': unpin_blend[side],
                        })

                    set_bone_world_position(rig, bone, pos_scaled)
                    bone.keyframe_insert(data_path="location")

                    # Hand rotation: point hand along forearm direction.
                    # Same approach as FK rotation — use compute_fk_rotation
                    # to orient the bone's Y axis along elbow→wrist.
                    if self.hand_rotation and "hand" in bone_name and not finger_data_available:
                        h_side = "L" if ".L" in bone_name else "R"
                        elbow_idx = LM.LEFT_ELBOW if h_side == "L" else LM.RIGHT_ELBOW
                        wrist_idx = LM.LEFT_WRIST if h_side == "L" else LM.RIGHT_WRIST
                        p_elbow = get_landmark(landmarks_3d, elbow_idx)
                        p_wrist = get_landmark(landmarks_3d, wrist_idx)

                        if p_elbow and p_wrist:
                            forearm_dir = (armature_inv_33 @ (p_wrist - p_elbow)).normalized()
                            bone.rotation_mode = 'QUATERNION'
                            bone.rotation_quaternion = compute_fk_rotation(bone, forearm_dir)
                            bone.keyframe_insert(data_path="rotation_quaternion")

                # Pole targets (also scaled)
                for bone_name, (root_idx, mid_idx, end_idx) in POLE_TARGETS.items():
                    # Hybrid mode: skip arm pole targets (arms use FK)
                    if hybrid_mode and 'arm' in bone_name:
                        continue
                    # Sitting: skip leg pole targets (legs use FK)
                    if is_sitting and 'thigh' in bone_name:
                        continue

                    bone = rig.pose.bones.get(bone_name)
                    if not bone:
                        continue

                    p_root = get_landmark(landmarks_3d, root_idx)
                    p_mid = get_landmark(landmarks_3d, mid_idx)
                    p_end = get_landmark(landmarks_3d, end_idx)

                    if None in (p_root, p_mid, p_end):
                        continue

                    is_arm = 'arm' in bone_name
                    chain_key = 'arm.L' if is_arm and '.L' in bone_name else \
                                'arm.R' if is_arm and '.R' in bone_name else \
                                'leg.L' if 'thigh' in bone_name and '.L' in bone_name else \
                                'leg.R' if 'thigh' in bone_name and '.R' in bone_name else 'global'
                    cs = scales.get(chain_key, global_scale)

                    if is_arm:
                        # Arm poles: use rig's actual shoulder position (same as hand IK).
                        # Scale arm joint positions (elbow, wrist) from mocap shoulder
                        # by arm_scale, then offset to rig's evaluated shoulder.
                        side = "L" if ".L" in bone_name else "R"
                        rig_sh = rig_shoulder_L if side == "L" else rig_shoulder_R
                        if rig_sh:
                            p_root_s = rig_sh
                            p_mid_s = rig_sh + (p_mid - p_root) * cs
                            p_end_s = rig_sh + (p_end - p_root) * cs
                        else:
                            p_root_s = scale_position(p_root, hip_center, global_scale)
                            p_mid_s = p_root_s + (p_mid - p_root) * cs
                            p_end_s = p_root_s + (p_end - p_root) * cs
                    else:
                        p_root_s = scale_position(p_root, hip_center, cs)
                        p_mid_s = scale_position(p_mid, hip_center, cs)
                        p_end_s = scale_position(p_end, hip_center, cs)

                    # Increase pole offset for bent arms — when ratio is low
                    # the elbow is close to the shoulder-wrist line and a small
                    # offset lets the IK solver pick a random bend direction.
                    # Larger offset forces the elbow forward more aggressively.
                    if is_arm:
                        arm_side = "L" if ".L" in bone_name else "R"
                        a_len = rig_props.get(f'arm.{arm_side}', 0)
                        a_sh = rig_shoulder_L if arm_side == "L" else rig_shoulder_R
                        a_hand = prev_hand_pos.get(arm_side)
                        if a_len > 0 and a_sh and a_hand:
                            a_ratio = (a_hand - a_sh).length / a_len
                            # Scale offset: 0.3 at ratio=1.0, up to 0.6 at ratio=0.55
                            pole_offset = 0.3 + max(0, (1.0 - a_ratio)) * 0.7
                        else:
                            pole_offset = 0.3
                    else:
                        pole_offset = 0.3

                    pole_pos = compute_pole_position(p_root_s, p_mid_s, p_end_s, offset=pole_offset)

                    if not (is_arm and (rig_shoulder_L or rig_shoulder_R)):
                        # Only apply height/XY offset for non-arm poles (legs).
                        # Arm poles already use the rig's evaluated position.
                        pole_pos.z += (hip_height_scale - 1.0) * hip_center.z
                        mocap_hip_frame0 = mocap_props.get('hip_pos')
                        if mocap_hip_frame0:
                            pole_pos.x += (global_scale - 1.0) * mocap_hip_frame0.x
                            pole_pos.y += (global_scale - 1.0) * mocap_hip_frame0.y

                    set_bone_world_position(rig, bone, pole_pos)
                    bone.keyframe_insert(data_path="location")

            prev_timestamp = timestamp

        # =====================
        # ARM RATIO DIAGNOSTICS
        # =====================
        DiagLog.section("ARM DIAGNOSTICS")
        if hybrid_mode:
            DiagLog.info("  Arms: FK mode (hybrid) — no IK ratio data")
            DiagLog.info("  Arm rotations computed directly from mocap shoulder→elbow→wrist")
        for side_label in ["L", "R"]:
            entries = arm_ratios[side_label]
            if not entries:
                if not hybrid_mode:
                    DiagLog.info(f"  arm.{side_label}: no data")
                continue
            ratios = [e['ratio'] for e in entries]
            r_min = min(ratios)
            r_max = max(ratios)
            r_avg = sum(ratios) / len(ratios)
            bad_frames = sum(1 for r in ratios if r < 0.7)
            clamped_frames = sum(1 for r in ratios if abs(r - ARM_MIN_RATIO) < 0.001)
            DiagLog.info(f"  arm.{side_label}:")
            DiagLog.data(f"    Ratio range", f"{r_min:.3f} to {r_max:.3f} (avg={r_avg:.3f})")
            DiagLog.data(f"    Frames <0.7 ratio", f"{bad_frames}/{len(entries)} ({100*bad_frames/len(entries):.0f}%)")
            DiagLog.data(f"    Frames clamped at {ARM_MIN_RATIO}", f"{clamped_frames}/{len(entries)} ({100*clamped_frames/len(entries):.0f}%)")
            if bad_frames > 0:
                first_bad = next(e for e in entries if e['ratio'] < 0.7)
                DiagLog.data(f"    First bad frame", f"{first_bad['frame']} (ratio={first_bad['ratio']:.3f})")
                worst = min(entries, key=lambda e: e['ratio'])
                DiagLog.data(f"    Worst frame", f"{worst['frame']} (ratio={worst['ratio']:.3f})")
            # Show ratio progression (sample 10 evenly spaced)
            sample_count = min(10, len(entries))
            step = max(1, len(entries) // sample_count)
            samples = [entries[i * step] for i in range(sample_count) if i * step < len(entries)]
            DiagLog.info(f"    --- Ratio timeline (sampled) ---")
            for s in samples:
                marker = " !!" if s['ratio'] < 0.7 else ""
                DiagLog.info(f"    f{s['frame']:3d}: ratio={s['ratio']:.3f} mocap_dist={s['mocap_dist']:.4f}{marker}")

        # =====================
        # FOOT DIAGNOSTICS
        # =====================
        DiagLog.section("FOOT DIAGNOSTICS")
        for side_label in ["L", "R"]:
            entries = foot_diag[side_label]
            if not entries:
                DiagLog.info(f"  foot.{side_label}: no data")
                continue

            z_vals = [e['raw_z'] for e in entries]
            speeds = [e['speed'] for e in entries if e['speed'] > 0]
            pinned_count = sum(1 for e in entries if e['pinned'])
            clamped_count = sum(1 for e in entries if e['clamped'])

            z_min = min(z_vals)
            z_max = max(z_vals)
            z_range = z_max - z_min

            # Frame-to-frame Z deltas
            z_deltas = [abs(z_vals[i] - z_vals[i-1]) for i in range(1, len(z_vals))]
            z_delta_avg = sum(z_deltas) / len(z_deltas) if z_deltas else 0
            z_delta_max = max(z_deltas) if z_deltas else 0

            # XY deltas (horizontal jitter)
            x_vals = [e['raw_x'] for e in entries]
            y_vals = [e['raw_y'] for e in entries]
            xy_deltas = []
            for i in range(1, len(entries)):
                dx = x_vals[i] - x_vals[i-1]
                dy = y_vals[i] - y_vals[i-1]
                xy_deltas.append(math.sqrt(dx*dx + dy*dy))
            xy_delta_avg = sum(xy_deltas) / len(xy_deltas) if xy_deltas else 0
            xy_delta_max = max(xy_deltas) if xy_deltas else 0

            speed_avg = sum(speeds) / len(speeds) if speeds else 0
            speed_max = max(speeds) if speeds else 0

            # Count "jitter frames" — Z delta > 1cm when speed < 0.3 m/s
            jitter_frames = 0
            for i in range(1, len(entries)):
                if entries[i]['speed'] < 0.3 and abs(z_vals[i] - z_vals[i-1]) > 0.01:
                    jitter_frames += 1

            DiagLog.info(f"  foot_ik.{side_label}:")
            DiagLog.data(f"    Z range", f"{z_min:.4f} to {z_max:.4f} (span: {z_range:.4f}m)")
            DiagLog.data(f"    Z delta avg/max", f"{z_delta_avg:.4f}m / {z_delta_max:.4f}m")
            DiagLog.data(f"    XY delta avg/max", f"{xy_delta_avg:.4f}m / {xy_delta_max:.4f}m")
            DiagLog.data(f"    Speed avg/max", f"{speed_avg:.3f} / {speed_max:.3f} m/s")
            DiagLog.data(f"    Pinned frames", f"{pinned_count}/{len(entries)}")
            DiagLog.data(f"    Ground-clamped", f"{clamped_count}/{len(entries)}")
            DiagLog.data(f"    Jitter frames", f"{jitter_frames}/{len(entries)} (Z>1cm while slow)")

            # Print first 20 frames of raw data for detailed analysis
            DiagLog.info(f"    --- First 20 frames detail ---")
            for e in entries[:20]:
                pin_str = " PINNED" if e['pinned'] else ""
                clamp_str = " CLAMPED" if e['clamped'] else ""
                DiagLog.info(f"    f{e['frame']:3d}: Z={e['raw_z']:+.4f} finalZ={e['final_z']:+.4f} "
                             f"spd={e['speed']:.3f}{pin_str}{clamp_str}")

            # Print worst jitter frames
            worst_z = sorted(range(1, len(entries)),
                             key=lambda i: abs(z_vals[i] - z_vals[i-1]),
                             reverse=True)[:5]
            if worst_z:
                DiagLog.info(f"    --- Worst Z-jitter frames ---")
                for i in worst_z:
                    e = entries[i]
                    delta = z_vals[i] - z_vals[i-1]
                    DiagLog.info(f"    f{e['frame']:3d}: deltaZ={delta:+.4f} Z={e['raw_z']:+.4f} "
                                 f"spd={e['speed']:.3f}")

            # Print pin/unpin transitions (where pinned state changes)
            transitions = []
            for i in range(1, len(entries)):
                prev_pin = entries[i-1]['pinned']
                cur_pin = entries[i]['pinned']
                blend = entries[i].get('blend_t', 0)
                if prev_pin != cur_pin or (0 < blend < 1):
                    transitions.append(i)
            if transitions:
                DiagLog.info(f"    --- Pin/Unpin transitions ({len(transitions)} frames) ---")
                for i in transitions[:30]:  # cap output
                    e = entries[i]
                    prev_e = entries[i-1]
                    state = "PINNED" if e['pinned'] else "FREE"
                    prev_state = "PINNED" if prev_e['pinned'] else "FREE"
                    blend_t = e.get('blend_t', 0)
                    # Show position jump
                    dx = e['final_x'] - prev_e['final_x']
                    dy = e['final_y'] - prev_e['final_y']
                    dz = e['final_z'] - prev_e['final_z']
                    jump = math.sqrt(dx*dx + dy*dy + dz*dz)
                    DiagLog.info(
                        f"    f{e['frame']:3d}: {prev_state}->{state} "
                        f"blend={blend_t:.2f} spd={e['speed']:.3f} "
                        f"rawZ={e['raw_z']:+.4f} finalZ={e['final_z']:+.4f} "
                        f"jump={jump:.4f}m"
                    )

        DiagLog.section("IMPORT COMPLETE")
        DiagLog.data("Total frames processed", len(frames))
        DiagLog.data("Blender frame range", f"0 - {int(frames[-1].get('timestamp', 0) * fps)}")

        self.report({'INFO'},
                    f"Imported {len(frames)} frames from {os.path.basename(self.filepath)} "
                    f"({format_name}, scale={global_scale:.2f}x)")
        return {'FINISHED'}


class MELODICCAP_OT_clear_animation(bpy.types.Operator):
    """Clear all animation from the active armature"""
    bl_idname = "melodiccap.clear_animation"
    bl_label = "Clear Animation"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rig = context.active_object
        if not rig or rig.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature")
            return {'CANCELLED'}

        if rig.animation_data and rig.animation_data.action:
            bpy.data.actions.remove(rig.animation_data.action)

        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.pose.transforms_clear()

        self.report({'INFO'}, "Animation cleared")
        return {'FINISHED'}


class MELODICCAP_OT_set_fk_mode(bpy.types.Operator):
    """Set all limbs to FK mode"""
    bl_idname = "melodiccap.set_fk_mode"
    bl_label = "Set FK Mode"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rig = context.active_object
        if not rig or rig.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature")
            return {'CANCELLED'}

        for bone in rig.pose.bones:
            if "IK_FK" in bone.keys():
                bone["IK_FK"] = 1.0

        self.report({'INFO'}, "Set to FK mode")
        return {'FINISHED'}


class MELODICCAP_OT_set_ik_mode(bpy.types.Operator):
    """Set all limbs to IK mode"""
    bl_idname = "melodiccap.set_ik_mode"
    bl_label = "Set IK Mode"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rig = context.active_object
        if not rig or rig.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature")
            return {'CANCELLED'}

        for bone in rig.pose.bones:
            if "IK_FK" in bone.keys():
                bone["IK_FK"] = 0.0

        self.report({'INFO'}, "Set to IK mode")
        return {'FINISHED'}


# =============================================================================
# PANEL
# =============================================================================

class MELODICCAP_PT_main_panel(bpy.types.Panel):
    bl_label = "MelodicCap RTM"
    bl_idname = "MELODICCAP_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'MelodicCap'

    def draw(self, context):
        layout = self.layout

        # Import section
        box = layout.box()
        box.label(text="Import", icon='IMPORT')
        box.operator("melodiccap.import_json", text="Import JSON Take")
        box.label(text="Supports RTM + legacy MediaPipe formats", icon='INFO')

        # Mode section
        box = layout.box()
        box.label(text="IK/FK Mode", icon='CON_KINEMATIC')
        row = box.row(align=True)
        row.operator("melodiccap.set_ik_mode", text="IK")
        row.operator("melodiccap.set_fk_mode", text="FK")

        # Utilities section
        box = layout.box()
        box.label(text="Utilities", icon='TOOL_SETTINGS')
        box.operator("melodiccap.clear_animation", text="Clear Animation", icon='X')

        # Info section
        if context.active_object and context.active_object.type == 'ARMATURE':
            box = layout.box()
            box.label(text="Active Rig:", icon='ARMATURE_DATA')
            box.label(text=f"  {context.active_object.name}")

            rig = context.active_object
            arm_parent = rig.pose.bones.get("upper_arm_parent.L")
            if arm_parent and "IK_FK" in arm_parent.keys():
                val = arm_parent["IK_FK"]
                mode = "FK" if val > 0.5 else "IK"
                box.label(text=f"  Limbs: {mode} mode")


# =============================================================================
# REGISTRATION
# =============================================================================

classes = [
    MELODICCAP_OT_import_json,
    MELODICCAP_OT_clear_animation,
    MELODICCAP_OT_set_fk_mode,
    MELODICCAP_OT_set_ik_mode,
    MELODICCAP_PT_main_panel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
