"""
MelodicCap RTM Blender Addon v5.12
====================================
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
    "version": (5, 14),
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
    Returns a dict with 'critical' (hard-block) and 'warnings' (soft-info) lists.

    HARD-BLOCK policy (v5.7): critical failures stop the import rather than
    being rescued silently. A bad A-pose poisons proportions, rest pitch/yaw,
    hip Z baseline, and foot Z offset — papering over it downstream is worse
    than forcing a re-record.
    """
    quality = {'critical': [], 'warnings': [], 'ok': True}

    # Arm symmetry: >10% is triangulation asymmetry (HARD BLOCK),
    # 3-10% is suspect but may be legitimate handedness.
    arm_l = mocap_props.get('arm.L', 0)
    arm_r = mocap_props.get('arm.R', 0)
    if arm_l > 0.01 and arm_r > 0.01:
        arm_asym = abs(arm_l - arm_r) / max(arm_l, arm_r) * 100
        quality['arm_asymmetry_pct'] = arm_asym
        if arm_asym > 10.0:
            quality['critical'].append(
                f"ARM ASYMMETRY {arm_asym:.1f}% (L={arm_l:.3f} R={arm_r:.3f}) — "
                f"frame 0 triangulation is broken on one side. Re-record with a "
                f"cleaner A-pose and check camera coverage.")
        elif arm_asym > 3.0:
            quality['warnings'].append(
                f"arm asymmetry {arm_asym:.1f}% (L={arm_l:.3f} R={arm_r:.3f})")

    # Leg symmetry: same treatment
    leg_l = mocap_props.get('leg.L', 0)
    leg_r = mocap_props.get('leg.R', 0)
    if leg_l > 0.01 and leg_r > 0.01:
        leg_asym = abs(leg_l - leg_r) / max(leg_l, leg_r) * 100
        quality['leg_asymmetry_pct'] = leg_asym
        if leg_asym > 10.0:
            quality['critical'].append(
                f"LEG ASYMMETRY {leg_asym:.1f}% (L={leg_l:.3f} R={leg_r:.3f}) — "
                f"triangulation imbalance across sides.")
        elif leg_asym > 3.0:
            quality['warnings'].append(
                f"leg asymmetry {leg_asym:.1f}% (L={leg_l:.3f} R={leg_r:.3f})")

    # Spine uprightness: >25° is almost certainly not standing (HARD BLOCK)
    hip_mid = compute_midpoint(landmarks_3d, LM.LEFT_HIP, LM.RIGHT_HIP)
    shoulder_mid = compute_midpoint(landmarks_3d, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)
    if hip_mid and shoulder_mid:
        spine_vec = shoulder_mid - hip_mid
        if spine_vec.length > 0.01:
            spine_up = spine_vec.normalized()
            spine_tilt = math.degrees(math.acos(max(-1, min(1, spine_up.z))))
            quality['spine_tilt_deg'] = spine_tilt
            if spine_tilt > 25:
                quality['critical'].append(
                    f"SPINE TILT {spine_tilt:.1f}° — calibration frame is not a "
                    f"standing pose. Re-record with a clean A-pose.")
            elif spine_tilt > 15:
                quality['warnings'].append(f"spine tilt {spine_tilt:.1f}°")

    # Hip height: standing hip should be 0.8-1.1m
    hip_pos = mocap_props.get('hip_pos')
    if hip_pos:
        quality['hip_height'] = hip_pos.z
        if hip_pos.z < 0.6:
            quality['critical'].append(
                f"HIP Z={hip_pos.z:.3f}m — person is sitting/crouching in the "
                f"calibration frame. Re-record with A-pose at the start.")
        elif hip_pos.z > 1.2:
            quality['warnings'].append(f"hip Z={hip_pos.z:.3f}m unusually high")

    # Shoulder width sanity
    p_ls = get_landmark(landmarks_3d, LM.LEFT_SHOULDER)
    p_rs = get_landmark(landmarks_3d, LM.RIGHT_SHOULDER)
    if p_ls and p_rs:
        shoulder_width = (p_ls - p_rs).length
        quality['shoulder_width'] = shoulder_width
        if shoulder_width < 0.2 or shoulder_width > 0.6:
            quality['warnings'].append(f"shoulder width {shoulder_width:.3f}m")

    # Wrist height relative to hip — informational only (not blocking since
    # arm position varies with A-pose style)
    for side, wrist_idx in [("L", LM.LEFT_WRIST), ("R", LM.RIGHT_WRIST)]:
        p_wr = get_landmark(landmarks_3d, wrist_idx)
        if p_wr and hip_pos:
            wrist_hip_dz = p_wr.z - hip_pos.z
            quality[f'wrist_{side}_dz'] = wrist_hip_dz
            if abs(wrist_hip_dz) > 0.25:
                quality['warnings'].append(
                    f"wrist {side} dZ={wrist_hip_dz:+.3f}m (arms far from hip height)")

    quality['ok'] = len(quality['critical']) == 0
    return quality


def check_lr_bone_symmetry(frames, bad_frame_fraction_max=0.05,
                            length_ratio_max=1.15):
    """
    Scan triangulated landmarks across the whole take and flag runs where
    L/R pair lengths diverge. Returns (ok, stats_dict).

    The solver enforces calibrated bone lengths on its output, but the
    DIAGNOSTIC value of this check is catching takes where the RAW input
    was already broken on more than `bad_frame_fraction_max` of frames
    (L/R bias exceeding `length_ratio_max`). A bad majority means either
    the wrong calibration pair, or a camera is misaligned.
    """
    pairs = [
        ('upper_arm', LM.LEFT_SHOULDER, LM.LEFT_ELBOW,
                       LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW),
        ('forearm',   LM.LEFT_ELBOW, LM.LEFT_WRIST,
                       LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
        ('thigh',     LM.LEFT_HIP, LM.LEFT_KNEE,
                       LM.RIGHT_HIP, LM.RIGHT_KNEE),
        ('shin',      LM.LEFT_KNEE, LM.LEFT_ANKLE,
                       LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
    ]
    stats = {name: {'bad': 0, 'total': 0, 'max_ratio': 1.0} for name, *_ in pairs}

    for f in frames:
        lm = f.get('landmarks_3d', {})
        if not lm:
            continue
        for name, lp, lc, rp, rc in pairs:
            p_lp = get_landmark(lm, lp)
            p_lc = get_landmark(lm, lc)
            p_rp = get_landmark(lm, rp)
            p_rc = get_landmark(lm, rc)
            if not (p_lp and p_lc and p_rp and p_rc):
                continue
            ll = (p_lc - p_lp).length
            lr = (p_rc - p_rp).length
            if ll < 0.03 or lr < 0.03:
                continue
            stats[name]['total'] += 1
            ratio = max(ll, lr) / min(ll, lr)
            if ratio > stats[name]['max_ratio']:
                stats[name]['max_ratio'] = ratio
            if ratio > length_ratio_max:
                stats[name]['bad'] += 1

    problems = []
    for name, s in stats.items():
        if s['total'] == 0:
            continue
        pct_bad = s['bad'] / s['total']
        s['bad_pct'] = pct_bad * 100
        if pct_bad > bad_frame_fraction_max:
            problems.append(
                f"{name} L/R diverges on {pct_bad*100:.1f}% of frames "
                f"(max ratio {s['max_ratio']:.2f}x)"
            )

    return (len(problems) == 0, stats, problems)


def detect_monocular_data(frames, sample_count=10):
    """
    Detect MediaPipe-single-camera data that can't be retargeted.
    Returns (is_monocular, reason) tuple.

    MediaPipe monocular outputs hip-centered 3D — hip never translates.
    RTM stereo has hip_raw varying across frames in world space.
    """
    if len(frames) < sample_count:
        return False, ""

    hip_xs = []
    hip_ys = []
    hip_zs = []
    step = max(1, len(frames) // sample_count)
    for i in range(0, len(frames), step):
        if len(hip_xs) >= sample_count:
            break
        lm = frames[i].get('landmarks_3d', {})
        hip_mid = compute_midpoint(lm, LM.LEFT_HIP, LM.RIGHT_HIP)
        if hip_mid:
            hip_xs.append(hip_mid.x)
            hip_ys.append(hip_mid.y)
            hip_zs.append(hip_mid.z)

    if len(hip_xs) < 3:
        return False, ""

    # If all hip coordinates are within 1mm of origin, it's hip-centered data.
    max_magnitude = max(max(abs(x) for x in hip_xs),
                        max(abs(y) for y in hip_ys),
                        max(abs(z) for z in hip_zs))
    if max_magnitude < 0.001:
        return True, "hip stays at origin (MediaPipe single-camera format)"

    # If hip XYZ variance is essentially zero, same thing even if not at origin.
    x_range = max(hip_xs) - min(hip_xs)
    y_range = max(hip_ys) - min(hip_ys)
    z_range = max(hip_zs) - min(hip_zs)
    total_range = x_range + y_range + z_range
    if total_range < 0.003:
        return True, f"hip total XYZ range {total_range*1000:.1f}mm across sampled frames — not stereo data"

    return False, ""


def scale_position(pos, hip_center_mocap, scale_factor):
    """Scale a position relative to the mocap hip center."""
    offset = pos - hip_center_mocap
    return hip_center_mocap + offset * scale_factor


# =============================================================================
# TEMPORAL SMOOTHING
# =============================================================================

def smooth_frames(frames, window=5):
    """Apply centered moving average to landmark positions.

    v5.11: per-key count fix. Pre-v5.11 the divisor was the number of
    frames in the window that had ANY landmarks_3d data — but the
    SUM for a specific key was over only the frames where THAT key
    was present. When triangulate_pose drops a keypoint for low
    confidence (e.g. wrist briefly occluded), that key would be
    missing in some frames and the resulting average would divide
    by too large a count, pulling the keypoint toward origin in
    proportion to how often it was missing. Result: a wrist dropped
    on 1 of 3 frames inflated the forearm length by ~33%, easily
    producing the 7.81x L/R ratio observed in the addon's L/R bone
    symmetry check despite solver output being clean. Fix is
    per-key count tracking.
    """
    if window < 2 or len(frames) < window:
        return frames

    half = window // 2
    smoothed = []

    for i in range(len(frames)):
        start = max(0, i - half)
        end = min(len(frames), i + half + 1)

        avg_landmarks = {}
        key_counts = {}
        for j in range(start, end):
            lm = frames[j].get('landmarks_3d', {})
            if not lm:
                continue
            for key, coords in lm.items():
                if key not in avg_landmarks:
                    avg_landmarks[key] = [0.0, 0.0, 0.0]
                    key_counts[key] = 0
                avg_landmarks[key][0] += coords[0]
                avg_landmarks[key][1] += coords[1]
                avg_landmarks[key][2] += coords[2]
                key_counts[key] += 1

        for key in list(avg_landmarks.keys()):
            c = key_counts[key]
            if c > 0:
                avg_landmarks[key] = [v / c for v in avg_landmarks[key]]
            else:
                del avg_landmarks[key]

        new_frame = dict(frames[i])
        new_frame['landmarks_3d'] = avg_landmarks
        smoothed.append(new_frame)

    return smoothed


# =============================================================================
# BUTTERWORTH LOW-PASS FILTER
# =============================================================================

def butterworth_filter_landmarks(frames, fps, cutoff_body=4.0, cutoff_feet=2.0):
    """
    Apply a 4th-order Butterworth low-pass filter to landmark positions (v4.7).

    Professional mocap systems (Vicon, OptiTrack) use 4th-order Butterworth
    at 6-10 Hz. We use 4th order via two cascaded biquad sections, with
    forward-backward filtering for zero phase lag.

    Feet (ankles/knees) get a lower cutoff to kill jitter since they need to
    be stable on the ground. Everything else gets a moderate cutoff.
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


def _butter_lowpass_filtfilt(data, cutoff, fs, order=4):
    """
    Zero-phase Butterworth filter implemented without scipy (v4.7).

    4th-order filter via two cascaded 2nd-order biquad sections,
    each with forward-backward passes for zero phase lag.
    This matches professional mocap pipeline standards (Vicon, Pose2Sim).
    """
    nyq = fs / 2.0
    wc = cutoff / nyq
    if wc >= 1.0:
        return data

    n = len(data)
    if n < 5:
        return list(data)

    # Bilinear pre-warp
    warp = math.tan(math.pi * wc / 2.0)
    k = warp * warp

    # 4th-order Butterworth = two 2nd-order sections with different Q factors.
    # Pole angles for 4th-order: pi*(2*m+1)/(2*N) for m=0,1
    # Section 1: Q1 = 1/(2*cos(pi*1/8)) ≈ 0.541
    # Section 2: Q2 = 1/(2*cos(pi*3/8)) ≈ 1.307
    sections = []
    for q_factor in [1.0 / (2.0 * math.cos(math.pi * 1 / 8)),
                     1.0 / (2.0 * math.cos(math.pi * 3 / 8))]:
        alpha = warp / q_factor
        norm = 1.0 / (1.0 + alpha + k)
        b0 = k * norm
        b1 = 2.0 * b0
        b2 = b0
        a1 = 2.0 * (k - 1.0) * norm
        a2 = (1.0 - alpha + k) * norm
        sections.append((b0, b1, b2, a1, a2))

    def _apply_biquad(signal, coeffs):
        b0, b1, b2, a1, a2 = coeffs
        out = list(signal)
        for i in range(2, len(out)):
            out[i] = (b0 * signal[i] + b1 * signal[i - 1] + b2 * signal[i - 2]
                       - a1 * out[i - 1] - a2 * out[i - 2])
        return out

    # Edge padding to reduce transients
    pad = min(12, n - 1)
    y = list(data)
    front_pad = [2.0 * y[0] - y[i] for i in range(pad, 0, -1)]
    back_pad = [2.0 * y[-1] - y[-(i + 2)] for i in range(pad)]
    padded = front_pad + y + back_pad

    # Apply each biquad section with forward-backward passes
    result = padded
    for coeffs in sections:
        # Forward
        fwd = _apply_biquad(result, coeffs)
        # Backward
        fwd.reverse()
        bwd = _apply_biquad(fwd, coeffs)
        bwd.reverse()
        result = bwd

    # Strip padding
    return result[pad:pad + n]


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


def compute_spine_fk_chain(rig, spine_points, armature_inv_33):
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
    permissive_import: bpy.props.BoolProperty(
        name="Permissive (skip L/R asymmetry block)",
        description="Bypass the L/R bone-length divergence hard block. "
                    "Use when 2-camera stereo produces intermittent depth-collapse "
                    "on one limb (e.g. forearm during reach) but the rest of the "
                    "take is usable. Logs as a warning instead. You'll likely "
                    "need to keyframe-clean the bad bone in Blender",
        default=False
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

        # HARD BLOCK v5.7: reject monocular (hip-centered) data upfront.
        # Retargeting hip-centered data produces a character that stretches in
        # place — there's no way to recover world-space hip translation.
        is_mono, mono_reason = detect_monocular_data(frames)
        if is_mono:
            msg = (f"Monocular/hip-centered data detected: {mono_reason}. "
                   f"Only RTM stereo takes can be retargeted.")
            DiagLog.info(f"[HARD BLOCK] {msg}")
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}

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
        # v5.12: detect solver-output JSONs. The skeleton solver does its
        # own temporal direction smoothing in _temporal_smooth_directions
        # and rebuilds the chain with calibrated bone lengths. Running
        # the addon's position-space smoothing on top compresses the
        # bone lengths whenever frame-to-frame forearm direction varies
        # (the "L=0.085 ≈ 0.242/3" pattern documented in
        # trace_forearm.py output). For solver-output, skip both the
        # 3-frame moving average and the Butterworth filter.
        solver_smoothed = bool(
            data.get('processing_settings', {}).get('skeleton_solver')
        )
        if solver_smoothed:
            DiagLog.info(
                "[INFO] Solver-smoothed take detected; skipping addon's "
                "moving-average and Butterworth filters (would compress "
                "bone lengths via position-averaging — see v5.12)."
            )
        elif self.smooth_window > 1:
            DiagLog.info(f"Applying {self.smooth_window}-frame moving average...")
            frames = smooth_frames(frames, self.smooth_window)

        # =====================
        # BUTTERWORTH FILTER
        # =====================
        if not solver_smoothed and self.butterworth:
            DiagLog.info(f"Applying Butterworth filter (body={self.butter_cutoff_body}Hz, feet={self.butter_cutoff_feet}Hz)...")
            frames = butterworth_filter_landmarks(
                frames, fps,
                cutoff_body=self.butter_cutoff_body,
                cutoff_feet=self.butter_cutoff_feet
            )

        # HARD BLOCK v5.7: L/R bone-length symmetry across the whole take.
        # The solver will enforce calibrated lengths downstream, but a take
        # where >5% of raw frames show >15% L/R divergence is a sign of
        # camera misalignment or using the wrong calibration pair, and the
        # solver can't rescue that.
        sym_ok, sym_stats, sym_problems = check_lr_bone_symmetry(frames)
        for bone_name, s in sym_stats.items():
            if s.get('total', 0):
                DiagLog.data(f"L/R {bone_name}",
                    f"max={s['max_ratio']:.2f}x bad={s.get('bad_pct', 0):.1f}%")
        if not sym_ok:
            if getattr(self, 'permissive_import', False):
                DiagLog.info("[WARNING] L/R bone lengths diverge "
                             "(permissive_import enabled — proceeding anyway):")
                for p in sym_problems:
                    DiagLog.info(f"    ! {p}")
                DiagLog.info("    Visually inspect; keyframe-clean the bad "
                             "bone in pose mode if motion is wrong.")
            else:
                DiagLog.info("[HARD BLOCK] L/R bone lengths diverge "
                             "systematically:")
                for p in sym_problems:
                    DiagLog.info(f"    ✗ {p}")
                DiagLog.info("    Re-run import with 'Permissive' option "
                             "checked to bypass this block.")
                msg = "L/R bone asymmetry: " + "; ".join(sym_problems)
                self.report({'ERROR'}, msg)
                return {'CANCELLED'}

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

        # Validate calibration frame pose quality (v5.7: HARD BLOCK on critical)
        DiagLog.section(f"CALIBRATION FRAME {cal_idx} POSE QUALITY CHECK")
        pose_quality = validate_frame0_pose(first_landmarks, mocap_props)
        if 'spine_tilt_deg' in pose_quality:
            DiagLog.data("Spine tilt from vertical", f"{pose_quality['spine_tilt_deg']:.1f}°")
        if 'arm_asymmetry_pct' in pose_quality:
            DiagLog.data("Arm length asymmetry", f"{pose_quality['arm_asymmetry_pct']:.1f}%")
        if 'leg_asymmetry_pct' in pose_quality:
            DiagLog.data("Leg length asymmetry", f"{pose_quality['leg_asymmetry_pct']:.1f}%")
        if 'shoulder_width' in pose_quality:
            DiagLog.data("Shoulder width", f"{pose_quality['shoulder_width']:.3f}m")
        for side in ['L', 'R']:
            if f'wrist_{side}_dz' in pose_quality:
                DiagLog.data(f"Wrist {side} vs hip dZ", f"{pose_quality[f'wrist_{side}_dz']:+.3f}m")

        for w in pose_quality['warnings']:
            DiagLog.info(f"  ⚠ {w}")

        if not pose_quality['ok']:
            DiagLog.info(f"[HARD BLOCK] Frame {cal_idx} A-pose quality check failed:")
            for c in pose_quality['critical']:
                DiagLog.info(f"    ✗ {c}")
            msg = (f"A-pose check failed ({len(pose_quality['critical'])} critical): "
                   + "; ".join(pose_quality['critical']))
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}
        DiagLog.info(f"Frame {cal_idx} pose: OK (clean A-pose)")

        # Hip height is determined by leg length, not spine length.
        # Use average leg scale for vertical positioning to prevent hunching.
        leg_scale_avg = (scales.get('leg.L', global_scale) + scales.get('leg.R', global_scale)) / 2.0
        hip_height_scale = leg_scale_avg
        DiagLog.data("Hip height scale (leg avg)", f"{hip_height_scale:.3f}")

        # v5.1: Delta-based hip Z — standard mocap retargeting approach
        # (Rokoko, Pose2Sim, FreeMoCap all anchor root to rest pose + scaled deltas)
        torso_bone = rig.pose.bones.get("torso")
        rig_rest_hip_z = (rig.matrix_world @ Vector(torso_bone.bone.head_local)).z
        # Precompute constant Z shift for limb IK targets:
        # shift = delta_hip_z - absolute_hip_z = rig_rest_hip_z - f0.z * hhs
        hip_z_shift = rig_rest_hip_z - mocap_props['hip_pos'].z * hip_height_scale
        DiagLog.data("Rig rest hip Z (world)", f"{rig_rest_hip_z:.4f}m")
        DiagLog.data("Hip Z shift (delta vs absolute)", f"{hip_z_shift:+.4f}m")

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
                    pos_z = pos.z + (hip_height_scale - 1.0) * hip_c.z + hip_z_shift
                    z_samples.append(pos_z)
            if z_samples:
                foot_z_offset[side_key] = sum(z_samples) / len(z_samples)
        DiagLog.data("Foot Z offset L", f"{foot_z_offset['L']:.4f}m")
        DiagLog.data("Foot Z offset R", f"{foot_z_offset['R']:.4f}m")

        # v5.8: A-pose hold quality — if the performer was already walking
        # during the foot_z_offset sample window, the averaged ankle Z bakes
        # in swing-phase ankle height (foot in air) and feet float for the
        # whole take. This is a soft warning, not a hard block — there's
        # legitimate small wobble in any A-pose.
        max_hip_xy_vel = 0.0
        prev_hip = None
        for i in range(OFFSET_SAMPLE_FRAMES):
            lm = frames[i].get('landmarks_3d', {})
            hip_c = compute_midpoint(lm, LM.LEFT_HIP, LM.RIGHT_HIP)
            if hip_c is not None:
                if prev_hip is not None:
                    dx = hip_c.x - prev_hip.x
                    dy = hip_c.y - prev_hip.y
                    vel = (dx * dx + dy * dy) ** 0.5
                    if vel > max_hip_xy_vel:
                        max_hip_xy_vel = vel
                prev_hip = hip_c
        DiagLog.data("A-pose hip max XY vel",
                     f"{max_hip_xy_vel*100:.1f}cm/frame "
                     f"(over first {OFFSET_SAMPLE_FRAMES} frames)")
        if max_hip_xy_vel > 0.05:
            DiagLog.info(
                f"[WARNING] A-pose calibration window contained motion: "
                f"max hip XY velocity {max_hip_xy_vel*100:.1f}cm/frame "
                f"(threshold 5cm/frame). Foot Z offsets may be biased by "
                f"swing-phase ankle height, causing feet to float during "
                f"the take. Hold A-pose still for the first ~2 seconds "
                f"of next recording.")

        # HARD BLOCK v5.7: foot Z offset asymmetry.
        # This was the #1 failure in take_20260420_230051 — L=0.67m vs R=0.19m.
        # Asymmetric per-side ankle Z baseline bakes triangulation bias into
        # every frame's pin/unpin decision; L foot pinned 0/293 while R pinned
        # 250/293 because the two feet oscillated around totally different
        # nominal Z. No downstream damper can fix this; re-record instead.
        foot_z_asym = abs(foot_z_offset['L'] - foot_z_offset['R'])
        DiagLog.data("Foot Z symmetry", f"|L-R|={foot_z_asym*100:.1f}cm")
        if foot_z_asym > 0.08:
            msg = (f"FOOT Z ASYMMETRY {foot_z_asym*100:.1f}cm "
                   f"(L={foot_z_offset['L']:.3f}m vs R={foot_z_offset['R']:.3f}m) — "
                   f"the two ankles triangulate to very different floor heights "
                   f"across the first {OFFSET_SAMPLE_FRAMES} frames. This is "
                   f"camera geometry asymmetry (one camera resolves one foot "
                   f"better than the other). Move cameras to a more symmetric "
                   f"position or re-record with performer further from cameras.")
            DiagLog.info(f"[HARD BLOCK] {msg}")
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}

        # Arm splay limit (v4.8): high fixed safety net.
        # v4.4's ARM_SPLAY_MAX=0.10 destroyed arm raises. v4.7 tried context-
        # aware clamp based on elbow height above shoulder — failed because
        # lateral arm raises keep elbows at/below shoulder height, so splay_lim
        # stayed at 0.15, still crushing X. Also caused seated chicken-wing
        # (elbows pushed inside torso).
        # The original chicken-wing was an IK solver artifact (v3.x), not a
        # triangulation noise issue. In FK mode, arm_fk_conf + velocity clamp
        # handle bad data. This is just a safety net for extreme artifacts.
        ARM_SPLAY_LIMIT = 0.80  # max |X| in armature space — allows full lateral raises

        # v4.8: Seated leg lateral damping.
        # Camera placement asymmetry causes systematic triangulation bias in
        # the lateral (X) direction of leg FK bones. During sitting, shins
        # hang mostly straight down — the X component is almost entirely
        # camera bias (left shin X=-0.14, right X=-0.08, BOTH negative =
        # same direction = clearly not real anatomy, it's camera geometry).
        # Damping X to 25% reduces the bias while allowing real lateral
        # movement to still show through.
        SEATED_LEG_LATERAL_DAMP = 0.25

        # v4.9: Seated arm depth damping.
        # The depth axis (Y in Blender/armature space) is the noisiest axis
        # in stereo triangulation — each camera measures it as a derived
        # quantity, not a direct pixel measurement. At 90° camera angle,
        # depth errors from the two cameras compound instead of canceling.
        # During sitting, upper arms should hang roughly straight down from
        # shoulders. Instead, elbow Y offset pushes arms behind the body
        # (upper_arm Y=+0.70 means elbow 16cm forward of shoulder in
        # Blender Y — physically wrong for armrests). Damping Y to 30%
        # during sitting keeps arms roughly at the body's sides in depth.
        SEATED_ARM_DEPTH_DAMP = 0.30

        # v5.0: Yaw depth-axis damping.
        # Y (depth) is the noisiest axis in stereo — damping it before
        # computing yaw reduces spurious rotation from depth noise.
        # Always-on (not just seated) because depth is always the worst axis.
        # At 0.35: small depth noise is heavily attenuated, real 90° turns
        # are barely affected (atan2 dominated by X shrinking, not Y).
        YAW_DEPTH_DAMP = 0.35

        # Spine rest direction: logged for diagnostics but NOT subtracted
        # from spine FK. The torso bone already applies rest-subtracted pitch.
        # Subtracting tilt from spine FK too causes double-rotation (v4.4 bug:
        # character leaned too far back when sitting because both torso pitch
        # and spine FK both corrected the same lean from the same data).
        spine_rest_dir = None
        cal_hip_mid = compute_midpoint(first_landmarks, LM.LEFT_HIP, LM.RIGHT_HIP)
        cal_sh_mid = compute_midpoint(first_landmarks, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)
        if cal_hip_mid and cal_sh_mid:
            spine_rest_dir = (armature_inv_33 @ (cal_sh_mid - cal_hip_mid)).normalized()
            DiagLog.data("Spine rest dir", f"({spine_rest_dir.x:.3f}, {spine_rest_dir.y:.3f}, {spine_rest_dir.z:.3f})")
            DiagLog.data("Spine rest tilt Y", f"{spine_rest_dir.y:.3f} (NOT subtracted — torso pitch handles this)")

        # Foot diagnostics: collect per-frame data for analysis
        foot_diag = {"L": [], "R": []}
        # Arm diagnostics: collect per-frame ratios for summary
        arm_ratios = {"L": [], "R": []}
        # Arm velocity clamp: track previous hand IK positions to reject spikes
        prev_hand_pos = {"L": None, "R": None}
        ARM_MAX_SPEED = 8.0  # m/s — fast arm swing is ~6 m/s, reject above this
        # v5.14: Lowered from 0.55 → 0.15. The 0.55 threshold dates from
        # the wrist-on-shoulder bug era (v4.7) and treated any normal bent-arm
        # pose (elbow ~70-90° = ratio 0.50-0.70 — guitar hold, hand on hip,
        # eating) as a triangulation error. With v5.13 fixing the actual
        # source of wrist-on-shoulder, this only needs to catch the genuine
        # degenerate case.
        ARM_MIN_RATIO = 0.15  # minimum arm reach ratio — safety net only

        # v4.6: Last-good arm FK pose — when arm_fk_conf drops below threshold,
        # hold the last good rotation instead of blending to identity (which puts
        # arms at the character's sides, wrong for armrest sitting).
        last_good_arm_rot = {
            "upper_arm_fk.L": None, "forearm_fk.L": None,
            "upper_arm_fk.R": None, "forearm_fk.R": None,
        }
        # v5.7: Track how long each bone has been held so we can release
        # toward rest pose instead of freezing forever. Frames 263–315 in the
        # v5.5 take had upper_arm_fk.L stuck at (0.485, 0.246, -0.839) for 50+
        # frames because the stability boost kept confidence alive while the
        # direction never actually changed. These counters + timeout break that.
        frames_since_arm_update = {
            "upper_arm_fk.L": 0, "forearm_fk.L": 0,
            "upper_arm_fk.R": 0, "forearm_fk.R": 0,
        }
        ARM_HOLD_CONF_THRESHOLD = 0.3  # below this, use last-good instead of rest
        ARM_FREEZE_TIMEOUT_FRAMES = 15  # after this many identical holds, release
        ARM_FREEZE_RELEASE_FRAMES = 8   # slerp back to rest over this many frames

        # v4.7: Track previous arm FK directions for stability-based confidence.
        # When direction is stable frame-to-frame (dot product near 1.0), the FK
        # data is trustworthy even if wrist-shoulder distance is short (armrest).
        # v5.7: stability boost now only applies when the BASE confidence is
        # already non-zero. Previously it could rescue genuinely bad data
        # (ratio < 0.55 → base conf 0) just because the direction was "stable",
        # which is exactly the feedback loop that caused the 50-frame freeze.
        prev_arm_dir = {"L": None, "R": None}
        ARM_STABILITY_BOOST = 0.5  # max confidence boost from direction stability
        ARM_STABILITY_DOT_THRESHOLD = 0.95  # dot product above this = stable
        ARM_STABILITY_BASE_CONF_MIN = 0.15  # require this much base conf to boost

        # v4.6: Track sit transition frame for dense logging
        sit_transition_frame = None  # frame_idx when sit state last changed
        prev_hip_pos_for_velocity = None  # for hip path velocity logging

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

            # v4.6: Dense logging — log every frame near sit transitions
            # and at regular intervals otherwise
            near_transition = (sit_transition_frame is not None and
                               abs(frame_idx - sit_transition_frame) <= 12)
            do_log = (frame_idx % log_every == 0) or near_transition

            # v5.3: Smooth sit_blend ramp instead of binary is_sitting.
            # All seated damping uses sit_blend (0→1 over SIT_BLEND_FRAMES)
            # so behaviors crossfade instead of snapping on/off.
            SIT_BLEND_FRAMES = 8
            prev_blend = mocap_props.get('_sit_blend', 0.0)

            # v4.6: Hip path velocity (XY displacement per frame)
            hip_xy_velocity = 0.0
            hip_xy_disp = None
            if hip_center and prev_hip_pos_for_velocity is not None:
                hip_dx = hip_center.x - prev_hip_pos_for_velocity.x
                hip_dy = hip_center.y - prev_hip_pos_for_velocity.y
                hip_dz_vel = hip_center.z - prev_hip_pos_for_velocity.z
                hip_xy_disp = math.sqrt(hip_dx*hip_dx + hip_dy*hip_dy)
                dt_hip = timestamp - prev_timestamp if prev_timestamp > 0 else 0.033
                hip_xy_velocity = hip_xy_disp / dt_hip if dt_hip > 0 else 0
            if hip_center:
                prev_hip_pos_for_velocity = hip_center.copy()

            if do_log:
                DiagLog.info(f"Frame {frame_idx}/{len(frames)} (t={timestamp:.2f}s, blender={frame_num}){' [TRANSITION]' if near_transition else ''}")
                if hip_center:
                    DiagLog.data("  hip_raw", f"({hip_center.x:.3f}, {hip_center.y:.3f}, {hip_center.z:.3f})")
                    # Show what the scaled hip will be (v5.1: delta-based)
                    sh_z = rig_rest_hip_z + (hip_center.z - mocap_props['hip_pos'].z) * hip_height_scale
                    DiagLog.data("  hip_scaled_z", f"{sh_z:.3f} (delta from rest={rig_rest_hip_z:.3f})")
                    # v5.1: Delta hip Z component breakdown
                    f0z = mocap_props['hip_pos'].z
                    perframe_offset = (hip_height_scale - 1.0) * hip_center.z
                    DiagLog.data("  hip_z_components", f"rest={rig_rest_hip_z:.3f} delta={(hip_center.z - f0z):+.3f} hhs={hip_height_scale:.3f} shift={hip_z_shift:+.4f} limb_perframe={perframe_offset:+.4f}")
                    # Hip drop from frame 0
                    dz = hip_center.z - f0z
                    DiagLog.data("  hip_dz_from_f0", f"{dz:+.3f}m")
                    # v4.6: Hip XY velocity for sit-down arc investigation
                    if hip_xy_disp is not None:
                        DiagLog.data("  hip_xy_disp", f"{hip_xy_disp:.4f}m  vel={hip_xy_velocity:.3f}m/s")

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
                    sit_transition_frame = frame_idx  # v4.6: track for dense logging
                    for ikfk_bone in ik_fk_bones:
                        is_leg = "thigh" in ikfk_bone.name.lower()
                        if is_leg:
                            # Sitting: legs → FK (1.0). Standing: legs → IK (0.0)
                            ikfk_bone["IK_FK"] = 1.0 if is_sitting else 0.0
                            ikfk_bone.keyframe_insert(data_path='["IK_FK"]', frame=frame_num)
                    DiagLog.info(f"  SIT DETECT: {'SITTING' if is_sitting else 'STANDING'} (hip_dz={hip_dz:+.3f}m) at frame_idx={frame_idx}")
                    mocap_props['_prev_sitting'] = is_sitting

            # v5.3: Ramp sit_blend toward target (1.0 sitting, 0.0 standing)
            target_blend = 1.0 if is_sitting else 0.0
            step = 1.0 / max(SIT_BLEND_FRAMES, 1)
            if prev_blend < target_blend:
                sit_blend = min(target_blend, prev_blend + step)
            elif prev_blend > target_blend:
                sit_blend = max(target_blend, prev_blend - step)
            else:
                sit_blend = target_blend
            mocap_props['_sit_blend'] = sit_blend

            # =====================
            # ROOT / TORSO POSITION (scaled)
            # =====================
            torso = rig.pose.bones.get("torso")
            if torso and hip_center:
                scaled_hip = hip_center.copy()
                # v5.1: Delta-based Z — anchored to rig rest position
                mocap_hip_f0_z = mocap_props['hip_pos'].z
                scaled_hip.z = rig_rest_hip_z + (hip_center.z - mocap_hip_f0_z) * hip_height_scale
                mocap_hip_frame0 = mocap_props.get('hip_pos')
                if mocap_hip_frame0:
                    dx = hip_center.x - mocap_hip_frame0.x
                    dy = hip_center.y - mocap_hip_frame0.y
                    # v5.2: Damp lateral (X) displacement when seated.
                    # 90° stereo setup exaggerates X drift during sit-down
                    # (20cm lateral shift is triangulation noise, not real movement).
                    SEATED_HIP_LATERAL_DAMP = 0.3
                    if sit_blend > 0:
                        effective_damp = 1.0 - sit_blend * (1.0 - SEATED_HIP_LATERAL_DAMP)
                        dx *= effective_damp
                    scaled_hip.x = mocap_hip_frame0.x * global_scale + dx * global_scale
                    scaled_hip.y = mocap_hip_frame0.y * global_scale + dy * global_scale
                    if do_log and sit_blend > 0:
                        raw_dx = hip_center.x - mocap_hip_frame0.x
                        DiagLog.data("  hip_xdamp", f"raw_dx={raw_dx:+.3f} damped_dx={dx:+.3f} damp={SEATED_HIP_LATERAL_DAMP:.2f}")

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
                    raw_yaw = 0.0
                    if body_right.length > 0.01:
                        body_right.y *= YAW_DEPTH_DAMP
                        body_right = body_right.normalized()
                        raw_yaw = math.atan2(-body_right.y, body_right.x)
                        # Subtract rest yaw (frame 0 bias) — mirrors torso_rest_pitch pattern
                        if 'torso_rest_yaw' not in mocap_props:
                            mocap_props['torso_rest_yaw'] = raw_yaw
                        yaw_angle = raw_yaw - mocap_props['torso_rest_yaw']

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

                    # v5.2: Clamp torso pitch when seated to prevent extreme lean
                    SEATED_PITCH_MAX = math.radians(35)   # max forward lean
                    SEATED_PITCH_MIN = math.radians(-20)  # max backward lean
                    if sit_blend > 0:
                        clamped = max(SEATED_PITCH_MIN, min(SEATED_PITCH_MAX, pitch_angle))
                        # v5.4: Store how much pitch was clamped — neck FK uses this
                        # to avoid compensating for depth error already handled here
                        mocap_props['_pitch_correction'] = sit_blend * (clamped - pitch_angle)
                        pitch_angle = pitch_angle + sit_blend * (clamped - pitch_angle)
                    else:
                        mocap_props['_pitch_correction'] = 0.0

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

                    if do_log:
                        DiagLog.data("  torso_yaw", f"{math.degrees(yaw_angle):.1f}° (raw={math.degrees(raw_yaw):.1f}° rest={math.degrees(mocap_props.get('torso_rest_yaw',0)):.1f}°)")
                        DiagLog.data("  torso_pitch", f"{math.degrees(pitch_angle):.1f}° (from vertical, rest-subtracted)")

            # =====================
            # FK ROTATIONS
            # =====================
            if self.use_fk:
                # --- Spine FK ---
                if self.spine_chain and spine_points.get('hip_mid') and spine_points.get('shoulder_mid'):
                    # Full chain: distribute rotation across all 4 spine FK bones
                    # Subtract frame-0 tilt before computing FK
                    chain_results = compute_spine_fk_chain(rig, spine_points, armature_inv_33)
                    for bone, quat in chain_results:
                        bone.rotation_mode = 'QUATERNION'
                        bone.rotation_quaternion = quat
                        bone.keyframe_insert(data_path="rotation_quaternion")
                    if do_log:
                        spine_dir = (armature_inv_33 @ (spine_points['shoulder_mid'] - spine_points['hip_mid'])).normalized()
                        DiagLog.data("  spine_dir", f"({spine_dir.x:.3f}, {spine_dir.y:.3f}, {spine_dir.z:.3f})")
                        DiagLog.data("  spine_chain_bones", f"{len(chain_results)}")
                else:
                    # Single bone fallback (default, proven stable)
                    spine_fk = rig.pose.bones.get("spine_fk")
                    if spine_fk and spine_points.get('hip_mid') and spine_points.get('shoulder_mid'):
                        spine_dir = (armature_inv_33 @ (spine_points['shoulder_mid'] - spine_points['hip_mid'])).normalized()
                        if do_log:
                            DiagLog.data("  spine_dir", f"({spine_dir.x:.3f}, {spine_dir.y:.3f}, {spine_dir.z:.3f})")
                        spine_fk.rotation_mode = 'QUATERNION'
                        spine_fk.rotation_quaternion = compute_fk_rotation(spine_fk, spine_dir)
                        spine_fk.keyframe_insert(data_path="rotation_quaternion")

                # --- v4.6: Flush depsgraph AFTER torso+spine, BEFORE neck ---
                # Neck FK uses 'auto' parent mode (reads bone.parent.matrix).
                # Without this flush, the parent matrices still reflect the
                # PREVIOUS frame, so neck 'auto' computes against stale torso
                # rotation. This was the cause of 59.9° neck pitch in v4.5.
                bpy.context.view_layer.update()

                # --- Head/Neck FK from face keypoints ---
                p_nose = get_landmark(landmarks_3d, LM.NOSE)
                p_lear = get_landmark(landmarks_3d, LM.LEFT_EAR)
                p_rear = get_landmark(landmarks_3d, LM.RIGHT_EAR)

                if p_lear and p_rear:
                    ear_mid = (p_lear + p_rear) / 2.0

                    # Head/neck confidence: attenuate when turned sideways to cameras.
                    torso_yaw_abs = abs(mocap_props.get('_current_torso_yaw', 0.0))
                    HEAD_YAW_FULL = math.radians(30)
                    HEAD_YAW_ZERO = math.radians(60)
                    if torso_yaw_abs <= HEAD_YAW_FULL:
                        head_confidence = 1.0
                    elif torso_yaw_abs >= HEAD_YAW_ZERO:
                        head_confidence = 0.0
                    else:
                        t = (torso_yaw_abs - HEAD_YAW_FULL) / (HEAD_YAW_ZERO - HEAD_YAW_FULL)
                        head_confidence = 1.0 - t * t * (3 - 2 * t)

                    neck_bone = rig.pose.bones.get("neck")
                    if neck_bone and spine_points.get('shoulder_mid'):
                        neck_dir = (armature_inv_33 @ (ear_mid - spine_points['shoulder_mid'])).normalized()

                        # v5.4: Correct neck_dir for torso pitch clamping.
                        # When torso pitch is clamped (e.g. -32° → -20°), ear_mid
                        # still reflects the raw spine tilt. Without this, the neck
                        # rotates forward to compensate for depth error that was
                        # already clamped at the torso level ("snake neck").
                        pitch_corr = mocap_props.get('_pitch_correction', 0)
                        if abs(pitch_corr) > math.radians(1):
                            corr_mat = Matrix.Rotation(pitch_corr, 3, 'X')
                            neck_dir = (corr_mat @ neck_dir).normalized()

                        neck_bone.rotation_mode = 'QUATERNION'
                        # 'auto' parent mode: now works correctly because
                        # depsgraph was flushed above with current torso+spine
                        neck_rot = compute_fk_rotation(neck_bone, neck_dir, 'auto')
                        NECK_ROT_MAX_STAND = math.radians(50)
                        NECK_ROT_MAX_SIT = math.radians(25)
                        neck_cap = NECK_ROT_MAX_STAND - sit_blend * (NECK_ROT_MAX_STAND - NECK_ROT_MAX_SIT)
                        if neck_rot.angle > neck_cap:
                            neck_rot = Quaternion(neck_rot.axis, neck_cap)

                        # v5.4: Neck angular velocity limiting.
                        # Prevents violent head-snap from depth spikes — caps
                        # change to 8°/frame (~160°/s at 20fps).
                        # v5.8: count fires for end-of-import diagnostic.
                        # If the clamp limited >5% of frames, the user's head
                        # turn was probably faster than the clamp allows and
                        # got smeared.
                        NECK_MAX_ANGULAR_VEL = math.radians(8)
                        prev_neck_angle = mocap_props.get('_prev_neck_angle', neck_rot.angle)
                        if neck_rot.angle - prev_neck_angle > NECK_MAX_ANGULAR_VEL:
                            neck_rot = Quaternion(neck_rot.axis, prev_neck_angle + NECK_MAX_ANGULAR_VEL)
                            mocap_props['_neck_clamp_fires'] = mocap_props.get('_neck_clamp_fires', 0) + 1
                        elif prev_neck_angle - neck_rot.angle > NECK_MAX_ANGULAR_VEL:
                            neck_rot = Quaternion(neck_rot.axis, max(0, prev_neck_angle - NECK_MAX_ANGULAR_VEL))
                            mocap_props['_neck_clamp_fires'] = mocap_props.get('_neck_clamp_fires', 0) + 1
                        mocap_props['_prev_neck_angle'] = neck_rot.angle

                        if head_confidence < 1.0:
                            neck_rot = Quaternion().slerp(neck_rot, head_confidence)
                        neck_bone.rotation_quaternion = neck_rot
                        neck_bone.keyframe_insert(data_path="rotation_quaternion")

                        if do_log:
                            # Log neck rotation magnitude for debugging
                            neck_angle = neck_rot.angle
                            DiagLog.data("  neck_rot", f"{math.degrees(neck_angle):.1f}° (conf={head_confidence:.2f})")

                    head_bone = rig.pose.bones.get("head")
                    if head_bone and p_nose:
                        p_ls = get_landmark(landmarks_3d, LM.LEFT_SHOULDER)
                        p_rs = get_landmark(landmarks_3d, LM.RIGHT_SHOULDER)
                        if p_ls and p_rs:
                            ear_vec = p_lear - p_rear
                            shoulder_vec = p_ls - p_rs
                            ear_angle = math.atan2(ear_vec.y, ear_vec.x)
                            shoulder_angle = math.atan2(shoulder_vec.y, shoulder_vec.x)
                            head_yaw = ear_angle - shoulder_angle
                            head_yaw = max(-1.05, min(1.05, head_yaw))  # v5.2: ±60° (was ±40.1°)
                            head_yaw *= head_confidence

                            head_bone.rotation_mode = 'QUATERNION'
                            head_bone.rotation_quaternion = Quaternion((0, 1, 0), head_yaw)
                            head_bone.keyframe_insert(data_path="rotation_quaternion")

                            if do_log:
                                DiagLog.data("  head_yaw", f"{math.degrees(head_yaw):.1f}° (conf={head_confidence:.2f})")
                                nose_ear_vec = p_nose - ear_mid
                                head_pitch = math.degrees(math.atan2(-nose_ear_vec.z, math.sqrt(nose_ear_vec.x**2 + nose_ear_vec.y**2)))
                                DiagLog.data("  head_pitch_via_neck", f"{head_pitch:.1f}°")

                # --- Second depsgraph flush: neck/head set, now update for limb FK ---
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
                        # AND direction stability (v4.7).
                        # Ratio-only confidence dropped arms to sides on armrests.
                        # Now: if the FK direction is STABLE frame-to-frame, boost
                        # confidence even at short ratios (armrest = stable + short).
                        sh_idx = LM.LEFT_SHOULDER if side == "L" else LM.RIGHT_SHOULDER
                        el_idx = LM.LEFT_ELBOW if side == "L" else LM.RIGHT_ELBOW
                        wr_idx = LM.LEFT_WRIST if side == "L" else LM.RIGHT_WRIST
                        p_sh_raw = get_landmark(landmarks_3d, sh_idx)
                        p_el_raw = get_landmark(landmarks_3d, el_idx)
                        p_wr_raw = get_landmark(landmarks_3d, wr_idx)
                        rig_arm_len = rig_props.get(f'arm.{side}', 0.53)
                        arm_fk_conf = 1.0
                        arm_ratio = 1.0
                        stability_boost = 0.0
                        if p_sh_raw is not None and p_wr_raw is not None and rig_arm_len > 0:
                            raw_dist = (p_wr_raw - p_sh_raw).length
                            arm_ratio = raw_dist / rig_arm_len
                            # v5.14: Threshold lowered from (0.55, 0.80) to (0.10, 0.20).
                            # Old thresholds treated any natural bent-elbow pose as low
                            # confidence; with v5.13 fixing the wrist-on-shoulder solver
                            # bug, we only need to catch the genuine degenerate case
                            # (wrist literally at shoulder, ratio < 0.10).
                            if arm_ratio < 0.20:
                                arm_fk_conf = max(0.0, (arm_ratio - 0.10) / 0.10)

                            # v4.7: Direction stability boost.
                            # v5.7: Gated on base confidence > ARM_STABILITY_BASE_CONF_MIN.
                            # Before v5.7, zero-base-conf frames (arm on armrest, ratio
                            # ≤ 0.55) could still get boosted because the direction
                            # matched itself trivially — creating a 50+ frame freeze
                            # since every frame "confirmed" the last one.
                            if p_el_raw is not None:
                                cur_dir = (p_el_raw - p_sh_raw)
                                if cur_dir.length > 0.01:
                                    cur_dir = cur_dir.normalized()
                                    if (prev_arm_dir[side] is not None
                                            and arm_fk_conf >= ARM_STABILITY_BASE_CONF_MIN):
                                        dot = cur_dir.dot(prev_arm_dir[side])
                                        if dot > ARM_STABILITY_DOT_THRESHOLD:
                                            stab_t = (dot - ARM_STABILITY_DOT_THRESHOLD) / (1.0 - ARM_STABILITY_DOT_THRESHOLD)
                                            ratio_scale = max(0.0, min(1.0, (arm_ratio - 0.55) / 0.25))
                                            stability_boost = stab_t * ARM_STABILITY_BOOST * ratio_scale
                                            arm_fk_conf = min(1.0, arm_fk_conf + stability_boost)
                                    prev_arm_dir[side] = cur_dir.copy()

                        # v4.7: Log arm_fk_conf with stability info
                        if do_log:
                            boost_str = f" stab_boost={stability_boost:.2f}" if stability_boost > 0 else ""
                            DiagLog.data(f"  arm_fk_conf.{side}",
                                f"{arm_fk_conf:.2f} (ratio={arm_ratio:.3f}{boost_str})")

                        # Upper arm first
                        ua_mapping = V2R_MAPPING.get(ua_name)
                        ua_expected_matrix = None
                        ua_damped_dir = None
                        # v5.8: shared with forearm — bypass seated depth damping
                        # when the arm is genuinely extended forward (e.g. holding
                        # a guitar in front of the body). Set inside the upper-arm
                        # block once raw_y is known.
                        arm_extended_forward = False
                        if ua_bone and ua_mapping:
                            p_start = get_landmark(landmarks_3d, ua_mapping[0])
                            p_end = get_landmark(landmarks_3d, ua_mapping[1])
                            if p_start is not None and p_end is not None:
                                target_dir = (armature_inv_33 @ (p_end - p_start)).normalized()

                                # v4.8: Fixed high splay limit (safety net only).
                                # Apply clamp: L arm outward is +X, R arm is -X
                                splay_clamped = False
                                if side == "L" and target_dir.x > ARM_SPLAY_LIMIT:
                                    target_dir.x = ARM_SPLAY_LIMIT
                                    target_dir = target_dir.normalized()
                                    splay_clamped = True
                                elif side == "R" and target_dir.x < -ARM_SPLAY_LIMIT:
                                    target_dir.x = -ARM_SPLAY_LIMIT
                                    target_dir = target_dir.normalized()
                                    splay_clamped = True

                                # v4.9: Seated arm depth damping — reduce depth axis noise
                                # in Y component when sitting (arms should hang down, not
                                # project forward/backward from triangulation error)
                                # v5.8: bypass when arm is reaching forward — long arm
                                # (ratio > 0.70) with strong positive Y is real reach,
                                # not depth noise. Damping it collapses guitar-hold pose.
                                raw_ua_y = target_dir.y
                                # v5.14: bypass depth damping whenever arm is reaching
                                # forward (raw_y > 0.20), regardless of how bent the
                                # elbow is. Previously required arm_ratio > 0.70 too,
                                # which excluded normal bent-arm reach poses (guitar
                                # hold has elbows ~80° = ratio ~0.55) and damped them
                                # back into the body.
                                arm_extended_forward = (raw_ua_y > 0.20)
                                if sit_blend > 0 and not arm_extended_forward:
                                    damp = 1.0 - sit_blend * (1.0 - SEATED_ARM_DEPTH_DAMP)
                                    target_dir.y *= damp
                                    target_dir = target_dir.normalized()
                                    # v5.1: Log upper arm depth damping
                                    if do_log:
                                        DiagLog.data(f"    {ua_name} ua_damp", f"raw_y={raw_ua_y:.3f} damp={damp:.2f} near_trans={near_transition}")
                                elif sit_blend > 0 and arm_extended_forward:
                                    if do_log:
                                        DiagLog.data(f"    {ua_name} ua_damp",
                                            f"raw_y={raw_ua_y:.3f} BYPASS (extended forward, ratio={arm_ratio:.2f})")

                                # Save upper arm's final direction for forearm rest context
                                ua_damped_dir = target_dir.copy()

                                ua_bone.rotation_mode = 'QUATERNION'
                                ua_rot = compute_fk_rotation(ua_bone, target_dir, 'auto')

                                # v4.6 / v5.7: Last-good-pose fallback with freeze timeout.
                                # Instead of blending to identity, we hold the last good
                                # FK rotation when confidence drops. But if we've been
                                # holding for >ARM_FREEZE_TIMEOUT_FRAMES, slerp back
                                # toward rest over ARM_FREEZE_RELEASE_FRAMES so a bad
                                # low-conf patch can't freeze the arm indefinitely.
                                ua_freeze_released = False
                                if arm_fk_conf >= ARM_HOLD_CONF_THRESHOLD:
                                    if arm_fk_conf < 1.0:
                                        fallback = last_good_arm_rot[ua_name] or Quaternion()
                                        ua_rot = fallback.slerp(ua_rot, arm_fk_conf)
                                    last_good_arm_rot[ua_name] = ua_rot.copy()
                                    frames_since_arm_update[ua_name] = 0
                                else:
                                    frames_since_arm_update[ua_name] += 1
                                    if last_good_arm_rot[ua_name] is not None:
                                        hold_frames = frames_since_arm_update[ua_name]
                                        if hold_frames > ARM_FREEZE_TIMEOUT_FRAMES:
                                            # Release: slerp held rotation toward rest
                                            release_t = min(
                                                1.0,
                                                (hold_frames - ARM_FREEZE_TIMEOUT_FRAMES)
                                                / ARM_FREEZE_RELEASE_FRAMES
                                            )
                                            ua_rot = last_good_arm_rot[ua_name].slerp(
                                                Quaternion(), release_t)
                                            last_good_arm_rot[ua_name] = ua_rot.copy()
                                            ua_freeze_released = True
                                            if do_log or release_t >= 1.0:
                                                DiagLog.info(
                                                    f"    ARM_FREEZE_RESET {ua_name} "
                                                    f"held={hold_frames}f t={release_t:.2f}")
                                        else:
                                            ua_rot = last_good_arm_rot[ua_name].copy()

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

                                if do_log:
                                    hold_str = ""
                                    if arm_fk_conf < ARM_HOLD_CONF_THRESHOLD and last_good_arm_rot[ua_name] is not None:
                                        hold_str = " RELEASE" if ua_freeze_released else " HOLD"
                                    clamp_str = " CLAMPED" if splay_clamped else ""
                                    ydamp_str = f" raw_y={raw_ua_y:.3f} YDAMP" if sit_blend > 0 else ""
                                    DiagLog.data(f"  arm_fk.{ua_name}",
                                        f"dir=({target_dir.x:.3f},{target_dir.y:.3f},{target_dir.z:.3f}){clamp_str}{ydamp_str}{hold_str}")

                        # Forearm second (uses upper_arm's computed matrix)
                        fa_mapping = V2R_MAPPING.get(fa_name)
                        if fa_bone and fa_mapping:
                            p_start = get_landmark(landmarks_3d, fa_mapping[0])
                            p_end = get_landmark(landmarks_3d, fa_mapping[1])
                            if p_start is not None and p_end is not None:
                                target_dir = (armature_inv_33 @ (p_end - p_start)).normalized()

                                # v5.0: Zero forearm depth (Y) when sitting.
                                # Elbow depth is overestimated by stereo, making
                                # forearm point backward (-Y) instead of down (-Z).
                                # Multiplicative damping fails here because Y dominates
                                # (raw_y=-0.96) — normalization re-inflates it.
                                # v5.1: Seated forearm rest direction blend.
                                # When arm_ratio < 0.55, the elbow-to-wrist vector is ~13cm
                                # (vs ~26cm arm length), making ALL three direction axes
                                # noise-dominated. Instead of trying to salvage bad data
                                # (Y=0 amplifies X noise via normalization), blend toward
                                # a neutral rest direction based on data quality (arm_ratio).
                                # v5.8: bypass the rest blend when the arm is reaching
                                # forward (set in upper-arm block). Mirrors the upper-arm
                                # bypass — guitar hold has hands forward at center, not
                                # depth noise.
                                raw_fa_y = target_dir.y
                                if sit_blend > 0 and not arm_extended_forward:
                                    FOREARM_INHERIT_LATERAL = 0.5
                                    fa_rest_x = ua_damped_dir.x * FOREARM_INHERIT_LATERAL if ua_damped_dir else 0.0
                                    FOREARM_REST_DIR = Vector((fa_rest_x, 0, -1)).normalized()
                                    # v5.14: Same threshold relaxation as arm_fk_conf.
                                    # Previously (0.55, 0.70) replaced legitimate bent-arm
                                    # forearms with rest direction during sitting,
                                    # collapsing held-object poses into "hands in lap".
                                    FA_RATIO_GOOD = 0.20
                                    FA_RATIO_BAD = 0.10
                                    if arm_ratio < FA_RATIO_BAD:
                                        seated_dir = FOREARM_REST_DIR.copy()
                                    elif arm_ratio < FA_RATIO_GOOD:
                                        blend = (arm_ratio - FA_RATIO_BAD) / (FA_RATIO_GOOD - FA_RATIO_BAD)
                                        seated_dir = FOREARM_REST_DIR.lerp(target_dir, blend).normalized()
                                    else:
                                        seated_dir = target_dir.copy()
                                    target_dir = target_dir.lerp(seated_dir, sit_blend).normalized()
                                    if do_log:
                                        DiagLog.data(f"    {fa_name} forearm", f"ratio={arm_ratio:.3f} raw_y={raw_fa_y:.3f} rest=({FOREARM_REST_DIR.x:.2f},{FOREARM_REST_DIR.y:.2f},{FOREARM_REST_DIR.z:.2f}) dir=({target_dir.x:.2f},{target_dir.y:.2f},{target_dir.z:.2f})")
                                elif sit_blend > 0 and arm_extended_forward:
                                    if do_log:
                                        DiagLog.data(f"    {fa_name} forearm",
                                            f"ratio={arm_ratio:.3f} raw_y={raw_fa_y:.3f} BYPASS (extended forward)")

                                fa_bone.rotation_mode = 'QUATERNION'
                                fa_rot = compute_fk_rotation(
                                    fa_bone, target_dir, ua_expected_matrix)

                                # v4.6 / v5.7: Same last-good-pose + freeze timeout for forearm
                                fa_freeze_released = False
                                if arm_fk_conf >= ARM_HOLD_CONF_THRESHOLD:
                                    if arm_fk_conf < 1.0:
                                        fallback = last_good_arm_rot[fa_name] or Quaternion()
                                        fa_rot = fallback.slerp(fa_rot, arm_fk_conf)
                                    last_good_arm_rot[fa_name] = fa_rot.copy()
                                    frames_since_arm_update[fa_name] = 0
                                else:
                                    frames_since_arm_update[fa_name] += 1
                                    if last_good_arm_rot[fa_name] is not None:
                                        hold_frames = frames_since_arm_update[fa_name]
                                        if hold_frames > ARM_FREEZE_TIMEOUT_FRAMES:
                                            release_t = min(
                                                1.0,
                                                (hold_frames - ARM_FREEZE_TIMEOUT_FRAMES)
                                                / ARM_FREEZE_RELEASE_FRAMES
                                            )
                                            fa_rot = last_good_arm_rot[fa_name].slerp(
                                                Quaternion(), release_t)
                                            last_good_arm_rot[fa_name] = fa_rot.copy()
                                            fa_freeze_released = True
                                            if do_log or release_t >= 1.0:
                                                DiagLog.info(
                                                    f"    ARM_FREEZE_RESET {fa_name} "
                                                    f"held={hold_frames}f t={release_t:.2f}")
                                        else:
                                            fa_rot = last_good_arm_rot[fa_name].copy()

                                fa_bone.rotation_quaternion = fa_rot
                                fa_bone.keyframe_insert(data_path="rotation_quaternion")

                                if do_log:
                                    hold_str = ""
                                    if arm_fk_conf < ARM_HOLD_CONF_THRESHOLD and last_good_arm_rot[fa_name] is not None:
                                        hold_str = " RELEASE" if fa_freeze_released else " HOLD"
                                    ydamp_fa_str = f" raw_y={raw_fa_y:.3f} YDAMP" if sit_blend > 0 else ""
                                    DiagLog.data(f"  arm_fk.{fa_name}",
                                        f"dir=({target_dir.x:.3f},{target_dir.y:.3f},{target_dir.z:.3f}){ydamp_fa_str}{hold_str}")

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

                                # v4.8: Seated lateral damping — reduce camera
                                # placement bias in X component of leg FK
                                raw_x = target_dir.x
                                if sit_blend > 0:
                                    eff_leg_damp = 1.0 - sit_blend * (1.0 - SEATED_LEG_LATERAL_DAMP)
                                    target_dir.x *= eff_leg_damp
                                target_dir = target_dir.normalized()

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

                                # v4.8: Log leg FK with raw vs damped X
                                if do_log:
                                    rot_angle = math.degrees(rot.angle)
                                    DiagLog.data(f"  leg_fk.{bone_name}",
                                        f"dir=({target_dir.x:.3f},{target_dir.y:.3f},{target_dir.z:.3f}) raw_x={raw_x:.3f} rot={rot_angle:.1f}°")

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

                            # v4.8: Seated lateral damping
                            raw_x = target_dir.x
                            target_dir.x *= SEATED_LEG_LATERAL_DAMP
                            target_dir = target_dir.normalized()

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

                            # v4.8: Log leg FK with raw vs damped X
                            if do_log:
                                rot_angle = math.degrees(rot.angle)
                                DiagLog.data(f"  leg_fk.{bone_name}",
                                    f"dir=({target_dir.x:.3f},{target_dir.y:.3f},{target_dir.z:.3f}) raw_x={raw_x:.3f} rot={rot_angle:.1f}°")
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

                # Arm diagnostics
                if do_log:
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
                            # v5.14: threshold lowered from 0.8 → 0.20. The 0.8 cutoff
                            # was based on the old assumption that ratio<0.8 meant a
                            # capture/retargeting error; now ratio 0.4-0.7 is normal
                            # bent-elbow geometry (guitar hold, hands on hips). Only
                            # warn when ratio is genuinely degenerate.
                            if arm_len > 0 and ratio < 0.20:
                                DiagLog.info(f"  !! ARM {side_label} RATIO {ratio:.3f} < 0.20 — "
                                    f"wrist target very close to shoulder. "
                                    f"Capture or retargeting error!")

                for bone_name, landmark_idx in IK_TARGETS.items():
                    if is_sitting and "foot" in bone_name:
                        foot_side = "L" if ".L" in bone_name else "R"
                        prev_foot_raw[foot_side] = None
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
                                    if do_log:
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
                            pos_scaled.z += (hip_height_scale - 1.0) * hip_center.z + hip_z_shift
                            mocap_hip_frame0 = mocap_props.get('hip_pos')
                            if mocap_hip_frame0:
                                pos_scaled.x += (global_scale - 1.0) * mocap_hip_frame0.x
                                pos_scaled.y += (global_scale - 1.0) * mocap_hip_frame0.y
                    else:
                        # Legs: scale from hip_center (correct for leg reach)
                        pos = scale_position(pos, hip_center, chain_scale)

                        # Height/XY offset for leg IK targets
                        pos_scaled = pos.copy()
                        pos_scaled.z += (hip_height_scale - 1.0) * hip_center.z + hip_z_shift
                        mocap_hip_frame0 = mocap_props.get('hip_pos')
                        if mocap_hip_frame0:
                            pos_scaled.x += (global_scale - 1.0) * mocap_hip_frame0.x
                            pos_scaled.y += (global_scale - 1.0) * mocap_hip_frame0.y

                    is_foot = "foot" in bone_name
                    side = "L" if ".L" in bone_name else "R"

                    # v5.1: Log Z offset components for feet
                    if do_log and is_foot:
                        z_perframe = (hip_height_scale - 1.0) * hip_center.z
                        DiagLog.data(f"    {bone_name} z_offset", f"perframe={z_perframe:+.4f} + shift={hip_z_shift:+.4f} = {z_perframe + hip_z_shift:+.4f}")

                    if is_foot:
                        # Apply ankle-to-ground offset so feet naturally sit near floor.
                        # The mocap ankle keypoint is ~5-7cm above ground anatomically.
                        # Without this, unpinned feet visibly float.
                        pos_scaled.z -= foot_z_offset[side]

                        # v5.6: FOOT_Z_FLOOR removed — it created an artificial Z gap
                        # between ground-clamped pos_scaled (z=0) and pre-clamp
                        # prev_foot_raw (z=-0.15), producing ~3.2 m/s phantom velocity
                        # every frame and preventing the foot from ever pinning.
                        # ground_clamp already handles the visual, and the velocity
                        # clamp below catches genuine spikes.
                        FOOT_MAX_SPEED = 6.0
                        if prev_foot_raw[side] is not None:
                            foot_dt = timestamp - prev_timestamp if prev_timestamp > 0 else 0.033
                            foot_delta = (pos_scaled - prev_foot_raw[side]).length
                            foot_vel = foot_delta / max(foot_dt, 0.001)
                            if foot_vel > FOOT_MAX_SPEED:
                                pos_scaled = prev_foot_raw[side].copy()

                        raw_z = pos_scaled.z
                        raw_pos = pos_scaled.copy()

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
                                # v5.6: compare raw_pos (pre-ground-clamp) to prev_foot_raw
                                # so small mocap Z oscillations near/below 0 don't get
                                # flattened into a phantom gap by ground_clamp, which
                                # was preventing pinning from ever latching.
                                dist = (raw_pos - prev_foot_raw[side]).length
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
                        pole_pos.z += (hip_height_scale - 1.0) * hip_center.z + hip_z_shift
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

        # v5.8: head-turn velocity diagnostic.
        # NECK_MAX_ANGULAR_VEL = 8°/frame catches depth-noise spikes but also
        # smears intentional fast head turns. Surface the count so the user
        # can decide whether to slow head turns in the next take.
        neck_fires = mocap_props.get('_neck_clamp_fires', 0)
        neck_fires_pct = (neck_fires / max(1, len(frames))) * 100.0
        if neck_fires == 0:
            DiagLog.data("HEAD_TURN_LIMITED", "0 frames")
        elif neck_fires_pct > 5.0:
            DiagLog.info(
                f"[WARNING] HEAD_TURN_LIMITED: neck angular velocity clamp "
                f"fired on {neck_fires}/{len(frames)} frames "
                f"({neck_fires_pct:.1f}%). >5% suggests the head turn was "
                f"faster than the 8°/frame clamp; slow head turns in the "
                f"next take or raise NECK_MAX_ANGULAR_VEL.")
        else:
            DiagLog.info(
                f"[INFO] HEAD_TURN_LIMITED: neck angular velocity clamp "
                f"fired on {neck_fires}/{len(frames)} frames "
                f"({neck_fires_pct:.1f}%). Within expected range for "
                f"noise-only fires.")

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
