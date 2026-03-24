"""
MelodicCap RTM Blender Addon v3.0
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
    "version": (3, 0),
    "blender": (4, 4, 0),
    "location": "View3D > Sidebar > MelodicCap",
    "description": "Import MelodicCap RTM/Fresh JSON motion capture to JaxRigify",
    "category": "Animation",
}

import bpy
import json
import os
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
# FK ROTATION HELPERS
# =============================================================================

def compute_fk_rotation(bone, target_dir_armature):
    """
    Compute rotation_quaternion for a pose bone to point along target_dir.

    rotation_quaternion is applied in bone-local space. The bone's Y axis
    (0,1,0) points along its length at rest. bone.bone.matrix_local transforms
    from bone-local to armature space.

    For a root-level bone (or one whose parent is at rest):
      final_dir = matrix_local @ pose_rotation @ (0,1,0)
    So:
      pose_rotation @ (0,1,0) = matrix_local.inv() @ target_dir
      pose_rotation = (0,1,0).rotation_difference(matrix_local.inv() @ target_dir)
    """
    rest_inv = bone.bone.matrix_local.to_3x3().inverted()
    target_local = (rest_inv @ target_dir_armature).normalized()
    return Vector((0, 1, 0)).rotation_difference(target_local)


def compute_spine_fk_chain(rig, spine_points, armature_inv_33):
    """
    Compute FK rotations for the spine chain with parent-space tracking.

    Processes bones in order, tracking the cumulative posed rotation so each
    child bone correctly accounts for its parent's pose.

    Returns list of (bone, quaternion) pairs to apply.
    """
    spine_chain = [
        ("spine_fk", 'hip_mid', 'spine_low'),
        ("spine_fk.001", 'spine_low', 'spine_mid'),
        ("spine_fk.002", 'spine_mid', 'chest'),
        ("spine_fk.003", 'chest', 'shoulder_mid'),
    ]

    results = []

    first_bone = rig.pose.bones.get("spine_fk")
    if first_bone and first_bone.parent:
        cumulative_quat = first_bone.parent.bone.matrix_local.to_quaternion()
    elif first_bone:
        cumulative_quat = Quaternion()
    else:
        return results

    for bone_name, start_key, end_key in spine_chain:
        bone = rig.pose.bones.get(bone_name)
        if not bone:
            continue

        p_start = spine_points.get(start_key)
        p_end = spine_points.get(end_key)
        if p_start is None or p_end is None:
            continue

        # Target direction in armature space
        target_dir = (armature_inv_33 @ (p_end - p_start)).normalized()

        # This bone's rest rotation relative to its parent
        if bone.parent:
            rest_rel = (bone.parent.bone.matrix_local.to_quaternion().inverted()
                        @ bone.bone.matrix_local.to_quaternion())
        else:
            rest_rel = bone.bone.matrix_local.to_quaternion()

        # Effective orientation = cumulative parent posed + this bone's rest
        effective = cumulative_quat @ rest_rel

        # Target direction in this bone's effective local space
        target_local = (effective.inverted() @ target_dir).normalized()

        # Rotation from (0,1,0) to target in local space
        pose_quat = Vector((0, 1, 0)).rotation_difference(target_local)

        results.append((bone, pose_quat))

        # Update cumulative for the next bone in chain
        cumulative_quat = effective @ pose_quat

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
        description="Apply finger rotations from wholebody data (requires RTM format)",
        default=True
    )

    ground_clamp: bpy.props.BoolProperty(
        name="Ground Clamp Feet",
        description="Prevent feet from going below Z=0",
        default=True
    )

    pin_threshold: bpy.props.FloatProperty(
        name="Foot Pin Threshold",
        description="Velocity threshold for foot pinning (0 = disabled)",
        default=0.02,
        min=0.0,
        max=0.1
    )

    smooth_window: bpy.props.IntProperty(
        name="Smoothing Window",
        description="Moving average window size (1 = off, 5 = moderate, 9 = heavy)",
        default=5,
        min=1,
        max=15
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

        # Set IK/FK sliders
        ik_fk_count = 0
        for bone in rig.pose.bones:
            if "IK_FK" in bone.keys():
                bone["IK_FK"] = 0.0 if self.use_ik else 1.0
                ik_fk_count += 1
        DiagLog.data("IK_FK sliders found", ik_fk_count)

        # Convert legacy frames if needed
        if not is_rtm:
            DiagLog.info("Converting legacy MediaPipe indices...")
            for i, f in enumerate(frames):
                lm = f.get('landmarks_3d', {})
                if lm:
                    frames[i]['landmarks_3d'] = _convert_mp_frame(lm)

        # Check hand data availability (first frame)
        first_landmarks = frames[0].get('landmarks_3d', {})
        finger_data_available = has_hand_data(first_landmarks)
        DiagLog.data("Finger data available", finger_data_available)

        # =====================
        # TEMPORAL SMOOTHING
        # =====================
        if self.smooth_window > 1:
            DiagLog.info(f"Applying {self.smooth_window}-frame moving average...")
            frames = smooth_frames(frames, self.smooth_window)

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

        # Precompute armature inverse matrix (3x3 for directions, 4x4 for positions)
        armature_inv = rig.matrix_world.inverted()
        armature_inv_33 = armature_inv.to_3x3()

        # Track previous foot positions for pinning
        prev_foot_pos = {"L": None, "R": None}
        pinned_foot_pos = {"L": None, "R": None}

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
                    DiagLog.data("  hip_center", f"({hip_center.x:.3f}, {hip_center.y:.3f}, {hip_center.z:.3f})")

            # =====================
            # ROOT / TORSO POSITION (scaled)
            # =====================
            torso = rig.pose.bones.get("torso")
            if torso and hip_center:
                scaled_hip = hip_center.copy()
                scaled_hip.z *= global_scale
                mocap_hip_frame0 = mocap_props.get('hip_pos')
                if mocap_hip_frame0:
                    dx = hip_center.x - mocap_hip_frame0.x
                    dy = hip_center.y - mocap_hip_frame0.y
                    scaled_hip.x = mocap_hip_frame0.x * global_scale + dx * global_scale
                    scaled_hip.y = mocap_hip_frame0.y * global_scale + dy * global_scale

                set_bone_world_position(rig, torso, scaled_hip)
                torso.keyframe_insert(data_path="location")

            # =====================
            # FK ROTATIONS
            # =====================
            if self.use_fk:
                # --- Spine chain (proper parent-space tracking) ---
                spine_results = compute_spine_fk_chain(rig, spine_points, armature_inv_33)
                for bone, pose_quat in spine_results:
                    bone.rotation_mode = 'QUATERNION'
                    bone.rotation_quaternion = pose_quat
                    bone.keyframe_insert(data_path="rotation_quaternion")

                # --- Limb FK (only in FK mode, skipped when IK active) ---
                if not self.use_ik:
                    for bone_name, (start_idx, end_idx) in V2R_MAPPING.items():
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
                for bone_name, landmark_idx in IK_TARGETS.items():
                    bone = rig.pose.bones.get(bone_name)
                    if not bone:
                        continue

                    pos = get_landmark(landmarks_3d, landmark_idx)
                    if pos is None:
                        continue

                    # Scale position relative to hip center using chain-specific scale
                    chain_key = IK_SCALE_KEY.get(bone_name, 'global')
                    chain_scale = scales.get(chain_key, global_scale)
                    pos = scale_position(pos, hip_center, chain_scale)

                    # Also scale the hip center's own position (height + XY)
                    pos_scaled = pos.copy()
                    pos_scaled.z += (global_scale - 1.0) * hip_center.z
                    mocap_hip_frame0 = mocap_props.get('hip_pos')
                    if mocap_hip_frame0:
                        pos_scaled.x += (global_scale - 1.0) * mocap_hip_frame0.x
                        pos_scaled.y += (global_scale - 1.0) * mocap_hip_frame0.y

                    is_foot = "foot" in bone_name
                    side = "L" if ".L" in bone_name else "R"

                    if is_foot:
                        if self.ground_clamp and pos_scaled.z < 0:
                            pos_scaled.z = 0

                        if self.pin_threshold > 0:
                            if prev_foot_pos[side] is not None:
                                velocity = (pos_scaled - prev_foot_pos[side]).length

                                if velocity < self.pin_threshold and pos_scaled.z < 0.1:
                                    if pinned_foot_pos[side] is None:
                                        pinned_foot_pos[side] = pos_scaled.copy()
                                        pinned_foot_pos[side].z = 0
                                    pos_scaled = pinned_foot_pos[side]
                                else:
                                    pinned_foot_pos[side] = None

                            prev_foot_pos[side] = pos_scaled.copy()

                    set_bone_world_position(rig, bone, pos_scaled)
                    bone.keyframe_insert(data_path="location")

                # Pole targets (also scaled)
                for bone_name, (root_idx, mid_idx, end_idx) in POLE_TARGETS.items():
                    bone = rig.pose.bones.get(bone_name)
                    if not bone:
                        continue

                    p_root = get_landmark(landmarks_3d, root_idx)
                    p_mid = get_landmark(landmarks_3d, mid_idx)
                    p_end = get_landmark(landmarks_3d, end_idx)

                    if None in (p_root, p_mid, p_end):
                        continue

                    # Scale the joint positions before computing pole
                    chain_key = 'arm.L' if 'arm' in bone_name and '.L' in bone_name else \
                                'arm.R' if 'arm' in bone_name and '.R' in bone_name else \
                                'leg.L' if 'thigh' in bone_name and '.L' in bone_name else \
                                'leg.R' if 'thigh' in bone_name and '.R' in bone_name else 'global'
                    cs = scales.get(chain_key, global_scale)
                    p_root_s = scale_position(p_root, hip_center, cs)
                    p_mid_s = scale_position(p_mid, hip_center, cs)
                    p_end_s = scale_position(p_end, hip_center, cs)

                    pole_pos = compute_pole_position(p_root_s, p_mid_s, p_end_s)

                    # Apply same global offset as IK targets
                    pole_pos.z += (global_scale - 1.0) * hip_center.z
                    mocap_hip_frame0 = mocap_props.get('hip_pos')
                    if mocap_hip_frame0:
                        pole_pos.x += (global_scale - 1.0) * mocap_hip_frame0.x
                        pole_pos.y += (global_scale - 1.0) * mocap_hip_frame0.y

                    set_bone_world_position(rig, bone, pole_pos)
                    bone.keyframe_insert(data_path="location")

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
