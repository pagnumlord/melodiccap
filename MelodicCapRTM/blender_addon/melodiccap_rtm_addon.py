"""
MelodicCap RTM Blender Addon v1.0
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
    "version": (2, 0),
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

# Rest pose axes (direction bone points in rest pose, armature space)
BONE_REST_AXES = {
    "upper_arm_fk.L": Vector((1, 0, 0)),
    "forearm_fk.L": Vector((1, 0, 0)),
    "upper_arm_fk.R": Vector((-1, 0, 0)),
    "forearm_fk.R": Vector((-1, 0, 0)),
    "thigh_fk.L": Vector((0, 0, -1)),
    "shin_fk.L": Vector((0, 0, -1)),
    "foot_fk.L": Vector((0, 1, 0)),
    "thigh_fk.R": Vector((0, 0, -1)),
    "shin_fk.R": Vector((0, 0, -1)),
    "foot_fk.R": Vector((0, 1, 0)),
    "spine_fk": Vector((0, 0, 1)),
    "spine_fk.001": Vector((0, 0, 1)),
    "spine_fk.002": Vector((0, 0, 1)),
    "spine_fk.003": Vector((0, 0, 1)),
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

# Rest axes for finger bones (all point outward along the finger)
FINGER_REST_AXES = {}
for bone_name in FINGER_FK_MAPPING:
    if ".L" in bone_name:
        FINGER_REST_AXES[bone_name] = Vector((1, 0, 0))
    else:
        FINGER_REST_AXES[bone_name] = Vector((-1, 0, 0))
# Thumbs have different rest orientation
for side in [".L", ".R"]:
    for seg in ["thumb.01", "thumb.02", "thumb.03"]:
        name = seg + side
        if side == ".L":
            FINGER_REST_AXES[name] = Vector((0.7, 0.7, 0))
        else:
            FINGER_REST_AXES[name] = Vector((-0.7, 0.7, 0))


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


def compute_rotation(rest_axis, target_direction):
    """Compute quaternion to rotate rest_axis to target_direction."""
    rest_axis = rest_axis.normalized()
    target_direction = target_direction.normalized()
    return rest_axis.rotation_difference(target_direction)


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
    # Check if any left hand keypoints exist (indices 91-111)
    for i in range(91, 112):
        if str(i) in landmarks_3d:
            return True
    return False


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
        description="Apply rotations to FK bones",
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

        if is_rtm:
            print(f"[MelodicCap] Loading RTM format ({data.get('detector', 'unknown')})")
        else:
            print(f"[MelodicCap] Loading legacy MediaPipe format, converting indices...")

        # Switch to pose mode
        bpy.ops.object.mode_set(mode='POSE')

        # Set IK/FK sliders
        for bone in rig.pose.bones:
            if "IK_FK" in bone.keys():
                if self.use_ik:
                    bone["IK_FK"] = 0.0
                else:
                    bone["IK_FK"] = 1.0

        # Check if hand data is available (first frame)
        first_landmarks = frames[0].get('landmarks_3d', {})
        if not is_rtm:
            first_landmarks = _convert_mp_frame(first_landmarks)
        finger_data_available = has_hand_data(first_landmarks)

        if finger_data_available and self.use_fingers:
            print("[MelodicCap] Wholebody hand data detected — finger tracking enabled")
        elif self.use_fingers and not finger_data_available:
            print("[MelodicCap] No hand keypoint data — finger tracking disabled")

        # Track previous foot positions for pinning
        prev_foot_pos = {"L": None, "R": None}
        pinned_foot_pos = {"L": None, "R": None}

        # Process each frame
        for frame_data in frames:
            timestamp = frame_data.get('timestamp', 0)
            frame_num = int(timestamp * fps)
            context.scene.frame_set(frame_num)

            landmarks_3d = frame_data.get('landmarks_3d', {})
            if not landmarks_3d:
                continue

            # Convert legacy format
            if not is_rtm:
                landmarks_3d = _convert_mp_frame(landmarks_3d)

            # Compute virtual spine points
            spine_points = compute_virtual_spine_points(landmarks_3d)

            # =====================
            # ROOT / TORSO POSITION
            # =====================
            torso = rig.pose.bones.get("torso")
            if torso and spine_points.get('hip_mid'):
                pos = spine_points['hip_mid']
                set_bone_world_position(rig, torso, pos)
                torso.keyframe_insert(data_path="location")

            # =====================
            # FK ROTATIONS
            # =====================
            if self.use_fk:
                # Limbs
                for bone_name, (start_idx, end_idx) in V2R_MAPPING.items():
                    bone = rig.pose.bones.get(bone_name)
                    if not bone:
                        continue

                    p_start = get_landmark(landmarks_3d, start_idx)
                    p_end = get_landmark(landmarks_3d, end_idx)

                    if p_start is None or p_end is None:
                        continue

                    direction = (p_end - p_start).normalized()
                    rest_axis = BONE_REST_AXES.get(bone_name, Vector((0, 1, 0)))

                    local_dir = rig.matrix_world.inverted().to_quaternion() @ direction

                    bone.rotation_mode = 'QUATERNION'
                    bone.rotation_quaternion = compute_rotation(rest_axis, local_dir)
                    bone.keyframe_insert(data_path="rotation_quaternion")

                # Spine chain
                spine_chain = [
                    ("spine_fk", 'hip_mid', 'spine_low'),
                    ("spine_fk.001", 'spine_low', 'spine_mid'),
                    ("spine_fk.002", 'spine_mid', 'chest'),
                    ("spine_fk.003", 'chest', 'shoulder_mid'),
                ]

                for bone_name, start_key, end_key in spine_chain:
                    bone = rig.pose.bones.get(bone_name)
                    if not bone:
                        continue

                    p_start = spine_points.get(start_key)
                    p_end = spine_points.get(end_key)

                    if p_start is None or p_end is None:
                        continue

                    direction = (p_end - p_start).normalized()
                    rest_axis = BONE_REST_AXES.get(bone_name, Vector((0, 0, 1)))
                    local_dir = rig.matrix_world.inverted().to_quaternion() @ direction

                    bone.rotation_mode = 'QUATERNION'
                    bone.rotation_quaternion = compute_rotation(rest_axis, local_dir)
                    bone.keyframe_insert(data_path="rotation_quaternion")

                # Fingers (only with wholebody data)
                if finger_data_available and self.use_fingers:
                    for bone_name, (start_idx, end_idx) in FINGER_FK_MAPPING.items():
                        bone = rig.pose.bones.get(bone_name)
                        if not bone:
                            continue

                        p_start = get_landmark(landmarks_3d, start_idx)
                        p_end = get_landmark(landmarks_3d, end_idx)

                        if p_start is None or p_end is None:
                            continue

                        direction = (p_end - p_start).normalized()
                        rest_axis = FINGER_REST_AXES.get(bone_name, Vector((1, 0, 0)))
                        local_dir = rig.matrix_world.inverted().to_quaternion() @ direction

                        bone.rotation_mode = 'QUATERNION'
                        bone.rotation_quaternion = compute_rotation(rest_axis, local_dir)
                        bone.keyframe_insert(data_path="rotation_quaternion")

            # =====================
            # IK POSITIONS
            # =====================
            if self.use_ik:
                for bone_name, landmark_idx in IK_TARGETS.items():
                    bone = rig.pose.bones.get(bone_name)
                    if not bone:
                        continue

                    pos = get_landmark(landmarks_3d, landmark_idx)
                    if pos is None:
                        continue

                    pos = pos.copy()

                    is_foot = "foot" in bone_name
                    side = "L" if ".L" in bone_name else "R"

                    if is_foot:
                        if self.ground_clamp and pos.z < 0:
                            pos.z = 0

                        if self.pin_threshold > 0:
                            if prev_foot_pos[side] is not None:
                                velocity = (pos - prev_foot_pos[side]).length

                                if velocity < self.pin_threshold and pos.z < 0.1:
                                    if pinned_foot_pos[side] is None:
                                        pinned_foot_pos[side] = pos.copy()
                                        pinned_foot_pos[side].z = 0
                                    pos = pinned_foot_pos[side]
                                else:
                                    pinned_foot_pos[side] = None

                            prev_foot_pos[side] = pos.copy()

                    set_bone_world_position(rig, bone, pos)
                    bone.keyframe_insert(data_path="location")

                # Pole targets
                for bone_name, (root_idx, mid_idx, end_idx) in POLE_TARGETS.items():
                    bone = rig.pose.bones.get(bone_name)
                    if not bone:
                        continue

                    p_root = get_landmark(landmarks_3d, root_idx)
                    p_mid = get_landmark(landmarks_3d, mid_idx)
                    p_end = get_landmark(landmarks_3d, end_idx)

                    if None in (p_root, p_mid, p_end):
                        continue

                    pole_pos = compute_pole_position(p_root, p_mid, p_end)
                    set_bone_world_position(rig, bone, pole_pos)
                    bone.keyframe_insert(data_path="location")

        self.report({'INFO'},
                    f"Imported {len(frames)} frames from {os.path.basename(self.filepath)} "
                    f"({format_name})")
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
