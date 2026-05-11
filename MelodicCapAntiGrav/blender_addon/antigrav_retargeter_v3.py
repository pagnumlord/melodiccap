bl_info = {
    "name": "AntiGrav V3 Retargeter",
    "author": "Antigravity AI",
    "version": (3, 2),
    "blender": (4, 4, 0),
    "location": "View3D > Sidebar > AntiGrav",
    "description": "Unified Mocap Retargeter for Rigify - IK/FK with correct mirroring and standardized format.",
    "category": "Animation",
}

import bpy
import json
import os
import math
from mathutils import Vector, Quaternion, Matrix, Euler
from bpy_extras.io_utils import ImportHelper

# =============================================================================
# UNIFIED DATA FORMAT HELPERS
# =============================================================================

def load_take(filepath):
    """Load a take file and normalize to unified format.
    Handles all historical formats:
      - v8/Fresh: frames[].landmarks = {"0": [x,y,z], ...}
      - AntiGrav: frames[].landmarks_3d = {"0": [x,y,z], ...}
      - OldPipeline: frames[].landmarks_3d = [{index, x, y, z, vis}, ...]
      - OldPipeline: frames[].landmarks_3d = {} (empty, but has raw_2d)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    frames = data.get('frames', [])
    normalized = []

    for frame in frames:
        # Try unified format first
        lms = frame.get('landmarks')

        # Fallback to landmarks_3d
        if not lms:
            lms = frame.get('landmarks_3d')

        # Handle list-of-dicts format (oldest pipeline)
        if isinstance(lms, list):
            lms_dict = {}
            for item in lms:
                if isinstance(item, dict) and 'index' in item:
                    lms_dict[str(item['index'])] = [item['x'], item['y'], item['z']]
            lms = lms_dict

        # Ensure string keys
        if isinstance(lms, dict):
            clean = {}
            for k, v in lms.items():
                clean[str(k)] = v
            lms = clean
        else:
            lms = {}

        normalized.append({
            'timestamp': frame.get('timestamp', 0),
            'landmarks': lms,
            'hands_3d': frame.get('hands_3d', []),
        })

    calib = data.get('calibration', {})
    return {
        'frames': normalized,
        'fps': data.get('fps', 30),
        'floor_offset': calib.get('floor_z_offset', calib.get('floor_offset', 0.0)),
    }


def get_lm(landmarks, idx):
    """Get landmark by index, returns Vector or None."""
    key = str(idx)
    if key in landmarks:
        p = landmarks[key]
        if isinstance(p, (list, tuple)) and len(p) >= 3:
            return Vector((p[0], p[1], p[2]))
    return None


def get_hip_center(landmarks):
    """Average of left and right hip."""
    l = get_lm(landmarks, 23)
    r = get_lm(landmarks, 24)
    if l and r:
        return (l + r) / 2
    return None


def mp_to_blender(v):
    """Convert MelodicCap Blender-space vector with X-mirroring for facing-camera setup.
    When performer faces camera: their LEFT appears on screen RIGHT.
    MediaPipe LEFT landmarks (11,13,15,23,25,27) = Rigify RIGHT bones.
    """
    return Vector((-v.x, v.y, v.z))


# =============================================================================
# BONE MAPPINGS
# =============================================================================

# MediaPipe LEFT limbs map to Rigify RIGHT (performer faces camera)
FK_MAP = {
    'upper_arm_fk.R': (11, 13),   # person's left shoulder->elbow
    'forearm_fk.R': (13, 15),     # person's left elbow->wrist
    'thigh_fk.R': (23, 25),       # person's left hip->knee
    'shin_fk.R': (25, 27),        # person's left knee->ankle
    'upper_arm_fk.L': (12, 14),   # person's right shoulder->elbow
    'forearm_fk.L': (14, 16),     # person's right elbow->wrist
    'thigh_fk.L': (24, 26),       # person's right hip->knee
    'shin_fk.L': (26, 28),        # person's right knee->ankle
}

IK_MAP = {
    'hand_ik.R': 15,   # person's left wrist
    'hand_ik.L': 16,   # person's right wrist
    'foot_ik.R': 27,   # person's left ankle
    'foot_ik.L': 28,   # person's right ankle
}

IK_FK_SWITCHES = {
    'upper_arm_parent.L': 'IK_FK',
    'upper_arm_parent.R': 'IK_FK',
    'thigh_parent.L': 'IK_FK',
    'thigh_parent.R': 'IK_FK',
}

# Virtual spine segments from MediaPipe landmarks
SPINE_MAP = {
    'spine_fk': ('hip_mid', 'spine_low'),
    'spine_fk.001': ('spine_low', 'spine_mid'),
    'spine_fk.002': ('spine_mid', 'neck_mid'),
    'spine_fk.003': ('neck_mid', 'shoulder_mid'),
}


def compute_spine_virtuals(lms):
    """Compute virtual spine landmarks from hip/shoulder midpoints."""
    v11 = get_lm(lms, 11)
    v12 = get_lm(lms, 12)
    v23 = get_lm(lms, 23)
    v24 = get_lm(lms, 24)
    if not all([v11, v12, v23, v24]):
        return lms

    hip_mid = (v23 + v24) / 2
    sh_mid = (v11 + v12) / 2
    spine_mid = (hip_mid + sh_mid) / 2
    spine_low = (hip_mid + spine_mid) / 2
    neck_mid = (spine_mid + sh_mid) / 2

    # Store as lists for consistency
    lms['hip_mid'] = list(hip_mid)
    lms['spine_low'] = list(spine_low)
    lms['spine_mid'] = list(spine_mid)
    lms['neck_mid'] = list(neck_mid)
    lms['shoulder_mid'] = list(sh_mid)
    return lms


# =============================================================================
# OPERATORS
# =============================================================================

class ANTIGRAV_OT_create_prop_empty(bpy.types.Operator):
    bl_idname = "antigrav.create_prop_empty"
    bl_label = "Create Prop Empty"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rig = context.object
        if not rig or rig.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the Armature first.")
            return {'CANCELLED'}

        bpy.ops.object.mode_set(mode='OBJECT')
        empty = bpy.data.objects.new("Prop_Anchor", None)
        empty.empty_display_type = 'CUBE'
        empty.empty_display_size = 0.1
        context.collection.objects.link(empty)

        active_bone = context.active_pose_bone
        if active_bone:
            empty.parent = rig
            empty.parent_type = 'BONE'
            empty.parent_bone = active_bone.name
            empty.location = (0, 0, 0)
            self.report({'INFO'}, f"Prop Empty parented to {active_bone.name}")
        else:
            self.report({'WARNING'}, "No bone selected, created world-space empty.")

        return {'FINISHED'}


class ANTIGRAV_OT_sanitize_rig(bpy.types.Operator):
    """Remove previous retargeting constraints from bones"""
    bl_idname = "antigrav.sanitize_rig"
    bl_label = "Sanitize Rig"

    def execute(self, context):
        rig = context.active_object
        if not rig or rig.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the Rigify Armature")
            return {'CANCELLED'}

        count = 0
        for bone in rig.pose.bones:
            if bone.name.startswith("DEF-") or "_fk" in bone.name:
                for con in list(bone.constraints):
                    if con.type in {'COPY_TRANSFORMS', 'COPY_LOCATION', 'COPY_ROTATION'} and not con.name.startswith("RIGIFY"):
                        bone.constraints.remove(con)
                        count += 1

        self.report({'INFO'}, f"Cleaned {count} constraints.")
        return {'FINISHED'}


class ANTIGRAV_OT_set_ik(bpy.types.Operator):
    bl_idname = "antigrav.set_ik"
    bl_label = "Set IK Mode"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rig = context.active_object
        if rig and rig.type == 'ARMATURE':
            for bone, prop in IK_FK_SWITCHES.items():
                if bone in rig.pose.bones and prop in rig.pose.bones[bone]:
                    rig.pose.bones[bone][prop] = 0.0
            self.report({'INFO'}, "IK mode set")
        return {'FINISHED'}


class ANTIGRAV_OT_set_fk(bpy.types.Operator):
    bl_idname = "antigrav.set_fk"
    bl_label = "Set FK Mode"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rig = context.active_object
        if rig and rig.type == 'ARMATURE':
            for bone, prop in IK_FK_SWITCHES.items():
                if bone in rig.pose.bones and prop in rig.pose.bones[bone]:
                    rig.pose.bones[bone][prop] = 1.0
            self.report({'INFO'}, "FK mode set")
        return {'FINISHED'}


class ANTIGRAV_OT_import_ik(bpy.types.Operator, ImportHelper):
    """Import mocap using IK targets with root motion"""
    bl_idname = "antigrav.import_ik"
    bl_label = "Import IK (.json)"

    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(default="*.json", options={'HIDDEN'})

    def execute(self, context):
        rig = context.active_object
        if not rig or rig.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the Rigify Armature")
            return {'CANCELLED'}

        take = load_take(self.filepath)
        frames = take['frames']
        if not frames:
            self.report({'ERROR'}, "No frames found")
            return {'CANCELLED'}

        pose_bones = rig.pose.bones
        world = rig.matrix_world

        # Set IK mode
        for bone, prop in IK_FK_SWITCHES.items():
            if bone in pose_bones and prop in pose_bones[bone]:
                pose_bones[bone][prop] = 0.0

        # Reference frame
        ref_lms = frames[0]['landmarks']
        ref_hip = get_hip_center(ref_lms)
        if not ref_hip:
            self.report({'ERROR'}, "No hip landmarks in first frame")
            return {'CANCELLED'}

        # Calculate scale from arm length
        l_shoulder = get_lm(ref_lms, 11)
        l_wrist = get_lm(ref_lms, 15)
        scale = 1.0
        if l_shoulder and l_wrist:
            person_arm = (l_wrist - l_shoulder).length
            char_arm = 0.52  # default Rigify arm length
            if 'upper_arm_fk.L' in pose_bones and 'hand_fk.L' in pose_bones:
                s = world @ pose_bones['upper_arm_fk.L'].bone.head_local
                w = world @ pose_bones['hand_fk.L'].bone.head_local
                char_arm = (w - s).length
            if person_arm > 0.01:
                scale = char_arm / person_arm
        print(f"[AntiGrav] IK import - scale: {scale:.3f}")

        # Reference limb positions relative to hip
        ref_limb_rel = {}
        for ik_bone, lm_idx in IK_MAP.items():
            lm = get_lm(ref_lms, lm_idx)
            if lm:
                ref_limb_rel[ik_bone] = lm - ref_hip

        # Create action
        bpy.ops.object.mode_set(mode='POSE')
        if not rig.animation_data:
            rig.animation_data_create()
        action = bpy.data.actions.new(name="AntiGrav_IK")
        rig.animation_data.action = action

        pin_threshold = context.scene.antigrav_pin_threshold
        prev_foot = {'.L': None, '.R': None}

        for fidx, fdata in enumerate(frames):
            f_idx = fidx + 1
            context.scene.frame_set(f_idx)
            lms = fdata['landmarks']

            current_hip = get_hip_center(lms)
            if not current_hip:
                continue

            # Root motion
            hip_delta = current_hip - ref_hip
            scaled_delta = hip_delta * scale

            if 'torso' in pose_bones:
                pose_bones['torso'].location = mp_to_blender(scaled_delta)
                pose_bones['torso'].keyframe_insert(data_path="location", frame=f_idx)

            # IK targets
            for ik_bone, lm_idx in IK_MAP.items():
                if ik_bone not in pose_bones or ik_bone not in ref_limb_rel:
                    continue

                lm = get_lm(lms, lm_idx)
                if not lm:
                    continue

                current_rel = lm - current_hip
                ref_rel = ref_limb_rel[ik_bone]
                limb_delta = (current_rel - ref_rel) * scale

                pos = mp_to_blender(limb_delta)

                # Foot pinning: if foot barely moved, keep it pinned
                if 'foot' in ik_bone:
                    side = '.L' if ik_bone.endswith('.L') else '.R'
                    if prev_foot[side] is not None:
                        if (pos - prev_foot[side]).length < pin_threshold:
                            pos = prev_foot[side]
                    prev_foot[side] = pos.copy()

                    # Ground clamp for feet
                    world_pos = lm * scale
                    if mp_to_blender(world_pos).z < 0:
                        pos.z = max(pos.z, 0)

                pose_bones[ik_bone].location = pos
                pose_bones[ik_bone].keyframe_insert(data_path="location", frame=f_idx)

        context.scene.frame_start = 1
        context.scene.frame_end = len(frames)
        self.report({'INFO'}, f"Imported {len(frames)} frames (IK)")
        return {'FINISHED'}


class ANTIGRAV_OT_import_fk(bpy.types.Operator, ImportHelper):
    """Import mocap using FK bone rotations with spine"""
    bl_idname = "antigrav.import_fk"
    bl_label = "Import FK (.json)"

    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(default="*.json", options={'HIDDEN'})

    def execute(self, context):
        rig = context.active_object
        if not rig or rig.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the Rigify Armature")
            return {'CANCELLED'}

        take = load_take(self.filepath)
        frames = take['frames']
        if not frames:
            self.report({'ERROR'}, "No frames found")
            return {'CANCELLED'}

        pose_bones = rig.pose.bones
        world = rig.matrix_world

        # Set FK mode
        for bone, prop in IK_FK_SWITCHES.items():
            if bone in pose_bones and prop in pose_bones[bone]:
                pose_bones[bone][prop] = 1.0

        # Reference
        ref_lms = frames[0]['landmarks']
        ref_hip = get_hip_center(ref_lms)
        if not ref_hip:
            self.report({'ERROR'}, "No hip landmarks in first frame")
            return {'CANCELLED'}

        # Scale
        l_shoulder = get_lm(ref_lms, 11)
        l_wrist = get_lm(ref_lms, 15)
        scale = 1.0
        if l_shoulder and l_wrist:
            person_arm = (l_wrist - l_shoulder).length
            char_arm = 0.52
            if 'upper_arm_fk.L' in pose_bones and 'hand_fk.L' in pose_bones:
                s = world @ pose_bones['upper_arm_fk.L'].bone.head_local
                w = world @ pose_bones['hand_fk.L'].bone.head_local
                char_arm = (w - s).length
            if person_arm > 0.01:
                scale = char_arm / person_arm
        print(f"[AntiGrav] FK import - scale: {scale:.3f}")

        # Get rest directions for each FK bone
        rest_dirs = {}
        for fk_bone in list(FK_MAP.keys()) + list(SPINE_MAP.keys()):
            if fk_bone in rig.data.bones:
                bone = rig.data.bones[fk_bone]
                head = world @ bone.head_local
                tail = world @ bone.tail_local
                rest_dirs[fk_bone] = (tail - head).normalized()

        # Spine rest axes point +Z typically
        spine_rest_axis = Vector((0, 0, 1))

        # Create action
        bpy.ops.object.mode_set(mode='POSE')
        if not rig.animation_data:
            rig.animation_data_create()
        action = bpy.data.actions.new(name="AntiGrav_FK")
        rig.animation_data.action = action

        for pb in pose_bones:
            pb.rotation_mode = 'QUATERNION'

        for fidx, fdata in enumerate(frames):
            f_idx = fidx + 1
            context.scene.frame_set(f_idx)
            lms = fdata['landmarks']

            current_hip = get_hip_center(lms)
            if not current_hip:
                continue

            # Compute spine virtuals
            lms = compute_spine_virtuals(lms)

            # Root motion (hip translation)
            hip_delta = current_hip - ref_hip
            scaled_delta = hip_delta * scale

            if 'torso' in pose_bones:
                pose_bones['torso'].location = mp_to_blender(scaled_delta)
                pose_bones['torso'].keyframe_insert(data_path="location", frame=f_idx)

            # Limb FK rotations
            for fk_bone, (start_lm, end_lm) in FK_MAP.items():
                if fk_bone not in pose_bones or fk_bone not in rest_dirs:
                    continue

                start = get_lm(lms, start_lm)
                end = get_lm(lms, end_lm)
                if not start or not end:
                    continue

                cur_dir = mp_to_blender((end - start).normalized()).normalized()
                rest_dir = rest_dirs[fk_bone]

                rotation = rest_dir.rotation_difference(cur_dir)

                pb = pose_bones[fk_bone]
                if pb.parent:
                    parent_rot = pb.parent.matrix.to_quaternion()
                    local_rot = parent_rot.inverted() @ rotation
                else:
                    arm_rot = world.to_quaternion()
                    local_rot = arm_rot.inverted() @ rotation

                pb.rotation_quaternion = local_rot
                pb.keyframe_insert(data_path="rotation_quaternion", frame=f_idx)

            # Spine FK rotations
            for spine_bone, (s_key, e_key) in SPINE_MAP.items():
                if spine_bone not in pose_bones:
                    continue

                s_val = lms.get(s_key)
                e_val = lms.get(e_key)
                if not s_val or not e_val:
                    continue

                s_vec = Vector(s_val) if isinstance(s_val, (list, tuple)) else s_val
                e_vec = Vector(e_val) if isinstance(e_val, (list, tuple)) else e_val

                cur_dir = mp_to_blender((e_vec - s_vec).normalized()).normalized()
                rest_dir = rest_dirs.get(spine_bone, spine_rest_axis)

                rotation = rest_dir.rotation_difference(cur_dir)

                pb = pose_bones[spine_bone]
                if pb.parent:
                    parent_rot = pb.parent.matrix.to_quaternion()
                    local_rot = parent_rot.inverted() @ rotation
                else:
                    arm_rot = world.to_quaternion()
                    local_rot = arm_rot.inverted() @ rotation

                pb.rotation_quaternion = local_rot
                pb.keyframe_insert(data_path="rotation_quaternion", frame=f_idx)

            # Finger FK (if hand data exists)
            hands_3d = fdata.get('hands_3d', [])
            for h_idx, hand_lms in enumerate(hands_3d):
                h_side = ".R" if h_idx == 0 else ".L"  # Mirrored: first hand is person's left

                finger_map = {
                    "thumb": [1, 2, 3, 4],
                    "index": [5, 6, 7, 8],
                    "middle": [9, 10, 11, 12],
                    "ring": [13, 14, 15, 16],
                    "pinky": [17, 18, 19, 20]
                }

                for f_name, mp_indices in finger_map.items():
                    for i, (s_idx, e_idx) in enumerate(zip(mp_indices[:-1], mp_indices[1:])):
                        bone_name = f"f_{f_name}.0{i+1}{h_side}"
                        if f_name == "thumb":
                            bone_name = f"thumb.0{i+1}{h_side}"

                        bone = rig.pose.bones.get(bone_name)
                        if bone and s_idx < len(hand_lms) and e_idx < len(hand_lms):
                            v_start = Vector(hand_lms[s_idx])
                            v_end = Vector(hand_lms[e_idx])
                            target_dir = mp_to_blender((v_end - v_start).normalized()).normalized()
                            bone.rotation_mode = 'QUATERNION'
                            bone.rotation_quaternion = Vector((0, 1, 0)).rotation_difference(
                                rig.matrix_world.inverted().to_quaternion() @ target_dir
                            )
                            bone.keyframe_insert(data_path="rotation_quaternion", frame=f_idx)

        context.scene.frame_start = 1
        context.scene.frame_end = len(frames)
        self.report({'INFO'}, f"Imported {len(frames)} frames (FK + Spine)")
        return {'FINISHED'}


class ANTIGRAV_OT_analyze(bpy.types.Operator, ImportHelper):
    """Analyze a take file for data quality"""
    bl_idname = "antigrav.analyze"
    bl_label = "Analyze Take"
    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(default="*.json", options={'HIDDEN'})

    def execute(self, context):
        take = load_take(self.filepath)
        frames = take['frames']

        print("=" * 60)
        print("TAKE ANALYSIS")
        print("=" * 60)
        print(f"Frames: {len(frames)}")
        print(f"Floor offset: {take['floor_offset']:.3f}m")

        # Check data quality
        empty_frames = sum(1 for f in frames if not f['landmarks'])
        print(f"Empty frames: {empty_frames}/{len(frames)}")

        if frames and frames[0]['landmarks']:
            lms = frames[0]['landmarks']
            hip = get_hip_center(lms)
            if hip:
                print(f"Hip center: ({hip.x:.3f}, {hip.y:.3f}, {hip.z:.3f})")

            # Check shoulder distance (should be ~0.35-0.45m)
            sl = get_lm(lms, 11)
            sr = get_lm(lms, 12)
            if sl and sr:
                sh_dist = (sl - sr).length
                print(f"Shoulder distance: {sh_dist:.3f}m (expected ~0.4m)")
                if sh_dist > 1.0:
                    print("  WARNING: Distance too large - possible calibration issue!")
                elif sh_dist < 0.1:
                    print("  WARNING: Distance too small - possible scale issue!")

            # Check foot Z (should be near 0 for standing)
            fl = get_lm(lms, 27)
            fr = get_lm(lms, 28)
            if fl and fr:
                avg_foot_z = (fl.z + fr.z) / 2
                print(f"Avg foot Z: {avg_foot_z:.3f}m (expected ~0 for standing)")

        self.report({'INFO'}, "Check console for analysis")
        return {'FINISHED'}


class ANTIGRAV_OT_clear(bpy.types.Operator):
    bl_idname = "antigrav.clear"
    bl_label = "Clear Animation"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rig = context.active_object
        if rig and rig.type == 'ARMATURE':
            if rig.animation_data:
                rig.animation_data.action = None
            for pb in rig.pose.bones:
                pb.location = Vector((0, 0, 0))
                pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
                pb.rotation_euler = Euler((0, 0, 0))
        self.report({'INFO'}, "Cleared animation")
        return {'FINISHED'}


# =============================================================================
# PANEL
# =============================================================================

class ANTIGRAV_PT_main_panel(bpy.types.Panel):
    bl_label = "AntiGrav V3.2"
    bl_idname = "ANTIGRAV_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'AntiGrav'

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)

        col.label(text="Step 1: Preparation")
        col.operator("antigrav.sanitize_rig", icon='BRUSH')

        col.separator()
        col.label(text="Step 2: Mode")
        row = col.row(align=True)
        row.operator("antigrav.set_ik", text="IK Mode")
        row.operator("antigrav.set_fk", text="FK Mode")

        col.separator()
        col.label(text="Step 3: Settings")
        col.prop(bpy.context.scene, "antigrav_pin_threshold", text="Foot Pin Sensitivity")

        col.separator()
        col.label(text="Step 4: Import")
        col.operator("antigrav.import_ik", icon='ANIM_DATA', text="Import IK")
        col.operator("antigrav.import_fk", icon='ANIM_DATA', text="Import FK + Spine")

        col.separator()
        col.label(text="Step 5: Tools")
        row = col.row(align=True)
        row.operator("antigrav.analyze", text="Analyze Take")
        row.operator("antigrav.clear", text="Clear")
        col.operator("antigrav.create_prop_empty", icon='EMPTY_AXIS', text="Prop Anchor")


# =============================================================================
# REGISTRATION
# =============================================================================

classes = [
    ANTIGRAV_OT_sanitize_rig,
    ANTIGRAV_OT_set_ik,
    ANTIGRAV_OT_set_fk,
    ANTIGRAV_OT_import_ik,
    ANTIGRAV_OT_import_fk,
    ANTIGRAV_OT_analyze,
    ANTIGRAV_OT_clear,
    ANTIGRAV_OT_create_prop_empty,
    ANTIGRAV_PT_main_panel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.antigrav_pin_threshold = bpy.props.FloatProperty(
        name="Pin Threshold", default=0.02, min=0.0, max=0.2,
        description="Foot velocity threshold for pinning (meters). Higher = stickier feet."
    )


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.antigrav_pin_threshold


if __name__ == "__main__":
    register()
