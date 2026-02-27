"""
MelodicCap Blender Addon v1.0
=============================
Imports JSON motion capture data and retargets to JaxRigify armature.

Bone names verified against JaxRigify diagnostic dump:
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
    "name": "MelodicCap Importer",
    "author": "Karsten Allen",
    "version": (1, 0),
    "blender": (4, 4, 0),
    "location": "View3D > Sidebar > MelodicCap",
    "description": "Import MelodicCap JSON motion capture to JaxRigify",
    "category": "Animation",
}

import bpy
import json
import os
from mathutils import Vector, Quaternion, Matrix, Euler
from bpy_extras.io_utils import ImportHelper


# =============================================================================
# MEDIAPIPE LANDMARK INDICES
# =============================================================================

class MP:
    """MediaPipe pose landmark indices"""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# =============================================================================
# BONE MAPPING
# =============================================================================

# Vector-to-Rotation mapping: bone_name -> (start_landmark, end_landmark)
# The bone will be rotated to point from start to end

V2R_MAPPING = {
    # Arms
    "upper_arm_fk.L": (MP.LEFT_SHOULDER, MP.LEFT_ELBOW),
    "forearm_fk.L": (MP.LEFT_ELBOW, MP.LEFT_WRIST),
    "upper_arm_fk.R": (MP.RIGHT_SHOULDER, MP.RIGHT_ELBOW),
    "forearm_fk.R": (MP.RIGHT_ELBOW, MP.RIGHT_WRIST),
    
    # Legs
    "thigh_fk.L": (MP.LEFT_HIP, MP.LEFT_KNEE),
    "shin_fk.L": (MP.LEFT_KNEE, MP.LEFT_ANKLE),
    "foot_fk.L": (MP.LEFT_ANKLE, MP.LEFT_FOOT_INDEX),
    "thigh_fk.R": (MP.RIGHT_HIP, MP.RIGHT_KNEE),
    "shin_fk.R": (MP.RIGHT_KNEE, MP.RIGHT_ANKLE),
    "foot_fk.R": (MP.RIGHT_ANKLE, MP.RIGHT_FOOT_INDEX),
}

# IK position targets: bone_name -> landmark_index
IK_TARGETS = {
    "hand_ik.L": MP.LEFT_WRIST,
    "hand_ik.R": MP.RIGHT_WRIST,
    "foot_ik.L": MP.LEFT_ANKLE,
    "foot_ik.R": MP.RIGHT_ANKLE,
}

# Pole targets for IK chains
POLE_TARGETS = {
    "upper_arm_ik_target.L": (MP.LEFT_SHOULDER, MP.LEFT_ELBOW, MP.LEFT_WRIST),
    "upper_arm_ik_target.R": (MP.RIGHT_SHOULDER, MP.RIGHT_ELBOW, MP.RIGHT_WRIST),
    "thigh_ik_target.L": (MP.LEFT_HIP, MP.LEFT_KNEE, MP.LEFT_ANKLE),
    "thigh_ik_target.R": (MP.RIGHT_HIP, MP.RIGHT_KNEE, MP.RIGHT_ANKLE),
}

# Rest pose axes for each bone (direction bone points in rest pose, in armature space)
# These are determined by the Rigify rig structure
BONE_REST_AXES = {
    # Arms point outward from body
    "upper_arm_fk.L": Vector((1, 0, 0)),
    "forearm_fk.L": Vector((1, 0, 0)),
    "upper_arm_fk.R": Vector((-1, 0, 0)),
    "forearm_fk.R": Vector((-1, 0, 0)),
    
    # Legs point down
    "thigh_fk.L": Vector((0, 0, -1)),
    "shin_fk.L": Vector((0, 0, -1)),
    "foot_fk.L": Vector((0, 1, 0)),  # Feet point forward
    "thigh_fk.R": Vector((0, 0, -1)),
    "shin_fk.R": Vector((0, 0, -1)),
    "foot_fk.R": Vector((0, 1, 0)),
    
    # Spine points up
    "spine_fk": Vector((0, 0, 1)),
    "spine_fk.001": Vector((0, 0, 1)),
    "spine_fk.002": Vector((0, 0, 1)),
    "spine_fk.003": Vector((0, 0, 1)),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_landmark(landmarks_3d, idx):
    """Get a landmark as Vector, handling string keys"""
    key = str(idx)
    if key in landmarks_3d:
        return Vector(landmarks_3d[key])
    return None


def compute_midpoint(landmarks_3d, idx1, idx2):
    """Compute midpoint between two landmarks"""
    p1 = get_landmark(landmarks_3d, idx1)
    p2 = get_landmark(landmarks_3d, idx2)
    if p1 and p2:
        return (p1 + p2) / 2
    return None


def compute_virtual_spine_points(landmarks_3d):
    """
    Compute virtual points for spine chain.
    Returns dict with 'hip_mid', 'spine_low', 'spine_mid', 'chest', 'shoulder_mid'
    """
    hip_mid = compute_midpoint(landmarks_3d, MP.LEFT_HIP, MP.RIGHT_HIP)
    shoulder_mid = compute_midpoint(landmarks_3d, MP.LEFT_SHOULDER, MP.RIGHT_SHOULDER)
    
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
        'shoulder_mid': shoulder_mid
    }


def compute_rotation(rest_axis, target_direction):
    """
    Compute quaternion to rotate rest_axis to target_direction.
    Both should be normalized vectors.
    """
    rest_axis = rest_axis.normalized()
    target_direction = target_direction.normalized()
    return rest_axis.rotation_difference(target_direction)


def compute_pole_position(p_root, p_mid, p_end, offset=0.3):
    """
    Compute pole target position for IK.
    Projects the middle joint perpendicular to the root-end line.
    """
    line_dir = (p_end - p_root).normalized()
    proj_length = (p_mid - p_root).dot(line_dir)
    proj_point = p_root + line_dir * proj_length
    pole_dir = (p_mid - proj_point).normalized()
    return p_mid + pole_dir * offset


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
    
    # Options
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
        
        fps = data.get('fps', 30)
        
        # Switch to pose mode
        bpy.ops.object.mode_set(mode='POSE')
        
        # Set IK/FK sliders based on settings
        for bone in rig.pose.bones:
            if "IK_FK" in bone.keys():
                if self.use_ik:
                    bone["IK_FK"] = 0.0  # IK mode
                else:
                    bone["IK_FK"] = 1.0  # FK mode
        
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
            
            # Compute virtual spine points
            spine_points = compute_virtual_spine_points(landmarks_3d)
            
            # =====================
            # ROOT / TORSO POSITION
            # =====================
            torso = rig.pose.bones.get("torso")
            if torso and spine_points.get('hip_mid'):
                pos = spine_points['hip_mid']
                torso.location = rig.matrix_world.inverted() @ pos
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
                    
                    # Transform direction to armature local space
                    local_dir = rig.matrix_world.inverted().to_quaternion() @ direction
                    
                    bone.rotation_mode = 'QUATERNION'
                    bone.rotation_quaternion = compute_rotation(rest_axis, local_dir)
                    bone.keyframe_insert(data_path="rotation_quaternion")
                
                # Spine chain (using virtual points)
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
                    
                    # Foot-specific processing
                    is_foot = "foot" in bone_name
                    side = "L" if ".L" in bone_name else "R"
                    
                    if is_foot:
                        # Ground clamp
                        if self.ground_clamp and pos.z < 0:
                            pos.z = 0
                        
                        # Foot pinning
                        if self.pin_threshold > 0:
                            if prev_foot_pos[side] is not None:
                                velocity = (pos - prev_foot_pos[side]).length
                                
                                if velocity < self.pin_threshold and pos.z < 0.1:
                                    # Pin the foot
                                    if pinned_foot_pos[side] is None:
                                        pinned_foot_pos[side] = pos.copy()
                                        pinned_foot_pos[side].z = 0
                                    pos = pinned_foot_pos[side]
                                else:
                                    # Unpin
                                    pinned_foot_pos[side] = None
                            
                            prev_foot_pos[side] = pos.copy()
                    
                    # Apply position
                    bone.location = rig.matrix_world.inverted() @ pos
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
                    bone.location = rig.matrix_world.inverted() @ pole_pos
                    bone.keyframe_insert(data_path="location")
        
        self.report({'INFO'}, f"Imported {len(frames)} frames from {os.path.basename(self.filepath)}")
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
        
        # Reset pose
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
    bl_label = "MelodicCap"
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
            
            # Show IK/FK state for one limb
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
