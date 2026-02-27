"""
MelodicCap Blender Addon v12.0
==============================
SKELETON PREVIEW + FULL RIGIFY MAPPING

Based on Rokoko/professional retargeting:
1. PREVIEW MODE - See raw mocap skeleton as empties
2. FK ROTATIONS - Calculate bone rotations from positions
3. IK POSITIONS - Hand/foot target positions
4. ROOT MOTION - Torso translation + hips rotation

Rokoko bone mapping insight:
- Map to CONTROL bones only (not DEF/MCH/ORG)
- FK bones get ROTATION
- IK bones get POSITION  
- Root (hips) gets TRANSLATION + ROTATION

For Blender 4.4.3+ and JaxRigify (1.87m)
"""

bl_info = {
    "name": "MelodicCap Motion Capture Importer",
    "author": "Karsten / MelodicCap Studio",
    "version": (12, 0, 0),
    "blender": (4, 4, 0),
    "location": "View3D > Sidebar > MelodicCap",
    "description": "Import MelodicCap mocap with skeleton preview",
    "category": "Animation",
}

import bpy
import json
import math
from mathutils import Vector, Matrix, Quaternion, Euler
from bpy.props import StringProperty, FloatProperty, BoolProperty, IntProperty
from bpy_extras.io_utils import ImportHelper

# =============================================================================
# MEDIAPIPE LANDMARK INDICES
# =============================================================================

LANDMARKS = {
    0: "nose",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
}

# Key landmarks for skeleton visualization
SKELETON_LANDMARKS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# IK targets that get POSITION
# MediaPipe LEFT = person's left = Rigify RIGHT (mirror)
IK_POSITION_MAP = {
    'hand_ik.R': 15,   # Person's left wrist
    'hand_ik.L': 16,   # Person's right wrist
    'foot_ik.R': 27,   # Person's left ankle
    'foot_ik.L': 28,   # Person's right ankle
}

# IK/FK switches
IK_FK_SWITCHES = {
    'upper_arm_parent.L': 'IK_FK',
    'upper_arm_parent.R': 'IK_FK',
    'thigh_parent.L': 'IK_FK',
    'thigh_parent.R': 'IK_FK',
}

def debug(msg):
    print(f"[INFO] MelodicCap: {msg}")

def get_lm(landmarks, idx):
    """Get landmark as Vector"""
    key = str(idx)
    if key in landmarks:
        p = landmarks[key]
        return Vector((p[0], p[1], p[2]))
    return None

def get_hip_center(landmarks):
    """Get hip center"""
    l = get_lm(landmarks, 23)
    r = get_lm(landmarks, 24)
    if l and r:
        return (l + r) / 2
    return None


# =============================================================================
# SKELETON PREVIEW OPERATOR
# =============================================================================

class MELODICCAP_OT_preview_skeleton(bpy.types.Operator, ImportHelper):
    """Create skeleton preview from take file (empties)"""
    bl_idname = "melodiccap.preview_skeleton"
    bl_label = "Preview Skeleton"
    bl_options = {'REGISTER', 'UNDO'}
    
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    start_frame: IntProperty(name="Start Frame", default=1, min=1)
    
    def execute(self, context):
        with open(self.filepath, 'r') as f:
            data = json.load(f)
        
        frames = data.get('frames', [])
        if not frames:
            self.report({'ERROR'}, "No frames!")
            return {'CANCELLED'}
        
        debug("\n" + "="*70)
        debug("SKELETON PREVIEW")
        debug("="*70)
        
        # Delete existing preview
        for obj in list(bpy.data.objects):
            if obj.name.startswith("MoCap_"):
                bpy.data.objects.remove(obj, do_unlink=True)
        
        ref_lms = frames[0].get('landmarks', {})
        ref_hip = get_hip_center(ref_lms)
        
        if not ref_hip:
            self.report({'ERROR'}, "No hip data!")
            return {'CANCELLED'}
        
        # Create parent
        parent = bpy.data.objects.new("MoCap_Skeleton", None)
        parent.empty_display_type = 'ARROWS'
        parent.empty_display_size = 0.1
        context.scene.collection.objects.link(parent)
        
        # Create empties for landmarks
        empties = {}
        colors = {
            0: (1, 1, 0),      # Head - yellow
            11: (1, 0.5, 0), 12: (0, 0.5, 1),  # Shoulders
            13: (1, 0.3, 0), 14: (0, 0.3, 1),  # Elbows
            15: (1, 0, 0), 16: (0, 0, 1),      # Wrists - red/blue
            23: (1, 0.5, 0), 24: (0, 0.5, 1),  # Hips
            25: (1, 0.3, 0), 26: (0, 0.3, 1),  # Knees
            27: (1, 0, 0), 28: (0, 0, 1),      # Ankles
        }
        
        for idx in SKELETON_LANDMARKS:
            name = f"MoCap_{LANDMARKS[idx]}"
            empty = bpy.data.objects.new(name, None)
            empty.empty_display_type = 'SPHERE'
            empty.empty_display_size = 0.04 if idx == 0 else 0.03
            empty.parent = parent
            context.scene.collection.objects.link(empty)
            empties[idx] = empty
        
        # Animate
        for fidx, fdata in enumerate(frames):
            bf = self.start_frame + fidx
            lms = fdata.get('landmarks', {})
            
            for idx, empty in empties.items():
                pos = get_lm(lms, idx)
                if pos:
                    # Relative to reference hip, mirror X
                    rel = pos - ref_hip
                    empty.location = Vector((-rel.x, rel.y, rel.z))
                    empty.keyframe_insert(data_path="location", frame=bf)
        
        debug(f"Created {len(empties)} empties, {len(frames)} frames")
        debug("Press PLAY to see the skeleton move!")
        
        context.scene.frame_start = self.start_frame
        context.scene.frame_end = self.start_frame + len(frames) - 1
        context.scene.frame_set(self.start_frame)
        
        self.report({'INFO'}, f"Created skeleton preview!")
        return {'FINISHED'}


# =============================================================================
# MAIN IMPORT OPERATOR
# =============================================================================

class MELODICCAP_OT_import(bpy.types.Operator, ImportHelper):
    """Import with Rigify mapping"""
    bl_idname = "melodiccap.import_take"
    bl_label = "Import Take"
    bl_options = {'REGISTER', 'UNDO'}
    
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    start_frame: IntProperty(name="Start Frame", default=1, min=1)
    
    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature!")
            return {'CANCELLED'}
        
        with open(self.filepath, 'r') as f:
            data = json.load(f)
        
        frames = data.get('frames', [])
        if not frames:
            self.report({'ERROR'}, "No frames!")
            return {'CANCELLED'}
        
        debug("\n" + "="*70)
        debug("MELODICCAP v12.0 IMPORT")
        debug("="*70)
        
        pose_bones = obj.pose.bones
        world = obj.matrix_world
        
        # Reference frame
        ref_lms = frames[0].get('landmarks', {})
        ref_hip = get_hip_center(ref_lms)
        
        if not ref_hip:
            self.report({'ERROR'}, "No hip!")
            return {'CANCELLED'}
        
        debug(f"Reference hip: ({ref_hip.x:.3f}, {ref_hip.y:.3f}, {ref_hip.z:.3f})")
        
        # Scale
        scale = 1.0
        l_shoulder = get_lm(ref_lms, 11)
        l_wrist = get_lm(ref_lms, 15)
        if l_shoulder and l_wrist:
            person_arm = (l_wrist - l_shoulder).length
            if 'upper_arm_fk.L' in pose_bones and 'hand_fk.L' in pose_bones:
                s = world @ pose_bones['upper_arm_fk.L'].bone.head_local
                w = world @ pose_bones['hand_fk.L'].bone.head_local
                char_arm = (w - s).length
                scale = char_arm / person_arm
        debug(f"Scale: {scale:.4f}")
        
        # Reference angles
        ref_hip_angle = 0.0
        l_hip = get_lm(ref_lms, 23)
        r_hip = get_lm(ref_lms, 24)
        if l_hip and r_hip:
            hip_vec = l_hip - r_hip
            ref_hip_angle = math.atan2(hip_vec.y, hip_vec.x)
        
        # Reference limb positions
        ref_limb_rel = {}
        for ik_bone, lm_idx in IK_POSITION_MAP.items():
            lm = get_lm(ref_lms, lm_idx)
            if lm:
                ref_limb_rel[ik_bone] = lm - ref_hip
        
        # Set IK mode
        for bone_name, prop_name in IK_FK_SWITCHES.items():
            if bone_name in pose_bones and prop_name in pose_bones[bone_name]:
                pose_bones[bone_name][prop_name] = 0.0
        
        # Rotation mode
        for pb in pose_bones:
            pb.rotation_mode = 'QUATERNION'
        
        # Action
        if not obj.animation_data:
            obj.animation_data_create()
        action = bpy.data.actions.new(name="MelodicCapAction")
        obj.animation_data.action = action
        
        # Animate
        keyframes = 0
        bones_animated = set()
        
        for fidx, fdata in enumerate(frames):
            bf = self.start_frame + fidx
            lms = fdata.get('landmarks', {})
            
            current_hip = get_hip_center(lms)
            if not current_hip:
                continue
            
            # ROOT MOTION (torso)
            hip_delta = current_hip - ref_hip
            scaled_delta = hip_delta * scale
            
            if 'torso' in pose_bones:
                torso = pose_bones['torso']
                torso.location = Vector((-scaled_delta.x, scaled_delta.y, scaled_delta.z))
                torso.keyframe_insert(data_path="location", frame=bf)
                keyframes += 1
                bones_animated.add('torso')
            
            # HIP ROTATION
            l_hip = get_lm(lms, 23)
            r_hip = get_lm(lms, 24)
            if l_hip and r_hip and 'hips' in pose_bones:
                hip_vec = l_hip - r_hip
                current_angle = math.atan2(hip_vec.y, hip_vec.x)
                rotation = current_angle - ref_hip_angle
                
                hips = pose_bones['hips']
                hips.rotation_quaternion = Quaternion((0, 0, 1), rotation)
                hips.keyframe_insert(data_path="rotation_quaternion", frame=bf)
                keyframes += 1
                bones_animated.add('hips')
            
            # IK TARGETS
            for ik_bone, lm_idx in IK_POSITION_MAP.items():
                if ik_bone not in pose_bones or ik_bone not in ref_limb_rel:
                    continue
                
                lm = get_lm(lms, lm_idx)
                if not lm:
                    continue
                
                current_rel = lm - current_hip
                ref_rel = ref_limb_rel[ik_bone]
                limb_delta = (current_rel - ref_rel) * scale
                
                pb = pose_bones[ik_bone]
                pb.location = Vector((-limb_delta.x, limb_delta.y, limb_delta.z))
                pb.keyframe_insert(data_path="location", frame=bf)
                keyframes += 1
                bones_animated.add(ik_bone)
            
            if fidx % 50 == 0:
                debug(f"Frame {fidx}/{len(frames)}")
        
        debug("\n" + "="*70)
        debug("SUMMARY")
        debug("="*70)
        debug(f"Frames: {len(frames)}, Keyframes: {keyframes}")
        debug(f"Bones: {sorted(bones_animated)}")
        
        self.report({'INFO'}, f"Imported {len(frames)} frames!")
        return {'FINISHED'}


# =============================================================================
# UTILITY OPERATORS
# =============================================================================

class MELODICCAP_OT_diagnose(bpy.types.Operator):
    bl_idname = "melodiccap.diagnose"
    bl_label = "Diagnose"
    
    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select armature!")
            return {'CANCELLED'}
        
        debug("\n" + "="*70)
        debug(f"RIG: {obj.name}, Scale: {obj.scale}")
        debug("="*70)
        
        for bone, prop in IK_FK_SWITCHES.items():
            if bone in obj.pose.bones and prop in obj.pose.bones[bone]:
                val = obj.pose.bones[bone][prop]
                debug(f"{bone}.{prop} = {val}")
        
        self.report({'INFO'}, "Check console!")
        return {'FINISHED'}


class MELODICCAP_OT_analyze(bpy.types.Operator, ImportHelper):
    bl_idname = "melodiccap.analyze"
    bl_label = "Analyze"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    
    def execute(self, context):
        with open(self.filepath, 'r') as f:
            data = json.load(f)
        
        frames = data.get('frames', [])
        debug(f"\nTAKE: {len(frames)} frames")
        
        if frames:
            ref_hip = get_hip_center(frames[0].get('landmarks', {}))
            if ref_hip:
                max_move = max((get_hip_center(f.get('landmarks', {})) - ref_hip).length 
                               for f in frames if get_hip_center(f.get('landmarks', {})))
                debug(f"Max movement: {max_move:.3f}m")
        
        self.report({'INFO'}, "Check console!")
        return {'FINISHED'}


class MELODICCAP_OT_set_ik(bpy.types.Operator):
    bl_idname = "melodiccap.set_ik"
    bl_label = "Set IK"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        if obj and obj.type == 'ARMATURE':
            for bone, prop in IK_FK_SWITCHES.items():
                if bone in obj.pose.bones and prop in obj.pose.bones[bone]:
                    obj.pose.bones[bone][prop] = 0.0
        self.report({'INFO'}, "IK mode!")
        return {'FINISHED'}


class MELODICCAP_OT_clear(bpy.types.Operator):
    bl_idname = "melodiccap.clear"
    bl_label = "Clear"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        if obj and obj.type == 'ARMATURE':
            if obj.animation_data:
                obj.animation_data.action = None
            for pb in obj.pose.bones:
                pb.location = Vector((0, 0, 0))
                pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
                pb.rotation_euler = Euler((0, 0, 0))
        
        for o in list(bpy.data.objects):
            if o.name.startswith("MoCap_"):
                bpy.data.objects.remove(o, do_unlink=True)
        
        self.report({'INFO'}, "Cleared!")
        return {'FINISHED'}


# =============================================================================
# PANEL
# =============================================================================

class MELODICCAP_PT_panel(bpy.types.Panel):
    bl_label = "MelodicCap v12"
    bl_idname = "MELODICCAP_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'MelodicCap'
    
    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        
        box = layout.box()
        box.label(text="Status", icon='INFO')
        if obj and obj.type == 'ARMATURE':
            box.label(text=f"Rig: {obj.name}")
        else:
            box.label(text="Select armature!", icon='ERROR')
        
        box = layout.box()
        box.label(text="👁️ Preview", icon='HIDE_OFF')
        box.operator("melodiccap.preview_skeleton", text="Preview Skeleton", icon='OUTLINER_OB_ARMATURE')
        box.label(text="See raw mocap before import!")
        
        box = layout.box()
        box.label(text="🔍 Tools", icon='TOOL_SETTINGS')
        row = box.row()
        row.operator("melodiccap.diagnose", text="Diagnose")
        row.operator("melodiccap.analyze", text="Analyze")
        row = box.row()
        row.operator("melodiccap.set_ik", text="Set IK")
        row.operator("melodiccap.clear", text="Clear")
        
        box = layout.box()
        box.label(text="📥 Import", icon='IMPORT')
        box.operator("melodiccap.import_take", text="Import Take", icon='ARMATURE_DATA')


# =============================================================================
# REGISTRATION
# =============================================================================

classes = [
    MELODICCAP_OT_preview_skeleton,
    MELODICCAP_OT_import,
    MELODICCAP_OT_diagnose,
    MELODICCAP_OT_analyze,
    MELODICCAP_OT_set_ik,
    MELODICCAP_OT_clear,
    MELODICCAP_PT_panel,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    debug("MelodicCap v12.0 - Skeleton Preview + Hip Rotation!")

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
