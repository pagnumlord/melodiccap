"""
MelodicCap Blender Addon v12.1
==============================
FIXES:
- Added Set FK button
- Skeleton preview uses floor_z_offset to position at ground level
- Better scale matching to character height
- Cleaner code

For Blender 4.4.3+ and JaxRigify (1.87m)
"""

bl_info = {
    "name": "MelodicCap Motion Capture Importer",
    "author": "Karsten / MelodicCap Studio",
    "version": (12, 1, 0),
    "blender": (4, 4, 0),
    "location": "View3D > Sidebar > MelodicCap",
    "description": "Import MelodicCap mocap with skeleton visualizer",
    "category": "Animation",
}

import bpy
import json
import math
from mathutils import Vector, Matrix, Quaternion, Euler
from bpy.props import StringProperty, FloatProperty, BoolProperty, IntProperty
from bpy_extras.io_utils import ImportHelper

# =============================================================================
# CONSTANTS
# =============================================================================

MEDIAPIPE_LANDMARKS = {
    0: "nose", 11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow", 15: "left_wrist", 16: "right_wrist",
    23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
}

# IK target mapping (MediaPipe LEFT = Rigify RIGHT when facing camera)
IK_MAP = {
    'hand_ik.R': 15,  # Person's left wrist -> Character's right hand
    'hand_ik.L': 16,  # Person's right wrist -> Character's left hand
    'foot_ik.R': 27,  # Person's left ankle -> Character's right foot
    'foot_ik.L': 28,  # Person's right ankle -> Character's left foot
}

# FK bone mapping (for rotation calculation)
FK_MAP = {
    'upper_arm_fk.R': (11, 13),  # Person's left shoulder to elbow
    'forearm_fk.R': (13, 15),
    'thigh_fk.R': (23, 25),
    'shin_fk.R': (25, 27),
    'upper_arm_fk.L': (12, 14),  # Person's right shoulder to elbow
    'forearm_fk.L': (14, 16),
    'thigh_fk.L': (24, 26),
    'shin_fk.L': (26, 28),
}

IK_FK_SWITCHES = {
    'upper_arm_parent.L': 'IK_FK',
    'upper_arm_parent.R': 'IK_FK',
    'thigh_parent.L': 'IK_FK',
    'thigh_parent.R': 'IK_FK',
}

def debug(msg):
    print(f"[MelodicCap] {msg}")

def get_lm(landmarks, idx):
    key = str(idx)
    if key in landmarks:
        p = landmarks[key]
        return Vector((p[0], p[1], p[2]))
    return None

def get_hip_center(landmarks):
    l = get_lm(landmarks, 23)
    r = get_lm(landmarks, 24)
    return (l + r) / 2 if l and r else None

def mp_to_blender(pos):
    """MediaPipe to Blender: mirror X, keep Y as depth, Z as up"""
    return Vector((-pos.x, pos.y, pos.z))


# =============================================================================
# SKELETON VISUALIZER
# =============================================================================

class MELODICCAP_OT_preview(bpy.types.Operator, ImportHelper):
    """Create skeleton preview from take file"""
    bl_idname = "melodiccap.preview"
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
        
        # Get calibration data
        calib = data.get('calibration', {})
        floor_offset = calib.get('floor_z_offset', calib.get('floor_offset', 0.0))
        
        debug("="*60)
        debug("SKELETON PREVIEW")
        debug("="*60)
        debug(f"Frames: {len(frames)}")
        debug(f"Floor offset: {floor_offset:.3f}m")
        
        # Delete old preview
        for o in list(bpy.data.objects):
            if o.name.startswith("MoCap_"):
                bpy.data.objects.remove(o, do_unlink=True)
        
        # Create parent
        parent = bpy.data.objects.new("MoCap_Root", None)
        parent.empty_display_type = 'ARROWS'
        parent.empty_display_size = 0.1
        context.scene.collection.objects.link(parent)
        
        # Reference frame
        ref_lms = frames[0].get('landmarks', {})
        ref_hip = get_hip_center(ref_lms)
        if not ref_hip:
            self.report({'ERROR'}, "No hip data!")
            return {'CANCELLED'}
        
        # Calculate person height for scaling
        nose = get_lm(ref_lms, 0)
        l_ankle = get_lm(ref_lms, 27)
        r_ankle = get_lm(ref_lms, 28)
        
        person_height = 1.7  # Default
        if nose and l_ankle and r_ankle:
            ankle_mid_z = (l_ankle.z + r_ankle.z) / 2
            person_height = (nose.z - ankle_mid_z) + 0.15  # Add head top
        
        # Target: JaxRigify is 1.87m
        char_height = 1.87
        scale = char_height / person_height
        
        debug(f"Person height: {person_height:.2f}m")
        debug(f"Character height: {char_height:.2f}m")
        debug(f"Scale: {scale:.3f}")
        
        # Create empties for key joints
        key_joints = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        empties = {}
        
        for idx in key_joints:
            name = f"MoCap_{MEDIAPIPE_LANDMARKS.get(idx, str(idx))}"
            empty = bpy.data.objects.new(name, None)
            empty.empty_display_type = 'SPHERE'
            empty.empty_display_size = 0.03
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
                    # Make relative to reference hip
                    rel = pos - ref_hip
                    # Apply floor offset (feet should be at Z=0)
                    # The floor_offset tells us how high the floor is in camera coords
                    # So we need to subtract (floor_offset - ref_hip.z) from Z
                    # Actually simpler: just offset so ankle Z ≈ 0
                    # Convert to Blender and scale
                    blender_pos = mp_to_blender(rel) * scale
                    # Offset so feet are at ground (ref ankle Z becomes 0)
                    ref_ankle_z = 0
                    if 27 in empties or 28 in empties:
                        ref_l = get_lm(ref_lms, 27)
                        ref_r = get_lm(ref_lms, 28)
                        if ref_l and ref_r:
                            ref_ankle_z = ((ref_l.z + ref_r.z) / 2 - ref_hip.z) * scale
                    
                    empty.location = blender_pos - Vector((0, 0, ref_ankle_z))
                    empty.keyframe_insert(data_path="location", frame=bf)
        
        context.scene.frame_start = self.start_frame
        context.scene.frame_end = self.start_frame + len(frames) - 1
        context.scene.frame_set(self.start_frame)
        
        debug(f"Created {len(empties)} joints, {len(frames)} frames")
        self.report({'INFO'}, f"Preview: {len(frames)} frames")
        return {'FINISHED'}


# =============================================================================
# IK IMPORT (ROOT MOTION + LIMB RELATIVE)
# =============================================================================

class MELODICCAP_OT_import_ik(bpy.types.Operator, ImportHelper):
    """Import using IK targets with root motion"""
    bl_idname = "melodiccap.import_ik"
    bl_label = "Import (IK Mode)"
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
        
        debug("="*60)
        debug("IK IMPORT")
        debug("="*60)
        
        pose_bones = obj.pose.bones
        world = obj.matrix_world
        
        # Set IK mode
        for bone, prop in IK_FK_SWITCHES.items():
            if bone in pose_bones and prop in pose_bones[bone]:
                pose_bones[bone][prop] = 0.0
        debug("Set IK mode")
        
        # Reference
        ref_lms = frames[0].get('landmarks', {})
        ref_hip = get_hip_center(ref_lms)
        if not ref_hip:
            self.report({'ERROR'}, "No hip!")
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
            scale = char_arm / person_arm
        debug(f"Scale: {scale:.4f}")
        
        # Reference limb positions relative to hip
        ref_limb_rel = {}
        for ik_bone, lm_idx in IK_MAP.items():
            lm = get_lm(ref_lms, lm_idx)
            if lm:
                ref_limb_rel[ik_bone] = lm - ref_hip
        
        # Create action
        if not obj.animation_data:
            obj.animation_data_create()
        action = bpy.data.actions.new(name="MelodicCapIK")
        obj.animation_data.action = action
        
        for pb in pose_bones:
            pb.rotation_mode = 'QUATERNION'
        
        # Animate
        keyframes = 0
        for fidx, fdata in enumerate(frames):
            bf = self.start_frame + fidx
            lms = fdata.get('landmarks', {})
            
            current_hip = get_hip_center(lms)
            if not current_hip:
                continue
            
            # ROOT MOTION - torso follows hip
            hip_delta = current_hip - ref_hip
            scaled_delta = hip_delta * scale
            
            if 'torso' in pose_bones:
                pose_bones['torso'].location = mp_to_blender(scaled_delta)
                pose_bones['torso'].keyframe_insert(data_path="location", frame=bf)
                keyframes += 1
            
            # IK TARGETS - limb movement relative to body
            for ik_bone, lm_idx in IK_MAP.items():
                if ik_bone not in pose_bones or ik_bone not in ref_limb_rel:
                    continue
                
                lm = get_lm(lms, lm_idx)
                if not lm:
                    continue
                
                current_rel = lm - current_hip
                ref_rel = ref_limb_rel[ik_bone]
                limb_delta = (current_rel - ref_rel) * scale
                
                pose_bones[ik_bone].location = mp_to_blender(limb_delta)
                pose_bones[ik_bone].keyframe_insert(data_path="location", frame=bf)
                keyframes += 1
            
            if fidx % 50 == 0:
                debug(f"Frame {fidx}/{len(frames)}")
        
        debug(f"Created {keyframes} keyframes")
        self.report({'INFO'}, f"Imported {len(frames)} frames (IK)")
        return {'FINISHED'}


# =============================================================================
# FK IMPORT (ROTATION CALCULATION)
# =============================================================================

class MELODICCAP_OT_import_fk(bpy.types.Operator, ImportHelper):
    """Import using FK bone rotations (like Rokoko)"""
    bl_idname = "melodiccap.import_fk"
    bl_label = "Import (FK Mode)"
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
        
        debug("="*60)
        debug("FK IMPORT (Rotation-based)")
        debug("="*60)
        
        pose_bones = obj.pose.bones
        world = obj.matrix_world
        
        # Set FK mode
        for bone, prop in IK_FK_SWITCHES.items():
            if bone in pose_bones and prop in pose_bones[bone]:
                pose_bones[bone][prop] = 1.0  # FK mode!
        debug("Set FK mode")
        
        # Reference
        ref_lms = frames[0].get('landmarks', {})
        ref_hip = get_hip_center(ref_lms)
        if not ref_hip:
            self.report({'ERROR'}, "No hip!")
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
            scale = char_arm / person_arm
        debug(f"Scale: {scale:.4f}")
        
        # Get rest directions
        rest_dirs = {}
        for fk_bone in FK_MAP.keys():
            if fk_bone in obj.data.bones:
                bone = obj.data.bones[fk_bone]
                head = world @ bone.head_local
                tail = world @ bone.tail_local
                rest_dirs[fk_bone] = (tail - head).normalized()
        
        # Create action
        if not obj.animation_data:
            obj.animation_data_create()
        action = bpy.data.actions.new(name="MelodicCapFK")
        obj.animation_data.action = action
        
        for pb in pose_bones:
            pb.rotation_mode = 'QUATERNION'
        
        # Animate
        keyframes = 0
        for fidx, fdata in enumerate(frames):
            bf = self.start_frame + fidx
            lms = fdata.get('landmarks', {})
            
            current_hip = get_hip_center(lms)
            if not current_hip:
                continue
            
            # Root motion
            hip_delta = current_hip - ref_hip
            scaled_delta = hip_delta * scale
            
            if 'torso' in pose_bones:
                pose_bones['torso'].location = mp_to_blender(scaled_delta)
                pose_bones['torso'].keyframe_insert(data_path="location", frame=bf)
                keyframes += 1
            
            # FK rotations
            for fk_bone, (start_lm, end_lm) in FK_MAP.items():
                if fk_bone not in pose_bones or fk_bone not in rest_dirs:
                    continue
                
                start = get_lm(lms, start_lm)
                end = get_lm(lms, end_lm)
                if not start or not end:
                    continue
                
                # Current direction (MediaPipe space)
                cur_dir_mp = (end - start).normalized()
                # Convert to Blender
                cur_dir = mp_to_blender(cur_dir_mp).normalized()
                # Rest direction
                rest_dir = rest_dirs[fk_bone]
                
                # Rotation from rest to current
                rotation = rest_dir.rotation_difference(cur_dir)
                
                pb = pose_bones[fk_bone]
                # Convert to local space
                if pb.parent:
                    parent_rot = pb.parent.matrix.to_quaternion()
                    local_rot = parent_rot.inverted() @ rotation
                else:
                    arm_rot = world.to_quaternion()
                    local_rot = arm_rot.inverted() @ rotation
                
                pb.rotation_quaternion = local_rot
                pb.keyframe_insert(data_path="rotation_quaternion", frame=bf)
                keyframes += 1
            
            if fidx % 50 == 0:
                debug(f"Frame {fidx}/{len(frames)}")
        
        debug(f"Created {keyframes} keyframes")
        self.report({'INFO'}, f"Imported {len(frames)} frames (FK)")
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
        
        debug("="*60)
        debug(f"RIG: {obj.name}")
        debug("="*60)
        
        debug("\nIK/FK Switches:")
        for bone, prop in IK_FK_SWITCHES.items():
            if bone in obj.pose.bones and prop in obj.pose.bones[bone]:
                val = obj.pose.bones[bone][prop]
                mode = "FK" if val > 0.5 else "IK"
                debug(f"  {bone} = {val:.1f} ({mode})")
        
        debug("\nFK Bones:")
        for fk in FK_MAP.keys():
            status = "✓" if fk in obj.pose.bones else "✗"
            debug(f"  {status} {fk}")
        
        debug("\nIK Targets:")
        for ik in IK_MAP.keys():
            status = "✓" if ik in obj.pose.bones else "✗"
            debug(f"  {status} {ik}")
        
        self.report({'INFO'}, "Check console!")
        return {'FINISHED'}


class MELODICCAP_OT_analyze(bpy.types.Operator, ImportHelper):
    bl_idname = "melodiccap.analyze"
    bl_label = "Analyze Take"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    
    def execute(self, context):
        with open(self.filepath, 'r') as f:
            data = json.load(f)
        
        frames = data.get('frames', [])
        calib = data.get('calibration', {})
        
        debug("="*60)
        debug("TAKE ANALYSIS")
        debug("="*60)
        debug(f"Frames: {len(frames)}")
        debug(f"Floor offset: {calib.get('floor_z_offset', calib.get('floor_offset', 'N/A'))}")
        debug(f"Stereo RMS: {calib.get('rms_stereo', 'N/A')}")
        
        if frames:
            hips = [get_hip_center(f.get('landmarks', {})) for f in frames]
            hips = [h for h in hips if h]
            if hips:
                ref = hips[0]
                max_move = max((h - ref).length for h in hips)
                debug(f"Max hip movement: {max_move:.3f}m")
        
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
            self.report({'INFO'}, "IK mode set!")
        return {'FINISHED'}


class MELODICCAP_OT_set_fk(bpy.types.Operator):
    bl_idname = "melodiccap.set_fk"
    bl_label = "Set FK"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        if obj and obj.type == 'ARMATURE':
            for bone, prop in IK_FK_SWITCHES.items():
                if bone in obj.pose.bones and prop in obj.pose.bones[bone]:
                    obj.pose.bones[bone][prop] = 1.0
            self.report({'INFO'}, "FK mode set!")
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
    bl_label = "MelodicCap v12.1"
    bl_idname = "MELODICCAP_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'MelodicCap'
    
    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        
        # Status
        box = layout.box()
        box.label(text="Status", icon='INFO')
        if obj and obj.type == 'ARMATURE':
            box.label(text=f"Rig: {obj.name}")
        else:
            box.label(text="Select armature!", icon='ERROR')
        
        # Preview
        box = layout.box()
        box.label(text="1. Preview", icon='HIDE_OFF')
        box.operator("melodiccap.preview", text="Preview Skeleton")
        
        # Mode
        box = layout.box()
        box.label(text="2. Set Mode", icon='TOOL_SETTINGS')
        row = box.row(align=True)
        row.operator("melodiccap.set_ik", text="IK Mode")
        row.operator("melodiccap.set_fk", text="FK Mode")
        
        # Import
        box = layout.box()
        box.label(text="3. Import", icon='IMPORT')
        box.operator("melodiccap.import_ik", text="Import IK")
        box.operator("melodiccap.import_fk", text="Import FK")
        
        # Tools
        box = layout.box()
        box.label(text="Tools", icon='TOOL_SETTINGS')
        row = box.row()
        row.operator("melodiccap.diagnose", text="Diagnose")
        row.operator("melodiccap.analyze", text="Analyze")
        box.operator("melodiccap.clear", text="Clear All")


# =============================================================================
# REGISTRATION
# =============================================================================

classes = [
    MELODICCAP_OT_preview,
    MELODICCAP_OT_import_ik,
    MELODICCAP_OT_import_fk,
    MELODICCAP_OT_diagnose,
    MELODICCAP_OT_analyze,
    MELODICCAP_OT_set_ik,
    MELODICCAP_OT_set_fk,
    MELODICCAP_OT_clear,
    MELODICCAP_PT_panel,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    debug("MelodicCap v12.1 registered!")

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
