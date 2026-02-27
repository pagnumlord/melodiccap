"""
MelodicCap Blender Addon v9.0
=============================
FIXES FROM v8 (based on Rigify hierarchy analysis):
- Torso: Translation ONLY (no rotation!)
- Hips: Gets the hip Z-rotation  
- Chest: Gets the spine twist (shoulder - hip rotation difference)
- spine_fk: NOT animated (MCH-spine has COPY_TRANSFORMS from hips, so it auto-follows!)
- IK Targets: World space position deltas (they're NOT children of torso!)

RIGIFY HIERARCHY (verified from JaxRigify dump):
  torso (master) - children: hips, chest, MCH-spine.001
  MCH-hand_ik.parent.L (Parent: None) → hand_ik.L  ← NOT child of torso!
  MCH-foot_ik.parent.L (Parent: None) → foot_ik.L  ← NOT child of torso!
  MCH-spine has COPY_TRANSFORMS from hips → spine_fk follows hips automatically

For Blender 4.4.3+ and JaxRigify (1.87m)
"""

bl_info = {
    "name": "MelodicCap Motion Capture Importer",
    "author": "Karsten / MelodicCap Studio",
    "version": (9, 0, 0),
    "blender": (4, 4, 0),
    "location": "View3D > Sidebar > MelodicCap",
    "description": "Import MelodicCap mocap data to Rigify",
    "category": "Animation",
}

import bpy
import json
import math
from mathutils import Vector, Matrix, Quaternion, Euler
from bpy.props import StringProperty, FloatProperty, BoolProperty, IntProperty
from bpy_extras.io_utils import ImportHelper

# =============================================================================
# LANDMARK INDICES
# =============================================================================

LM_NOSE = 0
LM_LEFT_SHOULDER = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_ELBOW = 13
LM_RIGHT_ELBOW = 14
LM_LEFT_WRIST = 15
LM_RIGHT_WRIST = 16
LM_LEFT_HIP = 23
LM_RIGHT_HIP = 24
LM_LEFT_KNEE = 25
LM_RIGHT_KNEE = 26
LM_LEFT_ANKLE = 27
LM_RIGHT_ANKLE = 28

# =============================================================================
# BONE MAPPING (Person's LEFT -> Blender RIGHT due to mirror)
# =============================================================================

IK_CONFIG = {
    'hand_ik.R': {
        'endpoint': LM_LEFT_WRIST,
        'root': LM_LEFT_SHOULDER,
        'max_length_bones': ['upper_arm_fk.R', 'forearm_fk.R'],
    },
    'hand_ik.L': {
        'endpoint': LM_RIGHT_WRIST,
        'root': LM_RIGHT_SHOULDER,
        'max_length_bones': ['upper_arm_fk.L', 'forearm_fk.L'],
    },
    'foot_ik.R': {
        'endpoint': LM_LEFT_ANKLE,
        'root': LM_LEFT_HIP,
        'max_length_bones': ['thigh_fk.R', 'shin_fk.R'],
    },
    'foot_ik.L': {
        'endpoint': LM_RIGHT_ANKLE,
        'root': LM_RIGHT_HIP,
        'max_length_bones': ['thigh_fk.L', 'shin_fk.L'],
    },
}

IK_FK_SWITCHES = {
    'upper_arm_parent.L': 'IK_FK',
    'upper_arm_parent.R': 'IK_FK',
    'thigh_parent.L': 'IK_FK',
    'thigh_parent.R': 'IK_FK',
}

# =============================================================================
# UTILITIES
# =============================================================================

def debug(msg, level="INFO"):
    print(f"[{level}] MelodicCap: {msg}")

def get_lm(landmarks, idx):
    key = str(idx) if str(idx) in landmarks else idx
    if key in landmarks:
        p = landmarks[key]
        return Vector((p[0], p[1], p[2]))
    return None

def get_mid(landmarks, i1, i2):
    p1, p2 = get_lm(landmarks, i1), get_lm(landmarks, i2)
    return (p1 + p2) / 2 if p1 and p2 else None

def clamp_length(vec, max_len):
    if vec.length > max_len and vec.length > 0.0001:
        return vec * (max_len / vec.length)
    return vec

def calculate_z_rotation(l_pos, r_pos):
    """Calculate Z-axis rotation (yaw) from left/right positions."""
    vec = r_pos - l_pos  # Vector from left to right
    # Facing direction is perpendicular to this
    # atan2 of the perpendicular gives us the facing angle
    angle = math.atan2(-vec.y, -vec.x)
    return angle

# =============================================================================
# IMPORTER
# =============================================================================

class MelodicCapImporter:
    
    def __init__(self, armature, take_data, settings):
        self.armature = armature
        self.take_data = take_data
        self.settings = settings
        
        # Reference data
        self.ref_landmarks = None
        self.ref_hip = None
        self.ref_hip_rotation = 0.0
        self.ref_shoulder_rotation = 0.0
        
        # Character data
        self.char_max_lengths = {}
        
        # Reference limb positions (relative to hip, in mocap space)
        self.ref_limb_positions = {}
        
        # Scale
        self.scale = 1.0
        
        self.stats = {'frames': 0, 'keys': 0, 'bones': set()}
    
    def analyze(self):
        """Analyze character and mocap data"""
        debug("="*60)
        debug("MELODICCAP v9.0 - RIGIFY HIERARCHY FIX")
        debug("="*60)
        
        bones = self.armature.data.bones
        pose_bones = self.armature.pose.bones
        world = self.armature.matrix_world
        
        # === CHARACTER ANALYSIS ===
        debug(f"\n  CHARACTER ANALYSIS:")
        
        # Check Rigify bone hierarchy
        debug(f"    Checking Rigify hierarchy:")
        for bone_name in ['torso', 'hips', 'spine_fk', 'spine_fk.001', 'chest', 'head']:
            if bone_name in bones:
                debug(f"      ✓ {bone_name}")
            else:
                # Try alternate names
                alt_names = {'spine_fk': 'spine', 'spine_fk.001': 'spine.001'}
                alt = alt_names.get(bone_name)
                if alt and alt in bones:
                    debug(f"      ✓ {alt} (as {bone_name})")
                else:
                    debug(f"      ✗ {bone_name}")
        
        # Max limb lengths
        for ik_bone, config in IK_CONFIG.items():
            total_length = 0.0
            for bone_name in config['max_length_bones']:
                if bone_name in bones:
                    bone = bones[bone_name]
                    head = world @ bone.head_local
                    tail = world @ bone.tail_local
                    total_length += (tail - head).length
            self.char_max_lengths[ik_bone] = total_length
            debug(f"    {ik_bone} max length: {total_length:.3f}m")
        
        # Character height
        min_z, max_z = float('inf'), float('-inf')
        for bone in bones:
            h = (world @ bone.head_local).z
            t = (world @ bone.tail_local).z
            min_z, max_z = min(min_z, h, t), max(max_z, h, t)
        char_height = max_z - min_z
        debug(f"    Character height: {char_height:.3f}m")
        
        # === MOCAP ANALYSIS ===
        frames = self.take_data.get('frames', [])
        if not frames:
            debug("ERROR: No frames!", "ERROR")
            return False
        
        debug(f"\n  MOCAP ANALYSIS:")
        debug(f"    Frames: {len(frames)}")
        
        # Reference frame
        self.ref_landmarks = frames[0].get('landmarks', {})
        self.ref_hip = get_mid(self.ref_landmarks, LM_LEFT_HIP, LM_RIGHT_HIP)
        
        if not self.ref_hip:
            debug("ERROR: No hip!", "ERROR")
            return False
        
        # Scale
        nose = get_lm(self.ref_landmarks, LM_NOSE)
        ankle_l = get_lm(self.ref_landmarks, LM_LEFT_ANKLE)
        ankle_r = get_lm(self.ref_landmarks, LM_RIGHT_ANKLE)
        
        if nose and ankle_l and ankle_r:
            ankle_mid = (ankle_l + ankle_r) / 2
            mocap_height = (nose.z - ankle_mid.z) + 0.15
            self.scale = char_height / mocap_height
            debug(f"    Scale factor: {self.scale:.4f}")
        
        # Reference body orientation
        l_shoulder = get_lm(self.ref_landmarks, LM_LEFT_SHOULDER)
        r_shoulder = get_lm(self.ref_landmarks, LM_RIGHT_SHOULDER)
        l_hip = get_lm(self.ref_landmarks, LM_LEFT_HIP)
        r_hip = get_lm(self.ref_landmarks, LM_RIGHT_HIP)
        
        if l_hip and r_hip:
            self.ref_hip_rotation = calculate_z_rotation(l_hip, r_hip)
            debug(f"\n  REFERENCE ORIENTATION:")
            debug(f"    Hip Z-rotation: {math.degrees(self.ref_hip_rotation):.1f}°")
        
        if l_shoulder and r_shoulder:
            self.ref_shoulder_rotation = calculate_z_rotation(l_shoulder, r_shoulder)
            debug(f"    Shoulder Z-rotation: {math.degrees(self.ref_shoulder_rotation):.1f}°")
            debug(f"    Initial spine twist: {math.degrees(self.ref_shoulder_rotation - self.ref_hip_rotation):.1f}°")
        
        # Reference limb positions (relative to hip)
        debug(f"\n  REFERENCE LIMB POSITIONS (relative to hip):")
        for ik_bone, config in IK_CONFIG.items():
            endpoint = get_lm(self.ref_landmarks, config['endpoint'])
            if endpoint:
                rel_pos = endpoint - self.ref_hip
                self.ref_limb_positions[ik_bone] = rel_pos.copy()
                debug(f"    {ik_bone}: ({rel_pos.x:.3f}, {rel_pos.y:.3f}, {rel_pos.z:.3f})")
        
        return True
    
    def set_ik_mode(self):
        pose_bones = self.armature.pose.bones
        for switch_bone, prop_name in IK_FK_SWITCHES.items():
            if switch_bone in pose_bones:
                pb = pose_bones[switch_bone]
                if prop_name in pb:
                    pb[prop_name] = 0.0
    
    def apply_animation(self):
        """Apply animation with proper Rigify hierarchy"""
        debug("\n" + "="*60)
        debug("APPLYING ANIMATION (Rigify Hierarchy)")
        debug("="*60)
        debug("  Strategy (based on JaxRigify bone dump):")
        debug("    - torso: Translation ONLY (IK targets are NOT its children!)")
        debug("    - hips: Hip Z-rotation (spine_fk auto-follows via MCH)")
        debug("    - chest: Spine twist (shoulder-hip rotation difference)")
        debug("    - spine_fk: NOT animated (MCH-spine COPY_TRANSFORMS from hips)")
        debug("    - IK targets: World-space position deltas")
        
        frames = self.take_data.get('frames', [])
        pose_bones = self.armature.pose.bones
        start = self.settings.get('start_frame', 1)
        
        self.set_ik_mode()
        
        for pb in pose_bones:
            pb.rotation_mode = 'QUATERNION'
        
        avail_ik = {b: c for b, c in IK_CONFIG.items() if b in pose_bones}
        
        has_torso = 'torso' in pose_bones
        has_hips = 'hips' in pose_bones
        has_chest = 'chest' in pose_bones
        
        debug(f"\n  Available bones:")
        debug(f"    torso: {has_torso}")
        debug(f"    hips: {has_hips}")
        debug(f"    chest: {has_chest}")
        debug(f"    IK targets: {list(avail_ik.keys())}")
        
        debug(f"\n  Processing {len(frames)} frames...")
        
        for fidx, fdata in enumerate(frames):
            bf = start + fidx
            bpy.context.scene.frame_set(bf)
            
            lms = fdata.get('landmarks', {})
            
            # Get positions
            hip = get_mid(lms, LM_LEFT_HIP, LM_RIGHT_HIP)
            l_shoulder = get_lm(lms, LM_LEFT_SHOULDER)
            r_shoulder = get_lm(lms, LM_RIGHT_SHOULDER)
            l_hip = get_lm(lms, LM_LEFT_HIP)
            r_hip = get_lm(lms, LM_RIGHT_HIP)
            
            if not hip or not l_hip or not r_hip:
                continue
            
            # === CALCULATE ROTATIONS ===
            current_hip_rot = calculate_z_rotation(l_hip, r_hip)
            hip_rotation_delta = current_hip_rot - self.ref_hip_rotation
            
            shoulder_rotation_delta = 0.0
            if l_shoulder and r_shoulder:
                current_shoulder_rot = calculate_z_rotation(l_shoulder, r_shoulder)
                shoulder_rotation_delta = current_shoulder_rot - self.ref_shoulder_rotation
            
            # Spine twist (difference between shoulder and hip rotation)
            # This is the upper body rotation RELATIVE to the hips
            spine_twist = shoulder_rotation_delta - hip_rotation_delta
            
            # Hip translation
            hip_delta = hip - self.ref_hip
            scaled_hip_delta = hip_delta * self.scale
            
            # === TORSO: TRANSLATION ONLY ===
            # Torso is the master - we DON'T rotate it because IK targets are NOT its children
            if has_torso:
                torso = pose_bones['torso']
                torso.location = Vector((scaled_hip_delta.x, scaled_hip_delta.y, scaled_hip_delta.z))
                torso.rotation_quaternion = Quaternion()  # Identity - NO ROTATION!
                torso.keyframe_insert(data_path="location", frame=bf)
                torso.keyframe_insert(data_path="rotation_quaternion", frame=bf)
                self.stats['keys'] += 2
                self.stats['bones'].add('torso')
            
            # === HIPS: ROTATION ===
            # The hips bone controls lower body rotation
            # spine_fk bones will auto-follow via MCH-spine COPY_TRANSFORMS constraint
            if has_hips:
                hips = pose_bones['hips']
                rot_quat = Quaternion((0, 0, 1), hip_rotation_delta)
                hips.rotation_quaternion = rot_quat
                hips.keyframe_insert(data_path="rotation_quaternion", frame=bf)
                self.stats['keys'] += 1
                self.stats['bones'].add('hips')
            
            # === CHEST: SPINE TWIST ===
            # Chest gets the full spine twist (shoulder rotation relative to hips)
            # We DON'T animate spine_fk because MCH-spine has COPY_TRANSFORMS from hips
            if has_chest:
                chest = pose_bones['chest']
                chest_rot = Quaternion((0, 0, 1), spine_twist)
                chest.rotation_quaternion = chest_rot
                chest.keyframe_insert(data_path="rotation_quaternion", frame=bf)
                self.stats['keys'] += 1
                self.stats['bones'].add('chest')
            
            # === IK TARGETS ===
            # Since we're NOT rotating torso, the IK targets stay in world space
            # We just need to move them relative to reference position + hip movement
            
            for ik_bone, config in avail_ik.items():
                endpoint = get_lm(lms, config['endpoint'])
                
                if not endpoint:
                    continue
                
                # Current position relative to hip
                current_rel = endpoint - hip
                
                # Reference position relative to hip
                ref_rel = self.ref_limb_positions.get(ik_bone)
                if not ref_rel:
                    continue
                
                # Limb delta (how hand moved relative to hip)
                limb_delta = current_rel - ref_rel
                scaled_limb_delta = limb_delta * self.scale
                
                # Total movement = hip translation + limb movement
                # NO rotation needed because torso isn't rotating!
                total_delta = scaled_hip_delta + scaled_limb_delta
                
                # Clamp to prevent stretching
                max_len = self.char_max_lengths.get(ik_bone, 1.0)
                if scaled_limb_delta.length > max_len * 0.5:
                    scaled_limb_delta = clamp_length(scaled_limb_delta, max_len * 0.5)
                    total_delta = scaled_hip_delta + scaled_limb_delta
                
                pb = pose_bones[ik_bone]
                pb.location = total_delta
                pb.keyframe_insert(data_path="location", frame=bf)
                
                self.stats['keys'] += 1
                self.stats['bones'].add(ik_bone)
            
            self.stats['frames'] += 1
            
            if fidx % 50 == 0:
                debug(f"    Frame {fidx}/{len(frames)}")
                if fidx > 0:
                    debug(f"      Hip rot: {math.degrees(hip_rotation_delta):.1f}°")
                    debug(f"      Spine twist: {math.degrees(spine_twist):.1f}° (to chest)")
            
            if fidx == 0:
                debug(f"    Frame 0:")
                debug(f"      Torso loc: ({scaled_hip_delta.x:.4f}, {scaled_hip_delta.y:.4f}, {scaled_hip_delta.z:.4f})")
                debug(f"      Hip rot: {math.degrees(hip_rotation_delta):.1f}° (spine_fk follows via MCH)")
                debug(f"      Chest twist: {math.degrees(spine_twist):.1f}°")
        
        return True
    
    def summary(self):
        debug("\n" + "="*60)
        debug("IMPORT SUMMARY")
        debug("="*60)
        debug(f"  Frames: {self.stats['frames']}")
        debug(f"  Keyframes: {self.stats['keys']}")
        debug(f"  Bones animated: {sorted(self.stats['bones'])}")


# =============================================================================
# OPERATORS
# =============================================================================

class MELODICCAP_OT_import(bpy.types.Operator, ImportHelper):
    bl_idname = "melodiccap.import_take"
    bl_label = "Import Take"
    bl_options = {'REGISTER', 'UNDO'}
    
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    
    def execute(self, context):
        debug("\n" + "="*60)
        debug("MELODICCAP v9.0 IMPORT")
        debug("="*60)
        
        arm = context.active_object
        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature!")
            return {'CANCELLED'}
        
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        
        settings = {
            'start_frame': context.scene.melodiccap_start_frame,
        }
        
        imp = MelodicCapImporter(arm, data, settings)
        
        if not imp.analyze():
            self.report({'ERROR'}, "Analysis failed")
            return {'CANCELLED'}
        
        bpy.ops.object.mode_set(mode='POSE')
        
        imp.apply_animation()
        imp.summary()
        
        self.report({'INFO'}, f"Imported {imp.stats['frames']} frames")
        return {'FINISHED'}


class MELODICCAP_OT_clear(bpy.types.Operator):
    bl_idname = "melodiccap.clear"
    bl_label = "Clear Animation"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        arm = context.active_object
        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select armature!")
            return {'CANCELLED'}
        
        if arm.animation_data:
            arm.animation_data_clear()
        
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.pose.transforms_clear()
        
        debug("Cleared animation")
        self.report({'INFO'}, "Cleared")
        return {'FINISHED'}


class MELODICCAP_OT_set_ik(bpy.types.Operator):
    bl_idname = "melodiccap.set_ik"
    bl_label = "Set IK Mode"
    
    def execute(self, context):
        arm = context.active_object
        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select armature!")
            return {'CANCELLED'}
        
        for switch_bone, prop_name in IK_FK_SWITCHES.items():
            if switch_bone in arm.pose.bones:
                pb = arm.pose.bones[switch_bone]
                if prop_name in pb:
                    pb[prop_name] = 0.0
        
        self.report({'INFO'}, "Set IK mode")
        return {'FINISHED'}


# =============================================================================
# PANEL
# =============================================================================

class MELODICCAP_PT_panel(bpy.types.Panel):
    bl_label = "MelodicCap v9"
    bl_idname = "MELODICCAP_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'MelodicCap'
    
    def draw(self, context):
        layout = self.layout
        
        box = layout.box()
        box.label(text="Target:", icon='ARMATURE_DATA')
        if context.active_object and context.active_object.type == 'ARMATURE':
            box.label(text=f"  {context.active_object.name}")
        else:
            box.label(text="  (Select armature)")
        
        box = layout.box()
        box.prop(context.scene, "melodiccap_start_frame")
        
        layout.operator("melodiccap.import_take", icon='IMPORT')
        layout.operator("melodiccap.set_ik", icon='CON_KINEMATIC')
        layout.operator("melodiccap.clear", icon='X')


# =============================================================================
# REGISTER
# =============================================================================

classes = [MELODICCAP_OT_import, MELODICCAP_OT_clear, MELODICCAP_OT_set_ik, MELODICCAP_PT_panel]

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.melodiccap_start_frame = IntProperty(name="Start Frame", default=1, min=1)
    debug("MelodicCap v9.0 registered - Rigify hierarchy fix!")

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.melodiccap_start_frame

if __name__ == "__main__":
    register()
