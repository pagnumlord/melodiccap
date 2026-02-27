"""
MelodicCap Blender Addon v7.0
=============================
NEW IN v7:
- BODY ROTATION from shoulder/hip orientation
- Hip rotation applied to torso
- Spine twist (chest rotation relative to hips)
- Proper Z-axis rotation for body turning

For Blender 4.4.3+ and JaxRigify (1.87m)
"""

bl_info = {
    "name": "MelodicCap Motion Capture Importer",
    "author": "Karsten / MelodicCap Studio",
    "version": (7, 0, 0),
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
# BONE MAPPING
# Person's LEFT -> Blender RIGHT (mirror)
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

def calculate_facing_angle(left_pos, right_pos):
    """
    Calculate the facing angle (rotation around Z axis) from left/right positions.
    Returns angle in radians.
    
    The facing direction is perpendicular to the left-right vector.
    In XY plane: perpendicular to (dx, dy) is (-dy, dx)
    """
    vec = right_pos - left_pos  # Vector from left to right
    
    # Facing direction (perpendicular, pointing "forward")
    facing_x = -vec.y
    facing_y = vec.x
    
    # Angle from positive X axis
    angle = math.atan2(facing_y, facing_x)
    
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
        self.ref_hip_angle = 0.0      # Reference hip facing angle
        self.ref_shoulder_angle = 0.0  # Reference shoulder facing angle
        
        # Character data
        self.char_max_lengths = {}
        self.mocap_root_positions = {}
        
        # Scale
        self.scale = 1.0
        
        self.stats = {'frames': 0, 'keys': 0, 'bones': set()}
    
    def analyze(self):
        """Analyze character and mocap data"""
        debug("="*60)
        debug("MELODICCAP v7.0 - WITH BODY ROTATION")
        debug("="*60)
        
        bones = self.armature.data.bones
        world = self.armature.matrix_world
        
        # === CHARACTER ANALYSIS ===
        debug(f"\n  CHARACTER ANALYSIS:")
        
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
            debug("ERROR: No hip in reference frame!", "ERROR")
            return False
        
        # Person height for scale
        nose = get_lm(self.ref_landmarks, LM_NOSE)
        ankle_l = get_lm(self.ref_landmarks, LM_LEFT_ANKLE)
        ankle_r = get_lm(self.ref_landmarks, LM_RIGHT_ANKLE)
        
        if nose and ankle_l and ankle_r:
            ankle_mid = (ankle_l + ankle_r) / 2
            mocap_height = (nose.z - ankle_mid.z) + 0.15
            self.scale = char_height / mocap_height
            debug(f"    Person height: {mocap_height:.3f}m")
            debug(f"    Scale factor: {self.scale:.4f}")
        
        # === REFERENCE BODY ORIENTATION ===
        l_shoulder = get_lm(self.ref_landmarks, LM_LEFT_SHOULDER)
        r_shoulder = get_lm(self.ref_landmarks, LM_RIGHT_SHOULDER)
        l_hip = get_lm(self.ref_landmarks, LM_LEFT_HIP)
        r_hip = get_lm(self.ref_landmarks, LM_RIGHT_HIP)
        
        if l_shoulder and r_shoulder:
            self.ref_shoulder_angle = calculate_facing_angle(l_shoulder, r_shoulder)
            debug(f"\n  REFERENCE BODY ORIENTATION:")
            debug(f"    Shoulder facing: {math.degrees(self.ref_shoulder_angle):.1f}°")
        
        if l_hip and r_hip:
            self.ref_hip_angle = calculate_facing_angle(l_hip, r_hip)
            debug(f"    Hip facing:      {math.degrees(self.ref_hip_angle):.1f}°")
            debug(f"    Spine twist:     {math.degrees(self.ref_shoulder_angle - self.ref_hip_angle):.1f}°")
        
        # Reference limb root positions
        for ik_bone, config in IK_CONFIG.items():
            pos = get_lm(self.ref_landmarks, config['root'])
            if pos:
                self.mocap_root_positions[ik_bone] = pos.copy()
        
        return True
    
    def set_ik_mode(self):
        """Set rig to IK mode"""
        pose_bones = self.armature.pose.bones
        for switch_bone, prop_name in IK_FK_SWITCHES.items():
            if switch_bone in pose_bones:
                pb = pose_bones[switch_bone]
                if prop_name in pb:
                    pb[prop_name] = 0.0
    
    def apply_animation(self):
        """Apply animation with body rotation"""
        debug("\n" + "="*60)
        debug("APPLYING ANIMATION (With Body Rotation)")
        debug("="*60)
        
        frames = self.take_data.get('frames', [])
        pose_bones = self.armature.pose.bones
        start = self.settings.get('start_frame', 1)
        
        # Force IK mode
        self.set_ik_mode()
        
        # Set rotation modes
        for pb in pose_bones:
            pb.rotation_mode = 'QUATERNION'
        
        # Check available bones
        avail_ik = {b: c for b, c in IK_CONFIG.items() if b in pose_bones}
        debug(f"  Available IK: {list(avail_ik.keys())}")
        
        has_torso = 'torso' in pose_bones
        has_hips = 'hips' in pose_bones
        has_chest = 'chest' in pose_bones
        
        debug(f"  Torso: {has_torso}, Hips: {has_hips}, Chest: {has_chest}")
        
        debug(f"\n  Processing {len(frames)} frames...")
        
        for fidx, fdata in enumerate(frames):
            bf = start + fidx
            bpy.context.scene.frame_set(bf)
            
            lms = fdata.get('landmarks', {})
            
            # Get key positions
            hip = get_mid(lms, LM_LEFT_HIP, LM_RIGHT_HIP)
            l_shoulder = get_lm(lms, LM_LEFT_SHOULDER)
            r_shoulder = get_lm(lms, LM_RIGHT_SHOULDER)
            l_hip = get_lm(lms, LM_LEFT_HIP)
            r_hip = get_lm(lms, LM_RIGHT_HIP)
            
            if not hip:
                continue
            
            # Calculate hip delta for root motion
            hip_delta = hip - self.ref_hip
            
            # === BODY ROTATION ===
            hip_rotation = 0.0
            shoulder_rotation = 0.0
            
            if l_hip and r_hip:
                current_hip_angle = calculate_facing_angle(l_hip, r_hip)
                hip_rotation = current_hip_angle - self.ref_hip_angle
            
            if l_shoulder and r_shoulder:
                current_shoulder_angle = calculate_facing_angle(l_shoulder, r_shoulder)
                shoulder_rotation = current_shoulder_angle - self.ref_shoulder_angle
            
            # Spine twist = difference between shoulder and hip rotation
            spine_twist = shoulder_rotation - hip_rotation
            
            # === TORSO (Location + Rotation) ===
            if has_torso:
                torso = pose_bones['torso']
                
                # Location (hip movement)
                scaled_delta = hip_delta * self.scale
                torso.location = Vector((scaled_delta.x, scaled_delta.y, scaled_delta.z))
                torso.keyframe_insert(data_path="location", frame=bf)
                
                # Rotation (hip facing direction)
                # Rotate around Z axis (vertical) by hip_rotation
                rot_quat = Quaternion((0, 0, 1), hip_rotation)
                torso.rotation_quaternion = rot_quat
                torso.keyframe_insert(data_path="rotation_quaternion", frame=bf)
                
                self.stats['keys'] += 2
                self.stats['bones'].add('torso')
                
                if fidx == 0:
                    debug(f"    Frame 0 torso:")
                    debug(f"      Location: ({scaled_delta.x:.4f}, {scaled_delta.y:.4f}, {scaled_delta.z:.4f})")
                    debug(f"      Rotation: {math.degrees(hip_rotation):.1f}° (Z-axis)")
            
            # === CHEST (Spine twist) ===
            if has_chest:
                chest = pose_bones['chest']
                
                # Apply spine twist (shoulder rotation relative to hips)
                twist_quat = Quaternion((0, 0, 1), spine_twist)
                chest.rotation_quaternion = twist_quat
                chest.keyframe_insert(data_path="rotation_quaternion", frame=bf)
                
                self.stats['keys'] += 1
                self.stats['bones'].add('chest')
                
                if fidx == 0:
                    debug(f"    Frame 0 chest:")
                    debug(f"      Spine twist: {math.degrees(spine_twist):.1f}°")
            
            # === IK TARGETS ===
            for ik_bone, config in avail_ik.items():
                endpoint = get_lm(lms, config['endpoint'])
                root = get_lm(lms, config['root'])
                
                if not endpoint or not root:
                    continue
                
                # Limb vector (shoulder to wrist / hip to ankle)
                mocap_limb_vec = endpoint - root
                
                # Reference limb vector
                ref_root = self.mocap_root_positions.get(ik_bone)
                ref_endpoint = get_lm(self.ref_landmarks, config['endpoint'])
                
                if not ref_root or not ref_endpoint:
                    continue
                
                ref_limb_vec = ref_endpoint - ref_root
                
                # Limb delta
                limb_delta = mocap_limb_vec - ref_limb_vec
                scaled_limb_delta = limb_delta * self.scale
                
                # Root motion
                root_motion = hip_delta * self.scale
                
                # Total movement
                total_delta = root_motion + scaled_limb_delta
                
                # Clamp to prevent stretching
                max_len = self.char_max_lengths.get(ik_bone, 1.0)
                if scaled_limb_delta.length > max_len * 0.5:
                    scaled_limb_delta = clamp_length(scaled_limb_delta, max_len * 0.5)
                    total_delta = root_motion + scaled_limb_delta
                
                # Apply
                pb = pose_bones[ik_bone]
                pb.location = total_delta
                pb.keyframe_insert(data_path="location", frame=bf)
                
                self.stats['keys'] += 1
                self.stats['bones'].add(ik_bone)
            
            self.stats['frames'] += 1
            
            if fidx % 50 == 0:
                debug(f"    Frame {fidx}/{len(frames)}")
                if fidx > 0:
                    debug(f"      Hip rotation: {math.degrees(hip_rotation):.1f}°, Spine twist: {math.degrees(spine_twist):.1f}°")
        
        return True
    
    def summary(self):
        debug("\n" + "="*60)
        debug("IMPORT SUMMARY")
        debug("="*60)
        debug(f"  Frames: {self.stats['frames']}")
        debug(f"  Keyframes: {self.stats['keys']}")
        debug(f"  Bones: {sorted(self.stats['bones'])}")


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
        debug("MELODICCAP v7.0 IMPORT")
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
        
        self.report({'INFO'}, f"Imported {imp.stats['frames']} frames with rotation")
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
    bl_label = "MelodicCap v7"
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
        box.label(text="Settings:", icon='SETTINGS')
        box.prop(context.scene, "melodiccap_start_frame")
        
        box = layout.box()
        box.label(text="Actions:", icon='ACTION')
        box.operator("melodiccap.import_take", icon='IMPORT')
        box.operator("melodiccap.set_ik", icon='CON_KINEMATIC')
        box.operator("melodiccap.clear", icon='X')


# =============================================================================
# REGISTER
# =============================================================================

classes = [MELODICCAP_OT_import, MELODICCAP_OT_clear, MELODICCAP_OT_set_ik, MELODICCAP_PT_panel]

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.melodiccap_start_frame = IntProperty(name="Start Frame", default=1, min=1)
    debug("MelodicCap v7.0 registered - Now with body ROTATION!")

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.melodiccap_start_frame

if __name__ == "__main__":
    register()
