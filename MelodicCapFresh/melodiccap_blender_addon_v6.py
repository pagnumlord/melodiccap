"""
MelodicCap Blender Addon v6.0
=============================
PROFESSIONAL RETARGETING APPROACH:
- Shoulder-relative hand positioning (not world-relative)
- Hip-relative foot positioning
- Limb length clamping (prevents stretching)
- Floor offset enforcement
- Proper proportional scaling

Based on Rokoko/FreeMoCap retargeting methodology.
For Blender 4.4.3+ and JaxRigify (1.87m)
"""

bl_info = {
    "name": "MelodicCap Motion Capture Importer",
    "author": "Karsten / MelodicCap Studio",
    "version": (6, 0, 0),
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
# RETARGETING STRATEGY
# =============================================================================
# Instead of: hand_position = mocap_hand - mocap_hip (world relative)
# We use:     hand_position = jax_shoulder + (mocap_hand - mocap_shoulder) * scale
#
# This keeps the hand relative to the shoulder, so if the torso moves,
# the hands stay attached to the body.
#
# Mirroring is handled by LANDMARK MAPPING:
#   Person's LEFT arm (landmarks 11,13,15) -> Jax's RIGHT arm (.R bones)
#   Person's RIGHT arm (landmarks 12,14,16) -> Jax's LEFT arm (.L bones)
# =============================================================================

# =============================================================================
# LANDMARK & BONE MAPPING
# =============================================================================

# MediaPipe landmark indices
LM_NOSE = 0
LM_LEFT_SHOULDER = 11   # Person's left (camera right)
LM_RIGHT_SHOULDER = 12  # Person's right (camera left)
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

# IK target configuration
# Format: bone_name -> {wrist/ankle landmark, shoulder/hip landmark for relative positioning}
# Person's LEFT limbs -> Blender RIGHT bones (mirror)
# Person's RIGHT limbs -> Blender LEFT bones (mirror)
IK_CONFIG = {
    'hand_ik.R': {
        'endpoint': LM_LEFT_WRIST,      # Person's left wrist
        'root': LM_LEFT_SHOULDER,        # Person's left shoulder
        'char_root': 'upper_arm_fk.R',   # Jax's right upper arm (head = shoulder)
        'max_length_bones': ['upper_arm_fk.R', 'forearm_fk.R'],
    },
    'hand_ik.L': {
        'endpoint': LM_RIGHT_WRIST,     # Person's right wrist
        'root': LM_RIGHT_SHOULDER,       # Person's right shoulder  
        'char_root': 'upper_arm_fk.L',   # Jax's left upper arm
        'max_length_bones': ['upper_arm_fk.L', 'forearm_fk.L'],
    },
    'foot_ik.R': {
        'endpoint': LM_LEFT_ANKLE,      # Person's left ankle
        'root': LM_LEFT_HIP,             # Person's left hip
        'char_root': 'thigh_fk.R',       # Jax's right thigh
        'max_length_bones': ['thigh_fk.R', 'shin_fk.R'],
    },
    'foot_ik.L': {
        'endpoint': LM_RIGHT_ANKLE,     # Person's right ankle
        'root': LM_RIGHT_HIP,            # Person's right hip
        'char_root': 'thigh_fk.L',       # Jax's left thigh
        'max_length_bones': ['thigh_fk.L', 'shin_fk.L'],
    },
}

# IK/FK switches
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
    """Get landmark as Vector"""
    key = str(idx) if str(idx) in landmarks else idx
    if key in landmarks:
        p = landmarks[key]
        return Vector((p[0], p[1], p[2]))
    return None

def get_mid(landmarks, i1, i2):
    """Get midpoint"""
    p1, p2 = get_lm(landmarks, i1), get_lm(landmarks, i2)
    return (p1 + p2) / 2 if p1 and p2 else None

def clamp_vector_length(vec, max_length):
    """Clamp vector to maximum length (prevents stretching)"""
    length = vec.length
    if length > max_length and length > 0.0001:
        return vec * (max_length / length)
    return vec

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
        self.floor_z = 0.0
        
        # Character data
        self.char_root_positions = {}  # Shoulder/hip positions for each IK target
        self.char_max_lengths = {}     # Maximum limb lengths
        self.char_ik_rest = {}         # IK target rest positions
        
        # Mocap reference data
        self.mocap_root_positions = {}  # Reference shoulder/hip positions
        
        # Scale
        self.scale = 1.0
        
        self.stats = {'frames': 0, 'keys': 0, 'bones': set()}
    
    def analyze(self):
        """Analyze character and mocap data"""
        debug("="*60)
        debug("MELODICCAP v6.0 - BODY-RELATIVE RETARGETING")
        debug("="*60)
        
        bones = self.armature.data.bones
        world = self.armature.matrix_world
        
        # === CHARACTER ANALYSIS ===
        debug(f"\n  CHARACTER ANALYSIS:")
        
        # Get root positions and max lengths for each IK target
        for ik_bone, config in IK_CONFIG.items():
            # Root position (shoulder or hip)
            root_bone = config['char_root']
            if root_bone in bones:
                pos = world @ bones[root_bone].head_local
                self.char_root_positions[ik_bone] = pos.copy()
                debug(f"    {ik_bone} root ({root_bone}): ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")
            
            # Maximum limb length
            total_length = 0.0
            for bone_name in config['max_length_bones']:
                if bone_name in bones:
                    bone = bones[bone_name]
                    head = world @ bone.head_local
                    tail = world @ bone.tail_local
                    total_length += (tail - head).length
            
            self.char_max_lengths[ik_bone] = total_length
            debug(f"    {ik_bone} max length: {total_length:.3f}m")
            
            # IK target rest position
            if ik_bone in bones:
                pos = world @ bones[ik_bone].head_local
                self.char_ik_rest[ik_bone] = pos.copy()
        
        # Character height (for general scale)
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
        
        # Calibration data
        calib = self.take_data.get('calibration', {})
        self.floor_z = calib.get('floor_offset', 0.0)
        debug(f"    Floor offset: {self.floor_z:.3f}m")
        
        # Reference frame
        self.ref_landmarks = frames[0].get('landmarks', {})
        self.ref_hip = get_mid(self.ref_landmarks, LM_LEFT_HIP, LM_RIGHT_HIP)
        
        if not self.ref_hip:
            debug("ERROR: No hip in reference frame!", "ERROR")
            return False
        
        debug(f"    Reference hip: ({self.ref_hip.x:.3f}, {self.ref_hip.y:.3f}, {self.ref_hip.z:.3f})")
        
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
        
        # Reference root positions (shoulders/hips in mocap space)
        debug(f"\n  REFERENCE LIMB ROOTS (mocap space):")
        for ik_bone, config in IK_CONFIG.items():
            root_lm = config['root']
            pos = get_lm(self.ref_landmarks, root_lm)
            if pos:
                self.mocap_root_positions[ik_bone] = pos.copy()
                debug(f"    {ik_bone} root (lm {root_lm}): ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")
        
        # Reference endpoint positions
        debug(f"\n  REFERENCE ENDPOINTS (relative to root):")
        for ik_bone, config in IK_CONFIG.items():
            endpoint = get_lm(self.ref_landmarks, config['endpoint'])
            root = self.mocap_root_positions.get(ik_bone)
            if endpoint and root:
                rel = endpoint - root
                debug(f"    {ik_bone}: ({rel.x:.3f}, {rel.y:.3f}, {rel.z:.3f}) len={rel.length:.3f}m")
        
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
        """Apply animation with body-relative positioning"""
        debug("\n" + "="*60)
        debug("APPLYING ANIMATION (Body-Relative)")
        debug("="*60)
        
        frames = self.take_data.get('frames', [])
        pose_bones = self.armature.pose.bones
        bones = self.armature.data.bones
        world = self.armature.matrix_world
        start = self.settings.get('start_frame', 1)
        
        # Force IK mode
        self.set_ik_mode()
        
        # Set quaternion mode
        for pb in pose_bones:
            pb.rotation_mode = 'QUATERNION'
        
        # Check available bones
        avail_ik = {b: c for b, c in IK_CONFIG.items() if b in pose_bones}
        debug(f"  Available IK targets: {list(avail_ik.keys())}")
        
        has_torso = 'torso' in pose_bones
        debug(f"  Torso: {'Yes' if has_torso else 'No'}")
        
        debug(f"\n  Processing {len(frames)} frames...")
        
        for fidx, fdata in enumerate(frames):
            bf = start + fidx
            bpy.context.scene.frame_set(bf)
            
            lms = fdata.get('landmarks', {})
            hip = get_mid(lms, LM_LEFT_HIP, LM_RIGHT_HIP)
            
            if not hip:
                continue
            
            # Calculate hip delta (needed for both torso and IK targets)
            hip_delta = hip - self.ref_hip
            
            # === ROOT MOTION (TORSO) ===
            if has_torso:
                # Scale and apply
                # Note: We DON'T mirror X for torso - it's center of body
                scaled_delta = hip_delta * self.scale
                
                torso = pose_bones['torso']
                torso.location = Vector((scaled_delta.x, scaled_delta.y, scaled_delta.z))
                torso.keyframe_insert(data_path="location", frame=bf)
                
                self.stats['keys'] += 1
                self.stats['bones'].add('torso')
                
                if fidx == 0:
                    debug(f"    Frame 0 torso: ({scaled_delta.x:.4f}, {scaled_delta.y:.4f}, {scaled_delta.z:.4f})")
            
            # === IK TARGETS (BODY-RELATIVE) ===
            # IMPORTANT: In Rigify, IK targets are NOT parented to torso!
            # So when torso moves, IK targets need to move with it + their own motion
            # Otherwise the arm will stretch to reach a stationary IK target
            
            for ik_bone, config in avail_ik.items():
                # Get current mocap positions
                endpoint = get_lm(lms, config['endpoint'])
                root = get_lm(lms, config['root'])
                
                if not endpoint or not root:
                    continue
                
                # Vector from root (shoulder/hip) to endpoint (wrist/ankle) in MOCAP space
                mocap_limb_vec = endpoint - root
                
                # Reference vector (frame 0)
                ref_root = self.mocap_root_positions.get(ik_bone)
                ref_endpoint = get_lm(self.ref_landmarks, config['endpoint'])
                
                if not ref_root or not ref_endpoint:
                    continue
                
                ref_limb_vec = ref_endpoint - ref_root
                
                # === THE KEY INSIGHT ===
                # IK target position = torso_movement + limb_relative_movement
                # Because IK targets are in world space, not parented to torso
                
                # 1. Limb delta (how the hand moved relative to shoulder)
                limb_delta = mocap_limb_vec - ref_limb_vec
                scaled_limb_delta = limb_delta * self.scale
                
                # 2. Root motion (torso movement) - IK needs this too!
                root_motion = hip_delta * self.scale  # hip_delta calculated above
                
                # 3. Total IK movement = root motion + limb movement
                total_delta = root_motion + scaled_limb_delta
                
                # CLAMP to prevent stretching
                max_len = self.char_max_lengths.get(ik_bone, 1.0)
                
                # Clamp the limb delta part (not root motion)
                if scaled_limb_delta.length > max_len * 0.5:
                    scaled_limb_delta = clamp_vector_length(scaled_limb_delta, max_len * 0.5)
                    total_delta = root_motion + scaled_limb_delta
                
                # Apply to IK bone
                pb = pose_bones[ik_bone]
                pb.location = total_delta
                pb.keyframe_insert(data_path="location", frame=bf)
                
                self.stats['keys'] += 1
                self.stats['bones'].add(ik_bone)
                
                if fidx == 0:
                    debug(f"    Frame 0 {ik_bone}:")
                    debug(f"      Root motion: ({root_motion.x:.4f}, {root_motion.y:.4f}, {root_motion.z:.4f})")
                    debug(f"      Limb delta:  ({scaled_limb_delta.x:.4f}, {scaled_limb_delta.y:.4f}, {scaled_limb_delta.z:.4f})")
                    debug(f"      Total:       ({total_delta.x:.4f}, {total_delta.y:.4f}, {total_delta.z:.4f})")
            
            self.stats['frames'] += 1
            
            if fidx % 50 == 0:
                debug(f"    Frame {fidx}/{len(frames)}")
        
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
        debug("MELODICCAP v6.0 IMPORT")
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
    bl_label = "MelodicCap v6"
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
    debug("MelodicCap v6.0 registered - Body-relative positioning!")

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.melodiccap_start_frame

if __name__ == "__main__":
    register()
