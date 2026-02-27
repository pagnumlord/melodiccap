"""
MelodicCap Blender Addon v5.0
=============================
FIXES FROM v4:
- PROPER X-axis mirroring (mocap left = Blender right)
- Body-relative IK positioning (hands/feet relative to shoulders/hips)
- No more extreme stretching
- Coordinate system alignment
- Proportional bone length matching

For Blender 4.4.3+ and JaxRigify (1.87m)
"""

bl_info = {
    "name": "MelodicCap Motion Capture Importer",
    "author": "Karsten / MelodicCap Studio",
    "version": (5, 0, 0),
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
# COORDINATE SYSTEM NOTES
# =============================================================================
# MoCap space (MediaPipe + our triangulation):
#   X = left/right (positive = person's left when facing cameras)
#   Y = depth (positive = toward cameras, away from person)
#   Z = height (positive = up)
#
# Blender space:
#   X = left/right (positive = character's left)
#   Y = forward/back (positive = forward, typically)  
#   Z = height (positive = up)
#
# MIRRORING: Person faces cameras, so person's LEFT = viewer's RIGHT
#   Person's left hand (mocap +X) = Character's right hand (Blender -X)
#   We need to NEGATE X when transferring!
# =============================================================================

# =============================================================================
# LANDMARK & BONE MAPPING
# =============================================================================

LANDMARKS = {
    0: "nose", 11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow", 15: "left_wrist", 16: "right_wrist",
    23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
}

# IK targets: mocap landmark -> Blender IK bone
# Person's LEFT (landmarks 11,13,15,23,25,27) -> Blender RIGHT (.R)
# Person's RIGHT (landmarks 12,14,16,24,26,28) -> Blender LEFT (.L)
IK_TARGETS = {
    'hand_ik.R': {'landmark': 15, 'parent_lm': 11},  # Person's left wrist, relative to left shoulder
    'hand_ik.L': {'landmark': 16, 'parent_lm': 12},  # Person's right wrist, relative to right shoulder
    'foot_ik.R': {'landmark': 27, 'parent_lm': 23},  # Person's left ankle, relative to left hip
    'foot_ik.L': {'landmark': 28, 'parent_lm': 24},  # Person's right ankle, relative to right hip
}

# FK chains for rotation
FK_CHAINS = {
    'upper_arm_fk.R': (11, 13),
    'forearm_fk.R': (13, 15),
    'thigh_fk.R': (23, 25),
    'shin_fk.R': (25, 27),
    'upper_arm_fk.L': (12, 14),
    'forearm_fk.L': (14, 16),
    'thigh_fk.L': (24, 26),
    'shin_fk.L': (26, 28),
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

def mocap_to_blender(vec):
    """
    Convert mocap coordinates to Blender coordinates.
    - Negate X (mirror left/right)
    - Y stays same (depth)
    - Z stays same (height)
    """
    return Vector((-vec.x, vec.y, vec.z))

# =============================================================================
# IMPORTER
# =============================================================================

class MelodicCapImporter:
    
    def __init__(self, armature, take_data, settings):
        self.armature = armature
        self.take_data = take_data
        self.settings = settings
        
        # Reference data
        self.ref_hip = None
        self.ref_landmarks = None
        
        # Scaling
        self.scale = 1.0
        self.char_height = 1.87
        
        # Character bone data
        self.char_ik_rest = {}  # IK target rest positions
        self.char_shoulder_pos = {}  # Shoulder positions for arm IK
        self.char_hip_pos = {}  # Hip positions for leg IK
        
        # Limb lengths for proportional scaling
        self.mocap_arm_length = 0.0
        self.char_arm_length = 0.0
        self.mocap_leg_length = 0.0
        self.char_leg_length = 0.0
        
        self.stats = {'frames': 0, 'keys': 0, 'bones': set()}
    
    def analyze(self):
        """Analyze character and mocap data"""
        debug("="*60)
        debug("MELODICCAP v5.0 - ANALYSIS")
        debug("="*60)
        
        bones = self.armature.data.bones
        pose_bones = self.armature.pose.bones
        world = self.armature.matrix_world
        
        # === CHARACTER HEIGHT ===
        min_z, max_z = float('inf'), float('-inf')
        for bone in bones:
            h = (world @ bone.head_local).z
            t = (world @ bone.tail_local).z
            min_z, max_z = min(min_z, h, t), max(max_z, h, t)
        
        self.char_height = max_z - min_z if max_z > min_z else 1.87
        debug(f"  Character height: {self.char_height:.3f}m")
        
        # === CHARACTER BONE POSITIONS ===
        debug(f"\n  Character key positions (world space):")
        
        # Get IK target rest positions
        for ik_bone in ['hand_ik.L', 'hand_ik.R', 'foot_ik.L', 'foot_ik.R']:
            if ik_bone in bones:
                pos = world @ bones[ik_bone].head_local
                self.char_ik_rest[ik_bone] = pos.copy()
                debug(f"    {ik_bone}: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")
        
        # Get shoulder positions (for arm IK reference)
        for shoulder_bone in ['shoulder.L', 'shoulder.R', 'ORG-shoulder.L', 'ORG-shoulder.R']:
            if shoulder_bone in bones:
                pos = world @ bones[shoulder_bone].head_local
                side = '.L' if '.L' in shoulder_bone else '.R'
                self.char_shoulder_pos[side] = pos.copy()
                debug(f"    Shoulder{side}: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")
        
        # Get hip positions (for leg IK reference)  
        if 'ORG-spine' in bones:
            pos = world @ bones['ORG-spine'].head_local
            self.char_hip_pos['center'] = pos.copy()
            debug(f"    Hip center: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")
        
        # === CHARACTER LIMB LENGTHS ===
        debug(f"\n  Character limb lengths:")
        
        # Arm length (upper arm + forearm)
        if 'upper_arm_fk.L' in bones and 'forearm_fk.L' in bones:
            upper = bones['upper_arm_fk.L']
            fore = bones['forearm_fk.L']
            upper_len = (world @ upper.tail_local - world @ upper.head_local).length
            fore_len = (world @ fore.tail_local - world @ fore.head_local).length
            self.char_arm_length = upper_len + fore_len
            debug(f"    Arm (upper+forearm): {self.char_arm_length:.3f}m")
        
        # Leg length (thigh + shin)
        if 'thigh_fk.L' in bones and 'shin_fk.L' in bones:
            thigh = bones['thigh_fk.L']
            shin = bones['shin_fk.L']
            thigh_len = (world @ thigh.tail_local - world @ thigh.head_local).length
            shin_len = (world @ shin.tail_local - world @ shin.head_local).length
            self.char_leg_length = thigh_len + shin_len
            debug(f"    Leg (thigh+shin): {self.char_leg_length:.3f}m")
        
        # === MOCAP DATA ===
        frames = self.take_data.get('frames', [])
        if not frames:
            debug("ERROR: No frames!", "ERROR")
            return False
        
        debug(f"\n  Mocap data: {len(frames)} frames")
        
        # === REFERENCE FRAME ===
        self.ref_landmarks = frames[0].get('landmarks', {})
        self.ref_hip = get_mid(self.ref_landmarks, 23, 24)
        
        if not self.ref_hip:
            debug("ERROR: No hip center in frame 0!", "ERROR")
            return False
        
        debug(f"\n  REFERENCE FRAME:")
        debug(f"    Hip center (mocap): ({self.ref_hip.x:.3f}, {self.ref_hip.y:.3f}, {self.ref_hip.z:.3f})")
        
        # Person height
        nose = get_lm(self.ref_landmarks, 0)
        ankle_l = get_lm(self.ref_landmarks, 27)
        ankle_r = get_lm(self.ref_landmarks, 28)
        
        if nose and ankle_l and ankle_r:
            ankle_mid = (ankle_l + ankle_r) / 2
            mocap_height = (nose.z - ankle_mid.z) + 0.15
            self.scale = self.char_height / mocap_height
            debug(f"    Person height: {mocap_height:.3f}m")
            debug(f"    SCALE FACTOR: {self.scale:.4f}")
        
        # === MOCAP LIMB LENGTHS ===
        debug(f"\n  Mocap limb lengths (reference frame):")
        
        # Arm length
        shoulder_l = get_lm(self.ref_landmarks, 11)
        elbow_l = get_lm(self.ref_landmarks, 13)
        wrist_l = get_lm(self.ref_landmarks, 15)
        
        if shoulder_l and elbow_l and wrist_l:
            upper = (elbow_l - shoulder_l).length
            fore = (wrist_l - elbow_l).length
            self.mocap_arm_length = (upper + fore) * self.scale
            debug(f"    Arm (scaled): {self.mocap_arm_length:.3f}m")
        
        # Leg length
        hip_l = get_lm(self.ref_landmarks, 23)
        knee_l = get_lm(self.ref_landmarks, 25)
        ankle_l = get_lm(self.ref_landmarks, 27)
        
        if hip_l and knee_l and ankle_l:
            thigh = (knee_l - hip_l).length
            shin = (ankle_l - knee_l).length
            self.mocap_leg_length = (thigh + shin) * self.scale
            debug(f"    Leg (scaled): {self.mocap_leg_length:.3f}m")
        
        # Limb scale factors
        if self.mocap_arm_length > 0 and self.char_arm_length > 0:
            arm_scale = self.char_arm_length / self.mocap_arm_length
            debug(f"    Arm scale factor: {arm_scale:.3f}")
        
        if self.mocap_leg_length > 0 and self.char_leg_length > 0:
            leg_scale = self.char_leg_length / self.mocap_leg_length
            debug(f"    Leg scale factor: {leg_scale:.3f}")
        
        # === REFERENCE POSITIONS (Blender space) ===
        debug(f"\n  Reference positions (converted to Blender space):")
        for idx in [11, 12, 15, 16, 23, 24, 27, 28]:
            pos = get_lm(self.ref_landmarks, idx)
            if pos:
                rel = pos - self.ref_hip
                blender_rel = mocap_to_blender(rel)
                name = LANDMARKS.get(idx, str(idx))
                debug(f"    [{idx:2d}] {name:15s}: mocap({rel.x:6.3f},{rel.y:6.3f},{rel.z:6.3f}) -> blender({blender_rel.x:6.3f},{blender_rel.y:6.3f},{blender_rel.z:6.3f})")
        
        return True
    
    def set_ik_mode(self):
        """Set rig to IK mode"""
        pose_bones = self.armature.pose.bones
        for switch_bone, prop_name in IK_FK_SWITCHES.items():
            if switch_bone in pose_bones:
                pb = pose_bones[switch_bone]
                if prop_name in pb:
                    pb[prop_name] = 0.0  # IK mode
    
    def apply_animation(self):
        """Apply animation with proper coordinate conversion"""
        debug("\n" + "="*60)
        debug("APPLYING ANIMATION (v5 - Proper Coordinates)")
        debug("="*60)
        
        frames = self.take_data.get('frames', [])
        pose_bones = self.armature.pose.bones
        start = self.settings.get('start_frame', 1)
        
        # Force IK mode
        self.set_ik_mode()
        
        # Set quaternion mode
        for pb in pose_bones:
            pb.rotation_mode = 'QUATERNION'
        
        # Get transforms
        world = self.armature.matrix_world
        world_inv = world.inverted()
        
        # Check available bones
        debug(f"\n  Available bones:")
        
        avail_ik = {}
        for bone, info in IK_TARGETS.items():
            if bone in pose_bones:
                avail_ik[bone] = info
                debug(f"    ✓ {bone}")
        
        has_torso = 'torso' in pose_bones
        debug(f"    {'✓' if has_torso else '✗'} torso")
        
        # Get reference IK positions (for delta calculation)
        ref_ik_pos = {}
        for bone, info in avail_ik.items():
            lm_idx = info['landmark']
            pos = get_lm(self.ref_landmarks, lm_idx)
            if pos:
                ref_ik_pos[bone] = pos - self.ref_hip
        
        debug(f"\n  Processing {len(frames)} frames...")
        
        for fidx, fdata in enumerate(frames):
            bf = start + fidx
            bpy.context.scene.frame_set(bf)
            
            lms = fdata.get('landmarks', {})
            hip = get_mid(lms, 23, 24)
            
            if not hip:
                continue
            
            # === ROOT MOTION ===
            if has_torso:
                # Delta from reference, converted to Blender space
                mocap_delta = hip - self.ref_hip
                blender_delta = mocap_to_blender(mocap_delta) * self.scale
                
                torso = pose_bones['torso']
                
                # Transform to armature local space
                local_delta = world_inv.to_3x3() @ blender_delta
                
                torso.location = local_delta
                torso.keyframe_insert(data_path="location", frame=bf)
                
                self.stats['keys'] += 1
                self.stats['bones'].add('torso')
                
                if fidx == 0:
                    debug(f"    Frame 0 torso: ({local_delta.x:.4f}, {local_delta.y:.4f}, {local_delta.z:.4f})")
            
            # === IK TARGETS ===
            for ik_bone, info in avail_ik.items():
                lm_idx = info['landmark']
                pos = get_lm(lms, lm_idx)
                
                if not pos:
                    continue
                
                # Current position relative to hip
                rel_to_hip = pos - hip
                
                # Reference position relative to hip
                ref_rel = ref_ik_pos.get(ik_bone)
                if not ref_rel:
                    continue
                
                # Delta from reference position
                mocap_delta = rel_to_hip - ref_rel
                
                # Convert to Blender space and scale
                blender_delta = mocap_to_blender(mocap_delta) * self.scale
                
                # Apply to IK target
                pb = pose_bones[ik_bone]
                
                # Transform to armature local space
                local_delta = world_inv.to_3x3() @ blender_delta
                
                pb.location = local_delta
                pb.keyframe_insert(data_path="location", frame=bf)
                
                self.stats['keys'] += 1
                self.stats['bones'].add(ik_bone)
                
                if fidx == 0:
                    debug(f"    Frame 0 {ik_bone}: ({local_delta.x:.4f}, {local_delta.y:.4f}, {local_delta.z:.4f})")
            
            self.stats['frames'] += 1
            
            if fidx % 50 == 0:
                debug(f"    Frame {fidx}/{len(frames)}")
        
        return True
    
    def summary(self):
        """Print summary"""
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
        debug("MELODICCAP v5.0 IMPORT")
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
    bl_label = "MelodicCap v5"
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
    debug("MelodicCap v5.0 registered - X-axis mirroring fix!")

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.melodiccap_start_frame

if __name__ == "__main__":
    register()
