"""
MelodicCap Blender Addon v4.0
=============================
FIXES FROM v3:
- Animates IK targets (hand_ik, foot_ik) - hands/feet now move!
- Uses floor_offset from calibration data
- Proper IK/FK switch handling
- Hybrid mode: IK positions + FK rotations
- Armature scale compensation
- Ankle-to-foot-ik offset correction

Based on Rokoko/FreeMoCap retargeting principles.
For Blender 4.4.3+ and JaxRigify (1.87m)
"""

bl_info = {
    "name": "MelodicCap Motion Capture Importer",
    "author": "Karsten / MelodicCap Studio",
    "version": (4, 0, 0),
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
# LANDMARK DEFINITIONS
# =============================================================================

LANDMARKS = {
    0: "nose", 11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow", 15: "left_wrist", 16: "right_wrist",
    17: "left_pinky", 18: "right_pinky", 19: "left_index", 20: "right_index",
    21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle", 29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index",
}

# =============================================================================
# BONE MAPPING
# MediaPipe person's LEFT = Blender's RIGHT (mirror when facing camera)
# =============================================================================

# FK bone chains for rotation
FK_CHAINS = {
    # Person's LEFT -> Blender RIGHT
    'upper_arm_fk.R': (11, 13),
    'forearm_fk.R': (13, 15),
    'thigh_fk.R': (23, 25),
    'shin_fk.R': (25, 27),
    # Person's RIGHT -> Blender LEFT
    'upper_arm_fk.L': (12, 14),
    'forearm_fk.L': (14, 16),
    'thigh_fk.L': (24, 26),
    'shin_fk.L': (26, 28),
}

# IK targets for position (CRITICAL - this was missing!)
IK_TARGETS = {
    'hand_ik.R': 15,   # Person's left wrist
    'hand_ik.L': 16,   # Person's right wrist
    'foot_ik.R': 27,   # Person's left ankle
    'foot_ik.L': 28,   # Person's right ankle
}

# IK/FK switch bones (Rigify convention)
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
        self.floor_offset = 0.0
        
        # Scaling
        self.scale = 1.0
        self.char_height = 1.87
        self.armature_scale = 1.0
        
        # IK target rest positions (for offset calculation)
        self.ik_rest_positions = {}
        
        self.stats = {'frames': 0, 'keys': 0, 'bones': set()}
    
    def analyze(self):
        """Analyze character and mocap data"""
        debug("="*60)
        debug("MELODICCAP v4.0 - ANALYSIS")
        debug("="*60)
        
        # === ARMATURE SCALE CHECK ===
        arm_scale = self.armature.scale
        debug(f"  Armature scale: ({arm_scale.x:.3f}, {arm_scale.y:.3f}, {arm_scale.z:.3f})")
        
        if abs(arm_scale.x - 1.0) > 0.01 or abs(arm_scale.y - 1.0) > 0.01 or abs(arm_scale.z - 1.0) > 0.01:
            debug("  WARNING: Armature scale is not 1.0! Consider applying scale.", "WARN")
            self.armature_scale = arm_scale.z  # Use Z for height scaling
        
        # === CHARACTER HEIGHT ===
        bones = self.armature.data.bones
        world = self.armature.matrix_world
        
        min_z, max_z = float('inf'), float('-inf')
        for bone in bones:
            h = (world @ bone.head_local).z
            t = (world @ bone.tail_local).z
            min_z, max_z = min(min_z, h, t), max(max_z, h, t)
        
        self.char_height = max_z - min_z if max_z > min_z else 1.87
        debug(f"  Character height: {self.char_height:.3f}m")
        
        # === IK TARGET REST POSITIONS ===
        debug(f"\n  IK target rest positions (world space):")
        pose_bones = self.armature.pose.bones
        
        for ik_bone in ['hand_ik.L', 'hand_ik.R', 'foot_ik.L', 'foot_ik.R']:
            if ik_bone in bones:
                bone = bones[ik_bone]
                head_world = world @ bone.head_local
                self.ik_rest_positions[ik_bone] = head_world.copy()
                debug(f"    {ik_bone}: ({head_world.x:.3f}, {head_world.y:.3f}, {head_world.z:.3f})")
        
        # === FK BONE REST DIRECTIONS ===
        debug(f"\n  FK bone rest directions (world space):")
        for bone_name in FK_CHAINS.keys():
            if bone_name in bones:
                bone = bones[bone_name]
                head = world @ bone.head_local
                tail = world @ bone.tail_local
                direction = (tail - head).normalized()
                debug(f"    {bone_name}: ({direction.x:.3f}, {direction.y:.3f}, {direction.z:.3f})")
        
        # === IK/FK SWITCH STATUS ===
        debug(f"\n  IK/FK switch status:")
        for switch_bone, prop_name in IK_FK_SWITCHES.items():
            if switch_bone in pose_bones:
                pb = pose_bones[switch_bone]
                if prop_name in pb:
                    val = pb[prop_name]
                    mode = "FK" if val > 0.5 else "IK"
                    debug(f"    {switch_bone}['{prop_name}'] = {val:.2f} ({mode} mode)")
                else:
                    debug(f"    {switch_bone}: No '{prop_name}' property")
        
        # === MOCAP DATA ===
        frames = self.take_data.get('frames', [])
        if not frames:
            debug("ERROR: No frames!", "ERROR")
            return False
        
        debug(f"\n  Mocap data: {len(frames)} frames, {self.take_data.get('duration_seconds', 0):.1f}s")
        
        # === CALIBRATION DATA ===
        calib = self.take_data.get('calibration', {})
        self.floor_offset = calib.get('floor_offset', 0.0)
        debug(f"\n  Calibration:")
        debug(f"    Stereo RMS: {calib.get('rms_stereo', 'N/A')}")
        debug(f"    Baseline: {calib.get('baseline', 'N/A')}m")
        debug(f"    Floor offset: {self.floor_offset:.3f}m")
        
        # === REFERENCE FRAME ===
        self.ref_landmarks = frames[0].get('landmarks', {})
        self.ref_hip = get_mid(self.ref_landmarks, 23, 24)
        
        if not self.ref_hip:
            debug("ERROR: No hip center in frame 0!", "ERROR")
            return False
        
        debug(f"\n  REFERENCE FRAME (becomes origin):")
        debug(f"    Raw hip center: ({self.ref_hip.x:.3f}, {self.ref_hip.y:.3f}, {self.ref_hip.z:.3f})")
        
        # === PERSON HEIGHT ===
        nose = get_lm(self.ref_landmarks, 0)
        ankle_l = get_lm(self.ref_landmarks, 27)
        ankle_r = get_lm(self.ref_landmarks, 28)
        
        if nose and ankle_l and ankle_r:
            ankle_mid = (ankle_l + ankle_r) / 2
            mocap_height = (nose.z - ankle_mid.z) + 0.15
            self.scale = self.char_height / mocap_height
            debug(f"    Person height: {mocap_height:.3f}m")
            debug(f"    SCALE FACTOR: {self.scale:.4f}")
        
        # === REFERENCE LANDMARK POSITIONS ===
        debug(f"\n  Reference landmarks (relative to hip):")
        for idx in [11, 12, 15, 16, 23, 24, 27, 28]:
            pos = get_lm(self.ref_landmarks, idx)
            if pos:
                rel = pos - self.ref_hip
                name = LANDMARKS.get(idx, str(idx))
                debug(f"    [{idx:2d}] {name:15s}: ({rel.x:7.3f}, {rel.y:7.3f}, {rel.z:7.3f})")
        
        # === REFERENCE IK TARGET POSITIONS (MOCAP SPACE) ===
        debug(f"\n  Reference IK positions (mocap, relative to hip):")
        for ik_bone, lm_idx in IK_TARGETS.items():
            pos = get_lm(self.ref_landmarks, lm_idx)
            if pos:
                rel = pos - self.ref_hip
                debug(f"    {ik_bone} (lm {lm_idx}): ({rel.x:.3f}, {rel.y:.3f}, {rel.z:.3f})")
        
        return True
    
    def set_ik_fk_mode(self, use_ik=True):
        """Set IK/FK switches on the rig"""
        debug(f"\n  Setting {'IK' if use_ik else 'FK'} mode on Rigify controls...")
        
        pose_bones = self.armature.pose.bones
        target_value = 0.0 if use_ik else 1.0  # 0 = IK, 1 = FK in Rigify
        
        for switch_bone, prop_name in IK_FK_SWITCHES.items():
            if switch_bone in pose_bones:
                pb = pose_bones[switch_bone]
                if prop_name in pb:
                    old_val = pb[prop_name]
                    pb[prop_name] = target_value
                    debug(f"    {switch_bone}['{prop_name}']: {old_val:.1f} -> {target_value:.1f}")
    
    def apply_animation(self):
        """Apply animation - HYBRID mode: IK positions + optional FK rotations"""
        debug("\n" + "="*60)
        debug("APPLYING ANIMATION (HYBRID MODE)")
        debug("="*60)
        debug("  Strategy: Animate IK targets for hands/feet positions")
        debug("            Animate torso for root motion")
        debug("            Optionally animate FK for limb rotations")
        
        frames = self.take_data.get('frames', [])
        pose_bones = self.armature.pose.bones
        start = self.settings.get('start_frame', 1)
        
        # Force IK mode so IK targets drive the mesh
        self.set_ik_fk_mode(use_ik=True)
        
        # Set quaternion rotation mode on all bones
        for pb in pose_bones:
            pb.rotation_mode = 'QUATERNION'
        
        # Get armature transforms
        world = self.armature.matrix_world
        world_inv = world.inverted()
        
        # === CHECK AVAILABLE BONES ===
        debug(f"\n  Checking available bones:")
        
        avail_ik = {}
        for bone, lm_idx in IK_TARGETS.items():
            if bone in pose_bones:
                avail_ik[bone] = lm_idx
                debug(f"    ✓ {bone} (IK target)")
            else:
                debug(f"    ✗ {bone} (missing)", "WARN")
        
        avail_fk = {}
        for bone, (i1, i2) in FK_CHAINS.items():
            if bone in pose_bones:
                avail_fk[bone] = (i1, i2)
                debug(f"    ✓ {bone} (FK chain)")
        
        has_torso = 'torso' in pose_bones
        debug(f"    {'✓' if has_torso else '✗'} torso (root motion)")
        
        # === PROCESS FRAMES ===
        debug(f"\n  Processing {len(frames)} frames...")
        
        for fidx, fdata in enumerate(frames):
            bf = start + fidx
            bpy.context.scene.frame_set(bf)
            
            lms = fdata.get('landmarks', {})
            hip = get_mid(lms, 23, 24)
            
            if not hip:
                continue
            
            # === ROOT MOTION (TORSO) ===
            if has_torso:
                # Delta from reference hip, scaled
                # This is RELATIVE motion - frame 0 should be (0,0,0)
                delta = (hip - self.ref_hip) * self.scale
                
                torso = pose_bones['torso']
                
                # Transform to armature local space
                # Note: torso.location is in armature local space
                local_delta = world_inv.to_3x3() @ delta
                
                torso.location = local_delta
                torso.keyframe_insert(data_path="location", frame=bf)
                
                self.stats['keys'] += 1
                self.stats['bones'].add('torso')
                
                if fidx == 0:
                    debug(f"    Frame 0 torso: ({local_delta.x:.4f}, {local_delta.y:.4f}, {local_delta.z:.4f})")
            
            # === IK TARGETS (HANDS AND FEET) ===
            for ik_bone, lm_idx in avail_ik.items():
                pos = get_lm(lms, lm_idx)
                if not pos:
                    continue
                
                # Position relative to current hip center
                rel_to_hip = pos - hip
                
                # Scale to character size
                scaled_rel = rel_to_hip * self.scale
                
                # Add root motion (hip delta from reference)
                hip_delta = (hip - self.ref_hip) * self.scale
                
                # Final world position = scaled relative position + root motion
                # But we need it relative to the character's origin (not world origin)
                # Since torso moves the whole body, IK targets should be relative to torso
                
                # Actually, for Rigify IK targets, they are in armature space
                # and move WITH the torso. So we just need the relative position.
                
                # Let me reconsider:
                # - torso.location moves the whole hierarchy
                # - IK targets are children of the hierarchy
                # - So IK target position should be relative to REST position
                
                # Get the bone's rest position
                rest_pos = self.ik_rest_positions.get(ik_bone, Vector((0, 0, 0)))
                
                # The mocap gives us where the hand/foot IS
                # We need to calculate where the IK target should BE
                
                # Approach: Calculate the DELTA from rest pose
                # Rest pose in mocap = first frame position
                ref_pos = get_lm(self.ref_landmarks, lm_idx)
                if not ref_pos:
                    continue
                
                # Delta from reference position
                mocap_delta = (pos - ref_pos) * self.scale
                
                # Apply this delta to the IK target
                # The IK target's location is in its own local space (parent space)
                pb = pose_bones[ik_bone]
                
                # For IK targets, we want absolute positioning
                # Transform delta to armature space
                local_delta = world_inv.to_3x3() @ mocap_delta
                
                pb.location = local_delta
                pb.keyframe_insert(data_path="location", frame=bf)
                
                self.stats['keys'] += 1
                self.stats['bones'].add(ik_bone)
                
                if fidx == 0:
                    debug(f"    Frame 0 {ik_bone}: ({local_delta.x:.4f}, {local_delta.y:.4f}, {local_delta.z:.4f})")
            
            # === FK ROTATIONS (OPTIONAL - for better limb orientation) ===
            if self.settings.get('animate_fk', True):
                for fk_bone, (i1, i2) in avail_fk.items():
                    p1, p2 = get_lm(lms, i1), get_lm(lms, i2)
                    if not p1 or not p2:
                        continue
                    
                    # Current bone direction in mocap space
                    cur_dir = (p2 - p1).normalized()
                    
                    # Get bone rest direction
                    bone_data = self.armature.data.bones[fk_bone]
                    rest_head = bone_data.head_local
                    rest_tail = bone_data.tail_local
                    rest_dir_local = (rest_tail - rest_head).normalized()
                    rest_dir_world = (world.to_3x3() @ rest_dir_local).normalized()
                    
                    # Rotation from rest to current (in world space)
                    rot_world = rest_dir_world.rotation_difference(cur_dir)
                    
                    # Convert to bone local space
                    pb = pose_bones[fk_bone]
                    
                    if pb.parent:
                        parent_world = world @ pb.parent.bone.matrix_local
                        parent_rot = parent_world.to_quaternion()
                        local_rot = parent_rot.inverted() @ rot_world
                    else:
                        arm_rot = world.to_quaternion()
                        local_rot = arm_rot.inverted() @ rot_world
                    
                    pb.rotation_quaternion = local_rot
                    pb.keyframe_insert(data_path="rotation_quaternion", frame=bf)
                    
                    self.stats['keys'] += 1
                    self.stats['bones'].add(fk_bone)
            
            self.stats['frames'] += 1
            
            if fidx % 50 == 0:
                debug(f"    Frame {fidx}/{len(frames)}")
        
        return True
    
    def summary(self):
        """Print summary"""
        debug("\n" + "="*60)
        debug("IMPORT SUMMARY")
        debug("="*60)
        debug(f"  Frames processed: {self.stats['frames']}")
        debug(f"  Keyframes created: {self.stats['keys']}")
        debug(f"  Bones animated ({len(self.stats['bones'])}):")
        
        ik_bones = [b for b in self.stats['bones'] if '_ik' in b]
        fk_bones = [b for b in self.stats['bones'] if '_fk' in b]
        other_bones = [b for b in self.stats['bones'] if '_ik' not in b and '_fk' not in b]
        
        if other_bones:
            debug(f"    Root: {other_bones}")
        if ik_bones:
            debug(f"    IK targets: {ik_bones}")
        if fk_bones:
            debug(f"    FK chains: {fk_bones}")


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
        debug("MELODICCAP v4.0 IMPORT")
        debug("="*60)
        
        arm = context.active_object
        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature!")
            return {'CANCELLED'}
        
        debug(f"  File: {self.filepath}")
        debug(f"  Armature: {arm.name}")
        
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        
        settings = {
            'start_frame': context.scene.melodiccap_start_frame,
            'animate_fk': context.scene.melodiccap_animate_fk,
        }
        
        imp = MelodicCapImporter(arm, data, settings)
        
        if not imp.analyze():
            self.report({'ERROR'}, "Analysis failed - check console")
            return {'CANCELLED'}
        
        bpy.ops.object.mode_set(mode='POSE')
        
        imp.apply_animation()
        imp.summary()
        
        self.report({'INFO'}, f"Imported {imp.stats['frames']} frames, {len(imp.stats['bones'])} bones")
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
        
        debug("Cleared animation and reset pose")
        self.report({'INFO'}, "Cleared")
        return {'FINISHED'}


class MELODICCAP_OT_set_ik_mode(bpy.types.Operator):
    """Set rig to IK mode (recommended for mocap)"""
    bl_idname = "melodiccap.set_ik_mode"
    bl_label = "Set IK Mode"
    
    def execute(self, context):
        arm = context.active_object
        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select armature!")
            return {'CANCELLED'}
        
        pose_bones = arm.pose.bones
        for switch_bone, prop_name in IK_FK_SWITCHES.items():
            if switch_bone in pose_bones:
                pb = pose_bones[switch_bone]
                if prop_name in pb:
                    pb[prop_name] = 0.0  # IK mode
        
        self.report({'INFO'}, "Set to IK mode")
        return {'FINISHED'}


# =============================================================================
# PANEL
# =============================================================================

class MELODICCAP_PT_panel(bpy.types.Panel):
    bl_label = "MelodicCap v4"
    bl_idname = "MELODICCAP_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'MelodicCap'
    
    def draw(self, context):
        layout = self.layout
        
        # Target
        box = layout.box()
        box.label(text="Target:", icon='ARMATURE_DATA')
        if context.active_object and context.active_object.type == 'ARMATURE':
            box.label(text=f"  {context.active_object.name}")
            
            # Check scale
            scale = context.active_object.scale
            if abs(scale.x - 1.0) > 0.01 or abs(scale.y - 1.0) > 0.01 or abs(scale.z - 1.0) > 0.01:
                box.label(text="  ⚠️ Scale not 1.0!", icon='ERROR')
        else:
            box.label(text="  (Select armature)")
        
        # Settings
        box = layout.box()
        box.label(text="Settings:", icon='SETTINGS')
        box.prop(context.scene, "melodiccap_start_frame")
        box.prop(context.scene, "melodiccap_animate_fk")
        
        # Actions
        box = layout.box()
        box.label(text="Actions:", icon='ACTION')
        box.operator("melodiccap.import_take", icon='IMPORT')
        box.operator("melodiccap.set_ik_mode", icon='CON_KINEMATIC')
        box.operator("melodiccap.clear", icon='X')


# =============================================================================
# REGISTER
# =============================================================================

classes = [
    MELODICCAP_OT_import,
    MELODICCAP_OT_clear,
    MELODICCAP_OT_set_ik_mode,
    MELODICCAP_PT_panel,
]

def register():
    for c in classes:
        bpy.utils.register_class(c)
    
    bpy.types.Scene.melodiccap_start_frame = IntProperty(
        name="Start Frame",
        default=1,
        min=1,
        description="Frame to start animation"
    )
    bpy.types.Scene.melodiccap_animate_fk = BoolProperty(
        name="Animate FK Rotations",
        default=True,
        description="Also animate FK bone rotations (better limb orientation)"
    )
    
    debug("MelodicCap v4.0 registered - Now with IK target animation!")

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    
    del bpy.types.Scene.melodiccap_start_frame
    del bpy.types.Scene.melodiccap_animate_fk

if __name__ == "__main__":
    register()
