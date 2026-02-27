"""
MelodicCap Blender Addon v11.0
==============================
FIX: Animate TORSO with body movement!

The v10 problem was clear:
- IK targets (hands/feet) moved with your walking motion
- But TORSO stayed at origin
- Since IK targets are parented to torso (torso_parent=1)
- The limbs tried to reach targets 1.5m away = folding/stretching

v11 Solution:
1. Animate TORSO with hip movement (root motion)
2. Animate IK targets with LIMB-ONLY movement (relative to body)
3. The body moves, limbs stay attached

For Blender 4.4.3+ and JaxRigify (1.87m)
"""

bl_info = {
    "name": "MelodicCap Motion Capture Importer",
    "author": "Karsten / MelodicCap Studio",
    "version": (11, 0, 0),
    "blender": (4, 4, 0),
    "location": "View3D > Sidebar > MelodicCap",
    "description": "Import MelodicCap mocap data with proper root motion",
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

# Rigify IK control mapping
# MediaPipe LEFT (person's left) -> Rigify RIGHT (character's right when facing camera)
RIGIFY_IK_MAP = {
    'hand_ik.R': {'landmark': 15, 'type': 'hand'},   # Person's left wrist
    'hand_ik.L': {'landmark': 16, 'type': 'hand'},   # Person's right wrist
    'foot_ik.R': {'landmark': 27, 'type': 'foot'},   # Person's left ankle
    'foot_ik.L': {'landmark': 28, 'type': 'foot'},   # Person's right ankle
}

# IK/FK switch bones
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
    """Get hip center from landmarks"""
    l_hip = get_lm(landmarks, 23)
    r_hip = get_lm(landmarks, 24)
    if l_hip and r_hip:
        return (l_hip + r_hip) / 2
    return None


# =============================================================================
# MAIN IMPORT OPERATOR
# =============================================================================

class MELODICCAP_OT_import(bpy.types.Operator, ImportHelper):
    """Import MelodicCap take with proper root motion"""
    bl_idname = "melodiccap.import_take"
    bl_label = "Import Take"
    bl_options = {'REGISTER', 'UNDO'}
    
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    
    start_frame: IntProperty(
        name="Start Frame",
        default=1,
        min=1
    )
    
    clamp_limbs: BoolProperty(
        name="Clamp Limb Length",
        description="Prevent stretching beyond character's limb length",
        default=True
    )
    
    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature!")
            return {'CANCELLED'}
        
        # Load take
        with open(self.filepath, 'r') as f:
            data = json.load(f)
        
        frames = data.get('frames', [])
        if not frames:
            self.report({'ERROR'}, "No frames in take!")
            return {'CANCELLED'}
        
        debug("\n" + "="*70)
        debug("MELODICCAP v11.0 - WITH ROOT MOTION")
        debug("="*70)
        debug("Fix: Torso moves with hip, IK targets move relative to body")
        
        pose_bones = obj.pose.bones
        world = obj.matrix_world
        
        # === STEP 1: ANALYZE CHARACTER ===
        debug("\n" + "-"*50)
        debug("STEP 1: CHARACTER ANALYSIS")
        debug("-"*50)
        
        # Get character measurements
        char_arm_length = 0.52  # Default
        char_leg_length = 0.88  # Default
        
        if 'upper_arm_fk.L' in pose_bones and 'hand_fk.L' in pose_bones:
            s = world @ pose_bones['upper_arm_fk.L'].bone.head_local
            w = world @ pose_bones['hand_fk.L'].bone.head_local
            char_arm_length = (w - s).length
            debug(f"  Character arm length: {char_arm_length:.4f}m")
        
        if 'thigh_fk.L' in pose_bones and 'foot_fk.L' in pose_bones:
            h = world @ pose_bones['thigh_fk.L'].bone.head_local
            a = world @ pose_bones['foot_fk.L'].bone.head_local
            char_leg_length = (a - h).length
            debug(f"  Character leg length: {char_leg_length:.4f}m")
        
        # Max limb lengths for clamping
        max_lengths = {
            'hand_ik.R': char_arm_length,
            'hand_ik.L': char_arm_length,
            'foot_ik.R': char_leg_length,
            'foot_ik.L': char_leg_length,
        }
        
        # === STEP 2: ANALYZE MOCAP DATA ===
        debug("\n" + "-"*50)
        debug("STEP 2: MOCAP ANALYSIS")
        debug("-"*50)
        
        ref_lms = frames[0].get('landmarks', {})
        ref_hip = get_hip_center(ref_lms)
        
        if not ref_hip:
            self.report({'ERROR'}, "No hip in frame 0!")
            return {'CANCELLED'}
        
        debug(f"  Reference hip: ({ref_hip.x:.4f}, {ref_hip.y:.4f}, {ref_hip.z:.4f})")
        
        # Calculate scale from arm length
        l_shoulder = get_lm(ref_lms, 11)
        l_wrist = get_lm(ref_lms, 15)
        scale = 1.0
        if l_shoulder and l_wrist:
            person_arm = (l_wrist - l_shoulder).length
            scale = char_arm_length / person_arm
            debug(f"  Person arm: {person_arm:.4f}m")
            debug(f"  Scale factor: {scale:.4f}")
        
        # Store reference limb positions (relative to hip)
        ref_limb_rel = {}
        for ik_bone, config in RIGIFY_IK_MAP.items():
            lm = get_lm(ref_lms, config['landmark'])
            if lm:
                ref_limb_rel[ik_bone] = lm - ref_hip
                debug(f"  Ref {ik_bone}: ({ref_limb_rel[ik_bone].x:.3f}, {ref_limb_rel[ik_bone].y:.3f}, {ref_limb_rel[ik_bone].z:.3f})")
        
        # === STEP 3: SET UP RIG ===
        debug("\n" + "-"*50)
        debug("STEP 3: SET UP RIG")
        debug("-"*50)
        
        # Set IK mode
        for bone_name, prop_name in IK_FK_SWITCHES.items():
            if bone_name in pose_bones:
                pb = pose_bones[bone_name]
                if prop_name in pb:
                    pb[prop_name] = 0.0
                    debug(f"  Set {bone_name} to IK mode")
        
        # Set rotation mode
        for pb in pose_bones:
            pb.rotation_mode = 'QUATERNION'
        
        # Create action
        if not obj.animation_data:
            obj.animation_data_create()
        action = bpy.data.actions.new(name="MelodicCapAction")
        obj.animation_data.action = action
        
        # === STEP 4: ANIMATE ===
        debug("\n" + "-"*50)
        debug("STEP 4: ANIMATE")
        debug("-"*50)
        debug("  Strategy:")
        debug("    - TORSO: Moves with hip (ROOT MOTION)")
        debug("    - IK targets: Move relative to body ONLY")
        debug("    - This keeps limbs attached to body!")
        
        keyframes = 0
        bones_animated = set()
        
        for fidx, fdata in enumerate(frames):
            bf = self.start_frame + fidx
            lms = fdata.get('landmarks', {})
            
            current_hip = get_hip_center(lms)
            if not current_hip:
                continue
            
            # === ROOT MOTION (TORSO) ===
            # How much has the body moved from frame 0?
            hip_delta = current_hip - ref_hip
            scaled_hip_delta = hip_delta * scale
            
            if 'torso' in pose_bones:
                torso = pose_bones['torso']
                # Apply hip movement to torso (negate X for mirror)
                torso.location = Vector((-scaled_hip_delta.x, scaled_hip_delta.y, scaled_hip_delta.z))
                torso.keyframe_insert(data_path="location", frame=bf)
                keyframes += 1
                bones_animated.add('torso')
            
            # === IK TARGETS (LIMB MOVEMENT ONLY) ===
            # Since torso moves with hip, IK targets only need LIMB movement
            for ik_bone, config in RIGIFY_IK_MAP.items():
                if ik_bone not in pose_bones or ik_bone not in ref_limb_rel:
                    continue
                
                lm = get_lm(lms, config['landmark'])
                if not lm:
                    continue
                
                # Current limb position relative to current hip
                current_rel = lm - current_hip
                
                # Reference limb position relative to hip
                ref_rel = ref_limb_rel[ik_bone]
                
                # LIMB-ONLY delta (how much has limb moved relative to body?)
                limb_delta = current_rel - ref_rel
                scaled_limb_delta = limb_delta * scale
                
                # Clamp to max limb length if enabled
                if self.clamp_limbs:
                    max_len = max_lengths.get(ik_bone, 1.0)
                    if scaled_limb_delta.length > max_len * 0.5:  # Only clamp extreme movements
                        scaled_limb_delta = scaled_limb_delta.normalized() * max_len * 0.5
                
                # Apply to IK target (negate X for mirror)
                pb = pose_bones[ik_bone]
                pb.location = Vector((-scaled_limb_delta.x, scaled_limb_delta.y, scaled_limb_delta.z))
                pb.keyframe_insert(data_path="location", frame=bf)
                keyframes += 1
                bones_animated.add(ik_bone)
            
            # Progress
            if fidx % 50 == 0:
                debug(f"  Frame {fidx}/{len(frames)}")
                if fidx == 0:
                    debug(f"    Torso delta: ({-scaled_hip_delta.x:.4f}, {scaled_hip_delta.y:.4f}, {scaled_hip_delta.z:.4f})")
        
        # === SUMMARY ===
        debug("\n" + "="*70)
        debug("IMPORT SUMMARY")
        debug("="*70)
        debug(f"  Frames: {len(frames)}")
        debug(f"  Keyframes: {keyframes}")
        debug(f"  Scale: {scale:.4f}")
        debug(f"  Bones animated: {sorted(bones_animated)}")
        debug(f"  Method: ROOT MOTION (torso) + relative IK targets")
        
        self.report({'INFO'}, f"Imported {len(frames)} frames!")
        return {'FINISHED'}


# =============================================================================
# UTILITY OPERATORS
# =============================================================================

class MELODICCAP_OT_diagnose(bpy.types.Operator):
    """Print diagnostic info about the rig"""
    bl_idname = "melodiccap.diagnose"
    bl_label = "Diagnose Rig"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature!")
            return {'CANCELLED'}
        
        debug("\n" + "="*70)
        debug("RIG DIAGNOSTIC")
        debug("="*70)
        debug(f"Armature: {obj.name}")
        debug(f"Scale: {obj.scale}")
        
        if obj.scale != Vector((1, 1, 1)):
            debug("⚠️  WARNING: Scale is not (1,1,1)!")
        
        pose_bones = obj.pose.bones
        
        # IK/FK switches
        debug("\nIK/FK Switches:")
        for bone_name, prop_name in IK_FK_SWITCHES.items():
            if bone_name in pose_bones:
                pb = pose_bones[bone_name]
                if prop_name in pb:
                    val = pb[prop_name]
                    mode = "IK" if val == 0 else "FK"
                    debug(f"  {bone_name}.{prop_name} = {val} ({mode})")
        
        # Torso parent switch
        if 'torso' in pose_bones:
            torso = pose_bones['torso']
            if 'torso_parent' in torso:
                val = torso['torso_parent']
                parent = "Root" if val == 0 else "Torso"
                debug(f"\ntorso.torso_parent = {val} ({parent})")
                debug("  (IK targets follow torso when = 1)")
        
        debug("\n" + "="*70)
        self.report({'INFO'}, "Diagnostic printed to console!")
        return {'FINISHED'}


class MELODICCAP_OT_analyze_take(bpy.types.Operator, ImportHelper):
    """Analyze take file without importing"""
    bl_idname = "melodiccap.analyze_take"
    bl_label = "Analyze Take"
    bl_options = {'REGISTER'}
    
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    
    def execute(self, context):
        with open(self.filepath, 'r') as f:
            data = json.load(f)
        
        frames = data.get('frames', [])
        debug("\n" + "="*70)
        debug("TAKE ANALYSIS")
        debug("="*70)
        debug(f"Frames: {len(frames)}")
        
        if not frames:
            return {'CANCELLED'}
        
        # Track hip movement
        hip_positions = []
        for fdata in frames:
            lms = fdata.get('landmarks', {})
            hip = get_hip_center(lms)
            if hip:
                hip_positions.append(hip)
        
        if hip_positions:
            ref = hip_positions[0]
            debug(f"\nReference hip: ({ref.x:.3f}, {ref.y:.3f}, {ref.z:.3f})")
            
            deltas = [(p - ref) for p in hip_positions]
            max_delta = max(d.length for d in deltas)
            
            debug(f"Max movement from start: {max_delta:.3f}m")
            
            if max_delta > 0.5:
                debug(f"\n⚠️  Large movement ({max_delta:.2f}m) - ROOT MOTION is critical!")
        
        debug("\n" + "="*70)
        self.report({'INFO'}, f"Analyzed {len(frames)} frames")
        return {'FINISHED'}


class MELODICCAP_OT_set_ik_mode(bpy.types.Operator):
    """Set rig to IK mode"""
    bl_idname = "melodiccap.set_ik_mode"
    bl_label = "Set IK Mode"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature!")
            return {'CANCELLED'}
        
        for bone_name, prop_name in IK_FK_SWITCHES.items():
            if bone_name in obj.pose.bones:
                pb = obj.pose.bones[bone_name]
                if prop_name in pb:
                    pb[prop_name] = 0.0
        
        self.report({'INFO'}, "IK mode set!")
        return {'FINISHED'}


class MELODICCAP_OT_clear(bpy.types.Operator):
    """Clear animation"""
    bl_idname = "melodiccap.clear"
    bl_label = "Clear Animation"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature!")
            return {'CANCELLED'}
        
        if obj.animation_data:
            obj.animation_data.action = None
        
        for pb in obj.pose.bones:
            pb.location = Vector((0, 0, 0))
            pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
            pb.rotation_euler = Euler((0, 0, 0))
            pb.scale = Vector((1, 1, 1))
        
        debug("Cleared animation")
        self.report({'INFO'}, "Cleared!")
        return {'FINISHED'}


# =============================================================================
# PANEL
# =============================================================================

class MELODICCAP_PT_panel(bpy.types.Panel):
    bl_label = "MelodicCap v11"
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
            box.label(text=f"Armature: {obj.name}")
            if obj.scale != Vector((1, 1, 1)):
                box.label(text="⚠️ Scale not 1.0!", icon='ERROR')
        else:
            box.label(text="Select an armature!", icon='ERROR')
        
        # Diagnostics
        box = layout.box()
        box.label(text="🔍 Diagnostics", icon='VIEWZOOM')
        box.operator("melodiccap.diagnose", text="Diagnose Rig")
        box.operator("melodiccap.analyze_take", text="Analyze Take")
        
        # Preparation
        box = layout.box()
        box.label(text="⚙️ Preparation", icon='TOOL_SETTINGS')
        box.operator("melodiccap.set_ik_mode", text="Set IK Mode")
        box.operator("melodiccap.clear", text="Clear Animation")
        
        # Import
        box = layout.box()
        box.label(text="📥 Import", icon='IMPORT')
        box.operator("melodiccap.import_take", text="Import Take", icon='ARMATURE_DATA')
        
        # Info
        box = layout.box()
        box.label(text="v11 FIX:", icon='CHECKMARK')
        box.label(text="• Torso now moves with hip!")
        box.label(text="• IK = limb-relative only")
        box.label(text="• No more folded legs!")


# =============================================================================
# REGISTRATION
# =============================================================================

classes = [
    MELODICCAP_OT_import,
    MELODICCAP_OT_diagnose,
    MELODICCAP_OT_analyze_take,
    MELODICCAP_OT_set_ik_mode,
    MELODICCAP_OT_clear,
    MELODICCAP_PT_panel,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    debug("MelodicCap v11.0 registered - Now with ROOT MOTION!")

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
