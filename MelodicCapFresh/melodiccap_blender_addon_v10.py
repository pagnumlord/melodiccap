"""
MelodicCap Blender Addon v10.0
==============================
DIAGNOSTIC + GHOST RIG METHOD

Based on professional tools (Rokoko, FreeMoCap, Mocap Blender):
1. Creates EMPTIES for each MediaPipe landmark
2. Animates the empties from JSON data
3. CONSTRAINS Rigify IK controls to empties
4. BAKES animation with visual keying
5. Removes constraints (clean keyframes)

This is how professional tools work - let Blender's constraint
system and IK solver do the heavy lifting!

For Blender 4.4.3+ and JaxRigify (1.87m)
"""

bl_info = {
    "name": "MelodicCap Motion Capture Importer",
    "author": "Karsten / MelodicCap Studio",
    "version": (10, 0, 0),
    "blender": (4, 4, 0),
    "location": "View3D > Sidebar > MelodicCap",
    "description": "Import MelodicCap mocap data with diagnostic tools",
    "category": "Animation",
}

import bpy
import json
import math
from mathutils import Vector, Matrix, Quaternion, Euler
from bpy.props import StringProperty, FloatProperty, BoolProperty, IntProperty, EnumProperty
from bpy_extras.io_utils import ImportHelper

# =============================================================================
# CONSTANTS
# =============================================================================

# MediaPipe landmark indices
LANDMARK_NAMES = {
    0: "nose",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
}

# Rigify IK control mapping
# MediaPipe LEFT (person's left) -> Rigify RIGHT (character's right when facing camera)
RIGIFY_IK_MAP = {
    'hand_ik.R': {'landmark': 15, 'type': 'hand'},   # Person's left wrist
    'hand_ik.L': {'landmark': 16, 'type': 'hand'},   # Person's right wrist
    'foot_ik.R': {'landmark': 27, 'type': 'foot'},   # Person's left ankle
    'foot_ik.L': {'landmark': 28, 'type': 'foot'},   # Person's right ankle
}

# IK/FK switch bones and properties
IK_FK_SWITCHES = {
    'upper_arm_parent.L': 'IK_FK',
    'upper_arm_parent.R': 'IK_FK',
    'thigh_parent.L': 'IK_FK',
    'thigh_parent.R': 'IK_FK',
}

# Parent switch properties
PARENT_SWITCHES = {
    'torso': 'torso_parent',  # 0=root, 1=torso for IK targets
}

def debug(msg):
    """Print debug message"""
    print(f"[INFO] MelodicCap: {msg}")


# =============================================================================
# DIAGNOSTIC OPERATOR
# =============================================================================

class MELODICCAP_OT_diagnose_rig(bpy.types.Operator):
    """Diagnose the current state of the Rigify rig"""
    bl_idname = "melodiccap.diagnose_rig"
    bl_label = "Diagnose Rig"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj or obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature!")
            return {'CANCELLED'}
        
        debug("\n" + "="*70)
        debug("RIGIFY RIG DIAGNOSTIC REPORT")
        debug("="*70)
        debug(f"Armature: {obj.name}")
        debug(f"Location: {obj.location}")
        debug(f"Scale: {obj.scale}")
        
        if obj.scale != Vector((1, 1, 1)):
            debug("⚠️  WARNING: Scale is not (1,1,1)! Apply scale with Ctrl+A!")
        
        pose_bones = obj.pose.bones
        
        # --- IK/FK SWITCHES ---
        debug("\n" + "-"*50)
        debug("IK/FK SWITCH VALUES")
        debug("-"*50)
        
        for bone_name, prop_name in IK_FK_SWITCHES.items():
            if bone_name in pose_bones:
                pb = pose_bones[bone_name]
                if prop_name in pb:
                    value = pb[prop_name]
                    mode = "IK" if value == 0.0 else ("FK" if value == 1.0 else f"Blend {value}")
                    debug(f"  {bone_name}.{prop_name} = {value} ({mode})")
                else:
                    debug(f"  {bone_name}: Property '{prop_name}' NOT FOUND!")
            else:
                debug(f"  {bone_name}: Bone NOT FOUND!")
        
        # --- PARENT SWITCHES ---
        debug("\n" + "-"*50)
        debug("PARENT SWITCH VALUES (IK Target Parents)")
        debug("-"*50)
        
        for bone_name, prop_name in PARENT_SWITCHES.items():
            if bone_name in pose_bones:
                pb = pose_bones[bone_name]
                if prop_name in pb:
                    value = pb[prop_name]
                    parent_info = "Root" if value == 0 else ("Torso" if value == 1 else f"Custom({value})")
                    debug(f"  {bone_name}.{prop_name} = {value} ({parent_info})")
                else:
                    debug(f"  {bone_name}: Property '{prop_name}' NOT FOUND")
            else:
                debug(f"  {bone_name}: Bone NOT FOUND!")
        
        # Check hand_ik parent switch
        for side in ['.L', '.R']:
            bone_name = f'hand_ik{side}'
            if bone_name in pose_bones:
                pb = pose_bones[bone_name]
                # Check for any custom properties
                custom_props = [p for p in pb.keys() if not p.startswith('_')]
                debug(f"  {bone_name} custom properties: {custom_props}")
        
        # --- IK TARGET BONES ---
        debug("\n" + "-"*50)
        debug("IK TARGET BONE POSITIONS (World Space)")
        debug("-"*50)
        
        for ik_bone in RIGIFY_IK_MAP.keys():
            if ik_bone in pose_bones:
                pb = pose_bones[ik_bone]
                # Get world space position
                world_pos = obj.matrix_world @ pb.head
                rest_pos = obj.matrix_world @ pb.bone.head_local
                
                debug(f"\n  {ik_bone}:")
                debug(f"    Rest position:    ({rest_pos.x:.4f}, {rest_pos.y:.4f}, {rest_pos.z:.4f})")
                debug(f"    Current position: ({world_pos.x:.4f}, {world_pos.y:.4f}, {world_pos.z:.4f})")
                debug(f"    Local location:   ({pb.location.x:.4f}, {pb.location.y:.4f}, {pb.location.z:.4f})")
                debug(f"    Rotation mode:    {pb.rotation_mode}")
                
                # Check parent
                if pb.parent:
                    debug(f"    Parent bone:      {pb.parent.name}")
                
                # Check constraints
                if pb.constraints:
                    debug(f"    Constraints:      {[c.type for c in pb.constraints]}")
            else:
                debug(f"  {ik_bone}: NOT FOUND!")
        
        # --- TORSO/HIPS/CHEST ---
        debug("\n" + "-"*50)
        debug("BODY CONTROL BONES")
        debug("-"*50)
        
        for bone_name in ['torso', 'hips', 'chest', 'spine_fk', 'spine_fk.001']:
            if bone_name in pose_bones:
                pb = pose_bones[bone_name]
                world_pos = obj.matrix_world @ pb.head
                debug(f"\n  {bone_name}:")
                debug(f"    World position:   ({world_pos.x:.4f}, {world_pos.y:.4f}, {world_pos.z:.4f})")
                debug(f"    Local location:   ({pb.location.x:.4f}, {pb.location.y:.4f}, {pb.location.z:.4f})")
                debug(f"    Local rotation:   {pb.rotation_quaternion if pb.rotation_mode == 'QUATERNION' else pb.rotation_euler}")
                if pb.parent:
                    debug(f"    Parent bone:      {pb.parent.name}")
        
        # --- BONE HIERARCHY ---
        debug("\n" + "-"*50)
        debug("IK TARGET PARENT CHAIN")
        debug("-"*50)
        
        for ik_bone in ['hand_ik.L', 'foot_ik.L']:
            if ik_bone in pose_bones:
                pb = pose_bones[ik_bone]
                chain = [ik_bone]
                current = pb.parent
                while current and len(chain) < 10:
                    chain.append(current.name)
                    current = current.parent
                debug(f"  {ik_bone} chain: {' -> '.join(chain)}")
        
        # --- ANIMATION DATA ---
        debug("\n" + "-"*50)
        debug("ANIMATION DATA")
        debug("-"*50)
        
        if obj.animation_data and obj.animation_data.action:
            action = obj.animation_data.action
            debug(f"  Action: {action.name}")
            debug(f"  Frame range: {action.frame_range}")
            
            # Count keyframes per bone
            bone_keys = {}
            for fc in action.fcurves:
                if fc.data_path.startswith('pose.bones'):
                    bone = fc.data_path.split('"')[1]
                    bone_keys[bone] = bone_keys.get(bone, 0) + len(fc.keyframe_points)
            
            debug(f"  Bones with keyframes: {len(bone_keys)}")
            for bone, count in sorted(bone_keys.items()):
                debug(f"    {bone}: {count} keys")
        else:
            debug("  No animation data!")
        
        debug("\n" + "="*70)
        debug("END DIAGNOSTIC REPORT")
        debug("="*70 + "\n")
        
        self.report({'INFO'}, "Diagnostic report printed to console!")
        return {'FINISHED'}


# =============================================================================
# ANALYZE TAKE OPERATOR
# =============================================================================

class MELODICCAP_OT_analyze_take(bpy.types.Operator, ImportHelper):
    """Analyze a take file without importing"""
    bl_idname = "melodiccap.analyze_take"
    bl_label = "Analyze Take"
    bl_options = {'REGISTER'}
    
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    
    def execute(self, context):
        with open(self.filepath, 'r') as f:
            data = json.load(f)
        
        debug("\n" + "="*70)
        debug("TAKE FILE ANALYSIS")
        debug("="*70)
        debug(f"File: {self.filepath}")
        
        # Calibration info
        if 'calibration' in data:
            cal = data['calibration']
            debug(f"\nCalibration:")
            debug(f"  Floor offset: {cal.get('floor_offset', 'N/A')}")
            debug(f"  Stereo RMS: {cal.get('stereo_rms', 'N/A')}")
        
        frames = data.get('frames', [])
        debug(f"\nFrames: {len(frames)}")
        
        if not frames:
            self.report({'ERROR'}, "No frames in take!")
            return {'CANCELLED'}
        
        # Analyze landmarks
        first_frame = frames[0].get('landmarks', {})
        last_frame = frames[-1].get('landmarks', {}) if len(frames) > 1 else first_frame
        
        debug(f"Landmarks per frame: {len(first_frame)}")
        
        # Calculate person dimensions from first frame
        debug("\n" + "-"*50)
        debug("PERSON MEASUREMENTS (Frame 0)")
        debug("-"*50)
        
        def get_lm(lms, idx):
            data = lms.get(str(idx))
            if data:
                return Vector(data)
            return None
        
        nose = get_lm(first_frame, 0)
        l_hip = get_lm(first_frame, 23)
        r_hip = get_lm(first_frame, 24)
        l_ankle = get_lm(first_frame, 27)
        r_ankle = get_lm(first_frame, 28)
        l_shoulder = get_lm(first_frame, 11)
        r_shoulder = get_lm(first_frame, 12)
        l_wrist = get_lm(first_frame, 15)
        r_wrist = get_lm(first_frame, 16)
        
        if nose and l_ankle and r_ankle:
            ankle_mid_z = (l_ankle.z + r_ankle.z) / 2
            height = (nose.z - ankle_mid_z) + 0.15  # Add head top estimate
            debug(f"  Person height estimate: {height:.3f}m")
        
        if l_hip and r_hip:
            hip_center = (l_hip + r_hip) / 2
            debug(f"  Hip center: ({hip_center.x:.3f}, {hip_center.y:.3f}, {hip_center.z:.3f})")
            hip_width = (l_hip - r_hip).length
            debug(f"  Hip width: {hip_width:.3f}m")
        
        if l_shoulder and r_shoulder:
            shoulder_center = (l_shoulder + r_shoulder) / 2
            debug(f"  Shoulder center: ({shoulder_center.x:.3f}, {shoulder_center.y:.3f}, {shoulder_center.z:.3f})")
            shoulder_width = (l_shoulder - r_shoulder).length
            debug(f"  Shoulder width: {shoulder_width:.3f}m")
        
        # Arm lengths
        if l_shoulder and l_wrist:
            l_arm = (l_wrist - l_shoulder).length
            debug(f"  Left arm length: {l_arm:.3f}m")
        
        if r_shoulder and r_wrist:
            r_arm = (r_wrist - r_shoulder).length
            debug(f"  Right arm length: {r_arm:.3f}m")
        
        # Body orientation
        debug("\n" + "-"*50)
        debug("BODY ORIENTATION")
        debug("-"*50)
        
        if l_hip and r_hip:
            hip_vec = l_hip - r_hip
            hip_angle = math.degrees(math.atan2(hip_vec.y, hip_vec.x))
            debug(f"  Frame 0 hip facing: {hip_angle:.1f}°")
        
        if l_shoulder and r_shoulder:
            shoulder_vec = l_shoulder - r_shoulder
            shoulder_angle = math.degrees(math.atan2(shoulder_vec.y, shoulder_vec.x))
            debug(f"  Frame 0 shoulder facing: {shoulder_angle:.1f}°")
        
        # Movement range
        debug("\n" + "-"*50)
        debug("MOVEMENT RANGE ANALYSIS")
        debug("-"*50)
        
        # Track hip center movement through all frames
        hip_positions = []
        for fdata in frames:
            lms = fdata.get('landmarks', {})
            lh = get_lm(lms, 23)
            rh = get_lm(lms, 24)
            if lh and rh:
                hip_positions.append((lh + rh) / 2)
        
        if hip_positions:
            min_x = min(p.x for p in hip_positions)
            max_x = max(p.x for p in hip_positions)
            min_y = min(p.y for p in hip_positions)
            max_y = max(p.y for p in hip_positions)
            min_z = min(p.z for p in hip_positions)
            max_z = max(p.z for p in hip_positions)
            
            debug(f"  Hip X range: {min_x:.3f} to {max_x:.3f} (delta: {max_x-min_x:.3f}m)")
            debug(f"  Hip Y range: {min_y:.3f} to {max_y:.3f} (delta: {max_y-min_y:.3f}m)")
            debug(f"  Hip Z range: {min_z:.3f} to {max_z:.3f} (delta: {max_z-min_z:.3f}m)")
        
        # Rotation range
        debug("\n" + "-"*50)
        debug("ROTATION RANGE ANALYSIS")
        debug("-"*50)
        
        hip_rotations = []
        shoulder_rotations = []
        for fdata in frames:
            lms = fdata.get('landmarks', {})
            lh = get_lm(lms, 23)
            rh = get_lm(lms, 24)
            ls = get_lm(lms, 11)
            rs = get_lm(lms, 12)
            
            if lh and rh:
                vec = lh - rh
                angle = math.degrees(math.atan2(vec.y, vec.x))
                hip_rotations.append(angle)
            
            if ls and rs:
                vec = ls - rs
                angle = math.degrees(math.atan2(vec.y, vec.x))
                shoulder_rotations.append(angle)
        
        if hip_rotations:
            ref_hip = hip_rotations[0]
            deltas = [r - ref_hip for r in hip_rotations]
            debug(f"  Hip rotation range: {min(deltas):.1f}° to {max(deltas):.1f}° (from frame 0)")
        
        if shoulder_rotations:
            ref_shoulder = shoulder_rotations[0]
            deltas = [r - ref_shoulder for r in shoulder_rotations]
            debug(f"  Shoulder rotation range: {min(deltas):.1f}° to {max(deltas):.1f}° (from frame 0)")
        
        debug("\n" + "="*70)
        debug("END TAKE ANALYSIS")
        debug("="*70 + "\n")
        
        self.report({'INFO'}, f"Take analysis complete: {len(frames)} frames")
        return {'FINISHED'}


# =============================================================================
# SET IK MODE OPERATOR
# =============================================================================

class MELODICCAP_OT_set_ik_mode(bpy.types.Operator):
    """Force IK mode on all limbs"""
    bl_idname = "melodiccap.set_ik_mode"
    bl_label = "Set IK Mode"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature!")
            return {'CANCELLED'}
        
        pose_bones = obj.pose.bones
        
        for bone_name, prop_name in IK_FK_SWITCHES.items():
            if bone_name in pose_bones:
                pb = pose_bones[bone_name]
                if prop_name in pb:
                    pb[prop_name] = 0.0  # IK mode
                    debug(f"Set {bone_name}.{prop_name} = 0.0 (IK mode)")
        
        self.report({'INFO'}, "IK mode set on all limbs!")
        return {'FINISHED'}


# =============================================================================
# CLEAR ANIMATION OPERATOR
# =============================================================================

class MELODICCAP_OT_clear_animation(bpy.types.Operator):
    """Clear all animation from rig"""
    bl_idname = "melodiccap.clear_animation"
    bl_label = "Clear Animation"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature!")
            return {'CANCELLED'}
        
        # Clear action
        if obj.animation_data:
            obj.animation_data.action = None
        
        # Reset pose
        for pb in obj.pose.bones:
            pb.location = Vector((0, 0, 0))
            pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
            pb.rotation_euler = Euler((0, 0, 0))
            pb.scale = Vector((1, 1, 1))
        
        # Delete empties from previous import
        empties = [o for o in bpy.data.objects if o.name.startswith("MoCap_")]
        for empty in empties:
            bpy.data.objects.remove(empty, do_unlink=True)
        
        debug("Cleared animation and mocap empties")
        self.report({'INFO'}, "Animation cleared!")
        return {'FINISHED'}


# =============================================================================
# GHOST RIG IMPORT OPERATOR (THE MAIN ONE - PROFESSIONAL METHOD)
# =============================================================================

class MELODICCAP_OT_import_ghost_rig(bpy.types.Operator, ImportHelper):
    """Import mocap using Ghost Rig method (empties + constraints) - RECOMMENDED"""
    bl_idname = "melodiccap.import_ghost_rig"
    bl_label = "Import (Ghost Rig Method)"
    bl_options = {'REGISTER', 'UNDO'}
    
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    
    start_frame: IntProperty(
        name="Start Frame",
        default=1,
        min=1
    )
    
    auto_bake: BoolProperty(
        name="Auto Bake",
        description="Automatically bake constraints to keyframes",
        default=True
    )
    
    delete_empties: BoolProperty(
        name="Delete Empties After Bake",
        description="Remove empties and constraints after baking",
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
        debug("MELODICCAP v10.0 - GHOST RIG IMPORT")
        debug("="*70)
        debug("Method: Create empties → Constrain IK → Bake → Clean")
        debug("This is how Rokoko/FreeMoCap work - let Blender do the IK!")
        
        # --- STEP 1: CALCULATE PARAMETERS ---
        debug("\n" + "-"*50)
        debug("STEP 1: CALCULATE PARAMETERS")
        debug("-"*50)
        
        def get_lm(lms, idx):
            d = lms.get(str(idx))
            if d:
                return Vector(d)
            return None
        
        # Reference frame (frame 0)
        ref_lms = frames[0].get('landmarks', {})
        ref_hip = self._get_hip_center(ref_lms)
        
        if not ref_hip:
            self.report({'ERROR'}, "No hip data in frame 0!")
            return {'CANCELLED'}
        
        debug(f"  Reference hip (frame 0): ({ref_hip.x:.4f}, {ref_hip.y:.4f}, {ref_hip.z:.4f})")
        
        # Character measurements
        pose_bones = obj.pose.bones
        
        # Get Jax's shoulder to wrist length
        char_arm_length = 0.0
        if 'upper_arm_fk.L' in pose_bones and 'hand_fk.L' in pose_bones:
            shoulder = obj.matrix_world @ pose_bones['upper_arm_fk.L'].bone.head_local
            wrist = obj.matrix_world @ pose_bones['hand_fk.L'].bone.head_local
            char_arm_length = (wrist - shoulder).length
            debug(f"  Character arm length: {char_arm_length:.4f}m")
        
        # Get person's arm length
        person_arm_length = 0.0
        l_shoulder = get_lm(ref_lms, 11)
        l_wrist = get_lm(ref_lms, 15)
        if l_shoulder and l_wrist:
            person_arm_length = (l_wrist - l_shoulder).length
            debug(f"  Person arm length: {person_arm_length:.4f}m")
        
        # Calculate scale
        scale = 1.0
        if person_arm_length > 0 and char_arm_length > 0:
            scale = char_arm_length / person_arm_length
            debug(f"  Scale factor: {scale:.4f}")
        
        # Get character hip position (rest)
        char_hip_rest = Vector((0, 0, 0))
        if 'torso' in pose_bones:
            char_hip_rest = obj.matrix_world @ pose_bones['torso'].bone.head_local
            debug(f"  Character hip rest: ({char_hip_rest.x:.4f}, {char_hip_rest.y:.4f}, {char_hip_rest.z:.4f})")
        
        # --- STEP 2: CREATE EMPTIES ---
        debug("\n" + "-"*50)
        debug("STEP 2: CREATE EMPTIES")
        debug("-"*50)
        
        # Create parent empty for all mocap data
        parent_empty = bpy.data.objects.new("MoCap_Parent", None)
        parent_empty.empty_display_type = 'ARROWS'
        parent_empty.empty_display_size = 0.2
        context.scene.collection.objects.link(parent_empty)
        
        # Position parent at character's hip
        parent_empty.location = char_hip_rest
        debug(f"  Created MoCap_Parent at {char_hip_rest}")
        
        # Create empties for IK targets
        empties = {}
        for ik_bone, config in RIGIFY_IK_MAP.items():
            empty_name = f"MoCap_{ik_bone}"
            empty = bpy.data.objects.new(empty_name, None)
            empty.empty_display_type = 'SPHERE'
            empty.empty_display_size = 0.05
            empty.parent = parent_empty
            context.scene.collection.objects.link(empty)
            empties[ik_bone] = {'empty': empty, 'config': config}
            debug(f"  Created empty: {empty_name}")
        
        # --- STEP 3: ANIMATE EMPTIES ---
        debug("\n" + "-"*50)
        debug("STEP 3: ANIMATE EMPTIES")
        debug("-"*50)
        
        num_frames = len(frames)
        keyframes_created = 0
        
        # Store reference limb positions
        ref_limb_pos = {}
        for ik_bone, config in RIGIFY_IK_MAP.items():
            lm = get_lm(ref_lms, config['landmark'])
            if lm:
                ref_limb_pos[ik_bone] = lm - ref_hip
        
        for fidx, fdata in enumerate(frames):
            bf = self.start_frame + fidx
            lms = fdata.get('landmarks', {})
            
            current_hip = self._get_hip_center(lms)
            if not current_hip:
                continue
            
            # Hip movement from reference
            hip_delta = current_hip - ref_hip
            scaled_hip_delta = hip_delta * scale
            
            for ik_bone, emp_data in empties.items():
                empty = emp_data['empty']
                config = emp_data['config']
                
                # Get landmark position
                lm_pos = get_lm(lms, config['landmark'])
                
                if lm_pos and ik_bone in ref_limb_pos:
                    # Current position relative to hip
                    current_rel = lm_pos - current_hip
                    ref_rel = ref_limb_pos[ik_bone]
                    
                    # Limb delta
                    limb_delta = current_rel - ref_rel
                    scaled_limb_delta = limb_delta * scale
                    
                    # Total position = hip movement + limb movement
                    total = scaled_hip_delta + scaled_limb_delta
                    
                    # Apply to empty (relative to parent, which is at char hip)
                    # NEGATE X for mirroring (person facing camera)
                    empty.location = Vector((-total.x, total.y, total.z))
                    empty.keyframe_insert(data_path="location", frame=bf)
                    keyframes_created += 1
            
            if fidx % 50 == 0:
                debug(f"  Frame {fidx}/{num_frames}")
                if fidx == 0:
                    for ik_bone, emp_data in empties.items():
                        loc = emp_data['empty'].location
                        debug(f"    {ik_bone}: ({loc.x:.4f}, {loc.y:.4f}, {loc.z:.4f})")
        
        debug(f"  Total keyframes created: {keyframes_created}")
        
        # --- STEP 4: SET UP CONSTRAINTS ---
        debug("\n" + "-"*50)
        debug("STEP 4: SET UP CONSTRAINTS")
        debug("-"*50)
        
        # Force IK mode
        for bone_name, prop_name in IK_FK_SWITCHES.items():
            if bone_name in pose_bones:
                pb = pose_bones[bone_name]
                if prop_name in pb:
                    pb[prop_name] = 0.0
                    debug(f"  Set {bone_name} to IK mode")
        
        # Add constraints to IK targets
        constraints_added = []
        for ik_bone, emp_data in empties.items():
            if ik_bone in pose_bones:
                pb = pose_bones[ik_bone]
                empty = emp_data['empty']
                
                # Remove existing MoCap constraints
                to_remove = [c for c in pb.constraints if c.name.startswith('MoCap')]
                for c in to_remove:
                    pb.constraints.remove(c)
                
                # Add new constraint
                constraint = pb.constraints.new('COPY_LOCATION')
                constraint.name = 'MoCap_Location'
                constraint.target = empty
                constraint.use_offset = False
                constraints_added.append((ik_bone, constraint.name))
                debug(f"  Added COPY_LOCATION: {ik_bone} -> {empty.name}")
        
        # --- STEP 5: BAKE ANIMATION ---
        if self.auto_bake:
            debug("\n" + "-"*50)
            debug("STEP 5: BAKE ANIMATION (Visual Keying)")
            debug("-"*50)
            
            # Set frame range
            context.scene.frame_start = self.start_frame
            context.scene.frame_end = self.start_frame + num_frames - 1
            
            # Select armature and enter pose mode
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='POSE')
            
            # Select bones to bake
            bpy.ops.pose.select_all(action='DESELECT')
            for ik_bone in empties.keys():
                if ik_bone in pose_bones:
                    pose_bones[ik_bone].bone.select = True
                    debug(f"  Selected for bake: {ik_bone}")
            
            # Bake with visual keying
            debug(f"  Baking frames {self.start_frame} to {self.start_frame + num_frames - 1}...")
            try:
                bpy.ops.nla.bake(
                    frame_start=self.start_frame,
                    frame_end=self.start_frame + num_frames - 1,
                    visual_keying=True,
                    only_selected=True,
                    clear_constraints=self.delete_empties,
                    bake_types={'POSE'}
                )
                debug(f"  ✓ Baked {num_frames} frames with visual keying")
            except Exception as e:
                debug(f"  ✗ Bake error: {e}")
                self.report({'WARNING'}, f"Bake failed: {e}")
            
            # Clean up empties if requested
            if self.delete_empties:
                for emp_data in empties.values():
                    bpy.data.objects.remove(emp_data['empty'], do_unlink=True)
                bpy.data.objects.remove(parent_empty, do_unlink=True)
                debug("  Removed empties and constraints")
        else:
            debug("\n  Auto-bake disabled - empties and constraints left in scene")
            debug("  You can manually bake with: Pose > Animation > Bake Action")
        
        # --- SUMMARY ---
        debug("\n" + "="*70)
        debug("IMPORT SUMMARY")
        debug("="*70)
        debug(f"  Frames: {num_frames}")
        debug(f"  Scale: {scale:.4f}")
        debug(f"  Auto bake: {self.auto_bake}")
        debug(f"  IK bones: {list(empties.keys())}")
        debug(f"  Method: Ghost Rig (empties + constraints + visual keying)")
        
        self.report({'INFO'}, f"Imported {num_frames} frames!")
        return {'FINISHED'}
    
    def _get_hip_center(self, lms):
        """Get hip center from landmarks"""
        l_hip = lms.get('23')
        r_hip = lms.get('24')
        if l_hip and r_hip:
            return (Vector(l_hip) + Vector(r_hip)) / 2
        return None


# =============================================================================
# SIMPLE IK IMPORT (v4-style but with better debugging)
# =============================================================================

class MELODICCAP_OT_import_simple_ik(bpy.types.Operator, ImportHelper):
    """Import mocap with simple IK positioning (direct keyframes)"""
    bl_idname = "melodiccap.import_simple_ik"
    bl_label = "Import (Simple IK)"
    bl_options = {'REGISTER', 'UNDO'}
    
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    
    start_frame: IntProperty(
        name="Start Frame",
        default=1,
        min=1
    )
    
    clamp_limbs: BoolProperty(
        name="Clamp Limb Length",
        description="Prevent arms/legs from stretching beyond max length",
        default=True
    )
    
    add_torso_movement: BoolProperty(
        name="Add Torso Movement",
        description="Move torso with hip movement",
        default=True
    )
    
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
        debug("MELODICCAP v10.0 - SIMPLE IK IMPORT")
        debug("="*70)
        debug("Method: Direct IK target keyframing")
        
        pose_bones = obj.pose.bones
        
        # --- SETUP ---
        def get_lm(lms, idx):
            d = lms.get(str(idx))
            return Vector(d) if d else None
        
        ref_lms = frames[0].get('landmarks', {})
        ref_hip = self._get_hip_center(ref_lms)
        
        if not ref_hip:
            self.report({'ERROR'}, "No hip in frame 0!")
            return {'CANCELLED'}
        
        debug(f"  Reference hip: ({ref_hip.x:.4f}, {ref_hip.y:.4f}, {ref_hip.z:.4f})")
        
        # Calculate scale from arm length
        scale = 1.0
        l_shoulder = get_lm(ref_lms, 11)
        l_wrist = get_lm(ref_lms, 15)
        
        if l_shoulder and l_wrist:
            person_arm = (l_wrist - l_shoulder).length
            if 'upper_arm_fk.L' in pose_bones and 'hand_fk.L' in pose_bones:
                shoulder_pos = obj.matrix_world @ pose_bones['upper_arm_fk.L'].bone.head_local
                wrist_pos = obj.matrix_world @ pose_bones['hand_fk.L'].bone.head_local
                char_arm = (wrist_pos - shoulder_pos).length
                scale = char_arm / person_arm
                debug(f"  Person arm: {person_arm:.4f}m, Character arm: {char_arm:.4f}m")
        
        debug(f"  Scale factor: {scale:.4f}")
        
        # Calculate max limb lengths for clamping
        max_lengths = {}
        for ik_bone, config in RIGIFY_IK_MAP.items():
            if config['type'] == 'hand':
                if 'upper_arm_fk.L' in pose_bones and 'hand_fk.L' in pose_bones:
                    s = obj.matrix_world @ pose_bones['upper_arm_fk.L'].bone.head_local
                    w = obj.matrix_world @ pose_bones['hand_fk.L'].bone.head_local
                    max_lengths[ik_bone] = (w - s).length
            else:
                if 'thigh_fk.L' in pose_bones and 'foot_fk.L' in pose_bones:
                    h = obj.matrix_world @ pose_bones['thigh_fk.L'].bone.head_local
                    a = obj.matrix_world @ pose_bones['foot_fk.L'].bone.head_local
                    max_lengths[ik_bone] = (a - h).length
        
        debug(f"  Max limb lengths: {max_lengths}")
        
        # Store reference limb positions
        ref_limb_pos = {}
        for ik_bone, config in RIGIFY_IK_MAP.items():
            lm = get_lm(ref_lms, config['landmark'])
            if lm:
                ref_limb_pos[ik_bone] = lm - ref_hip
                debug(f"  Ref {ik_bone}: ({ref_limb_pos[ik_bone].x:.4f}, {ref_limb_pos[ik_bone].y:.4f}, {ref_limb_pos[ik_bone].z:.4f})")
        
        # Set IK mode
        for bone_name, prop_name in IK_FK_SWITCHES.items():
            if bone_name in pose_bones:
                pb = pose_bones[bone_name]
                if prop_name in pb:
                    pb[prop_name] = 0.0
        
        # Set rotation mode
        for pb in pose_bones:
            pb.rotation_mode = 'QUATERNION'
        
        # Create/get action
        if not obj.animation_data:
            obj.animation_data_create()
        action = bpy.data.actions.new(name="MelodicCapAction")
        obj.animation_data.action = action
        
        # --- ANIMATE ---
        debug("\n" + "-"*50)
        debug("ANIMATING FRAMES")
        debug("-"*50)
        
        keyframes = 0
        bones_animated = set()
        
        for fidx, fdata in enumerate(frames):
            bf = self.start_frame + fidx
            lms = fdata.get('landmarks', {})
            
            current_hip = self._get_hip_center(lms)
            if not current_hip:
                continue
            
            # Hip delta from reference
            hip_delta = current_hip - ref_hip
            scaled_hip_delta = hip_delta * scale
            
            # Apply to torso
            if self.add_torso_movement and 'torso' in pose_bones:
                torso = pose_bones['torso']
                # Negate X for mirroring
                torso.location = Vector((-scaled_hip_delta.x, scaled_hip_delta.y, scaled_hip_delta.z))
                torso.keyframe_insert(data_path="location", frame=bf)
                keyframes += 1
                bones_animated.add('torso')
            
            # IK targets
            for ik_bone, config in RIGIFY_IK_MAP.items():
                if ik_bone not in pose_bones:
                    continue
                
                lm = get_lm(lms, config['landmark'])
                if not lm or ik_bone not in ref_limb_pos:
                    continue
                
                # Current position relative to hip
                current_rel = lm - current_hip
                ref_rel = ref_limb_pos[ik_bone]
                
                # Limb delta
                limb_delta = current_rel - ref_rel
                scaled_limb_delta = limb_delta * scale
                
                # Total = hip movement + limb movement
                total = scaled_hip_delta + scaled_limb_delta
                
                # Clamp to max length
                if self.clamp_limbs and ik_bone in max_lengths:
                    max_len = max_lengths[ik_bone]
                    if total.length > max_len * 0.95:
                        total = total.normalized() * max_len * 0.95
                
                # Apply (negate X for mirror)
                pb = pose_bones[ik_bone]
                pb.location = Vector((-total.x, total.y, total.z))
                pb.keyframe_insert(data_path="location", frame=bf)
                keyframes += 1
                bones_animated.add(ik_bone)
            
            if fidx % 50 == 0:
                debug(f"  Frame {fidx}/{len(frames)}")
        
        # --- SUMMARY ---
        debug("\n" + "="*70)
        debug("IMPORT SUMMARY")  
        debug("="*70)
        debug(f"  Frames: {len(frames)}")
        debug(f"  Keyframes: {keyframes}")
        debug(f"  Bones: {sorted(bones_animated)}")
        
        self.report({'INFO'}, f"Imported {len(frames)} frames!")
        return {'FINISHED'}
    
    def _get_hip_center(self, lms):
        l = lms.get('23')
        r = lms.get('24')
        if l and r:
            return (Vector(l) + Vector(r)) / 2
        return None


# =============================================================================
# PANEL
# =============================================================================

class MELODICCAP_PT_panel(bpy.types.Panel):
    bl_label = "MelodicCap v10"
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
                box.label(text="⚠️ Scale not (1,1,1)!", icon='ERROR')
        else:
            box.label(text="Select an armature!", icon='ERROR')
        
        # Diagnostic Tools
        box = layout.box()
        box.label(text="🔍 Diagnostic Tools", icon='VIEWZOOM')
        box.operator("melodiccap.diagnose_rig", text="Diagnose Rig", icon='ARMATURE_DATA')
        box.operator("melodiccap.analyze_take", text="Analyze Take File", icon='FILE')
        
        # Preparation
        box = layout.box()
        box.label(text="⚙️ Preparation", icon='TOOL_SETTINGS')
        box.operator("melodiccap.set_ik_mode", text="Set IK Mode", icon='CON_KINEMATIC')
        box.operator("melodiccap.clear_animation", text="Clear Animation", icon='TRASH')
        
        # Import Methods
        box = layout.box()
        box.label(text="📥 Import Methods", icon='IMPORT')
        
        col = box.column()
        col.label(text="Ghost Rig (Recommended):")
        col.operator("melodiccap.import_ghost_rig", text="Import with Empties+Bake", icon='GHOST_ENABLED')
        
        col = box.column()
        col.label(text="Simple IK (v4-style):")
        col.operator("melodiccap.import_simple_ik", text="Import Direct Keyframes", icon='KEY_HLT')
        
        # Info
        box = layout.box()
        box.label(text="ℹ️ Tips", icon='QUESTION')
        box.label(text="1. Run 'Diagnose Rig' first")
        box.label(text="2. Use 'Set IK Mode' before import")
        box.label(text="3. Check console for detailed output")


# =============================================================================
# REGISTRATION
# =============================================================================

classes = [
    MELODICCAP_OT_diagnose_rig,
    MELODICCAP_OT_analyze_take,
    MELODICCAP_OT_set_ik_mode,
    MELODICCAP_OT_clear_animation,
    MELODICCAP_OT_import_ghost_rig,
    MELODICCAP_OT_import_simple_ik,
    MELODICCAP_PT_panel,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    debug("MelodicCap v10.0 registered - Diagnostic + Ghost Rig Method!")

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
