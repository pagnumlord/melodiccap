"""
MelodicCap Blender Addon v2.0
=============================
Imports motion capture data and retargets to Rigify characters.

Features:
- Full debugging output
- Automatic scale detection  
- FK and IK mode support
- Bone rotation calculation from positions
- Animation baking
- JaxRigify 1.87m character support

For Blender 4.4.3+
"""

bl_info = {
    "name": "MelodicCap Motion Capture Importer",
    "author": "Karsten / MelodicCap Studio",
    "version": (2, 0, 0),
    "blender": (4, 4, 0),
    "location": "View3D > Sidebar > MelodicCap",
    "description": "Import MelodicCap motion capture data to Rigify characters",
    "category": "Animation",
}

import bpy
import json
import math
from mathutils import Vector, Matrix, Quaternion, Euler
from bpy.props import StringProperty, FloatProperty, BoolProperty, EnumProperty, IntProperty
from bpy_extras.io_utils import ImportHelper

# =============================================================================
# MEDIAPIPE LANDMARK DEFINITIONS
# =============================================================================

MEDIAPIPE_LANDMARKS = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def debug_print(msg, level="INFO"):
    """Print debug message to console"""
    prefix = {"INFO": "[INFO]", "WARN": "[WARN]", "ERROR": "[ERROR]", "DEBUG": "[DEBUG]"}
    print(f"{prefix.get(level, '[???]')} MelodicCap: {msg}")


def get_landmark_position(landmarks, idx):
    """Get landmark position as Vector, handling string keys from JSON"""
    key = str(idx) if str(idx) in landmarks else idx
    if key in landmarks:
        pos = landmarks[key]
        return Vector((pos[0], pos[1], pos[2]))
    return None


def calculate_midpoint(landmarks, idx1, idx2):
    """Calculate midpoint between two landmarks"""
    p1 = get_landmark_position(landmarks, idx1)
    p2 = get_landmark_position(landmarks, idx2)
    if p1 and p2:
        return (p1 + p2) / 2
    return None


def vector_to_rotation(direction, up_hint=None):
    """Convert direction vector to quaternion rotation"""
    if direction.length < 0.0001:
        return Quaternion()
    
    direction = direction.normalized()
    
    # Default up direction
    if up_hint is None:
        up_hint = Vector((0, 0, 1))
    
    # Build rotation matrix
    # Z axis = up, Y axis = forward (bone direction), X axis = right
    forward = direction
    right = forward.cross(up_hint)
    
    if right.length < 0.0001:
        # direction is parallel to up, choose different up
        up_hint = Vector((0, 1, 0))
        right = forward.cross(up_hint)
    
    right.normalize()
    up = right.cross(forward).normalized()
    
    # Create rotation matrix (column vectors are the axes)
    mat = Matrix((
        (right.x, forward.x, up.x),
        (right.y, forward.y, up.y),
        (right.z, forward.z, up.z),
    ))
    
    return mat.to_quaternion()


# =============================================================================
# MAIN IMPORTER CLASS
# =============================================================================

class MelodicCapImporter:
    """Main importer class with extensive debugging"""
    
    # Jax's height in meters
    JAX_HEIGHT = 1.87
    
    def __init__(self, armature_obj, take_data, settings):
        self.armature = armature_obj
        self.take_data = take_data
        self.settings = settings
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'keyframes_set': 0,
            'bones_animated': set(),
            'errors': [],
            'warnings': [],
        }
        
        # Character analysis
        self.char_height = 0
        self.mocap_height = 0
        self.scale_factor = 1.0
        
        # Rest pose reference positions
        self.rest_positions = {}
    
    def analyze_character(self):
        """Analyze character rig dimensions"""
        debug_print("="*60)
        debug_print("ANALYZING CHARACTER RIG")
        debug_print("="*60)
        debug_print(f"  Armature: {self.armature.name}")
        
        armature_data = self.armature.data
        
        # Get world transform
        world_matrix = self.armature.matrix_world
        
        # Find height by looking at bone positions
        min_z = float('inf')
        max_z = float('-inf')
        
        # Key bone positions for reference
        key_bones = {}
        bone_names_to_find = [
            'head', 'neck', 'chest', 'torso', 'hips', 'root',
            'ORG-spine', 'ORG-spine.006',
            'thigh_fk.L', 'thigh_fk.R', 
            'shin_fk.L', 'shin_fk.R',
            'foot_fk.L', 'foot_fk.R',
            'upper_arm_fk.L', 'upper_arm_fk.R',
            'forearm_fk.L', 'forearm_fk.R',
            'hand_fk.L', 'hand_fk.R',
            'hand_ik.L', 'hand_ik.R',
            'foot_ik.L', 'foot_ik.R',
        ]
        
        debug_print("\n  Key bone positions (world space):")
        for bone_name in bone_names_to_find:
            if bone_name in armature_data.bones:
                bone = armature_data.bones[bone_name]
                head_world = world_matrix @ bone.head_local
                tail_world = world_matrix @ bone.tail_local
                
                key_bones[bone_name] = {
                    'head': head_world.copy(),
                    'tail': tail_world.copy(),
                    'length': (tail_world - head_world).length
                }
                
                min_z = min(min_z, head_world.z, tail_world.z)
                max_z = max(max_z, head_world.z, tail_world.z)
                
                debug_print(f"    {bone_name}:")
                debug_print(f"      Head: ({head_world.x:.3f}, {head_world.y:.3f}, {head_world.z:.3f})")
                debug_print(f"      Tail: ({tail_world.x:.3f}, {tail_world.y:.3f}, {tail_world.z:.3f})")
                debug_print(f"      Length: {key_bones[bone_name]['length']:.4f}m")
        
        # Calculate character height
        if max_z > min_z:
            self.char_height = max_z - min_z
            debug_print(f"\n  Character height from bones: {self.char_height:.3f}m")
        else:
            self.char_height = self.JAX_HEIGHT
            debug_print(f"\n  Using default Jax height: {self.char_height:.3f}m")
        
        self.rest_positions = key_bones
        
        # Count available bones
        fk_bones = ['upper_arm_fk.L', 'upper_arm_fk.R', 'forearm_fk.L', 'forearm_fk.R',
                    'thigh_fk.L', 'thigh_fk.R', 'shin_fk.L', 'shin_fk.R',
                    'hand_fk.L', 'hand_fk.R', 'foot_fk.L', 'foot_fk.R']
        ik_bones = ['hand_ik.L', 'hand_ik.R', 'foot_ik.L', 'foot_ik.R']
        
        fk_found = sum(1 for b in fk_bones if b in armature_data.bones)
        ik_found = sum(1 for b in ik_bones if b in armature_data.bones)
        
        debug_print(f"\n  FK bones available: {fk_found}/{len(fk_bones)}")
        debug_print(f"  IK bones available: {ik_found}/{len(ik_bones)}")
        
        return True
    
    def analyze_mocap_data(self):
        """Analyze motion capture data dimensions"""
        debug_print("\n" + "="*60)
        debug_print("ANALYZING MOCAP DATA")
        debug_print("="*60)
        
        frames = self.take_data.get('frames', [])
        if not frames:
            debug_print("  ERROR: No frames in take data!", "ERROR")
            return False
        
        debug_print(f"  Version: {self.take_data.get('version', 'unknown')}")
        debug_print(f"  Total frames: {len(frames)}")
        debug_print(f"  Duration: {self.take_data.get('duration_seconds', 0):.2f}s")
        debug_print(f"  Predicted frames: {self.take_data.get('predicted_frames', 0)}")
        debug_print(f"  Dropped frames: {self.take_data.get('dropped_frames', 0)}")
        
        # Calibration info
        calib = self.take_data.get('calibration', {})
        debug_print(f"\n  Calibration:")
        debug_print(f"    Stereo RMS: {calib.get('rms_stereo', 'N/A')}")
        debug_print(f"    Baseline: {calib.get('baseline', 'N/A')}m")
        debug_print(f"    Floor offset: {calib.get('floor_offset', 'N/A')}m")
        
        # Analyze first frame
        first_frame = frames[0]
        landmarks = first_frame.get('landmarks', {})
        debug_print(f"\n  Landmarks in first frame: {len(landmarks)}")
        
        # Calculate mocap body dimensions
        debug_print("\n  First frame landmark positions:")
        
        # Key landmarks
        nose = get_landmark_position(landmarks, 0)
        left_shoulder = get_landmark_position(landmarks, 11)
        right_shoulder = get_landmark_position(landmarks, 12)
        left_hip = get_landmark_position(landmarks, 23)
        right_hip = get_landmark_position(landmarks, 24)
        left_ankle = get_landmark_position(landmarks, 27)
        right_ankle = get_landmark_position(landmarks, 28)
        
        for idx in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            pos = get_landmark_position(landmarks, idx)
            if pos:
                name = MEDIAPIPE_LANDMARKS.get(idx, f"lm_{idx}")
                debug_print(f"    [{idx:2d}] {name:20s}: ({pos.x:7.3f}, {pos.y:7.3f}, {pos.z:7.3f})")
        
        # Calculate heights and dimensions
        if nose and left_ankle and right_ankle:
            ankle_mid = (left_ankle + right_ankle) / 2
            head_to_ankle = nose.z - ankle_mid.z
            # Add estimate for top of head above nose (~15cm)
            self.mocap_height = head_to_ankle + 0.15
            debug_print(f"\n  MoCap person height: {self.mocap_height:.3f}m")
            debug_print(f"    (Nose Z: {nose.z:.3f}, Ankle mid Z: {ankle_mid.z:.3f}, +0.15 for head)")
        
        if left_shoulder and right_shoulder:
            shoulder_width = (left_shoulder - right_shoulder).length
            debug_print(f"  MoCap shoulder width: {shoulder_width:.3f}m")
        
        if left_hip and right_hip:
            hip_width = (left_hip - right_hip).length
            hip_mid = (left_hip + right_hip) / 2
            debug_print(f"  MoCap hip width: {hip_width:.3f}m")
            debug_print(f"  MoCap hip height: {hip_mid.z:.3f}m")
        
        # Calculate scale factor
        if self.mocap_height > 0 and self.char_height > 0:
            self.scale_factor = self.char_height / self.mocap_height
            debug_print(f"\n  === SCALE FACTOR: {self.scale_factor:.4f} ===")
            debug_print(f"    Character: {self.char_height:.3f}m")
            debug_print(f"    MoCap: {self.mocap_height:.3f}m")
        
        # Analyze motion range
        debug_print("\n  Analyzing motion range across all frames...")
        
        min_pos = Vector((float('inf'), float('inf'), float('inf')))
        max_pos = Vector((float('-inf'), float('-inf'), float('-inf')))
        
        for frame in frames:
            landmarks = frame.get('landmarks', {})
            for idx in range(33):
                pos = get_landmark_position(landmarks, idx)
                if pos:
                    min_pos.x = min(min_pos.x, pos.x)
                    min_pos.y = min(min_pos.y, pos.y)
                    min_pos.z = min(min_pos.z, pos.z)
                    max_pos.x = max(max_pos.x, pos.x)
                    max_pos.y = max(max_pos.y, pos.y)
                    max_pos.z = max(max_pos.z, pos.z)
        
        debug_print(f"  Motion bounding box:")
        debug_print(f"    Min: ({min_pos.x:.3f}, {min_pos.y:.3f}, {min_pos.z:.3f})")
        debug_print(f"    Max: ({max_pos.x:.3f}, {max_pos.y:.3f}, {max_pos.z:.3f})")
        debug_print(f"    Range: ({max_pos.x-min_pos.x:.3f}, {max_pos.y-min_pos.y:.3f}, {max_pos.z-min_pos.z:.3f})")
        
        return True
    
    def apply_animation_fk(self):
        """Apply animation using FK bones"""
        debug_print("\n" + "="*60)
        debug_print("APPLYING FK ANIMATION")
        debug_print("="*60)
        
        frames = self.take_data.get('frames', [])
        pose_bones = self.armature.pose.bones
        
        # Define bone chains to animate
        # Each chain: (bone_name, start_landmark, end_landmark)
        bone_chains = [
            # Left arm
            ('upper_arm_fk.L', 11, 13),  # shoulder to elbow
            ('forearm_fk.L', 13, 15),    # elbow to wrist
            
            # Right arm  
            ('upper_arm_fk.R', 12, 14),  # shoulder to elbow
            ('forearm_fk.R', 14, 16),    # elbow to wrist
            
            # Left leg
            ('thigh_fk.L', 23, 25),      # hip to knee
            ('shin_fk.L', 25, 27),       # knee to ankle
            
            # Right leg
            ('thigh_fk.R', 24, 26),      # hip to knee
            ('shin_fk.R', 26, 28),       # knee to ankle
        ]
        
        # Check which bones exist
        available_chains = []
        for bone_name, start_lm, end_lm in bone_chains:
            if bone_name in pose_bones:
                available_chains.append((bone_name, start_lm, end_lm))
                debug_print(f"  ✓ {bone_name}")
            else:
                debug_print(f"  ✗ {bone_name} (missing)", "WARN")
        
        debug_print(f"\n  Animating {len(available_chains)} bone chains...")
        
        # Get armature world matrix inverse for converting to local space
        armature_inv = self.armature.matrix_world.inverted()
        
        # Process each frame
        start_frame = self.settings.get('start_frame', 1)
        
        for frame_idx, frame_data in enumerate(frames):
            blender_frame = start_frame + frame_idx
            bpy.context.scene.frame_set(blender_frame)
            
            landmarks = frame_data.get('landmarks', {})
            
            # Calculate hip center for root motion
            hip_left = get_landmark_position(landmarks, 23)
            hip_right = get_landmark_position(landmarks, 24)
            
            if hip_left and hip_right:
                hip_center = (hip_left + hip_right) / 2 * self.scale_factor
                
                # Move root/torso bone
                if 'torso' in pose_bones:
                    torso_bone = pose_bones['torso']
                    # Convert to armature local space
                    local_pos = armature_inv @ hip_center
                    torso_bone.location = local_pos
                    torso_bone.keyframe_insert(data_path="location", frame=blender_frame)
                    self.stats['keyframes_set'] += 1
            
            # Process each bone chain
            for bone_name, start_lm, end_lm in available_chains:
                start_pos = get_landmark_position(landmarks, start_lm)
                end_pos = get_landmark_position(landmarks, end_lm)
                
                if start_pos is None or end_pos is None:
                    continue
                
                # Scale positions
                start_pos = start_pos * self.scale_factor
                end_pos = end_pos * self.scale_factor
                
                # Calculate direction
                direction = (end_pos - start_pos).normalized()
                
                if direction.length < 0.001:
                    continue
                
                pose_bone = pose_bones[bone_name]
                
                # Get bone's rest axis (Y axis in bone local space is along the bone)
                bone_rest_matrix = pose_bone.bone.matrix_local
                rest_direction = (bone_rest_matrix @ Vector((0, 1, 0)) - 
                                  bone_rest_matrix @ Vector((0, 0, 0))).normalized()
                
                # Calculate rotation needed
                rotation = rest_direction.rotation_difference(direction)
                
                # Apply to bone
                pose_bone.rotation_mode = 'QUATERNION'
                pose_bone.rotation_quaternion = rotation
                pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=blender_frame)
                
                self.stats['keyframes_set'] += 1
                self.stats['bones_animated'].add(bone_name)
            
            self.stats['frames_processed'] += 1
            
            if frame_idx % 50 == 0:
                debug_print(f"    Frame {frame_idx}/{len(frames)}")
        
        return True
    
    def apply_animation_ik(self):
        """Apply animation using IK targets"""
        debug_print("\n" + "="*60)
        debug_print("APPLYING IK ANIMATION")
        debug_print("="*60)
        
        frames = self.take_data.get('frames', [])
        pose_bones = self.armature.pose.bones
        
        # IK targets: (bone_name, landmark_index)
        ik_targets = [
            ('hand_ik.L', 15),  # left wrist
            ('hand_ik.R', 16),  # right wrist
            ('foot_ik.L', 27),  # left ankle
            ('foot_ik.R', 28),  # right ankle
        ]
        
        # Check which IK bones exist
        available_targets = []
        for bone_name, lm_idx in ik_targets:
            if bone_name in pose_bones:
                available_targets.append((bone_name, lm_idx))
                debug_print(f"  ✓ {bone_name}")
            else:
                debug_print(f"  ✗ {bone_name} (missing)", "WARN")
        
        debug_print(f"\n  Animating {len(available_targets)} IK targets...")
        
        armature_inv = self.armature.matrix_world.inverted()
        start_frame = self.settings.get('start_frame', 1)
        
        for frame_idx, frame_data in enumerate(frames):
            blender_frame = start_frame + frame_idx
            bpy.context.scene.frame_set(blender_frame)
            
            landmarks = frame_data.get('landmarks', {})
            
            # Root motion
            hip_left = get_landmark_position(landmarks, 23)
            hip_right = get_landmark_position(landmarks, 24)
            
            if hip_left and hip_right:
                hip_center = (hip_left + hip_right) / 2 * self.scale_factor
                
                if 'torso' in pose_bones:
                    torso_bone = pose_bones['torso']
                    local_pos = armature_inv @ hip_center
                    torso_bone.location = local_pos
                    torso_bone.keyframe_insert(data_path="location", frame=blender_frame)
                    self.stats['keyframes_set'] += 1
            
            # IK targets
            for bone_name, lm_idx in available_targets:
                pos = get_landmark_position(landmarks, lm_idx)
                if pos is None:
                    continue
                
                scaled_pos = pos * self.scale_factor
                pose_bone = pose_bones[bone_name]
                
                # Convert to bone's local space
                local_pos = armature_inv @ scaled_pos
                
                pose_bone.location = local_pos
                pose_bone.keyframe_insert(data_path="location", frame=blender_frame)
                
                self.stats['keyframes_set'] += 1
                self.stats['bones_animated'].add(bone_name)
            
            self.stats['frames_processed'] += 1
            
            if frame_idx % 50 == 0:
                debug_print(f"    Frame {frame_idx}/{len(frames)}")
        
        return True
    
    def apply_animation(self):
        """Apply animation based on settings"""
        if self.settings.get('use_ik', False):
            return self.apply_animation_ik()
        else:
            return self.apply_animation_fk()
    
    def print_summary(self):
        """Print final summary"""
        debug_print("\n" + "="*60)
        debug_print("IMPORT SUMMARY")
        debug_print("="*60)
        debug_print(f"  Target: {self.armature.name}")
        debug_print(f"  Mode: {'IK' if self.settings.get('use_ik') else 'FK'}")
        debug_print(f"  Character height: {self.char_height:.3f}m")
        debug_print(f"  MoCap height: {self.mocap_height:.3f}m")
        debug_print(f"  Scale factor: {self.scale_factor:.4f}")
        debug_print(f"  Frames processed: {self.stats['frames_processed']}")
        debug_print(f"  Keyframes created: {self.stats['keyframes_set']}")
        debug_print(f"  Bones animated: {len(self.stats['bones_animated'])}")
        
        if self.stats['bones_animated']:
            debug_print("\n  Animated bones:")
            for bone in sorted(self.stats['bones_animated']):
                debug_print(f"    - {bone}")


# =============================================================================
# BLENDER OPERATORS
# =============================================================================

class MELODICCAP_OT_import_take(bpy.types.Operator, ImportHelper):
    """Import MelodicCap motion capture take"""
    bl_idname = "melodiccap.import_take"
    bl_label = "Import MelodicCap Take"
    bl_options = {'REGISTER', 'UNDO'}
    
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    
    def execute(self, context):
        debug_print("\n" + "="*60)
        debug_print("MELODICCAP IMPORT STARTED")
        debug_print("="*60)
        debug_print(f"  File: {self.filepath}")
        
        # Check for selected armature
        armature = context.active_object
        if not armature or armature.type != 'ARMATURE':
            self.report({'ERROR'}, "Please select an armature first!")
            return {'CANCELLED'}
        
        debug_print(f"  Target armature: {armature.name}")
        
        # Load take data
        try:
            with open(self.filepath, 'r') as f:
                take_data = json.load(f)
            debug_print(f"  Take version: {take_data.get('version', 'unknown')}")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load take: {e}")
            debug_print(f"  ERROR: {e}", "ERROR")
            return {'CANCELLED'}
        
        # Get settings
        settings = {
            'use_ik': context.scene.melodiccap_use_ik,
            'start_frame': context.scene.melodiccap_start_frame,
            'auto_scale': context.scene.melodiccap_auto_scale,
            'scale_override': context.scene.melodiccap_scale_override,
        }
        
        # Create importer
        importer = MelodicCapImporter(armature, take_data, settings)
        
        # Analyze
        importer.analyze_character()
        importer.analyze_mocap_data()
        
        # Override scale if needed
        if not settings['auto_scale']:
            importer.scale_factor = settings['scale_override']
            debug_print(f"  Using manual scale override: {importer.scale_factor}")
        
        # Ensure pose mode
        bpy.ops.object.mode_set(mode='POSE')
        
        # Set rotation mode on all bones
        for pose_bone in armature.pose.bones:
            pose_bone.rotation_mode = 'QUATERNION'
        
        # Apply animation
        success = importer.apply_animation()
        
        # Print summary
        importer.print_summary()
        
        if success:
            self.report({'INFO'}, f"Imported {importer.stats['frames_processed']} frames")
        else:
            self.report({'WARNING'}, "Import completed with issues - check console")
        
        return {'FINISHED'}


class MELODICCAP_OT_analyze_rig(bpy.types.Operator):
    """Analyze selected rig for MelodicCap compatibility"""
    bl_idname = "melodiccap.analyze_rig"
    bl_label = "Analyze Rig"
    
    def execute(self, context):
        armature = context.active_object
        if not armature or armature.type != 'ARMATURE':
            self.report({'ERROR'}, "Please select an armature!")
            return {'CANCELLED'}
        
        debug_print("\n" + "="*60)
        debug_print(f"RIG ANALYSIS: {armature.name}")
        debug_print("="*60)
        
        bones = armature.data.bones
        
        # Check FK bones
        fk_list = ['upper_arm_fk.L', 'upper_arm_fk.R', 'forearm_fk.L', 'forearm_fk.R',
                   'thigh_fk.L', 'thigh_fk.R', 'shin_fk.L', 'shin_fk.R',
                   'hand_fk.L', 'hand_fk.R', 'foot_fk.L', 'foot_fk.R']
        
        debug_print("\n  FK Bones:")
        fk_count = 0
        for b in fk_list:
            if b in bones:
                debug_print(f"    ✓ {b}")
                fk_count += 1
            else:
                debug_print(f"    ✗ {b}")
        
        # Check IK bones
        ik_list = ['hand_ik.L', 'hand_ik.R', 'foot_ik.L', 'foot_ik.R']
        
        debug_print("\n  IK Bones:")
        ik_count = 0
        for b in ik_list:
            if b in bones:
                debug_print(f"    ✓ {b}")
                ik_count += 1
            else:
                debug_print(f"    ✗ {b}")
        
        # Check core bones
        core_list = ['torso', 'hips', 'chest', 'head', 'neck', 'root']
        
        debug_print("\n  Core Bones:")
        core_count = 0
        for b in core_list:
            if b in bones:
                debug_print(f"    ✓ {b}")
                core_count += 1
            else:
                debug_print(f"    ✗ {b}")
        
        debug_print(f"\n  Summary: FK={fk_count}/{len(fk_list)}, IK={ik_count}/{len(ik_list)}, Core={core_count}/{len(core_list)}")
        
        self.report({'INFO'}, f"FK={fk_count}, IK={ik_count}, Core={core_count} - see console")
        return {'FINISHED'}


class MELODICCAP_OT_clear_animation(bpy.types.Operator):
    """Clear all animation from selected armature"""
    bl_idname = "melodiccap.clear_animation"
    bl_label = "Clear Animation"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        armature = context.active_object
        if not armature or armature.type != 'ARMATURE':
            self.report({'ERROR'}, "Please select an armature!")
            return {'CANCELLED'}
        
        if armature.animation_data:
            armature.animation_data_clear()
        
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.pose.transforms_clear()
        
        debug_print(f"Cleared animation from {armature.name}")
        self.report({'INFO'}, f"Cleared animation from {armature.name}")
        return {'FINISHED'}


# =============================================================================
# UI PANEL
# =============================================================================

class MELODICCAP_PT_main_panel(bpy.types.Panel):
    """MelodicCap main panel"""
    bl_label = "MelodicCap"
    bl_idname = "MELODICCAP_PT_main_panel"
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
        else:
            box.label(text="  (Select armature)", icon='ERROR')
        
        # Settings
        box = layout.box()
        box.label(text="Settings:", icon='SETTINGS')
        box.prop(context.scene, "melodiccap_use_ik")
        box.prop(context.scene, "melodiccap_start_frame")
        box.prop(context.scene, "melodiccap_auto_scale")
        if not context.scene.melodiccap_auto_scale:
            box.prop(context.scene, "melodiccap_scale_override")
        
        # Actions
        box = layout.box()
        box.label(text="Actions:", icon='ACTION')
        box.operator("melodiccap.import_take", icon='IMPORT')
        box.operator("melodiccap.analyze_rig", icon='VIEWZOOM')
        box.operator("melodiccap.clear_animation", icon='X')


# =============================================================================
# REGISTRATION
# =============================================================================

classes = [
    MELODICCAP_OT_import_take,
    MELODICCAP_OT_analyze_rig,
    MELODICCAP_OT_clear_animation,
    MELODICCAP_PT_main_panel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.melodiccap_use_ik = BoolProperty(
        name="Use IK Mode",
        description="Use IK targets instead of FK rotations",
        default=False
    )
    bpy.types.Scene.melodiccap_start_frame = IntProperty(
        name="Start Frame",
        description="Frame to start animation",
        default=1,
        min=1
    )
    bpy.types.Scene.melodiccap_auto_scale = BoolProperty(
        name="Auto Scale",
        description="Automatically scale to character height",
        default=True
    )
    bpy.types.Scene.melodiccap_scale_override = FloatProperty(
        name="Scale Override",
        description="Manual scale factor",
        default=1.0,
        min=0.01,
        max=10.0
    )
    
    debug_print("MelodicCap addon v2.0 registered")


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    del bpy.types.Scene.melodiccap_use_ik
    del bpy.types.Scene.melodiccap_start_frame
    del bpy.types.Scene.melodiccap_auto_scale
    del bpy.types.Scene.melodiccap_scale_override
    
    debug_print("MelodicCap addon unregistered")


if __name__ == "__main__":
    register()
