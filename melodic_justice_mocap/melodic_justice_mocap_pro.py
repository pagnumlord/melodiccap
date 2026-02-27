"""
Melodic Justice Motion Capture - Blender Addon (Professional Version)
======================================================================

A professional retargeting addon incorporating best practices from:
- FreeMoCap: Ground plane detection, T-pose/A-pose support
- Rokoko: Bone mapping workflows, IK control support  
- Rigging Dojo: Map ONLY control bones (NOT MCH/ORG/DEF)

Installation:
1. In Blender: Edit > Preferences > Add-ons
2. Click "Install..." and select this file
3. Enable "Animation: Melodic Justice MoCap Pro"

Usage:
1. Select your Rigify rig (JaxRigify)
2. Open Sidebar (N key) > MoCap tab
3. Load your mocap JSON file
4. Adjust settings (scale, IK/FK, smoothing)
5. Click "Apply to Rigify" or "Setup & Bake"

CRITICAL: This addon targets Rigify CONTROL bones only!
- ✓ torso, chest, hips
- ✓ hand_ik.L/R, foot_ik.L/R
- ✓ upper_arm_ik_target.L/R, thigh_ik_target.L/R  
- ✗ DEF-* (deformation bones - NEVER keyframe)
- ✗ MCH-* (mechanism bones - internal)
- ✗ ORG-* (original metarig - reference only)
"""

bl_info = {
    "name": "Melodic Justice MoCap Pro",
    "author": "Melodic Justice Production",
    "version": (2, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > MoCap",
    "description": "Professional motion capture retargeting for Rigify rigs",
    "category": "Animation",
    "doc_url": "https://github.com/melodicjustice/mocap",
}

import bpy
import json
import os
import math
from mathutils import Vector, Matrix, Euler, Quaternion
from bpy.props import (
    StringProperty,
    BoolProperty,
    FloatProperty,
    EnumProperty,
    IntProperty,
    PointerProperty
)
from bpy.types import Operator, Panel, PropertyGroup


# ============================================================================
# RIGIFY CONTROL BONE MAPPING
# ============================================================================
# 
# CRITICAL: Only map to CONTROL bones, never to:
#   - DEF-* (deformation bones driven by constraints)
#   - MCH-* (mechanism/helper bones)
#   - ORG-* (original metarig reference bones)
#
# Rigify control bones are what animators actually manipulate!

MOCAP_TO_RIGIFY_CONTROLS = {
    # === ROOT AND TORSO ===
    # These control overall body position and rotation
    "hip_center": {
        "bone": "torso",
        "type": "location_rotation",
        "description": "Main body control"
    },
    "spine_center": {
        "bone": "chest", 
        "type": "location",
        "description": "Chest/spine control"
    },
    
    # === HEAD AND NECK ===
    "nose": {
        "bone": "head",
        "type": "rotation",
        "description": "Head rotation from nose direction"
    },
    "neck_base": {
        "bone": "neck",
        "type": "rotation",
        "description": "Neck rotation"
    },
    
    # === LEFT ARM (IK MODE) ===
    "left_wrist": {
        "bone": "hand_ik.L",
        "type": "location",
        "description": "Left hand IK target"
    },
    "left_elbow": {
        "bone": "upper_arm_ik_target.L",
        "type": "location", 
        "description": "Left elbow IK pole target"
    },
    "left_shoulder": {
        "bone": "shoulder.L",
        "type": "rotation",
        "description": "Left shoulder rotation"
    },
    
    # === RIGHT ARM (IK MODE) ===
    "right_wrist": {
        "bone": "hand_ik.R",
        "type": "location",
        "description": "Right hand IK target"
    },
    "right_elbow": {
        "bone": "upper_arm_ik_target.R",
        "type": "location",
        "description": "Right elbow IK pole target"
    },
    "right_shoulder": {
        "bone": "shoulder.R",
        "type": "rotation",
        "description": "Right shoulder rotation"
    },
    
    # === LEFT LEG (IK MODE) ===
    "left_ankle": {
        "bone": "foot_ik.L",
        "type": "location",
        "description": "Left foot IK target"
    },
    "left_knee": {
        "bone": "thigh_ik_target.L",
        "type": "location",
        "description": "Left knee IK pole target"
    },
    "left_foot_index": {
        "bone": "toe_ik.L",
        "type": "rotation",
        "description": "Left toe control"
    },
    
    # === RIGHT LEG (IK MODE) ===
    "right_ankle": {
        "bone": "foot_ik.R",
        "type": "location",
        "description": "Right foot IK target"
    },
    "right_knee": {
        "bone": "thigh_ik_target.R",
        "type": "location",
        "description": "Right knee IK pole target"
    },
    "right_foot_index": {
        "bone": "toe_ik.R",
        "type": "rotation",
        "description": "Right toe control"
    },
}

# FK bone mapping (alternative to IK)
MOCAP_TO_RIGIFY_FK = {
    # Left arm FK
    "left_shoulder": {"bone": "upper_arm_fk.L", "type": "rotation"},
    "left_elbow": {"bone": "forearm_fk.L", "type": "rotation"},
    "left_wrist": {"bone": "hand_fk.L", "type": "rotation"},
    
    # Right arm FK
    "right_shoulder": {"bone": "upper_arm_fk.R", "type": "rotation"},
    "right_elbow": {"bone": "forearm_fk.R", "type": "rotation"},
    "right_wrist": {"bone": "hand_fk.R", "type": "rotation"},
    
    # Left leg FK
    "left_hip": {"bone": "thigh_fk.L", "type": "rotation"},
    "left_knee": {"bone": "shin_fk.L", "type": "rotation"},
    "left_ankle": {"bone": "foot_fk.L", "type": "rotation"},
    
    # Right leg FK
    "right_hip": {"bone": "thigh_fk.R", "type": "rotation"},
    "right_knee": {"bone": "shin_fk.R", "type": "rotation"},
    "right_ankle": {"bone": "foot_fk.R", "type": "rotation"},
}

# Bones that should never be keyframed (for validation)
FORBIDDEN_BONE_PREFIXES = ["DEF-", "MCH-", "ORG-"]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_valid_control_bone(bone_name: str) -> bool:
    """Check if a bone is a valid control bone (not DEF/MCH/ORG)."""
    for prefix in FORBIDDEN_BONE_PREFIXES:
        if bone_name.startswith(prefix):
            return False
    return True


def get_rigify_ik_fk_property(armature, limb: str, side: str) -> str:
    """
    Get the IK/FK switch property name for a Rigify limb.
    
    Args:
        armature: The armature object
        limb: "arm" or "leg"
        side: "L" or "R"
        
    Returns:
        Property name string, or None if not found
    """
    # Rigify uses these property patterns
    prop_patterns = [
        f"IK_FK_{limb}_{side}",
        f"IK_FK.{limb}.{side}",
        f"ik_fk_{limb.lower()}_{side.lower()}",
    ]
    
    if armature.pose:
        # Check on pose bones (usually on the IK control)
        ik_bone_name = f"hand_ik.{side}" if limb == "arm" else f"foot_ik.{side}"
        if ik_bone_name in armature.pose.bones:
            pose_bone = armature.pose.bones[ik_bone_name]
            for prop in prop_patterns:
                if prop in pose_bone:
                    return prop
    
    return None


def set_rigify_ik_mode(armature, use_ik: bool = True):
    """
    Set all Rigify limbs to IK or FK mode.
    
    Args:
        armature: The armature object
        use_ik: True for IK mode (0.0), False for FK mode (1.0)
    """
    ik_value = 0.0 if use_ik else 1.0
    
    for limb in ["arm", "leg"]:
        for side in ["L", "R"]:
            prop_name = get_rigify_ik_fk_property(armature, limb, side)
            if prop_name:
                ik_bone_name = f"hand_ik.{side}" if limb == "arm" else f"foot_ik.{side}"
                if ik_bone_name in armature.pose.bones:
                    armature.pose.bones[ik_bone_name][prop_name] = ik_value


def calculate_rotation_from_points(
    point1: Vector,
    point2: Vector,
    reference_up: Vector = Vector((0, 0, 1))
) -> Quaternion:
    """
    Calculate rotation quaternion from two points.
    
    Args:
        point1: Starting point (e.g., shoulder)
        point2: Ending point (e.g., ear for head rotation)
        reference_up: Up vector for orientation
        
    Returns:
        Quaternion representing the rotation
    """
    direction = (point2 - point1).normalized()
    
    # Create rotation matrix
    forward = direction
    right = forward.cross(reference_up).normalized()
    up = right.cross(forward).normalized()
    
    rot_matrix = Matrix((
        (right.x, up.x, forward.x),
        (right.y, up.y, forward.y),
        (right.z, up.z, forward.z)
    ))
    
    return rot_matrix.to_quaternion()


def smooth_keyframes(fcurve, window_size: int = 5):
    """
    Apply simple moving average smoothing to an fcurve.
    
    Args:
        fcurve: The F-Curve to smooth
        window_size: Number of frames for averaging
    """
    if not fcurve or len(fcurve.keyframe_points) < window_size:
        return
    
    # Extract values
    values = [kp.co[1] for kp in fcurve.keyframe_points]
    
    # Apply moving average
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(values)):
        start = max(0, i - half_window)
        end = min(len(values), i + half_window + 1)
        avg = sum(values[start:end]) / (end - start)
        smoothed.append(avg)
    
    # Apply smoothed values
    for i, kp in enumerate(fcurve.keyframe_points):
        kp.co[1] = smoothed[i]
    
    fcurve.update()


# ============================================================================
# MOCAP DATA LOADER
# ============================================================================

class MocapDataLoader:
    """Loads and processes motion capture JSON data."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None
        self.frames = []
        self.metadata = {}
        self.load()
    
    def load(self):
        """Load JSON data from file."""
        with open(self.filepath, 'r') as f:
            self.data = json.load(f)
        
        self.metadata = self.data.get('metadata', {})
        self.frames = self.data.get('frames', [])
        
        # Handle old format
        if not self.frames and 'frames' not in self.data:
            # Try legacy format
            if isinstance(self.data, list):
                self.frames = self.data
            elif 'frame_data' in self.data:
                self.frames = self.data['frame_data']
    
    @property
    def frame_count(self) -> int:
        return len(self.frames)
    
    @property
    def fps(self) -> float:
        return self.metadata.get('fps', self.metadata.get('target_fps', 30))
    
    @property
    def duration(self) -> float:
        return self.metadata.get('duration_seconds', self.frame_count / self.fps)
    
    def get_frame(self, index: int) -> dict:
        """Get frame data by index."""
        if 0 <= index < len(self.frames):
            return self.frames[index]
        return {}
    
    def get_landmark_3d(self, frame_index: int, landmark_name: str) -> Vector:
        """
        Get 3D landmark position for a frame.
        
        Args:
            frame_index: Frame number
            landmark_name: Name of the landmark (e.g., "left_wrist")
            
        Returns:
            Vector with (x, y, z) in Blender coordinates, or None
        """
        frame = self.get_frame(frame_index)
        
        # Try different data sources in order of preference
        sources = ['landmarks_3d', 'world_landmarks', 'landmarks_2d_left']
        
        for source in sources:
            if source in frame and landmark_name in frame[source]:
                lm = frame[source][landmark_name]
                return Vector((lm['x'], lm['y'], lm['z']))
        
        return None
    
    def get_landmark_visibility(self, frame_index: int, landmark_name: str) -> float:
        """Get visibility/confidence for a landmark."""
        frame = self.get_frame(frame_index)
        
        for source in ['landmarks_3d', 'world_landmarks']:
            if source in frame and landmark_name in frame[source]:
                return frame[source][landmark_name].get('visibility', 1.0)
        
        return 0.0


# ============================================================================
# RIGIFY RETARGETER
# ============================================================================

class RigifyRetargeter:
    """
    Retargets motion capture data to Rigify control bones.
    
    IMPORTANT: This class only targets control bones, never DEF/MCH/ORG bones!
    """
    
    def __init__(
        self,
        armature,
        mocap_data: MocapDataLoader,
        use_ik: bool = True,
        scale_factor: float = 1.0,
        apply_smoothing: bool = True,
        smoothing_window: int = 5
    ):
        self.armature = armature
        self.mocap_data = mocap_data
        self.use_ik = use_ik
        self.scale_factor = scale_factor
        self.apply_smoothing = apply_smoothing
        self.smoothing_window = smoothing_window
        
        # Select bone mapping based on IK/FK mode
        self.bone_mapping = MOCAP_TO_RIGIFY_CONTROLS if use_ik else MOCAP_TO_RIGIFY_FK
        
        # Track which bones we successfully mapped
        self.mapped_bones = set()
        self.failed_bones = set()
        
        # Character reference point (hip position in rest pose)
        self.reference_hip_pos = None
    
    def validate_armature(self) -> bool:
        """
        Validate that the armature is a proper Rigify rig.
        
        Returns:
            True if valid, False otherwise
        """
        if not self.armature or self.armature.type != 'ARMATURE':
            return False
        
        # Check for key Rigify bones
        required_bones = ["torso", "head", "hand_ik.L", "hand_ik.R"]
        
        for bone_name in required_bones:
            if bone_name not in self.armature.pose.bones:
                print(f"Warning: Required bone '{bone_name}' not found")
                return False
        
        return True
    
    def get_rest_hip_position(self) -> Vector:
        """Get the hip position in rest pose for reference."""
        if "torso" in self.armature.pose.bones:
            bone = self.armature.pose.bones["torso"]
            return self.armature.matrix_world @ bone.bone.head_local
        return Vector((0, 0, 0))
    
    def apply_to_bone(
        self,
        bone_name: str,
        position: Vector,
        keyframe_type: str,
        frame: int
    ):
        """
        Apply position/rotation to a control bone and insert keyframe.
        
        Args:
            bone_name: Name of the Rigify control bone
            position: World position from mocap
            keyframe_type: "location", "rotation", or "location_rotation"
            frame: Frame number to keyframe
        """
        # Validate bone name
        if not is_valid_control_bone(bone_name):
            if bone_name not in self.failed_bones:
                print(f"ERROR: Attempted to keyframe forbidden bone: {bone_name}")
                self.failed_bones.add(bone_name)
            return
        
        if bone_name not in self.armature.pose.bones:
            if bone_name not in self.failed_bones:
                print(f"Warning: Bone '{bone_name}' not found in rig")
                self.failed_bones.add(bone_name)
            return
        
        pose_bone = self.armature.pose.bones[bone_name]
        
        # Apply scale factor
        scaled_pos = position * self.scale_factor
        
        # Convert to bone local space
        if pose_bone.parent:
            parent_matrix = pose_bone.parent.matrix.inverted()
            local_pos = parent_matrix @ scaled_pos
        else:
            armature_matrix = self.armature.matrix_world.inverted()
            local_pos = armature_matrix @ scaled_pos
        
        # Apply and keyframe
        if keyframe_type in ["location", "location_rotation"]:
            pose_bone.location = local_pos
            pose_bone.keyframe_insert(data_path="location", frame=frame)
        
        self.mapped_bones.add(bone_name)
    
    def apply_rotation_from_landmarks(
        self,
        bone_name: str,
        landmark_from: str,
        landmark_to: str,
        frame_index: int,
        frame: int
    ):
        """
        Apply rotation to a bone calculated from two landmarks.
        
        Args:
            bone_name: Control bone to rotate
            landmark_from: Starting landmark name
            landmark_to: Ending landmark name
            frame_index: Index in mocap data
            frame: Blender frame number
        """
        if not is_valid_control_bone(bone_name):
            return
        
        if bone_name not in self.armature.pose.bones:
            return
        
        pos_from = self.mocap_data.get_landmark_3d(frame_index, landmark_from)
        pos_to = self.mocap_data.get_landmark_3d(frame_index, landmark_to)
        
        if pos_from is None or pos_to is None:
            return
        
        pose_bone = self.armature.pose.bones[bone_name]
        
        # Calculate rotation
        rotation = calculate_rotation_from_points(pos_from, pos_to)
        
        # Apply to bone
        pose_bone.rotation_quaternion = rotation
        pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        
        self.mapped_bones.add(bone_name)
    
    def retarget_frame(self, frame_index: int, blender_frame: int):
        """
        Retarget a single frame of mocap data.
        
        Args:
            frame_index: Index in mocap data
            blender_frame: Target Blender frame number
        """
        # Process each mapped landmark
        for landmark_name, mapping in self.bone_mapping.items():
            bone_name = mapping["bone"]
            keyframe_type = mapping["type"]
            
            # Get 3D position
            position = self.mocap_data.get_landmark_3d(frame_index, landmark_name)
            
            if position is None:
                continue
            
            # Check visibility
            visibility = self.mocap_data.get_landmark_visibility(frame_index, landmark_name)
            if visibility < 0.5:
                continue
            
            # Apply to bone
            if keyframe_type in ["location", "location_rotation"]:
                self.apply_to_bone(bone_name, position, keyframe_type, blender_frame)
        
        # Calculate head rotation from landmarks
        self.apply_rotation_from_landmarks(
            "head", "nose", "neck_base", frame_index, blender_frame
        )
    
    def retarget_all(self, start_frame: int = 1, report=None):
        """
        Retarget all frames of mocap data.
        
        Args:
            start_frame: Starting Blender frame
            report: Optional reporting function for UI feedback
        """
        # Setup
        if self.use_ik:
            set_rigify_ik_mode(self.armature, use_ik=True)
        
        self.reference_hip_pos = self.get_rest_hip_position()
        
        # Set frame range
        total_frames = self.mocap_data.frame_count
        bpy.context.scene.frame_start = start_frame
        bpy.context.scene.frame_end = start_frame + total_frames - 1
        
        # Process each frame
        for i in range(total_frames):
            blender_frame = start_frame + i
            bpy.context.scene.frame_set(blender_frame)
            
            self.retarget_frame(i, blender_frame)
            
            # Progress report
            if report and i % 30 == 0:
                progress = (i / total_frames) * 100
                report({'INFO'}, f"Retargeting: {progress:.0f}%")
        
        # Apply smoothing if enabled
        if self.apply_smoothing:
            self._apply_smoothing()
        
        # Report results
        print(f"\nRetargeting complete!")
        print(f"  Frames: {total_frames}")
        print(f"  Mapped bones: {len(self.mapped_bones)}")
        if self.failed_bones:
            print(f"  Failed bones: {self.failed_bones}")
    
    def _apply_smoothing(self):
        """Apply smoothing to all keyframed curves."""
        if not self.armature.animation_data or not self.armature.animation_data.action:
            return
        
        action = self.armature.animation_data.action
        
        for fcurve in action.fcurves:
            # Only smooth location curves (rotation often needs sharp changes)
            if "location" in fcurve.data_path:
                smooth_keyframes(fcurve, self.smoothing_window)


# ============================================================================
# BLENDER OPERATORS
# ============================================================================

class MOCAP_OT_LoadFile(Operator):
    """Load a motion capture JSON file"""
    bl_idname = "mocap.load_file"
    bl_label = "Load MoCap File"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: StringProperty(
        subtype='FILE_PATH',
        default=""
    )
    
    filter_glob: StringProperty(
        default="*.json",
        options={'HIDDEN'}
    )
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        props = context.scene.mocap_props
        props.filepath = self.filepath
        
        # Validate file
        try:
            loader = MocapDataLoader(self.filepath)
            props.frame_count = loader.frame_count
            props.detected_fps = loader.fps
            self.report({'INFO'}, f"Loaded {loader.frame_count} frames @ {loader.fps:.1f} fps")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load: {e}")
            return {'CANCELLED'}
        
        return {'FINISHED'}


class MOCAP_OT_ApplyToRigify(Operator):
    """Apply motion capture data to selected Rigify rig"""
    bl_idname = "mocap.apply_to_rigify"
    bl_label = "Apply to Rigify"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        props = context.scene.mocap_props
        obj = context.active_object
        return (
            props.filepath and
            os.path.exists(props.filepath) and
            obj and obj.type == 'ARMATURE'
        )
    
    def execute(self, context):
        props = context.scene.mocap_props
        armature = context.active_object
        
        # Load mocap data
        try:
            loader = MocapDataLoader(props.filepath)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load mocap data: {e}")
            return {'CANCELLED'}
        
        # Create retargeter
        retargeter = RigifyRetargeter(
            armature=armature,
            mocap_data=loader,
            use_ik=props.use_ik,
            scale_factor=props.scale_factor,
            apply_smoothing=props.apply_smoothing,
            smoothing_window=props.smoothing_window
        )
        
        # Validate
        if not retargeter.validate_armature():
            self.report({'ERROR'}, "Selected object is not a valid Rigify rig")
            return {'CANCELLED'}
        
        # Apply retargeting
        retargeter.retarget_all(
            start_frame=props.start_frame,
            report=self.report
        )
        
        self.report({'INFO'}, 
            f"Retargeted {loader.frame_count} frames to {len(retargeter.mapped_bones)} bones")
        
        return {'FINISHED'}


class MOCAP_OT_SetIKMode(Operator):
    """Set Rigify rig to IK mode"""
    bl_idname = "mocap.set_ik_mode"
    bl_label = "Set IK Mode"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'ARMATURE'
    
    def execute(self, context):
        set_rigify_ik_mode(context.active_object, use_ik=True)
        self.report({'INFO'}, "Set all limbs to IK mode")
        return {'FINISHED'}


class MOCAP_OT_SetFKMode(Operator):
    """Set Rigify rig to FK mode"""
    bl_idname = "mocap.set_fk_mode"
    bl_label = "Set FK Mode"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'ARMATURE'
    
    def execute(self, context):
        set_rigify_ik_mode(context.active_object, use_ik=False)
        self.report({'INFO'}, "Set all limbs to FK mode")
        return {'FINISHED'}


class MOCAP_OT_ClearAnimation(Operator):
    """Clear all animation from selected rig"""
    bl_idname = "mocap.clear_animation"
    bl_label = "Clear Animation"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'ARMATURE'
    
    def execute(self, context):
        armature = context.active_object
        
        if armature.animation_data and armature.animation_data.action:
            bpy.data.actions.remove(armature.animation_data.action)
            self.report({'INFO'}, "Cleared animation data")
        else:
            self.report({'WARNING'}, "No animation data to clear")
        
        return {'FINISHED'}


class MOCAP_OT_BakeToKeyframes(Operator):
    """Bake constraints to keyframes"""
    bl_idname = "mocap.bake_to_keyframes"
    bl_label = "Bake to Keyframes"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'ARMATURE'
    
    def execute(self, context):
        # Use Blender's built-in bake
        bpy.ops.nla.bake(
            frame_start=context.scene.frame_start,
            frame_end=context.scene.frame_end,
            only_selected=False,
            visual_keying=True,
            clear_constraints=False,
            bake_types={'POSE'}
        )
        
        self.report({'INFO'}, "Baked animation to keyframes")
        return {'FINISHED'}


# ============================================================================
# PROPERTIES
# ============================================================================

class MocapProperties(PropertyGroup):
    """Properties for the MoCap addon."""
    
    filepath: StringProperty(
        name="MoCap File",
        description="Path to motion capture JSON file",
        subtype='FILE_PATH',
        default=""
    )
    
    frame_count: IntProperty(
        name="Frame Count",
        description="Number of frames in loaded file",
        default=0
    )
    
    detected_fps: FloatProperty(
        name="Detected FPS",
        description="FPS detected in mocap file",
        default=30.0
    )
    
    use_ik: BoolProperty(
        name="Use IK Controls",
        description="Use IK mode for retargeting (recommended)",
        default=True
    )
    
    scale_factor: FloatProperty(
        name="Scale",
        description="Scale factor for mocap data",
        default=1.0,
        min=0.01,
        max=100.0,
        soft_min=0.1,
        soft_max=10.0
    )
    
    start_frame: IntProperty(
        name="Start Frame",
        description="Starting frame for animation",
        default=1,
        min=1
    )
    
    apply_smoothing: BoolProperty(
        name="Apply Smoothing",
        description="Smooth keyframes after retargeting",
        default=True
    )
    
    smoothing_window: IntProperty(
        name="Smoothing Window",
        description="Number of frames for smoothing",
        default=5,
        min=3,
        max=15
    )


# ============================================================================
# PANELS
# ============================================================================

class MOCAP_PT_MainPanel(Panel):
    """Main panel for motion capture tools"""
    bl_label = "Melodic Justice MoCap"
    bl_idname = "MOCAP_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "MoCap"
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.mocap_props
        obj = context.active_object
        
        # File loading
        box = layout.box()
        box.label(text="1. Load Data", icon='FILE')
        
        row = box.row()
        row.prop(props, "filepath", text="")
        row.operator("mocap.load_file", text="", icon='FILEBROWSER')
        
        if props.frame_count > 0:
            box.label(text=f"Frames: {props.frame_count} @ {props.detected_fps:.1f} fps")
        
        # Target selection
        box = layout.box()
        box.label(text="2. Select Target", icon='ARMATURE_DATA')
        
        if obj and obj.type == 'ARMATURE':
            box.label(text=f"✓ {obj.name}", icon='CHECKMARK')
        else:
            box.label(text="Select a Rigify armature", icon='ERROR')
        
        # Settings
        box = layout.box()
        box.label(text="3. Settings", icon='SETTINGS')
        
        row = box.row()
        row.prop(props, "use_ik")
        
        box.prop(props, "scale_factor")
        box.prop(props, "start_frame")
        
        row = box.row()
        row.prop(props, "apply_smoothing")
        if props.apply_smoothing:
            row.prop(props, "smoothing_window", text="Window")
        
        # IK/FK mode buttons
        row = box.row(align=True)
        row.operator("mocap.set_ik_mode", text="Set IK")
        row.operator("mocap.set_fk_mode", text="Set FK")
        
        # Apply button
        layout.separator()
        box = layout.box()
        box.label(text="4. Apply", icon='PLAY')
        
        row = box.row(align=True)
        row.scale_y = 1.5
        row.operator("mocap.apply_to_rigify", text="Apply to Rigify", icon='ARMATURE_DATA')
        
        # Utilities
        box = layout.box()
        box.label(text="Utilities", icon='TOOL_SETTINGS')
        
        row = box.row(align=True)
        row.operator("mocap.bake_to_keyframes", text="Bake")
        row.operator("mocap.clear_animation", text="Clear", icon='X')
        
        # Tips
        layout.separator()
        box = layout.box()
        box.label(text="Tips:", icon='INFO')
        col = box.column(align=True)
        col.scale_y = 0.8
        col.label(text="• Set IK/FK to 1.0 first")
        col.label(text="• Preview before baking")
        col.label(text="• Jax is 1.87m tall")


# ============================================================================
# REGISTRATION
# ============================================================================

classes = [
    MocapProperties,
    MOCAP_OT_LoadFile,
    MOCAP_OT_ApplyToRigify,
    MOCAP_OT_SetIKMode,
    MOCAP_OT_SetFKMode,
    MOCAP_OT_ClearAnimation,
    MOCAP_OT_BakeToKeyframes,
    MOCAP_PT_MainPanel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.mocap_props = PointerProperty(type=MocapProperties)
    
    print("Melodic Justice MoCap Pro addon registered")


def unregister():
    del bpy.types.Scene.mocap_props
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    print("Melodic Justice MoCap Pro addon unregistered")


if __name__ == "__main__":
    register()
