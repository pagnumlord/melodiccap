"""
Melodic Justice Motion Capture - Blender Addon
==============================================
Import motion capture data and retarget to Rigify rigs.

Installation:
1. In Blender, go to Edit > Preferences > Add-ons
2. Click "Install..." and select this file
3. Enable "Animation: Melodic Justice MoCap"

Usage:
1. Select your Rigify rig
2. Go to the sidebar (N key) > MoCap tab
3. Load your mocap JSON file
4. Click "Retarget Animation"
"""

bl_info = {
    "name": "Melodic Justice MoCap",
    "author": "Melodic Justice Production",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > MoCap",
    "description": "Import and retarget motion capture data to Rigify rigs",
    "category": "Animation",
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
    IntProperty
)
from bpy.types import Operator, Panel, PropertyGroup


# ============================================================================
# RIGIFY BONE MAPPING
# ============================================================================

# MediaPipe landmark to Rigify CONTROL bone mapping
# These are the actual control bones that should be animated!
MEDIAPIPE_TO_RIGIFY = {
    # === CORE BODY (Location) ===
    "hip_center": "torso",
    "spine_center": "chest",
    "left_hip": None,  # Part of torso
    "right_hip": None,  # Part of torso
    
    # === HEAD (Rotation) ===
    "nose": "head",
    "left_ear": "head",
    "right_ear": "head",
    
    # === LEFT ARM IK (Location) ===
    "left_shoulder": "shoulder.L",
    "left_wrist": "hand_ik.L",
    "left_elbow": "upper_arm_ik_target.L",
    
    # === RIGHT ARM IK (Location) ===
    "right_shoulder": "shoulder.R",
    "right_wrist": "hand_ik.R",
    "right_elbow": "upper_arm_ik_target.R",
    
    # === LEFT LEG IK (Location) ===
    "left_ankle": "foot_ik.L",
    "left_knee": "thigh_ik_target.L",
    "left_foot_index": "toe_ik.L",
    
    # === RIGHT LEG IK (Location) ===
    "right_ankle": "foot_ik.R",
    "right_knee": "thigh_ik_target.R",
    "right_foot_index": "toe_ik.R",
}

# Which bones need location vs rotation keyframes
LOCATION_BONES = {
    "torso", "chest", "hips",
    "hand_ik.L", "hand_ik.R",
    "foot_ik.L", "foot_ik.R",
    "upper_arm_ik_target.L", "upper_arm_ik_target.R",
    "thigh_ik_target.L", "thigh_ik_target.R",
}

ROTATION_BONES = {
    "head", "neck",
    "shoulder.L", "shoulder.R",
}


# ============================================================================
# COMPUTED LANDMARKS
# ============================================================================

def compute_derived_landmarks(landmarks):
    """
    Compute derived landmarks like hip_center, spine_center, etc.
    from raw MediaPipe landmarks.
    """
    derived = dict(landmarks)
    
    # Hip center (midpoint of hips)
    if "left_hip" in landmarks and "right_hip" in landmarks:
        lh = landmarks["left_hip"]
        rh = landmarks["right_hip"]
        derived["hip_center"] = {
            'x': (lh['x'] + rh['x']) / 2,
            'y': (lh['y'] + rh['y']) / 2,
            'z': (lh['z'] + rh['z']) / 2,
            'visibility': min(lh['visibility'], rh['visibility'])
        }
    
    # Shoulder center
    if "left_shoulder" in landmarks and "right_shoulder" in landmarks:
        ls = landmarks["left_shoulder"]
        rs = landmarks["right_shoulder"]
        derived["shoulder_center"] = {
            'x': (ls['x'] + rs['x']) / 2,
            'y': (ls['y'] + rs['y']) / 2,
            'z': (ls['z'] + rs['z']) / 2,
            'visibility': min(ls['visibility'], rs['visibility'])
        }
    
    # Spine center (between hip center and shoulder center)
    if "hip_center" in derived and "shoulder_center" in derived:
        hc = derived["hip_center"]
        sc = derived["shoulder_center"]
        derived["spine_center"] = {
            'x': (hc['x'] + sc['x']) / 2,
            'y': (hc['y'] + sc['y']) / 2,
            'z': (hc['z'] + sc['z']) / 2,
            'visibility': min(hc['visibility'], sc['visibility'])
        }
    
    return derived


# ============================================================================
# MOTION DATA LOADER
# ============================================================================

class MotionDataLoader:
    """Loads and processes motion capture data."""
    
    def __init__(self):
        self.data = None
        self.filepath = ""
        self.frame_count = 0
        self.fps = 30
    
    def load(self, filepath):
        """Load motion data from JSON file."""
        with open(filepath, 'r') as f:
            self.data = json.load(f)
        
        self.filepath = filepath
        self.frame_count = self.data.get('frame_count', len(self.data.get('frames', [])))
        self.fps = self.data.get('fps', 30)
        
        return True
    
    def get_frame(self, frame_idx):
        """Get landmarks for a specific frame."""
        if not self.data or 'frames' not in self.data:
            return None
        
        if frame_idx >= len(self.data['frames']):
            return None
        
        frame = self.data['frames'][frame_idx]
        landmarks = frame.get('landmarks', frame.get('landmarks_3d', {}))
        
        # Compute derived landmarks
        return compute_derived_landmarks(landmarks)
    
    def get_all_frames(self):
        """Generator yielding all frames."""
        for i in range(self.frame_count):
            yield i, self.get_frame(i)


# ============================================================================
# RIGIFY RETARGETER
# ============================================================================

class RigifyRetargeter:
    """Retargets motion data to a Rigify rig."""
    
    def __init__(self, armature_obj, scale=1.0):
        self.armature = armature_obj
        self.scale = scale
        
        # Store rest pose bone positions
        self.rest_positions = {}
        self._cache_rest_pose()
    
    def _cache_rest_pose(self):
        """Cache the rest pose positions of control bones."""
        if not self.armature or self.armature.type != 'ARMATURE':
            return
        
        # Store current mode
        mode = self.armature.mode if bpy.context.active_object == self.armature else 'OBJECT'
        
        # Switch to pose mode to read pose bones
        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.mode_set(mode='POSE')
        
        for pbone in self.armature.pose.bones:
            # Get bone head in armature space
            self.rest_positions[pbone.name] = pbone.bone.head_local.copy()
        
        # Restore mode
        bpy.ops.object.mode_set(mode=mode)
    
    def get_bone(self, bone_name):
        """Get a pose bone by name."""
        if not self.armature:
            return None
        return self.armature.pose.bones.get(bone_name)
    
    def apply_frame(self, frame_idx, landmarks, use_ik=True):
        """
        Apply motion data for a single frame to the rig.
        
        Args:
            frame_idx: Frame number (1-indexed for Blender)
            landmarks: Dictionary of landmark positions
            use_ik: Use IK controls (recommended) vs FK
        """
        if not landmarks:
            return
        
        # Get hip center as root reference
        root_offset = Vector((0, 0, 0))
        if "hip_center" in landmarks:
            hc = landmarks["hip_center"]
            root_offset = Vector((hc['x'], hc['y'], hc['z'])) * self.scale
        
        # Apply torso/root
        self._apply_torso(frame_idx, landmarks, root_offset)
        
        # Apply limbs
        if use_ik:
            self._apply_limbs_ik(frame_idx, landmarks, root_offset)
        else:
            self._apply_limbs_fk(frame_idx, landmarks, root_offset)
        
        # Apply head
        self._apply_head(frame_idx, landmarks)
    
    def _apply_torso(self, frame_idx, landmarks, root_offset):
        """Apply torso movement."""
        torso = self.get_bone("torso")
        if not torso:
            return
        
        if "hip_center" in landmarks:
            hc = landmarks["hip_center"]
            pos = Vector((hc['x'], hc['y'], hc['z'])) * self.scale
            
            # Apply relative to rest position
            rest = self.rest_positions.get("torso", Vector())
            torso.location = pos - rest
            torso.keyframe_insert(data_path="location", frame=frame_idx)
        
        # Calculate torso rotation from hip/shoulder orientation
        if all(k in landmarks for k in ["left_hip", "right_hip", "left_shoulder", "right_shoulder"]):
            lh = Vector((landmarks["left_hip"]['x'], landmarks["left_hip"]['y'], landmarks["left_hip"]['z']))
            rh = Vector((landmarks["right_hip"]['x'], landmarks["right_hip"]['y'], landmarks["right_hip"]['z']))
            
            # Hip direction (left to right)
            hip_dir = (rh - lh).normalized()
            
            # Forward direction (perpendicular to hip direction, in XY plane)
            forward = Vector((-hip_dir.y, hip_dir.x, 0)).normalized()
            
            # Create rotation matrix
            up = Vector((0, 0, 1))
            right = forward.cross(up).normalized()
            
            # Calculate rotation
            rotation = Matrix((right, forward, up)).transposed().to_euler()
            
            torso.rotation_euler = rotation
            torso.keyframe_insert(data_path="rotation_euler", frame=frame_idx)
    
    def _apply_limbs_ik(self, frame_idx, landmarks, root_offset):
        """Apply limb positions using IK controls."""
        
        # === ARMS ===
        arm_mappings = [
            ("left_wrist", "hand_ik.L"),
            ("right_wrist", "hand_ik.R"),
            ("left_elbow", "upper_arm_ik_target.L"),
            ("right_elbow", "upper_arm_ik_target.R"),
        ]
        
        for landmark_name, bone_name in arm_mappings:
            if landmark_name in landmarks:
                bone = self.get_bone(bone_name)
                if bone:
                    lm = landmarks[landmark_name]
                    pos = Vector((lm['x'], lm['y'], lm['z'])) * self.scale
                    
                    # Position relative to torso rest
                    rest = self.rest_positions.get(bone_name, Vector())
                    bone.location = pos - root_offset
                    bone.keyframe_insert(data_path="location", frame=frame_idx)
        
        # === LEGS ===
        leg_mappings = [
            ("left_ankle", "foot_ik.L"),
            ("right_ankle", "foot_ik.R"),
            ("left_knee", "thigh_ik_target.L"),
            ("right_knee", "thigh_ik_target.R"),
        ]
        
        for landmark_name, bone_name in leg_mappings:
            if landmark_name in landmarks:
                bone = self.get_bone(bone_name)
                if bone:
                    lm = landmarks[landmark_name]
                    pos = Vector((lm['x'], lm['y'], lm['z'])) * self.scale
                    
                    bone.location = pos - root_offset
                    bone.keyframe_insert(data_path="location", frame=frame_idx)
        
        # === SHOULDERS ===
        shoulder_mappings = [
            ("left_shoulder", "shoulder.L"),
            ("right_shoulder", "shoulder.R"),
        ]
        
        for landmark_name, bone_name in shoulder_mappings:
            if landmark_name in landmarks:
                bone = self.get_bone(bone_name)
                if bone:
                    # Shoulders get rotation based on elevation
                    lm = landmarks[landmark_name]
                    
                    # Calculate shoulder rotation from position relative to spine
                    if "shoulder_center" in landmarks:
                        sc = landmarks["shoulder_center"]
                        offset = Vector((
                            lm['x'] - sc['x'],
                            lm['y'] - sc['y'],
                            lm['z'] - sc['z']
                        ))
                        
                        # Simple shoulder rotation approximation
                        angle = math.atan2(offset.z, abs(offset.x)) * 0.5
                        
                        bone.rotation_euler.x = angle
                        bone.keyframe_insert(data_path="rotation_euler", frame=frame_idx)
    
    def _apply_limbs_fk(self, frame_idx, landmarks, root_offset):
        """Apply limb rotations using FK controls."""
        # FK implementation would calculate rotations between connected joints
        # This is more complex and less reliable than IK for mocap
        pass
    
    def _apply_head(self, frame_idx, landmarks):
        """Apply head rotation."""
        head_bone = self.get_bone("head")
        if not head_bone:
            return
        
        # Calculate head direction from facial landmarks
        if all(k in landmarks for k in ["nose", "left_ear", "right_ear"]):
            nose = Vector((landmarks["nose"]['x'], landmarks["nose"]['y'], landmarks["nose"]['z']))
            left_ear = Vector((landmarks["left_ear"]['x'], landmarks["left_ear"]['y'], landmarks["left_ear"]['z']))
            right_ear = Vector((landmarks["right_ear"]['x'], landmarks["right_ear"]['y'], landmarks["right_ear"]['z']))
            
            # Head center
            head_center = (left_ear + right_ear) / 2
            
            # Forward direction (toward nose)
            forward = (nose - head_center).normalized()
            
            # Right direction
            right = (right_ear - left_ear).normalized()
            
            # Up direction
            up = right.cross(forward).normalized()
            
            # Recalculate forward for orthogonality
            forward = up.cross(right).normalized()
            
            # Create rotation
            mat = Matrix((right, forward, up)).transposed()
            rotation = mat.to_euler()
            
            head_bone.rotation_euler = rotation
            head_bone.keyframe_insert(data_path="rotation_euler", frame=frame_idx)


# ============================================================================
# BLENDER OPERATORS
# ============================================================================

class MOCAP_OT_load_file(Operator):
    """Load motion capture data file"""
    bl_idname = "mocap.load_file"
    bl_label = "Load MoCap File"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: StringProperty(
        name="File Path",
        subtype='FILE_PATH'
    )
    
    filter_glob: StringProperty(
        default="*.json",
        options={'HIDDEN'}
    )
    
    def execute(self, context):
        props = context.scene.mocap_props
        
        if not os.path.exists(self.filepath):
            self.report({'ERROR'}, f"File not found: {self.filepath}")
            return {'CANCELLED'}
        
        props.mocap_filepath = self.filepath
        
        # Load and validate
        loader = MotionDataLoader()
        if loader.load(self.filepath):
            props.frame_count = loader.frame_count
            props.fps = loader.fps
            self.report({'INFO'}, f"Loaded {loader.frame_count} frames at {loader.fps} FPS")
        else:
            self.report({'ERROR'}, "Failed to load mocap file")
            return {'CANCELLED'}
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class MOCAP_OT_retarget(Operator):
    """Retarget motion capture to selected Rigify rig"""
    bl_idname = "mocap.retarget"
    bl_label = "Retarget Animation"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        props = context.scene.mocap_props
        obj = context.active_object
        return (
            props.mocap_filepath and 
            os.path.exists(props.mocap_filepath) and
            obj and 
            obj.type == 'ARMATURE'
        )
    
    def execute(self, context):
        props = context.scene.mocap_props
        armature = context.active_object
        
        # Load motion data
        loader = MotionDataLoader()
        if not loader.load(props.mocap_filepath):
            self.report({'ERROR'}, "Failed to load mocap file")
            return {'CANCELLED'}
        
        # Create retargeter
        retargeter = RigifyRetargeter(armature, scale=props.scale)
        
        # Set scene FPS
        context.scene.render.fps = loader.fps
        
        # Set frame range
        start_frame = props.start_frame
        context.scene.frame_start = start_frame
        context.scene.frame_end = start_frame + loader.frame_count - 1
        
        # Ensure we're in pose mode
        bpy.ops.object.mode_set(mode='POSE')
        
        # Apply each frame
        self.report({'INFO'}, f"Retargeting {loader.frame_count} frames...")
        
        for frame_idx, landmarks in loader.get_all_frames():
            if landmarks:
                blender_frame = start_frame + frame_idx
                retargeter.apply_frame(
                    blender_frame, 
                    landmarks,
                    use_ik=props.use_ik
                )
        
        self.report({'INFO'}, f"Retargeted {loader.frame_count} frames to {armature.name}")
        
        # Go to first frame
        context.scene.frame_set(start_frame)
        
        return {'FINISHED'}


class MOCAP_OT_clear_animation(Operator):
    """Clear all keyframes from the selected rig"""
    bl_idname = "mocap.clear_animation"
    bl_label = "Clear Animation"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'ARMATURE'
    
    def execute(self, context):
        armature = context.active_object
        
        # Clear animation data
        if armature.animation_data:
            armature.animation_data_clear()
        
        # Reset pose
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.pose.transforms_clear()
        
        self.report({'INFO'}, f"Cleared animation from {armature.name}")
        return {'FINISHED'}


class MOCAP_OT_detect_cameras(Operator):
    """List available cameras for reference"""
    bl_idname = "mocap.detect_cameras"
    bl_label = "Detect Cameras"
    
    def execute(self, context):
        self.report({'INFO'}, "Camera detection requires the capture app (not Blender)")
        return {'FINISHED'}


# ============================================================================
# PROPERTIES
# ============================================================================

class MocapProperties(PropertyGroup):
    """Properties for the mocap addon."""
    
    mocap_filepath: StringProperty(
        name="MoCap File",
        description="Path to motion capture JSON file",
        subtype='FILE_PATH'
    )
    
    frame_count: IntProperty(
        name="Frame Count",
        default=0,
        min=0
    )
    
    fps: IntProperty(
        name="FPS",
        default=30,
        min=1,
        max=120
    )
    
    start_frame: IntProperty(
        name="Start Frame",
        description="Frame to start the animation",
        default=1,
        min=1
    )
    
    scale: FloatProperty(
        name="Scale",
        description="Scale factor for motion data",
        default=1.0,
        min=0.01,
        max=100.0
    )
    
    use_ik: BoolProperty(
        name="Use IK",
        description="Use IK controls (recommended for Rigify)",
        default=True
    )
    
    smooth_motion: BoolProperty(
        name="Smooth Motion",
        description="Apply smoothing to the animation",
        default=True
    )
    
    smooth_factor: FloatProperty(
        name="Smooth Factor",
        description="Amount of smoothing to apply",
        default=0.5,
        min=0.0,
        max=1.0
    )


# ============================================================================
# UI PANEL
# ============================================================================

class MOCAP_PT_main_panel(Panel):
    """Main panel for motion capture tools"""
    bl_label = "Melodic Justice MoCap"
    bl_idname = "MOCAP_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "MoCap"
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.mocap_props
        
        # File section
        box = layout.box()
        box.label(text="Motion Data", icon='FILE')
        
        row = box.row()
        row.prop(props, "mocap_filepath", text="")
        row.operator("mocap.load_file", text="", icon='FILE_FOLDER')
        
        if props.mocap_filepath and os.path.exists(props.mocap_filepath):
            box.label(text=f"Frames: {props.frame_count}")
            box.label(text=f"FPS: {props.fps}")
        
        # Settings section
        box = layout.box()
        box.label(text="Settings", icon='PREFERENCES')
        
        box.prop(props, "start_frame")
        box.prop(props, "scale")
        box.prop(props, "use_ik")
        
        row = box.row()
        row.prop(props, "smooth_motion")
        if props.smooth_motion:
            row.prop(props, "smooth_factor", text="")
        
        # Rig section
        box = layout.box()
        box.label(text="Target Rig", icon='ARMATURE_DATA')
        
        obj = context.active_object
        if obj and obj.type == 'ARMATURE':
            box.label(text=f"Selected: {obj.name}", icon='CHECKMARK')
            
            # Check for Rigify
            is_rigify = any(
                bone.name in ["torso", "hand_ik.L", "foot_ik.L"] 
                for bone in obj.pose.bones
            )
            if is_rigify:
                box.label(text="Rigify rig detected", icon='CHECKMARK')
            else:
                box.label(text="Warning: May not be Rigify", icon='ERROR')
        else:
            box.label(text="Select a Rigify armature", icon='ERROR')
        
        # Actions
        layout.separator()
        
        col = layout.column(align=True)
        col.scale_y = 1.5
        col.operator("mocap.retarget", icon='PLAY')
        
        layout.separator()
        
        layout.operator("mocap.clear_animation", icon='TRASH')


# ============================================================================
# REGISTRATION
# ============================================================================

classes = (
    MocapProperties,
    MOCAP_OT_load_file,
    MOCAP_OT_retarget,
    MOCAP_OT_clear_animation,
    MOCAP_OT_detect_cameras,
    MOCAP_PT_main_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.mocap_props = bpy.props.PointerProperty(type=MocapProperties)
    
    print("Melodic Justice MoCap addon registered")


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    del bpy.types.Scene.mocap_props


if __name__ == "__main__":
    register()
