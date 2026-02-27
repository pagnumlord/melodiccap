"""
Melodic Justice MoCap - FIXED Retargeting Addon
================================================

Key fixes:
1. Uses RELATIVE positions (offset from hip) not absolute
2. Better coordinate space handling
3. Debug output to see what's happening
4. Option to lock root/torso in place

Install: Edit > Preferences > Add-ons > Install > Select this file
"""

bl_info = {
    "name": "Melodic Justice MoCap v2",
    "author": "Melodic Justice Production",
    "version": (2, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > MoCap",
    "description": "Fixed motion capture retargeting for Rigify",
    "category": "Animation",
}

import bpy
import json
import math
from mathutils import Vector, Matrix, Euler, Quaternion
from bpy.props import (
    StringProperty, BoolProperty, FloatProperty,
    EnumProperty, IntProperty, PointerProperty
)
from bpy.types import Operator, Panel, PropertyGroup


# Rigify control bone mapping
# Maps mocap landmark -> rigify control bone
BONE_MAPPING = {
    # IK targets (location)
    "left_wrist": "hand_ik.L",
    "right_wrist": "hand_ik.R",
    "left_ankle": "foot_ik.L",
    "right_ankle": "foot_ik.R",
    
    # Pole targets (location) - for elbow/knee direction
    "left_elbow": "upper_arm_ik_target.L",
    "right_elbow": "upper_arm_ik_target.R",
    "left_knee": "thigh_ik_target.L",
    "right_knee": "thigh_ik_target.R",
    
    # Torso (location + rotation)
    "hip_center": "torso",
    
    # Spine/Chest
    "spine_center": "chest",
}


class MocapLoader:
    """Load and parse mocap JSON data."""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.frames = []
        self.fps = 30
        
        self._load()
    
    def _load(self):
        with open(self.filepath, 'r') as f:
            self.data = json.load(f)
        
        # Get metadata
        meta = self.data.get('metadata', {})
        self.fps = meta.get('fps', meta.get('target_fps', 30))
        
        # Get frames
        self.frames = self.data.get('frames', [])
        
        print(f"Loaded {len(self.frames)} frames @ {self.fps} fps")
    
    def get_landmark(self, frame_idx, landmark_name):
        """Get 3D position for a landmark at a frame."""
        if frame_idx >= len(self.frames):
            return None
        
        frame = self.frames[frame_idx]
        
        # Try landmarks_3d first, then world_landmarks
        for source in ['landmarks_3d', 'world_landmarks']:
            if source in frame and landmark_name in frame[source]:
                lm = frame[source][landmark_name]
                return Vector((lm['x'], lm['y'], lm['z']))
        
        return None
    
    def get_hip_center(self, frame_idx):
        """Get hip center position for a frame."""
        # Try computed hip_center first
        hip = self.get_landmark(frame_idx, 'hip_center')
        if hip:
            return hip
        
        # Calculate from left/right hip
        left = self.get_landmark(frame_idx, 'left_hip')
        right = self.get_landmark(frame_idx, 'right_hip')
        
        if left and right:
            return (left + right) / 2
        
        return Vector((0, 0, 0))


class FixedRetargeter:
    """
    Fixed retargeting that uses relative positions.
    
    Key insight: MediaPipe world_landmarks are centered at hip.
    We need to apply these as OFFSETS from the character's hip position.
    """
    
    def __init__(self, armature, mocap: MocapLoader, settings):
        self.armature = armature
        self.mocap = mocap
        self.settings = settings
        
        # Get character's rest hip position (in world space)
        self.rest_hip_pos = self._get_rest_hip_position()
        print(f"Rest hip position: {self.rest_hip_pos}")
    
    def _get_rest_hip_position(self):
        """Get the hip/torso position in rest pose."""
        if "torso" in self.armature.pose.bones:
            bone = self.armature.pose.bones["torso"]
            # Get world position of bone head
            return self.armature.matrix_world @ bone.bone.head_local
        return Vector((0, 0, 1))  # Default ~1m up
    
    def _get_bone(self, bone_name):
        """Safely get a pose bone."""
        if bone_name in self.armature.pose.bones:
            return self.armature.pose.bones[bone_name]
        return None
    
    def apply_frame(self, frame_idx, blender_frame):
        """Apply mocap data for one frame."""
        
        # Get hip center for this frame (reference point)
        mocap_hip = self.mocap.get_hip_center(frame_idx)
        
        scale = self.settings.scale_factor
        
        for landmark_name, bone_name in BONE_MAPPING.items():
            bone = self._get_bone(bone_name)
            if not bone:
                continue
            
            # Get mocap position
            mocap_pos = self.mocap.get_landmark(frame_idx, landmark_name)
            if mocap_pos is None:
                continue
            
            # === KEY FIX: Use position relative to hip ===
            if landmark_name == "hip_center":
                if self.settings.lock_root:
                    # Don't move the root at all
                    continue
                else:
                    # Move root by the hip offset only (small movements)
                    offset = mocap_pos * scale
                    # Only apply horizontal movement, minimal vertical
                    bone.location = Vector((offset.x, offset.y, offset.z * 0.1))
            else:
                # For limbs: position is RELATIVE to hip already in MediaPipe
                # So mocap_pos IS the offset from hip
                
                # Scale the offset
                offset = mocap_pos * scale
                
                # Apply to bone location
                bone.location = offset
            
            # Insert keyframe
            bone.keyframe_insert(data_path="location", frame=blender_frame)
    
    def retarget_all(self, start_frame=1):
        """Retarget all frames."""
        
        total = len(self.mocap.frames)
        print(f"\nRetargeting {total} frames...")
        print(f"Scale: {self.settings.scale_factor}")
        print(f"Lock root: {self.settings.lock_root}")
        
        # Set frame range
        bpy.context.scene.frame_start = start_frame
        bpy.context.scene.frame_end = start_frame + total - 1
        
        # Process frames
        for i in range(total):
            bpy.context.scene.frame_set(start_frame + i)
            self.apply_frame(i, start_frame + i)
            
            if i % 50 == 0:
                print(f"  Frame {i}/{total}")
        
        print(f"Done! Retargeted {total} frames")
        
        # Apply smoothing if enabled
        if self.settings.apply_smoothing:
            self._smooth_animation()
    
    def _smooth_animation(self):
        """Smooth the animation curves."""
        if not self.armature.animation_data or not self.armature.animation_data.action:
            return
        
        action = self.armature.animation_data.action
        window = self.settings.smoothing_window
        
        for fcurve in action.fcurves:
            if len(fcurve.keyframe_points) < window:
                continue
            
            # Get values
            values = [kp.co[1] for kp in fcurve.keyframe_points]
            
            # Simple moving average
            smoothed = []
            half = window // 2
            for i in range(len(values)):
                start = max(0, i - half)
                end = min(len(values), i + half + 1)
                smoothed.append(sum(values[start:end]) / (end - start))
            
            # Apply
            for i, kp in enumerate(fcurve.keyframe_points):
                kp.co[1] = smoothed[i]
            
            fcurve.update()


# =============================================================================
# UI Properties
# =============================================================================

class MocapSettings(PropertyGroup):
    filepath: StringProperty(
        name="File",
        subtype='FILE_PATH'
    )
    
    frame_count: IntProperty(default=0)
    detected_fps: FloatProperty(default=30.0)
    
    scale_factor: FloatProperty(
        name="Scale",
        default=1.0,
        min=0.001,
        max=100.0,
        description="Scale factor for mocap data"
    )
    
    start_frame: IntProperty(
        name="Start Frame",
        default=1,
        min=1
    )
    
    lock_root: BoolProperty(
        name="Lock Root",
        default=True,
        description="Keep the torso/root stationary (recommended for testing)"
    )
    
    apply_smoothing: BoolProperty(
        name="Smooth",
        default=True
    )
    
    smoothing_window: IntProperty(
        name="Window",
        default=5,
        min=3,
        max=15
    )


# =============================================================================
# Operators
# =============================================================================

class MOCAP_OT_LoadFile(Operator):
    bl_idname = "mocap.load_file"
    bl_label = "Load MoCap File"
    bl_options = {'REGISTER'}
    
    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        props = context.scene.mocap_settings
        props.filepath = self.filepath
        
        try:
            loader = MocapLoader(self.filepath)
            props.frame_count = len(loader.frames)
            props.detected_fps = loader.fps
            self.report({'INFO'}, f"Loaded {props.frame_count} frames")
        except Exception as e:
            self.report({'ERROR'}, f"Failed: {e}")
            return {'CANCELLED'}
        
        return {'FINISHED'}


class MOCAP_OT_ApplyRetarget(Operator):
    bl_idname = "mocap.apply_retarget"
    bl_label = "Apply to Rigify"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        props = context.scene.mocap_settings
        obj = context.active_object
        return props.filepath and obj and obj.type == 'ARMATURE'
    
    def execute(self, context):
        props = context.scene.mocap_settings
        armature = context.active_object
        
        # Load mocap
        try:
            loader = MocapLoader(props.filepath)
        except Exception as e:
            self.report({'ERROR'}, f"Load failed: {e}")
            return {'CANCELLED'}
        
        # Retarget
        retargeter = FixedRetargeter(armature, loader, props)
        retargeter.retarget_all(props.start_frame)
        
        self.report({'INFO'}, f"Applied {len(loader.frames)} frames")
        return {'FINISHED'}


class MOCAP_OT_ClearAnimation(Operator):
    bl_idname = "mocap.clear_anim"
    bl_label = "Clear"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'ARMATURE'
    
    def execute(self, context):
        arm = context.active_object
        if arm.animation_data and arm.animation_data.action:
            bpy.data.actions.remove(arm.animation_data.action)
            self.report({'INFO'}, "Cleared animation")
        return {'FINISHED'}


class MOCAP_OT_DebugPrint(Operator):
    """Print debug info about the mocap data"""
    bl_idname = "mocap.debug_print"
    bl_label = "Debug Info"
    
    def execute(self, context):
        props = context.scene.mocap_settings
        
        if not props.filepath:
            self.report({'WARNING'}, "No file loaded")
            return {'CANCELLED'}
        
        try:
            loader = MocapLoader(props.filepath)
            
            print("\n" + "="*50)
            print("MOCAP DEBUG INFO")
            print("="*50)
            print(f"Frames: {len(loader.frames)}")
            print(f"FPS: {loader.fps}")
            
            if loader.frames:
                frame = loader.frames[0]
                print(f"\nFrame 0 data sources: {list(frame.keys())}")
                
                # Print sample landmarks
                for source in ['landmarks_3d', 'world_landmarks']:
                    if source in frame:
                        print(f"\n{source} landmarks:")
                        for name in ['hip_center', 'left_hip', 'right_hip', 
                                    'left_wrist', 'right_wrist', 'left_ankle']:
                            if name in frame[source]:
                                lm = frame[source][name]
                                print(f"  {name}: ({lm['x']:.3f}, {lm['y']:.3f}, {lm['z']:.3f})")
            
            # Print rig info
            arm = context.active_object
            if arm and arm.type == 'ARMATURE':
                print(f"\nRig: {arm.name}")
                print("Available control bones:")
                for bn in BONE_MAPPING.values():
                    status = "✓" if bn in arm.pose.bones else "✗"
                    print(f"  {status} {bn}")
            
            print("="*50 + "\n")
            
            self.report({'INFO'}, "Debug info printed to console")
            
        except Exception as e:
            self.report({'ERROR'}, f"Error: {e}")
        
        return {'FINISHED'}


# =============================================================================
# UI Panel
# =============================================================================

class MOCAP_PT_MainPanel(Panel):
    bl_label = "Melodic Justice MoCap v2"
    bl_idname = "MOCAP_PT_main_v2"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "MoCap"
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.mocap_settings
        obj = context.active_object
        
        # Load
        box = layout.box()
        box.label(text="1. Load Data", icon='FILE')
        row = box.row()
        row.prop(props, "filepath", text="")
        row.operator("mocap.load_file", text="", icon='FILEBROWSER')
        
        if props.frame_count > 0:
            box.label(text=f"✓ {props.frame_count} frames @ {props.detected_fps:.1f} fps")
        
        # Target
        box = layout.box()
        box.label(text="2. Select Rig", icon='ARMATURE_DATA')
        if obj and obj.type == 'ARMATURE':
            box.label(text=f"✓ {obj.name}")
        else:
            box.label(text="Select a Rigify armature!", icon='ERROR')
        
        # Settings
        box = layout.box()
        box.label(text="3. Settings", icon='SETTINGS')
        
        box.prop(props, "scale_factor")
        box.prop(props, "lock_root")
        box.prop(props, "start_frame")
        
        row = box.row()
        row.prop(props, "apply_smoothing")
        if props.apply_smoothing:
            row.prop(props, "smoothing_window")
        
        # Apply
        box = layout.box()
        box.label(text="4. Apply", icon='PLAY')
        row = box.row()
        row.scale_y = 1.5
        row.operator("mocap.apply_retarget", icon='ARMATURE_DATA')
        
        # Utilities
        box = layout.box()
        box.label(text="Utilities", icon='TOOL_SETTINGS')
        row = box.row()
        row.operator("mocap.clear_anim")
        row.operator("mocap.debug_print")
        
        # Help
        box = layout.box()
        box.label(text="Tips:", icon='INFO')
        col = box.column(align=True)
        col.scale_y = 0.8
        col.label(text="• Start with Lock Root ON")
        col.label(text="• Try Scale 0.5 - 2.0")
        col.label(text="• Use Debug to check data")


# =============================================================================
# Registration
# =============================================================================

classes = [
    MocapSettings,
    MOCAP_OT_LoadFile,
    MOCAP_OT_ApplyRetarget,
    MOCAP_OT_ClearAnimation,
    MOCAP_OT_DebugPrint,
    MOCAP_PT_MainPanel,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mocap_settings = PointerProperty(type=MocapSettings)
    print("Melodic Justice MoCap v2 registered")

def unregister():
    del bpy.types.Scene.mocap_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
