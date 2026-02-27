"""
Complete Motion Capture to Rigify System
Production-ready addon for Melodic Justice short film

Handles:
- JSON import with proper coordinate conversion
- Bone rotation calculation from positions
- Auto-scaling to character height
- A-pose recording support
- Full Rigify control bone retargeting
- IK/FK mode support
- Constraint-based animation transfer
- One-click baking

Installation:
1. Save as mocap_rigify_final.py
2. Blender > Edit > Preferences > Add-ons > Install
3. Enable "Animation: Complete Mocap to Rigify System"
4. Panel in 3D View > Sidebar (N) > Mocap

Author: Melodic Justice Production Team
Version: 4.0.0 Final
Blender: 4.4.3
"""

bl_info = {
    "name": "Complete Mocap to Rigify System",
    "author": "Melodic Justice Production",
    "version": (4, 0, 0),
    "blender": (4, 4, 3),
    "location": "3D View > Sidebar > Mocap Tab",
    "description": "Complete motion capture import and Rigify retargeting with rotation fixing",
    "category": "Animation",
}

import bpy
import json
import os
import math
from mathutils import Vector, Matrix, Quaternion, Euler
from bpy.props import StringProperty, BoolProperty, EnumProperty, FloatProperty, IntProperty
from bpy_extras.io_utils import ImportHelper
from bpy.types import Operator, Panel, PropertyGroup


# ============================================================================
# COORDINATE SYSTEM CONVERSION
# ============================================================================

def mediapipe_to_blender(x, y, z):
    """
    Convert MediaPipe coordinates to Blender coordinates
    
    MediaPipe: X=right, Y=down, Z=forward (away from camera)
    Blender: X=right, Y=forward, Z=up
    
    Conversion: Blender_X = MediaPipe_X
                Blender_Y = MediaPipe_Z
                Blender_Z = -MediaPipe_Y
    """
    return Vector((x, z, -y))


# ============================================================================
# MOCAP ARMATURE BUILDER WITH ROTATION CALCULATION
# ============================================================================

class MocapArmatureBuilder:
    """
    Builds animated armature from mocap data with proper rotations
    
    Key improvements:
    - Calculates bone rotations from position data
    - Handles coordinate conversion
    - Auto-scales to character height
    - Preserves bone hierarchy
    """
    
    # MediaPipe landmark indices
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    
    def __init__(self, name="MocapArmature"):
        self.name = name
        self.bone_hierarchy = {}
        
    def get_landmark_position(self, landmarks, idx, scale=1.0):
        """Get landmark position in Blender space"""
        if idx >= len(landmarks):
            return Vector((0, 0, 0))
        
        lm = landmarks[idx]
        x = float(lm.get('x', 0)) * scale
        y = float(lm.get('y', 0)) * scale
        z = float(lm.get('z', 0)) * scale
        
        return mediapipe_to_blender(x, y, z)
    
    def create_armature(self, context, frame_data, target_height=1.87, auto_scale=True):
        """
        Create armature with proper scaling and bone hierarchy
        
        Args:
            target_height: Character height in meters (default 1.87m for Jax)
            auto_scale: Automatically scale mocap to match target height
        """
        
        if not frame_data or 'landmarks_3d' not in frame_data[0]:
            print("ERROR: No landmarks_3d in frame data")
            return None
        
        landmarks = frame_data[0]['landmarks_3d']
        
        if not isinstance(landmarks, list) or len(landmarks) < 33:
            print(f"ERROR: Expected 33 landmarks, got {len(landmarks)}")
            return None
        
        # Calculate scale factor
        if auto_scale:
            # Measure height in mocap data (hip to head)
            left_hip = self.get_landmark_position(landmarks, self.LEFT_HIP, scale=1.0)
            right_hip = self.get_landmark_position(landmarks, self.RIGHT_HIP, scale=1.0)
            hips_pos = (left_hip + right_hip) / 2
            
            nose = self.get_landmark_position(landmarks, self.NOSE, scale=1.0)
            
            mocap_height = (nose - hips_pos).length
            scale = target_height / mocap_height if mocap_height > 0 else 1.0
            
            print(f"Auto-scaling: mocap height={mocap_height:.3f}m, target={target_height}m, scale={scale:.3f}")
        else:
            scale = 1.0
        
        # Create armature
        armature = bpy.data.armatures.new(name=self.name)
        armature_obj = bpy.data.objects.new(self.name, armature)
        context.collection.objects.link(armature_obj)
        
        context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Build skeleton
        self.create_skeleton(armature, landmarks, scale)
        
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Position at origin
        armature_obj.location = (0, 0, 0)
        
        print(f"Created armature: {self.name} with {len(armature.edit_bones)} bones")
        
        return armature_obj
    
    def create_skeleton(self, armature, landmarks, scale):
        """Create bone hierarchy with proper structure"""
        
        # Get key positions
        left_hip = self.get_landmark_position(landmarks, self.LEFT_HIP, scale)
        right_hip = self.get_landmark_position(landmarks, self.RIGHT_HIP, scale)
        hips_pos = (left_hip + right_hip) / 2
        
        left_shoulder = self.get_landmark_position(landmarks, self.LEFT_SHOULDER, scale)
        right_shoulder = self.get_landmark_position(landmarks, self.RIGHT_SHOULDER, scale)
        shoulders_pos = (left_shoulder + right_shoulder) / 2
        
        nose = self.get_landmark_position(landmarks, self.NOSE, scale)
        
        # ROOT (at hips)
        root = armature.edit_bones.new('root')
        root.head = hips_pos
        root.tail = hips_pos + Vector((0, 0, 0.1))
        self.bone_hierarchy['root'] = None
        
        # SPINE (hips to shoulders)
        spine = armature.edit_bones.new('spine')
        spine.head = hips_pos
        spine.tail = shoulders_pos
        spine.parent = root
        self.bone_hierarchy['spine'] = 'root'
        
        # HEAD (shoulders to nose)
        head = armature.edit_bones.new('head')
        head.head = shoulders_pos
        head.tail = nose + Vector((0, 0, 0.1))
        head.parent = spine
        self.bone_hierarchy['head'] = 'spine'
        
        # LEFT ARM CHAIN
        left_elbow = self.get_landmark_position(landmarks, self.LEFT_ELBOW, scale)
        left_wrist = self.get_landmark_position(landmarks, self.LEFT_WRIST, scale)
        
        upper_arm_l = armature.edit_bones.new('upper_arm.L')
        upper_arm_l.head = left_shoulder
        upper_arm_l.tail = left_elbow
        upper_arm_l.parent = spine
        self.bone_hierarchy['upper_arm.L'] = 'spine'
        
        forearm_l = armature.edit_bones.new('forearm.L')
        forearm_l.head = left_elbow
        forearm_l.tail = left_wrist
        forearm_l.parent = upper_arm_l
        self.bone_hierarchy['forearm.L'] = 'upper_arm.L'
        
        hand_l = armature.edit_bones.new('hand.L')
        hand_l.head = left_wrist
        hand_l.tail = left_wrist + (left_wrist - left_elbow).normalized() * 0.05
        hand_l.parent = forearm_l
        self.bone_hierarchy['hand.L'] = 'forearm.L'
        
        # RIGHT ARM CHAIN
        right_elbow = self.get_landmark_position(landmarks, self.RIGHT_ELBOW, scale)
        right_wrist = self.get_landmark_position(landmarks, self.RIGHT_WRIST, scale)
        
        upper_arm_r = armature.edit_bones.new('upper_arm.R')
        upper_arm_r.head = right_shoulder
        upper_arm_r.tail = right_elbow
        upper_arm_r.parent = spine
        self.bone_hierarchy['upper_arm.R'] = 'spine'
        
        forearm_r = armature.edit_bones.new('forearm.R')
        forearm_r.head = right_elbow
        forearm_r.tail = right_wrist
        forearm_r.parent = upper_arm_r
        self.bone_hierarchy['forearm.R'] = 'upper_arm.R'
        
        hand_r = armature.edit_bones.new('hand.R')
        hand_r.head = right_wrist
        hand_r.tail = right_wrist + (right_wrist - right_elbow).normalized() * 0.05
        hand_r.parent = forearm_r
        self.bone_hierarchy['hand.R'] = 'forearm.R'
        
        # LEFT LEG CHAIN
        left_knee = self.get_landmark_position(landmarks, self.LEFT_KNEE, scale)
        left_ankle = self.get_landmark_position(landmarks, self.LEFT_ANKLE, scale)
        
        thigh_l = armature.edit_bones.new('thigh.L')
        thigh_l.head = left_hip
        thigh_l.tail = left_knee
        thigh_l.parent = root
        self.bone_hierarchy['thigh.L'] = 'root'
        
        shin_l = armature.edit_bones.new('shin.L')
        shin_l.head = left_knee
        shin_l.tail = left_ankle
        shin_l.parent = thigh_l
        self.bone_hierarchy['shin.L'] = 'thigh.L'
        
        foot_l = armature.edit_bones.new('foot.L')
        foot_l.head = left_ankle
        foot_l.tail = left_ankle + Vector((0, 0.05, 0))
        foot_l.parent = shin_l
        self.bone_hierarchy['foot.L'] = 'shin.L'
        
        # RIGHT LEG CHAIN
        right_knee = self.get_landmark_position(landmarks, self.RIGHT_KNEE, scale)
        right_ankle = self.get_landmark_position(landmarks, self.RIGHT_ANKLE, scale)
        
        thigh_r = armature.edit_bones.new('thigh.R')
        thigh_r.head = right_hip
        thigh_r.tail = right_knee
        thigh_r.parent = root
        self.bone_hierarchy['thigh.R'] = 'root'
        
        shin_r = armature.edit_bones.new('shin.R')
        shin_r.head = right_knee
        shin_r.tail = right_ankle
        shin_r.parent = thigh_r
        self.bone_hierarchy['shin.R'] = 'thigh.R'
        
        foot_r = armature.edit_bones.new('foot.R')
        foot_r.head = right_ankle
        foot_r.tail = right_ankle + Vector((0, 0.05, 0))
        foot_r.parent = shin_r
        self.bone_hierarchy['foot.R'] = 'shin.R'
    
    def animate_with_rotations(self, armature_obj, frame_data, scale, fps=30):
        """
        Animate armature with both positions AND rotations
        
        This is the critical fix - calculates rotations from position changes
        """
        
        bpy.context.scene.render.fps = fps
        
        if armature_obj.animation_data:
            armature_obj.animation_data_clear()
        
        action = bpy.data.actions.new(name=f"{armature_obj.name}_Action")
        armature_obj.animation_data_create()
        armature_obj.animation_data.action = action
        
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='POSE')
        
        print(f"Animating {len(frame_data)} frames...")
        
        # Process each frame
        for frame_num, frame in enumerate(frame_data):
            if 'landmarks_3d' not in frame:
                continue
            
            landmarks = frame['landmarks_3d']
            bpy.context.scene.frame_set(frame_num + 1)
            
            # Update bone positions and rotations
            self.update_bone_transforms(armature_obj, landmarks, scale)
            
            # Keyframe everything
            for bone in armature_obj.pose.bones:
                bone.keyframe_insert(data_path='location', frame=frame_num + 1)
                bone.keyframe_insert(data_path='rotation_quaternion', frame=frame_num + 1)
            
            if (frame_num + 1) % 50 == 0:
                print(f"  Frame {frame_num + 1}/{len(frame_data)}")
        
        bpy.ops.object.mode_set(mode='OBJECT')
        
        print(f"✅ Animation complete: {len(frame_data)} frames")
    
    def update_bone_transforms(self, armature_obj, landmarks, scale):
        """Update bone positions and calculate rotations for current frame"""
        
        # Get key positions for this frame
        left_hip = self.get_landmark_position(landmarks, self.LEFT_HIP, scale)
        right_hip = self.get_landmark_position(landmarks, self.RIGHT_HIP, scale)
        hips_pos = (left_hip + right_hip) / 2
        
        # Update root position
        if 'root' in armature_obj.pose.bones:
            armature_obj.pose.bones['root'].location = hips_pos
        
        # For each bone with children, calculate rotation to point at child
        for bone in armature_obj.pose.bones:
            if bone.children:
                self.calculate_bone_rotation(armature_obj, bone, landmarks, scale)
    
    def calculate_bone_rotation(self, armature_obj, bone, landmarks, scale):
        """Calculate rotation for bone to point at its child"""
        
        child = bone.children[0]
        
        # Map bone names to landmark indices
        landmark_map = {
            'spine': (self.LEFT_HIP, self.RIGHT_HIP, self.LEFT_SHOULDER, self.RIGHT_SHOULDER),
            'head': (self.LEFT_SHOULDER, self.RIGHT_SHOULDER, self.NOSE, self.NOSE),
            'upper_arm.L': (self.LEFT_SHOULDER, self.LEFT_ELBOW),
            'forearm.L': (self.LEFT_ELBOW, self.LEFT_WRIST),
            'upper_arm.R': (self.RIGHT_SHOULDER, self.RIGHT_ELBOW),
            'forearm.R': (self.RIGHT_ELBOW, self.RIGHT_WRIST),
            'thigh.L': (self.LEFT_HIP, self.LEFT_KNEE),
            'shin.L': (self.LEFT_KNEE, self.LEFT_ANKLE),
            'thigh.R': (self.RIGHT_HIP, self.RIGHT_KNEE),
            'shin.R': (self.RIGHT_KNEE, self.RIGHT_ANKLE),
        }
        
        if bone.name in landmark_map:
            indices = landmark_map[bone.name]
            
            if len(indices) == 2:
                # Simple bone - head to tail
                head_pos = self.get_landmark_position(landmarks, indices[0], scale)
                tail_pos = self.get_landmark_position(landmarks, indices[1], scale)
            else:
                # Averaged bone (spine)
                pos1 = self.get_landmark_position(landmarks, indices[0], scale)
                pos2 = self.get_landmark_position(landmarks, indices[1], scale)
                head_pos = (pos1 + pos2) / 2
                
                pos3 = self.get_landmark_position(landmarks, indices[2], scale)
                pos4 = self.get_landmark_position(landmarks, indices[3], scale)
                tail_pos = (pos3 + pos4) / 2
            
            # Calculate direction vector
            direction = (tail_pos - head_pos).normalized()
            
            # Bone's local Y-axis points along its length
            up_vector = Vector((0, 1, 0))
            
            # Calculate rotation from up_vector to direction
            if direction.length > 0.001:
                rotation = up_vector.rotation_difference(direction)
                bone.rotation_quaternion = rotation


# ============================================================================
# RIGIFY RETARGETER
# ============================================================================

class RigifyRetargeter:
    """
    Retargets mocap to Rigify control bones
    
    Supports:
    - IK mode (hand_ik, foot_ik)
    - FK mode (hand_fk, foot_fk, etc.)
    - Proper bone mapping for JaxRigify structure
    """
    
    def __init__(self, source_rig, target_rig, use_ik=True):
        self.source = source_rig
        self.target = target_rig
        self.use_ik = use_ik
        
        # Bone mapping based on JaxRigify structure
        if use_ik:
            self.bone_map = {
                # Core
                'root': ('torso', 'COPY_LOCATION'),
                'spine': ('chest', 'COPY_ROTATION'),
                'head': ('head', 'COPY_ROTATION'),
                # Arms IK
                'hand.L': ('hand_ik.L', 'COPY_LOCATION'),
                'hand.R': ('hand_ik.R', 'COPY_LOCATION'),
                # Legs IK
                'foot.L': ('foot_ik.L', 'COPY_LOCATION'),
                'foot.R': ('foot_ik.R', 'COPY_LOCATION'),
            }
        else:
            self.bone_map = {
                # Core
                'root': ('torso', 'COPY_LOCATION'),
                'spine': ('chest', 'COPY_ROTATION'),
                'head': ('head', 'COPY_ROTATION'),
                # Arms FK
                'upper_arm.L': ('upper_arm_fk.L', 'COPY_ROTATION'),
                'forearm.L': ('forearm_fk.L', 'COPY_ROTATION'),
                'hand.L': ('hand_fk.L', 'COPY_ROTATION'),
                'upper_arm.R': ('upper_arm_fk.R', 'COPY_ROTATION'),
                'forearm.R': ('forearm_fk.R', 'COPY_ROTATION'),
                'hand.R': ('hand_fk.R', 'COPY_ROTATION'),
                # Legs FK
                'thigh.L': ('thigh_fk.L', 'COPY_ROTATION'),
                'shin.L': ('shin_fk.L', 'COPY_ROTATION'),
                'foot.L': ('foot_fk.L', 'COPY_ROTATION'),
                'thigh.R': ('thigh_fk.R', 'COPY_ROTATION'),
                'shin.R': ('shin_fk.R', 'COPY_ROTATION'),
                'foot.R': ('foot_fk.R', 'COPY_ROTATION'),
            }
    
    def setup_constraints(self):
        """Add retargeting constraints"""
        
        constraints_added = 0
        missing_bones = []
        
        for source_bone_name, (target_bone_name, constraint_type) in self.bone_map.items():
            # Check source
            if source_bone_name not in self.source.pose.bones:
                continue
            
            # Check target
            if target_bone_name not in self.target.pose.bones:
                missing_bones.append(target_bone_name)
                continue
            
            target_bone = self.target.pose.bones[target_bone_name]
            
            # Remove old mocap constraints
            for con in list(target_bone.constraints):
                if con.name.startswith('MOCAP_'):
                    target_bone.constraints.remove(con)
            
            # Add new constraint
            if constraint_type == 'COPY_LOCATION':
                con = target_bone.constraints.new('COPY_LOCATION')
                con.name = f'MOCAP_{source_bone_name}'
                con.target = self.source
                con.subtarget = source_bone_name
                con.use_offset = False
                con.target_space = 'WORLD'
                con.owner_space = 'WORLD'
                con.influence = 1.0
                constraints_added += 1
                
            elif constraint_type == 'COPY_ROTATION':
                con = target_bone.constraints.new('COPY_ROTATION')
                con.name = f'MOCAP_{source_bone_name}'
                con.target = self.source
                con.subtarget = source_bone_name
                con.use_offset = False
                con.target_space = 'WORLD'
                con.owner_space = 'WORLD'
                con.mix_mode = 'REPLACE'
                con.influence = 1.0
                constraints_added += 1
        
        if missing_bones:
            print(f"Warning: {len(missing_bones)} target bones not found: {missing_bones[:5]}")
        
        return constraints_added
    
    def bake_animation(self, frame_start, frame_end):
        """Bake constraints to keyframes"""
        
        bpy.context.view_layer.objects.active = self.target
        self.target.select_set(True)
        
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.select_all(action='DESELECT')
        
        # Select bones with mocap constraints
        for bone in self.target.pose.bones:
            for con in bone.constraints:
                if con.name.startswith('MOCAP_'):
                    bone.bone.select = True
                    break
        
        # Bake with visual keying
        try:
            bpy.ops.nla.bake(
                frame_start=frame_start,
                frame_end=frame_end,
                only_selected=True,
                visual_keying=True,
                clear_constraints=False,
                use_current_action=True,
                bake_types={'POSE'}
            )
            return True
        except Exception as e:
            print(f"Baking error: {e}")
            return False
        finally:
            bpy.ops.object.mode_set(mode='OBJECT')
    
    def remove_constraints(self):
        """Remove mocap constraints"""
        count = 0
        for bone in self.target.pose.bones:
            for con in list(bone.constraints):
                if con.name.startswith('MOCAP_'):
                    bone.constraints.remove(con)
                    count += 1
        return count


# ============================================================================
# OPERATORS
# ============================================================================

class MOCAP_OT_import_json(Operator, ImportHelper):
    """Import motion capture JSON file"""
    bl_idname = "mocap.import_json"
    bl_label = "Import Mocap JSON"
    bl_options = {'REGISTER', 'UNDO'}
    
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    
    target_height: FloatProperty(
        name="Character Height (m)",
        description="Height of your character in meters",
        default=1.87,
        min=0.5,
        max=3.0
    )
    
    auto_scale: BoolProperty(
        name="Auto Scale",
        description="Automatically scale mocap to match character height",
        default=True
    )
    
    def execute(self, context):
        print("\n" + "="*70)
        print(f"IMPORTING: {self.filepath}")
        print("="*70 + "\n")
        
        # Load JSON
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load JSON: {e}")
            return {'CANCELLED'}
        
        if 'frames' not in data:
            self.report({'ERROR'}, "Invalid JSON: no 'frames' key")
            return {'CANCELLED'}
        
        frame_data = data['frames']
        
        if not frame_data:
            self.report({'ERROR'}, "No frames in file")
            return {'CANCELLED'}
        
        # Verify 3D landmarks exist
        if 'landmarks_3d' not in frame_data[0]:
            self.report({'ERROR'}, "No 3D landmarks in frames")
            return {'CANCELLED'}
        
        if len(frame_data[0]['landmarks_3d']) == 0:
            self.report({'ERROR'}, "3D landmarks array is empty - triangulation failed during recording")
            return {'CANCELLED'}
        
        # Create armature
        take_name = data.get('take_name', 'Take')
        builder = MocapArmatureBuilder(name=f"Mocap_{take_name}")
        
        armature_obj = builder.create_armature(
            context, 
            frame_data, 
            target_height=self.target_height,
            auto_scale=self.auto_scale
        )
        
        if not armature_obj:
            self.report({'ERROR'}, "Failed to create armature")
            return {'CANCELLED'}
        
        # Calculate scale
        if self.auto_scale:
            landmarks = frame_data[0]['landmarks_3d']
            left_hip = builder.get_landmark_position(landmarks, builder.LEFT_HIP, scale=1.0)
            right_hip = builder.get_landmark_position(landmarks, builder.RIGHT_HIP, scale=1.0)
            hips_pos = (left_hip + right_hip) / 2
            nose = builder.get_landmark_position(landmarks, builder.NOSE, scale=1.0)
            mocap_height = (nose - hips_pos).length
            scale = self.target_height / mocap_height if mocap_height > 0 else 1.0
        else:
            scale = 1.0
        
        # Animate with rotations
        builder.animate_with_rotations(
            armature_obj,
            frame_data,
            scale,
            fps=data.get('fps', 30)
        )
        
        self.report({'INFO'}, f"Imported {len(frame_data)} frames")
        print("\n" + "="*70)
        print("IMPORT COMPLETE")
        print("="*70 + "\n")
        
        return {'FINISHED'}


class MOCAP_OT_retarget(Operator):
    """Setup retargeting constraints"""
    bl_idname = "mocap.retarget"
    bl_label = "Setup Retargeting"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.mocap_props
        
        if not props.source_rig or not props.target_rig:
            self.report({'ERROR'}, "Select both source and target rigs")
            return {'CANCELLED'}
        
        source = bpy.data.objects.get(props.source_rig)
        target = bpy.data.objects.get(props.target_rig)
        
        if not source or not target:
            self.report({'ERROR'}, "Invalid rigs")
            return {'CANCELLED'}
        
        retargeter = RigifyRetargeter(source, target, props.use_ik)
        count = retargeter.setup_constraints()
        
        self.report({'INFO'}, f"Added {count} constraints")
        print(f"\n✅ Setup {count} retargeting constraints")
        
        return {'FINISHED'}


class MOCAP_OT_bake(Operator):
    """Bake animation to keyframes"""
    bl_idname = "mocap.bake"
    bl_label = "Bake Animation"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.mocap_props
        
        if not props.source_rig or not props.target_rig:
            self.report({'ERROR'}, "Select rigs first")
            return {'CANCELLED'}
        
        source = bpy.data.objects.get(props.source_rig)
        target = bpy.data.objects.get(props.target_rig)
        
        if not source or not target:
            self.report({'ERROR'}, "Invalid rigs")
            return {'CANCELLED'}
        
        # Get frame range
        if source.animation_data and source.animation_data.action:
            action = source.animation_data.action
            frame_start = int(action.frame_range[0])
            frame_end = int(action.frame_range[1])
        else:
            frame_start = context.scene.frame_start
            frame_end = context.scene.frame_end
        
        print(f"\nBaking frames {frame_start}-{frame_end}...")
        
        retargeter = RigifyRetargeter(source, target, props.use_ik)
        success = retargeter.bake_animation(frame_start, frame_end)
        
        if success:
            self.report({'INFO'}, f"Baked {frame_end - frame_start + 1} frames")
            print(f"✅ Baking complete!")
        else:
            self.report({'ERROR'}, "Baking failed")
            print(f"❌ Baking failed")
        
        return {'FINISHED' if success else 'CANCELLED'}


class MOCAP_OT_remove_constraints(Operator):
    """Remove retargeting constraints"""
    bl_idname = "mocap.remove_constraints"
    bl_label = "Remove Constraints"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.mocap_props
        
        if not props.target_rig:
            self.report({'ERROR'}, "Select target rig")
            return {'CANCELLED'}
        
        target = bpy.data.objects.get(props.target_rig)
        
        if not target:
            self.report({'ERROR'}, "Invalid target")
            return {'CANCELLED'}
        
        retargeter = RigifyRetargeter(None, target, False)
        count = retargeter.remove_constraints()
        
        self.report({'INFO'}, f"Removed {count} constraints")
        print(f"\n✅ Removed {count} constraints")
        
        return {'FINISHED'}


# ============================================================================
# PROPERTIES
# ============================================================================

class MocapProperties(PropertyGroup):
    """Properties for mocap system"""
    
    source_rig: StringProperty(
        name="Source (Mocap)",
        description="Mocap armature to retarget from"
    )
    
    target_rig: StringProperty(
        name="Target (Rigify)",
        description="Rigify character rig to retarget to"
    )
    
    use_ik: BoolProperty(
        name="Use IK Controls",
        description="Use IK controls (hand_ik, foot_ik) instead of FK",
        default=True
    )


# ============================================================================
# UI PANEL
# ============================================================================

class MOCAP_PT_main(Panel):
    """Main control panel"""
    bl_label = "Mocap to Rigify"
    bl_idname = "MOCAP_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Mocap'
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.mocap_props
        
        # STEP 1: IMPORT
        box = layout.box()
        box.label(text="1. Import Mocap", icon='IMPORT')
        box.operator("mocap.import_json", text="Import JSON File", icon='FILE_FOLDER')
        
        # STEP 2: RETARGET
        box = layout.box()
        box.label(text="2. Retarget to Rigify", icon='ARMATURE_DATA')
        
        box.prop_search(props, "source_rig", bpy.data, "objects", text="Source")
        box.prop_search(props, "target_rig", bpy.data, "objects", text="Target")
        box.prop(props, "use_ik")
        
        box.operator("mocap.retarget", text="Setup Retargeting", icon='CON_ARMATURE')
        
        # STEP 3: BAKE
        box = layout.box()
        box.label(text="3. Finalize", icon='RENDER_ANIMATION')
        box.operator("mocap.bake", text="Bake to Keyframes", icon='KEYFRAME')
        box.operator("mocap.remove_constraints", text="Remove Constraints", icon='X')
        
        # TIPS
        box = layout.box()
        box.label(text="Tips:", icon='INFO')
        box.label(text="• Set IK/FK switches to 1.0 first")
        box.label(text="• Preview animation before baking")
        box.label(text="• Remove constraints after bake")
        box.label(text="• Jax is 1.87m tall by default")


# ============================================================================
# REGISTRATION
# ============================================================================

classes = (
    MocapProperties,
    MOCAP_OT_import_json,
    MOCAP_OT_retarget,
    MOCAP_OT_bake,
    MOCAP_OT_remove_constraints,
    MOCAP_PT_main,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.mocap_props = bpy.props.PointerProperty(type=MocapProperties)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    del bpy.types.Scene.mocap_props


if __name__ == "__main__":
    register()
