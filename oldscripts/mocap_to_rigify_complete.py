"""
Motion Capture to Rigify - Complete Working System
Imports mocap JSON and retargets to Rigify rigs with proper scaling

Installation:
1. Save as mocap_to_rigify_complete.py
2. Blender > Edit > Preferences > Add-ons > Install
3. Enable "Animation: Motion Capture to Rigify Complete"
4. Find panel in 3D View > Sidebar (N) > Mocap tab
"""

bl_info = {
    "name": "Motion Capture to Rigify Complete",
    "author": "Melodic Justice Production",
    "version": (3, 0, 0),
    "blender": (4, 4, 3),
    "location": "3D View > Sidebar > Mocap Tab",
    "description": "Complete mocap import and Rigify retargeting system",
    "category": "Animation",
}

import bpy
import json
import os
from mathutils import Vector, Matrix
from bpy.props import StringProperty, BoolProperty, EnumProperty, FloatProperty
from bpy_extras.io_utils import ImportHelper
from bpy.types import Operator, Panel, PropertyGroup


# ============================================================================
# MOCAP ARMATURE BUILDER
# ============================================================================

class MocapArmatureBuilder:
    """Builds and animates armature from mocap data"""
    
    def __init__(self, name="MocapArmature"):
        self.name = name
        
    def create_armature(self, context, frame_data, scale=0.15):
        """Create armature with proper scaling"""
        
        if not frame_data or 'landmarks_3d' not in frame_data[0]:
            return None
            
        landmarks = frame_data[0]['landmarks_3d']
        
        if not isinstance(landmarks, list) or len(landmarks) == 0:
            print("ERROR: landmarks_3d is not a list or is empty")
            return None
        
        # Create armature
        armature = bpy.data.armatures.new(name=self.name)
        armature_obj = bpy.data.objects.new(self.name, armature)
        context.collection.objects.link(armature_obj)
        
        context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Create skeleton
        self.create_skeleton(armature, landmarks, scale)
        
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Center at origin
        armature_obj.location = (0, 0, 0)
        
        return armature_obj
        
    def create_skeleton(self, armature, landmarks, scale):
        """Create bone hierarchy"""
        
        def get_pos(idx, scale=1.0):
            if idx < len(landmarks):
                lm = landmarks[idx]
                x = float(lm.get('x', 0)) * scale
                y = float(lm.get('y', 0)) * scale
                z = float(lm.get('z', 0)) * scale
                # Blender coords: X=right, Y=forward, Z=up
                return Vector((x, z, -y))
            return Vector((0, 0, 0))
        
        # Get key positions
        left_hip = get_pos(23, scale)
        right_hip = get_pos(24, scale)
        hips_pos = (left_hip + right_hip) / 2
        
        left_shoulder = get_pos(11, scale)
        right_shoulder = get_pos(12, scale)
        shoulders_pos = (left_shoulder + right_shoulder) / 2
        
        # ROOT
        root = armature.edit_bones.new('root')
        root.head = hips_pos
        root.tail = hips_pos + Vector((0, 0, 0.05))
        
        # SPINE
        spine = armature.edit_bones.new('spine')
        spine.head = hips_pos
        spine.tail = shoulders_pos
        spine.parent = root
        
        # HEAD
        nose = get_pos(0, scale)
        head = armature.edit_bones.new('head')
        head.head = shoulders_pos
        head.tail = nose + Vector((0, 0, 0.05))
        head.parent = spine
        
        # LEFT ARM
        left_elbow = get_pos(13, scale)
        left_wrist = get_pos(15, scale)
        
        upper_arm_l = armature.edit_bones.new('upper_arm.L')
        upper_arm_l.head = left_shoulder
        upper_arm_l.tail = left_elbow
        upper_arm_l.parent = spine
        
        forearm_l = armature.edit_bones.new('forearm.L')
        forearm_l.head = left_elbow
        forearm_l.tail = left_wrist
        forearm_l.parent = upper_arm_l
        
        hand_l = armature.edit_bones.new('hand.L')
        hand_l.head = left_wrist
        hand_l.tail = left_wrist + (left_wrist - left_elbow).normalized() * 0.05
        hand_l.parent = forearm_l
        
        # RIGHT ARM
        right_elbow = get_pos(14, scale)
        right_wrist = get_pos(16, scale)
        
        upper_arm_r = armature.edit_bones.new('upper_arm.R')
        upper_arm_r.head = right_shoulder
        upper_arm_r.tail = right_elbow
        upper_arm_r.parent = spine
        
        forearm_r = armature.edit_bones.new('forearm.R')
        forearm_r.head = right_elbow
        forearm_r.tail = right_wrist
        forearm_r.parent = upper_arm_r
        
        hand_r = armature.edit_bones.new('hand.R')
        hand_r.head = right_wrist
        hand_r.tail = right_wrist + (right_wrist - right_elbow).normalized() * 0.05
        hand_r.parent = forearm_r
        
        # LEFT LEG
        left_knee = get_pos(25, scale)
        left_ankle = get_pos(27, scale)
        
        thigh_l = armature.edit_bones.new('thigh.L')
        thigh_l.head = left_hip
        thigh_l.tail = left_knee
        thigh_l.parent = root
        
        shin_l = armature.edit_bones.new('shin.L')
        shin_l.head = left_knee
        shin_l.tail = left_ankle
        shin_l.parent = thigh_l
        
        foot_l = armature.edit_bones.new('foot.L')
        foot_l.head = left_ankle
        foot_l.tail = left_ankle + Vector((0, 0.05, 0))
        foot_l.parent = shin_l
        
        # RIGHT LEG
        right_knee = get_pos(26, scale)
        right_ankle = get_pos(28, scale)
        
        thigh_r = armature.edit_bones.new('thigh.R')
        thigh_r.head = right_hip
        thigh_r.tail = right_knee
        thigh_r.parent = root
        
        shin_r = armature.edit_bones.new('shin.R')
        shin_r.head = right_knee
        shin_r.tail = right_ankle
        shin_r.parent = thigh_r
        
        foot_r = armature.edit_bones.new('foot.R')
        foot_r.head = right_ankle
        foot_r.tail = right_ankle + Vector((0, 0.05, 0))
        foot_r.parent = shin_r
        
    def animate_armature(self, armature_obj, frame_data, scale=0.15, fps=30):
        """Animate armature from mocap data"""
        
        bpy.context.scene.render.fps = fps
        
        if armature_obj.animation_data:
            armature_obj.animation_data_clear()
            
        action = bpy.data.actions.new(name=f"{armature_obj.name}_Action")
        armature_obj.animation_data_create()
        armature_obj.animation_data.action = action
        
        def get_pos(landmarks, idx, scale=1.0):
            if idx < len(landmarks):
                lm = landmarks[idx]
                x = float(lm.get('x', 0)) * scale
                y = float(lm.get('y', 0)) * scale
                z = float(lm.get('z', 0)) * scale
                return Vector((x, z, -y))
            return Vector((0, 0, 0))
        
        for frame_num, frame in enumerate(frame_data):
            if 'landmarks_3d' not in frame:
                continue
                
            landmarks = frame['landmarks_3d']
            bpy.context.scene.frame_set(frame_num + 1)
            
            # Animate root
            left_hip = get_pos(landmarks, 23, scale)
            right_hip = get_pos(landmarks, 24, scale)
            hips_pos = (left_hip + right_hip) / 2
            
            if 'root' in armature_obj.pose.bones:
                bone = armature_obj.pose.bones['root']
                bone.location = hips_pos
                bone.keyframe_insert(data_path='location', frame=frame_num + 1)
            
            # Keyframe all bones
            for pose_bone in armature_obj.pose.bones:
                pose_bone.keyframe_insert(data_path='location', frame=frame_num + 1)
                pose_bone.keyframe_insert(data_path='rotation_quaternion', frame=frame_num + 1)


# ============================================================================
# RIGIFY RETARGETER
# ============================================================================

class RigifyRetargeter:
    """Handles retargeting to Rigify control bones"""
    
    def __init__(self, source_rig, target_rig, use_ik=True):
        self.source = source_rig
        self.target = target_rig
        self.use_ik = use_ik
        
        # Bone mapping: source -> (target_control_bone, constraint_type)
        self.bone_map = {
            'root': ('torso', 'COPY_LOCATION'),
            'spine': ('chest', 'COPY_ROTATION'),
            'head': ('head', 'COPY_ROTATION'),
        }
        
        if use_ik:
            # IK mode
            self.bone_map.update({
                'hand.L': ('hand_ik.L', 'COPY_LOCATION'),
                'hand.R': ('hand_ik.R', 'COPY_LOCATION'),
                'foot.L': ('foot_ik.L', 'COPY_LOCATION'),
                'foot.R': ('foot_ik.R', 'COPY_LOCATION'),
            })
        else:
            # FK mode
            self.bone_map.update({
                'upper_arm.L': ('upper_arm_fk.L', 'COPY_ROTATION'),
                'forearm.L': ('forearm_fk.L', 'COPY_ROTATION'),
                'hand.L': ('hand_fk.L', 'COPY_ROTATION'),
                'upper_arm.R': ('upper_arm_fk.R', 'COPY_ROTATION'),
                'forearm.R': ('forearm_fk.R', 'COPY_ROTATION'),
                'hand.R': ('hand_fk.R', 'COPY_ROTATION'),
                'thigh.L': ('thigh_fk.L', 'COPY_ROTATION'),
                'shin.L': ('shin_fk.L', 'COPY_ROTATION'),
                'foot.L': ('foot_fk.L', 'COPY_ROTATION'),
                'thigh.R': ('thigh_fk.R', 'COPY_ROTATION'),
                'shin.R': ('shin_fk.R', 'COPY_ROTATION'),
                'foot.R': ('foot_fk.R', 'COPY_ROTATION'),
            })
    
    def setup_constraints(self):
        """Add constraints to target rig"""
        
        constraints_added = 0
        
        for source_bone_name, (target_bone_name, constraint_type) in self.bone_map.items():
            if source_bone_name not in self.source.pose.bones:
                continue
            if target_bone_name not in self.target.pose.bones:
                print(f"Warning: Target bone '{target_bone_name}' not found")
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
                con.target_space = 'WORLD'
                con.owner_space = 'WORLD'
                con.influence = 1.0
                constraints_added += 1
                
            elif constraint_type == 'COPY_ROTATION':
                con = target_bone.constraints.new('COPY_ROTATION')
                con.name = f'MOCAP_{source_bone_name}'
                con.target = self.source
                con.subtarget = source_bone_name
                con.target_space = 'WORLD'
                con.owner_space = 'WORLD'
                con.influence = 1.0
                constraints_added += 1
        
        return constraints_added
    
    def bake_animation(self, frame_start, frame_end):
        """Bake constraints to keyframes"""
        
        # Select target rig
        bpy.context.view_layer.objects.active = self.target
        self.target.select_set(True)
        
        # Select bones with mocap constraints
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.select_all(action='DESELECT')
        
        for bone in self.target.pose.bones:
            for con in bone.constraints:
                if con.name.startswith('MOCAP_'):
                    bone.bone.select = True
                    break
        
        # Bake
        bpy.ops.nla.bake(
            frame_start=frame_start,
            frame_end=frame_end,
            only_selected=True,
            visual_keying=True,
            clear_constraints=False,
            use_current_action=True,
            bake_types={'POSE'}
        )
        
        bpy.ops.object.mode_set(mode='OBJECT')
    
    def remove_constraints(self):
        """Remove all mocap constraints"""
        
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
    """Import mocap JSON file"""
    bl_idname = "mocap.import_json"
    bl_label = "Import Mocap JSON"
    bl_options = {'REGISTER', 'UNDO'}
    
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    
    scale: FloatProperty(
        name="Scale",
        description="Scale factor for mocap data",
        default=0.15,
        min=0.01,
        max=1.0
    )
    
    def execute(self, context):
        # Load JSON
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load JSON: {e}")
            return {'CANCELLED'}
        
        if 'frames' not in data:
            self.report({'ERROR'}, "Invalid JSON format")
            return {'CANCELLED'}
        
        frame_data = data['frames']
        
        if not frame_data:
            self.report({'ERROR'}, "No frames in file")
            return {'CANCELLED'}
        
        # Create armature
        builder = MocapArmatureBuilder(name=f"Mocap_{data.get('take_name', 'Take')}")
        armature_obj = builder.create_armature(context, frame_data, self.scale)
        
        if not armature_obj:
            self.report({'ERROR'}, "Failed to create armature")
            return {'CANCELLED'}
        
        # Animate
        builder.animate_armature(armature_obj, frame_data, self.scale, data.get('fps', 30))
        
        self.report({'INFO'}, f"Imported {len(frame_data)} frames")
        return {'FINISHED'}


class MOCAP_OT_retarget_to_rigify(Operator):
    """Retarget mocap to Rigify rig"""
    bl_idname = "mocap.retarget_to_rigify"
    bl_label = "Retarget to Rigify"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.mocap_props
        
        if not props.source_rig or not props.target_rig:
            self.report({'ERROR'}, "Select both source and target rigs")
            return {'CANCELLED'}
        
        source = bpy.data.objects.get(props.source_rig)
        target = bpy.data.objects.get(props.target_rig)
        
        if not source or not target:
            self.report({'ERROR'}, "Invalid rigs selected")
            return {'CANCELLED'}
        
        # Setup retargeter
        retargeter = RigifyRetargeter(source, target, props.use_ik)
        count = retargeter.setup_constraints()
        
        self.report({'INFO'}, f"Added {count} constraints")
        return {'FINISHED'}


class MOCAP_OT_bake_animation(Operator):
    """Bake mocap animation to keyframes"""
    bl_idname = "mocap.bake_animation"
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
        
        retargeter = RigifyRetargeter(source, target, props.use_ik)
        retargeter.bake_animation(frame_start, frame_end)
        
        self.report({'INFO'}, f"Baked frames {frame_start}-{frame_end}")
        return {'FINISHED'}


class MOCAP_OT_remove_constraints(Operator):
    """Remove mocap constraints"""
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
        return {'FINISHED'}


# ============================================================================
# UI PANEL
# ============================================================================

class MocapProperties(PropertyGroup):
    """Properties for mocap panel"""
    
    source_rig: StringProperty(
        name="Source (Mocap)",
        description="Mocap armature"
    )
    
    target_rig: StringProperty(
        name="Target (Rigify)",
        description="Rigify character rig"
    )
    
    use_ik: BoolProperty(
        name="Use IK",
        description="Use IK controls instead of FK",
        default=True
    )


class MOCAP_PT_main_panel(Panel):
    """Main mocap panel"""
    bl_label = "Mocap to Rigify"
    bl_idname = "MOCAP_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Mocap'
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.mocap_props
        
        # IMPORT
        box = layout.box()
        box.label(text="1. Import Mocap", icon='IMPORT')
        box.operator("mocap.import_json", text="Import JSON File", icon='FILE_FOLDER')
        
        # RETARGET
        box = layout.box()
        box.label(text="2. Retarget", icon='ARMATURE_DATA')
        
        # Source rig dropdown
        box.prop_search(props, "source_rig", bpy.data, "objects", text="Source")
        
        # Target rig dropdown
        box.prop_search(props, "target_rig", bpy.data, "objects", text="Target")
        
        # IK/FK toggle
        box.prop(props, "use_ik")
        
        box.operator("mocap.retarget_to_rigify", text="Setup Retargeting", icon='CON_ARMATURE')
        
        # BAKE
        box = layout.box()
        box.label(text="3. Bake", icon='RENDER_ANIMATION')
        box.operator("mocap.bake_animation", text="Bake to Keyframes", icon='KEYFRAME')
        box.operator("mocap.remove_constraints", text="Remove Constraints", icon='X')
        
        # TIPS
        box = layout.box()
        box.label(text="Tips:", icon='INFO')
        box.label(text="• Set IK/FK switches to 1.0")
        box.label(text="• Preview before baking")
        box.label(text="• Remove constraints after bake")


# ============================================================================
# REGISTRATION
# ============================================================================

classes = (
    MocapProperties,
    MOCAP_OT_import_json,
    MOCAP_OT_retarget_to_rigify,
    MOCAP_OT_bake_animation,
    MOCAP_OT_remove_constraints,
    MOCAP_PT_main_panel,
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
