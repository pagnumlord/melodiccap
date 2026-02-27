bl_info = {
    "name": "MelodicCap Addon",
    "author": "Karsten Allen / Antigravity AI",
    "version": (3, 1),
    "blender": (4, 4, 0),  # Updated for Blender 4.4.3
    "location": "View3D > Sidebar > MelodicCap",
    "description": "Professional Retargeting for Melodic Justice characters.",
    "category": "Animation",
}

import bpy
import math
from mathutils import Vector, Quaternion, Matrix, Euler

class MELODICCAP_OT_sanitize_rig(bpy.types.Operator):
    """Clean up stray constraints from previous retargeting attempts"""
    bl_idname = "melodiccap.sanitize_rig"
    bl_label = "Sanitize Rig"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        rig = context.active_object
        if not rig or rig.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the Rigify Armature")
            return {'CANCELLED'}
        
        count = 0
        for bone in rig.pose.bones:
            # Clean DEF bones and FK controls
            if bone.name.startswith("DEF-") or "_fk" in bone.name:
                for con in list(bone.constraints):
                    # Don't remove built-in Rigify constraints
                    if con.type in {'COPY_TRANSFORMS', 'COPY_LOCATION', 'COPY_ROTATION'}:
                        if not con.name.startswith("RIGIFY"):
                            bone.constraints.remove(con)
                            count += 1
        
        self.report({'INFO'}, f"Cleaned {count} constraints.")
        return {'FINISHED'}


class MELODICCAP_OT_align_poses(bpy.types.Operator):
    """Align the character to a T-pose based on the selected preset"""
    bl_idname = "melodiccap.align_poses"
    bl_label = "Align Rest Pose"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rig = context.active_object
        if not rig or rig.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the Rigify Armature")
            return {'CANCELLED'}
        
        preset = context.scene.melodiccap_character_preset
        
        # Rotation offsets to go from A-pose to T-pose
        # These are Euler angles (X, Y, Z) in radians
        offsets = {
            "upper_arm_fk.L": (0, 0, 0.785),   # +45 degrees in Z
            "upper_arm_fk.R": (0, 0, -0.785),  # -45 degrees in Z
        }
        
        bpy.ops.object.mode_set(mode='POSE')
        
        for bone_name, euler_rot in offsets.items():
            bone = rig.pose.bones.get(bone_name)
            if bone:
                bone.rotation_mode = 'QUATERNION'
                # FIXED: Create Euler from tuple, then convert to Quaternion
                euler = Euler(euler_rot, 'XYZ')
                bone.rotation_quaternion = euler.to_quaternion()
        
        self.report({'INFO'}, f"Aligned {preset} to T-Pose.")
        return {'FINISHED'}


class MELODICCAP_OT_retarget_v2r(bpy.types.Operator):
    """V2R Retargeting Engine: Automated FK alignment and IK syncing"""
    bl_idname = "melodiccap.retarget_v2r"
    bl_label = "MelodicCap Retarget"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        source = context.scene.melodiccap_source_rig
        target = context.active_object
        
        if not source or not target:
            self.report({'ERROR'}, "Select Target (active) and set Source in panel.")
            return {'CANCELLED'}
        
        if source.type != 'ARMATURE' or target.type != 'ARMATURE':
            self.report({'ERROR'}, "Both source and target must be armatures.")
            return {'CANCELLED'}

        # Proven bone mapping (Mixamo to Rigify)
        # From successful Rokoko testing - CGDive method
        mapping = {
            # Torso
            "Hips": "torso",
            "Spine": "spine_fk.001",
            "Spine1": "spine_fk.002",
            "Spine2": "spine_fk.003",
            
            # Head
            "Neck": "neck",
            "Head": "head",
            
            # Left Arm
            "LeftShoulder": "shoulder.L",
            "LeftArm": "upper_arm_fk.L",
            "LeftForeArm": "forearm_fk.L",
            "LeftHand": "hand_fk.L",
            
            # Right Arm
            "RightShoulder": "shoulder.R",
            "RightArm": "upper_arm_fk.R",
            "RightForeArm": "forearm_fk.R",
            "RightHand": "hand_fk.R",
            
            # Left Leg
            "LeftUpLeg": "thigh_fk.L",
            "LeftLeg": "shin_fk.L",
            "LeftFoot": "foot_fk.L",
            "LeftToeBase": "toe_fk.L",
            
            # Right Leg
            "RightUpLeg": "thigh_fk.R",
            "RightLeg": "shin_fk.R",
            "RightFoot": "foot_fk.R",
            "RightToeBase": "toe_fk.R",
        }

        # Finger mapping
        finger_names = ["Index", "Middle", "Pinky", "Ring", "Thumb"]
        for f in finger_names:
            for i in range(1, 4):
                src_l = f"LeftHand{f}{i}"
                src_r = f"RightHand{f}{i}"
                
                if f == "Thumb":
                    tgt_l = f"thumb.0{i}.L"
                    tgt_r = f"thumb.0{i}.R"
                else:
                    tgt_l = f"f_{f.lower()}.0{i}.L"
                    tgt_r = f"f_{f.lower()}.0{i}.R"
                
                mapping[src_l] = tgt_l
                mapping[src_r] = tgt_r

        # IK targets (CGDive method - position-based)
        ik_sync = {
            "LeftHand": "hand_ik.L",
            "RightHand": "hand_ik.R",
            "LeftFoot": "foot_ik.L",
            "RightFoot": "foot_ik.R",
        }

        # Set IK/FK sliders to FK mode
        bpy.ops.object.mode_set(mode='POSE')
        for bone in target.pose.bones:
            if "IK_FK" in bone.keys():
                bone["IK_FK"] = 1.0  # FK mode

        scene = context.scene
        frame_start = scene.frame_start
        frame_end = scene.frame_end
        
        for frame in range(frame_start, frame_end + 1):
            scene.frame_set(frame)
            
            # FK rotation mapping
            for src_name, tgt_name in mapping.items():
                # Try with and without mixamorig prefix
                s_bone = source.pose.bones.get(src_name)
                if not s_bone:
                    s_bone = source.pose.bones.get(f"mixamorig:{src_name}")
                
                t_bone = target.pose.bones.get(tgt_name)
                
                if s_bone and t_bone:
                    t_bone.rotation_mode = 'QUATERNION'
                    t_bone.rotation_quaternion = s_bone.rotation_quaternion.copy()
                    t_bone.keyframe_insert(data_path="rotation_quaternion")

            # IK position sync (for extremities)
            for src_name, tgt_name in ik_sync.items():
                s_bone = source.pose.bones.get(src_name)
                if not s_bone:
                    s_bone = source.pose.bones.get(f"mixamorig:{src_name}")
                
                t_bone = target.pose.bones.get(tgt_name)
                
                if s_bone and t_bone:
                    # Get world position of source bone
                    src_world = source.matrix_world @ s_bone.matrix @ Vector((0, 0, 0))
                    
                    # Convert to target's local space
                    tgt_local = target.matrix_world.inverted() @ src_world
                    
                    # Apply with scale factor if needed
                    # (assumes source and target are similar scale)
                    t_bone.location = tgt_local
                    t_bone.keyframe_insert(data_path="location")

            # Root motion (hips translation)
            hips_src = source.pose.bones.get("Hips") or source.pose.bones.get("mixamorig:Hips")
            hips_tgt = target.pose.bones.get("torso")
            
            if hips_src and hips_tgt:
                src_world = source.matrix_world @ hips_src.matrix.translation
                hips_tgt.location = target.matrix_world.inverted() @ src_world
                hips_tgt.keyframe_insert(data_path="location")

        self.report({'INFO'}, f"Retargeted {frame_end - frame_start + 1} frames.")
        return {'FINISHED'}


class MELODICCAP_PT_main(bpy.types.Panel):
    bl_label = "MelodicCap Studio"
    bl_idname = "MELODICCAP_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'MelodicCap'

    def draw(self, context):
        # FIXED: Correct signature and layout access
        layout = self.layout
        
        col = layout.column(align=True)
        col.label(text="1. Setup", icon='PREFERENCES')
        col.operator("melodiccap.sanitize_rig", icon='BRUSH_DATA')
        
        col.separator()
        col.label(text="2. Character", icon='ARMATURE_DATA')
        col.prop(context.scene, "melodiccap_character_preset")
        col.operator("melodiccap.align_poses", icon='POSE_HLT')
        
        col.separator()
        col.label(text="3. Retarget", icon='ACTION')
        col.prop(context.scene, "melodiccap_source_rig", text="Source")
        
        # Show frame range
        row = col.row(align=True)
        row.prop(context.scene, "frame_start", text="Start")
        row.prop(context.scene, "frame_end", text="End")
        
        col.operator("melodiccap.retarget_v2r", icon='ANIM_DATA', text="Retarget Animation")
        
        # Settings box
        box = layout.box()
        box.label(text="Settings", icon='SETTINGS')
        
        # Show current IK/FK state if armature selected
        if context.active_object and context.active_object.type == 'ARMATURE':
            for bone_name in ["upper_arm_parent.L", "upper_arm_parent.R", 
                             "thigh_parent.L", "thigh_parent.R"]:
                bone = context.active_object.pose.bones.get(bone_name)
                if bone and "IK_FK" in bone.keys():
                    row = box.row()
                    row.label(text=bone_name.replace("_parent", ""))
                    row.prop(bone, '["IK_FK"]', text="IK/FK", slider=True)


def register():
    bpy.utils.register_class(MELODICCAP_OT_sanitize_rig)
    bpy.utils.register_class(MELODICCAP_OT_align_poses)
    bpy.utils.register_class(MELODICCAP_OT_retarget_v2r)
    bpy.utils.register_class(MELODICCAP_PT_main)
    
    bpy.types.Scene.melodiccap_source_rig = bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Source Rig",
        description="Source armature with animation (e.g., Mixamo import)",
        poll=lambda self, obj: obj.type == 'ARMATURE'
    )
    
    bpy.types.Scene.melodiccap_character_preset = bpy.props.EnumProperty(
        items=[
            ('JAX', 'Jax', 'Main character'),
            ('KIKO', 'Kiko', ''),
            ('KAI', 'Kai', ''),
            ('SHADOW', 'Shadow', ''),
            ('DR_WHITE', 'Dr. White', ''),
            ('HIRO', 'Hiro', ''),
        ],
        name="Character",
        default='JAX'
    )


def unregister():
    bpy.utils.unregister_class(MELODICCAP_OT_sanitize_rig)
    bpy.utils.unregister_class(MELODICCAP_OT_align_poses)
    bpy.utils.unregister_class(MELODICCAP_OT_retarget_v2r)
    bpy.utils.unregister_class(MELODICCAP_PT_main)
    
    del bpy.types.Scene.melodiccap_source_rig
    del bpy.types.Scene.melodiccap_character_preset


if __name__ == "__main__":
    register()
