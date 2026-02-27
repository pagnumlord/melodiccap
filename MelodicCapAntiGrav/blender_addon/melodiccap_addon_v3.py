bl_info = {
    "name": "MelodicCap Addon",
    "author": "Antigravity AI",
    "version": (3, 1),
    "blender": (4, 4, 0),
    "location": "View3D > Sidebar > MelodicCap",
    "description": "Professional Retargeting for Melodic Justice characters.",
    "category": "Animation",
}

import bpy
import math
from mathutils import Vector, Quaternion, Matrix

class MELODICCAP_OT_sanitize_rig(bpy.types.Operator):
    """Clean up stray constraints from previous retargeting attempts"""
    bl_idname = "melodiccap.sanitize_rig"
    bl_label = "Sanitize Rig"
    
    def execute(self, context):
        rig = context.active_object
        if not rig or rig.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the Rigify Armature")
            return {'CANCELLED'}
        
        count = 0
        for bone in rig.pose.bones:
            if bone.name.startswith("DEF-") or "_fk" in bone.name:
                for con in bone.constraints:
                    if con.type in {'COPY_TRANSFORMS', 'COPY_LOCATION', 'COPY_ROTATION'} and not con.name.startswith("RIGIFY"):
                        bone.constraints.remove(con)
                        count += 1
        
        self.report({'INFO'}, f"Cleaned {count} suspicious constraints.")
        return {'FINISHED'}

class MELODICCAP_OT_align_poses(bpy.types.Operator):
    """Align the character to a T-pose based on the selected preset"""
    bl_idname = "melodiccap.align_poses"
    bl_label = "Align Rest Pose"

    def execute(self, context):
        rig = context.active_object
        preset = context.scene.melodiccap_character_preset
        
        # In A-pose, shoulders are down ~45 deg. To reach T-pose, rotate UP.
        offsets = {
            "upper_arm_fk.L": (0, 0, 0.785),
            "upper_arm_fk.R": (0, 0, -0.785),
        }
        
        for bone_name, rot in offsets.items():
            bone = rig.pose.bones.get(bone_name)
            if bone:
                bone.rotation_mode = 'QUATERNION'
                from mathutils import Euler
                bone.rotation_quaternion = Euler(rot, 'XYZ').to_quaternion()
        
        self.report({'INFO'}, f"Aligned {preset} to T-Pose.")
        return {'FINISHED'}

class MELODICCAP_OT_retarget_v2r(bpy.types.Operator):
    """V2R Retargeting Engine: Automated FK alignment and IK syncing"""
    bl_idname = "melodiccap.retarget_v2r"
    bl_label = "MelodicCap Retarget"
    
    def execute(self, context):
        source = context.scene.melodiccap_source_rig
        target = context.active_object
        
        if not source or not target:
            self.report({'ERROR'}, "Select Target (Jax) and set Source in panel.")
            return {'CANCELLED'}

        # Mapping dictionary (Mixamo to Rigify)
        mapping = {
            "Hips": "torso",
            "Spine": "spine_fk.001",
            "Spine1": "spine_fk.002",
            "Spine2": "spine_fk.003",
            "LeftArm": "upper_arm_fk.L",
            "LeftForeArm": "forearm_fk.L",
            "LeftHand": "hand_fk.L",
            "RightArm": "upper_arm_fk.R",
            "RightForeArm": "forearm_fk.R",
            "RightHand": "hand_fk.R",
            "LeftUpLeg": "thigh_fk.L",
            "LeftLeg": "shin_fk.L",
            "LeftFoot": "foot_fk.L",
            "RightUpLeg": "thigh_fk.R",
            "RightLeg": "shin_fk.R",
            "RightFoot": "foot_fk.R",
            "Neck": "neck",
            "Head": "head"
        }

        # Fingers
        fingers = ["index", "middle", "pinky", "ring", "thumb"]
        for f in fingers:
            for i in range(1, 4):
                mapping[f"LeftHand{f.capitalize()}{i}"] = f"f_{f}.0{i}.L"
                mapping[f"RightHand{f.capitalize()}{i}"] = f"f_{f}.0{i}.R"

        # IK Sync
        ik_sync = {
            "LeftHand": "hand_ik.L", "RightHand": "hand_ik.R",
            "LeftFoot": "foot_ik.L", "RightFoot": "foot_ik.R"
        }

        # Set sliders to FK for capture
        for bone in target.pose.bones:
            if "IK_FK" in bone.keys():
                bone["IK_FK"] = 1.0

        scene = context.scene
        for frame in range(scene.frame_start, scene.frame_end + 1):
            scene.frame_set(frame)
            for s_bone_name, t_bone_name in mapping.items():
                s_bone = source.pose.bones.get(s_bone_name) or source.pose.bones.get(f"mixamorig:{s_bone_name}")
                t_bone = target.pose.bones.get(t_bone_name)
                if s_bone and t_bone:
                    t_bone.rotation_mode = 'QUATERNION'
                    t_bone.rotation_quaternion = s_bone.rotation_quaternion.copy()
                    t_bone.keyframe_insert(data_path="rotation_quaternion")

            for s_bone_name, t_bone_name in ik_sync.items():
                s_bone = source.pose.bones.get(s_bone_name) or source.pose.bones.get(f"mixamorig:{s_bone_name}")
                t_bone = target.pose.bones.get(t_bone_name)
                if s_bone and t_bone:
                    t_bone.location = target.convert_space(pose_bone=t_bone, matrix=s_bone.matrix, from_space='WORLD', to_space='POSE').to_translation()
                    t_bone.keyframe_insert(data_path="location")

        self.report({'INFO'}, "Retargeting Complete.")
        return {'FINISHED'}

class MELODICCAP_PT_main(bpy.types.Panel):
    bl_label = "MelodicCap Studio"
    bl_idname = "MELODICCAP_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'MelodicCap'

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="1. Setup")
        col.operator("melodiccap.sanitize_rig", icon='BRUSH')
        
        col.separator()
        col.label(text="2. Character")
        col.prop(bpy.context.scene, "melodiccap_character_preset")
        col.operator("melodiccap.align_poses", icon='POSE_HLT')
        
        col.separator()
        col.label(text="3. Retarget")
        col.prop(bpy.context.scene, "melodiccap_source_rig", text="Source")
        col.operator("melodiccap.retarget_v2r", icon='ANIM_DATA')

def register():
    bpy.utils.register_class(MELODICCAP_OT_sanitize_rig)
    bpy.utils.register_class(MELODICCAP_OT_align_poses)
    bpy.utils.register_class(MELODICCAP_OT_retarget_v2r)
    bpy.utils.register_class(MELODICCAP_PT_main)
    bpy.types.Scene.melodiccap_source_rig = bpy.props.PointerProperty(type=bpy.types.Object)
    bpy.types.Scene.melodiccap_character_preset = bpy.props.EnumProperty(
        items=[('JAX', 'Jax', ''), ('KIKO', 'Kiko', ''), ('KAI', 'Kai', ''), ('SHADOW', 'Shadow', '')],
        name="Preset", default='JAX'
    )

def unregister():
    bpy.utils.unregister_class(MELODICCAP_OT_sanitize_rig)
    bpy.utils.unregister_class(MELODICCAP_OT_align_poses)
    bpy.utils.unregister_class(MELODICCAP_OT_retarget_v2r)
    bpy.utils.unregister_class(MELODICCAP_PT_main)

if __name__ == "__main__":
    register()
