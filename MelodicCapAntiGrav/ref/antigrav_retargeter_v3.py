bl_info = {
    "name": "AntiGrav V3 Retargeter",
    "author": "Karsten Allen / Antigravity AI",
    "version": (3, 1),
    "blender": (4, 4, 0),  # Updated for Blender 4.4.3
    "location": "View3D > Sidebar > AntiGrav",
    "description": "Scientific Mocap Retargeter for Rigify using Vector-to-Rotation math with Smart Pinning.",
    "category": "Animation",
}

import bpy
import json
import os
import math
from mathutils import Vector, Quaternion, Matrix, Euler
from bpy_extras.io_utils import ImportHelper


class ANTIGRAV_OT_create_prop_empty(bpy.types.Operator):
    """Creates a tracking empty at the selection for prop alignment"""
    bl_idname = "antigrav.create_prop_empty"
    bl_label = "Create Prop Empty"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rig = context.object
        if not rig or rig.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the Armature first.")
            return {'CANCELLED'}
        
        # Create Empty
        bpy.ops.object.mode_set(mode='OBJECT')
        empty = bpy.data.objects.new("Prop_Anchor", None)
        empty.empty_display_type = 'CUBE'
        empty.empty_display_size = 0.1
        context.collection.objects.link(empty)
        
        # Parent to selection (usually a hand bone)
        active_bone = context.active_pose_bone
        if active_bone:
            empty.parent = rig
            empty.parent_type = 'BONE'
            empty.parent_bone = active_bone.name
            empty.location = (0, 0, 0)
            self.report({'INFO'}, f"Prop Empty parented to {active_bone.name}")
        else:
            self.report({'WARNING'}, "No bone selected, created world-space empty.")
            
        return {'FINISHED'}


class ANTIGRAV_OT_sanitize_rig(bpy.types.Operator):
    """Remove previous retargeting constraints from DEF/FK bones"""
    bl_idname = "antigrav.sanitize_rig"
    bl_label = "Sanitize Rig"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        rig = context.active_object
        if not rig or rig.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the Rigify Armature")
            return {'CANCELLED'}
        
        count = 0
        for bone in rig.pose.bones:
            if bone.name.startswith("DEF-") or "_fk" in bone.name:
                for con in list(bone.constraints):
                    if con.type in {'COPY_TRANSFORMS', 'COPY_LOCATION', 'COPY_ROTATION'}:
                        if not con.name.startswith("RIGIFY"):
                            bone.constraints.remove(con)
                            count += 1
        
        self.report({'INFO'}, f"Cleaned {count} suspicious constraints.")
        return {'FINISHED'}


class ANTIGRAV_OT_align_poses(bpy.types.Operator):
    """Align the character's rest pose based on the selected cast member"""
    bl_idname = "antigrav.align_poses"
    bl_label = "Align Poses"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rig = context.active_object
        if not rig or rig.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the Rigify Armature")
            return {'CANCELLED'}
        
        preset = context.scene.antigrav_character_preset
        
        # Rotation offsets (Euler XYZ in radians)
        offsets = {
            "upper_arm_fk.L": (0, 0, 0.785),
            "upper_arm_fk.R": (0, 0, -0.785),
        }
        
        bpy.ops.object.mode_set(mode='POSE')
        
        for bone_name, euler_rot in offsets.items():
            bone = rig.pose.bones.get(bone_name)
            if bone:
                bone.rotation_mode = 'QUATERNION'
                # FIXED: Proper Euler to Quaternion conversion
                euler = Euler(euler_rot, 'XYZ')
                bone.rotation_quaternion = euler.to_quaternion()
        
        self.report({'INFO'}, f"Aligned {preset} to T-Pose.")
        return {'FINISHED'}


class ANTIGRAV_OT_import_v2r(bpy.types.Operator, ImportHelper):
    """Import scientific 3D JSON and apply V2R retargeting with Smart Pinning"""
    bl_idname = "antigrav.import_v2r"
    bl_label = "Import Scientific Mocap (.json)"
    bl_options = {'REGISTER', 'UNDO'}
    
    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(default="*.json", options={'HIDDEN'})

    def execute(self, context):
        rig = context.active_object
        if not rig or rig.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the Jax Rigify Armature")
            return {'CANCELLED'}

        with open(self.filepath, 'r') as f:
            data = json.load(f)

        frames = data.get('frames', [])
        if not frames:
            self.report({'ERROR'}, "No frames found in JSON")
            return {'CANCELLED'}

        # Vector-to-Rotation mapping
        # Maps bone name to (start_landmark_idx, end_landmark_idx)
        v2r_map = {
            # Arms
            "upper_arm_fk.L": (11, 13),
            "forearm_fk.L": (13, 15),
            "upper_arm_fk.R": (12, 14),
            "forearm_fk.R": (14, 16),
            
            # Legs
            "thigh_fk.L": (23, 25),
            "shin_fk.L": (25, 27),
            "thigh_fk.R": (24, 26),
            "shin_fk.R": (26, 28),
            
            # Spine (uses virtual midpoints)
            "spine_fk": ("hip_mid", "spine_low"),
            "spine_fk.001": ("spine_low", "spine_mid"),
            "spine_fk.002": ("spine_mid", "neck_mid"),
            "spine_fk.003": ("neck_mid", "shoulder_mid"),
        }
        
        # Rest direction for each bone (local Y axis in rest pose)
        # Rigify bones generally point along local Y
        bone_rest_axes = {
            "upper_arm_fk.L": Vector((1, 0, 0)),    # Points outward
            "upper_arm_fk.R": Vector((-1, 0, 0)),   # Points outward
            "forearm_fk.L": Vector((1, 0, 0)),
            "forearm_fk.R": Vector((-1, 0, 0)),
            "thigh_fk.L": Vector((0, 0, -1)),       # Points down
            "thigh_fk.R": Vector((0, 0, -1)),
            "shin_fk.L": Vector((0, 0, -1)),
            "shin_fk.R": Vector((0, 0, -1)),
            "spine_fk": Vector((0, 0, 1)),          # Points up
            "spine_fk.001": Vector((0, 0, 1)),
            "spine_fk.002": Vector((0, 0, 1)),
            "spine_fk.003": Vector((0, 0, 1)),
        }

        # Pinning state (for foot/hand locking)
        prev_pos = {}
        pin_threshold = context.scene.antigrav_pin_threshold

        bpy.ops.object.mode_set(mode='POSE')
        
        # Set FK mode on all IK/FK switches
        for bone in rig.pose.bones:
            if "IK_FK" in bone.keys():
                bone["IK_FK"] = 1.0

        for frame_data in frames:
            timestamp = frame_data.get('timestamp', 0)
            f_idx = int(timestamp * 30)
            context.scene.frame_set(f_idx)
            
            lms = frame_data.get('landmarks_3d', {})
            if not lms:
                continue
            
            # Calculate virtual spine midpoints
            if "11" in lms and "12" in lms and "23" in lms and "24" in lms:
                v11 = Vector(lms["11"])
                v12 = Vector(lms["12"])
                v23 = Vector(lms["23"])
                v24 = Vector(lms["24"])
                
                hip_mid = (v23 + v24) / 2
                sh_mid = (v11 + v12) / 2
                spine_mid = (hip_mid + sh_mid) / 2
                spine_low = (hip_mid + spine_mid) / 2
                neck_mid = (spine_mid + sh_mid) / 2
                
                lms["hip_mid"] = hip_mid
                lms["spine_low"] = spine_low
                lms["spine_mid"] = spine_mid
                lms["neck_mid"] = neck_mid
                lms["shoulder_mid"] = sh_mid

            # 1. Root translation (torso)
            hips = rig.pose.bones.get("torso")
            if hips and "hip_mid" in lms:
                target_pos = lms["hip_mid"]
                if isinstance(target_pos, Vector):
                    hips.location = rig.matrix_world.inverted() @ target_pos
                else:
                    hips.location = rig.matrix_world.inverted() @ Vector(target_pos)
                hips.keyframe_insert(data_path="location")

            # 2. FK rotations from V2R mapping
            for bone_name, (s, e) in v2r_map.items():
                bone = rig.pose.bones.get(bone_name)
                if not bone:
                    continue
                
                # Get start/end positions
                s_key = str(s) if isinstance(s, int) else s
                e_key = str(e) if isinstance(e, int) else e
                
                if s_key not in lms or e_key not in lms:
                    continue
                
                v_start = lms[s_key]
                v_end = lms[e_key]
                
                if not isinstance(v_start, Vector):
                    v_start = Vector(v_start)
                if not isinstance(v_end, Vector):
                    v_end = Vector(v_end)
                
                target_dir = (v_end - v_start).normalized()
                
                # Get rest axis for this bone
                rest_axis = bone_rest_axes.get(bone_name, Vector((0, 1, 0)))
                
                # Convert target direction to armature local space
                target_dir_local = rig.matrix_world.inverted().to_quaternion() @ target_dir
                
                bone.rotation_mode = 'QUATERNION'
                quat = rest_axis.rotation_difference(target_dir_local)
                bone.rotation_quaternion = quat
                bone.keyframe_insert(data_path="rotation_quaternion")

            # 3. IK targets with Smart Pinning
            for side in [".L", ".R"]:
                # FOOT IK
                foot_ik = rig.pose.bones.get(f"foot_ik{side}")
                if foot_ik:
                    idx = "27" if side == ".L" else "28"
                    if idx in lms:
                        pos = Vector(lms[idx]) if not isinstance(lms[idx], Vector) else lms[idx]
                        
                        # Ground clamp
                        if pos.z < 0:
                            pos.z = 0
                        
                        # Smart pinning
                        pin_key = f"foot{side}"
                        if pin_key in prev_pos:
                            dist = (pos - prev_pos[pin_key]).length
                            if dist < pin_threshold:
                                pos = prev_pos[pin_key].copy()
                        
                        prev_pos[pin_key] = pos.copy()
                        foot_ik.location = rig.matrix_world.inverted() @ pos
                        
                        # Foot rotation from shin direction
                        shin_idx = "25" if side == ".L" else "26"
                        if shin_idx in lms:
                            shin_pos = lms[shin_idx]
                            if not isinstance(shin_pos, Vector):
                                shin_pos = Vector(shin_pos)
                            vec_shin = (pos - shin_pos).normalized()
                            foot_ik.rotation_mode = 'QUATERNION'
                            local_vec = rig.matrix_world.inverted().to_quaternion() @ vec_shin
                            foot_ik.rotation_quaternion = Vector((0, 0, -1)).rotation_difference(local_vec)

                        foot_ik.keyframe_insert(data_path="location")
                        foot_ik.keyframe_insert(data_path="rotation_quaternion")

                # HAND IK
                hand_ik = rig.pose.bones.get(f"hand_ik{side}")
                if hand_ik:
                    idx = "15" if side == ".L" else "16"
                    if idx in lms:
                        pos = Vector(lms[idx]) if not isinstance(lms[idx], Vector) else lms[idx]
                        
                        # Smart pinning
                        pin_key = f"hand{side}"
                        if pin_key in prev_pos:
                            dist = (pos - prev_pos[pin_key]).length
                            if dist < pin_threshold:
                                pos = prev_pos[pin_key].copy()
                        
                        prev_pos[pin_key] = pos.copy()
                        hand_ik.location = rig.matrix_world.inverted() @ pos

                        # Hand rotation from forearm direction
                        elbow_idx = "13" if side == ".L" else "14"
                        if elbow_idx in lms:
                            elbow_pos = lms[elbow_idx]
                            if not isinstance(elbow_pos, Vector):
                                elbow_pos = Vector(elbow_pos)
                            vec_arm = (pos - elbow_pos).normalized()
                            hand_ik.rotation_mode = 'QUATERNION'
                            local_vec = rig.matrix_world.inverted().to_quaternion() @ vec_arm
                            hand_ik.rotation_quaternion = Vector((0, 1, 0)).rotation_difference(local_vec)

                        hand_ik.keyframe_insert(data_path="location")
                        hand_ik.keyframe_insert(data_path="rotation_quaternion")

                # POLE TARGETS (elbow/knee orientation)
                p_side = "L" if side == ".L" else "R"
                elbow_pole = rig.pose.bones.get(f"upper_arm_ik_target.{p_side}")
                if elbow_pole:
                    wrist_idx = "15" if side == ".L" else "16"
                    elbow_idx = "13" if side == ".L" else "14"
                    sh_idx = "11" if side == ".L" else "12"
                    
                    if all(str(i) in lms for i in [wrist_idx, elbow_idx, sh_idx]):
                        v_sh = Vector(lms[str(sh_idx)]) if not isinstance(lms[str(sh_idx)], Vector) else lms[str(sh_idx)]
                        v_elbow = Vector(lms[str(elbow_idx)]) if not isinstance(lms[str(elbow_idx)], Vector) else lms[str(elbow_idx)]
                        v_wrist = Vector(lms[str(wrist_idx)]) if not isinstance(lms[str(wrist_idx)], Vector) else lms[str(wrist_idx)]
                        
                        line = (v_wrist - v_sh).normalized()
                        proj = v_sh + line * (v_elbow - v_sh).dot(line)
                        pole_vec = (v_elbow - proj).normalized() * 0.5
                        elbow_pole.location = rig.matrix_world.inverted() @ (v_elbow + pole_vec)
                        elbow_pole.keyframe_insert(data_path="location")

            # 4. FINGER FIDELITY (21 points per hand)
            hands_3d = frame_data.get('hands_3d', [])
            for h_idx, hand_lms in enumerate(hands_3d):
                h_side = ".L" if h_idx == 0 else ".R"
                
                finger_map = {
                    "thumb": [1, 2, 3, 4],
                    "index": [5, 6, 7, 8],
                    "middle": [9, 10, 11, 12],
                    "ring": [13, 14, 15, 16],
                    "pinky": [17, 18, 19, 20]
                }
                
                for f_name, mp_indices in finger_map.items():
                    for i, (s_idx, e_idx) in enumerate(zip(mp_indices[:-1], mp_indices[1:])):
                        if f_name == "thumb":
                            bone_name = f"thumb.0{i+1}{h_side}"
                        else:
                            bone_name = f"f_{f_name}.0{i+1}{h_side}"
                        
                        bone = rig.pose.bones.get(bone_name)
                        if bone and s_idx < len(hand_lms) and e_idx < len(hand_lms):
                            v_start = Vector(hand_lms[s_idx])
                            v_end = Vector(hand_lms[e_idx])
                            target_dir = (v_end - v_start).normalized()
                            target_dir_local = rig.matrix_world.inverted().to_quaternion() @ target_dir
                            
                            bone.rotation_mode = 'QUATERNION'
                            bone.rotation_quaternion = Vector((0, 1, 0)).rotation_difference(target_dir_local)
                            bone.keyframe_insert(data_path="rotation_quaternion")

        self.report({'INFO'}, f"Imported {len(frames)} frames with Smart Pinning.")
        return {'FINISHED'}


class ANTIGRAV_PT_main_panel(bpy.types.Panel):
    bl_label = "AntiGrav V3"
    bl_idname = "ANTIGRAV_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'AntiGrav'

    def draw(self, context):
        # FIXED: Correct signature
        layout = self.layout
        
        col = layout.column(align=True)
        col.label(text="Step 1: Preparation", icon='PREFERENCES')
        col.operator("antigrav.sanitize_rig", icon='BRUSH_DATA')
        
        col.separator()
        col.label(text="Step 2: Cast Member", icon='ARMATURE_DATA')
        col.prop(context.scene, "antigrav_character_preset")
        col.operator("antigrav.align_poses", icon='POSE_HLT')
        
        col.separator()
        col.label(text="Step 3: Pro Settings", icon='SETTINGS')
        col.prop(context.scene, "antigrav_pin_threshold", text="Pin Sensitivity")
        
        col.separator()
        col.label(text="Step 4: Load Take", icon='ACTION')
        col.operator("antigrav.import_v2r", icon='IMPORT', text="Import Scientific JSON")
        
        col.separator()
        col.label(text="Step 5: Props", icon='EMPTY_AXIS')
        col.operator("antigrav.create_prop_empty", text="Create Prop Anchor")


def register():
    bpy.utils.register_class(ANTIGRAV_OT_sanitize_rig)
    bpy.utils.register_class(ANTIGRAV_OT_align_poses)
    bpy.utils.register_class(ANTIGRAV_OT_import_v2r)
    bpy.utils.register_class(ANTIGRAV_OT_create_prop_empty)
    bpy.utils.register_class(ANTIGRAV_PT_main_panel)
    
    bpy.types.Scene.antigrav_character_preset = bpy.props.EnumProperty(
        items=[
            ('JAX', 'Jax', 'Main character'),
            ('KIKO', 'Kiko', ''),
            ('KAI', 'Kai', ''),
            ('DR_WHITE', 'Dr. White', ''),
            ('HIRO', 'Hiro', ''),
            ('SHADOW', 'Shadow', '')
        ],
        name="Preset",
        default='JAX'
    )
    
    bpy.types.Scene.antigrav_pin_threshold = bpy.props.FloatProperty(
        name="Pin Threshold",
        default=0.02,
        min=0.0,
        max=0.2,
        description="Higher = Stickier Feet/Hands (velocity threshold in m/frame)"
    )


def unregister():
    bpy.utils.unregister_class(ANTIGRAV_OT_sanitize_rig)
    bpy.utils.unregister_class(ANTIGRAV_OT_align_poses)
    bpy.utils.unregister_class(ANTIGRAV_OT_import_v2r)
    bpy.utils.unregister_class(ANTIGRAV_OT_create_prop_empty)
    bpy.utils.unregister_class(ANTIGRAV_PT_main_panel)
    
    del bpy.types.Scene.antigrav_character_preset
    del bpy.types.Scene.antigrav_pin_threshold


if __name__ == "__main__":
    register()
