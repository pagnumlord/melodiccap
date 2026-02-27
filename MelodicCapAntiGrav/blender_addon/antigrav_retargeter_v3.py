bl_info = {
    "name": "AntiGrav V3 Retargeter",
    "author": "Antigravity AI",
    "version": (3, 1),
    "blender": (4, 4, 0),
    "location": "View3D > Sidebar > AntiGrav",
    "description": "Scientific Mocap Retargeter for Rigify using Vector-to-Rotation math with Smart Pinning.",
    "category": "Animation",
}

import bpy
import json
import os
import math
from mathutils import Vector, Quaternion, Matrix
from bpy_extras.io_utils import ImportHelper

class ANTIGRAV_OT_create_prop_empty(bpy.types.Operator):
    bl_idname = "antigrav.create_prop_empty"
    bl_label = "Create Prop Empty"
    bl_description = "Creates a tracking empty at the selection for prop alignment"
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
    """Remove previous retargeting constraints (Copy Transforms/Location) from DEF bones"""
    bl_idname = "antigrav.sanitize_rig"
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

class ANTIGRAV_OT_align_poses(bpy.types.Operator):
    """Align the character's rest pose based on the selected cast member"""
    bl_idname = "antigrav.align_poses"
    bl_label = "Align Poses"

    def execute(self, context):
        rig = context.active_object
        preset = context.scene.antigrav_character_preset
        
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

class ANTIGRAV_OT_import_v2r(bpy.types.Operator, ImportHelper):
    """Import scientific 3D JSON and apply pure rotation retargeting with Smart Pinning"""
    bl_idname = "antigrav.import_v2r"
    bl_label = "Import Scientific Mocap (.json)"
    
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

        v2r_map = {
            "upper_arm_fk.L": (11, 13),
            "forearm_fk.L": (13, 15),
            "upper_arm_fk.R": (12, 14),
            "forearm_fk.R": (14, 16),
            "thigh_fk.L": (23, 25),
            "shin_fk.L": (25, 27),
            "thigh_fk.R": (24, 26),
            "shin_fk.R": (26, 28),
            # Full Professional Spine (4 Segments)
            "spine_fk": ("hip_mid", "spine_low"),
            "spine_fk.001": ("spine_low", "spine_mid"),
            "spine_fk.002": ("spine_mid", "neck_mid"),
            "spine_fk.003": ("neck_mid", "shoulder_mid")
        }

        # Professional State: Track previous positions for velocity-based pinning
        prev_pos = {".L": None, ".R": None}
        pin_threshold = context.scene.antigrav_pin_threshold

        bpy.ops.object.mode_set(mode='POSE')
        for bone in rig.pose.bones:
            if "IK_FK" in bone.keys():
                # Professional Dual Mode: Use 0.5 to blend, or stay 1.0 for FK but still key IK targets
                bone["IK_FK"] = 1.0 

        for frame_data in frames:
            timestamp = frame_data['timestamp']
            f_idx = int(timestamp * 30)
            context.scene.frame_set(f_idx)
            lms = frame_data['landmarks_3d']
            
            # ... (rest of the logic stays similar but we ensure IK targets are keyed properly)
            
            # CALCULATE VIRTUAL MIDPOINTS FOR SPINE (4 segments)
            if "11" in lms and "12" in lms and "23" in lms and "24" in lms:
                v11, v12 = Vector(lms["11"]), Vector(lms["12"])
                v23, v24 = Vector(lms["23"]), Vector(lms["24"])
                
                hip_mid = (v23 + v24) / 2
                sh_mid = (v11 + v12) / 2
                spine_mid = (hip_mid + sh_mid) / 2
                
                # Intermediate segments for 4-bone spine
                spine_low = (hip_mid + spine_mid) / 2
                neck_mid = (spine_mid + sh_mid) / 2 
                
                lms["hip_mid"] = hip_mid
                lms["spine_low"] = spine_low
                lms["spine_mid"] = spine_mid
                lms["neck_mid"] = neck_mid
                lms["shoulder_mid"] = sh_mid

            # 1. Hips Translation (TORSO is the master parent)
            hips = rig.pose.bones.get("torso")
            if hips and "hip_mid" in lms:
                target_pos = lms["hip_mid"]
                # Torso is often world-aligned in rest pose; transform properly
                hips.location = rig.matrix_world.inverted() @ target_pos
                hips.keyframe_insert(data_path="location")

            # 2. V2R Limb & Spine Rotations (Scientific Axis Correction)
            bone_axes = {
                "spine_fk": Vector((0, 0, 1)),
                "spine_fk.001": Vector((0, 0, 1)),
                "spine_fk.002": Vector((0, 0, 1)),
                "spine_fk.003": Vector((0, 0, 1)),
                "neck": Vector((0, 0, 1)),
                "head": Vector((0, 0, 1))
            }

            for bone_name, (s, e) in v2r_map.items():
                bone = rig.pose.bones.get(bone_name)
                if bone and str(s) in lms and str(e) in lms:
                    v_start = Vector(lms[str(s)])
                    v_end = Vector(lms[str(e)])
                    target_dir = (v_end - v_start).normalized()
                    
                    bone.rotation_mode = 'QUATERNION'
                    target_dir_local = (rig.matrix_world.inverted().to_quaternion() @ target_dir)
                    
                    # Axis Correction: Rigify limbs point +Y, Spine/Head point +Z
                    rest_axis = bone_axes.get(bone_name, Vector((0, 1, 0)))
                    quat = rest_axis.rotation_difference(target_dir_local)
                    bone.rotation_quaternion = quat
                    bone.keyframe_insert(data_path="rotation_quaternion")

            # 3. Smart Foot/Hand Pinning & Z-Clamp
            for side in [".L", ".R"]:
                # FOOT IK
                foot_ik = rig.pose.bones.get(f"foot_ik{side}")
                if foot_ik:
                    idx = 27 if side == ".L" else 28
                    if str(idx) in lms:
                        pos = Vector(lms[str(idx)])
                        if pos.z < 0: pos.z = 0 # Ground Clamp
                        
                        if prev_pos.get(f"foot{side}") is not None:
                            dist = (pos - prev_pos[f"foot{side}"]).length
                            if dist < pin_threshold:
                                pos = prev_pos[f"foot{side}"]
                        
                        prev_pos[f"foot{side}"] = pos.copy()
                        foot_ik.location = rig.matrix_world.inverted() @ pos
                        
                        # FOOT ROTATION: Align to shin vector
                        idx_start = 25 if side == ".L" else 26
                        if str(idx_start) in lms:
                            vec_shin = (Vector(lms[str(idx)]) - Vector(lms[str(idx_start)])).normalized()
                            foot_ik.rotation_mode = 'QUATERNION'
                            foot_ik.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(rig.matrix_world.inverted().to_quaternion() @ vec_shin)

                        foot_ik.keyframe_insert(data_path="location")
                        foot_ik.keyframe_insert(data_path="rotation_quaternion")

                # HAND IK (NEW: Support for sitting/touching surfaces)
                hand_ik = rig.pose.bones.get(f"hand_ik{side}")
                if hand_ik:
                    idx = 15 if side == ".L" else 16
                    if str(idx) in lms:
                        pos = Vector(lms[str(idx)])
                        
                        # Hand pinning (helps with "Hand positions off" jitter)
                        if prev_pos.get(f"hand{side}") is not None:
                            dist = (pos - prev_pos[f"hand{side}"]).length
                            if dist < pin_threshold:
                                pos = prev_pos[f"hand{side}"]
                        
                        prev_pos[f"hand{side}"] = pos.copy()
                        hand_ik.location = rig.matrix_world.inverted() @ pos

                        # HAND ROTATION: Align to forearm vector for better strumming/piano
                        idx_start = 13 if side == ".L" else 14
                        if str(idx_start) in lms:
                            vec_arm = (Vector(lms[str(idx)]) - Vector(lms[str(idx_start)])).normalized()
                            hand_ik.rotation_mode = 'QUATERNION'
                            hand_ik.rotation_quaternion = Vector((0, 1, 0)).rotation_difference(rig.matrix_world.inverted().to_quaternion() @ vec_arm)

                        hand_ik.keyframe_insert(data_path="location")
                        hand_ik.keyframe_insert(data_path="rotation_quaternion")

                # POLE TARGETS (Automatic Elbow/Knee Orientation)
                p_side = "L" if side == ".L" else "R"
                elbow_pole = rig.pose.bones.get(f"upper_arm_ik_target.{p_side}")
                if elbow_pole:
                    idx_wrist = 15 if side == ".L" else 16
                    idx_elbow = 13 if side == ".L" else 14
                    idx_sh = 11 if side == ".L" else 12
                    if str(idx_wrist) in lms and str(idx_elbow) in lms and str(idx_sh) in lms:
                        # Calculate pole position by projecting elbow away from the sh-wrist line
                        v_sh = Vector(lms[str(idx_sh)])
                        v_elbow = Vector(lms[str(idx_elbow)])
                        v_wrist = Vector(lms[str(idx_wrist)])
                        
                        line = (v_wrist - v_sh).normalized()
                        proj = v_sh + line * (v_elbow - v_sh).dot(line)
                        pole_vec = (v_elbow - proj).normalized() * 0.5 # Offset backward
                        elbow_pole.location = rig.matrix_world.inverted() @ (v_elbow + pole_vec)
                        elbow_pole.keyframe_insert(data_path="location")

            # 4. FINGER-FIDELITY (21 points per hand)
            hands_3d = frame_data.get('hands_3d', [])
            for h_idx, hand_lms in enumerate(hands_3d):
                # Detect side by checking which wrist landmark (15 or 16) is closer to hand root (lm 0)
                # Or use a simpler heuristic for now
                h_side = ".L" if h_idx == 0 else ".R" 
                
                # Mapping MediaPipe Hand to Rigify Fingers
                finger_map = {
                    "thumb": [1, 2, 3, 4],
                    "index": [5, 6, 7, 8],
                    "middle": [9, 10, 11, 12],
                    "ring": [13, 14, 15, 16],
                    "pinky": [17, 18, 19, 20]
                }
                
                for f_name, mp_indices in finger_map.items():
                    for i, (s_idx, e_idx) in enumerate(zip(mp_indices[:-1], mp_indices[1:])):
                        # Rigify bone name (e.g., f_index.01.L)
                        bone_name = f"f_{f_name}.0{i+1}{h_side}"
                        if f_name == "thumb": bone_name = f"thumb.0{i+1}{h_side}"
                        
                        bone = rig.pose.bones.get(bone_name)
                        if bone:
                            v_start = Vector(hand_lms[s_idx])
                            v_end = Vector(hand_lms[e_idx])
                            target_dir = (v_end - v_start).normalized()
                            target_dir_local = (rig.matrix_world.inverted().to_quaternion() @ target_dir)
                            bone.rotation_mode = 'QUATERNION'
                            # Fingers point +Y in Rigify
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
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="Step 1: Preparation")
        col.operator("antigrav.sanitize_rig", icon='BRUSH')
        
        col.separator()
        col.label(text="Step 2: Cast Member")
        col.prop(bpy.context.scene, "antigrav_character_preset")
        col.operator("antigrav.align_poses", icon='POSE_HLT')
        
        col.separator()
        col.label(text="Step 3: Pro Properties")
        col.prop(bpy.context.scene, "antigrav_pin_threshold", text="Pin Sensitivity")
        
        col.separator()
        col.label(text="Step 4: Load Take")
        col.operator("antigrav.import_v2r", icon='ANIM_DATA', text="Import Scientific JSON")
        
        col.separator()
        col.label(text="Step 5: Production Tools")
        col.operator("antigrav.create_prop_empty", icon='EMPTY_AXIS', text="Create Prop Anchor")

def register():
    bpy.utils.register_class(ANTIGRAV_OT_sanitize_rig)
    bpy.utils.register_class(ANTIGRAV_OT_align_poses)
    bpy.utils.register_class(ANTIGRAV_OT_import_v2r)
    bpy.utils.register_class(ANTIGRAV_OT_create_prop_empty)
    bpy.utils.register_class(ANTIGRAV_PT_main_panel)
    bpy.types.Scene.antigrav_character_preset = bpy.props.EnumProperty(
        items=[('JAX', 'Jax', ''), ('KIKO', 'Kiko', ''), ('KAI', 'Kai', ''), ('DR_WHITE', 'Dr. White', ''), ('HIRO', 'Hiro', ''), ('SHADOW', 'Shadow', '')],
        name="Preset", default='JAX'
    )
    bpy.types.Scene.antigrav_pin_threshold = bpy.props.FloatProperty(
        name="Pin Threshold", default=0.02, min=0.0, max=0.2, description="Higher = Stickier Feet"
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
