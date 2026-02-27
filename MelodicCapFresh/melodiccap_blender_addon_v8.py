"""
MelodicCap Blender Addon v8.0
=============================
FIXES FROM v7:
- IK targets now ROTATE with the body (crucial fix!)
- Full 3D torso orientation (pitch, roll, yaw)
- Distributed spine rotation
- IK positions are body-relative, then transformed by body rotation

For Blender 4.4.3+ and JaxRigify (1.87m)
"""

bl_info = {
    "name": "MelodicCap Motion Capture Importer",
    "author": "Karsten / MelodicCap Studio",
    "version": (8, 0, 0),
    "blender": (4, 4, 0),
    "location": "View3D > Sidebar > MelodicCap",
    "description": "Import MelodicCap mocap data to Rigify",
    "category": "Animation",
}

import bpy
import json
import math
from mathutils import Vector, Matrix, Quaternion, Euler
from bpy.props import StringProperty, FloatProperty, BoolProperty, IntProperty
from bpy_extras.io_utils import ImportHelper

# =============================================================================
# LANDMARK INDICES
# =============================================================================

LM_NOSE = 0
LM_LEFT_SHOULDER = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_ELBOW = 13
LM_RIGHT_ELBOW = 14
LM_LEFT_WRIST = 15
LM_RIGHT_WRIST = 16
LM_LEFT_HIP = 23
LM_RIGHT_HIP = 24
LM_LEFT_KNEE = 25
LM_RIGHT_KNEE = 26
LM_LEFT_ANKLE = 27
LM_RIGHT_ANKLE = 28

# =============================================================================
# BONE MAPPING
# =============================================================================

IK_CONFIG = {
    'hand_ik.R': {
        'endpoint': LM_LEFT_WRIST,
        'root': LM_LEFT_SHOULDER,
        'max_length_bones': ['upper_arm_fk.R', 'forearm_fk.R'],
    },
    'hand_ik.L': {
        'endpoint': LM_RIGHT_WRIST,
        'root': LM_RIGHT_SHOULDER,
        'max_length_bones': ['upper_arm_fk.L', 'forearm_fk.L'],
    },
    'foot_ik.R': {
        'endpoint': LM_LEFT_ANKLE,
        'root': LM_LEFT_HIP,
        'max_length_bones': ['thigh_fk.R', 'shin_fk.R'],
    },
    'foot_ik.L': {
        'endpoint': LM_RIGHT_ANKLE,
        'root': LM_RIGHT_HIP,
        'max_length_bones': ['thigh_fk.L', 'shin_fk.L'],
    },
}

IK_FK_SWITCHES = {
    'upper_arm_parent.L': 'IK_FK',
    'upper_arm_parent.R': 'IK_FK',
    'thigh_parent.L': 'IK_FK',
    'thigh_parent.R': 'IK_FK',
}

# =============================================================================
# UTILITIES
# =============================================================================

def debug(msg, level="INFO"):
    print(f"[{level}] MelodicCap: {msg}")

def get_lm(landmarks, idx):
    key = str(idx) if str(idx) in landmarks else idx
    if key in landmarks:
        p = landmarks[key]
        return Vector((p[0], p[1], p[2]))
    return None

def get_mid(landmarks, i1, i2):
    p1, p2 = get_lm(landmarks, i1), get_lm(landmarks, i2)
    return (p1 + p2) / 2 if p1 and p2 else None

def clamp_length(vec, max_len):
    if vec.length > max_len and vec.length > 0.0001:
        return vec * (max_len / vec.length)
    return vec

def calculate_body_orientation(l_shoulder, r_shoulder, l_hip, r_hip):
    """
    Calculate full 3D body orientation from shoulder and hip positions.
    Returns a rotation matrix representing the body's orientation.
    
    Coordinate system:
    - X axis: Left to Right (shoulder line)
    - Y axis: Back to Front (perpendicular to shoulders, horizontal)
    - Z axis: Down to Up (vertical)
    """
    # Shoulder midpoint and hip midpoint
    shoulder_mid = (l_shoulder + r_shoulder) / 2
    hip_mid = (l_hip + r_hip) / 2
    
    # X axis: Left to Right shoulder direction
    x_axis = (r_shoulder - l_shoulder).normalized()
    
    # Spine direction (hip to shoulder) - this gives us the "up" with lean
    spine = (shoulder_mid - hip_mid).normalized()
    
    # Y axis: Forward direction (perpendicular to X, mostly horizontal)
    # Cross product of spine and X gives us forward
    y_axis = spine.cross(x_axis).normalized()
    
    # Recalculate Z axis to ensure orthogonality
    z_axis = x_axis.cross(y_axis).normalized()
    
    # Build rotation matrix (column vectors are the axes)
    rot_matrix = Matrix((
        (x_axis.x, y_axis.x, z_axis.x),
        (x_axis.y, y_axis.y, z_axis.y),
        (x_axis.z, y_axis.z, z_axis.z)
    )).to_3x3()
    
    return rot_matrix

def calculate_z_rotation(l_pos, r_pos):
    """Calculate just the Z-axis rotation (yaw) from left/right positions."""
    vec = r_pos - l_pos
    angle = math.atan2(-vec.y, -vec.x)  # Facing direction
    return angle

# =============================================================================
# IMPORTER
# =============================================================================

class MelodicCapImporter:
    
    def __init__(self, armature, take_data, settings):
        self.armature = armature
        self.take_data = take_data
        self.settings = settings
        
        # Reference data
        self.ref_landmarks = None
        self.ref_hip = None
        self.ref_hip_rotation = 0.0
        self.ref_shoulder_rotation = 0.0
        self.ref_body_matrix = None
        
        # Character data
        self.char_max_lengths = {}
        self.mocap_root_positions = {}
        
        # Reference limb vectors (in body-local space)
        self.ref_limb_vectors = {}
        
        # Scale
        self.scale = 1.0
        
        self.stats = {'frames': 0, 'keys': 0, 'bones': set()}
    
    def analyze(self):
        """Analyze character and mocap data"""
        debug("="*60)
        debug("MELODICCAP v8.0 - ROTATION-AWARE IK")
        debug("="*60)
        
        bones = self.armature.data.bones
        world = self.armature.matrix_world
        
        # === CHARACTER ANALYSIS ===
        debug(f"\n  CHARACTER ANALYSIS:")
        
        for ik_bone, config in IK_CONFIG.items():
            total_length = 0.0
            for bone_name in config['max_length_bones']:
                if bone_name in bones:
                    bone = bones[bone_name]
                    head = world @ bone.head_local
                    tail = world @ bone.tail_local
                    total_length += (tail - head).length
            self.char_max_lengths[ik_bone] = total_length
            debug(f"    {ik_bone} max length: {total_length:.3f}m")
        
        # Character height
        min_z, max_z = float('inf'), float('-inf')
        for bone in bones:
            h = (world @ bone.head_local).z
            t = (world @ bone.tail_local).z
            min_z, max_z = min(min_z, h, t), max(max_z, h, t)
        char_height = max_z - min_z
        debug(f"    Character height: {char_height:.3f}m")
        
        # === MOCAP ANALYSIS ===
        frames = self.take_data.get('frames', [])
        if not frames:
            debug("ERROR: No frames!", "ERROR")
            return False
        
        debug(f"\n  MOCAP ANALYSIS:")
        debug(f"    Frames: {len(frames)}")
        
        # Reference frame
        self.ref_landmarks = frames[0].get('landmarks', {})
        self.ref_hip = get_mid(self.ref_landmarks, LM_LEFT_HIP, LM_RIGHT_HIP)
        
        if not self.ref_hip:
            debug("ERROR: No hip!", "ERROR")
            return False
        
        # Scale
        nose = get_lm(self.ref_landmarks, LM_NOSE)
        ankle_l = get_lm(self.ref_landmarks, LM_LEFT_ANKLE)
        ankle_r = get_lm(self.ref_landmarks, LM_RIGHT_ANKLE)
        
        if nose and ankle_l and ankle_r:
            ankle_mid = (ankle_l + ankle_r) / 2
            mocap_height = (nose.z - ankle_mid.z) + 0.15
            self.scale = char_height / mocap_height
            debug(f"    Scale factor: {self.scale:.4f}")
        
        # === REFERENCE BODY ORIENTATION ===
        l_shoulder = get_lm(self.ref_landmarks, LM_LEFT_SHOULDER)
        r_shoulder = get_lm(self.ref_landmarks, LM_RIGHT_SHOULDER)
        l_hip = get_lm(self.ref_landmarks, LM_LEFT_HIP)
        r_hip = get_lm(self.ref_landmarks, LM_RIGHT_HIP)
        
        if l_shoulder and r_shoulder and l_hip and r_hip:
            self.ref_body_matrix = calculate_body_orientation(l_shoulder, r_shoulder, l_hip, r_hip)
            self.ref_hip_rotation = calculate_z_rotation(l_hip, r_hip)
            self.ref_shoulder_rotation = calculate_z_rotation(l_shoulder, r_shoulder)
            
            debug(f"\n  REFERENCE ORIENTATION:")
            debug(f"    Hip Z-rotation: {math.degrees(self.ref_hip_rotation):.1f}°")
            debug(f"    Shoulder Z-rotation: {math.degrees(self.ref_shoulder_rotation):.1f}°")
        
        # === REFERENCE LIMB VECTORS (relative to body center) ===
        debug(f"\n  REFERENCE LIMB POSITIONS (relative to hip):")
        for ik_bone, config in IK_CONFIG.items():
            endpoint = get_lm(self.ref_landmarks, config['endpoint'])
            if endpoint:
                # Store position relative to hip center
                rel_pos = endpoint - self.ref_hip
                self.ref_limb_vectors[ik_bone] = rel_pos.copy()
                debug(f"    {ik_bone}: ({rel_pos.x:.3f}, {rel_pos.y:.3f}, {rel_pos.z:.3f})")
            
            # Also store root positions
            root = get_lm(self.ref_landmarks, config['root'])
            if root:
                self.mocap_root_positions[ik_bone] = root.copy()
        
        return True
    
    def set_ik_mode(self):
        pose_bones = self.armature.pose.bones
        for switch_bone, prop_name in IK_FK_SWITCHES.items():
            if switch_bone in pose_bones:
                pb = pose_bones[switch_bone]
                if prop_name in pb:
                    pb[prop_name] = 0.0
    
    def apply_animation(self):
        """Apply animation with rotation-aware IK"""
        debug("\n" + "="*60)
        debug("APPLYING ANIMATION (Rotation-Aware IK)")
        debug("="*60)
        
        frames = self.take_data.get('frames', [])
        pose_bones = self.armature.pose.bones
        start = self.settings.get('start_frame', 1)
        
        self.set_ik_mode()
        
        for pb in pose_bones:
            pb.rotation_mode = 'QUATERNION'
        
        avail_ik = {b: c for b, c in IK_CONFIG.items() if b in pose_bones}
        debug(f"  Available IK: {list(avail_ik.keys())}")
        
        has_torso = 'torso' in pose_bones
        has_chest = 'chest' in pose_bones
        
        debug(f"  Torso: {has_torso}, Chest: {has_chest}")
        debug(f"\n  Processing {len(frames)} frames...")
        
        for fidx, fdata in enumerate(frames):
            bf = start + fidx
            bpy.context.scene.frame_set(bf)
            
            lms = fdata.get('landmarks', {})
            
            # Get positions
            hip = get_mid(lms, LM_LEFT_HIP, LM_RIGHT_HIP)
            l_shoulder = get_lm(lms, LM_LEFT_SHOULDER)
            r_shoulder = get_lm(lms, LM_RIGHT_SHOULDER)
            l_hip = get_lm(lms, LM_LEFT_HIP)
            r_hip = get_lm(lms, LM_RIGHT_HIP)
            
            if not hip or not l_hip or not r_hip:
                continue
            
            # === CALCULATE BODY ROTATION ===
            current_hip_rot = calculate_z_rotation(l_hip, r_hip)
            hip_rotation_delta = current_hip_rot - self.ref_hip_rotation
            
            shoulder_rotation_delta = 0.0
            if l_shoulder and r_shoulder:
                current_shoulder_rot = calculate_z_rotation(l_shoulder, r_shoulder)
                shoulder_rotation_delta = current_shoulder_rot - self.ref_shoulder_rotation
            
            spine_twist = shoulder_rotation_delta - hip_rotation_delta
            
            # Hip translation delta
            hip_delta = hip - self.ref_hip
            scaled_hip_delta = hip_delta * self.scale
            
            # === TORSO ===
            if has_torso:
                torso = pose_bones['torso']
                
                # Location
                torso.location = Vector((scaled_hip_delta.x, scaled_hip_delta.y, scaled_hip_delta.z))
                torso.keyframe_insert(data_path="location", frame=bf)
                
                # Rotation (Z-axis = yaw)
                rot_quat = Quaternion((0, 0, 1), hip_rotation_delta)
                torso.rotation_quaternion = rot_quat
                torso.keyframe_insert(data_path="rotation_quaternion", frame=bf)
                
                self.stats['keys'] += 2
                self.stats['bones'].add('torso')
            
            # === CHEST (Spine twist) ===
            if has_chest:
                chest = pose_bones['chest']
                twist_quat = Quaternion((0, 0, 1), spine_twist)
                chest.rotation_quaternion = twist_quat
                chest.keyframe_insert(data_path="rotation_quaternion", frame=bf)
                
                self.stats['keys'] += 1
                self.stats['bones'].add('chest')
            
            # === IK TARGETS (ROTATION-AWARE!) ===
            # The key insight: IK targets need to follow body rotation
            # 1. Calculate limb position relative to hip (in current mocap frame)
            # 2. Calculate the DELTA from reference position
            # 3. ROTATE that delta by the body rotation
            # 4. Add hip translation
            
            # Rotation matrix for transforming deltas
            rot_matrix = Matrix.Rotation(hip_rotation_delta, 3, 'Z')
            
            for ik_bone, config in avail_ik.items():
                endpoint = get_lm(lms, config['endpoint'])
                
                if not endpoint:
                    continue
                
                # Current position relative to hip
                current_rel = endpoint - hip
                
                # Reference position relative to hip
                ref_rel = self.ref_limb_vectors.get(ik_bone)
                if not ref_rel:
                    continue
                
                # Delta in mocap space (how the limb moved relative to body)
                limb_delta_mocap = current_rel - ref_rel
                
                # Scale the delta
                scaled_limb_delta = limb_delta_mocap * self.scale
                
                # ROTATE the limb delta by body rotation!
                # This keeps the hands attached when the body turns
                rotated_limb_delta = rot_matrix @ scaled_limb_delta
                
                # Add hip translation (already in world-ish space)
                total_delta = scaled_hip_delta + rotated_limb_delta
                
                # Clamp to prevent stretching
                max_len = self.char_max_lengths.get(ik_bone, 1.0)
                if rotated_limb_delta.length > max_len * 0.5:
                    rotated_limb_delta = clamp_length(rotated_limb_delta, max_len * 0.5)
                    total_delta = scaled_hip_delta + rotated_limb_delta
                
                # Apply
                pb = pose_bones[ik_bone]
                pb.location = total_delta
                pb.keyframe_insert(data_path="location", frame=bf)
                
                self.stats['keys'] += 1
                self.stats['bones'].add(ik_bone)
            
            self.stats['frames'] += 1
            
            if fidx % 50 == 0:
                debug(f"    Frame {fidx}/{len(frames)}")
                if fidx > 0:
                    debug(f"      Hip rot: {math.degrees(hip_rotation_delta):.1f}°, Twist: {math.degrees(spine_twist):.1f}°")
            
            if fidx == 0:
                debug(f"    Frame 0:")
                debug(f"      Torso location: ({scaled_hip_delta.x:.4f}, {scaled_hip_delta.y:.4f}, {scaled_hip_delta.z:.4f})")
                debug(f"      Torso rotation: {math.degrees(hip_rotation_delta):.1f}°")
                debug(f"      Chest twist: {math.degrees(spine_twist):.1f}°")
        
        return True
    
    def summary(self):
        debug("\n" + "="*60)
        debug("IMPORT SUMMARY")
        debug("="*60)
        debug(f"  Frames: {self.stats['frames']}")
        debug(f"  Keyframes: {self.stats['keys']}")
        debug(f"  Bones: {sorted(self.stats['bones'])}")


# =============================================================================
# OPERATORS
# =============================================================================

class MELODICCAP_OT_import(bpy.types.Operator, ImportHelper):
    bl_idname = "melodiccap.import_take"
    bl_label = "Import Take"
    bl_options = {'REGISTER', 'UNDO'}
    
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    
    def execute(self, context):
        debug("\n" + "="*60)
        debug("MELODICCAP v8.0 IMPORT")
        debug("="*60)
        
        arm = context.active_object
        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature!")
            return {'CANCELLED'}
        
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        
        settings = {
            'start_frame': context.scene.melodiccap_start_frame,
        }
        
        imp = MelodicCapImporter(arm, data, settings)
        
        if not imp.analyze():
            self.report({'ERROR'}, "Analysis failed")
            return {'CANCELLED'}
        
        bpy.ops.object.mode_set(mode='POSE')
        
        imp.apply_animation()
        imp.summary()
        
        self.report({'INFO'}, f"Imported {imp.stats['frames']} frames")
        return {'FINISHED'}


class MELODICCAP_OT_clear(bpy.types.Operator):
    bl_idname = "melodiccap.clear"
    bl_label = "Clear Animation"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        arm = context.active_object
        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select armature!")
            return {'CANCELLED'}
        
        if arm.animation_data:
            arm.animation_data_clear()
        
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.pose.transforms_clear()
        
        debug("Cleared animation")
        self.report({'INFO'}, "Cleared")
        return {'FINISHED'}


class MELODICCAP_OT_set_ik(bpy.types.Operator):
    bl_idname = "melodiccap.set_ik"
    bl_label = "Set IK Mode"
    
    def execute(self, context):
        arm = context.active_object
        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select armature!")
            return {'CANCELLED'}
        
        for switch_bone, prop_name in IK_FK_SWITCHES.items():
            if switch_bone in arm.pose.bones:
                pb = arm.pose.bones[switch_bone]
                if prop_name in pb:
                    pb[prop_name] = 0.0
        
        self.report({'INFO'}, "Set IK mode")
        return {'FINISHED'}


# =============================================================================
# PANEL
# =============================================================================

class MELODICCAP_PT_panel(bpy.types.Panel):
    bl_label = "MelodicCap v8"
    bl_idname = "MELODICCAP_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'MelodicCap'
    
    def draw(self, context):
        layout = self.layout
        
        box = layout.box()
        box.label(text="Target:", icon='ARMATURE_DATA')
        if context.active_object and context.active_object.type == 'ARMATURE':
            box.label(text=f"  {context.active_object.name}")
        else:
            box.label(text="  (Select armature)")
        
        box = layout.box()
        box.prop(context.scene, "melodiccap_start_frame")
        
        layout.operator("melodiccap.import_take", icon='IMPORT')
        layout.operator("melodiccap.set_ik", icon='CON_KINEMATIC')
        layout.operator("melodiccap.clear", icon='X')


# =============================================================================
# REGISTER
# =============================================================================

classes = [MELODICCAP_OT_import, MELODICCAP_OT_clear, MELODICCAP_OT_set_ik, MELODICCAP_PT_panel]

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.melodiccap_start_frame = IntProperty(name="Start Frame", default=1, min=1)
    debug("MelodicCap v8.0 registered - Rotation-aware IK!")

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.melodiccap_start_frame

if __name__ == "__main__":
    register()
