"""
MelodicCap Blender Addon v3.0
=============================
FIXES FROM v2:
- Frame 0 = reference origin (no world position offset)
- Proper LOCAL bone space rotation
- Left/Right mirroring (MediaPipe person's left = Blender's right when facing)
- Root motion relative to first frame
- IK/FK mode switching
- Extensive debugging

For Blender 4.4.3+ and JaxRigify (1.87m)
"""

bl_info = {
    "name": "MelodicCap Motion Capture Importer",
    "author": "Karsten / MelodicCap Studio",
    "version": (3, 0, 0),
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
# LANDMARK NAMES
# =============================================================================

LANDMARKS = {
    0: "nose", 11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow", 15: "left_wrist", 16: "right_wrist",
    23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle", 31: "left_foot", 32: "right_foot",
}

# =============================================================================
# BONE MAPPING
# MediaPipe uses PERSON's left/right (mirrored when facing camera)
# When we face the person, their LEFT is our RIGHT
# So MediaPipe landmarks 11,13,15 (person's left arm) -> Blender .R bones
# =============================================================================

FK_CHAINS = {
    # bone_name: (start_landmark, end_landmark)
    # Person's LEFT side -> Blender RIGHT (.R)
    'upper_arm_fk.R': (11, 13),
    'forearm_fk.R': (13, 15),
    'thigh_fk.R': (23, 25),
    'shin_fk.R': (25, 27),
    # Person's RIGHT side -> Blender LEFT (.L)
    'upper_arm_fk.L': (12, 14),
    'forearm_fk.L': (14, 16),
    'thigh_fk.L': (24, 26),
    'shin_fk.L': (26, 28),
}

IK_TARGETS = {
    # bone_name: landmark_index
    'hand_ik.R': 15,  # Person's left wrist
    'hand_ik.L': 16,  # Person's right wrist
    'foot_ik.R': 27,  # Person's left ankle
    'foot_ik.L': 28,  # Person's right ankle
}

# =============================================================================
# UTILITIES
# =============================================================================

def debug(msg, level="INFO"):
    print(f"[{level}] MelodicCap: {msg}")

def get_lm(landmarks, idx):
    """Get landmark as Vector"""
    key = str(idx) if str(idx) in landmarks else idx
    if key in landmarks:
        p = landmarks[key]
        return Vector((p[0], p[1], p[2]))
    return None

def get_mid(landmarks, i1, i2):
    """Get midpoint"""
    p1, p2 = get_lm(landmarks, i1), get_lm(landmarks, i2)
    return (p1 + p2) / 2 if p1 and p2 else None

# =============================================================================
# IMPORTER
# =============================================================================

class MelodicCapImporter:
    
    def __init__(self, armature, take_data, settings):
        self.armature = armature
        self.take_data = take_data
        self.settings = settings
        
        self.ref_hip = None
        self.ref_landmarks = None
        self.scale = 1.0
        self.char_height = 1.87
        
        self.stats = {'frames': 0, 'keys': 0, 'bones': set()}
    
    def analyze(self):
        """Analyze character and mocap data"""
        debug("="*60)
        debug("ANALYSIS")
        debug("="*60)
        
        # Character height
        bones = self.armature.data.bones
        world = self.armature.matrix_world
        
        min_z, max_z = float('inf'), float('-inf')
        for bone in bones:
            h = (world @ bone.head_local).z
            t = (world @ bone.tail_local).z
            min_z, max_z = min(min_z, h, t), max(max_z, h, t)
        
        self.char_height = max_z - min_z if max_z > min_z else 1.87
        debug(f"  Character height: {self.char_height:.3f}m")
        
        # Print bone positions
        debug(f"\n  Key bone rest positions:")
        for name in ['upper_arm_fk.L', 'upper_arm_fk.R', 'thigh_fk.L', 'thigh_fk.R', 'torso']:
            if name in bones:
                b = bones[name]
                h = world @ b.head_local
                t = world @ b.tail_local
                d = (t - h).normalized()
                debug(f"    {name}:")
                debug(f"      head: ({h.x:.3f}, {h.y:.3f}, {h.z:.3f})")
                debug(f"      dir:  ({d.x:.3f}, {d.y:.3f}, {d.z:.3f})")
        
        # Mocap data
        frames = self.take_data.get('frames', [])
        if not frames:
            debug("ERROR: No frames!", "ERROR")
            return False
        
        debug(f"\n  Mocap: {len(frames)} frames, {self.take_data.get('duration_seconds', 0):.1f}s")
        
        # Reference frame (frame 0)
        self.ref_landmarks = frames[0].get('landmarks', {})
        self.ref_hip = get_mid(self.ref_landmarks, 23, 24)
        
        if not self.ref_hip:
            debug("ERROR: No hip center in frame 0!", "ERROR")
            return False
        
        debug(f"\n  REFERENCE FRAME (origin):")
        debug(f"    Hip center: ({self.ref_hip.x:.3f}, {self.ref_hip.y:.3f}, {self.ref_hip.z:.3f})")
        
        # Person height
        nose = get_lm(self.ref_landmarks, 0)
        ankle_l = get_lm(self.ref_landmarks, 27)
        ankle_r = get_lm(self.ref_landmarks, 28)
        
        if nose and ankle_l and ankle_r:
            ankle_mid = (ankle_l + ankle_r) / 2
            mocap_height = (nose.z - ankle_mid.z) + 0.15
            self.scale = self.char_height / mocap_height
            debug(f"    Person height: {mocap_height:.3f}m")
            debug(f"    SCALE: {self.scale:.4f}")
        
        # Print reference landmarks relative to hip
        debug(f"\n  Landmarks relative to hip (frame 0):")
        for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            pos = get_lm(self.ref_landmarks, idx)
            if pos:
                rel = pos - self.ref_hip
                name = LANDMARKS.get(idx, str(idx))
                debug(f"    [{idx:2d}] {name:15s}: ({rel.x:7.3f}, {rel.y:7.3f}, {rel.z:7.3f})")
        
        # Reference bone directions from mocap
        debug(f"\n  Reference bone directions (mocap):")
        for bone, (i1, i2) in FK_CHAINS.items():
            p1, p2 = get_lm(self.ref_landmarks, i1), get_lm(self.ref_landmarks, i2)
            if p1 and p2:
                d = (p2 - p1).normalized()
                debug(f"    {bone}: ({d.x:.3f}, {d.y:.3f}, {d.z:.3f})")
        
        return True
    
    def apply_fk(self):
        """Apply FK animation"""
        debug("\n" + "="*60)
        debug("APPLYING FK ANIMATION")
        debug("="*60)
        
        frames = self.take_data.get('frames', [])
        pose_bones = self.armature.pose.bones
        start = self.settings.get('start_frame', 1)
        
        # Set quaternion mode
        for pb in pose_bones:
            pb.rotation_mode = 'QUATERNION'
        
        # Find available bones
        avail = {b: (i1, i2) for b, (i1, i2) in FK_CHAINS.items() if b in pose_bones}
        debug(f"  Available: {list(avail.keys())}")
        
        # Get bone rest data
        bone_data = {}
        world = self.armature.matrix_world
        
        for bone_name in avail:
            bone = self.armature.data.bones[bone_name]
            
            # Rest direction in armature space
            head = bone.head_local
            tail = bone.tail_local
            rest_dir = (tail - head).normalized()
            
            # Rest direction in world space
            rest_dir_world = (world.to_3x3() @ rest_dir).normalized()
            
            bone_data[bone_name] = {
                'rest_dir': rest_dir,
                'rest_dir_world': rest_dir_world,
                'matrix_local': bone.matrix_local.copy(),
            }
            
            debug(f"  {bone_name} rest dir (world): ({rest_dir_world.x:.3f}, {rest_dir_world.y:.3f}, {rest_dir_world.z:.3f})")
        
        debug(f"\n  Processing {len(frames)} frames...")
        
        for fidx, fdata in enumerate(frames):
            bf = start + fidx
            bpy.context.scene.frame_set(bf)
            
            lms = fdata.get('landmarks', {})
            hip = get_mid(lms, 23, 24)
            
            if not hip:
                continue
            
            # === ROOT MOTION ===
            if 'torso' in pose_bones:
                # Delta from reference, scaled
                delta = (hip - self.ref_hip) * self.scale
                
                # Torso bone location is in armature local space
                # But we just apply the delta directly for now
                torso = pose_bones['torso']
                torso.location = Vector((delta.x, delta.y, delta.z))
                torso.keyframe_insert(data_path="location", frame=bf)
                self.stats['keys'] += 1
                self.stats['bones'].add('torso')
                
                if fidx == 0:
                    debug(f"    Frame 0 root delta: ({delta.x:.4f}, {delta.y:.4f}, {delta.z:.4f})")
            
            # === BONE ROTATIONS ===
            for bone_name, (i1, i2) in avail.items():
                p1, p2 = get_lm(lms, i1), get_lm(lms, i2)
                if not p1 or not p2:
                    continue
                
                # Current direction in mocap/world space
                cur_dir = (p2 - p1).normalized()
                
                # Get bone's rest direction in world space
                rest_dir = bone_data[bone_name]['rest_dir_world']
                
                # Calculate rotation from rest to current
                # This rotation is in world space
                rot_world = rest_dir.rotation_difference(cur_dir)
                
                # Now convert to bone local space
                # The bone's local rotation is relative to its parent
                pb = pose_bones[bone_name]
                
                # For Rigify FK bones, we need to account for parent transforms
                # The simplest approach: apply the world rotation, adjusted for bone's basis
                
                # Get the bone's rest orientation in world
                bone_rest_mat = world @ self.armature.data.bones[bone_name].matrix_local
                bone_rest_rot = bone_rest_mat.to_quaternion()
                
                # The local rotation is: 
                # bone_rest^-1 * rot_world * bone_rest (to rotate in local space)
                # But this doesn't account for parent...
                
                # Actually, for FK bones, we want to set rotation relative to rest
                # If parent is transformed, Blender handles that
                
                # Try: convert world rotation to local
                if pb.parent:
                    # Parent's current world rotation
                    parent_world = world @ pb.parent.bone.matrix_local
                    parent_rot = parent_world.to_quaternion()
                    
                    # Transform world rotation to parent-local
                    local_rot = parent_rot.inverted() @ rot_world
                else:
                    arm_rot = world.to_quaternion()
                    local_rot = arm_rot.inverted() @ rot_world
                
                pb.rotation_quaternion = local_rot
                pb.keyframe_insert(data_path="rotation_quaternion", frame=bf)
                self.stats['keys'] += 1
                self.stats['bones'].add(bone_name)
            
            self.stats['frames'] += 1
            if fidx % 50 == 0:
                debug(f"    Frame {fidx}/{len(frames)}")
        
        return True
    
    def apply_ik(self):
        """Apply IK animation"""
        debug("\n" + "="*60)
        debug("APPLYING IK ANIMATION")
        debug("="*60)
        
        frames = self.take_data.get('frames', [])
        pose_bones = self.armature.pose.bones
        start = self.settings.get('start_frame', 1)
        
        avail = {b: idx for b, idx in IK_TARGETS.items() if b in pose_bones}
        debug(f"  Available: {list(avail.keys())}")
        
        # Get reference IK positions (relative to hip)
        ref_pos = {}
        for bone, idx in avail.items():
            p = get_lm(self.ref_landmarks, idx)
            if p:
                ref_pos[bone] = p - self.ref_hip
                debug(f"  {bone} ref pos: ({ref_pos[bone].x:.3f}, {ref_pos[bone].y:.3f}, {ref_pos[bone].z:.3f})")
        
        debug(f"\n  Processing {len(frames)} frames...")
        
        for fidx, fdata in enumerate(frames):
            bf = start + fidx
            bpy.context.scene.frame_set(bf)
            
            lms = fdata.get('landmarks', {})
            hip = get_mid(lms, 23, 24)
            
            if not hip:
                continue
            
            # Root motion
            if 'torso' in pose_bones:
                delta = (hip - self.ref_hip) * self.scale
                torso = pose_bones['torso']
                torso.location = Vector((delta.x, delta.y, delta.z))
                torso.keyframe_insert(data_path="location", frame=bf)
                self.stats['keys'] += 1
                self.stats['bones'].add('torso')
            
            # IK targets
            for bone, idx in avail.items():
                p = get_lm(lms, idx)
                if not p:
                    continue
                
                # Position relative to current hip, scaled
                rel = (p - hip) * self.scale
                
                # Add hip movement to get world position
                hip_delta = (hip - self.ref_hip) * self.scale
                final = rel + hip_delta
                
                pb = pose_bones[bone]
                pb.location = final
                pb.keyframe_insert(data_path="location", frame=bf)
                self.stats['keys'] += 1
                self.stats['bones'].add(bone)
            
            self.stats['frames'] += 1
            if fidx % 50 == 0:
                debug(f"    Frame {fidx}/{len(frames)}")
        
        return True
    
    def summary(self):
        debug("\n" + "="*60)
        debug("SUMMARY")
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
        debug("MELODICCAP v3.0 IMPORT")
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
            'use_ik': context.scene.melodiccap_use_ik,
            'start_frame': context.scene.melodiccap_start_frame,
        }
        
        imp = MelodicCapImporter(arm, data, settings)
        
        if not imp.analyze():
            self.report({'ERROR'}, "Analysis failed")
            return {'CANCELLED'}
        
        bpy.ops.object.mode_set(mode='POSE')
        
        if settings['use_ik']:
            imp.apply_ik()
        else:
            imp.apply_fk()
        
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
        
        self.report({'INFO'}, "Cleared")
        return {'FINISHED'}


# =============================================================================
# PANEL
# =============================================================================

class MELODICCAP_PT_panel(bpy.types.Panel):
    bl_label = "MelodicCap v3"
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
        box.prop(context.scene, "melodiccap_use_ik")
        box.prop(context.scene, "melodiccap_start_frame")
        
        layout.operator("melodiccap.import_take", icon='IMPORT')
        layout.operator("melodiccap.clear", icon='X')


# =============================================================================
# REGISTER
# =============================================================================

classes = [MELODICCAP_OT_import, MELODICCAP_OT_clear, MELODICCAP_PT_panel]

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.melodiccap_use_ik = BoolProperty(name="Use IK", default=False)
    bpy.types.Scene.melodiccap_start_frame = IntProperty(name="Start Frame", default=1, min=1)
    debug("MelodicCap v3.0 registered")

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.melodiccap_use_ik
    del bpy.types.Scene.melodiccap_start_frame

if __name__ == "__main__":
    register()
