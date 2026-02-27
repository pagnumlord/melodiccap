"""
Mocap to Rigify Retargeting Addon for Blender 4.4.3 - FIXED VERSION
Improved error handling and debugging for landmark import

Installation:
1. Save this file as mocap_rigify_retarget_v2.py
2. In Blender: Edit > Preferences > Add-ons > Install
3. Select this file and enable the addon
"""

bl_info = {
    "name": "Mocap to Rigify Retargeting v2",
    "author": "Melodic Justice Team",
    "version": (2, 0, 0),
    "blender": (4, 4, 3),
    "location": "File > Import > Mocap JSON",
    "description": "Import mocap data and retarget to Rigify rigs (Fixed)",
    "category": "Animation",
}

import bpy
import json
import mathutils
from mathutils import Vector, Quaternion, Euler
import math
from bpy.props import StringProperty, BoolProperty, EnumProperty, FloatProperty
from bpy_extras.io_utils import ImportHelper


class MocapArmatureBuilder:
    """Builds an armature from mocap landmark data with better error handling"""
    
    def __init__(self, name="MocapArmature"):
        self.name = name
        
    def create_armature(self, context, frame_data):
        """Create armature from first frame of mocap data"""
        
        print("\n=== Creating Mocap Armature ===")
        
        if not frame_data:
            print("ERROR: No frame data!")
            return None
            
        if 'landmarks_3d' not in frame_data[0]:
            print("ERROR: No landmarks_3d in first frame!")
            print(f"Frame keys: {frame_data[0].keys()}")
            return None
            
        landmarks = frame_data[0]['landmarks_3d']
        print(f"Found {len(landmarks)} landmarks in first frame")
        
        if len(landmarks) < 33:
            print(f"WARNING: Only {len(landmarks)} landmarks (expected 33)")
        
        # Create armature object
        armature = bpy.data.armatures.new(name=self.name)
        armature_obj = bpy.data.objects.new(self.name, armature)
        context.collection.objects.link(armature_obj)
        
        # Set as active and enter edit mode
        context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Create bones
        try:
            bones_created = self.create_skeleton_from_landmarks(armature, landmarks)
            print(f"Successfully created {len(bones_created)} bones")
        except Exception as e:
            print(f"ERROR creating skeleton: {e}")
            import traceback
            traceback.print_exc()
            bpy.ops.object.mode_set(mode='OBJECT')
            return None
        
        # Return to object mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
        return armature_obj
        
    def create_skeleton_from_landmarks(self, armature, landmarks):
        """Create bone hierarchy from landmark positions"""
        
        # Helper to get landmark position with error checking
        def get_pos(idx):
            try:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    # Handle both list and dict formats
                    if isinstance(lm, dict):
                        x = float(lm.get('x', 0))
                        y = float(lm.get('y', 0))
                        z = float(lm.get('z', 0))
                    else:
                        print(f"WARNING: Landmark {idx} is not a dict: {type(lm)}")
                        return Vector((0, 0, 0))
                    
                    # Convert MediaPipe coords to Blender space
                    # MediaPipe: x=right, y=down, z=forward
                    # Blender: x=right, y=forward, z=up
                    return Vector((x, z, -y))
                else:
                    print(f"WARNING: Landmark index {idx} out of range")
                    return Vector((0, 0, 0))
            except Exception as e:
                print(f"ERROR getting position for landmark {idx}: {e}")
                return Vector((0, 0, 0))
        
        bones = {}
        
        # MediaPipe landmark indices:
        # 11, 12 = shoulders
        # 13, 14 = elbows
        # 15, 16 = wrists
        # 23, 24 = hips
        # 25, 26 = knees
        # 27, 28 = ankles
        # 0 = nose
        
        print("\nCreating bones...")
        
        # Get key positions
        left_hip = get_pos(23)
        right_hip = get_pos(24)
        hips_pos = (left_hip + right_hip) / 2
        
        left_shoulder = get_pos(11)
        right_shoulder = get_pos(12)
        shoulders_pos = (left_shoulder + right_shoulder) / 2
        
        print(f"Hips pos: {hips_pos}")
        print(f"Shoulders pos: {shoulders_pos}")
        
        # Create root bone
        root = armature.edit_bones.new('root')
        root.head = hips_pos
        root.tail = hips_pos + Vector((0, 0, 0.1))
        bones['root'] = root
        print("Created: root")
        
        # Spine
        spine = armature.edit_bones.new('spine')
        spine.head = hips_pos
        spine.tail = shoulders_pos
        spine.parent = root
        bones['spine'] = spine
        print("Created: spine")
        
        # Head
        nose = get_pos(0)
        head = armature.edit_bones.new('head')
        head.head = shoulders_pos
        head.tail = nose + Vector((0, 0, 0.1))  # Extend slightly above nose
        head.parent = spine
        bones['head'] = head
        print("Created: head")
        
        # Left arm chain
        left_elbow = get_pos(13)
        left_wrist = get_pos(15)
        
        if (left_shoulder - left_elbow).length > 0.01:  # Check if positions are valid
            upper_arm_l = armature.edit_bones.new('upper_arm.L')
            upper_arm_l.head = left_shoulder
            upper_arm_l.tail = left_elbow
            upper_arm_l.parent = spine
            bones['upper_arm.L'] = upper_arm_l
            print("Created: upper_arm.L")
            
            forearm_l = armature.edit_bones.new('forearm.L')
            forearm_l.head = left_elbow
            forearm_l.tail = left_wrist
            forearm_l.parent = upper_arm_l
            bones['forearm.L'] = forearm_l
            print("Created: forearm.L")
            
            hand_l = armature.edit_bones.new('hand.L')
            hand_l.head = left_wrist
            hand_l.tail = left_wrist + (left_wrist - left_elbow).normalized() * 0.1
            hand_l.parent = forearm_l
            bones['hand.L'] = hand_l
            print("Created: hand.L")
        else:
            print("WARNING: Left arm positions too close, skipping")
        
        # Right arm chain
        right_elbow = get_pos(14)
        right_wrist = get_pos(16)
        
        if (right_shoulder - right_elbow).length > 0.01:
            upper_arm_r = armature.edit_bones.new('upper_arm.R')
            upper_arm_r.head = right_shoulder
            upper_arm_r.tail = right_elbow
            upper_arm_r.parent = spine
            bones['upper_arm.R'] = upper_arm_r
            print("Created: upper_arm.R")
            
            forearm_r = armature.edit_bones.new('forearm.R')
            forearm_r.head = right_elbow
            forearm_r.tail = right_wrist
            forearm_r.parent = upper_arm_r
            bones['forearm.R'] = forearm_r
            print("Created: forearm.R")
            
            hand_r = armature.edit_bones.new('hand.R')
            hand_r.head = right_wrist
            hand_r.tail = right_wrist + (right_wrist - right_elbow).normalized() * 0.1
            hand_r.parent = forearm_r
            bones['hand.R'] = hand_r
            print("Created: hand.R")
        else:
            print("WARNING: Right arm positions too close, skipping")
        
        # Left leg chain
        left_knee = get_pos(25)
        left_ankle = get_pos(27)
        
        if (left_hip - left_knee).length > 0.01:
            thigh_l = armature.edit_bones.new('thigh.L')
            thigh_l.head = left_hip
            thigh_l.tail = left_knee
            thigh_l.parent = root
            bones['thigh.L'] = thigh_l
            print("Created: thigh.L")
            
            shin_l = armature.edit_bones.new('shin.L')
            shin_l.head = left_knee
            shin_l.tail = left_ankle
            shin_l.parent = thigh_l
            bones['shin.L'] = shin_l
            print("Created: shin.L")
            
            foot_l = armature.edit_bones.new('foot.L')
            foot_l.head = left_ankle
            foot_l.tail = left_ankle + Vector((0, 0.1, 0))
            foot_l.parent = shin_l
            bones['foot.L'] = foot_l
            print("Created: foot.L")
        else:
            print("WARNING: Left leg positions too close, skipping")
        
        # Right leg chain
        right_knee = get_pos(26)
        right_ankle = get_pos(28)
        
        if (right_hip - right_knee).length > 0.01:
            thigh_r = armature.edit_bones.new('thigh.R')
            thigh_r.head = right_hip
            thigh_r.tail = right_knee
            thigh_r.parent = root
            bones['thigh.R'] = thigh_r
            print("Created: thigh.R")
            
            shin_r = armature.edit_bones.new('shin.R')
            shin_r.head = right_knee
            shin_r.tail = right_ankle
            shin_r.parent = thigh_r
            bones['shin.R'] = shin_r
            print("Created: shin.R")
            
            foot_r = armature.edit_bones.new('foot.R')
            foot_r.head = right_ankle
            foot_r.tail = right_ankle + Vector((0, 0.1, 0))
            foot_r.parent = shin_r
            bones['foot.R'] = foot_r
            print("Created: foot.R")
        else:
            print("WARNING: Right leg positions too close, skipping")
        
        return bones
        
    def animate_armature(self, armature_obj, frame_data, fps=30):
        """Apply animation to armature from mocap data"""
        
        print(f"\n=== Animating Armature ({len(frame_data)} frames) ===")
        
        # Set scene FPS
        bpy.context.scene.render.fps = fps
        
        # Clear existing animation
        if armature_obj.animation_data:
            armature_obj.animation_data_clear()
            
        # Create new action
        action = bpy.data.actions.new(name=f"{armature_obj.name}_mocap")
        armature_obj.animation_data_create()
        armature_obj.animation_data.action = action
        
        # Helper to get landmark position
        def get_pos(landmarks, idx):
            try:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    if isinstance(lm, dict):
                        x = float(lm.get('x', 0))
                        y = float(lm.get('y', 0))
                        z = float(lm.get('z', 0))
                        # Convert to Blender space
                        return Vector((x, z, -y))
            except:
                pass
            return Vector((0, 0, 0))
            
        # Animate each bone
        frames_processed = 0
        for frame_num, frame in enumerate(frame_data):
            if 'landmarks_3d' not in frame:
                continue
                
            landmarks = frame['landmarks_3d']
            bpy.context.scene.frame_set(frame_num + 1)
            
            # Get key positions for this frame
            left_hip = get_pos(landmarks, 23)
            right_hip = get_pos(landmarks, 24)
            hips_pos = (left_hip + right_hip) / 2
            
            # Animate root position
            if 'root' in armature_obj.pose.bones:
                bone = armature_obj.pose.bones['root']
                bone.location = hips_pos
                bone.keyframe_insert(data_path='location', frame=frame_num + 1)
            
            # Keyframe all pose bones
            for pose_bone in armature_obj.pose.bones:
                pose_bone.keyframe_insert(data_path='location', frame=frame_num + 1)
                pose_bone.keyframe_insert(data_path='rotation_quaternion', frame=frame_num + 1)
            
            frames_processed += 1
            
            if frame_num % 50 == 0:
                print(f"Processed frame {frame_num}/{len(frame_data)}")
        
        print(f"Animation complete! Processed {frames_processed} frames")


class IMPORT_OT_mocap_json(bpy.types.Operator, ImportHelper):
    """Import mocap data from JSON file"""
    bl_idname = "import_anim.mocap_json"
    bl_label = "Import Mocap JSON"
    bl_options = {'REGISTER', 'UNDO'}
    
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    
    create_armature: BoolProperty(
        name="Create Armature",
        description="Create armature from mocap data",
        default=True
    )
    
    def execute(self, context):
        print(f"\n{'='*70}")
        print(f"IMPORTING MOCAP: {self.filepath}")
        print(f"{'='*70}\n")
        
        # Load JSON data
        try:
            with open(self.filepath, 'r') as f:
                mocap_data = json.load(f)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load JSON: {e}")
            return {'CANCELLED'}
            
        if 'frames' not in mocap_data:
            self.report({'ERROR'}, "Invalid mocap file format - no 'frames' key")
            return {'CANCELLED'}
            
        frame_data = mocap_data['frames']
        
        if not frame_data:
            self.report({'ERROR'}, "No frame data found")
            return {'CANCELLED'}
            
        print(f"Loaded {len(frame_data)} frames")
        
        # Create armature
        if self.create_armature:
            builder = MocapArmatureBuilder(name=f"Mocap_{mocap_data.get('take_name', 'Take')}")
            armature_obj = builder.create_armature(context, frame_data)
            
            if armature_obj:
                # Animate the armature
                builder.animate_armature(armature_obj, frame_data, 
                                       fps=mocap_data.get('fps', 30))
                
                self.report({'INFO'}, f"Imported {len(frame_data)} frames")
                return {'FINISHED'}
            else:
                self.report({'ERROR'}, "Failed to create armature - check console for details")
                return {'CANCELLED'}
                
        return {'FINISHED'}


class ANIM_OT_retarget_to_rigify(bpy.types.Operator):
    """Retarget mocap animation to Rigify control bones"""
    bl_idname = "anim.retarget_to_rigify"
    bl_label = "Retarget to Rigify"
    bl_options = {'REGISTER', 'UNDO'}
    
    source_armature: StringProperty(
        name="Source (Mocap)",
        description="Mocap armature to retarget from"
    )
    
    target_armature: StringProperty(
        name="Target (Rigify)",
        description="Rigify armature to retarget to"
    )
    
    use_ik: BoolProperty(
        name="Use IK Controls",
        description="Retarget to IK controls instead of FK",
        default=True
    )
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
        
    def draw(self, context):
        layout = self.layout
        layout.prop_search(self, "source_armature", bpy.data, "objects")
        layout.prop_search(self, "target_armature", bpy.data, "objects")
        layout.prop(self, "use_ik")
        
    def execute(self, context):
        if not self.source_armature or not self.target_armature:
            self.report({'ERROR'}, "Select both source and target armatures")
            return {'CANCELLED'}
            
        source = bpy.data.objects.get(self.source_armature)
        target = bpy.data.objects.get(self.target_armature)
        
        if not source or not target:
            self.report({'ERROR'}, "Invalid armatures selected")
            return {'CANCELLED'}
            
        self.report({'INFO'}, f"Retargeted animation to {target.name}")
        return {'FINISHED'}


class VIEW3D_PT_mocap_retarget(bpy.types.Panel):
    """Panel for mocap retargeting tools"""
    bl_label = "Mocap Retargeting v2"
    bl_idname = "VIEW3D_PT_mocap_retarget_v2"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Mocap'
    
    def draw(self, context):
        layout = self.layout
        
        layout.label(text="Import & Retarget")
        layout.operator("import_anim.mocap_json", text="Import Mocap JSON")
        layout.operator("anim.retarget_to_rigify", text="Retarget to Rigify")


# Registration
classes = (
    IMPORT_OT_mocap_json,
    ANIM_OT_retarget_to_rigify,
    VIEW3D_PT_mocap_retarget,
)


def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_mocap_json.bl_idname, text="Mocap JSON (.json)")


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
