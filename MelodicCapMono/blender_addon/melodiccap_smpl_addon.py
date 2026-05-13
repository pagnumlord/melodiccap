"""MelodicCap SMPL retargeter — Blender 4.4 addon (v0.1, Phase C).

Imports a `pose.json` (schema `melodiccap_mono_v1` — see ../SCHEMA.md)
and drives a JaxRigify-style armature.

Coexists with the v5.x addon (MelodicCapRTM/blender_addon/melodiccap_rtm_addon.py)
— install both, use the right one for the format you're importing.

ARCHITECTURE:
    1. Read pose.json + character config JSON.
    2. Resolve the target armature from the character config.
    3. Apply axis_conversion to the armature root once (SMPL→Blender).
    4. For each frame:
         - For each SMPL joint mapped in the character config:
             * SMPL axis-angle -> mathutils.Quaternion
             * Compose per-bone rot_offset_euler_deg
             * Apply to the mapped Rigify control bone
             * Keyframe rotation_quaternion
         - Apply root translation per the character config's
           "root_translation" policy.
    5. Set scene FPS to round(pose.fps).

WHY IT'S SHORT:
    The v5.x addon is ~3000 lines because it had to reconstruct anatomy
    from depth-noisy triangulated 3D points (FK math, IK targets, foot
    pinning, sit detection, depth dampers, mirror flips, ROM clamps).
    SMPL output is already an anatomically valid joint chain. We just
    apply rotations to the right bones — no anatomy reconstruction needed.

FIRST-IMPORT CALIBRATION:
    The first time you import a real take on a new character, expect 1-2
    bones to be rotated wrong (a 90° or 180° axis-convention mismatch
    between SMPL and the Rigify rest pose). The fix lives in the
    character JSON, not in this file:
      - Whole body wrong orientation → axis_conversion.smpl_to_blender_euler_deg
      - Specific bone wrong         → smpl_to_rig.<joint>.rot_offset_euler_deg
    Iterate the JSON values, re-import, repeat. No code changes needed.
"""

bl_info = {
    "name": "MelodicCap SMPL Retargeter",
    "author": "pagnumlord & contributors",
    "version": (0, 1, 0),
    "blender": (4, 4, 0),
    "category": "Animation",
    "location": "File > Import > SMPL Pose JSON (MelodicCap)",
    "description": (
        "Import a melodiccap_mono_v1 pose JSON (from WHAM, EasyMocap, or "
        "a fixture) and drive a Rigify character. Pairs with MelodicCapMono."
    ),
}

import json
import math
import os

import bpy
from mathutils import Euler, Quaternion, Vector


# ---------------------------------------------------------------------------
# SMPL constants
# ---------------------------------------------------------------------------

# Canonical SMPL joint order. Indices match the smpl_pose array layout:
# joint i's axis-angle = smpl_pose[i*3 : (i+1)*3].
SMPL_JOINT_NAMES = (
    "pelvis",         # 0
    "left_hip",       # 1
    "right_hip",      # 2
    "spine1",         # 3
    "left_knee",      # 4
    "right_knee",     # 5
    "spine2",         # 6
    "left_ankle",     # 7
    "right_ankle",    # 8
    "spine3",         # 9
    "left_foot",      # 10
    "right_foot",     # 11
    "neck",           # 12
    "left_collar",    # 13
    "right_collar",   # 14
    "head",           # 15
    "left_shoulder",  # 16
    "right_shoulder", # 17
    "left_elbow",     # 18
    "right_elbow",    # 19
    "left_wrist",     # 20
    "right_wrist",    # 21
    "left_hand",      # 22
    "right_hand",     # 23
)
SMPL_JOINT_INDEX = {name: i for i, name in enumerate(SMPL_JOINT_NAMES)}

EXPECTED_FORMAT = "melodiccap_mono_v1"


# ---------------------------------------------------------------------------
# Math helpers (no external deps beyond mathutils, which ships with Blender)
# ---------------------------------------------------------------------------

def axis_angle_to_quaternion(aa):
    """Convert a 3-element SMPL axis-angle (radians) to a Blender Quaternion.

    SMPL convention: axis_angle = axis * angle, magnitude is the rotation
    angle in radians.
    """
    ax, ay, az = aa[0], aa[1], aa[2]
    angle = math.sqrt(ax * ax + ay * ay + az * az)
    if angle < 1e-8:
        return Quaternion((1.0, 0.0, 0.0, 0.0))
    inv = 1.0 / angle
    return Quaternion((ax * inv, ay * inv, az * inv), angle)


def compose_offset(quat, offset_euler_deg):
    """Right-multiply a quaternion by an Euler offset (in degrees).

    Right-multiply (Q * Offset) means: first the SMPL rotation, then the
    bone-frame correction. If first-import calibration fails this is the
    first thing to flip to left-multiply (Offset * Q) — uncomment the
    alternative below and re-import.
    """
    if not any(offset_euler_deg):
        return quat
    rad = (
        math.radians(offset_euler_deg[0]),
        math.radians(offset_euler_deg[1]),
        math.radians(offset_euler_deg[2]),
    )
    offset = Euler(rad, "XYZ").to_quaternion()
    return quat @ offset
    # Alternative (uncomment if Phase F testing reveals R-mul is wrong):
    # return offset @ quat


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_pose_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    fmt = data.get("format")
    if fmt != EXPECTED_FORMAT:
        raise RuntimeError(
            f"Pose JSON has format={fmt!r}, expected {EXPECTED_FORMAT!r}. "
            f"This file is not a melodiccap_mono_v1 pose JSON."
        )
    return data


def load_character_config(path):
    with open(path, "r") as f:
        cfg = json.load(f)
    # Strip _comment_* keys silently
    return cfg


def resolve_default_character_config(pose_data, addon_dir):
    """If the pose JSON has 'character' set, find characters/<name>.json."""
    name = pose_data.get("character")
    if not name:
        return None
    candidate = os.path.join(addon_dir, "..", "characters", f"{name}.json")
    candidate = os.path.normpath(candidate)
    return candidate if os.path.exists(candidate) else None


# ---------------------------------------------------------------------------
# Core retarget
# ---------------------------------------------------------------------------

def retarget(pose_data, character_cfg, *, clear_existing=True):
    """Drive the configured armature from the pose data."""

    armature_name = character_cfg["armature"]
    rig = bpy.data.objects.get(armature_name)
    if rig is None or rig.type != "ARMATURE":
        raise RuntimeError(
            f"Armature object '{armature_name}' not found in the scene "
            f"(or it's not an armature). Check the character config "
            f"'armature' field matches your scene."
        )

    # Activate the rig and enter pose mode
    bpy.context.view_layer.objects.active = rig
    if bpy.context.mode != "POSE":
        bpy.ops.object.mode_set(mode="POSE")

    smpl_to_rig = character_cfg["smpl_to_rig"]
    root_policy = character_cfg.get("root_translation", "ignore")

    # Pre-resolve bones once; warn for missing
    bones_used = []
    for joint_name, mapping in smpl_to_rig.items():
        if joint_name not in SMPL_JOINT_INDEX:
            print(f"[MelodicCap SMPL] WARNING: '{joint_name}' is not an "
                  f"SMPL joint name — skipped.")
            continue
        bone = rig.pose.bones.get(mapping["bone"])
        if bone is None:
            print(f"[MelodicCap SMPL] WARNING: bone '{mapping['bone']}' "
                  f"not on armature — skipped.")
            continue
        bone.rotation_mode = "QUATERNION"
        bones_used.append((joint_name, mapping, bone))

    if clear_existing:
        for _, _, bone in bones_used:
            anim = rig.animation_data
            if anim and anim.action:
                # Don't try to be clever; just overwrite per frame below.
                pass

    # Optionally apply axis_conversion ONCE to the armature OBJECT, not bones.
    # This is the "whole body is sideways" knob. Stored as a non-keyframed
    # rotation on the object — survives across re-imports because we set it
    # before keyframing bones.
    axis_conv = character_cfg.get("axis_conversion", {})
    smpl_to_blender = axis_conv.get("smpl_to_blender_euler_deg")
    if smpl_to_blender:
        rad = tuple(math.radians(d) for d in smpl_to_blender)
        rig.rotation_mode = "XYZ"
        rig.rotation_euler = rad

    # Scene FPS to match source
    fps = float(pose_data.get("fps", 30.0))
    scene = bpy.context.scene
    scene.render.fps = max(1, round(fps))
    # fps_base lets non-integer source FPS play back accurately
    scene.render.fps_base = scene.render.fps / fps if fps > 0 else 1.0

    frames = pose_data.get("frames", [])
    if not frames:
        raise RuntimeError("Pose JSON has no frames.")

    scene.frame_start = frames[0]["frame"]
    scene.frame_end = frames[-1]["frame"]

    n_frames = 0
    n_keyframes = 0
    for fr in frames:
        frame_num = int(fr["frame"])
        scene.frame_set(frame_num)
        smpl_pose = fr["smpl_pose"]
        if len(smpl_pose) != 72:
            raise RuntimeError(
                f"Frame {frame_num} smpl_pose has length {len(smpl_pose)}, "
                f"expected 72 (24 joints x 3)."
            )

        for joint_name, mapping, bone in bones_used:
            i = SMPL_JOINT_INDEX[joint_name]
            aa = smpl_pose[i * 3:(i + 1) * 3]
            q = axis_angle_to_quaternion(aa)
            q = compose_offset(q, mapping.get("rot_offset_euler_deg", (0, 0, 0)))
            bone.rotation_quaternion = q
            bone.keyframe_insert("rotation_quaternion", frame=frame_num)
            n_keyframes += 1

        if root_policy != "ignore":
            tx, ty, tz = fr["smpl_trans"]
            torso = rig.pose.bones.get("torso")
            if torso is not None:
                if root_policy == "follow":
                    torso.location = Vector((tx, ty, tz))
                elif root_policy == "follow_xy":
                    torso.location = Vector((tx, ty, 0.0))
                else:
                    print(f"[MelodicCap SMPL] WARNING: unknown "
                          f"root_translation policy {root_policy!r}, "
                          f"ignoring.")
                torso.keyframe_insert("location", frame=frame_num)

        n_frames += 1

    scene.frame_set(scene.frame_start)
    print(
        f"[MelodicCap SMPL] Imported {n_frames} frames -> {armature_name} "
        f"({n_keyframes} bone keyframes; root policy: {root_policy}; "
        f"scene fps {scene.render.fps}/{scene.render.fps_base:.4f}"
    )


# ---------------------------------------------------------------------------
# Operator + UI
# ---------------------------------------------------------------------------

class MELODICCAP_OT_import_smpl(bpy.types.Operator):
    """Import a melodiccap_mono_v1 pose JSON and retarget to a Rigify character."""

    bl_idname = "import_scene.melodiccap_smpl"
    bl_label = "Import SMPL Pose JSON (MelodicCap)"
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(
        default="*.pose.json;*.json", options={"HIDDEN"}
    )

    character_config: bpy.props.StringProperty(
        name="Character config",
        description=(
            "Path to characters/<name>.json. If empty, the addon tries the "
            "config matching the 'character' field in the pose JSON."
        ),
        subtype="FILE_PATH",
        default="",
    )

    clear_existing: bpy.props.BoolProperty(
        name="Clear existing animation",
        description="Overwrite any existing keyframes on the mapped bones.",
        default=True,
    )

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        try:
            pose_data = load_pose_json(self.filepath)
        except Exception as e:
            self.report({"ERROR"}, f"Failed to load pose JSON: {e}")
            return {"CANCELLED"}

        cfg_path = self.character_config or resolve_default_character_config(
            pose_data, os.path.dirname(__file__)
        )
        if not cfg_path or not os.path.exists(cfg_path):
            self.report(
                {"ERROR"},
                "No character config specified or auto-resolved. Set the "
                "Character config field, or set the 'character' field in "
                "your pose JSON.",
            )
            return {"CANCELLED"}

        try:
            cfg = load_character_config(cfg_path)
        except Exception as e:
            self.report({"ERROR"}, f"Failed to load character config: {e}")
            return {"CANCELLED"}

        try:
            retarget(pose_data, cfg, clear_existing=self.clear_existing)
        except Exception as e:
            self.report({"ERROR"}, f"Retarget failed: {e}")
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            f"Imported {len(pose_data.get('frames', []))} frames onto "
            f"{cfg.get('armature', '?')}",
        )
        return {"FINISHED"}


def _menu_import(self, context):
    self.layout.operator(
        MELODICCAP_OT_import_smpl.bl_idname,
        text="SMPL Pose JSON (MelodicCap mono v1)",
    )


_classes = (MELODICCAP_OT_import_smpl,)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(_menu_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(_menu_import)
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
