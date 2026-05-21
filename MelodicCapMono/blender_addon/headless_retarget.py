"""Headless rest-relative retarget. Run INSIDE Blender:

    blender --background --python headless_retarget.py -- \
        --config resolved.json --bvh take.bvh --out take.blend --fps 30

For each frame, sets each mapped target pose-bone's LOCAL rotation via
the standard Blender skeleton-to-skeleton retarget identity:

    target_local_rot = target_rest^-1 @ source_world
                       @ source_rest^-1 @ target_rest

(using each rig's own bone.matrix_local rest matrices). That is the same
math Rokoko applies internally: transfer the source's pose delta from
its own rest into the target's rest frame. It uses Blender's own
rest matrices -- no axis-angle / Euler-order guessing. The previous
WORLD/WORLD Copy-Rotation+bake approach forced JaxRigify into SMPL's
T-pose world orientations and produced a stretched/hunched character on
real takes; this fixes it.

The source BVH skeleton is height-scaled to match the target rig before
the per-frame loop, so `*_ik` location targets (foot_ik, hand_ik) land
at the right world positions for IK chains.

The master .blend is OPENED then SAVED-AS to --out -- the master is
never mutated. Config is the per-character JSON written by process_take
(armature, base_blend, ik_fk_one, bone_map, calibration).
"""

from __future__ import annotations

import json
import sys

import bpy
import mathutils


def _argv_after_ddash():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def _fail(msg: str):
    print(f"[headless_retarget] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _bone_height(arm, head_bone, foot_bone):
    """|head.z - foot.z| of two rest bones, in armature local space."""
    bones = arm.data.bones
    if head_bone not in bones or foot_bone not in bones:
        return 0.0
    return abs(bones[head_bone].head_local.z - bones[foot_bone].head_local.z)


def main():
    import argparse
    ap = argparse.ArgumentParser(prog="headless_retarget")
    ap.add_argument("--config", required=True)
    ap.add_argument("--bvh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args(_argv_after_ddash())

    # utf-8-sig: tolerate a BOM in case some upstream writer added one.
    with open(args.config, encoding="utf-8-sig") as fh:
        cfg = json.loads(fh.read())
    armature_name = cfg["armature"]
    base_blend = cfg.get("base_blend")
    bone_map = cfg.get("bone_map", {})
    ik_fk_one = cfg.get("ik_fk_one", [])
    calibration = cfg.get("calibration", {})
    if not base_blend:
        _fail("config has no base_blend")

    # Open the master read-only; everything below saves to --out instead.
    try:
        bpy.ops.wm.open_mainfile(filepath=base_blend)
    except Exception as e:  # noqa: BLE001
        _fail(f"could not open base_blend {base_blend!r}: {e}")

    target = bpy.data.objects.get(armature_name)
    if target is None or target.type != "ARMATURE":
        _fail(f"armature object {armature_name!r} not found in {base_blend!r}")

    # Import the BVH (defaults match the validated manual File>Import>BVH;
    # the up-axis/facing fix is already baked into the BVH root by
    # pose_json_to_bvh --global-rot).
    before = set(bpy.data.objects.keys())
    try:
        bpy.ops.import_anim.bvh(
            filepath=args.bvh, axis_forward="-Z", axis_up="Y",
            rotate_mode="NATIVE", global_scale=1.0,
            update_scene_fps=False, update_scene_duration=True,
        )
    except Exception as e:  # noqa: BLE001
        _fail(f"BVH import failed: {e}")
    new_objs = [bpy.data.objects[n] for n in bpy.data.objects.keys()
                if n not in before]
    src = next((o for o in new_objs if o.type == "ARMATURE"), None)
    if src is None:
        _fail("no armature object created by the BVH import")

    src_action = src.animation_data.action if src.animation_data else None
    if src_action is None:
        _fail("imported BVH armature has no action")
    fstart, fend = (int(round(src_action.frame_range[0])),
                    int(round(src_action.frame_range[1])))

    # Clear any stale action on the target rig.
    if target.animation_data and target.animation_data.action:
        target.animation_data.action = None

    # Force the mapped limbs to FK so the *_fk targets are followed.
    for bname in ik_fk_one:
        pb = target.pose.bones.get(bname)
        if pb is None:
            print(f"[headless_retarget] WARN: IK_FK bone {bname!r} missing")
            continue
        if "IK_FK" in pb:
            pb["IK_FK"] = 1.0
        else:
            print(f"[headless_retarget] WARN: {bname!r} has no IK_FK prop")

    # Height-fit the source skeleton to the target rig so IK target world
    # positions land correctly. SMPL mean-shape is ~1.6 m; JaxRigify is
    # 1.87 m. Without this, foot_ik/hand_ik COPY_LOCATION targets sit at
    # SMPL coords -> the leg IK over-extends and pulls the spine down.
    src_pelvis = bone_map.get("pelvis", {}).get("bone")
    src_h = _bone_height(src, "pelvis", "left_foot") or src.dimensions.z
    tgt_foot = bone_map.get("left_ankle", {}).get("bone", "foot_fk.L")
    tgt_h = _bone_height(target, src_pelvis or "torso", tgt_foot) \
        or target.dimensions.z
    scale = (tgt_h / src_h) if src_h > 0 else 1.0
    src.scale = (scale, scale, scale)
    bpy.context.view_layer.update()
    print(f"[headless_retarget] scale fit: src_h={src_h:.3f} "
          f"tgt_h={tgt_h:.3f} scale={scale:.4f}")

    # Sort pairs by retarget type.
    rot_pairs = []  # (src_bone, tgt_bone)
    loc_pairs = []
    for src_bone, spec in bone_map.items():
        tgt_name = spec["bone"]
        ctype = spec.get("type", "COPY_ROTATION")
        if src_bone not in src.pose.bones:
            print(f"[headless_retarget] WARN: source bone {src_bone!r} "
                  f"absent in BVH — skipped")
            continue
        if tgt_name not in target.pose.bones:
            print(f"[headless_retarget] WARN: target bone {tgt_name!r} "
                  f"absent on {armature_name} — skipped")
            continue
        if ctype == "COPY_LOCATION":
            loc_pairs.append((src_bone, tgt_name))
        else:
            rot_pairs.append((src_bone, tgt_name))

    if not rot_pairs and not loc_pairs:
        _fail("no bone pairs resolved — check bone_map vs the rig bone names")

    # Pre-compute rest rotation matrices (don't change per frame).
    src_rest_R = {n: src.data.bones[n].matrix_local.to_3x3()
                  for n, _ in rot_pairs}
    tgt_rest_R = {n: target.data.bones[n].matrix_local.to_3x3()
                  for _, n in rot_pairs}
    src_rest_R_inv = {n: m.inverted() for n, m in src_rest_R.items()}
    tgt_rest_R_inv = {n: m.inverted() for n, m in tgt_rest_R.items()}

    # Force rotation_mode QUATERNION on rotation targets so keyframing
    # picks the right channel; same for location targets' rotation_mode
    # (harmless for them).
    for _, n in rot_pairs:
        target.pose.bones[n].rotation_mode = "QUATERNION"
    for _, n in loc_pairs:
        target.pose.bones[n].rotation_mode = "QUATERNION"

    # Per-frame retarget loop -- the actual work.
    scn = bpy.context.scene
    target_inv_world = target.matrix_world.inverted()
    rot_keys = 0
    loc_keys = 0
    for f in range(fstart, fend + 1):
        scn.frame_set(f)
        bpy.context.view_layer.update()
        for src_name, tgt_name in rot_pairs:
            src_pb = src.pose.bones[src_name]
            tgt_pb = target.pose.bones[tgt_name]
            src_R = src_pb.matrix.to_3x3()
            local_R = (tgt_rest_R_inv[tgt_name] @ src_R
                       @ src_rest_R_inv[src_name] @ tgt_rest_R[tgt_name])
            cal = calibration.get(tgt_name, {})
            inv = cal.get("invert")
            if inv and len(inv) == 3:
                # axis-toggle calibration: flip components of the local
                # quaternion's vector part on selected axes.
                q = local_R.to_quaternion()
                qx = -q.x if inv[0] else q.x
                qy = -q.y if inv[1] else q.y
                qz = -q.z if inv[2] else q.z
                tgt_pb.rotation_quaternion = mathutils.Quaternion(
                    (q.w, qx, qy, qz))
            else:
                tgt_pb.rotation_quaternion = local_R.to_quaternion()
            tgt_pb.keyframe_insert("rotation_quaternion", frame=f)
            rot_keys += 1
        for src_name, tgt_name in loc_pairs:
            src_pb = src.pose.bones[src_name]
            tgt_pb = target.pose.bones[tgt_name]
            # world head of source bone, expressed in target armature space:
            src_world_head = src.matrix_world @ src_pb.head
            desired_arm = target_inv_world @ src_world_head
            # Build pose-bone matrix = rest rotation + desired translation,
            # so location channel encodes the offset cleanly.
            m = target.data.bones[tgt_name].matrix_local.copy()
            m.translation = desired_arm
            tgt_pb.matrix = m
            tgt_pb.keyframe_insert("location", frame=f)
            loc_keys += 1

    # Drop the BVH source so the per-take .blend is just rig + baked action.
    bpy.data.objects.remove(src, do_unlink=True)

    # Scene FPS to match source (fractional via fps_base, mirrors the addon).
    fps = float(args.fps) if args.fps > 0 else 30.0
    scn.render.fps = max(1, round(fps))
    scn.render.fps_base = scn.render.fps / fps
    scn.frame_start, scn.frame_end = fstart, fend

    bpy.ops.wm.save_as_mainfile(filepath=args.out)
    print(f"[headless_retarget] OK frames={fstart}-{fend} "
          f"rot_pairs={len(rot_pairs)} loc_pairs={len(loc_pairs)} "
          f"keys={rot_keys + loc_keys} "
          f"fps={scn.render.fps}/{scn.render.fps_base:.4f} "
          f"out={args.out}")


if __name__ == "__main__":
    main()
