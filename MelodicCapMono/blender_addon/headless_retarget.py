"""Headless constraint-bake retarget. Run INSIDE Blender:

    blender --background --python headless_retarget.py -- \
        --config resolved.json --bvh take.bvh --out take.blend --fps 30

Adapts the proven RigifyRetargeter from
oldscripts/mocap_to_rigify_complete.py (WORLD-space COPY_ROTATION /
COPY_LOCATION -> bpy.ops.nla.bake(visual_keying=True) -> strip MOCAP_*).
WORLD/WORLD copy constraints make Blender's own solver resolve the
SMPL-vs-Rigify rest-frame difference per frame — the exact thing the
Phase C local-axis-angle addon got wrong. No bespoke rotation math here.

The master .blend is OPENED then SAVED-AS to --out, so it is never
mutated. Config is the fully-resolved per-character JSON written by
process_take (armature, base_blend, ik_fk_one, bone_map, calibration).
"""

from __future__ import annotations

import json
import sys

import bpy


def _argv_after_ddash():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def _fail(msg: str):
    print(f"[headless_retarget] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


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

    # Add WORLD-space copy constraints (the proven mechanism).
    added = 0
    constrained = []
    for src_bone, spec in bone_map.items():
        tgt_name = spec["bone"]
        ctype = spec.get("type", "COPY_ROTATION")
        if src_bone not in src.pose.bones:
            print(f"[headless_retarget] WARN: source bone {src_bone!r} "
                  f"absent in BVH — skipped")
            continue
        tgt = target.pose.bones.get(tgt_name)
        if tgt is None:
            print(f"[headless_retarget] WARN: target bone {tgt_name!r} "
                  f"absent on {armature_name} — skipped")
            continue
        for c in list(tgt.constraints):
            if c.name.startswith("MOCAP_"):
                tgt.constraints.remove(c)
        con = tgt.constraints.new(ctype)
        con.name = f"MOCAP_{src_bone}"
        con.target = src
        con.subtarget = src_bone
        con.target_space = "WORLD"
        con.owner_space = "WORLD"
        con.influence = 1.0
        if ctype == "COPY_ROTATION":
            inv = calibration.get(tgt_name, {}).get("invert")
            if inv and len(inv) == 3:
                con.invert_x, con.invert_y, con.invert_z = (
                    bool(inv[0]), bool(inv[1]), bool(inv[2]))
        added += 1
        constrained.append(tgt_name)

    if added == 0:
        _fail("no constraints added — check bone_map vs the rig bone names")

    # Bake the constrained bones to keyframes (visual keying resolves the
    # constraints in world space), then strip the constraints.
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="DESELECT")
    for pb in target.pose.bones:
        if any(c.name.startswith("MOCAP_") for c in pb.constraints):
            pb.bone.select = True
    bpy.ops.nla.bake(
        frame_start=fstart, frame_end=fend, only_selected=True,
        visual_keying=True, clear_constraints=False,
        use_current_action=True, bake_types={"POSE"},
    )
    for pb in target.pose.bones:
        for c in list(pb.constraints):
            if c.name.startswith("MOCAP_"):
                pb.constraints.remove(c)
    bpy.ops.object.mode_set(mode="OBJECT")

    # Drop the BVH source so the per-take .blend is just rig + baked action.
    bpy.data.objects.remove(src, do_unlink=True)

    # Scene FPS to match source (fractional via fps_base, mirrors the addon).
    scn = bpy.context.scene
    fps = float(args.fps) if args.fps > 0 else 30.0
    scn.render.fps = max(1, round(fps))
    scn.render.fps_base = scn.render.fps / fps
    scn.frame_start, scn.frame_end = fstart, fend

    bpy.ops.wm.save_as_mainfile(filepath=args.out)
    print(f"[headless_retarget] OK frames={fstart}-{fend} "
          f"bones={added} fps={scn.render.fps}/{scn.render.fps_base:.4f} "
          f"out={args.out}")


if __name__ == "__main__":
    main()
