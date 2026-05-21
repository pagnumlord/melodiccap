"""Headless Rokoko retarget. Run INSIDE Blender:

    blender --background --python headless_retarget.py -- \
        --config resolved.json --bvh take.bvh --out take.blend --fps 30

The user's Blender has the Rokoko Studio Live plugin enabled (autoloads
on file open -- visible in every run as "Loaded Rokoko Studio Live for
Blender successfully!"). Rokoko exposes its retarget as standard bpy
operators (`bpy.ops.rsl.retarget_animation`) and scene properties
(`rsl_retargeting_*`). Their retarget engine handles rest-pose alignment,
auto-scaling, and per-bone basis differences correctly -- exactly the
class of artifacts the hand-rolled rest-relative approach kept
producing on real takes.

Earlier WORLD/WORLD copy constraints, then a rest-relative matrix loop,
each fixed one symptom and surfaced another. This path stops
re-implementing skeleton-to-skeleton retarget math and instead drives
the proven retargeter that the user's manual workflow already validated.

The master .blend is OPENED then SAVED-AS to --out -- never mutated.
Config is the per-character JSON written by process_take.
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

    # Import the BVH; defaults match the validated manual File>Import>BVH.
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

    # --- Rokoko Studio Live: configure + retarget ---
    scn = bpy.context.scene
    if not hasattr(scn, "rsl_retargeting_bone_list"):
        _fail(
            "Rokoko Studio Live addon properties not present "
            "(scn.rsl_retargeting_bone_list missing). Enable the plugin "
            "in this Blender, or open the master .blend interactively "
            "once so its preferences persist."
        )
    scn.rsl_retargeting_armature_source = src
    scn.rsl_retargeting_armature_target = target
    if hasattr(scn, "rsl_retargeting_auto_scaling"):
        scn.rsl_retargeting_auto_scaling = True
    if hasattr(scn, "rsl_retargeting_use_pose"):
        try:
            scn.rsl_retargeting_use_pose = "REST"
        except Exception:  # noqa: BLE001
            # Some Rokoko builds use a different enum value name
            pass

    # Populate the Rokoko bone list from cfg.bone_map (clear anything stale
    # left over from a previous interactive session in this .blend).
    bone_list = scn.rsl_retargeting_bone_list
    try:
        bone_list.clear()
    except Exception:  # noqa: BLE001
        while len(bone_list):
            bone_list.remove(0)

    pairs_added = 0
    for src_bone, spec in bone_map.items():
        tgt_name = spec["bone"]
        if src_bone not in src.pose.bones:
            print(f"[headless_retarget] WARN: source bone {src_bone!r} "
                  f"absent in BVH — skipped")
            continue
        if tgt_name not in target.pose.bones:
            print(f"[headless_retarget] WARN: target bone {tgt_name!r} "
                  f"absent on {armature_name} — skipped")
            continue
        item = bone_list.add()
        item.bone_name_source = src_bone
        item.bone_name_target = tgt_name
        pairs_added += 1

    if pairs_added == 0:
        _fail("no bone pairs added to the Rokoko list — check bone_map "
              "vs the source BVH bones and target rig bones")

    print(f"[headless_retarget] Rokoko: source={src.name!r} "
          f"target={target.name!r} pairs={pairs_added} "
          f"auto_scale=True use_pose=REST")

    try:
        bpy.ops.rsl.retarget_animation()
    except Exception as e:  # noqa: BLE001
        _fail(
            f"bpy.ops.rsl.retarget_animation() failed: {e}\n"
            f"If this is a headless-context error, the workaround is to "
            f"wrap the call in a context override; raise an issue with "
            f"the exact traceback above."
        )

    # Drop the BVH source so the per-take .blend is just rig + baked action.
    bpy.data.objects.remove(src, do_unlink=True)

    # Scene FPS to match source (fractional via fps_base, mirrors the addon).
    fps = float(args.fps) if args.fps > 0 else 30.0
    scn.render.fps = max(1, round(fps))
    scn.render.fps_base = scn.render.fps / fps
    scn.frame_start, scn.frame_end = fstart, fend

    bpy.ops.wm.save_as_mainfile(filepath=args.out)
    print(f"[headless_retarget] OK frames={fstart}-{fend} pairs={pairs_added} "
          f"fps={scn.render.fps}/{scn.render.fps_base:.4f} "
          f"out={args.out} (engine=rokoko)")


if __name__ == "__main__":
    main()
