"""One command per take: video -> baked per-take .blend, unattended.

    python -m MelodicCapMono.orchestrate.process_take \
        guitar.mp4 --character jax \
        --wham-python /path/to/envs/wham/bin/python \
        --base-blend "C:/.../Characters.blend" \
        --blender-exe "C:/.../Blender/blender.exe" \
        --out guitar.blend

Stages (each reuses existing, proven code — nothing reinvented here):
  1. Resolve characters/<name>.json (+ optional untracked
     characters/<name>.local.json deep-merged over it, + CLI overrides).
  2. WHAM:  subprocess `-m MelodicCapMono.wham.video2pose` in the WHAM
     conda env (skipped with --from-pose-json).
  3. BVH:   `MelodicCapMono.wham.pose_json_to_bvh.convert()` in-process
     (pure numpy), using the config's bvh_export block.
  4. Retarget: headless Blender runs blender_addon/headless_retarget.py
     against a COPY of the master .blend (master never mutated).
  5. Write <out>.report.json.

This module is light-env (stdlib + numpy via pose_json_to_bvh). It does
NOT import torch/bpy. The WHAM env stays separate; Blender is a
subprocess.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HEADLESS_SCRIPT = (
    REPO_ROOT / "MelodicCapMono" / "blender_addon" / "headless_retarget.py"
)
CHARACTERS_DIR = REPO_ROOT / "MelodicCapMono" / "characters"


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge `over` into a copy of `base` (over wins)."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_character(name: str) -> dict:
    cfg_path = CHARACTERS_DIR / f"{name}.json"
    if not cfg_path.exists():
        sys.exit(
            f"ERROR: no character config {cfg_path}. "
            f"Available: {sorted(p.stem for p in CHARACTERS_DIR.glob('*.json') if not p.stem.endswith('.local'))}"
        )
    # utf-8-sig transparently strips a BOM if the file has one — Windows
    # PowerShell's `Set-Content -Encoding utf8` writes a BOM that Python's
    # default json.loads chokes on at char 0. With sig: BOM or no-BOM works.
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    # Optional untracked personal override (machine paths etc.)
    local_path = CHARACTERS_DIR / f"{name}.local.json"
    if local_path.exists():
        cfg = _deep_merge(cfg, json.loads(local_path.read_text(encoding="utf-8-sig")))
    return cfg


def _resolve(cfg: dict, args) -> dict:
    """Apply CLI/env overrides on top of the merged config."""
    if args.base_blend:
        cfg["base_blend"] = args.base_blend
    if args.blender_exe:
        cfg["blender_exe"] = args.blender_exe
    cfg["blender_exe"] = cfg.get("blender_exe") or os.environ.get(
        "MELODICCAP_BLENDER"
    )
    return cfg


def _run(cmd, cwd=None, dry=False):
    printable = " ".join(str(c) for c in cmd)
    print(f"[process_take] $ {printable}" + (f"   (cwd={cwd})" if cwd else ""))
    if dry:
        return 0
    return subprocess.run(cmd, cwd=cwd).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="MelodicCapMono.orchestrate.process_take",
        description="Video -> baked per-take .blend (unattended).",
    )
    ap.add_argument("video", nargs="?", type=Path,
                    help="Input video (omit with --from-pose-json).")
    ap.add_argument("--character", default="jax",
                    help="Character config name (characters/<name>.json).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output .blend (default: <video stem>.blend in cwd).")
    ap.add_argument("--from-pose-json", type=Path, default=None,
                    help="Skip WHAM; use this melodiccap_mono_v1 JSON.")
    ap.add_argument("--from-pkl", type=Path, default=None,
                    help="Pass through to video2pose --from-pkl.")
    ap.add_argument("--fps", type=float, default=None,
                    help="Override source FPS.")
    ap.add_argument("--wham-python", default=sys.executable,
                    help="Python interpreter of the WHAM conda env "
                         "(default: current — i.e. run from the wham env).")
    ap.add_argument("--wham-dir", default=None,
                    help="WHAM checkout (else $WHAM_DIR).")
    ap.add_argument("--base-blend", default=None,
                    help="Master .blend with the rig (overrides config).")
    ap.add_argument("--blender-exe", default=None,
                    help="Blender binary (overrides config / $MELODICCAP_BLENDER).")
    ap.add_argument("--keep-intermediate", action="store_true",
                    help="Keep the intermediate .pose.json and .bvh.")
    ap.add_argument("--no-foot-contact", action="store_true",
                    help="Skip Stage 3.5 (foot-contact detection + IK lock). "
                         "Reproduces today's behavior bit-for-bit.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the planned stages without executing.")
    args = ap.parse_args(argv)

    if args.video is None and args.from_pose_json is None:
        ap.error("give a video, or --from-pose-json")

    cfg = _resolve(_load_character(args.character), args)

    stem = (args.from_pose_json or args.video).stem.split(".")[0]
    out_blend = (args.out or Path(f"{stem}.blend")).resolve()
    out_blend.parent.mkdir(parents=True, exist_ok=True)
    work = out_blend.parent
    pose_json = (args.from_pose_json
                 or work / f"{stem}.pose.json").resolve()
    bvh_path = (work / f"{stem}.bvh").resolve()

    report = {
        "tool": "process_take",
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "character": args.character,
        "video": str(args.video) if args.video else None,
        "stages": {},
    }

    # --- Stage 2: WHAM (skipped with --from-pose-json) ---
    if args.from_pose_json is None:
        cmd = [args.wham_python, "-m", "MelodicCapMono.wham.video2pose",
               str(args.video.resolve()), str(pose_json),
               "--character", args.character]
        if args.fps is not None:
            cmd += ["--fps", str(args.fps)]
        if args.wham_dir:
            cmd += ["--wham-dir", args.wham_dir]
        if args.from_pkl:
            cmd += ["--from-pkl", str(args.from_pkl.resolve())]
        rc = _run(cmd, cwd=str(REPO_ROOT), dry=args.dry_run)
        if rc != 0 and not args.dry_run:
            sys.exit(f"WHAM stage failed (exit {rc}).")
    report["stages"]["wham"] = {"pose_json": str(pose_json),
                                "skipped": args.from_pose_json is not None}

    # --- Stage 3: BVH (in-process, pure numpy) ---
    from MelodicCapMono.wham import pose_json_to_bvh
    from MelodicCapMono.wham._smpl_fk import (
        euler_deg_to_matrix, resolve_global_rot_deg)
    bvh_cfg = cfg.get("bvh_export", {})
    root_motion = bvh_cfg.get("root_motion", "zero")
    # "auto" (default) resolves from the pose JSON's coordinate_frame
    # tag: no correction for gravity-aligned converter output, the
    # legacy (180, 0, 0) flip for old camera-frame JSONs. An explicit
    # [x, y, z] in the config still overrides.
    global_rot_cfg = bvh_cfg.get("global_rot_euler_deg", "auto")
    global_rot = (None if global_rot_cfg in ("auto", None)
                  else tuple(global_rot_cfg))
    trans_span_m = None
    if args.dry_run:
        print(f"[process_take] BVH: convert {pose_json} -> {bvh_path} "
              f"(root_motion={root_motion}, global_rot={global_rot_cfg})")
        src_fps = args.fps or 30.0
        n_frames = "?"
    else:
        import numpy as np
        data = pose_json_to_bvh._load_pose_json(pose_json)
        resolved_rot = resolve_global_rot_deg(data, global_rot)
        bvh_text = pose_json_to_bvh.convert(
            data, root_motion=root_motion, global_rot_deg=global_rot)
        bvh_path.write_text(bvh_text)
        src_fps = args.fps or float(data.get("fps", 30.0))
        n_frames = len(data["frames"])
        # Horizontal root-path span (in the upright frame): how far the
        # performer actually travelled. Drives the pinned-root warning.
        R = euler_deg_to_matrix(*resolved_rot)
        t = np.array([fr.get("smpl_trans", [0.0, 0.0, 0.0])
                      for fr in data["frames"]], dtype=np.float64) @ R.T
        trans_span_m = float(np.hypot(t[:, 0].max() - t[:, 0].min(),
                                      t[:, 2].max() - t[:, 2].min()))
        print(f"[process_take] BVH: {n_frames} frames -> {bvh_path} "
              f"(global_rot={list(resolved_rot)}, "
              f"root path span {trans_span_m:.2f} m)")
    report["stages"]["bvh"] = {"bvh": str(bvh_path), "frames": n_frames,
                               "fps": src_fps, "root_motion": root_motion,
                               "global_rot": global_rot_cfg,
                               "trans_span_m": trans_span_m}

    # --- Stage 3.5: foot-contact detection (in-process, pure numpy) ---
    # Adds Phase 4's contact-aware IK lock layer on top of the rotation
    # retarget. Bypassed when foot_contact.enabled is false or
    # --no-foot-contact is set; output is identical to today's pipeline.
    foot_contact_cfg = cfg.get("foot_contact", {})
    footlock_enabled = (foot_contact_cfg.get("enabled", False)
                        and not args.no_foot_contact)
    footlock_path = None
    if footlock_enabled:
        from MelodicCapMono.orchestrate import footlock
        footlock_path = (work / f"{stem}.footlock.json").resolve()
        if args.dry_run:
            print(f"[process_take] footlock: -> {footlock_path}")
            report["stages"]["foot_contact"] = {
                "sidecar": str(footlock_path), "dry_run": True}
        else:
            fl = footlock.write_footlock_sidecar(data, footlock_path,
                                                 config=cfg)
            report["stages"]["foot_contact"] = {
                "sidecar": str(footlock_path),
                "floor_z": fl["floor_z"],
                "frames_in_contact": fl["frames_in_contact"],
                "intervals": fl["intervals"],
            }
            asym = abs(fl["floor_z"]["L"] - fl["floor_z"]["R"])
            print(f"[process_take] footlock: {len(fl['intervals'])} intervals, "
                  f"in_contact L={fl['frames_in_contact']['L']} "
                  f"R={fl['frames_in_contact']['R']}/{n_frames}, "
                  f"floor_z asym={asym:.3f} m -> {footlock_path}")
            if asym > 0.05:
                print(f"[process_take] WARN: floor_z asymmetry > 5 cm "
                      f"({asym:.3f}) — calibration may be biased.")
            # Contact intervals are detected in world space (root
            # translation included), but root_motion="zero" pins the
            # rig's root — on a travelling take the planted foot MUST
            # slide under a pinned root, so the IK lock would fight the
            # walk instead of fixing skate.
            if (root_motion == "zero" and trans_span_m is not None
                    and trans_span_m > 0.5):
                print(f"[process_take] WARN: take travels "
                      f"{trans_span_m:.2f} m but root_motion='zero' pins "
                      f"the root. Foot-lock will fight the walk. Set "
                      f"bvh_export.root_motion='trans' for travelling "
                      f"takes, or pass --no-foot-contact.")
    else:
        report["stages"]["foot_contact"] = {"skipped": True}

    # --- Stage 4: headless retarget (master .blend never mutated) ---
    base_blend = cfg.get("base_blend")
    blender_exe = cfg.get("blender_exe")
    if not base_blend:
        sys.exit("ERROR: base_blend not set. Pass --base-blend, or set it in "
                 f"characters/{args.character}.local.json (untracked).")
    if not blender_exe:
        sys.exit("ERROR: blender_exe not set. Pass --blender-exe or set "
                 "$MELODICCAP_BLENDER (Steam Blender binary; not on PATH).")

    # Write the fully-resolved config to a temp file the headless script reads.
    resolved_cfg = work / f".{stem}.cfg.json"
    if not args.dry_run:
        resolved_cfg.write_text(json.dumps(cfg, indent=2))
    cmd = [blender_exe, "--background", "--python", str(HEADLESS_SCRIPT), "--",
           "--config", str(resolved_cfg), "--bvh", str(bvh_path),
           "--out", str(out_blend), "--fps", str(src_fps)]
    if footlock_path is not None:
        cmd += ["--footlock", str(footlock_path)]
    rc = _run(cmd, cwd=str(REPO_ROOT), dry=args.dry_run)
    if rc != 0 and not args.dry_run:
        sys.exit(f"Headless retarget failed (exit {rc}). See Blender stderr.")
    report["stages"]["retarget"] = {"out_blend": str(out_blend),
                                    "base_blend": base_blend}

    # --- Cleanup + report ---
    if not args.dry_run:
        if not args.keep_intermediate:
            for p in (bvh_path, resolved_cfg):
                p.unlink(missing_ok=True)
            if footlock_path is not None:
                footlock_path.unlink(missing_ok=True)
            if args.from_pose_json is None:
                pose_json.unlink(missing_ok=True)
        report_path = out_blend.with_suffix(".report.json")
        report_path.write_text(json.dumps(report, indent=2))
        print(f"[process_take] DONE -> {out_blend}\n"
              f"[process_take] report -> {report_path}")
    else:
        print("[process_take] (dry run — nothing executed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
