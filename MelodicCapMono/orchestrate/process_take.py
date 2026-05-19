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
    cfg = json.loads(cfg_path.read_text())
    # Optional untracked personal override (machine paths etc.)
    local_path = CHARACTERS_DIR / f"{name}.local.json"
    if local_path.exists():
        cfg = _deep_merge(cfg, json.loads(local_path.read_text()))
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
    bvh_cfg = cfg.get("bvh_export", {})
    root_motion = bvh_cfg.get("root_motion", "zero")
    global_rot = tuple(bvh_cfg.get("global_rot_euler_deg", [180.0, 0.0, 0.0]))
    if args.dry_run:
        print(f"[process_take] BVH: convert {pose_json} -> {bvh_path} "
              f"(root_motion={root_motion}, global_rot={global_rot})")
        src_fps = args.fps or 30.0
        n_frames = "?"
    else:
        data = pose_json_to_bvh._load_pose_json(pose_json)
        bvh_text = pose_json_to_bvh.convert(
            data, root_motion=root_motion, global_rot_deg=global_rot)
        bvh_path.write_text(bvh_text)
        src_fps = args.fps or float(data.get("fps", 30.0))
        n_frames = len(data["frames"])
        print(f"[process_take] BVH: {n_frames} frames -> {bvh_path}")
    report["stages"]["bvh"] = {"bvh": str(bvh_path), "frames": n_frames,
                               "fps": src_fps, "root_motion": root_motion,
                               "global_rot": list(global_rot)}

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
