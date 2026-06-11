"""Batch processor: feed a manifest of takes, get a folder of baked .blends.

The production unlock for filming a short film. Drop a session's clips
in a folder, write one line per take, run overnight, open baked .blends
in the morning.

USAGE:
    python -m MelodicCapMono.orchestrate.batch shoot01.csv \\
        --wham-python /path/to/envs/wham/bin/python \\
        [--continue-on-error] [--dry-run] [--no-foot-contact]

MANIFEST FORMAT (CSV, header row required):

    video,from_pose_json,character,fps,out_name
    C:/clips/walk_01.mp4,,jax,30,walk_01
    C:/clips/turn_02.mp4,,jax,,
    ,C:/clips/cached.pose.json,kai,60,kai_take03

Per-row rules:
  * Exactly one of `video` or `from_pose_json` must be set per row.
  * `character` empty -> defaults to "jax".
  * `fps` empty -> use the video's native FPS (process_take detects it).
  * `out_name` empty -> derived from <video|pose_json stem>.blend, written
    in the manifest's directory.

EXECUTION:
  * Serial (WHAM is GPU-bound; parallelism contends for one GPU).
  * Per-take stdout/stderr captured to <out_name>.log next to the .blend.
  * <out_name>.report.json comes from process_take (BVH frames, foot
    contact intervals, retarget summary).
  * <manifest_dir>/batch_report.json aggregates per-take status, return
    code, duration, frame count, and a quick-link to each per-take
    report+log. Default: stop on first failure (pass
    `--continue-on-error` for unattended overnight runs).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_manifest(path: Path) -> list[dict]:
    """Parse the manifest CSV. Tolerate UTF-8 BOM (Excel saves with one)."""
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for i, raw in enumerate(reader, start=2):  # row 2 = first data line
            video = (raw.get("video") or "").strip()
            fpj = (raw.get("from_pose_json") or "").strip()
            if not video and not fpj:
                continue  # blank line — skip silently
            if video and fpj:
                sys.exit(
                    f"manifest row {i}: set EITHER video OR from_pose_json, "
                    f"not both"
                )
            rows.append({
                "video": video or None,
                "from_pose_json": fpj or None,
                "character": (raw.get("character") or "").strip() or "jax",
                "fps": (raw.get("fps") or "").strip() or None,
                "out_name": (raw.get("out_name") or "").strip() or None,
                "_row": i,
            })
    if not rows:
        sys.exit(f"manifest {path} has no rows")
    return rows


def _resolve_out_path(row, manifest_dir: Path) -> Path:
    name = row["out_name"]
    if not name:
        stem_src = Path(row["video"] or row["from_pose_json"]).stem
        # Strip .pose if the stem comes from a .pose.json
        if stem_src.endswith(".pose"):
            stem_src = stem_src[:-len(".pose")]
        name = f"{stem_src}.blend"
    elif not name.endswith(".blend"):
        name = f"{name}.blend"
    return (manifest_dir / name).resolve()


def _build_cmd(row, out_path, common_args):
    cmd = [sys.executable, "-m",
           "MelodicCapMono.orchestrate.process_take"]
    if row["video"]:
        cmd.append(row["video"])
    cmd += ["--character", row["character"], "--out", str(out_path)]
    if row["from_pose_json"]:
        cmd += ["--from-pose-json", row["from_pose_json"]]
    if row["fps"]:
        cmd += ["--fps", str(row["fps"])]
    cmd += common_args
    return cmd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="MelodicCapMono.orchestrate.batch",
        description="Process a manifest of takes serially.",
    )
    ap.add_argument("manifest", type=Path,
                    help="CSV manifest (see module docstring for schema).")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="Skip failed takes and continue (default: stop on "
                         "first failure).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print planned commands without executing.")
    ap.add_argument("--report", type=Path, default=None,
                    help="Where to write batch_report.json (default: next "
                         "to manifest).")
    # Pass-through to process_take (applied to every take)
    ap.add_argument("--no-foot-contact", action="store_true",
                    help="Disable foot-contact stage for all takes.")
    ap.add_argument("--wham-python", default=None,
                    help="Python in the WHAM conda env (passed to every take).")
    ap.add_argument("--wham-dir", default=None,
                    help="WHAM checkout root (else $WHAM_DIR).")
    ap.add_argument("--base-blend", default=None,
                    help="Master .blend; usually set in characters/<n>.local.json instead.")
    ap.add_argument("--blender-exe", default=None,
                    help="Blender binary; usually set in characters/<n>.local.json instead.")
    ap.add_argument("--keep-intermediate", action="store_true",
                    help="Keep per-take .bvh/.pose.json/.footlock.json/.cfg.")
    args = ap.parse_args(argv)

    if not args.manifest.exists():
        sys.exit(f"manifest not found: {args.manifest}")
    manifest_dir = args.manifest.parent.resolve()
    rows = _load_manifest(args.manifest)

    common = []
    if args.no_foot_contact:
        common.append("--no-foot-contact")
    if args.wham_python:
        common += ["--wham-python", args.wham_python]
    if args.wham_dir:
        common += ["--wham-dir", args.wham_dir]
    if args.base_blend:
        common += ["--base-blend", args.base_blend]
    if args.blender_exe:
        common += ["--blender-exe", args.blender_exe]
    if args.keep_intermediate:
        common.append("--keep-intermediate")
    if args.dry_run:
        common.append("--dry-run")

    report_path = (args.report or manifest_dir / "batch_report.json").resolve()
    batch_started = datetime.datetime.now().isoformat(timespec="seconds")
    batch_t0 = time.time()
    results = []

    print(f"[batch] {len(rows)} take(s) from {args.manifest}")
    print(f"[batch] report -> {report_path}")
    print(f"[batch] started {batch_started}")
    print()

    stopped_early = False
    for i, row in enumerate(rows, start=1):
        out_path = _resolve_out_path(row, manifest_dir)
        log_path = out_path.with_suffix(".log")
        cmd = _build_cmd(row, out_path, common)

        label = row["video"] or row["from_pose_json"]
        print(f"[batch] {i}/{len(rows)}  char={row['character']:<10s}  "
              f"-> {out_path.name}  (manifest row {row['_row']})")
        print(f"  src: {label}")
        printable = " ".join(str(c) for c in cmd)
        print(f"  $ {printable}")

        if args.dry_run:
            results.append({
                "row": row["_row"], "video": row["video"],
                "from_pose_json": row["from_pose_json"],
                "character": row["character"], "out": str(out_path),
                "status": "dry_run", "duration_s": 0.0,
            })
            print()
            continue

        t0 = time.time()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as logf:
            logf.write(f"# batch row {row['_row']}: {row}\n")
            logf.write(f"# $ {printable}\n\n")
            logf.flush()
            try:
                proc = subprocess.run(cmd, cwd=str(REPO_ROOT),
                                      stdout=logf, stderr=subprocess.STDOUT,
                                      text=True)
                rc = proc.returncode
            except Exception as e:  # noqa: BLE001
                logf.write(f"\n[batch] EXCEPTION {type(e).__name__}: {e}\n")
                rc = 99
        dur = time.time() - t0

        report_file = out_path.with_suffix(".report.json")
        per_take = None
        if report_file.exists():
            try:
                per_take = json.loads(report_file.read_text())
            except Exception:  # noqa: BLE001
                pass

        result = {
            "row": row["_row"], "video": row["video"],
            "from_pose_json": row["from_pose_json"],
            "character": row["character"], "out": str(out_path),
            "log": str(log_path),
            "report": str(report_file) if report_file.exists() else None,
            "status": "ok" if rc == 0 else "fail",
            "returncode": rc, "duration_s": round(dur, 1),
        }
        if per_take:
            bvh_stage = per_take.get("stages", {}).get("bvh", {})
            fc_stage = per_take.get("stages", {}).get("foot_contact", {})
            result["frames"] = bvh_stage.get("frames")
            result["fps"] = bvh_stage.get("fps")
            result["foot_contact_intervals"] = (
                len(fc_stage.get("intervals", []))
                if isinstance(fc_stage.get("intervals"), list) else None
            )
        results.append(result)

        status_lbl = "OK  " if rc == 0 else f"FAIL rc={rc}"
        extra = ""
        if per_take and "frames" in result:
            extra = (f"  {result['frames']} frames  "
                     f"{result.get('foot_contact_intervals', '-')} plants")
        print(f"  {status_lbl}  ({dur:.1f}s){extra}  log -> {log_path.name}")
        print()

        if rc != 0 and not args.continue_on_error:
            print(f"[batch] stopping after row {row['_row']} "
                  f"(pass --continue-on-error for unattended runs)")
            stopped_early = True
            break

    total_dur = time.time() - batch_t0
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_fail = sum(1 for r in results if r["status"] == "fail")
    n_dry = sum(1 for r in results if r["status"] == "dry_run")

    batch_report = {
        "tool": "batch", "manifest": str(args.manifest),
        "started_at": batch_started, "duration_s": round(total_dur, 1),
        "total": len(rows), "processed": len(results),
        "ok": n_ok, "fail": n_fail, "dry_run": n_dry,
        "stopped_early": stopped_early,
        "continue_on_error": args.continue_on_error,
        "takes": results,
    }
    report_path.write_text(json.dumps(batch_report, indent=2))

    print(f"[batch] DONE  ok={n_ok}  fail={n_fail}  dry={n_dry}  "
          f"({total_dur:.1f}s total) -> {report_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
