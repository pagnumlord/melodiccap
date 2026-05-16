"""Path A entry point: phone MP4 -> melodiccap_mono_v1 pose JSON, via WHAM.

This is a thin orchestration layer. The actual SMPL inference runs
inside the WHAM repo's demo.py via subprocess; we don't import WHAM's
modules directly because WHAM's deps (a specific torch + CUDA combo)
collide with everything else and pinning two stacks is a nightmare.
Keep WHAM in its own conda env and run this script from inside that
env.

Usage (Windows PowerShell):

    conda activate wham
    set WHAM_DIR=C:\\Users\\you\\src\\WHAM
    python -m MelodicCapMono.wham.video2pose `
        C:\\path\\to\\guitar.mp4 `
        C:\\path\\to\\guitar.pose.json `
        --character jax

Usage (macOS / Linux):

    conda activate wham
    export WHAM_DIR=/abs/path/to/WHAM
    python -m MelodicCapMono.wham.video2pose \\
        guitar.mp4 guitar.pose.json --character jax

If WHAM's pkl format has shifted between releases and the converter
can't read it, run:

    python -m MelodicCapMono.wham.video2pose --inspect-pkl <some>.pkl

to dump the dict structure. The fix is usually a one-line key rename
in wham_to_pose_json.py.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import wham_to_pose_json


def _find_wham_dir(cli_override: str | None) -> Path:
    candidate = cli_override or os.environ.get("WHAM_DIR")
    if not candidate:
        sys.exit(
            "ERROR: WHAM_DIR is not set and --wham-dir was not given.\n\n"
            "Clone WHAM, then point WHAM_DIR at it:\n"
            "  git clone https://github.com/yohanshin/WHAM.git\n"
            "  conda env create -f WHAM/environment.yml\n"
            "  conda activate wham\n"
            "  cd WHAM && bash fetch_demo_data.sh    # downloads ~3 GB of checkpoints\n"
            "  set WHAM_DIR=C:\\path\\to\\WHAM        # Windows\n"
            "  export WHAM_DIR=/abs/path/to/WHAM     # macOS / Linux\n"
        )
    p = Path(candidate).expanduser().resolve()
    if not p.exists():
        sys.exit(f"ERROR: WHAM_DIR={p} does not exist.")
    if not (p / "demo.py").exists():
        sys.exit(
            f"ERROR: WHAM_DIR={p} does not contain demo.py. "
            f"Is this actually a WHAM checkout?"
        )
    return p


def _run_wham(wham_dir: Path, video: Path, output_dir: Path) -> Path:
    """Invoke WHAM's demo.py on the video. Return the produced .pkl path.

    Raises SystemExit if WHAM exits non-zero or produces no .pkl.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "demo.py",
        "--video", str(video.resolve()),
        "--output_pth", str(output_dir.resolve()),
        "--save_pkl",
    ]
    print(f"[video2pose] Running WHAM:\n  cwd: {wham_dir}\n  {' '.join(cmd)}\n")
    proc = subprocess.run(cmd, cwd=str(wham_dir))
    if proc.returncode != 0:
        sys.exit(
            f"WHAM exited with code {proc.returncode}. "
            f"See its stderr above; common causes:\n"
            f"  - GPU OOM (try a shorter video or lower resolution)\n"
            f"  - missing checkpoints (re-run fetch_demo_data.sh)\n"
            f"  - no person detected in the video"
        )
    candidates = sorted(output_dir.rglob("*.pkl"))
    if not candidates:
        sys.exit(
            f"WHAM exited 0 but produced no .pkl in {output_dir}. "
            f"Check WHAM's stdout above — it may have written results "
            f"to a different path. Pass --keep-intermediate to inspect."
        )
    if len(candidates) > 1:
        print(f"[video2pose] Multiple .pkl files found, picking the first: "
              f"{candidates[0]}")
    return candidates[0]


def _load_wham_pkl(pkl_path: Path) -> Any:
    """Deserialize WHAM's output file.

    WHAM's demo.py writes results with ``joblib.dump``. joblib wraps
    numpy arrays in its own persistence format (and may compress), so a
    plain ``pickle.load`` fails with
    ``UnpicklingError: invalid load key``. ``joblib.load`` transparently
    reads both joblib-format files and plain pickles, so prefer it; fall
    back to ``pickle`` only if joblib can't be imported (it always can in
    the WHAM env, since WHAM itself used joblib to write the file).
    """
    try:
        import joblib
    except ImportError:
        with pkl_path.open("rb") as f:
            return pickle.load(f)
    return joblib.load(str(pkl_path))


def _inspect_pkl(pkl_path: Path) -> int:
    """Print the structure of a WHAM pkl. Used to debug schema drift."""
    data = _load_wham_pkl(pkl_path)

    def describe(obj, depth=0, prefix=""):
        pad = "  " * depth
        if isinstance(obj, dict):
            print(f"{pad}{prefix}dict with {len(obj)} keys:")
            for k, v in list(obj.items())[:25]:
                describe(v, depth + 1, prefix=f"[{k!r}] ")
            if len(obj) > 25:
                print(f"{pad}  ... and {len(obj) - 25} more keys")
        elif isinstance(obj, list):
            print(f"{pad}{prefix}list of length {len(obj)}")
            if obj:
                describe(obj[0], depth + 1, prefix="[0] ")
        else:
            try:
                import numpy as np
                if isinstance(obj, np.ndarray):
                    print(f"{pad}{prefix}ndarray shape={obj.shape} dtype={obj.dtype}")
                    return
            except ImportError:
                pass
            text = repr(obj)
            print(f"{pad}{prefix}{type(obj).__name__}: "
                  f"{text[:80]}{'...' if len(text) > 80 else ''}")

    describe(data)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="MelodicCapMono.wham.video2pose",
        description="Run WHAM on a video, convert its output to a melodiccap_mono_v1 pose JSON.",
    )
    ap.add_argument("video", nargs="?", type=Path,
                    help="Input video file (.mp4 or anything OpenCV reads).")
    ap.add_argument("output_json", nargs="?", type=Path,
                    help="Output pose JSON path.")
    ap.add_argument("--character", default=None,
                    help="Character name (embedded in JSON for Blender auto-resolve).")
    ap.add_argument("--fps", type=float, default=None,
                    help="Override detected video FPS.")
    ap.add_argument("--wham-dir", default=None,
                    help="Path to WHAM clone. Overrides WHAM_DIR env var.")
    ap.add_argument("--keep-intermediate", action="store_true",
                    help="Keep WHAM's pkl output for debugging.")
    ap.add_argument("--inspect-pkl", type=Path, default=None,
                    help="Skip WHAM, just print the structure of an existing .pkl.")
    ap.add_argument("--from-pkl", type=Path, default=None,
                    help="Skip WHAM; convert an existing WHAM .pkl straight to "
                         "pose JSON. Usage: --from-pkl <pkl> <output.json>")
    args = ap.parse_args(argv)

    if args.inspect_pkl is not None:
        if not args.inspect_pkl.exists():
            sys.exit(f"ERROR: {args.inspect_pkl} does not exist.")
        return _inspect_pkl(args.inspect_pkl)

    if args.from_pkl is not None:
        if not args.from_pkl.exists():
            sys.exit(f"ERROR: --from-pkl file not found: {args.from_pkl}")
        out_json = args.output_json or args.video
        if out_json is None:
            ap.error("--from-pkl requires an output path: "
                     "--from-pkl <pkl> <output.json>")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        print(f"[video2pose] Reading WHAM output: {args.from_pkl}")
        wham_data = _load_wham_pkl(args.from_pkl)
        print(f"[video2pose] Converting to pose JSON: {out_json}")
        pose = wham_to_pose_json.convert(
            wham_data,
            character=args.character,
            fps_override=args.fps,
            source_video=args.from_pkl.name,
        )
        with out_json.open("w") as f:
            json.dump(pose, f, indent=2)
        print(f"[video2pose] DONE. {len(pose['frames'])} frames -> {out_json}")
        return 0

    if args.video is None or args.output_json is None:
        ap.error("video and output_json are required "
                 "(unless --inspect-pkl / --from-pkl).")

    if not args.video.exists():
        sys.exit(f"ERROR: input video not found: {args.video}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    wham_dir = _find_wham_dir(args.wham_dir)

    intermediate_dir = args.output_json.parent / f".{args.output_json.stem}.wham"
    pkl_path = _run_wham(wham_dir, args.video, intermediate_dir)

    print(f"[video2pose] Reading WHAM output: {pkl_path}")
    wham_data = _load_wham_pkl(pkl_path)

    print(f"[video2pose] Converting to pose JSON: {args.output_json}")
    pose = wham_to_pose_json.convert(
        wham_data,
        character=args.character,
        fps_override=args.fps,
        source_video=args.video.name,
    )
    with args.output_json.open("w") as f:
        json.dump(pose, f, indent=2)

    if not args.keep_intermediate:
        shutil.rmtree(intermediate_dir, ignore_errors=True)

    print(f"[video2pose] DONE. {len(pose['frames'])} frames -> "
          f"{args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
