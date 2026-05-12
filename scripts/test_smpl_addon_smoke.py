"""Smoke test for the melodiccap_mono_v1 pose JSON schema.

Validates a pose JSON against the schema defined in
`MelodicCapMono/SCHEMA.md`. Runs as pure stdlib — no Blender, no model
installs, no extra deps. Lets a contributor verify the JSON contract is
intact in one command after clone+install.

Usage:
    python scripts/test_smpl_addon_smoke.py [path1.pose.json [path2 ...]]

With no arguments, runs against the committed fixtures:
    MelodicCapMono/fixtures/rest.pose.json
    MelodicCapMono/fixtures/wave.pose.json

Exits 0 if every file passes, 1 on any failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_FIXTURES = [
    REPO_ROOT / "MelodicCapMono" / "fixtures" / "rest.pose.json",
    REPO_ROOT / "MelodicCapMono" / "fixtures" / "wave.pose.json",
]

EXPECTED_FORMAT = "melodiccap_mono_v1"
EXPECTED_BETAS_LEN = 10
EXPECTED_POSE_LEN = 72  # 24 SMPL joints x 3 axis-angle components
EXPECTED_TRANS_LEN = 3
EXPECTED_GLOBAL_ORIENT_LEN = 3


def ok(msg: str) -> None:
    print(f"  PASS  {msg}")


def fail(msg: str) -> None:
    print(f"  FAIL  {msg}")


def check_top_level(data: dict) -> list[str]:
    errors: list[str] = []

    if data.get("format") != EXPECTED_FORMAT:
        errors.append(
            f"format mismatch: got {data.get('format')!r}, "
            f"expected {EXPECTED_FORMAT!r}"
        )

    for key, kind in (
        ("source_model", str),
        ("source_video", str),
        ("fps", (int, float)),
        ("smpl_betas", list),
        ("frames", list),
    ):
        if key not in data:
            errors.append(f"missing required top-level key: {key!r}")
        elif not isinstance(data[key], kind):
            errors.append(
                f"top-level {key!r} has wrong type: "
                f"got {type(data[key]).__name__}, "
                f"expected {kind if isinstance(kind, type) else 'number'}"
            )

    if isinstance(data.get("smpl_betas"), list):
        if len(data["smpl_betas"]) != EXPECTED_BETAS_LEN:
            errors.append(
                f"smpl_betas length {len(data['smpl_betas'])}, "
                f"expected {EXPECTED_BETAS_LEN}"
            )

    fps = data.get("fps")
    if isinstance(fps, (int, float)) and fps <= 0:
        errors.append(f"fps must be > 0, got {fps}")

    return errors


def check_frames(frames: list) -> list[str]:
    errors: list[str] = []
    if not frames:
        errors.append("frames array is empty")
        return errors

    seen_frame_indices: set[int] = set()
    last_timestamp: float | None = None

    for i, fr in enumerate(frames):
        if not isinstance(fr, dict):
            errors.append(f"frame[{i}] is not a dict")
            continue

        # required keys per frame
        for k, kind in (
            ("frame", int),
            ("timestamp", (int, float)),
            ("smpl_pose", list),
            ("smpl_trans", list),
        ):
            if k not in fr:
                errors.append(f"frame[{i}] missing {k!r}")
            elif not isinstance(fr[k], kind):
                errors.append(
                    f"frame[{i}].{k} wrong type: "
                    f"got {type(fr[k]).__name__}"
                )

        # pose length
        pose = fr.get("smpl_pose")
        if isinstance(pose, list) and len(pose) != EXPECTED_POSE_LEN:
            errors.append(
                f"frame[{i}].smpl_pose length {len(pose)}, "
                f"expected {EXPECTED_POSE_LEN}"
            )

        # trans length
        trans = fr.get("smpl_trans")
        if isinstance(trans, list) and len(trans) != EXPECTED_TRANS_LEN:
            errors.append(
                f"frame[{i}].smpl_trans length {len(trans)}, "
                f"expected {EXPECTED_TRANS_LEN}"
            )

        # optional global_orient shape
        go = fr.get("smpl_global_orient")
        if go is not None:
            if not isinstance(go, list) or len(go) != EXPECTED_GLOBAL_ORIENT_LEN:
                errors.append(
                    f"frame[{i}].smpl_global_orient must be a list of "
                    f"{EXPECTED_GLOBAL_ORIENT_LEN}, got {go!r}"
                )

        # frame index sanity
        idx = fr.get("frame")
        if isinstance(idx, int):
            if idx in seen_frame_indices:
                errors.append(f"duplicate frame index {idx}")
            seen_frame_indices.add(idx)

        # timestamp monotonic
        ts = fr.get("timestamp")
        if isinstance(ts, (int, float)):
            if last_timestamp is not None and ts < last_timestamp - 1e-6:
                errors.append(
                    f"frame[{i}].timestamp went backwards: "
                    f"{last_timestamp} -> {ts}"
                )
            last_timestamp = ts

    return errors


def validate_file(path: Path) -> list[str]:
    if not path.exists():
        return [f"file not found: {path}"]

    try:
        with path.open("r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [f"JSON parse error: {e}"]
    except Exception as e:
        return [f"failed to open: {e}"]

    if not isinstance(data, dict):
        return ["top-level JSON is not an object"]

    errors = check_top_level(data)
    if isinstance(data.get("frames"), list):
        errors.extend(check_frames(data["frames"]))
    return errors


def main() -> int:
    targets: list[Path] = (
        [Path(p).resolve() for p in sys.argv[1:]]
        if len(sys.argv) > 1
        else DEFAULT_FIXTURES
    )

    print("MelodicCap SMPL pose JSON smoke test")
    total_failures = 0
    for path in targets:
        rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        print(f"\nChecking {rel}:")
        errors = validate_file(path)
        if errors:
            for e in errors:
                fail(e)
            total_failures += len(errors)
        else:
            ok("schema valid")

    print()
    if total_failures:
        print(f"SMPL POSE SMOKE TEST FAILED — {total_failures} issue(s)")
        return 1
    print("SMPL POSE SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
