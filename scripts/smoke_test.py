"""Fixture-based smoke test for the v5.x take format and solver output.

What this validates:
    1. The repo contains the expected committed sample takes.
    2. Those JSONs parse and have the v5.x schema
       (format=melodiccap_rtm_v1, processing_settings.skeleton_solver).
    3. Per-frame bone lengths are consistent with the solver's claimed
       calibrated lengths (the solver enforces these per-frame; if a JSON
       claims to be solver-output but bone lengths drift, something is
       wrong with the file or the format has changed).
    4. Hip Z is stable across a known A-pose take (sanity check that the
       take is what its name implies — a static calibration pose).

What this does NOT do:
    - Run the cameras, RTMPose, or stereo triangulation. Those need
      hardware. See MelodicCapRTM/README for the full live-capture flow.
    - Validate the Blender retarget. That needs Blender installed.

Exit 0 on all pass, 1 on any failure. No external deps — pure stdlib +
numpy (numpy is in requirements.txt).

Run from the repo root:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TAKES_DIR = REPO_ROOT / "MelodicCapRTM" / "takes"

# COCO body-17 indices used to derive bones for invariant checks.
L_SHOULDER, R_SHOULDER = "5", "6"
L_ELBOW, R_ELBOW = "7", "8"
L_WRIST, R_WRIST = "9", "10"
L_HIP, R_HIP = "11", "12"

# (parent_idx, child_idx, solver_meta_key, max_per_frame_drift_m)
BONE_CHECKS = [
    (L_SHOULDER, L_ELBOW, "upper_arm_l", 1e-4),
    (R_SHOULDER, R_ELBOW, "upper_arm_r", 1e-4),
    (L_ELBOW, L_WRIST, "forearm_l", 1e-4),
    (R_ELBOW, R_WRIST, "forearm_r", 1e-4),
]


# ANSI-free output so it works on every terminal including Windows cmd.
def ok(msg: str) -> None:
    print(f"  PASS  {msg}")


def fail(msg: str) -> None:
    print(f"  FAIL  {msg}")


def dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def check_take(path: Path, expect_static_apose: bool) -> list[str]:
    """Run all invariants for one take. Returns list of failure messages."""
    errors: list[str] = []

    if not path.exists():
        return [f"missing fixture: {path.relative_to(REPO_ROOT)}"]

    with path.open() as f:
        data = json.load(f)

    # 1. Schema
    fmt = data.get("format")
    if fmt != "melodiccap_rtm_v1":
        errors.append(f"format mismatch: got {fmt!r}, expected 'melodiccap_rtm_v1'")

    settings = data.get("processing_settings", {})
    solver = settings.get("skeleton_solver", {})
    if not solver.get("enabled"):
        errors.append("processing_settings.skeleton_solver.enabled is not true")
        return errors  # downstream checks assume solver ran

    bone_lengths = solver.get("bone_lengths", {})
    if not bone_lengths:
        errors.append("processing_settings.skeleton_solver.bone_lengths missing/empty")
        return errors

    frames = data.get("frames", [])
    if not frames:
        errors.append("no frames in take")
        return errors

    # 2. Bone-length consistency per frame
    for parent, child, meta_key, tol in BONE_CHECKS:
        claimed = bone_lengths.get(meta_key)
        if claimed is None:
            continue  # solver chose not to calibrate this bone (e.g. virtual bone)
        for i, fr in enumerate(frames):
            lm = fr.get("landmarks_3d", {})
            if parent not in lm or child not in lm:
                continue
            observed = dist(lm[parent][:3], lm[child][:3])
            if abs(observed - claimed) > tol:
                errors.append(
                    f"bone {meta_key} drifts at frame {i}: "
                    f"observed {observed:.5f}m vs claimed {claimed:.5f}m "
                    f"(tol {tol}m)"
                )
                break  # one report per bone is enough

    # 3. Hip Z stability on the A-pose take
    if expect_static_apose:
        hip_zs = []
        for fr in frames:
            lm = fr.get("landmarks_3d", {})
            if L_HIP in lm and R_HIP in lm:
                hip_zs.append((lm[L_HIP][2] + lm[R_HIP][2]) / 2.0)
        if hip_zs:
            hip_range = max(hip_zs) - min(hip_zs)
            # An A-pose hold should not vary by more than 3cm
            if hip_range > 0.03:
                errors.append(
                    f"A-pose hip Z range {hip_range * 100:.1f}cm exceeds 3cm "
                    f"(min {min(hip_zs):.3f}, max {max(hip_zs):.3f})"
                )

    return errors


FIXTURES = [
    # (path, is_static_apose)
    (TAKES_DIR / "take_20260430_204607.json", True),
    (TAKES_DIR / "take_20260504_213341.json", True),
]


def main() -> int:
    print("MelodicCap smoke test")
    print(f"Repo root: {REPO_ROOT}")
    print()

    total_failures = 0

    for path, is_apose in FIXTURES:
        rel = path.relative_to(REPO_ROOT) if path.exists() else path.name
        print(f"Checking {rel}:")
        errors = check_take(path, expect_static_apose=is_apose)
        if not errors:
            ok("schema, bone-length consistency, hip stability")
        else:
            for e in errors:
                fail(e)
            total_failures += len(errors)
        print()

    if total_failures:
        print(f"SMOKE TEST FAILED — {total_failures} issue(s)")
        return 1
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
