"""Smoke test for the WHAM -> melodiccap_mono_v1 converter.

Builds synthetic WHAM-style dicts in memory (no actual WHAM install,
no .pkl on disk) and runs them through MelodicCapMono.wham.wham_to_pose_json.convert.
Validates the output against the same schema rules used by
test_smpl_addon_smoke.py.

Exercises every shape variant the converter is supposed to accept:
    - single-person dict OR {person_id: dict} OR list[dict]
    - pose as (N, 24, 3, 3) rotation matrices OR (N, 24, 3) AA OR (N, 72) flat AA
    - betas as (10,) constant OR (N, 10) per-frame
    - global_orient present (matrices OR axis-angle) OR absent
    - frame_ids present OR absent

Usage:
    python scripts/test_wham_converter_smoke.py

Requires numpy (the converter imports it). No torch, no WHAM, no Blender.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from MelodicCapMono.wham import wham_to_pose_json  # noqa: E402


SMPL_JOINTS = 24
EXPECTED_FORMAT = "melodiccap_mono_v1"


# ---------------------------------------------------------------------------
# Synthetic WHAM data builders
# ---------------------------------------------------------------------------

def _axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Tiny Rodrigues for verifying round-trip."""
    angle = float(np.linalg.norm(aa))
    if angle < 1e-9:
        return np.eye(3)
    axis = aa / angle
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def make_person(
    n_frames: int,
    *,
    pose_layout: str = "matrix",       # "matrix" | "axis_angle" | "flat"
    betas_layout: str = "constant",    # "constant" | "per_frame"
    include_global_orient: str = "none",  # "none" | "matrix" | "axis_angle"
    include_frame_ids: bool = True,
    fps: float = 30.0,
) -> dict:
    """Build a synthetic WHAM person record."""
    rng = np.random.default_rng(seed=42)

    # Base axis-angle: gentle random rotations on every joint.
    pose_aa = rng.normal(0, 0.05, size=(n_frames, SMPL_JOINTS, 3))

    if pose_layout == "matrix":
        mats = np.zeros((n_frames, SMPL_JOINTS, 3, 3), dtype=np.float64)
        for f in range(n_frames):
            for j in range(SMPL_JOINTS):
                mats[f, j] = _axis_angle_to_matrix(pose_aa[f, j])
        pose_field = mats
    elif pose_layout == "axis_angle":
        pose_field = pose_aa
    elif pose_layout == "flat":
        pose_field = pose_aa.reshape(n_frames, SMPL_JOINTS * 3)
    else:
        raise ValueError(pose_layout)

    if betas_layout == "constant":
        betas = rng.normal(0, 1.0, size=10)
    elif betas_layout == "per_frame":
        betas = rng.normal(0, 1.0, size=(n_frames, 10))
    else:
        raise ValueError(betas_layout)

    person = {
        "pose": pose_field,
        "trans": rng.normal(0, 0.1, size=(n_frames, 3)),
        "betas": betas,
        "fps": fps,
    }
    if include_frame_ids:
        person["frame_ids"] = np.arange(n_frames, dtype=np.int64)
    if include_global_orient == "matrix":
        go_aa = rng.normal(0, 0.1, size=(n_frames, 3))
        go_mats = np.zeros((n_frames, 3, 3))
        for f in range(n_frames):
            go_mats[f] = _axis_angle_to_matrix(go_aa[f])
        person["global_orient"] = go_mats
    elif include_global_orient == "axis_angle":
        person["global_orient"] = rng.normal(0, 0.1, size=(n_frames, 3))
    return person


# ---------------------------------------------------------------------------
# Schema validator (pared-down copy of test_smpl_addon_smoke.py rules)
# ---------------------------------------------------------------------------

def validate_schema(pose: dict) -> list[str]:
    errors: list[str] = []
    if pose.get("format") != EXPECTED_FORMAT:
        errors.append(f"format != {EXPECTED_FORMAT}: {pose.get('format')!r}")
    for k, t in (
        ("source_model", str),
        ("source_video", str),
        ("fps", (int, float)),
        ("smpl_betas", list),
        ("frames", list),
    ):
        if k not in pose:
            errors.append(f"missing top-level {k}")
        elif not isinstance(pose[k], t):
            errors.append(f"{k} wrong type: {type(pose[k]).__name__}")
    if isinstance(pose.get("smpl_betas"), list) and len(pose["smpl_betas"]) != 10:
        errors.append(f"smpl_betas length {len(pose['smpl_betas'])} != 10")
    fps = pose.get("fps")
    if isinstance(fps, (int, float)) and fps <= 0:
        errors.append(f"fps must be > 0, got {fps}")

    last_t = None
    for i, fr in enumerate(pose.get("frames", [])):
        if not isinstance(fr, dict):
            errors.append(f"frame[{i}] not a dict")
            continue
        for k, t in (
            ("frame", int),
            ("timestamp", (int, float)),
            ("smpl_pose", list),
            ("smpl_trans", list),
        ):
            if k not in fr:
                errors.append(f"frame[{i}] missing {k}")
            elif not isinstance(fr[k], t):
                errors.append(f"frame[{i}].{k} wrong type")
        sp = fr.get("smpl_pose")
        if isinstance(sp, list) and len(sp) != 72:
            errors.append(f"frame[{i}].smpl_pose len {len(sp)} != 72")
        st = fr.get("smpl_trans")
        if isinstance(st, list) and len(st) != 3:
            errors.append(f"frame[{i}].smpl_trans len {len(st)} != 3")
        go = fr.get("smpl_global_orient")
        if go is not None and (not isinstance(go, list) or len(go) != 3):
            errors.append(f"frame[{i}].smpl_global_orient bad: {go!r}")
        ts = fr.get("timestamp")
        if isinstance(ts, (int, float)):
            if last_t is not None and ts < last_t - 1e-6:
                errors.append(f"frame[{i}].timestamp went backwards")
            last_t = ts
    return errors


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def case(name: str, wham_data, *, expect_global=False, expect_frames=10):
    print(f"\nCase: {name}")
    pose = wham_to_pose_json.convert(
        wham_data,
        character="jax",
        fps_override=None,
        source_video="synthetic.mp4",
    )
    errs = validate_schema(pose)
    if errs:
        for e in errs:
            print(f"  FAIL  {e}")
        return False
    if len(pose["frames"]) != expect_frames:
        print(f"  FAIL  expected {expect_frames} frames, got {len(pose['frames'])}")
        return False
    if expect_global and "smpl_global_orient" not in pose["frames"][0]:
        print("  FAIL  expected smpl_global_orient on frames")
        return False
    if not expect_global and "smpl_global_orient" in pose["frames"][0]:
        print("  FAIL  unexpected smpl_global_orient on frames")
        return False
    print(f"  PASS  ({len(pose['frames'])} frames, character={pose['character']})")
    return True


def main() -> int:
    print("MelodicCap WHAM converter smoke test")

    results: list[bool] = []

    # 1. Single-person dict, matrix pose
    results.append(case(
        "single-person dict + (N,24,3,3) matrices + constant betas + frame_ids",
        make_person(10, pose_layout="matrix"),
    ))

    # 2. Multi-person dict (we should pick the longer track, person 1)
    multi = {
        0: make_person(3, pose_layout="matrix"),
        1: make_person(10, pose_layout="matrix"),
    }
    results.append(case("multi-person dict, picks longest track", multi))

    # 3. Flat axis-angle (N, 72)
    results.append(case(
        "flat (N, 72) axis-angle pose",
        make_person(10, pose_layout="flat"),
    ))

    # 4. Per-joint axis-angle (N, 24, 3)
    results.append(case(
        "(N, 24, 3) axis-angle pose",
        make_person(10, pose_layout="axis_angle"),
    ))

    # 5. Per-frame betas
    results.append(case(
        "per-frame betas (N, 10)",
        make_person(10, betas_layout="per_frame"),
    ))

    # 6. global_orient as matrices (now folded into smpl_pose[0:3] —
    #    the converter emits exactly one root channel)
    results.append(case(
        "global_orient matrices (folded into pose root)",
        make_person(10, include_global_orient="matrix"),
    ))

    # 7. global_orient as axis-angle (folded as above)
    results.append(case(
        "global_orient axis-angle (folded into pose root)",
        make_person(10, include_global_orient="axis_angle"),
    ))

    # 8. No frame_ids -> converter falls back to range(N)
    results.append(case(
        "no frame_ids",
        make_person(10, include_frame_ids=False),
    ))

    # 9. results dict wrapper: {"results": {0: person}}
    p = make_person(10)
    results.append(case(
        "wrapped under 'results' key",
        {"results": {0: p}},
    ))

    # 10. List of persons
    results.append(case(
        "list of persons",
        [make_person(3), make_person(10)],
    ))

    # 11. Round-trip math sanity: matrix -> AA -> matrix should preserve rotation
    aa_in = np.array([0.1, 0.2, 0.3])
    R_in = _axis_angle_to_matrix(aa_in)
    aa_out = wham_to_pose_json._rotation_matrix_to_axis_angle(R_in)
    err = float(np.max(np.abs(aa_in - aa_out)))
    print(f"\nCase: rotation matrix <-> axis-angle round trip")
    if err < 1e-6:
        print(f"  PASS  max error {err:.2e}")
        results.append(True)
    else:
        print(f"  FAIL  max error {err:.2e} too large")
        results.append(False)

    # ------------------------------------------------------------------
    # World-frame preference + gravity leveling (the backward-lean fix)
    # ------------------------------------------------------------------

    def _root_up(pose_json, frame=0):
        """Body-up unit vector from a converted frame's root rotation."""
        aa = np.asarray(pose_json["frames"][frame]["smpl_pose"][0:3])
        return _axis_angle_to_matrix(aa) @ np.array([0.0, 1.0, 0.0])

    def _tilted_person(n, tilt_deg, with_world=False):
        """Camera-frame person whose root is Rx(180 - tilt): an upright
        body seen by a camera pitched `tilt` degrees off level."""
        p = make_person(n, pose_layout="flat")
        pose = p["pose"].reshape(n, SMPL_JOINTS, 3)
        pose[:, 0] = [math.radians(180.0 - tilt_deg), 0.0, 0.0]
        p["pose"] = pose.reshape(n, SMPL_JOINTS * 3)
        if with_world:
            wpose = pose.copy()
            wpose[:, 0] = 0.0  # world keys: level, +Y up
            p["pose_world"] = wpose.reshape(n, SMPL_JOINTS * 3)
            p["trans_world"] = np.tile([0.0, 0.9, 0.0], (n, 1))
        return p

    # 12. pose_world preferred over tilted camera pose
    print("\nCase: prefers pose_world/trans_world over camera keys")
    pose = wham_to_pose_json.convert(
        _tilted_person(10, 20.0, with_world=True),
        character="jax", fps_override=None, source_video="synthetic.mp4")
    prov = pose.get("frame_provenance", {})
    up = _root_up(pose)
    ok = (prov.get("source_keys") == "pose_world/trans_world"
          and pose.get("coordinate_frame") == "y_up_world"
          and abs(up[1] - 1.0) < 1e-6
          and not validate_schema(pose))
    print(f"  {'PASS' if ok else 'FAIL'}  keys={prov.get('source_keys')} "
          f"up_y={up[1]:+.6f} frame={pose.get('coordinate_frame')}")
    results.append(ok)

    # 13. Camera-only + 20-deg tilted camera: leveling makes it upright
    print("\nCase: camera-frame fallback auto-levels a tilted camera")
    pose = wham_to_pose_json.convert(
        _tilted_person(10, 20.0, with_world=False),
        character="jax", fps_override=None, source_video="synthetic.mp4")
    prov = pose.get("frame_provenance", {})
    up = _root_up(pose)
    corr = prov.get("level_correction_deg") or 0.0
    ok = (prov.get("source_keys") == "pose/trans"
          and pose.get("coordinate_frame") == "y_up_world"
          and abs(up[1] - 1.0) < 1e-6
          and abs(corr - 160.0) < 1.0
          and not validate_schema(pose))
    print(f"  {'PASS' if ok else 'FAIL'}  correction={corr:.1f} deg "
          f"(expect ~160) up_y={up[1]:+.6f}")
    results.append(ok)

    # 14. --no-level on camera keys: legacy output, no y_up_world tag,
    #     resolver falls back to the historical 180-deg flip
    print("\nCase: level=False keeps legacy camera frame + 180 flip resolve")
    pose = wham_to_pose_json.convert(
        _tilted_person(10, 20.0, with_world=False),
        character="jax", fps_override=None, source_video="synthetic.mp4",
        level=False)
    from MelodicCapMono.wham._smpl_fk import resolve_global_rot_deg
    ok = ("coordinate_frame" not in pose
          and pose["frame_provenance"]["leveled"] is False
          and resolve_global_rot_deg(pose) == (180.0, 0.0, 0.0)
          and resolve_global_rot_deg({"coordinate_frame": "y_up_world"})
              == (0.0, 0.0, 0.0)
          and resolve_global_rot_deg(pose, (90.0, 0.0, 0.0))
              == (90.0, 0.0, 0.0)
          and not validate_schema(pose))
    print(f"  {'PASS' if ok else 'FAIL'}")
    results.append(ok)

    # 15. Recenter: frame-0 XZ at origin, height preserved
    print("\nCase: recenter zeroes frame-0 XZ, keeps Y")
    p = _tilted_person(10, 0.0, with_world=True)
    p["trans_world"] = p["trans_world"] + np.array([3.0, 0.0, -2.0])
    pose = wham_to_pose_json.convert(
        p, character="jax", fps_override=None, source_video="synthetic.mp4")
    t0 = pose["frames"][0]["smpl_trans"]
    ok = (abs(t0[0]) < 1e-9 and abs(t0[2]) < 1e-9
          and abs(t0[1] - 0.9) < 1e-6)
    print(f"  {'PASS' if ok else 'FAIL'}  t0={t0}")
    results.append(ok)

    # 16. world-only person record (no camera 'pose' at all)
    print("\nCase: person record with only pose_world keys")
    p = _tilted_person(10, 20.0, with_world=True)
    del p["pose"]
    del p["trans"]
    pose = wham_to_pose_json.convert(
        p, character="jax", fps_override=None, source_video="synthetic.mp4")
    ok = (pose["frame_provenance"]["source_keys"] == "pose_world/trans_world"
          and not validate_schema(pose))
    print(f"  {'PASS' if ok else 'FAIL'}")
    results.append(ok)

    print()
    if all(results):
        print(f"WHAM CONVERTER SMOKE TEST PASSED ({len(results)} cases)")
        return 0
    n_fail = sum(1 for r in results if not r)
    print(f"WHAM CONVERTER SMOKE TEST FAILED — {n_fail}/{len(results)} cases failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
