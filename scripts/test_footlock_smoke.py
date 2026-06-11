"""Smoke test for foot-contact detection (orchestrate/footlock.py).

Builds synthetic melodiccap_mono_v1 takes in memory and checks that
the detector finds plants where physics says they are:

  1. Standing take -> both feet planted the whole take.
  2. Left-leg lift mid-take -> left contact drops over the lift window,
     right stays planted; intervals split accordingly.
  3. Legacy-frame equivalence: the SAME motion expressed in the old
     camera frame (+Y down, no coordinate_frame tag) must produce the
     same contact pattern, proving resolve_global_rot_deg() feeds the
     FK the upright correction that footlock previously skipped.

Usage:
    python scripts/test_footlock_smoke.py

Requires numpy only. No torch, no WHAM, no Blender.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from MelodicCapMono.orchestrate.footlock import detect_foot_contacts  # noqa: E402


N_FRAMES = 40
LIFT_IN, LIFT_OUT = 12, 27   # frames where the left leg is raised
J_LEFT_HIP = 1


def make_take(*, lift_left=False, legacy_camera_frame=False):
    """Synthetic standing take, optionally lifting the left leg by
    flexing the left hip 0.6 rad over frames LIFT_IN..LIFT_OUT (raises
    the ankle ~14 cm — well past the 8 cm near-floor threshold)."""
    frames = []
    for f in range(N_FRAMES):
        pose = [0.0] * 72
        if lift_left and LIFT_IN <= f <= LIFT_OUT:
            pose[J_LEFT_HIP * 3] = -0.6  # hip flexion about X
        trans = [0.0, 0.9, 0.0]
        if legacy_camera_frame:
            # Express the same world motion in the old camera frame:
            # rotate everything by Rx(180) (Y down). Root rotation
            # becomes 180 deg about X; translation flips Y and Z.
            pose[0:3] = [math.pi, 0.0, 0.0]
            trans = [trans[0], -trans[1], -trans[2]]
        frames.append({
            "frame": f,
            "timestamp": f / 30.0,
            "smpl_pose": pose,
            "smpl_trans": trans,
        })
    take = {
        "format": "melodiccap_mono_v1",
        "source_model": "fixture",
        "source_video": "synthetic_footlock",
        "character": "jax",
        "fps": 30.0,
        "smpl_betas": [0.0] * 10,
        "frames": frames,
    }
    if not legacy_camera_frame:
        take["coordinate_frame"] = "y_up_world"
    return take


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    return ok


def main() -> int:
    print("MelodicCap footlock smoke test")
    results = []

    # 1. Standing take: both feet planted essentially the whole take.
    print("\nCase: standing take -> both feet planted")
    r = detect_foot_contacts(make_take())
    results.append(check(
        "both sides in contact >90% of frames",
        r["frames_in_contact"]["L"] > 0.9 * N_FRAMES
        and r["frames_in_contact"]["R"] > 0.9 * N_FRAMES,
        f"L={r['frames_in_contact']['L']} R={r['frames_in_contact']['R']}/{N_FRAMES}"))
    results.append(check(
        "one interval per side",
        len(r["intervals"]) == 2))
    results.append(check(
        "floor calibrated near ankle rest height",
        abs(r["floor_z"]["L"] - 0.0135) < 0.02
        and abs(r["floor_z"]["R"] - 0.0135) < 0.02,
        f"L={r['floor_z']['L']:+.3f} R={r['floor_z']['R']:+.3f}"))

    # 2. Left-leg lift: left contact drops over the lift, right holds.
    print("\nCase: left-leg lift mid-take")
    r = detect_foot_contacts(make_take(lift_left=True))
    c_l = r["per_frame"]["foot_contact_l"]
    iv_l = [iv for iv in r["intervals"] if iv["side"] == "L"]
    iv_r = [iv for iv in r["intervals"] if iv["side"] == "R"]
    results.append(check(
        "left contact < 0.5 during mid-lift",
        all(c < 0.5 for c in c_l[LIFT_IN + 6:LIFT_OUT]),
        f"mid values {['%.2f' % c for c in c_l[LIFT_IN + 6:LIFT_OUT:4]]}"))
    results.append(check(
        "left splits into 2 plant intervals", len(iv_l) == 2,
        f"{[(iv['frame_in'], iv['frame_out']) for iv in iv_l]}"))
    results.append(check(
        "right stays a single full plant", len(iv_r) == 1
        and iv_r[0]["frame_in"] <= 5
        and iv_r[0]["frame_out"] >= N_FRAMES - 2))

    # 3. Same motion in the legacy camera frame -> same detection.
    print("\nCase: legacy camera-frame JSON (no tag) matches y_up_world")
    r_legacy = detect_foot_contacts(make_take(lift_left=True,
                                              legacy_camera_frame=True))
    c_l_legacy = r_legacy["per_frame"]["foot_contact_l"]
    max_diff = float(np.max(np.abs(np.asarray(c_l) - np.asarray(c_l_legacy))))
    results.append(check(
        "per-frame left contact identical", max_diff < 1e-9,
        f"max diff {max_diff:.2e}"))
    results.append(check(
        "interval lists identical",
        r_legacy["intervals"] == r["intervals"]))

    print()
    if all(results):
        print(f"FOOTLOCK SMOKE TEST PASSED ({len(results)} checks)")
        return 0
    n_fail = sum(1 for x in results if not x)
    print(f"FOOTLOCK SMOKE TEST FAILED — {n_fail}/{len(results)} checks failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
