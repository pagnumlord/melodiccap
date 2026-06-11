"""Foot-contact detection from a melodiccap_mono_v1 pose JSON.

Output: per-frame foot_contact_l/r in [0, 1], a plant-interval list,
and a per-side floor calibration. Pure numpy + stdlib. Adapted from
MelodicCapRTM/blender_addon/melodiccap_rtm_addon.py:1418-1440
(median-of-N-frames floor calibration) and :2658-2761 (smoothstep
blend in/out, the production-tested foot-pinning state machine).

The headless retarget reads the sidecar and applies an IK lock pass
over Rokoko's bake (lifting foot_ik to the planted XYZ during plant
intervals, blending in/out at edges, toggling thigh_parent IK_FK).

Industry context: the canonical post-retarget pipeline for monocular
mocap is rotation transfer (Rokoko / MotionBuilder HumanIK / Unreal
IK Retargeter — all rest-relative quaternion math) FOLLOWED by a
contact-aware IK override on plant frames. WHAM is trajectory-based
and does not predict reliable ankle rotation; this module supplies
the missing contact channel.

CLI:
    python -m MelodicCapMono.orchestrate.footlock <in.pose.json> \\
        <out.footlock.json> [--config characters/<name>.json] [--report]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from MelodicCapMono.wham._smpl_fk import (
    HEIGHT_AXIS,
    SMPL_JOINT_NAMES,
    forward_kinematics,
    resolve_global_rot_deg,
)


# SMPL joint indices for the bones footlock tracks.
J_PELVIS = 0
J_LEFT_ANKLE = 7
J_RIGHT_ANKLE = 8
J_LEFT_TOE = 10
J_RIGHT_TOE = 11

EXPECTED_FORMAT = "melodiccap_mono_v1"

# Production-tested defaults from v5.x (MelodicCapRTM addon constants).
DEFAULTS = {
    "enabled": True,
    "floor_z": None,                     # None = auto-calibrate per side
    "ankle_z_threshold_m": 0.08,         # near_floor cutoff above floor
    "ankle_vel_threshold_mps": 0.15,     # slow cutoff for "planted"
    "calibration_frames": 20,            # median over first N frames
    "pin_blend_frames": 6,               # smoothstep in
    "unpin_blend_frames": 8,             # smoothstep out
    "pin_slide_rate": 0.15,              # walking-aware lock slide
    "hip_drift_unpin_m": 0.12,           # hip XY drift triggers slide
    "foot_ik_bones": {"L": "foot_ik.L", "R": "foot_ik.R"},
    "thigh_parent_bones": {"L": "thigh_parent.L", "R": "thigh_parent.R"},
}


def _merge_defaults(config):
    cfg = dict(DEFAULTS)
    if config and isinstance(config, dict) and "foot_contact" in config:
        for k, v in config["foot_contact"].items():
            cfg[k] = v
    return cfg


def _smoothstep(t):
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def detect_foot_contacts(pose_data, *, config=None):
    """Return per-frame contact + intervals + floor calibration.

    Output shape:
        {
          "fps": float,
          "floor_z": {"L": float, "R": float},
          "frames_in_contact": {"L": int, "R": int},
          "intervals": [{"side": "L"/"R", "frame_in": int, "frame_out": int}],
          "per_frame": {"foot_contact_l": [N floats], "foot_contact_r": [N floats]},
          "config_used": {...},
        }
    """
    if pose_data.get("format") != EXPECTED_FORMAT:
        raise ValueError(
            f"unexpected pose format: {pose_data.get('format')!r}"
        )
    cfg = _merge_defaults(config)
    fps = float(pose_data.get("fps", 30.0))
    n = len(pose_data["frames"])
    if n == 0:
        return {"fps": fps, "floor_z": {"L": 0.0, "R": 0.0},
                "frames_in_contact": {"L": 0, "R": 0},
                "intervals": [], "per_frame": {
                    "foot_contact_l": [], "foot_contact_r": []},
                "config_used": cfg}

    # Per-frame world joint positions, gravity-aligned +Y up. The root
    # correction resolves from the JSON's coordinate_frame tag: zero for
    # leveled converter output, the legacy 180-degree X flip for old
    # camera-frame JSONs (whose raw +Y points DOWN — without the flip,
    # a lifted foot reads as *closer* to the floor and the near-floor
    # test passes during swings).
    joints = forward_kinematics(
        pose_data, global_rot_deg=resolve_global_rot_deg(pose_data))
    h = HEIGHT_AXIS  # 1 — Y is the up axis in SMPL canonical

    ankle_l = joints[:, J_LEFT_ANKLE]
    ankle_r = joints[:, J_RIGHT_ANKLE]

    # Calibration: median of first N frames per side (median is robust to
    # an early step or motion in the calibration window, mirroring
    # v5.x's foot_z_offset block).
    cal_n = max(1, min(int(cfg["calibration_frames"]), n))
    if cfg["floor_z"] is not None:
        floor_l = floor_r = float(cfg["floor_z"])
    else:
        floor_l = float(np.median(ankle_l[:cal_n, h]))
        floor_r = float(np.median(ankle_r[:cal_n, h]))

    z_thresh = float(cfg["ankle_z_threshold_m"])
    vel_thresh = float(cfg["ankle_vel_threshold_mps"])
    pin_bf = max(1, int(cfg["pin_blend_frames"]))
    unpin_bf = max(1, int(cfg["unpin_blend_frames"]))
    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0

    # SMPL canonical: axis 1 (Y) is up; horizontal plane is axes 0 (X)
    # and 2 (Z). Ankle velocity uses those two axes.
    def _side_contacts(ankle, floor):
        contact = np.zeros(n, dtype=np.float64)
        prev_blend = 0.0
        for f in range(n):
            y = ankle[f, h]
            near = (y - floor) < z_thresh
            if f > 0:
                dx = ankle[f, 0] - ankle[f - 1, 0]
                dz = ankle[f, 2] - ankle[f - 1, 2]
                v_xy = math.sqrt(dx * dx + dz * dz) / dt
            else:
                v_xy = 0.0
            raw_plant = bool(near) and (v_xy < vel_thresh)

            if raw_plant:
                prev_blend = min(1.0, prev_blend + 1.0 / pin_bf)
            else:
                prev_blend = max(0.0, prev_blend - 1.0 / unpin_bf)
            contact[f] = _smoothstep(prev_blend)
        return contact

    c_l = _side_contacts(ankle_l, floor_l)
    c_r = _side_contacts(ankle_r, floor_r)

    def _intervals(side, contact):
        out = []
        in_plant = False
        start = -1
        for f in range(n):
            high = contact[f] > 0.5
            if high and not in_plant:
                in_plant, start = True, f
            elif (not high) and in_plant:
                out.append({"side": side, "frame_in": int(start),
                            "frame_out": int(f - 1)})
                in_plant, start = False, -1
        if in_plant:
            out.append({"side": side, "frame_in": int(start),
                        "frame_out": int(n - 1)})
        return out

    intervals = (_intervals("L", c_l) + _intervals("R", c_r))
    intervals.sort(key=lambda iv: (iv["frame_in"], iv["side"]))

    return {
        "fps": fps,
        "floor_z": {"L": floor_l, "R": floor_r},
        "frames_in_contact": {
            "L": int(np.sum(c_l > 0.5)),
            "R": int(np.sum(c_r > 0.5)),
        },
        "intervals": intervals,
        "per_frame": {
            "foot_contact_l": [float(x) for x in c_l.tolist()],
            "foot_contact_r": [float(x) for x in c_r.tolist()],
        },
        "config_used": {
            k: cfg[k] for k in (
                "floor_z", "ankle_z_threshold_m", "ankle_vel_threshold_mps",
                "calibration_frames", "pin_blend_frames",
                "unpin_blend_frames", "pin_slide_rate", "hip_drift_unpin_m",
                "foot_ik_bones", "thigh_parent_bones",
            )
        },
    }


def write_footlock_sidecar(pose_data, out_path, *, config=None):
    """Detect + write the sidecar JSON. Returns the summary dict."""
    result = detect_foot_contacts(pose_data, config=config)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="MelodicCapMono.orchestrate.footlock",
        description=(
            "Detect foot-contact plants from a melodiccap_mono_v1 pose JSON. "
            "Output sidecar feeds the headless retargeter's IK-lock pass."
        ),
    )
    ap.add_argument("input_json", type=Path)
    ap.add_argument("output_json", nargs="?", type=Path, default=None,
                    help="Output sidecar JSON path (omit with --report).")
    ap.add_argument("--config", type=Path, default=None,
                    help="characters/<name>.json (reads foot_contact block).")
    ap.add_argument("--report", action="store_true",
                    help="Print summary to stdout instead of writing.")
    args = ap.parse_args(argv)

    if not args.input_json.exists():
        sys.exit(f"ERROR: input not found: {args.input_json}")
    pose = json.loads(args.input_json.read_text(encoding="utf-8-sig"))
    cfg = None
    if args.config:
        if not args.config.exists():
            sys.exit(f"ERROR: config not found: {args.config}")
        cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))

    result = detect_foot_contacts(pose, config=cfg)

    if args.report or args.output_json is None:
        print(f"[footlock] floor_z L={result['floor_z']['L']:+.3f} "
              f"R={result['floor_z']['R']:+.3f}  "
              f"asym={abs(result['floor_z']['L'] - result['floor_z']['R']):.3f} m")
        print(f"[footlock] frames_in_contact L={result['frames_in_contact']['L']} "
              f"R={result['frames_in_contact']['R']} of {len(pose['frames'])} "
              f"({100.0 * result['frames_in_contact']['L'] / max(1, len(pose['frames'])):.0f}% / "
              f"{100.0 * result['frames_in_contact']['R'] / max(1, len(pose['frames'])):.0f}%)")
        print(f"[footlock] intervals ({len(result['intervals'])}):")
        for iv in result["intervals"]:
            print(f"  side={iv['side']} frames {iv['frame_in']:>4}-{iv['frame_out']:<4}")
    else:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2))
        print(f"[footlock] DONE intervals={len(result['intervals'])} "
              f"frames_in_contact L={result['frames_in_contact']['L']} "
              f"R={result['frames_in_contact']['R']} -> {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
