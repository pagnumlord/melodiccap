"""
Chain calibration: derive AC stereo calibration from AB + BC.

When Camera A and Camera C are far apart, the ChArUco board becomes too
small to detect at the distance needed for both to see it. But if B sits
between them, you can calibrate AB and BC directly (board is close enough
to B in both cases) and then compose the transforms to get AC.

Math:
  AB gives X_B = R_AB @ X_A + T_AB
  BC gives X_C = R_BC @ X_B + T_BC
  Substituting: X_C = (R_BC @ R_AB) @ X_A + (R_BC @ T_AB + T_BC)
  So R_AC = R_BC @ R_AB, T_AC = R_BC @ T_AB + T_BC

The A→B→C chain means AC shares Camera A's coordinate origin with AB,
so AC inherits AB's floor_z_offset for free — all three pairs then put
the skeleton in the same world frame.

Usage:
    python chain_calibration.py
    python chain_calibration.py --calibration-dir path/to/calibration
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def _load_pair(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return {
        'K1': np.array(data['K1']),
        'K2': np.array(data['K2']),
        'D1': np.array(data['D1']),
        'D2': np.array(data['D2']),
        'R':  np.array(data['R']),
        'T':  np.array(data['T']).reshape(3, 1),
        'image_size': tuple(data.get('image_size', [1280, 720])),
        'floor_z_offset': float(data.get('floor_z_offset', 0.0) or 0.0),
        'stereo_rms': float(data.get('stereo_rms', 0.0) or 0.0),
        'baseline_meters': float(data.get('baseline_meters', 0.0) or 0.0),
    }


def derive_ac(ab_path, bc_path, out_path):
    """Derive AC calibration by chaining AB and BC.

    Returns True on success.
    """
    ab_path = Path(ab_path)
    bc_path = Path(bc_path)
    out_path = Path(out_path)

    if not ab_path.exists():
        print(f"[ERROR] AB calibration not found: {ab_path}")
        return False
    if not bc_path.exists():
        print(f"[ERROR] BC calibration not found: {bc_path}")
        return False

    print(f"[CHAIN] Loading AB from {ab_path.name}")
    ab = _load_pair(ab_path)
    print(f"         AB baseline: {ab['baseline_meters']:.3f}m, "
          f"RMS: {ab['stereo_rms']:.3f}, "
          f"floor offset: {ab['floor_z_offset']:.3f}m")

    print(f"[CHAIN] Loading BC from {bc_path.name}")
    bc = _load_pair(bc_path)
    print(f"         BC baseline: {bc['baseline_meters']:.3f}m, "
          f"RMS: {bc['stereo_rms']:.3f}")

    if ab['image_size'] != bc['image_size']:
        print(f"[ERROR] Image size mismatch: AB {ab['image_size']} vs BC {bc['image_size']}")
        return False

    # Sanity: AB's K2 is Camera B. BC's K1 is also Camera B. They should
    # be nearly identical (same physical camera, same resolution).
    k_b_diff = np.linalg.norm(ab['K2'] - bc['K1']) / np.linalg.norm(ab['K2'])
    if k_b_diff > 0.05:
        print(f"[WARN] Camera B intrinsics differ between AB and BC by "
              f"{k_b_diff*100:.1f}% — calibrations may be inconsistent")
    else:
        print(f"[OK]   Camera B intrinsics match between AB and BC "
              f"(diff {k_b_diff*100:.2f}%)")

    # Chain extrinsics:
    #   X_B = R_AB @ X_A + T_AB
    #   X_C = R_BC @ X_B + T_BC
    #   => X_C = (R_BC @ R_AB) @ X_A + (R_BC @ T_AB + T_BC)
    R_AB = ab['R']
    T_AB = ab['T']
    R_BC = bc['R']
    T_BC = bc['T']

    R_AC = R_BC @ R_AB
    T_AC = R_BC @ T_AB + T_BC
    baseline_AC = float(np.linalg.norm(T_AC))

    print(f"[CHAIN] Derived AC extrinsics")
    print(f"         AC baseline: {baseline_AC:.3f}m")

    # Camera A intrinsics from AB, Camera C intrinsics from BC
    K_A = ab['K1']
    D_A = ab['D1']
    K_C = bc['K2']
    D_C = bc['D2']
    image_size = ab['image_size']  # (W, H)

    # Rectification (same as stereo_calibration.py's calibrate() flow)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_A, D_A, K_C, D_C, image_size, R_AC, T_AC,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0,
    )

    # AC inherits AB's floor offset because A is the shared reference frame
    floor_z_offset = ab['floor_z_offset']

    out_data = {
        'version': 'melodiccap_rtm_v1',
        'timestamp': datetime.now().isoformat(),
        'derived_from': ['stereo_calibration_ab.json', 'stereo_calibration_bc.json'],
        'derivation_method': 'chain (R_AC = R_BC @ R_AB, T_AC = R_BC @ T_AB + T_BC)',
        'image_size': list(image_size),
        'K1': K_A.tolist(),
        'K2': K_C.tolist(),
        'D1': D_A.tolist(),
        'D2': D_C.tolist(),
        'R':  R_AC.tolist(),
        'T':  T_AC.tolist(),
        'R1': R1.tolist(),
        'R2': R2.tolist(),
        'P1': P1.tolist(),
        'P2': P2.tolist(),
        'baseline_meters': baseline_AC,
        # Propagated RMS: worst-case sum of the contributing pairs
        'stereo_rms': ab['stereo_rms'] + bc['stereo_rms'],
        'cam_a_rms': None,
        'cam_b_rms': None,
        'floor_z_offset': floor_z_offset,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)

    print(f"[OK]   Saved derived AC calibration to {out_path}")
    print(f"         Propagated stereo RMS: {out_data['stereo_rms']:.3f}")
    print(f"         Floor offset (inherited from AB): {floor_z_offset:.3f}m")

    # Quality warnings
    if out_data['stereo_rms'] > 1.0:
        print(f"[WARN] Propagated RMS {out_data['stereo_rms']:.3f} > 1.0 — "
              f"the offline processor's quality gate will reject this pair. "
              f"Recalibrate AB and/or BC for better quality first.")
    if baseline_AC < 0.8:
        print(f"[WARN] AC baseline {baseline_AC:.3f}m < 0.8m — "
              f"the offline processor's baseline gate will reject this pair.")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Derive AC stereo calibration from AB + BC calibrations'
    )
    parser.add_argument('--calibration-dir', default=None,
                        help='Calibration directory (default: ./calibration)')
    parser.add_argument('--ab', default=None,
                        help='Path to AB calibration (default: <dir>/stereo_calibration_ab.json)')
    parser.add_argument('--bc', default=None,
                        help='Path to BC calibration (default: <dir>/stereo_calibration_bc.json)')
    parser.add_argument('--out', default=None,
                        help='Output path (default: <dir>/stereo_calibration_ac.json)')

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    cal_dir = Path(args.calibration_dir) if args.calibration_dir else base_dir / 'calibration'

    ab_path = Path(args.ab) if args.ab else cal_dir / 'stereo_calibration_ab.json'
    bc_path = Path(args.bc) if args.bc else cal_dir / 'stereo_calibration_bc.json'
    out_path = Path(args.out) if args.out else cal_dir / 'stereo_calibration_ac.json'

    ok = derive_ac(ab_path, bc_path, out_path)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
