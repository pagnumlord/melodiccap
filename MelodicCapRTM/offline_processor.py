"""
Offline Motion Capture Processor
=================================
Loads raw 2D detection recordings (melodiccap_raw_v1 format) and runs
stereo triangulation + filtering offline. Outputs standard melodiccap_rtm_v1
JSON files ready for the Blender addon.

Supports multi-pair triangulation: when 3 cameras are calibrated, per-frame
pair selection picks the best camera pair for each frame (by mean reprojection
error), then routes all keypoints through the full triangulate_pose() pipeline
(Kalman + outlier rejection + bone constraints).

Quality gates automatically reject inadequate calibration pairs:
  - stereo RMS > 1.0 (too noisy)
  - baseline < 0.8m (poor depth resolution)
  - no floor offset (would shift skeleton origin between pairs)

Usage:
    python offline_processor.py takes/take_20260328_143000_raw.json
    python offline_processor.py takes/take_20260328_143000_raw.json --no-smooth
    python offline_processor.py takes/take_20260328_143000_raw.json --pair AB
    python offline_processor.py takes/*.json  (batch mode)

Benefits over real-time triangulation:
    - No frame drops from GPU contention (pose detection vs triangulation)
    - Can re-process the same take with different filter settings
    - Can apply bi-directional smoothing (forward + backward Kalman)
    - Multi-pair triangulation picks best camera pair per frame
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

# Add parent dir to path so we can import from MelodicCapRTM
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stereo_calibration import StereoCalibration
from skeleton_solver import SkeletonSolver


class OfflineConfig:
    """Minimal config for StereoCalibration — only the fields it needs."""

    def __init__(self, min_conf=0.3, skip_face_hands=True,
                 kalman_process=1e-4, kalman_measure=1e-2):
        self.MIN_KEYPOINT_CONFIDENCE = min_conf
        self.SKIP_FACE_HANDS = skip_face_hands
        self.KALMAN_PROCESS_NOISE = kalman_process
        self.KALMAN_MEASUREMENT_NOISE = kalman_measure

        # ChArUco board config — required by StereoCalibration.__init__
        # but not used during offline triangulation
        import cv2
        self.ARUCO_DICT = cv2.aruco.DICT_4X4_50
        self.CHARUCO_SQUARES_X = 10
        self.CHARUCO_SQUARES_Y = 5
        self.CHARUCO_SQUARE_SIZE = 0.04286
        self.CHARUCO_MARKER_SIZE = 0.03016
        self.FRAME_WIDTH = 1280
        self.FRAME_HEIGHT = 720


def _triangulate_keypoint(calibration, px1, py1, px2, py2):
    """Triangulate a single keypoint and return (point_3d, reproj_error).

    Returns:
        (blender_point, reproj_error) or (None, inf) on failure.
        blender_point is [x, y, z] in Blender space.
        reproj_error is mean pixel error across both cameras.
    """
    result = calibration.triangulate([[px1, py1]], [[px2, py2]])
    if result is None:
        return None, float('inf')

    bpt = result[0]  # Blender coords
    # Convert back to CV for reprojection error
    cv_pt = [bpt[0], -bpt[2] + calibration.floor_z_offset, bpt[1]]
    err_a, err_b, _ = calibration.compute_reprojection_error(
        [[px1, py1]], [[px2, py2]], [cv_pt])
    return bpt.tolist(), (err_a + err_b) / 2.0


PAIR_RMS_MAX_DEFAULT = 1.0  # Reject calibration pairs with stereo RMS above this
MIN_BASELINE_DEFAULT = 0.8  # Reject pairs with baseline below this (meters)


def _load_calibrations(calibration_dir, pair_rms_max=PAIR_RMS_MAX_DEFAULT,
                       min_baseline=MIN_BASELINE_DEFAULT):
    """Load all available calibration pairs from a directory.

    Quality gates (any failure rejects the pair):
      - stereo RMS must be <= pair_rms_max
      - baseline must be >= min_baseline (short baselines have poor depth)
      - floor_z_offset must be nonzero (required for correct world coordinates)

    Returns:
        list of (label, StereoCalibration, cam_key_1, cam_key_2) tuples
    """
    calibration_dir = Path(calibration_dir)
    config = OfflineConfig()

    pairs = []

    # Try loading specific pair files
    pair_files = [
        ('AB', 'stereo_calibration_ab.json', 'raw_2d_a', 'raw_2d_b'),
        ('AC', 'stereo_calibration_ac.json', 'raw_2d_a', 'raw_2d_c'),
        ('BC', 'stereo_calibration_bc.json', 'raw_2d_b', 'raw_2d_c'),
    ]

    for label, filename, key1, key2 in pair_files:
        cal_file = calibration_dir / filename
        if cal_file.exists():
            cal = StereoCalibration(config)
            if not cal.load(cal_file):
                continue

            # Prefer metrics restored by load(); fall back to JSON for legacy
            # files that pre-date the load() fix.
            rms = cal._stereo_rms if getattr(cal, '_stereo_rms', None) is not None else 0.0
            baseline = (cal.baseline_meters
                        if getattr(cal, 'baseline_meters', None) is not None else 0.0)
            floor_offset = cal.floor_z_offset or 0.0
            derivation = getattr(cal, 'derivation_method', None)
            if rms == 0.0 or baseline == 0.0 or derivation is None:
                try:
                    with open(cal_file, 'r') as f:
                        cal_json = json.load(f)
                    if rms == 0.0:
                        rms = cal_json.get('stereo_rms', 0.0) or 0.0
                    if baseline == 0.0:
                        baseline = cal_json.get('baseline_meters', 0.0) or 0.0
                    if derivation is None:
                        derivation = cal_json.get('derivation_method')
                except Exception:
                    pass

            # Chain-derived pairs get a relaxed RMS gate. Their propagated
            # RMS is combined-in-quadrature from two real calibrations, so
            # it's systematically higher even when the geometry is fine.
            is_chain = (derivation == 'chain')
            effective_rms_max = pair_rms_max * 1.5 if is_chain else pair_rms_max

            # Quality gate: RMS
            if rms > effective_rms_max:
                gate_label = (f"{effective_rms_max:.2f} (chain relaxed)"
                              if is_chain else f"{pair_rms_max}")
                print(f"  [SKIP] {label}: stereo RMS {rms:.3f} > {gate_label} "
                      f"(too noisy)")
                continue

            # Quality gate: baseline
            if baseline > 0 and baseline < min_baseline:
                print(f"  [SKIP] {label}: baseline {baseline:.3f}m < {min_baseline}m "
                      f"(poor depth resolution)")
                continue

            # Quality gate: floor offset
            if abs(floor_offset) < 0.01:
                print(f"  [SKIP] {label}: no floor offset calibrated "
                      f"(would shift skeleton origin)")
                continue

            rms_str = f" (RMS {rms:.3f})" if rms > 0 else ""
            base_str = f", baseline {baseline:.3f}m" if baseline > 0 else ""
            chain_str = " [chain-derived]" if is_chain else ""
            print(f"  [OK] {label}{rms_str}{base_str}{chain_str}")
            pairs.append((label, cal, key1, key2))

    # If no pair-specific files found, fall back to the single calibration
    if not pairs:
        single_file = calibration_dir / 'stereo_calibration.json'
        if single_file.exists():
            cal = StereoCalibration(config)
            if cal.load(single_file):
                pairs.append(('AB', cal, 'raw_2d_a', 'raw_2d_b'))

    return pairs


def process_take(raw_path, calibration_path, smooth=True,
                 min_conf=0.3, skip_face_hands=True,
                 kalman_process=1e-4, kalman_measure=1e-2,
                 pair_rms_max=PAIR_RMS_MAX_DEFAULT,
                 min_baseline=MIN_BASELINE_DEFAULT, force_pair=None,
                 use_skeleton_solver=True,
                 direction_smooth_alpha=0.5):
    """
    Process a single raw detection file into triangulated 3D output.

    When multiple calibration pairs exist, uses per-frame pair selection:
    each frame is triangulated by the pair with lowest mean reprojection
    error, routed through the full triangulate_pose() pipeline (Kalman +
    outlier rejection + bone constraints).

    Args:
        raw_path: path to the _raw.json file
        calibration_path: path to calibration directory or single .json file
        smooth: apply Kalman smoothing
        min_conf: minimum keypoint confidence
        skip_face_hands: skip face/hand keypoints in wholebody mode
        kalman_process: Kalman process noise
        kalman_measure: Kalman measurement noise
        pair_rms_max: max stereo RMS for calibration pairs (default 1.0)
        min_baseline: min stereo baseline in meters (default 0.8)
        force_pair: force a specific calibration pair label (e.g. 'AB')
        use_skeleton_solver: apply skeleton solver post-processing (default True)

    Returns:
        output filepath, or None on failure
    """
    raw_path = Path(raw_path)
    calibration_path = Path(calibration_path)

    # Load raw recording
    with open(raw_path, 'r') as f:
        raw_data = json.load(f)

    if raw_data.get('format') not in ('melodiccap_raw_v1', 'melodiccap_rtm_v1'):
        print(f"[ERROR] Unknown format: {raw_data.get('format')} in {raw_path}")
        return None

    if raw_data.get('format') == 'melodiccap_rtm_v1':
        print(f"[SKIP] {raw_path.name} is already triangulated (melodiccap_rtm_v1)")
        return None

    frames = raw_data.get('frames', [])
    if not frames:
        print(f"[ERROR] No frames in {raw_path}")
        return None

    first = frames[0]
    if 'raw_2d_a' not in first or 'raw_2d_b' not in first:
        print(f"[ERROR] Frames missing raw_2d_a/raw_2d_b in {raw_path}")
        return None

    has_cam_c = 'raw_2d_c' in first

    # Determine calibration directory
    if calibration_path.is_dir():
        cal_dir = calibration_path
    else:
        cal_dir = calibration_path.parent

    # Load all available calibration pairs
    cal_pairs = _load_calibrations(cal_dir, pair_rms_max=pair_rms_max,
                                    min_baseline=min_baseline)

    # Force a specific pair if requested
    if force_pair and cal_pairs:
        filtered = [(l, c, k1, k2) for l, c, k1, k2 in cal_pairs
                     if l == force_pair.upper()]
        if filtered:
            cal_pairs = filtered
            print(f"  Forced pair: {force_pair.upper()}")
        else:
            print(f"  [WARN] Requested pair {force_pair.upper()} not available, "
                  f"using: {', '.join(l for l, _, _, _ in cal_pairs)}")

    if not cal_pairs:
        # Fall back to the explicit path
        config = OfflineConfig(
            min_conf=min_conf,
            skip_face_hands=skip_face_hands,
            kalman_process=kalman_process,
            kalman_measure=kalman_measure,
        )
        cal = StereoCalibration(config)
        if not cal.load(calibration_path):
            print(f"[ERROR] Failed to load calibration from {calibration_path}")
            return None
        cal_pairs = [('AB', cal, 'raw_2d_a', 'raw_2d_b')]

    # Filter pairs: only keep pairs where the data has the required camera keys
    available_keys = set(first.keys())
    cal_pairs = [(lbl, cal, k1, k2) for lbl, cal, k1, k2 in cal_pairs
                 if k1 in available_keys and k2 in available_keys]

    if not cal_pairs:
        print(f"[ERROR] No calibration pairs match the available camera data")
        return None

    multi_pair = len(cal_pairs) > 1
    pair_labels = [p[0] for p in cal_pairs]
    print(f"\n[PROCESSING] {raw_path.name}")
    print(f"  {len(frames)} frames, smooth={smooth}, min_conf={min_conf}")
    print(f"  Calibration pairs: {', '.join(pair_labels)}"
          f" {'(MULTI-PAIR)' if multi_pair else '(single pair)'}")
    if has_cam_c:
        print(f"  Camera C data: present")

    # For single-pair mode, use the existing triangulate_pose pipeline
    # (includes Kalman, outlier rejection, bone constraints)
    if not multi_pair:
        _, primary_cal, _, _ = cal_pairs[0]
        primary_cal.reset_filters()
        return _process_single_pair(raw_path, raw_data, frames, primary_cal,
                                    smooth, min_conf, skip_face_hands,
                                    kalman_process, kalman_measure,
                                    use_skeleton_solver=use_skeleton_solver,
                                    direction_smooth_alpha=direction_smooth_alpha)

    # Multi-pair mode: per-frame pair selection.
    # For each frame, score all pairs by mean reprojection error, then route
    # ALL keypoints through the winning pair's triangulate_pose() pipeline.
    # This keeps the skeleton in a single consistent coordinate frame per frame
    # and gets the full quality pipeline (Kalman + outlier rejection + bone
    # constraints) instead of the raw per-keypoint approach which bypassed it.

    # Reset filters on all calibrations
    for _, cal, _, _ in cal_pairs:
        cal.reset_filters()

    output_frames = []
    triangulated_count = 0
    pair_win_counts = {lbl: 0 for lbl, _, _, _ in cal_pairs}

    for i, frame in enumerate(frames):
        # Load detections for all cameras
        dets = {}
        for key in ['raw_2d_a', 'raw_2d_b', 'raw_2d_c']:
            if key in frame:
                dets[key] = {int(k): tuple(v) for k, v in frame[key].items()}

        # Pass 1: Score each pair for this frame by mean reprojection error
        best_pair_label = None
        best_pair_cal = None
        best_pair_keys = None
        best_mean_err = float('inf')

        for label, cal, key1, key2 in cal_pairs:
            if key1 not in dets or key2 not in dets:
                continue

            errors = []
            for idx in dets[key1]:
                if idx not in dets[key2]:
                    continue
                if skip_face_hands and idx > 22:
                    continue
                px1, py1, conf1 = dets[key1][idx]
                px2, py2, conf2 = dets[key2][idx]
                if conf1 < min_conf or conf2 < min_conf:
                    continue
                _, err = _triangulate_keypoint(cal, px1, py1, px2, py2)
                if err < float('inf'):
                    errors.append(err)

            if errors:
                mean_err = sum(errors) / len(errors)
                if mean_err < best_mean_err:
                    best_mean_err = mean_err
                    best_pair_label = label
                    best_pair_cal = cal
                    best_pair_keys = (key1, key2)

        # Pass 2: Route winning pair through full triangulate_pose()
        points_3d = None
        if best_pair_cal and best_pair_keys:
            det1 = dets.get(best_pair_keys[0], {})
            det2 = dets.get(best_pair_keys[1], {})
            points_3d = best_pair_cal.triangulate_pose(
                det1, det2, smooth=smooth,
                enforce_bones=not use_skeleton_solver)
            pair_win_counts[best_pair_label] += 1

        out_frame = {"timestamp": frame["timestamp"]}

        if points_3d:
            out_frame["landmarks_3d"] = {str(k): v for k, v in points_3d.items()}
            triangulated_count += 1
            # v5.7: store per-keypoint confidence (min of winning pair's 2D conf)
            # so the solver can use it for confidence-weighted smoothing.
            if best_pair_keys:
                conf_dict = {}
                d1 = dets.get(best_pair_keys[0], {})
                d2 = dets.get(best_pair_keys[1], {})
                for k in points_3d:
                    c1 = d1.get(k, (0, 0, 0))[2] if k in d1 else 0.0
                    c2 = d2.get(k, (0, 0, 0))[2] if k in d2 else 0.0
                    conf_dict[str(k)] = round(min(c1, c2), 3)
                out_frame['confidence'] = conf_dict

        # Preserve raw 2D
        for key in ['raw_2d_a', 'raw_2d_b', 'raw_2d_c']:
            if key in frame:
                out_frame[key] = frame[key]

        output_frames.append(out_frame)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(frames)} frames...")

    # Report pair selection stats
    total_selections = sum(pair_win_counts.values())
    if total_selections > 0:
        print(f"\n  [PAIR SELECTION] (per-frame)")
        for label in pair_labels:
            count = pair_win_counts[label]
            pct = count / total_selections * 100
            print(f"    {label}: {count} frames ({pct:.0f}%)")

    # Skeleton solver: enforce fixed bone lengths + joint angle limits
    solver_meta = None
    if use_skeleton_solver:
        solver = SkeletonSolver(
            n_cal_frames=30,
            direction_smooth_alpha=direction_smooth_alpha,
        )
        output_frames = solver.solve_sequence(output_frames)
        solver_meta = solver.get_metadata()

    # Build output
    return _write_output(raw_path, raw_data, output_frames, frames,
                         triangulated_count, smooth, min_conf,
                         skip_face_hands, kalman_process, kalman_measure,
                         multi_pair=True, pair_labels=pair_labels,
                         solver_meta=solver_meta)


def _process_single_pair(raw_path, raw_data, frames, calibration,
                         smooth, min_conf, skip_face_hands,
                         kalman_process, kalman_measure,
                         use_skeleton_solver=True,
                         direction_smooth_alpha=0.5):
    """Original single-pair processing pipeline."""
    output_frames = []
    triangulated_count = 0

    for i, frame in enumerate(frames):
        det_a = {int(k): tuple(v) for k, v in frame['raw_2d_a'].items()}
        det_b = {int(k): tuple(v) for k, v in frame['raw_2d_b'].items()}

        points_3d = calibration.triangulate_pose(
            det_a, det_b, smooth=smooth,
            enforce_bones=not use_skeleton_solver)

        out_frame = {"timestamp": frame["timestamp"]}

        if points_3d:
            out_frame["landmarks_3d"] = {str(k): v for k, v in points_3d.items()}
            triangulated_count += 1
            # v5.7: per-keypoint confidence = min(conf_a, conf_b)
            conf_dict = {}
            for k in points_3d:
                c_a = det_a.get(k, (0, 0, 0))[2] if k in det_a else 0.0
                c_b = det_b.get(k, (0, 0, 0))[2] if k in det_b else 0.0
                conf_dict[str(k)] = round(min(c_a, c_b), 3)
            out_frame['confidence'] = conf_dict

        # Preserve raw 2D
        for key in ['raw_2d_a', 'raw_2d_b', 'raw_2d_c']:
            if key in frame:
                out_frame[key] = frame[key]

        output_frames.append(out_frame)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(frames)} frames...")

    # Skeleton solver: enforce fixed bone lengths + joint angle limits
    solver_meta = None
    if use_skeleton_solver:
        solver = SkeletonSolver(
            n_cal_frames=30,
            direction_smooth_alpha=direction_smooth_alpha,
        )
        output_frames = solver.solve_sequence(output_frames)
        solver_meta = solver.get_metadata()

    return _write_output(raw_path, raw_data, output_frames, frames,
                         triangulated_count, smooth, min_conf,
                         skip_face_hands, kalman_process, kalman_measure,
                         solver_meta=solver_meta)


def _write_output(raw_path, raw_data, output_frames, frames,
                  triangulated_count, smooth, min_conf,
                  skip_face_hands, kalman_process, kalman_measure,
                  multi_pair=False, pair_labels=None, solver_meta=None):
    """Write output JSON file."""
    duration = raw_data.get('duration', 0)
    fps = len(output_frames) / duration if duration > 0 else 0

    processing_settings = {
        "smooth": smooth,
        "min_confidence": min_conf,
        "skip_face_hands": skip_face_hands,
        "kalman_process_noise": kalman_process,
        "kalman_measurement_noise": kalman_measure,
        "multi_pair": multi_pair,
        "calibration_pairs": pair_labels or [],
    }
    if solver_meta:
        processing_settings["skeleton_solver"] = solver_meta

    output_data = {
        "format": "melodiccap_rtm_v1",
        "keypoint_format": raw_data.get("keypoint_format", "coco_wholebody_133"),
        "detector": raw_data.get("detector", "unknown"),
        "take_name": raw_data.get("take_name", raw_path.stem),
        "duration": duration,
        "fps": fps,
        "frame_count": len(output_frames),
        "created": raw_data.get("created", ""),
        "processed_from": raw_path.name,
        "processing_settings": processing_settings,
        "frames": output_frames
    }

    out_name = raw_path.stem.replace('_raw', '') + '.json'
    out_path = raw_path.parent / out_name
    with open(out_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    success_pct = (triangulated_count / len(frames) * 100) if frames else 0
    mode_str = "multi-pair" if multi_pair else "single-pair"
    print(f"  [DONE] {out_path} ({mode_str})")
    print(f"         {triangulated_count}/{len(frames)} frames triangulated ({success_pct:.0f}%)")

    # v5.10: report per-keypoint fallback counts so we can see if the
    # velocity-rejection chain was firing on specific joints (the
    # "stuck at hip / shoulder" bug).
    fb = getattr(calibration, '_fallback_total', None) if not multi_pair else None
    if fb:
        landmark_names = {
            5: "L_shoulder", 6: "R_shoulder",
            7: "L_elbow", 8: "R_elbow",
            9: "L_wrist", 10: "R_wrist",
            11: "L_hip", 12: "R_hip",
            13: "L_knee", 14: "R_knee",
            15: "L_ankle", 16: "R_ankle",
        }
        bad = [(idx, n) for idx, n in fb.items() if n > 5]
        bad.sort(key=lambda x: -x[1])
        if bad:
            print(f"         Fallback fires (>5 frames): "
                  + ", ".join(f"{landmark_names.get(i, str(i))}={n}"
                              for i, n in bad[:6]))

    return str(out_path)


def main():
    parser = argparse.ArgumentParser(
        description='Process raw MelodicCap recordings into triangulated 3D motion data'
    )
    parser.add_argument('files', nargs='+', help='Raw recording JSON files to process')
    parser.add_argument('--calibration', '-c', default=None,
                        help='Path to calibration directory or stereo_calibration.json '
                             '(default: calibration/)')
    parser.add_argument('--no-smooth', action='store_true',
                        help='Disable Kalman smoothing')
    parser.add_argument('--min-conf', type=float, default=0.3,
                        help='Minimum keypoint confidence (default: 0.3)')
    parser.add_argument('--kalman-process', type=float, default=1e-4,
                        help='Kalman process noise (default: 1e-4)')
    parser.add_argument('--kalman-measure', type=float, default=1e-2,
                        help='Kalman measurement noise (default: 1e-2)')
    parser.add_argument('--include-face-hands', action='store_true',
                        help='Include face/hand keypoints (wholebody mode)')
    parser.add_argument('--pair', default=None,
                        help='Force specific calibration pair (e.g., AB, BC)')
    parser.add_argument('--pair-rms-max', type=float, default=1.0,
                        help='Max stereo RMS for calibration pairs (default: 1.0)')
    parser.add_argument('--min-baseline', type=float, default=0.8,
                        help='Min stereo baseline in meters (default: 0.8)')
    parser.add_argument('--no-skeleton', action='store_true',
                        help='Disable skeleton solver (bone length + angle limit enforcement)')
    parser.add_argument('--direction-smooth-alpha', type=float, default=0.5,
                        help='Baseline EMA α for solver direction smoothing (default: 0.5). '
                             'Lower values trust previous frame more. Per-frame α is '
                             'modulated by joint confidence; this sets the baseline.')

    args = parser.parse_args()

    # Default calibration path — try directory first for multi-pair
    base_dir = Path(__file__).resolve().parent
    if args.calibration:
        cal_path = Path(args.calibration)
    else:
        cal_path = base_dir / "calibration"

    processed = 0
    for filepath in args.files:
        result = process_take(
            raw_path=filepath,
            calibration_path=cal_path,
            smooth=not args.no_smooth,
            min_conf=args.min_conf,
            skip_face_hands=not args.include_face_hands,
            kalman_process=args.kalman_process,
            kalman_measure=args.kalman_measure,
            pair_rms_max=args.pair_rms_max,
            min_baseline=args.min_baseline,
            force_pair=args.pair,
            use_skeleton_solver=not args.no_skeleton,
            direction_smooth_alpha=args.direction_smooth_alpha,
        )
        if result:
            processed += 1

    print(f"\n[COMPLETE] Processed {processed} take(s)")


if __name__ == '__main__':
    main()
