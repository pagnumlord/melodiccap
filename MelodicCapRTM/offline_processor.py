"""
Offline Motion Capture Processor
=================================
Loads raw 2D detection recordings (melodiccap_raw_v1 format) and runs
stereo triangulation + filtering offline. Outputs standard melodiccap_rtm_v1
JSON files ready for the Blender addon.

Supports multi-pair triangulation: when 3 cameras are calibrated, each
keypoint is triangulated by all available pairs (AB, AC, BC) and the
result with the lowest reprojection error is selected. This significantly
improves accuracy, especially on the depth axis which is worst-resolved
with a single stereo pair.

Usage:
    python offline_processor.py takes/take_20260328_143000_raw.json
    python offline_processor.py takes/take_20260328_143000_raw.json --no-smooth
    python offline_processor.py takes/*.json  (batch mode)

Benefits over real-time triangulation:
    - No frame drops from GPU contention (pose detection vs triangulation)
    - Can re-process the same take with different filter settings
    - Can apply bi-directional smoothing (forward + backward Kalman)
    - Multi-pair triangulation picks best result per keypoint
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

# Add parent dir to path so we can import from MelodicCapRTM
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stereo_calibration import StereoCalibration


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


def _load_calibrations(calibration_dir):
    """Load all available calibration pairs from a directory.

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
            if cal.load(cal_file):
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
                 kalman_process=1e-4, kalman_measure=1e-2):
    """
    Process a single raw detection file into triangulated 3D output.

    Supports multi-pair triangulation: if multiple calibration files exist
    (stereo_calibration_ab.json, _ac.json, _bc.json), each keypoint is
    triangulated by all pairs and the lowest reprojection error wins.

    Args:
        raw_path: path to the _raw.json file
        calibration_path: path to calibration directory or single .json file
        smooth: apply Kalman smoothing
        min_conf: minimum keypoint confidence
        skip_face_hands: skip face/hand keypoints in wholebody mode
        kalman_process: Kalman process noise
        kalman_measure: Kalman measurement noise

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
    cal_pairs = _load_calibrations(cal_dir)
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
                                    kalman_process, kalman_measure)

    # Multi-pair mode: triangulate each keypoint with all pairs,
    # pick best per keypoint, then pass through primary pair's pipeline
    # for smoothing and bone constraints.
    primary_cal = cal_pairs[0][1]  # AB pair for smoothing/constraints
    primary_cal.reset_filters()

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

        # Find all keypoint indices across all cameras
        all_indices = set()
        for d in dets.values():
            all_indices.update(d.keys())

        if skip_face_hands:
            all_indices = {idx for idx in all_indices if idx <= 22}

        # For each keypoint, triangulate with all pairs, pick best
        best_points = {}
        for idx in all_indices:
            best_pt = None
            best_err = float('inf')
            best_pair = None

            for label, cal, key1, key2 in cal_pairs:
                if key1 not in dets or key2 not in dets:
                    continue
                if idx not in dets[key1] or idx not in dets[key2]:
                    continue

                px1, py1, conf1 = dets[key1][idx]
                px2, py2, conf2 = dets[key2][idx]

                if conf1 < min_conf or conf2 < min_conf:
                    continue

                pt, err = _triangulate_keypoint(cal, px1, py1, px2, py2)
                if pt is not None and err < best_err:
                    best_pt = pt
                    best_err = err
                    best_pair = label

            if best_pt is not None:
                best_points[idx] = best_pt
                if best_pair:
                    pair_win_counts[best_pair] += 1

        # Pass best points through primary calibration's smoothing pipeline
        # (Kalman + outlier rejection + bone constraints)
        if best_points:
            # Feed as if it were a single-pair result — primary_cal handles
            # smoothing, outlier rejection, bone constraints
            smoothed = primary_cal.triangulate_pose(
                dets.get('raw_2d_a', {}), dets.get('raw_2d_b', {}),
                smooth=smooth
            )

            # Replace with multi-pair results where available
            # For keypoints that only the primary pair saw, keep its result.
            # For keypoints with multi-pair data, use the best triangulation
            # but still apply the primary pair's Kalman filter state.
            if smoothed:
                for idx, pt in best_points.items():
                    if idx in smoothed:
                        # The Kalman filter in primary_cal already has state
                        # for this keypoint from the AB triangulation above.
                        # For multi-pair, we update the Kalman with the best point.
                        if smooth and idx in primary_cal.filters:
                            pt = [
                                primary_cal.filters[idx][0].update(pt[0]),
                                primary_cal.filters[idx][1].update(pt[1]),
                                primary_cal.filters[idx][2].update(pt[2]),
                            ]
                        smoothed[idx] = pt
                points_3d = smoothed
            else:
                points_3d = best_points
        else:
            points_3d = None

        out_frame = {"timestamp": frame["timestamp"]}

        if points_3d:
            out_frame["landmarks_3d"] = {str(k): v for k, v in points_3d.items()}
            triangulated_count += 1

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
        print(f"\n  [PAIR SELECTION]")
        for label in pair_labels:
            count = pair_win_counts[label]
            pct = count / total_selections * 100
            print(f"    {label}: {count} keypoints ({pct:.0f}%)")

    # Build output
    return _write_output(raw_path, raw_data, output_frames, frames,
                         triangulated_count, smooth, min_conf,
                         skip_face_hands, kalman_process, kalman_measure,
                         multi_pair=True, pair_labels=pair_labels)


def _process_single_pair(raw_path, raw_data, frames, calibration,
                         smooth, min_conf, skip_face_hands,
                         kalman_process, kalman_measure):
    """Original single-pair processing pipeline."""
    output_frames = []
    triangulated_count = 0

    for i, frame in enumerate(frames):
        det_a = {int(k): tuple(v) for k, v in frame['raw_2d_a'].items()}
        det_b = {int(k): tuple(v) for k, v in frame['raw_2d_b'].items()}

        points_3d = calibration.triangulate_pose(det_a, det_b, smooth=smooth)

        out_frame = {"timestamp": frame["timestamp"]}

        if points_3d:
            out_frame["landmarks_3d"] = {str(k): v for k, v in points_3d.items()}
            triangulated_count += 1

        # Preserve raw 2D
        for key in ['raw_2d_a', 'raw_2d_b', 'raw_2d_c']:
            if key in frame:
                out_frame[key] = frame[key]

        output_frames.append(out_frame)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(frames)} frames...")

    return _write_output(raw_path, raw_data, output_frames, frames,
                         triangulated_count, smooth, min_conf,
                         skip_face_hands, kalman_process, kalman_measure)


def _write_output(raw_path, raw_data, output_frames, frames,
                  triangulated_count, smooth, min_conf,
                  skip_face_hands, kalman_process, kalman_measure,
                  multi_pair=False, pair_labels=None):
    """Write output JSON file."""
    duration = raw_data.get('duration', 0)
    fps = len(output_frames) / duration if duration > 0 else 0

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
        "processing_settings": {
            "smooth": smooth,
            "min_confidence": min_conf,
            "skip_face_hands": skip_face_hands,
            "kalman_process_noise": kalman_process,
            "kalman_measurement_noise": kalman_measure,
            "multi_pair": multi_pair,
            "calibration_pairs": pair_labels or [],
        },
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
        )
        if result:
            processed += 1

    print(f"\n[COMPLETE] Processed {processed} take(s)")


if __name__ == '__main__':
    main()
