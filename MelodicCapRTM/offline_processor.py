"""
Offline Motion Capture Processor
=================================
Loads raw 2D detection recordings (melodiccap_raw_v1 format) and runs
stereo triangulation + filtering offline. Outputs standard melodiccap_rtm_v1
JSON files ready for the Blender addon.

Usage:
    python offline_processor.py takes/take_20260328_143000_raw.json
    python offline_processor.py takes/take_20260328_143000_raw.json --smooth --no-smooth
    python offline_processor.py takes/*.json  (batch mode)

Benefits over real-time triangulation:
    - No frame drops from GPU contention (pose detection vs triangulation)
    - Can re-process the same take with different filter settings
    - Can apply bi-directional smoothing (forward + backward Kalman)
"""

import argparse
import json
import sys
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


def process_take(raw_path, calibration_path, smooth=True,
                 min_conf=0.3, skip_face_hands=True,
                 kalman_process=1e-4, kalman_measure=1e-2):
    """
    Process a single raw detection file into triangulated 3D output.

    Args:
        raw_path: path to the _raw.json file
        calibration_path: path to stereo_calibration.json
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

    # If already processed (has 3D data) and not raw, skip unless forced
    if raw_data.get('format') == 'melodiccap_rtm_v1':
        print(f"[SKIP] {raw_path.name} is already triangulated (melodiccap_rtm_v1)")
        return None

    frames = raw_data.get('frames', [])
    if not frames:
        print(f"[ERROR] No frames in {raw_path}")
        return None

    # Check that frames have raw 2D data
    first = frames[0]
    if 'raw_2d_a' not in first or 'raw_2d_b' not in first:
        print(f"[ERROR] Frames missing raw_2d_a/raw_2d_b in {raw_path}")
        return None

    # Init calibration
    config = OfflineConfig(
        min_conf=min_conf,
        skip_face_hands=skip_face_hands,
        kalman_process=kalman_process,
        kalman_measure=kalman_measure,
    )
    calibration = StereoCalibration(config)
    if not calibration.load(calibration_path):
        print(f"[ERROR] Failed to load calibration from {calibration_path}")
        return None

    calibration.reset_filters()

    print(f"\n[PROCESSING] {raw_path.name}")
    print(f"  {len(frames)} frames, smooth={smooth}, min_conf={min_conf}")

    # Forward pass: triangulate each frame
    output_frames = []
    triangulated_count = 0

    for i, frame in enumerate(frames):
        # Reconstruct detection dicts with integer keys
        det_a = {int(k): tuple(v) for k, v in frame['raw_2d_a'].items()}
        det_b = {int(k): tuple(v) for k, v in frame['raw_2d_b'].items()}

        points_3d = calibration.triangulate_pose(det_a, det_b, smooth=smooth)

        out_frame = {
            "timestamp": frame["timestamp"],
        }

        if points_3d:
            out_frame["landmarks_3d"] = {str(k): v for k, v in points_3d.items()}
            triangulated_count += 1

        # Preserve raw 2D in output for potential re-processing
        out_frame["raw_2d_a"] = frame["raw_2d_a"]
        out_frame["raw_2d_b"] = frame["raw_2d_b"]

        output_frames.append(out_frame)

        # Progress
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(frames)} frames...")

    # Build output
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
        },
        "frames": output_frames
    }

    # Output path: replace _raw.json with .json
    out_name = raw_path.stem.replace('_raw', '') + '.json'
    out_path = raw_path.parent / out_name
    with open(out_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    success_pct = (triangulated_count / len(frames) * 100) if frames else 0
    print(f"  [DONE] {out_path}")
    print(f"         {triangulated_count}/{len(frames)} frames triangulated ({success_pct:.0f}%)")

    return str(out_path)


def main():
    parser = argparse.ArgumentParser(
        description='Process raw MelodicCap recordings into triangulated 3D motion data'
    )
    parser.add_argument('files', nargs='+', help='Raw recording JSON files to process')
    parser.add_argument('--calibration', '-c', default=None,
                        help='Path to stereo_calibration.json (default: calibration/stereo_calibration.json)')
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

    # Default calibration path
    base_dir = Path(__file__).resolve().parent
    cal_path = Path(args.calibration) if args.calibration else base_dir / "calibration" / "stereo_calibration.json"

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
