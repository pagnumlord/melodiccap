"""
MelodicCap Take Validator
=========================
Standalone script to validate and clean take JSON files.

Usage:
    python take_validator.py <take_file.json>
    python take_validator.py takes/take_20260221_010702.json

What it does:
1. Loads take JSON and reports frame count, duration, landmark coverage
2. Flags landmarks where any coordinate is > 5m from hip center (outliers)
3. Reports which landmarks have outliers and how many frames are affected
4. Writes cleaned version as <take>_clean.json with outliers removed
5. Reports coordinate ranges before and after cleaning

Known outlier patterns from production takes:
- Landmark 31 (left_foot_index): 11/230 frames
- Landmark 16 (right_wrist): 6/230 frames
- Landmark 18 (right_pinky): 4/230 frames
- Landmark 20 (right_index): 2/230 frames
These are extremity points where triangulation is less reliable.
"""

import json
import sys
from pathlib import Path

LANDMARK_NAMES = {
    0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow", 15: "left_wrist", 16: "right_wrist",
    17: "left_pinky", 18: "right_pinky", 19: "left_index", 20: "right_index",
    21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle", 29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index",
}

OUTLIER_THRESHOLD = 5.0  # meters from hip center


def get_hip_center(landmarks):
    """Get hip center from landmarks 23 and 24."""
    lm23 = landmarks.get('23') or landmarks.get(23)
    lm24 = landmarks.get('24') or landmarks.get(24)
    if lm23 and lm24:
        return [(lm23[i] + lm24[i]) / 2 for i in range(3)]
    return None


def validate_take(filepath):
    """Validate and clean a take JSON file."""
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"ERROR: File not found: {filepath}")
        return

    with open(filepath) as f:
        data = json.load(f)

    frames = data.get('frames', [])
    print("=" * 60)
    print(f"TAKE VALIDATOR: {filepath.name}")
    print("=" * 60)

    # Basic info
    print(f"\n  Version: {data.get('version', '?')}")
    print(f"  Frames: {len(frames)}")
    print(f"  Duration: {data.get('duration_seconds', 0):.1f}s")
    print(f"  Predicted frames: {data.get('predicted_frames', 0)}")
    print(f"  Dropped frames: {data.get('dropped_frames', 0)}")

    calib = data.get('calibration', {})
    print(f"  Stereo RMS: {calib.get('rms_stereo', 'N/A')}")
    print(f"  Baseline: {calib.get('baseline', 'N/A')}m")
    print(f"  Floor offset: {calib.get('floor_offset', 0):.3f}m")

    if not frames:
        print("\nERROR: No frames in take!")
        return

    # Analyze coordinate ranges and outliers
    print("\n--- COORDINATE ANALYSIS ---")

    all_x, all_y, all_z = [], [], []
    outlier_frames = set()
    outlier_landmarks = {}  # lm_idx -> count
    total_outlier_points = 0

    for fidx, fdata in enumerate(frames):
        lms = fdata.get('landmarks', {})
        hip = get_hip_center(lms)

        for key, pos in lms.items():
            idx = int(key) if isinstance(key, str) else key
            all_x.append(pos[0])
            all_y.append(pos[1])
            all_z.append(pos[2])

            # Check outlier relative to hip center
            is_outlier = False
            if hip:
                for i in range(3):
                    if abs(pos[i] - hip[i]) > OUTLIER_THRESHOLD:
                        is_outlier = True
                        break
            else:
                # No hip center — check absolute
                for i in range(3):
                    if abs(pos[i]) > OUTLIER_THRESHOLD:
                        is_outlier = True
                        break

            if is_outlier:
                outlier_frames.add(fidx)
                outlier_landmarks[idx] = outlier_landmarks.get(idx, 0) + 1
                total_outlier_points += 1

    print(f"\n  Coordinate ranges (BEFORE cleaning):")
    print(f"    X: [{min(all_x):.2f}, {max(all_x):.2f}]")
    print(f"    Y: [{min(all_y):.2f}, {max(all_y):.2f}]")
    print(f"    Z: [{min(all_z):.2f}, {max(all_z):.2f}]")

    # Landmark coverage
    lm_counts = {}
    for fdata in frames:
        for key in fdata.get('landmarks', {}).keys():
            idx = int(key) if isinstance(key, str) else key
            lm_counts[idx] = lm_counts.get(idx, 0) + 1

    print(f"\n  Landmark coverage ({len(lm_counts)} unique landmarks):")
    for idx in sorted(lm_counts.keys()):
        count = lm_counts[idx]
        pct = count * 100 / len(frames)
        name = LANDMARK_NAMES.get(idx, f"unknown_{idx}")
        outlier_note = ""
        if idx in outlier_landmarks:
            outlier_note = f"  ** {outlier_landmarks[idx]} outlier frames **"
        print(f"    [{idx:2d}] {name:20s}: {count:4d}/{len(frames)} ({pct:5.1f}%){outlier_note}")

    # Outlier summary
    print(f"\n--- OUTLIER SUMMARY ---")
    print(f"  Threshold: {OUTLIER_THRESHOLD}m from hip center")
    print(f"  Frames with outliers: {len(outlier_frames)}/{len(frames)} ({len(outlier_frames)*100/len(frames):.1f}%)")
    print(f"  Total outlier points: {total_outlier_points}")

    if outlier_landmarks:
        print(f"\n  Outlier landmarks (sorted by frequency):")
        for idx, count in sorted(outlier_landmarks.items(), key=lambda x: -x[1]):
            name = LANDMARK_NAMES.get(idx, f"unknown_{idx}")
            print(f"    [{idx:2d}] {name:20s}: {count} frames")
    else:
        print(f"\n  No outliers detected! Take is clean.")
        return

    # Person height from frame 0
    ref_lms = frames[0].get('landmarks', {})
    nose = ref_lms.get('0') or ref_lms.get(0)
    ankle_l = ref_lms.get('27') or ref_lms.get(27)
    ankle_r = ref_lms.get('28') or ref_lms.get(28)

    if nose and ankle_l and ankle_r:
        ankle_mid_z = (ankle_l[2] + ankle_r[2]) / 2
        person_height = (nose[2] - ankle_mid_z) + 0.15
        print(f"\n  Person height (frame 0): {person_height:.3f}m")

    # Clean the take
    print(f"\n--- CLEANING ---")

    cleaned_frames = []
    for fidx, fdata in enumerate(frames):
        lms = fdata.get('landmarks', {})
        hip = get_hip_center(lms)

        cleaned_lms = {}
        for key, pos in lms.items():
            idx = int(key) if isinstance(key, str) else key

            is_outlier = False
            if hip:
                for i in range(3):
                    if abs(pos[i] - hip[i]) > OUTLIER_THRESHOLD:
                        is_outlier = True
                        break
            else:
                for i in range(3):
                    if abs(pos[i]) > OUTLIER_THRESHOLD:
                        is_outlier = True
                        break

            if not is_outlier:
                cleaned_lms[key] = pos

        cleaned_frame = dict(fdata)
        cleaned_frame['landmarks'] = cleaned_lms
        cleaned_frame['num_landmarks'] = len(cleaned_lms)
        cleaned_frames.append(cleaned_frame)

    # Coordinate ranges after cleaning
    clean_x, clean_y, clean_z = [], [], []
    for fdata in cleaned_frames:
        for pos in fdata['landmarks'].values():
            clean_x.append(pos[0])
            clean_y.append(pos[1])
            clean_z.append(pos[2])

    print(f"  Coordinate ranges (AFTER cleaning):")
    print(f"    X: [{min(clean_x):.2f}, {max(clean_x):.2f}]")
    print(f"    Y: [{min(clean_y):.2f}, {max(clean_y):.2f}]")
    print(f"    Z: [{min(clean_z):.2f}, {max(clean_z):.2f}]")

    # Write cleaned file
    clean_data = dict(data)
    clean_data['frames'] = cleaned_frames
    clean_data['cleaned'] = True
    clean_data['outliers_removed'] = total_outlier_points

    output_path = filepath.parent / f"{filepath.stem}_clean{filepath.suffix}"
    with open(output_path, 'w') as f:
        json.dump(clean_data, f)

    output_size = output_path.stat().st_size / 1024 / 1024
    print(f"\n  Saved: {output_path.name} ({output_size:.1f} MB)")
    print(f"  Outlier points removed: {total_outlier_points}")
    print(f"\nDone!")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python take_validator.py <take_file.json>")
        print("Example: python take_validator.py takes/take_20260221_010702.json")
        sys.exit(1)

    validate_take(sys.argv[1])
