"""
Take Format Converter
=====================
Converts takes from any historical format to the unified format expected by
AntiGrav V3.2 retargeter.

Usage:
    python convert_takes.py <input.json> [output.json]
    python convert_takes.py --batch <directory>

Supported input formats:
  - melodic_justice_mocap: landmarks_3d as list of {index, x, y, z, visibility}
  - MelodicCapAntiGrav old: landmarks_3d as dict {"0": [x,y,z]}
  - MelodicCapFresh v8: landmarks as dict {"0": [x,y,z]}
  - MelodicCapAntiGrav recorder: landmarks_3d as dict + raw_2d

Output format:
  frames[].landmarks = {"0": [x, y, z], "1": [x, y, z], ...}
"""

import json
import os
import sys
from pathlib import Path


def convert_take(input_path, output_path=None):
    """Convert a single take file to unified format."""
    with open(input_path, 'r') as f:
        data = json.load(f)

    frames = data.get('frames', [])
    if not frames:
        print(f"  SKIP: No frames in {input_path}")
        return None

    converted_frames = []
    format_detected = "unknown"

    for frame in frames:
        new_frame = {
            'timestamp': frame.get('timestamp', 0),
            'landmarks': {},
        }

        # Try unified format first (MelodicCapFresh v8)
        lms = frame.get('landmarks')
        if lms and isinstance(lms, dict) and lms:
            format_detected = "v8_unified"
            new_frame['landmarks'] = {str(k): v for k, v in lms.items()}

        # Try landmarks_3d dict format (AntiGrav)
        elif frame.get('landmarks_3d'):
            lms3d = frame['landmarks_3d']

            if isinstance(lms3d, dict) and lms3d:
                format_detected = "antigrav_dict"
                new_frame['landmarks'] = {str(k): v for k, v in lms3d.items()}

            elif isinstance(lms3d, list) and lms3d:
                format_detected = "justice_list"
                for item in lms3d:
                    if isinstance(item, dict) and 'index' in item:
                        new_frame['landmarks'][str(item['index'])] = [
                            item['x'], item['y'], item['z']
                        ]

        # Preserve hands data if present
        if frame.get('hands_3d'):
            new_frame['hands_3d'] = frame['hands_3d']

        # Preserve raw 2D for possible re-triangulation
        raw_2d = frame.get('raw_2d', frame.get('landmarks_2d'))
        if raw_2d:
            new_frame['raw_2d'] = raw_2d

        converted_frames.append(new_frame)

    # Build output
    calib = data.get('calibration', {})
    output = {
        'version': 'unified_v1',
        'original_version': data.get('version', format_detected),
        'fps': data.get('fps', 30),
        'frame_count': len(converted_frames),
        'duration': data.get('duration', data.get('duration_seconds', 0)),
        'calibration': {
            'floor_z_offset': calib.get('floor_z_offset', calib.get('floor_offset', 0.0)),
            'rms_stereo': calib.get('rms_stereo'),
            'baseline': calib.get('baseline', calib.get('baseline_meters')),
        },
        'frames': converted_frames,
    }

    # Determine output path
    if output_path is None:
        stem = Path(input_path).stem
        parent = Path(input_path).parent
        output_path = str(parent / f"{stem}_unified.json")

    with open(output_path, 'w') as f:
        json.dump(output, f)

    # Stats
    non_empty = sum(1 for f in converted_frames if f['landmarks'])
    print(f"  Converted: {input_path}")
    print(f"    Format: {format_detected}")
    print(f"    Frames: {len(converted_frames)} ({non_empty} with 3D data)")
    print(f"    Output: {output_path}")

    return output_path


def batch_convert(directory):
    """Convert all JSON takes in a directory."""
    dir_path = Path(directory)
    files = sorted(dir_path.glob("*.json"))
    # Skip already-converted files
    files = [f for f in files if '_unified' not in f.stem and '_refined' not in f.stem]

    print(f"Found {len(files)} take files in {directory}")
    print("=" * 60)

    converted = 0
    for f in files:
        try:
            result = convert_take(str(f))
            if result:
                converted += 1
        except Exception as e:
            print(f"  ERROR: {f.name}: {e}")

    print("=" * 60)
    print(f"Converted {converted}/{len(files)} takes")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python convert_takes.py <input.json> [output.json]")
        print("  python convert_takes.py --batch <directory>")
        sys.exit(1)

    if sys.argv[1] == '--batch':
        if len(sys.argv) < 3:
            print("Provide directory path")
            sys.exit(1)
        batch_convert(sys.argv[2])
    else:
        output = sys.argv[2] if len(sys.argv) > 2 else None
        convert_take(sys.argv[1], output)
