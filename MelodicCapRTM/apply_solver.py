"""
Apply skeleton solver v2 to existing takes (including legacy MediaPipe format).

Reads a take JSON, converts landmarks to COCO-17 format if needed,
runs the skeleton solver, then writes back in the original format.

Usage:
    python apply_solver.py path/to/take.json
    python apply_solver.py path/to/take.json --no-skeleton   # bypass solver (passthrough)
    python apply_solver.py path/to/*.json                    # batch mode
"""

import json
import sys
import os
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skeleton_solver import SkeletonSolver
from keypoint_map import LM

MP_NAME_TO_COCO = {
    'nose': LM.NOSE,
    'left_eye': LM.LEFT_EYE,
    'right_eye': LM.RIGHT_EYE,
    'left_ear': LM.LEFT_EAR,
    'right_ear': LM.RIGHT_EAR,
    'left_shoulder': LM.LEFT_SHOULDER,
    'right_shoulder': LM.RIGHT_SHOULDER,
    'left_elbow': LM.LEFT_ELBOW,
    'right_elbow': LM.RIGHT_ELBOW,
    'left_wrist': LM.LEFT_WRIST,
    'right_wrist': LM.RIGHT_WRIST,
    'left_hip': LM.LEFT_HIP,
    'right_hip': LM.RIGHT_HIP,
    'left_knee': LM.LEFT_KNEE,
    'right_knee': LM.RIGHT_KNEE,
    'left_ankle': LM.LEFT_ANKLE,
    'right_ankle': LM.RIGHT_ANKLE,
    'left_heel': LM.LEFT_HEEL,
    'right_heel': LM.RIGHT_HEEL,
    'left_foot_index': LM.LEFT_BIG_TOE,
    'right_foot_index': LM.RIGHT_BIG_TOE,
}

COCO_TO_MP_NAME = {v: k for k, v in MP_NAME_TO_COCO.items()}


def detect_format(frame):
    lm = frame.get('landmarks_3d', {})
    if not lm:
        return 'empty'
    first_key = next(iter(lm))
    first_val = lm[first_key]
    if isinstance(first_val, dict) and 'x' in first_val:
        return 'mediapipe_named'
    if isinstance(first_val, list):
        try:
            int(first_key)
            return 'coco_indexed'
        except ValueError:
            return 'unknown'
    return 'unknown'


def mp_named_to_coco(landmarks_3d):
    converted = {}
    for name, val in landmarks_3d.items():
        if name in MP_NAME_TO_COCO:
            idx = MP_NAME_TO_COCO[name]
            converted[str(idx)] = [val['x'], val['y'], val['z']]
    return converted


def coco_to_mp_named(coco_lm, original_lm):
    result = dict(original_lm)
    for key, coords in coco_lm.items():
        idx = int(key)
        if idx in COCO_TO_MP_NAME:
            name = COCO_TO_MP_NAME[idx]
            if name in result:
                result[name] = dict(result[name])
                result[name]['x'] = coords[0]
                result[name]['y'] = coords[1]
                result[name]['z'] = coords[2]
    return result


def process_take(input_path, no_skeleton=False):
    with open(input_path) as f:
        data = json.load(f)

    frames = data.get('frames', [])
    if not frames:
        print(f"  No frames in {input_path}")
        return None

    fmt = detect_format(frames[0])
    print(f"  Format: {fmt}, Frames: {len(frames)}")

    if fmt == 'mediapipe_named':
        coco_frames = []
        for frame in frames:
            coco_frame = dict(frame)
            coco_frame['landmarks_3d'] = mp_named_to_coco(frame['landmarks_3d'])
            coco_frames.append(coco_frame)
    elif fmt == 'coco_indexed':
        coco_frames = frames
    else:
        print(f"  Unknown format — skipping")
        return None

    if no_skeleton:
        print(f"  --no-skeleton: passing through without solver")
        solved_frames = coco_frames
    else:
        solver = SkeletonSolver(n_cal_frames=30)
        solved_frames = solver.solve_sequence(coco_frames)

    if fmt == 'mediapipe_named':
        for i, frame in enumerate(solved_frames):
            solved_lm = frame['landmarks_3d']
            original_lm = frames[i]['landmarks_3d']
            frame['landmarks_3d'] = coco_to_mp_named(solved_lm, original_lm)

    data['frames'] = solved_frames
    if not no_skeleton:
        data.setdefault('metadata', {})['skeleton_solver_v2'] = True

    base, ext = os.path.splitext(input_path)
    suffix = '_noskel' if no_skeleton else '_solved'
    output_path = f"{base}{suffix}{ext}"

    with open(output_path, 'w') as f:
        json.dump(data, f)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Written: {output_path} ({size_mb:.1f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Apply skeleton solver v2 to takes')
    parser.add_argument('inputs', nargs='+', help='Take JSON files (supports glob)')
    parser.add_argument('--no-skeleton', action='store_true', help='Skip solver (passthrough)')
    args = parser.parse_args()

    files = []
    for pattern in args.inputs:
        expanded = glob.glob(pattern)
        files.extend(expanded if expanded else [pattern])

    print(f"\nProcessing {len(files)} take(s)...\n")
    for path in sorted(files):
        print(f"Processing: {path}")
        process_take(path, no_skeleton=args.no_skeleton)
        print()


if __name__ == '__main__':
    main()
