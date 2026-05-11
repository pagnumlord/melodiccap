"""
Diagnostic: dump first N frames of left/right hip & ankle 3D positions
to identify the systematic foot-Z asymmetry bug.

Usage:
    python diagnose_ankle.py takes/take_20260426_222320.json
    python diagnose_ankle.py takes/take_20260426_222320.json 60   # dump 60 frames
"""
import json
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_ankle.py <take.json> [num_frames]")
        sys.exit(1)

    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    with open(path) as f:
        t = json.load(f)

    frames = t["frames"]
    print(f"\n{path}")
    print(f"Total frames: {len(frames)}")
    print(f"Dumping first {min(n, len(frames))} frames\n")

    # Header: full XYZ for left hip vs left ankle, plus identity check
    print(f"{'frm':>4}  {'hip_l (xyz)':>26}  {'ankle_l (xyz)':>26}  "
          f"{'identical?':>11}  {'dist':>6}")
    print("-" * 90)

    identical_count = 0
    near_count = 0  # within 1mm

    for i, f in enumerate(frames[:n]):
        lm = f.get("landmarks_3d", {})
        hl = lm.get("11")
        al = lm.get("15")

        if hl is None or al is None:
            print(f"f{i:3d}: missing keypoint(s)")
            continue

        # Distance between left hip and left ankle (should be ~0.85m for a
        # standing adult — not zero!)
        dx = hl[0] - al[0]
        dy = hl[1] - al[1]
        dz = hl[2] - al[2]
        dist = (dx*dx + dy*dy + dz*dz) ** 0.5

        identical = (hl == al)
        near = dist < 0.001

        if identical:
            identical_count += 1
            tag = "EXACT"
        elif near:
            near_count += 1
            tag = "<1mm"
        elif dist < 0.05:
            tag = "<5cm"
        else:
            tag = ""

        print(f"f{i:3d}: "
              f"({hl[0]:+.3f},{hl[1]:+.3f},{hl[2]:+.3f})  "
              f"({al[0]:+.3f},{al[1]:+.3f},{al[2]:+.3f})  "
              f"{tag:>11}  {dist:.4f}")

    print(f"\nFrames where hip_l == ankle_l (exact): {identical_count}/{min(n, len(frames))}")
    print(f"Frames where hip_l ~= ankle_l (<1mm):   {near_count}/{min(n, len(frames))}")

    # Full-take stats on left ankle Z
    al_z = [f["landmarks_3d"].get("15", [0, 0, 0])[2]
            for f in frames if "15" in f.get("landmarks_3d", {})]
    ar_z = [f["landmarks_3d"].get("16", [0, 0, 0])[2]
            for f in frames if "16" in f.get("landmarks_3d", {})]

    if al_z and ar_z:
        print(f"\nFull-take left ankle Z:  min={min(al_z):+.3f} "
              f"max={max(al_z):+.3f} mean={sum(al_z)/len(al_z):+.3f}")
        print(f"Full-take right ankle Z: min={min(ar_z):+.3f} "
              f"max={max(ar_z):+.3f} mean={sum(ar_z)/len(ar_z):+.3f}")


if __name__ == "__main__":
    main()
