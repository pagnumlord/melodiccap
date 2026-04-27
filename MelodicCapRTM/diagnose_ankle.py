"""
Diagnostic: dump first N frames of left/right hip & ankle Z values
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

    print(f"{'frm':>4} {'hip_l_z':>8} {'hip_r_z':>8} "
          f"{'ank_l_z':>8} {'ank_r_z':>8} {'L_diff':>8} {'R_diff':>8}")
    print("-" * 60)

    for i, f in enumerate(frames[:n]):
        lm = f.get("landmarks_3d", {})
        hl = lm.get("11", [0, 0, None])[2]
        hr = lm.get("12", [0, 0, None])[2]
        al = lm.get("15", [0, 0, None])[2]
        ar = lm.get("16", [0, 0, None])[2]

        if any(v is None for v in (hl, hr, al, ar)):
            print(f"f{i:3d}: missing keypoint(s)")
            continue

        # Diff = hip_z - ankle_z (should be ~0.85m for an adult standing)
        l_diff = hl - al
        r_diff = hr - ar

        print(f"f{i:3d}: {hl:+8.3f} {hr:+8.3f} {al:+8.3f} {ar:+8.3f} "
              f"{l_diff:+8.3f} {r_diff:+8.3f}")

    # Summary
    al_vals = [f["landmarks_3d"].get("15", [0, 0, 0])[2]
               for f in frames if "15" in f.get("landmarks_3d", {})]
    ar_vals = [f["landmarks_3d"].get("16", [0, 0, 0])[2]
               for f in frames if "16" in f.get("landmarks_3d", {})]

    if al_vals and ar_vals:
        print(f"\nFull-take left ankle Z:  min={min(al_vals):+.3f} "
              f"max={max(al_vals):+.3f} mean={sum(al_vals)/len(al_vals):+.3f} "
              f"frames_present={len(al_vals)}/{len(frames)}")
        print(f"Full-take right ankle Z: min={min(ar_vals):+.3f} "
              f"max={max(ar_vals):+.3f} mean={sum(ar_vals)/len(ar_vals):+.3f} "
              f"frames_present={len(ar_vals)}/{len(frames)}")


if __name__ == "__main__":
    main()
