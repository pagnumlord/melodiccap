"""
Repair takes where the LEFT ANKLE was triangulated near the LEFT HIP
(rtmlib 'fast' / RTMPose-Lite lower-body keypoint bias).

For each frame, if left_ankle is anatomically impossible (within 30cm
of left_hip in Z), replace it with a mirror of right_ankle across the
hip midline. Same for left_knee if also broken.

Imperfect (loses real left-vs-right walking asymmetry) but gets the
take past v5.7's foot-Z asymmetry hard block so the rest can be cleaned
up with keyframes in Blender.

Usage:
    python repair_ankle.py takes/take_NAME.json
    # → writes takes/take_NAME_repaired.json
"""
import json
import sys
import copy


def is_broken(hip, ankle, threshold=0.30):
    """Anatomically impossible ankle position: ankle within `threshold`
    meters of hip in Z (real ankle is ~0.85m below hip)."""
    if hip is None or ankle is None:
        return False
    return abs(hip[2] - ankle[2]) < threshold


def mirror_across_hips(left_hip, right_hip, right_pt):
    """Mirror a right-side point across the hip midline.
    Preserves Y (forward/back) and Z (height); flips X around mid-X."""
    mid_x = (left_hip[0] + right_hip[0]) / 2.0
    mirror_x = 2 * mid_x - right_pt[0]
    return [mirror_x, right_pt[1], right_pt[2]]


def repair_take(in_path, out_path):
    with open(in_path) as f:
        take = json.load(f)

    frames = take["frames"]
    n_total = len(frames)
    n_ankle_repaired = 0
    n_knee_repaired = 0
    n_skipped = 0

    for frame in frames:
        lm = frame.get("landmarks_3d", {})

        hl = lm.get("11")  # left_hip
        hr = lm.get("12")  # right_hip
        kl = lm.get("13")  # left_knee
        kr = lm.get("14")  # right_knee
        al = lm.get("15")  # left_ankle
        ar = lm.get("16")  # right_ankle

        # Need both hips and the right side to mirror from
        if hl is None or hr is None or ar is None:
            n_skipped += 1
            continue

        # Repair left ankle if broken
        if is_broken(hl, al):
            lm["15"] = mirror_across_hips(hl, hr, ar)
            n_ankle_repaired += 1

        # Repair left knee if broken (and we have right knee to mirror)
        if kr is not None and is_broken(hl, kl, threshold=0.20):
            lm["13"] = mirror_across_hips(hl, hr, kr)
            n_knee_repaired += 1

    with open(out_path, "w") as f:
        json.dump(take, f, indent=2)

    print(f"\n[REPAIR] {in_path}")
    print(f"  Total frames:           {n_total}")
    print(f"  Left ankle repaired:    {n_ankle_repaired}/{n_total} "
          f"({100*n_ankle_repaired/max(1,n_total):.1f}%)")
    print(f"  Left knee repaired:     {n_knee_repaired}/{n_total} "
          f"({100*n_knee_repaired/max(1,n_total):.1f}%)")
    print(f"  Skipped (missing data): {n_skipped}")
    print(f"  Output:                 {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python repair_ankle.py <take.json>")
        sys.exit(1)

    in_path = sys.argv[1]
    if in_path.endswith("_raw.json"):
        print("ERROR: Run repair_ankle on the SOLVED take JSON "
              "(no '_raw' in name), not the raw triangulation file.")
        sys.exit(1)

    out_path = in_path.replace(".json", "_repaired.json")
    repair_take(in_path, out_path)


if __name__ == "__main__":
    main()
