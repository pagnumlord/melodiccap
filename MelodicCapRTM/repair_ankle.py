"""
Repair takes where one ankle was triangulated near its hip due to
occlusion in one camera. The bug is bidirectional — sometimes left,
sometimes right, depending on which foot is occluded from camera A's
view in any given frame.

For each frame, if EITHER ankle is anatomically impossible (within 30cm
of its same-side hip in Z), and the OTHER side has a plausible ankle,
mirror the good ankle across the hip midline to repair the bad one.
Same for knees if also broken.

Imperfect (loses real left-vs-right walking asymmetry on repaired
frames) but gets the take past v5.7's foot-Z asymmetry hard block so
the rest can be cleaned up with keyframes in Blender.

Usage:
    python repair_ankle.py takes/take_NAME.json
    # → writes takes/take_NAME_repaired.json
"""
import json
import sys


def is_broken(hip, ankle, threshold=0.30):
    """Anatomically impossible ankle position: ankle within `threshold`
    meters of hip in Z (real ankle is ~0.85m below hip)."""
    if hip is None or ankle is None:
        return False
    return abs(hip[2] - ankle[2]) < threshold


def mirror_across_hips(left_hip, right_hip, src_pt):
    """Mirror src point across the hip midline.
    Preserves Y (forward/back) and Z (height); flips X around mid-X."""
    mid_x = (left_hip[0] + right_hip[0]) / 2.0
    mirror_x = 2 * mid_x - src_pt[0]
    return [mirror_x, src_pt[1], src_pt[2]]


def repair_take(in_path, out_path):
    with open(in_path) as f:
        take = json.load(f)

    frames = take["frames"]
    n_total = len(frames)

    counts = {"L_ankle": 0, "R_ankle": 0, "L_knee": 0, "R_knee": 0,
              "skipped": 0, "both_bad": 0}

    for frame in frames:
        lm = frame.get("landmarks_3d", {})

        hl = lm.get("11")  # left_hip
        hr = lm.get("12")  # right_hip
        kl = lm.get("13")  # left_knee
        kr = lm.get("14")  # right_knee
        al = lm.get("15")  # left_ankle
        ar = lm.get("16")  # right_ankle

        if hl is None or hr is None:
            counts["skipped"] += 1
            continue

        # Repair ankles bidirectionally
        l_bad = is_broken(hl, al)
        r_bad = is_broken(hr, ar)
        if l_bad and r_bad:
            counts["both_bad"] += 1  # can't repair, no good reference
        elif l_bad and ar is not None:
            lm["15"] = mirror_across_hips(hl, hr, ar)
            counts["L_ankle"] += 1
        elif r_bad and al is not None:
            lm["16"] = mirror_across_hips(hl, hr, al)
            counts["R_ankle"] += 1

        # Repair knees bidirectionally (looser threshold — knees CAN be
        # near hip in some sit poses, so 0.20m not 0.30m)
        l_knee_bad = is_broken(hl, kl, threshold=0.20)
        r_knee_bad = is_broken(hr, kr, threshold=0.20)
        if l_knee_bad and not r_knee_bad and kr is not None:
            lm["13"] = mirror_across_hips(hl, hr, kr)
            counts["L_knee"] += 1
        elif r_knee_bad and not l_knee_bad and kl is not None:
            lm["14"] = mirror_across_hips(hl, hr, kl)
            counts["R_knee"] += 1

    with open(out_path, "w") as f:
        json.dump(take, f, indent=2)

    print(f"\n[REPAIR] {in_path}")
    print(f"  Total frames:        {n_total}")
    print(f"  L ankle repaired:    {counts['L_ankle']:>4}  "
          f"({100*counts['L_ankle']/max(1,n_total):.1f}%)")
    print(f"  R ankle repaired:    {counts['R_ankle']:>4}  "
          f"({100*counts['R_ankle']/max(1,n_total):.1f}%)")
    print(f"  L knee repaired:     {counts['L_knee']:>4}  "
          f"({100*counts['L_knee']/max(1,n_total):.1f}%)")
    print(f"  R knee repaired:     {counts['R_knee']:>4}  "
          f"({100*counts['R_knee']/max(1,n_total):.1f}%)")
    if counts["both_bad"]:
        print(f"  BOTH ankles bad:     {counts['both_bad']:>4}  "
              f"(unrepairable — both sides occluded)")
    if counts["skipped"]:
        print(f"  Skipped:             {counts['skipped']:>4}  "
              f"(missing hip data)")
    print(f"  Output:              {out_path}")


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
