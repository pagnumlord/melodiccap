"""
Dump 2D keypoint positions from each camera per frame, to identify
whether the bug is detector confusion (2D keypoints at wrong place)
or triangulation noise (2D correct, 3D wrong).

Reports the 2D pixel distance between child→parent pairs:
  wrist→shoulder, ankle→hip  (per camera, per side)

If wrist_2D is consistently near shoulder_2D (e.g. <50px) in EITHER
camera, the detector is confused — model can't localize wrist and
collapses it onto shoulder. Same for ankle→hip.

Healthy 2D for a 720p frame at ~2m subject distance: arm spans
~150-250 pixels from shoulder to wrist when extended.

Usage:
    python diagnose_2d.py takes/take_NAME_raw.json
"""
import json
import sys
import math


SHOULDER_L, SHOULDER_R = 5, 6
ELBOW_L, ELBOW_R = 7, 8
WRIST_L, WRIST_R = 9, 10
HIP_L, HIP_R = 11, 12
KNEE_L, KNEE_R = 13, 14
ANKLE_L, ANKLE_R = 15, 16


def get_2d(detections, idx):
    """detections is either:
       - dict {idx_str: [x, y, conf]} from one camera
       - list of [x, y] or [x, y, conf] indexed by idx
    """
    if detections is None:
        return None
    if isinstance(detections, dict):
        val = detections.get(str(idx)) or detections.get(idx)
    elif isinstance(detections, list):
        if idx >= len(detections):
            return None
        val = detections[idx]
    else:
        return None
    if val is None:
        return None
    return (float(val[0]), float(val[1]))


def dist_2d(a, b):
    if a is None or b is None:
        return None
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def summarize_pair(label, child_idx, parent_idx, frames, cam_key):
    """Compute distance child→parent across all frames for one camera."""
    dists = []
    for f in frames:
        det = f.get(cam_key)
        if det is None:
            continue
        c = get_2d(det, child_idx)
        p = get_2d(det, parent_idx)
        d = dist_2d(c, p)
        if d is not None:
            dists.append(d)

    if not dists:
        return f"  {label} ({cam_key}): NO DATA"

    n = len(dists)
    mn = min(dists)
    mx = max(dists)
    mean = sum(dists) / n
    near_parent = sum(1 for d in dists if d < 50)
    pct_near = 100 * near_parent / n

    flag = ""
    if mean < 60:
        flag = "  ← STUCK NEAR PARENT (detector confusion)"
    elif pct_near > 30:
        flag = f"  ← {pct_near:.0f}% of frames child is <50px from parent"

    return (f"  {label} ({cam_key}): "
            f"min={mn:.0f}px  max={mx:.0f}px  mean={mean:.0f}px  "
            f"<50px: {near_parent}/{n} ({pct_near:.0f}%){flag}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_2d.py <take_raw.json>")
        sys.exit(1)

    path = sys.argv[1]
    if not path.endswith("_raw.json"):
        print("ERROR: Run on the RAW take file (with _raw in name).")
        print("       The 2D detections are stored only in raw, not solved.")
        sys.exit(1)

    with open(path) as f:
        take = json.load(f)
    frames = take["frames"]

    # Try common keys for per-camera 2D detections
    sample = frames[0]
    cam_keys = []
    for k in ("raw_2d_a", "detections_a", "landmarks_2d_a", "kpts_a", "detections_2d_a"):
        if k in sample:
            cam_keys.append(k)
            break
    for k in ("raw_2d_b", "detections_b", "landmarks_2d_b", "kpts_b", "detections_2d_b"):
        if k in sample:
            cam_keys.append(k)
            break

    if len(cam_keys) < 2:
        print(f"\nFrame keys present: {list(sample.keys())}")
        print("\nCould not find per-camera 2D detection keys. The raw take")
        print("file does not contain 2D detections in a recognized format.")
        print("Look at the keys above and tell me which ones hold 2D data.")
        sys.exit(1)

    print(f"\n{path}")
    print(f"Total frames: {len(frames)}")
    print(f"Camera keys: {cam_keys}\n")

    print("=" * 80)
    print("2D distance (pixels): child keypoint → parent keypoint")
    print("Healthy values for a person 2m from a 720p camera:")
    print("  wrist→shoulder: ~150-250px when arm extended (A-pose)")
    print("  ankle→hip:      ~250-350px when standing")
    print("  <50px = child stuck on parent (detector confusion)")
    print("=" * 80)

    for cam_key in cam_keys:
        print(f"\n--- Camera {cam_key.upper()} ---")
        print(summarize_pair("L wrist → L shoulder", WRIST_L, SHOULDER_L,
                             frames, cam_key))
        print(summarize_pair("R wrist → R shoulder", WRIST_R, SHOULDER_R,
                             frames, cam_key))
        print(summarize_pair("L ankle → L hip      ", ANKLE_L, HIP_L,
                             frames, cam_key))
        print(summarize_pair("R ankle → R hip      ", ANKLE_R, HIP_R,
                             frames, cam_key))

    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("- If a row says STUCK NEAR PARENT: that camera + that joint = the bug source.")
    print("- If 2D is healthy in BOTH cameras but 3D is wrong, bug is triangulation.")
    print("- If 2D is bad in ONE camera, that camera's view of that limb is the issue")
    print("  (could be lighting, distance, angle, occlusion of small body part).")


if __name__ == "__main__":
    main()
