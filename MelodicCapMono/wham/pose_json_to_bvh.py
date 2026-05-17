"""Export a melodiccap_mono_v1 pose JSON to BVH on the canonical SMPL skeleton.

WHY THIS EXISTS:
    The Phase C addon tried to apply SMPL joint rotations directly to
    Rigify FK control bones. That is wrong: every Rigify bone has its
    own rest orientation, so the rotations land in the wrong basis and
    the body folds. The fix is NOT more axis guessing — it is to hand a
    *correct* SMPL-skeleton animation to a proven skeleton-to-skeleton
    retargeter (Rokoko / Auto-Rig Pro), which owns that hard math.

    BVH on the SMPL skeleton is correct *by construction*: SMPL pose
    params are, by definition, the per-joint local rotations of the SMPL
    kinematic tree relative to the SMPL rest (T-pose). Written onto that
    same skeleton there is no rest-frame mismatch and no axis convention
    to guess. Every retargeter reads BVH.

    This module is pure numpy + stdlib (no bpy, no WHAM imports) so it
    runs anywhere and is unit-testable without Blender.

USAGE:
    python -m MelodicCapMono.wham.pose_json_to_bvh in.pose.json out.bvh
    python -m MelodicCapMono.wham.pose_json_to_bvh in.pose.json out.bvh --root-motion zero

The skeleton is the SMPL *mean-shape* rest pose. Exact bone lengths are
not important: a skeleton-to-skeleton retargeter transfers rotations and
re-proportions to the target rig. The hierarchy and the per-frame
rotations are what must be exact, and those come straight from the SMPL
kinematic tree and the pose params.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


# SMPL 24-joint kinematic tree. Index i's parent is SMPL_PARENTS[i]
# (-1 = root). Order matches smpl_pose: joint i's axis-angle is
# smpl_pose[i*3:(i+1)*3].
SMPL_JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot",
    "right_foot", "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hand", "right_hand",
)
SMPL_PARENTS = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
    16, 17, 18, 19, 20, 21,
)

# SMPL neutral mean-shape rest joints, meters, SMPL frame (+X = subject
# left, +Y = up, +Z = forward). T-pose: spine along +Y, arms along X,
# legs along -Y. Approximate; only relative offsets and topology matter
# for a retarget source (lengths get re-proportioned to the target rig).
SMPL_REST_JOINTS = np.array([
    [ 0.0000,  0.0000,  0.0000],  # 0  pelvis
    [ 0.0586, -0.0820,  0.0140],  # 1  left_hip
    [-0.0589, -0.0827,  0.0135],  # 2  right_hip
    [ 0.0044,  0.1244, -0.0383],  # 3  spine1
    [ 0.1058, -0.4927,  0.0085],  # 4  left_knee
    [-0.1064, -0.4936,  0.0061],  # 5  right_knee
    [ 0.0047,  0.2728, -0.0383],  # 6  spine2
    [ 0.0855, -0.8865, -0.0265],  # 7  left_ankle
    [-0.0861, -0.8893, -0.0234],  # 8  right_ankle
    [ 0.0061,  0.3268, -0.0163],  # 9  spine3
    [ 0.1233, -0.9430,  0.1158],  # 10 left_foot
    [-0.1199, -0.9445,  0.1141],  # 11 right_foot
    [ 0.0028,  0.5375, -0.0410],  # 12 neck
    [ 0.0792,  0.4549, -0.0228],  # 13 left_collar
    [-0.0760,  0.4565, -0.0232],  # 14 right_collar
    [ 0.0070,  0.6336,  0.0146],  # 15 head
    [ 0.1827,  0.4602, -0.0375],  # 16 left_shoulder
    [-0.1771,  0.4567, -0.0407],  # 17 right_shoulder
    [ 0.4576,  0.4509, -0.0501],  # 18 left_elbow
    [-0.4561,  0.4495, -0.0490],  # 19 right_elbow
    [ 0.7138,  0.4536, -0.0526],  # 20 left_wrist
    [-0.7158,  0.4521, -0.0517],  # 21 right_wrist
    [ 0.8062,  0.4555, -0.0568],  # 22 left_hand
    [-0.8089,  0.4541, -0.0560],  # 23 right_hand
], dtype=np.float64)

EXPECTED_FORMAT = "melodiccap_mono_v1"


def _children(parents):
    kids = {i: [] for i in range(len(parents))}
    for i, p in enumerate(parents):
        if p >= 0:
            kids[p].append(i)
    return kids


def _dfs_order(parents):
    """Depth-first joint order (root first). HIERARCHY declaration order
    AND per-frame channel order must both follow this."""
    kids = _children(parents)
    order = []

    def visit(i):
        order.append(i)
        for c in kids[i]:
            visit(c)

    visit(0)
    return order


def _aa_to_matrix(aa):
    """Axis-angle (3,) radians -> (3,3) rotation matrix (Rodrigues)."""
    x, y, z = float(aa[0]), float(aa[1]), float(aa[2])
    theta = math.sqrt(x * x + y * y + z * z)
    if theta < 1e-12:
        return np.eye(3)
    kx, ky, kz = x / theta, y / theta, z / theta
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def _matrix_to_euler_zxy_deg(R):
    """Decompose R into BVH 'Zrotation Xrotation Yrotation' Euler degrees.

    BVH applies the listed channels as intrinsic rotations, so the local
    transform is R = Rz(z) @ Rx(x) @ Ry(y). Inverting that composition:
        R[2,1] =  sin(x)
        R[2,0] = -cos(x) sin(y)   R[2,2] = cos(x) cos(y)
        R[0,1] = -sin(z) cos(x)   R[1,1] = cos(z) cos(x)
    """
    sx = max(-1.0, min(1.0, R[2, 1]))
    x = math.asin(sx)
    cx = math.cos(x)
    if abs(cx) > 1e-6:
        z = math.atan2(-R[0, 1], R[1, 1])
        y = math.atan2(-R[2, 0], R[2, 2])
    else:
        # Gimbal lock (x = +/-90): fold y into z, pick y = 0.
        z = math.atan2(R[1, 0], R[0, 0])
        y = 0.0
    d = 180.0 / math.pi
    return z * d, x * d, y * d


def _load_pose_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    fmt = data.get("format")
    if fmt != EXPECTED_FORMAT:
        raise SystemExit(
            f"Pose JSON format={fmt!r}, expected {EXPECTED_FORMAT!r}."
        )
    if not data.get("frames"):
        raise SystemExit("Pose JSON has no frames.")
    return data


def _write_hierarchy(lines, joint, parents, kids, offsets, depth):
    pad = "  " * depth
    name = SMPL_JOINT_NAMES[joint]
    ox, oy, oz = offsets[joint]
    if joint == 0:
        lines.append(f"ROOT {name}")
        lines.append("{")
        lines.append(f"  OFFSET {ox:.6f} {oy:.6f} {oz:.6f}")
        lines.append("  CHANNELS 6 Xposition Yposition Zposition "
                      "Zrotation Xrotation Yrotation")
    else:
        lines.append(f"{pad}JOINT {name}")
        lines.append(f"{pad}{{")
        lines.append(f"{pad}  OFFSET {ox:.6f} {oy:.6f} {oz:.6f}")
        lines.append(f"{pad}  CHANNELS 3 Zrotation Xrotation Yrotation")
    if kids[joint]:
        for c in kids[joint]:
            _write_hierarchy(lines, c, parents, kids, offsets, depth + 1)
    else:
        # End Site: continue the bone a little so the leaf has length.
        ex, ey, ez = offsets[joint] * 0.5
        if abs(ex) + abs(ey) + abs(ez) < 1e-6:
            ey = 0.05
        lines.append(f"{pad}  End Site")
        lines.append(f"{pad}  {{")
        lines.append(f"{pad}    OFFSET {ex:.6f} {ey:.6f} {ez:.6f}")
        lines.append(f"{pad}  }}")
    close_pad = "  " * depth
    lines.append(f"{close_pad}}}")


def convert(pose_data, *, root_motion="trans"):
    """Return BVH text for a melodiccap_mono_v1 pose dict.

    root_motion: 'trans' writes smpl_trans into the root position
    channels; 'zero' pins the root at origin (in-place; monocular trans
    is drift-prone — pin it and keyframe the path on the target rig).
    """
    frames = pose_data["frames"]
    fps = float(pose_data.get("fps", 30.0))

    parents = SMPL_PARENTS
    kids = _children(parents)
    order = _dfs_order(parents)

    # OFFSET[i] = rest position of joint i relative to its parent.
    offsets = np.zeros((24, 3), dtype=np.float64)
    for i in range(24):
        p = parents[i]
        offsets[i] = SMPL_REST_JOINTS[i] if p < 0 else (
            SMPL_REST_JOINTS[i] - SMPL_REST_JOINTS[p]
        )

    lines = ["HIERARCHY"]
    _write_hierarchy(lines, 0, parents, kids, offsets, 0)
    lines.append("MOTION")
    lines.append(f"Frames: {len(frames)}")
    lines.append(f"Frame Time: {1.0 / fps:.8f}")

    for fr in frames:
        pose = fr["smpl_pose"]
        if len(pose) != 72:
            raise SystemExit(
                f"frame {fr.get('frame')}: smpl_pose len {len(pose)} != 72"
            )
        vals = []
        tx, ty, tz = fr.get("smpl_trans", [0.0, 0.0, 0.0])
        if root_motion == "zero":
            tx = ty = tz = 0.0
        for j in order:
            aa = pose[j * 3:(j + 1) * 3]
            if j == 0 and "smpl_global_orient" in fr:
                aa = fr["smpl_global_orient"]
            z, x, y = _matrix_to_euler_zxy_deg(_aa_to_matrix(aa))
            if j == 0:
                vals += [f"{tx:.6f}", f"{ty:.6f}", f"{tz:.6f}"]
            vals += [f"{z:.6f}", f"{x:.6f}", f"{y:.6f}"]
        lines.append(" ".join(vals))

    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="MelodicCapMono.wham.pose_json_to_bvh",
        description="Convert a melodiccap_mono_v1 pose JSON to SMPL-skeleton BVH.",
    )
    ap.add_argument("input_json", type=Path, help="melodiccap_mono_v1 pose JSON")
    ap.add_argument("output_bvh", type=Path, help="output .bvh path")
    ap.add_argument("--root-motion", choices=("trans", "zero"),
                    default="trans",
                    help="'trans' (default) uses smpl_trans for the root; "
                         "'zero' pins the root in place.")
    args = ap.parse_args(argv)

    if not args.input_json.exists():
        raise SystemExit(f"ERROR: input not found: {args.input_json}")

    data = _load_pose_json(args.input_json)
    bvh = convert(data, root_motion=args.root_motion)
    args.output_bvh.parent.mkdir(parents=True, exist_ok=True)
    with args.output_bvh.open("w") as f:
        f.write(bvh)
    print(f"[pose_json_to_bvh] DONE. {len(data['frames'])} frames, "
          f"SMPL skeleton -> {args.output_bvh}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
