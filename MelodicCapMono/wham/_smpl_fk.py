"""Shared SMPL primitives + forward kinematics.

Used by:
- pose_json_to_bvh.py (BVH export consumes aa_to_matrix and the rest
  joint template as bone OFFSETs in the HIERARCHY block).
- orchestrate/footlock.py (foot-contact detection needs per-frame world
  joint positions in pure numpy -- no torch, no Blender).

Pure numpy + math. SMPL canonical frame: +Y up, +X to subject's left,
+Z forward (the SMPL body stands along +Y). Floor calibration in
footlock is data-relative so it self-calibrates regardless of frame;
HEIGHT_AXIS = 1 makes the up axis explicit for downstream code.
"""

from __future__ import annotations

import math

import numpy as np


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

# SMPL neutral mean-shape rest joint locations (meters, SMPL frame +Y up).
# Approximate; downstream code only needs offsets + a consistent frame.
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

# Index of the "up" axis in SMPL canonical frame. Footlock measures
# ankle height along this axis when detecting plants.
HEIGHT_AXIS = 1


def aa_to_matrix(aa) -> np.ndarray:
    """Axis-angle (3,) radians -> (3,3) rotation matrix (Rodrigues)."""
    x, y, z = float(aa[0]), float(aa[1]), float(aa[2])
    theta = math.sqrt(x * x + y * y + z * z)
    if theta < 1e-12:
        return np.eye(3)
    kx, ky, kz = x / theta, y / theta, z / theta
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def euler_deg_to_matrix(xd, yd, zd) -> np.ndarray:
    """Rz @ Ry @ Rx (degrees) — same composition as pose_json_to_bvh's
    global-rotation correction."""
    x, y, z = math.radians(xd), math.radians(yd), math.radians(zd)
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def forward_kinematics(pose_data, global_rot_deg=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Per-frame world joint positions from a melodiccap_mono_v1 pose dict.

    Returns ndarray shape (n_frames, 24, 3) in SMPL canonical frame
    (+Y up). With default global_rot_deg=(0,0,0), output is in raw SMPL
    frame -- sufficient for foot-contact detection because the floor is
    calibrated from the data itself per take.

    SMPL FK identity (walked once per frame, parent-before-child since
    SMPL_PARENTS is topologically ordered):
        T_world[0] = trans                   (offsets[0] is (0,0,0))
        R_world[0] = R_local[0]
        for j >= 1:
            T_world[j] = T_world[p] + R_world[p] @ (rest[j] - rest[p])
            R_world[j] = R_world[p] @ R_local[j]
    where R_local[j] = aa_to_matrix(pose[j*3:(j+1)*3]), and root rotation
    is optionally pre-multiplied by R_global = euler_deg_to_matrix(*global_rot_deg).
    """
    frames = pose_data["frames"]
    n = len(frames)
    parents = SMPL_PARENTS
    rest = SMPL_REST_JOINTS

    R_global = euler_deg_to_matrix(*global_rot_deg)

    # Precompute per-joint rest offset from parent. offsets[0] = (0,0,0).
    offsets = np.zeros_like(rest)
    for j in range(24):
        p = parents[j]
        if p >= 0:
            offsets[j] = rest[j] - rest[p]

    out = np.zeros((n, 24, 3), dtype=np.float64)

    for fi, fr in enumerate(frames):
        pose = fr["smpl_pose"]
        if len(pose) != 72:
            raise ValueError(
                f"frame {fi}: smpl_pose length {len(pose)} != 72")
        trans = np.asarray(fr.get("smpl_trans", [0.0, 0.0, 0.0]),
                           dtype=np.float64)
        global_orient = fr.get("smpl_global_orient")

        R_local = [aa_to_matrix(pose[j * 3:(j + 1) * 3]) for j in range(24)]
        if global_orient is not None:
            R_local[0] = aa_to_matrix(global_orient)
        R_local[0] = R_global @ R_local[0]
        trans_world = R_global @ trans

        R_world = [None] * 24
        T_world = [None] * 24
        R_world[0] = R_local[0]
        T_world[0] = trans_world  # offsets[0] is (0,0,0)
        for j in range(1, 24):
            p = parents[j]
            R_world[j] = R_world[p] @ R_local[j]
            T_world[j] = T_world[p] + R_world[p] @ offsets[j]

        for j in range(24):
            out[fi, j] = T_world[j]

    return out
