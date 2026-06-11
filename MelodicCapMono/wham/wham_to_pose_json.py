"""Convert a WHAM pickle into a melodiccap_mono_v1 pose JSON dict.

WHAM's exact pkl layout has shifted across releases. As of the CVPR
2024 release, the most common form is:

    results = {
        person_id (int): {
            'pose':       np.ndarray (N, 24, 3, 3) rotation matrices
                          OR (N, 72) flat axis-angle in radians
            'trans':      np.ndarray (N, 3) root translation in meters
            'betas':      np.ndarray (10,) constant body shape OR (N, 10)
            'frame_ids':  np.ndarray (N,) source-video frame indices
            'global_orient': np.ndarray (N, 3) or (N, 3, 3)  [optional]
        },
        ...
    }

Some forks/versions return a single-person dict (no person_id wrapper)
or a list. We accept all three shapes. If a future WHAM release changes
the dict keys we don't recognize, this module raises a clear error
naming the keys we DID see, so the fix is one rename here, not a re-do
of the pipeline.

This module is pure numpy — no torch, no bpy, no WHAM imports. So it
runs in any env that has numpy and is straightforward to unit-test
without standing up the WHAM stack (see scripts/test_wham_converter_smoke.py).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


SMPL_NUM_JOINTS = 24
SMPL_POSE_LEN = SMPL_NUM_JOINTS * 3  # axis-angle, flat


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Convert (..., 3, 3) rotation matrices to (..., 3) axis-angle.

    Uses the closed-form trace identity. Stable for angles away from
    pi; near-pi falls back to the eigenvector route.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"expected (..., 3, 3), got {R.shape}")
    flat = R.reshape(-1, 3, 3)
    out = np.zeros((flat.shape[0], 3), dtype=np.float64)
    for i in range(flat.shape[0]):
        Ri = flat[i]
        cos_theta = (np.trace(Ri) - 1.0) * 0.5
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        theta = math.acos(cos_theta)
        if theta < 1e-6:
            continue
        if abs(math.pi - theta) < 1e-3:
            # Near 180 deg: extract axis from the diagonal of (R + I) / 2.
            M = (Ri + np.eye(3)) * 0.5
            diag = np.array([M[0, 0], M[1, 1], M[2, 2]])
            k = int(np.argmax(diag))
            axis = M[:, k] / max(math.sqrt(max(diag[k], 0.0)), 1e-9)
            out[i] = axis * theta
            continue
        axis = np.array([
            Ri[2, 1] - Ri[1, 2],
            Ri[0, 2] - Ri[2, 0],
            Ri[1, 0] - Ri[0, 1],
        ]) / (2.0 * math.sin(theta))
        out[i] = axis * theta
    return out.reshape(R.shape[:-2] + (3,))


def _to_axis_angle_per_joint(arr: np.ndarray) -> np.ndarray:
    """Coerce per-joint pose into shape (N_frames, 24, 3) axis-angle.

    Accepts:
        (N, 24, 3, 3)  rotation matrices per joint
        (N, 24, 3)     axis-angle per joint
        (N, 72)        flat axis-angle (24 * 3)
    """
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[1] == SMPL_NUM_JOINTS and arr.shape[2:] == (3, 3):
        return _rotation_matrix_to_axis_angle(arr).astype(np.float64)
    if arr.ndim == 3 and arr.shape[1] == SMPL_NUM_JOINTS and arr.shape[2] == 3:
        return arr.astype(np.float64)
    if arr.ndim == 2 and arr.shape[1] == SMPL_POSE_LEN:
        return arr.reshape(arr.shape[0], SMPL_NUM_JOINTS, 3).astype(np.float64)
    raise ValueError(
        f"WHAM 'pose' has unexpected shape {arr.shape}; "
        f"expected (N, 24, 3, 3), (N, 24, 3), or (N, 72)"
    )


def _to_axis_angle_global(arr: np.ndarray) -> np.ndarray:
    """Coerce global_orient into shape (N_frames, 3) axis-angle."""
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[1:] == (3, 3):
        return _rotation_matrix_to_axis_angle(arr).astype(np.float64)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr.astype(np.float64)
    raise ValueError(
        f"WHAM 'global_orient' has unexpected shape {arr.shape}; "
        f"expected (N, 3) or (N, 3, 3)"
    )


# ---------------------------------------------------------------------------
# Person-track selection
# ---------------------------------------------------------------------------

def _select_person(wham_data: Any) -> dict:
    """Pick one person track from WHAM's output.

    WHAM is a single-person model but its pipeline detects and tracks
    multiple people. For solo-performer footage there's typically one
    track; if there are several, pick the longest (most frames). Print
    a warning so the user knows we made a choice.
    """
    def _looks_like_person(d) -> bool:
        return isinstance(d, dict) and ("pose" in d or "pose_world" in d)

    if isinstance(wham_data, dict):
        if 0 in wham_data and _looks_like_person(wham_data[0]):
            people = wham_data
        elif _looks_like_person(wham_data):
            return wham_data
        elif "results" in wham_data and isinstance(wham_data["results"], dict):
            people = wham_data["results"]
        else:
            keys = list(wham_data.keys())
            raise KeyError(
                f"WHAM pkl is a dict but doesn't look like {{person_id: data}} "
                f"or a single-person dict. Top-level keys: {keys[:10]}"
            )
    elif isinstance(wham_data, list):
        if not wham_data:
            raise ValueError("WHAM pkl is an empty list")
        people = {i: p for i, p in enumerate(wham_data)}
    else:
        raise TypeError(
            f"WHAM pkl is type {type(wham_data).__name__}; expected dict or list"
        )

    if len(people) == 1:
        return next(iter(people.values()))

    def n_frames(p: dict) -> int:
        for k in ("pose", "pose_world", "trans", "frame_ids"):
            if k in p:
                return int(np.asarray(p[k]).shape[0])
        return 0

    best_id = max(people.keys(), key=lambda k: n_frames(people[k]))
    others = [k for k in people.keys() if k != best_id]
    print(
        f"[wham_to_pose_json] Detected {len(people)} person tracks; "
        f"picking id={best_id} ({n_frames(people[best_id])} frames). "
        f"Discarded: {others}."
    )
    return people[best_id]


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def _resolve_betas(person: dict, n_frames: int) -> list[float]:
    """Return a 10-float SMPL beta vector. WHAM may store per-frame; we
    average to a constant since SMPL betas are body-shape, not motion."""
    if "betas" not in person:
        print("[wham_to_pose_json] WARNING: no 'betas' in WHAM output, "
              "using zeros (SMPL mean shape).")
        return [0.0] * 10
    b = np.asarray(person["betas"], dtype=np.float64)
    if b.ndim == 2 and b.shape[1] == 10:
        b = b.mean(axis=0)
    if b.ndim != 1 or b.shape[0] < 10:
        raise ValueError(
            f"WHAM 'betas' has unexpected shape {b.shape}; expected (10,) or (N, 10)"
        )
    return b[:10].tolist()


def _resolve_global_orient(person: dict, pose_aa: np.ndarray) -> np.ndarray | None:
    """Return per-frame global orient as (N, 3) axis-angle, or None if
    WHAM encoded it into pose[:, 0] already."""
    if "global_orient" in person:
        return _to_axis_angle_global(person["global_orient"])
    return None


# ---------------------------------------------------------------------------
# Gravity leveling
# ---------------------------------------------------------------------------

SMPL_CANONICAL_UP = np.array([0.0, 1.0, 0.0])

# Leveling corrections above this are almost certainly a take that does
# not average to an upright torso (lying in bed, ground fight). The
# rotation is still applied, but the warning tells the user to re-run
# with --no-level if the result looks wrong.
LEVEL_WARN_DEG = 60.0


def _axis_angle_from_two_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix taking unit vector v_from onto v_to."""
    c = float(np.clip(np.dot(v_from, v_to), -1.0, 1.0))
    axis = np.cross(v_from, v_to)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        if c > 0.0:
            return np.eye(3)
        # Antiparallel: rotate 180 deg about any axis perpendicular to
        # v_from; +X works for any up-ish vector.
        axis = np.cross(v_from, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-9:
            axis = np.cross(v_from, np.array([0.0, 0.0, 1.0]))
        axis = axis / np.linalg.norm(axis)
        theta = math.pi
    else:
        axis = axis / norm
        theta = math.acos(c)
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def _aa_to_matrix_batch(aa: np.ndarray) -> np.ndarray:
    """(N, 3) axis-angle -> (N, 3, 3) rotation matrices (Rodrigues)."""
    aa = np.asarray(aa, dtype=np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)  # (N, 1)
    out = np.broadcast_to(np.eye(3), (aa.shape[0], 3, 3)).copy()
    nz = theta[:, 0] > 1e-12
    if not np.any(nz):
        return out
    k = np.zeros_like(aa)
    k[nz] = aa[nz] / theta[nz]
    K = np.zeros((aa.shape[0], 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s = np.sin(theta)[..., None]
    c1 = (1.0 - np.cos(theta))[..., None]
    out[nz] = (np.eye(3) + s[nz] * K[nz] + c1[nz] * (K[nz] @ K[nz]))
    return out


def _level_to_y_up(root_aa: np.ndarray, trans: np.ndarray) -> tuple:
    """Rotate the whole take so the mean body-up axis lands on +Y.

    Body-up per frame is the root rotation applied to SMPL canonical up
    (+Y through the spine in the SMPL rest pose). Averaging over the
    take makes the estimate robust to bends, turns, and noise: any
    normal standing/sitting/walking performance averages to vertical.
    This replaces a *blind* constant flip with a data-driven correction
    that also absorbs camera tilt when only camera-frame keys exist.

    Returns (root_aa_leveled, trans_leveled, correction_deg).
    """
    R0 = _aa_to_matrix_batch(root_aa)                    # (N, 3, 3)
    ups = R0 @ SMPL_CANONICAL_UP                         # (N, 3)
    mean_up = ups.mean(axis=0)
    n = float(np.linalg.norm(mean_up))
    if n < 1e-9:
        print("[wham_to_pose_json] WARNING: degenerate mean up-axis; "
              "skipping leveling.")
        return root_aa, trans, 0.0
    mean_up /= n
    R_fix = _axis_angle_from_two_vectors(mean_up, SMPL_CANONICAL_UP)
    correction_deg = math.degrees(
        math.acos(float(np.clip(np.dot(mean_up, SMPL_CANONICAL_UP), -1.0, 1.0))))
    root_lev = _rotation_matrix_to_axis_angle(R_fix[None] @ R0)
    trans_lev = trans @ R_fix.T
    return root_lev, trans_lev, correction_deg


def convert(
    wham_data: Any,
    *,
    character: str | None,
    fps_override: float | None,
    source_video: str,
    source_model: str = "wham",
    prefer_world: bool = True,
    level: bool = True,
    recenter: bool = True,
) -> dict:
    """Convert a deserialized WHAM pkl into our pose JSON dict.

    `wham_data` is the result of pickle.load(open(<wham_pkl>, 'rb')).
    Returns a dict that round-trips through json.dump and conforms to
    the schema in MelodicCapMono/SCHEMA.md.

    prefer_world: WHAM's demo saves BOTH camera-frame keys
        ('pose'/'trans') and gravity-aligned world keys
        ('pose_world'/'trans_world', from its trajectory decoder).
        Camera-frame poses inherit the physical camera tilt — a phone
        propped low and aimed up bakes a constant backward lean into
        every frame, which no downstream retargeter can undo. World
        keys are the correct source; camera keys remain the fallback
        when the pkl lacks world keys (e.g. WHAM run local-only).
    level: rotate the take so its mean body-up axis is exactly +Y
        (gravity-aligned). Near a no-op for world keys; for camera-frame
        fallback it replaces the blind 180-degree flip with a
        data-driven upright correction that absorbs camera tilt.
    recenter: subtract the frame-0 horizontal (X/Z) root translation so
        every take starts at the origin.
    """
    person = _select_person(wham_data)

    if "pose" not in person and "pose_world" not in person:
        raise KeyError(
            f"WHAM person record missing 'pose'/'pose_world'. Keys present: "
            f"{list(person.keys())}"
        )

    use_world = prefer_world and "pose_world" in person
    pose_key = "pose_world" if use_world else "pose"
    trans_key = "trans_world" if (use_world and "trans_world" in person) else "trans"
    if use_world and "trans_world" not in person:
        print("[wham_to_pose_json] WARNING: pkl has 'pose_world' but no "
              "'trans_world'; root translation falls back to camera-frame "
              "'trans' (orientation and translation frames will disagree).")
    if not use_world and prefer_world:
        print("[wham_to_pose_json] No 'pose_world' in pkl (WHAM run "
              "local-only?). Using camera-frame keys"
              + ("; auto-leveling will correct the up-axis."
                 if level else " WITHOUT leveling — downstream applies "
                 "the legacy 180-degree flip."))

    pose_aa = _to_axis_angle_per_joint(person[pose_key])  # (N, 24, 3)
    n_frames = pose_aa.shape[0]

    if trans_key in person:
        trans = np.asarray(person[trans_key], dtype=np.float64)
        if trans.shape != (n_frames, 3):
            raise ValueError(
                f"WHAM {trans_key!r} shape {trans.shape} doesn't match pose "
                f"frame count {n_frames}"
            )
    else:
        print("[wham_to_pose_json] WARNING: no 'trans' in WHAM output, "
              "using zeros (root translation will be all zero).")
        trans = np.zeros((n_frames, 3), dtype=np.float64)

    # Camera-frame 'global_orient' must not be mixed into world-frame
    # poses; world keys carry the root rotation in pose_world[:, 0].
    global_aa = None if use_world else _resolve_global_orient(person, pose_aa)
    if global_aa is not None and global_aa.shape != (n_frames, 3):
        raise ValueError(
            f"WHAM 'global_orient' shape {global_aa.shape} doesn't match "
            f"pose frame count {n_frames}"
        )

    # Fold the authoritative root rotation into pose[:, 0] so the JSON
    # has exactly one root channel, then level + recenter in place.
    if global_aa is not None:
        pose_aa[:, 0] = global_aa
    correction_deg = None
    if level:
        root_lev, trans, correction_deg = _level_to_y_up(pose_aa[:, 0], trans)
        pose_aa[:, 0] = root_lev
        if correction_deg > LEVEL_WARN_DEG:
            print(f"[wham_to_pose_json] WARNING: leveling rotated the take "
                  f"by {correction_deg:.1f} deg. Expected for camera-frame "
                  f"input (~180 deg); if this take is intentionally "
                  f"non-upright (lying down), re-run with --no-level.")
        else:
            print(f"[wham_to_pose_json] Leveled up-axis: "
                  f"{correction_deg:.1f} deg correction "
                  f"(keys={pose_key}/{trans_key}).")
    recentered = False
    if recenter and n_frames > 0:
        offset = trans[0].copy()
        offset[1] = 0.0  # keep absolute height; floor calib is data-relative
        trans = trans - offset
        recentered = True

    betas = _resolve_betas(person, n_frames)

    if fps_override is not None:
        fps = float(fps_override)
    elif "fps" in person:
        fps = float(person["fps"])
    elif isinstance(wham_data, dict) and "fps" in wham_data:
        fps = float(wham_data["fps"])
    else:
        print("[wham_to_pose_json] WARNING: no fps in WHAM output, "
              "defaulting to 30.0.")
        fps = 30.0

    if "frame_ids" in person:
        frame_ids = np.asarray(person["frame_ids"], dtype=np.int64).tolist()
        if len(frame_ids) != n_frames:
            raise ValueError(
                f"WHAM 'frame_ids' length {len(frame_ids)} doesn't match pose "
                f"frame count {n_frames}"
            )
    else:
        frame_ids = list(range(n_frames))

    frames_out = []
    for i in range(n_frames):
        frames_out.append({
            "frame": int(frame_ids[i]),
            "timestamp": float(frame_ids[i]) / fps,
            "smpl_pose": pose_aa[i].flatten().tolist(),
            "smpl_trans": trans[i].tolist(),
        })

    out = {
        "format": "melodiccap_mono_v1",
        "source_model": source_model,
        "source_video": source_video,
        "character": character,
        "fps": fps,
        "smpl_betas": betas,
        "frames": frames_out,
        "frame_provenance": {
            "source_keys": f"{pose_key}/{trans_key}",
            "leveled": bool(level),
            "level_correction_deg": (
                round(correction_deg, 2) if correction_deg is not None else None
            ),
            "recentered_xz": recentered,
        },
    }
    # Gravity-aligned output (world keys are already y-up; leveling
    # forces it for camera-frame fallback). Downstream consumers
    # (pose_json_to_bvh, footlock) skip the legacy 180-degree flip when
    # this tag is present — see _smpl_fk.resolve_global_rot_deg.
    if level or use_world:
        out["coordinate_frame"] = "y_up_world"
    return out
