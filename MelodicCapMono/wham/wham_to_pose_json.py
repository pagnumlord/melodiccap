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
    if isinstance(wham_data, dict):
        if 0 in wham_data and isinstance(wham_data[0], dict) and "pose" in wham_data[0]:
            people = wham_data
        elif "pose" in wham_data:
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
        for k in ("pose", "trans", "frame_ids"):
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


def convert(
    wham_data: Any,
    *,
    character: str | None,
    fps_override: float | None,
    source_video: str,
    source_model: str = "wham",
) -> dict:
    """Convert a deserialized WHAM pkl into our pose JSON dict.

    `wham_data` is the result of pickle.load(open(<wham_pkl>, 'rb')).
    Returns a dict that round-trips through json.dump and conforms to
    the schema in MelodicCapMono/SCHEMA.md.
    """
    person = _select_person(wham_data)

    if "pose" not in person:
        raise KeyError(
            f"WHAM person record missing 'pose'. Keys present: "
            f"{list(person.keys())}"
        )
    pose_aa = _to_axis_angle_per_joint(person["pose"])  # (N, 24, 3)
    n_frames = pose_aa.shape[0]

    if "trans" in person:
        trans = np.asarray(person["trans"], dtype=np.float64)
        if trans.shape != (n_frames, 3):
            raise ValueError(
                f"WHAM 'trans' shape {trans.shape} doesn't match pose frame "
                f"count {n_frames}"
            )
    else:
        print("[wham_to_pose_json] WARNING: no 'trans' in WHAM output, "
              "using zeros (root translation will be all zero).")
        trans = np.zeros((n_frames, 3), dtype=np.float64)

    global_aa = _resolve_global_orient(person, pose_aa)
    if global_aa is not None and global_aa.shape != (n_frames, 3):
        raise ValueError(
            f"WHAM 'global_orient' shape {global_aa.shape} doesn't match "
            f"pose frame count {n_frames}"
        )

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
        f = {
            "frame": int(frame_ids[i]),
            "timestamp": float(frame_ids[i]) / fps,
            "smpl_pose": pose_aa[i].flatten().tolist(),
            "smpl_trans": trans[i].tolist(),
        }
        if global_aa is not None:
            f["smpl_global_orient"] = global_aa[i].tolist()
        frames_out.append(f)

    return {
        "format": "melodiccap_mono_v1",
        "source_model": source_model,
        "source_video": source_video,
        "character": character,
        "fps": fps,
        "smpl_betas": betas,
        "frames": frames_out,
    }
