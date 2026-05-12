# Pose JSON schema — `melodiccap_mono_v1`

This is the contract between the input wrappers (WHAM for Path A,
EasyMocap for Path B) and the Blender SMPL retargeter. Anything that
produces a JSON matching this schema can drive the retargeter; anything
that consumes this schema works with both paths.

## Top-level object

```json
{
  "format":         "melodiccap_mono_v1",
  "source_model":   "wham" | "easymocap" | "fixture",
  "source_video":   "<filename or identifier>",
  "character":      "<character name, optional>",
  "fps":            <float>,
  "smpl_betas":     [<10 floats>],
  "frames":         [<frame objects, see below>]
}
```

### Field details

| Field | Required | Type | Notes |
|---|---|---|---|
| `format` | yes | string | Must be `"melodiccap_mono_v1"`. Loader rejects anything else. |
| `source_model` | yes | string | Identifies the upstream model. `"wham"`, `"easymocap"`, `"fixture"` for our synthetic test data. |
| `source_video` | yes | string | Filename or identifier of the input. Diagnostic only. |
| `character` | no | string | If set, Blender addon picks `characters/<character>.json` automatically. |
| `fps` | yes | float | Source video / take FPS. Blender scene FPS is set to round(fps). |
| `smpl_betas` | yes | array[10 floats] | SMPL body shape parameters, constant across the take. |
| `frames` | yes | array[frame] | One frame object per source frame. See below. |

## Frame object

```json
{
  "frame":               <int>,
  "timestamp":           <float, seconds>,
  "smpl_pose":           [<72 floats>],
  "smpl_trans":          [<3 floats>],
  "smpl_global_orient":  [<3 floats>]    // optional
}
```

| Field | Required | Type | Notes |
|---|---|---|---|
| `frame` | yes | int | Zero-based frame index. Used as Blender keyframe number. |
| `timestamp` | yes | float | Seconds from take start. Diagnostic. |
| `smpl_pose` | yes | array[72 floats] | 24 SMPL joints × 3-component axis-angle, **flat**. Index `i*3:(i+1)*3` is joint `i`'s rotation in radians, axis-angle convention (axis × magnitude = angle). |
| `smpl_trans` | yes | array[3 floats] | Root translation in meters, world space. Applied per character config's `root_translation` policy. |
| `smpl_global_orient` | no | array[3 floats] | Root rotation axis-angle. If absent, the rotation is encoded into `smpl_pose[0:3]` (pelvis). |

## SMPL joint order (24 joints)

| Index | Joint | Maps to JaxRigify (default) |
|---|---|---|
| 0 | pelvis | `torso` (+ root translation) |
| 1 | left_hip | `thigh_fk.L` |
| 2 | right_hip | `thigh_fk.R` |
| 3 | spine1 | `spine_fk.001` |
| 4 | left_knee | `shin_fk.L` |
| 5 | right_knee | `shin_fk.R` |
| 6 | spine2 | `spine_fk.002` |
| 7 | left_ankle | `foot_fk.L` |
| 8 | right_ankle | `foot_fk.R` |
| 9 | spine3 | `chest` |
| 10 | left_foot | (toe — unmapped by default) |
| 11 | right_foot | (toe — unmapped by default) |
| 12 | neck | `neck` |
| 13 | left_collar | (unmapped — Rigify chest absorbs) |
| 14 | right_collar | (unmapped) |
| 15 | head | `head` |
| 16 | left_shoulder | `upper_arm_fk.L` |
| 17 | right_shoulder | `upper_arm_fk.R` |
| 18 | left_elbow | `forearm_fk.L` |
| 19 | right_elbow | `forearm_fk.R` |
| 20 | left_wrist | `hand_fk.L` |
| 21 | right_wrist | `hand_fk.R` |
| 22 | left_hand | (fingers — SMPL doesn't model them) |
| 23 | right_hand | (fingers) |

Mapping is per-character in `characters/<name>.json` `smpl_to_rig`.
Unmapped joints are ignored. Custom characters can rebind any subset.

## Versioning

The `format` field is the schema version. If we ever break compatibility,
the version bumps (`melodiccap_mono_v2`) and the addon rejects v1. We
do NOT add new top-level fields silently — additive changes get a
minor-version note in this document.

## Example

A minimal one-frame fixture (all SMPL params zero = T-pose / rest pose):

```json
{
  "format": "melodiccap_mono_v1",
  "source_model": "fixture",
  "source_video": "synthetic_rest",
  "character": "jax",
  "fps": 30.0,
  "smpl_betas": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "frames": [
    {
      "frame": 0,
      "timestamp": 0.0,
      "smpl_pose": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      "smpl_trans": [0.0, 0.0, 0.0]
    }
  ]
}
```
