# Character config schema (`characters/<name>.json`)

Drives the automated pipeline (`MelodicCapMono/orchestrate/process_take.py`
→ `blender_addon/headless_retarget.py`). Adding a character is **config
only** — copy `jax.json`, change the few machine/rig fields, do one
ground-truth visual check. JSON has no comments; `_comment_*` keys are the
convention and are ignored by loaders.

## Fields used by the automated path

| Key | Type | Purpose |
|---|---|---|
| `name` | str | Display name. |
| `armature` | str | Blender **object** name of the target Rigify rig (must exist in `base_blend`). |
| `height_m` | float | Reference height (informational). |
| `base_blend` | str / null | Master `.blend` containing `armature`. **Machine-specific — keep `null` in the committed file**; supply via `--base-blend`, `$MELODICCAP_BLENDER`-adjacent override, or an untracked `<name>.local.json`. The pipeline opens it read-only and saves a **new** per-take `.blend`; the master is never mutated. |
| `blender_exe` | str / null | Blender binary (Steam build is not on PATH). Same machine-specific rule as `base_blend`. CLI `--blender-exe` or `$MELODICCAP_BLENDER` override it. |
| `bvh_export.root_motion` | `"zero"` \| `"trans"` | Passed to `pose_json_to_bvh`. `zero` pins the root (monocular trans drifts — keyframe the path by hand). |
| `bvh_export.global_rot_euler_deg` | `[x,y,z]` | Root orientation fix for WHAM's camera frame (default `[180,0,0]` = upright after Blender's BVH import). |
| `ik_fk_one` | `[bone,...]` | Rigify limb-settings bones whose `IK_FK` custom property is forced to `1.0` (full FK) so the `*_fk` retarget is followed. Default: the four `*_parent` bones. |
| `bone_map` | `{src: {bone, type?}}` | SMPL/BVH source joint → Rigify control bone. `type` defaults to `COPY_ROTATION`; use `COPY_LOCATION` for IK end-effectors (`*_foot→foot_ik`, `*_hand→hand_ik`). WORLD/WORLD copy constraints + `nla.bake(visual_keying=True)` resolve the rest-frame difference. |
| `calibration` | `{target_bone: {invert:[bx,by,bz]}}` | Optional escape hatch, used **only** if a real take shows a bone rotated wrong vs the ground-truth Rokoko bake. Default `{}`. |

## Local override (machine paths, untracked)

`characters/<name>.local.json` (gitignored) is deep-merged over
`<name>.json`. Put your personal `base_blend` / `blender_exe` here so the
public repo stays clean of absolute paths:

```json
{ "base_blend": "C:/Users/you/.../Characters.blend",
  "blender_exe": "C:/.../steamapps/common/Blender/blender.exe" }
```

## Legacy keys (deprecated)

`smpl_to_rig`, `root_translation`, `axis_conversion` are consumed only by
the superseded Phase C direct-import addon (`blender_addon/melodiccap_smpl_addon.py`,
fixtures/smoke-test only). Kept for backward compat; the automated path
ignores them.
