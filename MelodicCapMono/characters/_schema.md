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
| `bvh_export.root_motion` | `"zero"` \| `"trans"` | Passed to `pose_json_to_bvh`. `zero` pins the root (keyframe the path by hand). Use `trans` for travelling takes when foot-contact lock is enabled — plant detection happens in world space, so a pinned root makes the IK lock fight the walk (process_take warns when a `zero` take travels >0.5 m). |
| `bvh_export.global_rot_euler_deg` | `"auto"` \| `[x,y,z]` | Root orientation fix. `"auto"` (default) resolves from the pose JSON's `coordinate_frame` tag: gravity-aligned `y_up_world` JSONs get no correction, legacy camera-frame JSONs get the historical `[180,0,0]` flip. Explicit `[x,y,z]` overrides. |
| `ik_fk_one` | `[bone,...]` | Rigify limb-settings bones whose `IK_FK` custom property is forced to `1.0` (full FK) so the `*_fk` retarget is followed. Default: the four `*_parent` bones. |
| `bone_map` | `{src: {bone, type?}}` | SMPL/BVH source joint → Rigify control bone. Drives the headless Rokoko bone list (`bpy.ops.rsl.retarget_animation()`). Include both rotation pairs (`*_ankle → foot_fk`, etc.) **and** IK end-effector pairs (`*_foot → foot_ik`, `*_hand → hand_ik`) — Rokoko handles auto-scaling and rest-pose alignment, so both work correctly. The `type` field is documentation-only; Rokoko infers behavior from the source/target bone names. Mirror the proven 22-entry map the user validated manually in Rokoko's UI for each new character. |
| `calibration` | `{target_bone: {invert:[bx,by,bz]}}` | Reserved for future use; the current Rokoko engine does not consume it. Default `{}`. |
| `foot_contact` | object | Phase 4 IK-lock post-pass config. When `enabled: true`, `process_take` runs `orchestrate/footlock.py` between BVH and Blender to detect plant frames from SMPL ankle Z + XY velocity (data-relative floor calibration via median of the first `calibration_frames`). The headless retarget then keys `foot_ik_bones.{L,R}` location + rotation toward the planted snapshot (captured from Rokoko's own bake at each interval's frame_in — conservative) with smoothstep blend in / out, and ramps `thigh_parent_bones.{L,R}` `IK_FK` from `1.0` (FK on swing) to `0.0` (IK on plant). Defaults are the v5.x music-video constants (`ankle_z_threshold_m: 0.08`, `ankle_vel_threshold_mps: 0.15`, `pin_blend_frames: 6`, `unpin_blend_frames: 8`, `pin_slide_rate: 0.15`, `hip_drift_unpin_m: 0.12`, `calibration_frames: 20`). Set `enabled: false` (or pass `--no-foot-contact`) to bypass; output is identical to today's pipeline. The rotation chain (`thigh_fk/shin_fk/foot_fk`) Rokoko produced is **never** touched — this pass only writes keys on `foot_ik.*` and `thigh_parent.*` `IK_FK`. |

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
