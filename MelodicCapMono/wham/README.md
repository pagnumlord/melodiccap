# Path A — WHAM (monocular SMPL inference)

Single-camera video in, `melodiccap_mono_v1` pose JSON out. The JSON
then drives JaxRigify (or any other character config) via the Blender
addon in `../blender_addon/`.

This directory is a **wrapper**, not a fork of WHAM. WHAM stays in its
own conda env on your machine. We invoke its CLI via subprocess and
reformat its pickled output.

## What WHAM is

[WHAM](https://github.com/yohanshin/WHAM) (Shin et al., CVPR 2024) is a
monocular human-mesh recovery model with motion-prior smoothing. It
takes a single video stream and outputs:

- Per-frame SMPL pose parameters (24 joints × 3 axis-angle)
- Per-frame root translation (xyz, meters)
- Constant body shape (10 SMPL betas)

The motion prior is what we care about. Unlike the v5.x custom solver,
WHAM's output is anatomically valid by construction — no arm-through-
chest, no neck stretch, no shoulder collapse, no foot rotation lock.
The depth axis is inferred from learned priors rather than triangulated,
which is the structural fix for the entire class of v5.x bugs.

The tradeoff: monocular root translation drifts on long takes. Use
`"root_translation": "ignore"` in your character config and hand-
keyframe the walk path. SMPL-driven local body motion (arms, torso,
legs, head) stays accurate.

## Install (one time, ~1 hour)

```bash
# 1. Clone WHAM somewhere outside this repo.
git clone https://github.com/yohanshin/WHAM.git
cd WHAM

# 2. Build its conda env. Heavy: ~3 GB once checkpoints land.
conda env create -f environment.yml
conda activate wham

# 3. Download model checkpoints (~3 GB).
bash fetch_demo_data.sh

# 4. Confirm WHAM works on its own demo video.
python demo.py --video examples/demo_video.mp4 --output_pth out --save_pkl
ls out  # should contain a .pkl

# 5. Point our wrapper at this WHAM clone.
#    Windows PowerShell:
$env:WHAM_DIR = "C:\path\to\WHAM"
#    macOS / Linux:
export WHAM_DIR=/abs/path/to/WHAM
```

WHAM runs on a GPU (recommended) or CPU (slow — minutes per second of
video). Their docs cover system requirements; if you can run their
demo video, you can run our wrapper.

## Run (every take)

You must be inside the WHAM conda env so subprocess inherits its torch
and OpenCV. Then from the **MelodicCap repo root**:

```bash
conda activate wham

# Windows PowerShell:
python -m MelodicCapMono.wham.video2pose `
    C:\path\to\guitar.mp4 `
    C:\path\to\guitar.pose.json `
    --character jax

# macOS / Linux:
python -m MelodicCapMono.wham.video2pose \
    guitar.mp4 guitar.pose.json --character jax
```

Output: `guitar.pose.json` (schema in `../SCHEMA.md`).

> **Which retarget path:** the Phase C direct addon import
> (`../blender_addon/`) applies SMPL joint rotations straight onto
> Rigify FK bones. That is too naive — every Rigify control bone has
> its own rest orientation, so a real take folds (head between the
> knees) and **no `axis_conversion` value fixes it**. The proven path
> is **BVH → a skeleton-to-skeleton retargeter (Rokoko, free)**, which
> owns the rest-pose math. See "Retarget to Rigify" below. The addon
> is kept only for the synthetic fixtures / schema smoke test.

## Automated path (one command per take)

Once a character is calibrated, the whole pipeline is one unattended
command (it reuses everything below — nothing reinvented):

```
python -m MelodicCapMono.orchestrate.process_take guitar.mp4 \
    --character jax --wham-python /path/to/envs/wham/bin/python \
    --base-blend "C:/.../Characters.blend" \
    --blender-exe "C:/.../Blender/blender.exe" --out guitar.blend
```

video → WHAM → BVH → **headless Rokoko retarget** on a **copy** of the
master `.blend` (the master is never touched) → `guitar.blend` (rig +
baked action) + `guitar.report.json`. Put machine paths in an untracked
`characters/jax.local.json` (deep-merged) to drop the
`--base-blend/--blender-exe` flags. Config contract: `characters/_schema.md`.
The headless script calls `bpy.ops.rsl.retarget_animation()` from the
Rokoko Studio Live plugin (auto-loaded in the user's Blender), driven
by the proven bone list in `characters/<name>.json`'s `bone_map`. Hand-
rolled approaches (WORLD/WORLD copy constraints, then a rest-relative
matrix loop) each fixed one symptom and surfaced another on real takes;
Rokoko's retarget engine already handles rest-pose alignment, auto-
scaling, and per-bone basis differences — that's what makes the manual
workflow work, and that's what the headless run now invokes directly.

The manual Rokoko recipe below is now the **calibration / ground-truth
reference**: run it once per new character to produce the known-good
bake the automated path's `calibration` block is tuned against.

## Retarget to Rigify (BVH → Rokoko — manual, calibration reference)

1. **Pose JSON → BVH** on the canonical SMPL skeleton (correct by
   construction — SMPL pose params *are* that skeleton's local
   rotations, so there is no rest-frame mismatch):

   ```
   python -m MelodicCapMono.wham.pose_json_to_bvh \
       guitar.pose.json guitar.bvh --root-motion zero
   ```

   `--root-motion zero` pins the root (monocular trans drifts —
   keyframe the path by hand). `--global-rot` defaults to **auto**:
   gravity-aligned (`coordinate_frame: "y_up_world"`) JSONs need no
   correction; legacy camera-frame JSONs get the historical `180 0 0`
   upright flip. If a take still comes in upside down / facing wrong,
   override with `--global-rot X Y Z` (try `180 180 0` or `180 0 180`).

2. **Import** into Blender: File ▸ Import ▸ Motion Capture (.bvh).
   This is the motion *source* — a bare armature, no mesh, expected.

3. **Clean the target**: select JaxRigify, Dope Sheet ▸ Action Editor
   ▸ unlink any stale action.

4. **Set all four limbs to FK — required.** The retarget keys the
   `*_fk` bones; a Rigify limb only follows them when its `IK_FK = 1.0`.
   Pose Mode → Properties ▸ Bone ▸ Custom Properties on
   `upper_arm_parent.L`, `upper_arm_parent.R`, `thigh_parent.L`,
   `thigh_parent.R` → set `IK_FK = 1.0` (a static rig setting, not
   keyframed). If the embedded `rig_ui.py` is missing (version drift),
   set the property directly here — no panel needed.

5. **Rokoko panel** (free plugin) ▸ Retargeting: Source = the BVH
   armature, Target = JaxRigify, Build Bone List, set targets to this
   **proven map**, Auto Scale on, Retarget, then Pose ▸ Bake Action,
   and **save the bone list as a Custom Naming Scheme preset** —
   reusable for every future take and every character:

   | SMPL source | JaxRigify target |
   |---|---|
   | pelvis | torso |
   | left_hip / right_hip | thigh_fk.L / thigh_fk.R |
   | left_knee / right_knee | shin_fk.L / shin_fk.R |
   | left_ankle / right_ankle | foot_fk.L / foot_fk.R |
   | left_foot / right_foot | foot_ik.L / foot_ik.R |
   | left_wrist / right_wrist | hand_fk.L / hand_fk.R |
   | left_hand / right_hand | hand_ik.L / hand_ik.R |
   | spine1 / spine2 / spine3 | spine_fk.001 / spine_fk.002 / chest |
   | neck / head | neck / head |
   | left_shoulder / right_shoulder | upper_arm_fk.L / upper_arm_fk.R |
   | left_elbow / right_elbow | forearm_fk.L / forearm_fk.R |
   | left_collar / right_collar | (leave empty) |

   Mapping `*_ankle/*_wrist` → the **FK** bones *and* the near-static
   SMPL `*_foot/*_hand` tip joints → the **IK** controls is what makes
   hands/feet actually move on a Rigify rig. FK alone leaves them
   frozen, because the rig still reads the (un-keyed) IK targets for
   end-effector placement. Never retarget onto `DEF-` bones.

6. **Polish pass (hand work — not pipeline):** the body + limb motion
   is the deliverable. On top of the baked action you keyframe the
   root's path across the scene (a few keys — the mocap is in-place by
   design) and do a foot-contact / IK-lock pass to kill sliding. This
   is exactly how the v5.19 music video shipped: a mocap base layer
   with hand-keyframed gaps.

## Recording the input video

For best results:

- **Camera**: any 1080p phone, native camera app (NOT DroidCam — that
  streams instead of recording).
- **Orientation**: landscape. Full body in frame from head to feet,
  with margin so head and feet don't clip on extreme moves.
- **Distance**: 2-3 meters from camera. Closer is fine; farther loses
  pixel resolution on hands/face.
- **Lighting**: even, no strong backlight. Phone autoexposure handles
  the rest.
- **Background**: less busy is better but not required. WHAM's tracker
  is robust.
- **Length**: 15-60 seconds per take. Longer is fine but root
  translation drift accumulates.
- **Pre-roll**: stand still in a clean A-pose for 2 seconds at the
  start. Useful as a calibration reference even though WHAM doesn't
  require it the way v5.x did.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `WHAM_DIR is not set` | Env var missing | `set WHAM_DIR=...` (Windows) or `export WHAM_DIR=...` (macOS/Linux) |
| `WHAM exited with code 1` and "CUDA out of memory" in log | GPU too small for video resolution | Re-encode video to 720p, or shorter clip |
| `WHAM exited 0 but produced no .pkl` | No person detected | Check person is in frame the whole take; brighter lighting |
| `WHAM 'pose' has unexpected shape` from converter | WHAM pkl format drifted between versions | Run `python -m MelodicCapMono.wham.video2pose --inspect-pkl <file>.pkl` to see the actual structure, then update `wham_to_pose_json.py` with the new keys / shapes |
| **Character leans backward/forward constantly** (same lean in every retarget engine) | Pose JSON predates world-frame support: it carries WHAM's **camera-frame** pose, so the physical tripod tilt is baked into every frame | Regenerate the pose JSON (re-run `video2pose`, or `--from-pkl` if you kept the pkl with `--keep-intermediate`). The converter now prefers WHAM's gravity-aligned `pose_world`/`trans_world` keys and auto-levels the take's mean up-axis to +Y, writing `coordinate_frame: "y_up_world"` |
| Leveling warning `rotated the take by NN deg` on an intentionally non-upright take (lying down, ground fight) | Auto-level assumes the take's *average* body-up is vertical | Re-run `video2pose` with `--no-level` and set the orientation manually via `--global-rot` at the BVH step |
| BVH skeleton imports upside down / facing backward | Legacy camera-frame JSON ≠ Blender up-axis | Default is now `--global-rot` **auto**: `y_up_world` JSONs get no correction, legacy JSONs get `180 0 0`. Override explicitly (`180 180 0`, `180 0 180`) only if a take still faces wrong |
| Retargeted hands/feet frozen while the body moves | That limb is still IK, or only the FK bones were mapped | Set `IK_FK = 1.0` on all four `*_parent` bones; also map SMPL `*_foot→foot_ik` and `*_hand→hand_ik` alongside `*_ankle→foot_fk` / `*_wrist→hand_fk` |
| Rig drifts / slides over a long take | Monocular root drift + no foot ground-lock (both expected) | Export with `--root-motion zero`; keyframe the path and a foot-lock pass by hand on the baked action |

## Inspecting a pkl without running WHAM

Useful when WHAM updates change the dict structure:

```bash
python -m MelodicCapMono.wham.video2pose --inspect-pkl path/to/some.pkl
```

Prints the dict structure, ndarray shapes, and dtypes. Compare against
the layout assumed in `wham_to_pose_json.py`'s `_select_person` and
`_to_axis_angle_per_joint` — if a key has been renamed or reshaped,
update those functions, no re-architecture needed.

## Why subprocess instead of importing WHAM

WHAM pins specific torch + CUDA versions. The MelodicCap dev env runs
on a different stack (numpy + opencv-python only — see
`MelodicCapMono/requirements.txt`). Two stacks in one env is a
nightmare; subprocess + a separate conda env is the standard
workaround. This wrapper imports nothing from WHAM directly — only
spawns its CLI and reads the resulting pkl.

If a future monocular SMPL model (GVHMR, 4DHumans, NLF) replaces WHAM,
the swap touches only `video2pose.py`'s `_run_wham` function and
possibly `wham_to_pose_json.py`. The downstream pose JSON schema and
Blender addon are unchanged.
