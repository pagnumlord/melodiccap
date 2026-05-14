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

Output: `guitar.pose.json` — same schema the Blender addon already
imports. Open Blender → File → Import → SMPL Pose JSON → pick the
JSON. Character config auto-resolves from the `--character jax` flag.

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
| Imports into Blender but the rig faces sideways / lies down | SMPL → Rigify per-bone axis mismatch on a specific joint | Edit `characters/jax.json` `rot_offset_euler_deg` for that joint. See the addon's first-import calibration notes |
| Rig drifts across the floor over a long take | Monocular root translation drift (expected) | Set `"root_translation": "ignore"` in `characters/jax.json` and keyframe the walk path by hand |

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
