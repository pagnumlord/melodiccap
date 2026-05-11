# MelodicCap

Markerless motion capture for solo filmmakers. Originally built for a music video
that shipped May 2026; now being re-scoped for a short film with multiple characters.

This repo is in a **transition state**. The v5.19 stereo + custom-solver pipeline
is functional but fragile. The next-generation pipeline (monocular SMPL via WHAM /
GVHMR with a thin retargeting layer) is being designed. Outside review and PRs
welcome — see [What's needed](#whats-needed).

---

## What it does today (v5.19)

```
   2 cameras (phone + webcam)
              ↓
   RTMPose (CUDA) — 2D body keypoints
              ↓
   Stereo triangulation — 3D keypoints
              ↓
   skeleton_solver.py — bone-length + ROM constraints
              ↓
   melodiccap_rtm_addon.py — retarget to JaxRigify in Blender
              ↓
   Hand-keyframe the parts the solver gets wrong
```

This produced a usable take for the music video, but every fix to one symptom
created another (19 versions documented in `CLAUDE.md`). The root cause:
2-camera 90° stereo baselines collapse the depth axis, and a custom retargeter
has no built-in priors to know what a human body can or can't do.

## What it should do (next)

```
   1 phone camera
              ↓
   WHAM or GVHMR — SMPL parameters per frame
              ↓
   Character config JSON (per-character proportions + bone aliases)
              ↓
   SMPL → Rigify retargeter (one mapping covers all SMPL-skeleton chars)
              ↓
   Polish in Blender
```

Same Blender + hand-keyframing workflow at the end. Massively simpler capture
side. No clipping/depth-collapse/neck-stretching class of bugs because the
SMPL output is already an anatomically valid human pose.

---

## Repo layout

| Path | What it is | Status |
|---|---|---|
| `MelodicCapRTM/melodic_capture.py` | Real-time 2-camera capture loop | v5.x — shipped video, frozen |
| `MelodicCapRTM/stereo_calibration.py` | ChArUco stereo calibration | v5.x — frozen |
| `MelodicCapRTM/offline_processor.py` | Re-triangulate raw 2D → 3D from disk | v5.x — frozen |
| `MelodicCapRTM/skeleton_solver.py` | Bone-length + ROM solver | v5.16 — frozen |
| `MelodicCapRTM/blender_addon/melodiccap_rtm_addon.py` | JaxRigify retargeter | v5.19 — frozen |
| `CLAUDE.md` | Full project documentation, including v3.x → v5.19 patch history | maintained |
| `melodic_justice_mocap/`, `MelodicCapFresh/`, `MelodicCapAntiGrav/` | Earlier experiments | archive |

The `melodiccap_retargeter_v3.py` at the root is an old artifact and not used.

---

## Running the current (v5.19) pipeline

You probably shouldn't, unless you want to reproduce the music video. The
short-film pipeline isn't built yet. Steps that work today:

```bash
# Setup
pip install -r requirements.txt

# Calibrate stereo cameras (every session — they move between sessions)
python MelodicCapRTM/stereo_calibration.py

# Capture a take
python MelodicCapRTM/melodic_capture.py

# Re-process raw 2D to 3D + solver
python MelodicCapRTM/offline_processor.py path/to/take_*_raw.json

# Import the resulting *.json into Blender via the addon
# (install MelodicCapRTM/blender_addon/melodiccap_rtm_addon.py in Blender 4.4)
```

Read `CLAUDE.md` end-to-end before doing anything serious with the v5.19 path.
It documents 19 versions of accumulated patches and what each one is working
around.

## Hardware (legacy)

- Samsung S25 over DroidCam (camera A) + Logitech C615 (camera B)
- ~1.5m baseline, 30-45° angle spread, ~2m from performer
- NVIDIA GPU (RTX 4070-class or better) for real-time RTMPose

The short-film pipeline target is: one phone, no GPU at capture time
(inference happens offline on a desktop GPU).

---

## What's needed

If you're "new eyes" looking at this and want to help, these are the high-value
directions:

1. **WHAM or GVHMR integration**. Wrap the model behind a script that takes
   `input.mp4` and writes `pose.json` (frame index → SMPL params + global
   trajectory). PR a `MelodicCapMono/` directory parallel to `MelodicCapRTM/`.
2. **SMPL → JaxRigify retargeter**. New Blender addon that reads the SMPL
   pose JSON and drives the Rigify control bones. Lives alongside the v5.19
   addon, not on top of it. Same hand-keyframing UX after import.
3. **Character config schema**. JSON-per-character with proportions + bone
   aliases. Used by the retargeter to support Jax, Kai, Kiko, Dr White, Hiro,
   THE SHADOW without code changes.
4. **Camera-side root translation fix**. WHAM/GVHMR get global trajectory
   approximately right but drift over long takes. Document a reliable workflow
   (e.g., keyframe the character's root path manually, let mocap drive
   hips-up only). Possibly use a single AprilTag in frame as a world reference.

Smaller helpful items:

- A `CONTRIBUTING.md` covering branch conventions and PR style.
- A `scripts/test-current.sh` that runs the legacy v5.19 pipeline end-to-end
  on a fixture take.
- Per-character proportion configs (JSON files) for the five non-Jax
  characters.

---

## Music video

The v5.19 pipeline was used as the base for the "Melodic Justice" music video
released May 2026. The final shipped animation is the v5.19 retarget plus hand
keyframing for the rough sections (forward walking direction inversion, the
72° torso turn during the guitar grab, arm clipping during the forward bend).
The hand-keyframe load was high enough that this pipeline is not a viable
production tool for a full short film without the architectural pivot
described above.

## License

For "Melodic Justice" production. Free to use and modify for your own project.

---

See `CLAUDE.md` for the full version history, known issues by category, and
detailed troubleshooting tables.
