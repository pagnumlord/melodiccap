# MelodicCap - Markerless Motion Capture for Solo Filmmakers

## Project Goal
A motion capture pipeline that a single person can run **in a small room** to animate
characters for a **short film**, not a one-off music video. Originally built around
2-camera RTMPose + stereo triangulation + a custom Blender retargeter; that path
shipped the music video (v5.19, May 2026) but proved too fragile for repeated takes
across multiple characters. Current direction: pivot toward learned-prior monocular
mocap (SMPL-output models like WHAM/GVHMR) and a thinner retargeting layer that
generalizes across characters.

## Status — May 2026

- **Music video: SHIPPED** using v5.19 + manual hand-keyframing of the rough
  passages. The custom pipeline was usable as a base layer; hand-keyframing
  filled the gaps (forward walking direction, full-take torso turn, arm clipping
  during the guitar grab).
- **Short film: pivoting** the pipeline. The v3-v5 history below stays as reference
  but the pipeline going forward is being re-scoped, not iterated on.
- **Repo**: PUBLIC at https://github.com/pagnumlord/melodiccap. PR #1, #2, #4
  merged (docs refresh, repo onboarding, Phase C SMPL infra). Branch
  `claude/fix-mocap-retargeting-NGruo` open: WHAM Path A is now **proven
  end-to-end on a real take** (video → WHAM → melodiccap_mono_v1 JSON →
  SMPL BVH → Rigify retarget → bake). Landed this branch: joblib pkl load,
  `--from-pkl`, Windows POSIX path fix, the `pose_json_to_bvh` exporter and
  up-axis root fix, the documented Rokoko recipe, and **Phase 1 automation**
  (`MelodicCapMono/orchestrate/process_take.py` + `blender_addon/headless_retarget.py`
  — one unattended command per take, config-driven per `characters/_schema.md`).
  The Phase C direct-import addon is superseded (fixtures/smoke only).
- **Path A (monocular WHAM) is the chosen production pipeline.** Path B
  (stereo EasyMocap) is deferred to the rare climax shot needing
  root-translation accuracy. The legacy v5.x stereo solver
  (`MelodicCapRTM/`) is retired, not iterated.

## Short film: Melodic Justice

Cyberpunk setting (Metaneapolis, 2076). Music is the magic system. Stress
Relay handheld devices convert mental stress into power; the antagonist
group (Black Sun, led by THE SHADOW) corrupts the same tech to drain
people into "umbral" shells. The protagonist Jax fights back by joining a
band (Melodic Justice) and ultimately wielding a guitar that transforms
into a laser weapon synced to the beat. Tone: gritty cyberpunk + hopeful
musician core. The script (read into context this session) is the source
of truth for character / motion priorities below.

### Characters (mocap priority order)

| Rank | Character | Role | Mocap-relevant motion |
|---|---|---|---|
| 1 | **Jax** | Lead, every scene | Sitting, standing, walking, running, playing guitar, lying in bed, climbing, dodging, leaping, shooting "to the beat" |
| 2 | **Kai** | Drummer (band), comic relief | Drumming, choreographed beats, tackling, fighting with drumsticks |
| 3 | **Kiko** | Keytarist (band), leader | Playing keytar, swinging it as a weapon, expressive talking |
| 4 | **Hiro** | Mentor, music shop owner | Standing/talking, late-game weapon-staff combat (in a sling after) |
| 5 | **Dr. White** | Cybernetic CEO, ally/twist | Walking with red-orb cane, energy attacks, falling, formal posture |
| 6 | **THE SHADOW** | Antagonist | Massive imposing stance, energy attacks, throwing debris — mostly VFX-heavy (dark mask), may not need full SMPL retarget |
| 7 | **Young Jax** | Flashback only | Child proportions, separate rig — playing piano, struggling |
| 8 | **Alex / Umbrals** | Faceless antagonist extras | Choreographed group attacks — likely one shared "umbral" config |

Phase C–F target Jax only. Other characters become Phase G config
additions (~30 min per character, no code changes — just per-bone offsets
in JSON).

### Motion budget (rough)

- **~80% performance** — dialogue, instrument playing, walking, gestures,
  reactions. **Path A (monocular WHAM)** handles all of this; fast
  iteration, no calibration per session.
- **~15% light action** — running, dodging the necklace, sit-up snap from
  nightmare. **Path A** still fine.
- **~5% heavy action with prop contact** — concert fight (Jax climbing
  lighting rig, leaping with cane, shooting "to the beat" of Kai's drum
  loop), Hiro vs Dr. White staff combat, THE SHADOW energy melee. **Path
  B (stereo EasyMocap)** preferred for the climax sequence, with
  hand-keyframing on top for prop contact.

### Out-of-scope for the body-mocap pipeline (handled separately)

- **Facial mocap** — separate pipeline. Many close-ups in the script
  (Hiro's wisdom delivery, Jax's "Listen to the music" beats, THE SHADOW's
  mask cracks). Candidate tools: Rhubarb Lip Sync for dialogue, a separate
  face-tracking model if expressive needed.
- **Fingers on instruments** — SMPL doesn't include fingers. Hand-keyframe
  finger positions on guitar/keytar/drums. Maybe SMPL-X later if it
  becomes a major need.
- **Cybernetic effects** (Jax's pink mods, Dr. White's eye, Stress Relay
  glows, dark tendrils, guitar transformation, hologram comms) — all VFX,
  no mocap involvement.
- **Multi-person scenes** (band practice, fight choreography) — current
  pipeline is single-person. Capture each performer separately, composite
  in Blender's NLA editor.

## Pipeline Direction (forward-looking)

Two tracks under evaluation. Whichever wins is the one the short film is built on.

### Track 1 — Monocular SMPL mocap (preferred default)

One phone, one shot, no calibration:

1. **Capture**: 1080p phone video, decent lighting, full body in frame.
2. **Pose**: WHAM (2024) or GVHMR (2024) → per-frame SMPL pose + global
   trajectory. Both run on a single GPU, are open-source, BSD/MIT licensed,
   and embed motion priors so the output is anatomically valid by construction.
3. **Retarget**: SMPL → JaxRigify (and other characters) via a small mapping
   layer. Existing options: SMPL Blender addon, Rokoko Studio Live (free,
   retargets to Rigify), Auto-Rig Pro (paid, robust). One-time write of a
   SMPL→Rigify mapping covers all characters that share the SMPL skeleton.
4. **Polish**: same Blender + hand-keyframing workflow that worked for the
   music video.

Why this is the default: no calibration per session, no stereo geometry
constraints, anatomical priors eliminate the clipping/depth-collapse class of
bugs the v3-v5 stereo pipeline accumulated patches for.

Tradeoff: monocular root translation drifts on long takes — keyframe the
character's overall walk path, let mocap drive everything from the hips up.

### Track 2 — 3-camera stereo with SMPL fitting (specialty shots)

Keep the current calibration + RTMPose stack as the front-end. Replace the
custom solver with EasyMocap (multi-view SMPL fitting). Use only for shots
where root translation accuracy matters (precise reach-and-touch, contact
between characters). Higher setup cost, lower iteration speed, higher
spatial accuracy.

### Decision criterion

Pick Track 1 unless a specific shot's blocking requires Track 2. Don't iterate
the v5.x custom solver any further.

## Character scope (short film)

- Jax (JaxRigify, primary, 1.87m) — currently working
- Kai, Kiko, Dr White, Hiro, THE SHADOW — pending; need per-character
  proportion config + bone-name aliasing for non-Rigify variants

A character config JSON per character (height, arm length, leg length, bone
name overrides) is the cleanest way to support all six without code changes.

## Key Facts (legacy stereo pipeline, kept for v5.x context)
- **User height**: 6'1" (1.856m)
- **Primary rig**: JaxRigify (1.87284m tall)
- **Cameras (legacy)**: Samsung S25 via DroidCam (cam A) + Logitech C615 (cam B)
- **Previous 3rd camera (iPad via Camo)**: disabled — wifi-only, watermark, crashes.
- **Stereo geometry (legacy)**: 30-45° angle spread, ~1.5m baseline, ~2m from performer
- **3-camera multi-pair mode**: offline_processor picks best pair per frame by
  reprojection error. CAM_C_INDEX currently -1.
- **Cameras moved each session** — calibration done at the start of every capture.
- **No hardware sync** — sequential grab()/retrieve() (v4.9) reduces inter-cam
  delta to ~20ms.

## Architecture
- `MelodicCapRTM/` — Python capture pipeline
  - `melodic_capture.py` — Main capture loop (dual camera, parallel inference)
  - `stereo_calibration.py` — Stereo camera calibration (checkerboard + bone constraints)
  - `pose_detector.py` — RTMPose wrapper (single-person detection)
  - `kalman.py` — Kalman filter for 3D keypoint smoothing
  - `recorder.py` — JSON take recorder
  - `offline_processor.py` — Offline triangulation (multi-pair support)
  - `skeleton_solver.py` — Skeleton solver v2: direction-preserving chain fitting
    with calibrated bone lengths, soft spine/hip constraints, joint angle limits
  - `apply_solver.py` — Standalone retroactive solver for existing takes
    (`python apply_solver.py path/to/take.json` → writes `*_solved.json`)
  - `chain_calibration.py` — Derives AC stereo pair from AB+BC chain
- `MelodicCapRTM/blender_addon/` — Blender addon
  - `melodiccap_rtm_addon.py` — Main addon (v5.12): imports JSON takes, retargets to JaxRigify
  - `trace_forearm.py` — Forensic diagnostic for forearm L/R length per
    pipeline stage; copies addon's smooth_frames + butterworth_filter so
    it runs without Blender
- `MelodicCapMono/` — Forward-looking SMPL pipeline (Phases C, D landed)
  - `blender_addon/melodiccap_smpl_addon.py` — Phase C addon. Imports a
    `melodiccap_mono_v1` pose JSON, applies SMPL axis-angle rotations
    per joint to a Rigify rig per `characters/<name>.json` mapping.
    No anatomy reconstruction (vs. v5.12) because SMPL output is already
    valid by construction. ~380 lines vs. v5.12's ~3000.
  - `characters/jax.json` — JaxRigify mapping. 18 SMPL joints to Rigify
    bones, per-bone `rot_offset_euler_deg` for axis calibration. Real
    take calibration TBD (Phase F).
  - `fixtures/{rest,wave}.pose.json` — Synthetic pose JSONs that drive
    the addon without needing WHAM/EasyMocap installed. Verified frame
    0 of rest = JaxRigify A-pose (the user's weight-painted rest).
  - `wham/` — Phase D: Path A wrapper.
    - `video2pose.py` — CLI orchestrator. `python -m
      MelodicCapMono.wham.video2pose <mp4> <pose.json> --character jax`.
      Invokes WHAM's demo.py via subprocess (must be in WHAM conda env),
      reads the resulting pkl, hands it to `wham_to_pose_json.convert`.
      Has `--inspect-pkl` to dump WHAM's pkl structure when its layout
      drifts between releases.
    - `wham_to_pose_json.py` — Pure-numpy converter from WHAM's pkl
      shape to `melodiccap_mono_v1` schema. Accepts pose as
      `(N,24,3,3)` matrices, `(N,24,3)` axis-angle, or `(N,72)` flat;
      betas as `(10,)` or `(N,10)`; with/without `global_orient` and
      `frame_ids`; single-person dict, `{person_id: ...}` dict, or list.
    - `README.md` — install (~1hr WHAM env), run, troubleshooting.
- `scripts/`
  - `test_smpl_addon_smoke.py` — Phase C: validates pose JSON schema.
    Runs in stdlib only.
  - `test_wham_converter_smoke.py` — Phase D: 11 cases, builds
    synthetic WHAM-style pkl dicts in memory and verifies the converter
    handles every shape variant. Requires only numpy.

## Blender Addon - Current State (v5.12)
- Proportional retargeting: measures mocap vs rig proportions from frame 0
- **Hybrid mode (default)**: Arms use FK rotations, legs use IK positioning
- Torso: yaw (rest-subtracted + depth-damped) + pitch (rest-subtracted) (v5.0)
- Spine FK: single bone OR full 4-bone chain (distributes rotation 1/N per bone)
- Neck FK: parent-aware ('auto' mode) — relative to current torso/spine
- Head: yaw-only from ear line vs shoulder line, ±60° cap (v5.2)
- Neck FK: capped at 50° rotation (v5.2)
- Foot IK: positioned with per-chain scaling, speed-based pinning with smooth blend
- Foot pinning: walking-aware with hip-drift slide, smooth pin/unpin over 6-8 frames
- Ankle Z offset: precomputed from first 20 standing frames
- Sit/stand detection: hip Z drop with hysteresis (-0.15m sit, -0.10m stand)
- When sitting: legs switch to FK, foot pinning disabled
- Arm FK confidence: wrist-to-shoulder ratio + direction stability boost (v4.7)
- Arm splay: high fixed safety-net limit 0.80 (v4.8 — replaces broken context clamp)
- Seated hip lateral damping: 0.30x on X displacement to correct camera bias (v5.2)
- Seated torso pitch clamp: -20° to +35° to prevent extreme lean (v5.2)
- Seated leg lateral damping: 0.25x on X component to correct camera bias (v4.8)
- Seated arm depth damping: 0.30x on Y component of upper arm AND forearm FK (v5.0)
  — bypassed when arm is extended forward (ratio>0.70, raw_y>0.20) so guitar/reach poses survive (v5.8)
- Yaw depth damping: 0.35x on Y component before yaw atan2 (v5.0)
- Yaw rest subtraction: frame 0 yaw bias removed (mirrors pitch pattern) (v5.0)
- Arm velocity clamp: rejects >8 m/s hand IK spikes
- 4th-order Butterworth low-pass filter (two cascaded biquads, zero-phase) (v4.7)
- Per-keypoint confidence stored in JSON (min of both camera scores) (v4.7)
- Depsgraph flush: after torso+spine (before neck), after neck/head (before limbs)

## Version History (Retargeter, v3.x → v5.19)

> **Reference only.** This is the patch history of the custom retargeter used
> for the music video. Don't iterate further on this code — the short film
> pipeline pivot is the priority. Kept here so anyone looking at the v5.x
> branch can understand what each constant is for.


### v3.x — IK arms, worked well for arm POSITIONING
- Arms used IK wrist targets — character hands went where mocap wrists were
- Arm raises worked correctly because IK pulls hand to position
- Had chicken-wing problem during sitting (IK solver bends elbow sideways
  when wrist is close to shoulder)
- No torso pitch, no sit detection, no spine FK chain

### v4.0 — Hybrid FK arms + IK legs
- Introduced arm FK to fix chicken-wing: shoulder→elbow→wrist FK rotation
- Arms no longer chicken-wing when sitting, BUT:
- FK inherently worse for arm POSITION accuracy than IK (rotates bones
  vs positioning endpoints)

### v4.1 — Parent-aware FK transform
- Fixed FK rotation math to account for parent chain
- Added depsgraph flush for arm FK

### v4.2 — Torso pitch, sit/stand detection
- Added forward lean detection (hip→shoulder vs vertical)
- Sit/stand hysteresis: legs switch IK↔FK based on hip drop
- **Introduced double-rotation bug**: torso pitch AND spine FK both applied
  the same lean from the same data (spine_fk is child of torso)

### v4.3 — Walking, head bobbing, chicken wing
- Walking foot pinning improvements
- Arm splay bias subtraction (later replaced in v4.4)

### v4.4 — Arm splay clamp, spine tilt, walking pins
- **ARM_SPLAY_MAX = 0.10**: replaced bias subtraction with hard clamp on
  outward X component of upper arm FK direction
- Added spine rest tilt subtraction (caused double-rotation with torso pitch)
- Added PIN_SLIDE_RATE for walking foot IK
- Added arm_fk_conf with fade from 0.7→0.4

### v4.5 — Fix double-rotation, neck FK, seated arm fade
- Removed spine FK tilt subtraction (torso pitch handles it alone)
- Neck FK uses 'auto' parent mode (but depsgraph wasn't flushed — didn't work)
- Tightened arm_fk_conf to 0.7→0.55 range
- Arms blended to identity (rest pose = arms at sides) when conf low

### v4.6 — Depsgraph fix, arm hold-pose, dense diagnostics
- Fixed depsgraph flush ORDER: now flushes AFTER torso+spine, BEFORE neck FK
  (v4.5 had it after neck, so 'auto' parent mode read stale data)
- Arm hold-pose fallback: holds last good FK rotation instead of identity
  when confidence drops below 0.3
- Dense logging: every frame within ±12 of sit transitions
- Per-frame arm_fk_conf + leg FK direction logging

### v4.8 — Fix arm splay limit, seated leg lateral damping
- **ARM_SPLAY_LIMIT = 0.80 fixed**: v4.7's context-aware clamp using elbow height
  above shoulder FAILED — lateral arm raises keep elbows at/below shoulder height,
  so splay_lim stayed at 0.15, still crushing arm raises. Also caused seated
  chicken-wing (elbows pushed inside torso). Replaced with a single high fixed
  limit (0.80) that acts as a safety net for extreme triangulation artifacts only.
  The original chicken-wing was an IK solver artifact (v3.x), not applicable to FK.
  arm_fk_conf + velocity clamp handle noisy data.
- **Seated leg lateral damping (SEATED_LEG_LATERAL_DAMP = 0.25)**: camera placement
  asymmetry causes systematic triangulation bias in shin X component (left=-0.14,
  right=-0.08, both negative = same direction = clearly camera bias not anatomy).
  During sitting, X component of all leg FK directions is damped to 25%, making
  shins hang mostly straight down. Fixes the left leg inward bend during sitting.

### v5.0 — Yaw fix, forearm depth damping
- **Yaw depth damping (YAW_DEPTH_DAMP = 0.35)**: the Y (depth) component of the
  body-right vector was contaminating yaw computation. At 90° stereo angle, depth
  is the worst-resolved axis. Damping Y before `atan2` reduces spurious rotation.
  Always-on (not seated-only). Reduced sitting yaw from -13.5° to ~-5° (raw).
- **Yaw rest subtraction (torso_rest_yaw)**: stores frame 0 yaw and subtracts it
  from all frames, removing constant camera-geometry bias. Mirrors the existing
  `torso_rest_pitch` pattern. Combined with depth damping: sitting yaw -13.5° → -2.7°.
- **Forearm depth damping**: extended SEATED_ARM_DEPTH_DAMP to forearm FK (was
  only upper arm). Forearm Y was -0.96 during sitting (pointing almost straight
  backward from elbow depth overestimation). Damping fixes hands-pointing-down.

### v5.2 — Seated posture fixes, hip lateral stabilization
- **Hip lateral stabilization (SEATED_HIP_LATERAL_DAMP = 0.30)**: 90° stereo setup
  causes 20cm lateral X drift during sit-down (triangulation noise, not real movement).
  X displacement damped to 30% when seated, keeping character roughly vertical from
  front view instead of leaning sideways. Mirrors SEATED_LEG_LATERAL_DAMP pattern.
- **Neck rotation cap (NECK_ROT_MAX = 50°)**: global cap on neck FK rotation.
  Take 2 data showed 75.6° neck rotation (physically impossible — human max ~50°).
  Neck was over-compensating for backward torso lean.
- **Head yaw cap raised (±40.1° → ±60°)**: previous ±0.7 rad cap was too restrictive.
  Performer legitimately turned head past 40° and motion was flattened. Now ±1.05 rad.
- **Seated torso pitch clamp (SEATED_PITCH_MIN=-20°, SEATED_PITCH_MAX=+35°)**: prevents
  extreme backward lean when seated. Take 2 showed -30.2° pitch (extreme recline),
  now clamped to -20°. Forward lean up to 35° still allowed.

### v5.19 — Mirror also swaps LEFT/RIGHT keypoint labels

User tested v5.18's `mirror_x` toggle: "It is mirrored correctly and he
leans camera-right like I did in my take" (✓ orientation fixed) "BUT the
legs were switched, always crossed. Arms as well up until the sit pose."

v5.18 only negated X coordinates. The LEFT_HIP keypoint was still
labeled LEFT and got fed to the rig's LEFT leg bone — only its world
position was on the opposite side. Result: rig's left leg ended up
where the user's right leg actually was. Anatomical L/R labels
disagreed with world positions.

v5.19 fix: when `mirror_x` is ON, also swap the COCO body-17 LEFT/RIGHT
index pairs (eye, ear, shoulder, elbow, wrist, hip, knee, ankle).
Together with the X negation, this is a proper reflection across the
YZ plane — every body-side label follows the corresponding world
position consistently. Per-keypoint confidence values are swapped too.

Verification path: re-import a take that mirrored cross-legged with
v5.18+mirror_x ON. With v5.19+mirror_x ON, expect: orientation matches
the user (lean direction, body turn direction) AND limbs go to the
correct sides (no crossing).

### v5.18 — User-toggleable X mirror for rig orientation mismatch

User observed across multiple takes: "I always look to my left, but Jax's
animation is looking to his right. I lean toward camera-right (my left),
Jax leans toward camera-left and his right side."

The mocap world's +X axis is camera A's right (right-of-screen in playback).
The rig's armature-local +X axis depends on which way the rig was built —
JaxRigify and other Rigify variants differ. When they don't match, the
rig appears mirrored: every left becomes right and vice versa.

Quick fix without restructuring the addon's coordinate handling: a user-
toggleable `mirror_x` import setting that negates the X coordinate of
every 3D keypoint at load time, BEFORE any downstream processing. Default
OFF for backwards compatibility (existing rigs that were correctly
oriented stay correct). User toggles ON when they see the mirror.

Implementation lives in `melodiccap_rtm_addon.py`, applied in `execute()`
right after `frames` is loaded, before format detection. Modifies the
landmarks_3d arrays in-memory so every downstream consumer (proportions,
torso yaw/pitch, FK bone direction, IK target positions, foot pinning)
sees the corrected values consistently.

Out of scope for v5.18: auto-detection of the rig's facing direction
(would require inspecting bone matrices), or full coordinate-frame
calibration (would need rest-pose alignment logic). Manual toggle is the
minimum viable check for the user's tonight session.

### v5.17 — Torso turn under-rotation fix + Blender FPS match

User #1 complaint after every motion-capture iteration: "every single take
I've ever recorded where I turn my body a lot, the torso moves very oddly."
Root cause finally located in v5.17's diagnostic.

**The bug**: `YAW_DEPTH_DAMP = 0.35` from v5.0 was applied unconditionally
to `body_right.y` before `atan2`. For a real 70° body turn at frame 49 of
take 213321, body_right ≈ (-0.342, -0.940). Without damping,
atan2(0.940, -0.342) = 110° → rest-subtracted = -70° (correct). With y *=
0.35: y becomes -0.329, normalized to (-0.720, -0.693), atan2(0.693,
-0.720) = 136° → rest-subtracted = -44°. **The damp halved the measured
turn angle**. The capture log confirmed it: torso_yaw read -36.2° at
exactly the frame where the user's actual shoulder yaw rotated 72°.

The damp was sized for v5.0's noise level. v5.15's anatomical anchoring
(shoulder/hip width clamps) reduced shoulder triangulation noise to ±10%
of calibration, so the static damp is now over-correction.

**Fix** (`melodiccap_rtm_addon.py`, the yaw computation block):
context-aware damping. Compute `yz_ratio = |body_right.y| / |body_right.x|`,
which approximates how far the body has turned from rest. At rest
(yz_ratio = 0): damp = `YAW_DEPTH_DAMP` (0.35, full damping for noise
reduction). At yz_ratio ≥ 0.4 (≈22° turn): damp = 1.0 (no damping, full
turn passes through). Linear interpolation between. The threshold catches
small noise (<10° spurious yaw at sit-down) while letting real turns
through unattenuated.

**Plus**: Blender scene FPS now auto-set at import. The addon imports
keyframes at `frame_num = int(timestamp * fps)` — 8fps mocap ⇒ 8 Blender
frames per second of source. But Blender's default scene FPS is 24, so
playback runs at 24/8 = 3× speed. User noticed: "feels a little quicker
than my take." Now `scene.render.fps` is set to `round(source_fps)` and
`scene.render.fps_base` to `fps_int / source_fps` for fractional precision.

Verification: re-process the four test takes and re-import. Expected
behavior change:
- Turn-only 213321: rig should now sweep through ~72° of yaw (matches the
  user's actual turn) instead of the v5.16 ~36° under-rotation.
- All takes: real-time playback duration in Blender (a 9.2-second mocap
  reads as 9.2 seconds of timeline at fps=8 setting).

Out of scope: the Track B SMPL+motion-prior pivot (still deferred).

### v5.16 — More rules: anatomical ROM + spine rate limit + tighter neck

The user's complaint after v5.15: "neck bends too much in the movement, and
the arm does go inside Jax's body. The actual take. In the last photo
you'll see the take where im sitting, getting into the sitting motion was
very rough, and the elbows are pointed in. We don't need to fix these
small issues, we need to fix the overall capture or code or something to
make it better match our take and be realistic human motion." Per the
outsider perspective: "not enough rules restricting it to realistic
movement." The user picked Track A (tighten the existing pipeline) over
Track B (architectural pivot to SMPL+motion-prior).

Six changes, all serving the same goal: hard rules that prevent
geometrically valid but anatomically sloppy poses from reaching the rig.

**Solver (`skeleton_solver.py`):**

1. **Shoulder + hip ROM** (Step 7b, NEW): humerus-to-spine angle clamped
   to [25°, 178°], thigh-to-spine angle clamped to [15°, 178°]. The 25°
   shoulder floor is the direct fix for "arm inside Jax's body" — it
   prevents the upper arm from being within 25° of the spine direction,
   which on the rig's idealized chest mesh meant the arm bone was passing
   through the ribs. On take 213321 (turn-only) this clamp engaged
   16/74 frames (21.6%), all on the left shoulder (the side the user
   turned toward). On 213428 (full take) it engaged 28/150 frames
   (18.7%). Right shoulder/hip rarely tripped — asymmetry from the
   user's actual pose, not from the constraint.

2. **Spine pitch rate limit** (Step 1b, NEW): the angular change between
   consecutive spine vectors is capped at 8°/frame (≈64°/sec at 8fps).
   When exceeded, the current spine direction is rotated back toward the
   previous via Rodrigues rotation around the cross-product axis. Source
   data on the four test takes had max step 4-6°/frame so the rate
   limit didn't engage in v5.16's verification — it's a preventive
   constraint for the next take with the typical depth-axis spike during
   fast sit-down.

3. **Detector confidence-gated VERY_LOWCONF tier** in
   `_temporal_smooth_directions`: when the AVERAGE per-keypoint
   confidence (parent + child / 2) of a bone's endpoints is below 0.35,
   use α = 0.08 instead of the existing 0.15 LOWCONF tier. Both
   endpoints weak should propagate even less than one weak endpoint.

**Addon (`melodiccap_rtm_addon.py`):**

4. **Neck rotation cap** tightened from `NECK_ROT_MAX_STAND = 50°` to
   **35°** (and SIT cap 25°→20°). Logged data on the four takes had
   neck_rot peaking 20-43° while the user described the magnitude as
   "too much." 35° matches the realistic anatomical range; the reduced
   cap will trim the spurious 5-8° the user was seeing as wobble.

5. **Neck angular velocity cap** tightened from `NECK_MAX_ANGULAR_VEL =
   8°/frame` to **5°/frame** (=40°/sec at 8fps). Smoother frame-to-frame
   change kills the wobble contribution from depth-axis ear-keypoint
   noise.

6. **Sit blend frames** extended from `SIT_BLEND_FRAMES = 8` to **16**.
   At 8fps that's 2 seconds of crossfade. The hip drops -16cm in the
   8-frame v5.3 window, forcing both seated dampers AND the leg IK→FK
   posture switch to ease in over the same frames where the hip is
   falling — visible as the "rough sit motion" the user flagged. 16
   frames separates the leg posture switch from the hip drop in time.

7. **Arm splay clamp** rewritten from broken to angle-correct. Old
   code: clamp `target_dir.x` to `ARM_SPLAY_LIMIT = 0.80`, then
   `.normalized()` — but normalizing a (0.80, y, z) vector with small
   y/z restores X to ~0.93, so the clamp logged "CLAMPED" but the X
   barely changed (verified against take 213341 T-pose log frames 45-72
   where dir.x stayed ≥0.85 after the supposed clamp). New: when |X|
   exceeds the limit, scale Y/Z proportionally to absorb the X
   reduction so the vector stays unit length without inflating X back.
   Pure ±X (perfect T-pose) passes through unchanged because there's
   no Y/Z to absorb.

Verification on the four v5.15 test takes:

| Metric | 213321 | 213341 | 213401 | 213428 |
|---|---|---|---|---|
| Frames solved | 74/74 | 91/91 | 151/151 | 150/150 |
| Total angle clamps | 49 (was 33) | 264 | 111 (was 107) | 100 (was 69) |
| `shoulder_l` clamps | 16 (NEW) | 0 | 0 | 28 (NEW) |
| `hip_l` clamps | 0 | 0 | 4 (NEW) | 3 (NEW) |
| Shoulder width violations | 0/74 | 0/91 | 4/151 | 4/150 |
| Spine pitch range | unchanged | unchanged | unchanged | unchanged |
| Max spine pitch step | 5.2°/f | 3.8°/f | 5.6°/f | 4.1°/f |

Existing v5.15 stability holds; v5.16 adds anatomical ROM corrections
on the frames that needed them. Out of scope (Track B follow-up): SMPL
fitting via EasyMocap to replace the custom solver, evaluation deferred
until v5.16 ships and the user has visual data on the rig.

### v5.15 — Anatomical anchoring (shoulder_width + tightened hip_width)

The visual problems on take 204628 (hands inside body during sit, torso
"flying around" during turns) survived v5.14 because the upstream 3D data
itself was anatomically wrong, not just the addon's gates. A diagnostic on
the v5.13-fresh re-process showed shoulder width collapsing 44% during the
forward bend (calibration mean 0.349m → 0.195m at frame 79) — pure depth-
axis triangulation noise. The cameras are at a 90° baseline; when both
shoulders move toward the cameras during a forward bend, multi-view
correspondence weakens and the apparent inter-shoulder distance collapses.

`skeleton_solver.py` was already calibrating `hip_width` and `spine` as
virtual bones, but Step 3 (line 367-371 pre-fix) kept the raw L/R shoulder
offsets without any width enforcement, and Step 3b's `hip_width` threshold
of 20% allowed the diagnostic range 0.16-0.24m through (cal mean 0.199m).

Three changes to `skeleton_solver.py`:

1. **Add `shoulder_width` to `VIRTUAL_BONE_NAMES`** so it's calibrated
   exactly the way `hip_width` and `spine` already were (median of clean
   standing frames, gated by L/R asymmetry + hip-Z + slouch checks).
2. **Per-frame shoulder width enforcement**: when raw L↔R distance
   deviates >10% from calibration, project both shoulders symmetrically
   around the spine-corrected `shoulder_mid` along the raw L→R axis.
   ±10% allows real shoulder-girdle flex (shrugs, scapula rotation in
   arm raises) but kills camera-driven collapse beyond that.
3. **Tighten `hip_width` threshold from 20% → 10%** to match. Old 20%
   was too loose given the diagnostic data.

Verification on take 204628 (the walk + sit + grab-guitar take):

| Metric | Pre-v5.15 | Post-v5.15 |
|---|---|---|
| Shoulder width range | 0.195 → 0.366m (span 0.171m) | 0.312 → 0.366m (span 0.054m) |
| Worst shoulder violation | -44% from cal | -10% (at clamp) |
| Hip width range | 0.161 → 0.239m | 0.180 → 0.220m |
| Frames solved | 186/186 | 186/186 |
| Angle clamps total | 63 | 64 |

The 26 frames that hit the shoulder-width clamp (14% of the take) are
exactly the ones that would have been -25% to -44% under v5.13 — pulled
back to the anatomical -10% bound. The other 86% pass through unchanged.

Implications: the existing addon retargeter consumes shoulder positions
directly (rig_shoulder_L/R, FK arm parent matrix, torso yaw/pitch). With
shoulders no longer collapsing to half their anatomical width during
bends, the parent transform for arm FK should now be stable across
torso turns — the original "torso flying around" symptom. This needs to
be verified by re-importing the v5.15 take into Blender.

Out of scope: the IK_FK property convention check on JaxRigify (Track B
follow-up — possible cause of "hands at sides during sit" if the v5.15
shoulder fix doesn't surface them). Test plan: a turn-only isolation take
that exercises only torso yaw without depth collapse.

### v5.14 — Stop rejecting natural bent-arm poses as "low confidence"

After v5.13 fixed the upstream wrist-on-shoulder source, the addon's old
arm-confidence gates (sized for the v4.7-era pathology) started rejecting
valid bent-arm geometry as if it were broken triangulation. Symptom: hands
collapse "inside the body" during sitting, holding-guitar, or any reach
where the elbow bends past ~70°. The cascade was: low arm_ratio → conf=0
→ hold last-good FK rotation → ARM_FREEZE_TIMEOUT after 15 frames → slerp
to rest pose.

The geometric ratio (straight-line shoulder-to-wrist distance / fully-
extended arm length) is a poor signal for "data is bad". A natural elbow
bend at 90° gives ratio ≈ 0.71. At 60° (holding something close to body),
ratio ≈ 0.50. The old 0.55 threshold flagged every one of those as an
error.

Five constant changes in `melodiccap_rtm_addon.py`, all serving the same
goal: only flag a frame as "low confidence" when wrist is *actually* near
shoulder (ratio < 0.20), not when the elbow is just bent.

| Constant | Old | New | Where |
|---|---|---|---|
| `ARM_MIN_RATIO` (IK clamp) | 0.55 | 0.15 | line ~1458 |
| `arm_fk_conf` band | (0.55, 0.80) | (0.10, 0.20) | line ~1883 |
| `FA_RATIO_GOOD`, `FA_RATIO_BAD` (forearm rest blend) | 0.70, 0.55 | 0.20, 0.10 | line ~2060 |
| `arm_extended_forward` bypass | `ratio>0.70 AND raw_y>0.20` | `raw_y>0.20` | line ~1947 |
| Warning threshold | `ratio<0.8` | `ratio<0.20` | line ~2336 |

Verification on take 204628 (the walk + sit + grab-guitar take):
- Pre-v5.14: arm_fk_conf dropped to 0.00 from frame ~120 onward; arms
  HELD then ARM_FREEZE_RESET fired every frame for the rest of the take;
  visible result was hands snapping to rest pose during sitting.
- Post-v5.14: arm_fk_conf stays at 1.0 across the whole take; zero
  ARM_FREEZE_RESET events; hands stay where mocap captured them.

Out of scope (deferred to a follow-up audit): torso pitch math, head
pitch (currently yaw-only), the 11 depth-noise dampers, `compute_spine_fk_chain`
(opt-in, previously regressed), and the architectural question of whether
to swap the custom retargeter for a Rokoko-Studio-Live-style bone-mapping
layer with custom-entry support for Rigify's 4-spine chain. That audit is
in `/root/.claude/plans/we-need-the-torso-luminous-adleman.md`, Track B.

### v5.13 — Capture-pipeline root causes: clamp_angle math + framerate-aware velocity gate

Two distinct upstream bugs that produced the visible Blender symptoms
("L wrist stuck on shoulder", "walking contortion", "arm raise not as
high as v3"). Both are in the capture/processing pipeline, not the
addon. The addon has been faithfully rendering broken solver output for
weeks.

**Bug A — `_clamp_angle` math inverted** (`skeleton_solver.py:124-128`).
The Rodrigues rotation used `np.pi - target_rad` where it should have
used `target_rad`. When elbow angle exceeded the 175° max ROM (e.g.
178° on a near-straight arm with triangulation noise), the clamp
produced 5° (fully bent, hand on shoulder) instead of 175° (slightly
bent, hand still extended). This collapsed the wrist onto the shoulder
on any frame where elbow angle nudged past 175°.

Take 204607 (static A-pose, 80 frames): 32 frames had L_WRIST on
L_SHOULDER pre-fix (frames 0-14, 60-79). Post-fix: 0 frames. Verified
end-to-end: triangulation produced wrist at hip Z=0.85m for ALL 80
frames; the clamp was the single point of failure. Same bug also
explained "arm raise not as high as v3" — a near-straight raised arm
would similarly get clamped to "fully bent" and collapse.

R arm rarely tripped because its noise pattern kept elbow angle just
under 175°; L arm tripped because cam B's view of the user's left side
gave elbow angle ~178° from the same noise.

**Bug B — `OUTLIER_MAX_VELOCITY` was in m/frame** (`stereo_calibration.py:21`).
0.3 m/frame at 21fps = 6.3 m/s (the original tuning). At 8fps (the
body_accurate detector), the SAME constant became 2.4 m/s — well
within range of legitimate fast motion (reach for guitar, sit-down,
turn). Triangulated 3D positions exceeding the threshold got replaced
by held `_prev_points`, freezing the keypoint for up to 9 frames (the
v5.10 sticky-release window). The visible result: shoulder/nose freeze
in 3D while hips kept moving → "walking contortion" with a
~30° spurious spine pitch swing per freeze.

Fix: thresholds now in m/s (`OUTLIER_MAX_VELOCITY_MPS = 12.0`,
`OUTLIER_MAX_VELOCITY_FEET_MPS = 6.0`), divided at runtime by an
`_capture_fps` set on the `StereoCalibration` instance by
`offline_processor` from `frame_count / duration`. Default 21.0 keeps
the existing tuning for `melodic_capture` (real-time path).

Verification on take 204628_raw (186 frames at 7.91 fps):
- Pre-fix: 9-frame freezes on L_SHOULDER (70-78), nose (79-87),
  R_SHOULDER (65-73). Spine pitch jumped from -3° to -32° in one
  frame at frame 75. 24 L_ankle fallbacks.
- Post-fix: zero frozen-3-frames-in-a-row keypoints across all 186
  frames. Spine pitch ramps smoothly -3° → -29° over frames 65-79
  (the actual bend-to-grab motion), recovers to -5° by frame 99.

Implications for existing takes: re-run `offline_processor.py` on
`*_raw.json` files. Solver output is regenerated correctly. No
recapture needed.

### v5.12 — Bypass addon smoothing on solver-output JSONs (the actual fix)
v5.11's per-key count fix turned out to be a no-op for solver-output
data because no keys are missing post-solver. `trace_forearm.py`
showed the addon's `smooth_frames` produces identical buggy output
with both buggy and fixed versions. The real bug is that
position-space averaging on solver output **compresses bone lengths**
when frame-to-frame forearm direction varies — the addon averages
positions, but the solver enforces lengths *per frame* with
independently-computed directions, so even small direction noise
causes the averaged forearm to shrink. The trace numbers expose this
exactly: `0.085 ≈ 0.242 / 3` is the smoothed length when 1 of 3
frames in the window has the forearm pointing in a different
direction (cosine-of-half-angle effect).

The fix: detect `processing_settings.skeleton_solver` in the JSON
and skip the addon's `smooth_frames` + `butterworth_filter_landmarks`
when solver-output is detected. The solver already does temporal
direction smoothing in `_temporal_smooth_directions`, so the addon's
extra smoothing is redundant AND actively harmful.

Logged at import as `[INFO] Solver-smoothed take detected; skipping
addon's moving-average and Butterworth filters`. Legacy
MediaPipe-formatted JSONs (no solver_meta) still get the original
smoothing path, where `smooth_frames` is appropriate for raw
single-camera 3D data.

Implications: every take recorded in the past three days is usable
as-is. The JSONs have correct solver output. Just reinstall v5.12
and re-import — no re-recording, no re-processing.

### v5.11 — Forensic audit found smooth_frames divide-by-count bug
The single root cause of three days of "L/R bone divergence" hard
blocks. Symptom: addon reported forearm L/R up to 7.81x bad on 24%
of frames, despite clean 2D detections (verified per-camera) and
clean solver output (forearm_l=0.243m vs forearm_r=0.255m, 5%
asymmetry).

The bug, in `smooth_frames` (`melodiccap_rtm_addon.py:684-718`):
the moving-average divisor `count` was incremented per FRAME (any
frame with non-empty `landmarks_3d`), but each landmark's sum was
over only the frames where THAT specific key was present. When
`triangulate_pose` dropped a keypoint for low confidence (e.g.
wrist briefly occluded during arm motion), the wrist's average
divided by too large a divisor and pulled the wrist toward origin
proportional to how often it was missing. Result: a wrist dropped
on 1 of 3 frames inflated the forearm length by ~33%; the
Butterworth filter compounded this. The bug was triggered by RTM's
intermittent low-confidence wrist drops on arm raises and reaches —
exactly what every test sequence had.

Fix: per-key count tracking. The `count` is replaced by a
`key_counts` dict so each landmark's sum is divided by the number
of frames where THAT key was actually present.

Why this was hard to see: the SOLVER enforces calibrated bone
lengths in its output JSON, and `check_lr_bone_symmetry` is run on
the addon's POST-smoothed coordinates, so the post-solver JSON
looked fine on inspection but the addon's view was corrupted.

Verification: `trace_forearm.py` reports forearm L/R length at
each pipeline stage (post-solver from JSON / post-smooth_frames /
post-Butterworth) for both buggy and fixed `smooth_frames`. On a
broken take, stage 1 ratio stays ≤1.05x while stage 2 spikes
≫2x; with the fix, all stages stay ≤1.10x.

Implications: every take recorded in the past three days is
already usable — the JSON files have correct solver output. No
re-recording, no re-processing. Just reinstall the addon and
re-import. Camera placement, calibration, detector quality, and
recording technique were never the problem.

### v5.8 — Forward-reach bypass, A-pose-hold warning, head-turn diagnostic
- **Seated arm depth-damp bypass when reaching forward** (addon, upper arm
  ~line 1837 and forearm ~line 1937): the v4.9 `SEATED_ARM_DEPTH_DAMP=0.30`
  was tuned for "arms on armrests" where Y>0 is depth noise. But poses like
  "sitting with guitar" put hands forward at center — Y is real reach, not
  noise. Damping it to 30% pulled elbows back into the body and collapsed
  hands into the lap. Fix: detect `arm_extended_forward` when
  `arm_ratio > 0.70 AND raw_y > 0.20` and skip both the upper-arm Y damp
  and the forearm seated-rest blend. Logged as `BYPASS (extended forward)`.
- **A-pose-hold quality warning** (addon, after foot_z_offset block ~line 1285):
  `foot_z_offset[side]` averages the first 20 frames. If the performer
  walked during frames 5-20, swing-phase ankle Z biases the offset and feet
  float through the take. New scan tracks max hip-XY velocity over the
  same 20-frame window; if it exceeds 5cm/frame (~1m/s, clearly walking),
  logs a `[WARNING] A-pose calibration window contained motion`. Soft
  warning, not a hard block — A-poses can have legitimate small wobble.
- **Head-turn velocity diagnostic** (addon, neck angular-velocity clamp
  ~line 1727 + import-complete summary): `NECK_MAX_ANGULAR_VEL=8°/frame`
  catches depth-noise spikes but also smears intentional fast head turns.
  Now counts clamp fires in `mocap_props['_neck_clamp_fires']` and reports
  at end-of-import: `HEAD_TURN_LIMITED: N/total frames (X.X%)`. >5% logs
  a `[WARNING]` recommending a slower head turn or a higher cap.

### v5.7 — Hard A-pose block, arm freeze timeout, confidence-weighted solver
- **HARD BLOCK on bad takes** (addon `validate_frame0_pose` +
  `check_lr_bone_symmetry`): monocular data, A-pose spine tilt >25°,
  hip Z <0.6m, arm/leg asymmetry >10%, foot-Z offset asymmetry >8cm, or
  L/R bone length divergence >15% on >5% of frames all return
  `{'CANCELLED'}` with a clear error. The foot-Z asymmetry gate in
  particular catches the 48cm L/R catastrophe from take_20260420_230051
  that would have pinned L foot 0/293 frames if silently rescued.
- **Arm freeze timeout** (addon): the v4.7 direction-stability boost could
  rescue a zero-base-conf frame (wrist ratio ≤0.55 = arm on armrest) just
  because the direction matched itself, and the hold-pose fallback then
  kept that rotation for 50+ frames. Fixed two ways:
  (1) stability boost now requires base conf ≥ 0.15 before firing,
  (2) `last_good_arm_rot` has a 15-frame held-timeout that slerps back to
  rest over 8 frames, logged as `ARM_FREEZE_RESET`.
- **Confidence-weighted solver smoothing** (skeleton_solver):
  `DIRECTION_SMOOTH_ALPHA` is now a baseline; per-frame, per-bone α is
  `LOWCONF=0.15` when either endpoint has conf <0.4 (trust previous,
  don't let a bad frame propagate), `JUMP=0.85` when both endpoints
  ≥0.5 and direction changes sharply (dot <0.7 — legitimate sit/stand
  transition should flow through), else baseline. Exposed as
  `offline_processor.py --direction-smooth-alpha`.
- **Quality-gated bone calibration** (skeleton_solver): calibration
  samples now pass three gates — L/R pair asymmetry ≤12%, hip Z within
  ±3σ of running median, shoulder-hip dZ ≥0.25m. Fewer than 20 surviving
  frames → solve aborts with an error (previously silently calibrated
  on bad data).
- **Per-keypoint confidence in offline_processor output**: multi-pair and
  single-pair paths now emit `confidence[idx] = min(2D_conf_a, 2D_conf_b)`
  per triangulated keypoint so the solver (and future addon passes) can
  weight by it.

### v5.6 — Revert v5.5 FOOT_Z_FLOOR regression, fix pinning speed reference
- **Removed FOOT_Z_FLOOR clamp**: the v5.5 floor-at-15cm-below-ground check created
  a permanent 0.15m Z gap between `pos_scaled` (ground-clamped to 0) and the stored
  `prev_foot_raw` (clamped to -0.15). At 21fps that's a phantom ~3.2 m/s — above
  the pin threshold — so the pin logic concluded the foot was always moving fast.
  Symptom: left foot pinned 0/293 frames in take_20260420_230051. Feet snappy,
  never planted. ground_clamp still catches visual penetration; the velocity clamp
  still catches genuine spikes.
- **Pinning speed uses raw_pos**: previously `foot_speed = (pos_scaled - prev_foot_raw)`
  which mixed post-ground-clamp with pre-ground-clamp references. Now uses
  `(raw_pos - prev_foot_raw)` so small oscillations around z=0 stay consistent.
  This was the same class of bug as FOOT_Z_FLOOR — just less severe before v5.5.

### v5.5 — Foot clamp fixes, Camera C disabled (REGRESSED — fixed in v5.6)
- **Foot Z floor clamp (FOOT_Z_FLOOR = -0.15m)**: feet can't be more than 15cm below
  floor. Catches gradual depth drift that was too slow for the velocity clamp — left foot
  Z drifted from -0.01 to -0.61m over 6 frames (speeds 0.17→4.1 m/s, all under 6 m/s
  threshold). The Z floor clamp cuts off the drift before it accumulates.
- **Foot velocity clamp always-on**: removed `and not is_sitting` condition. The clamp
  was disabled during sitting and sit transitions, exactly when depth spikes are worst.
- **Stale prev_foot_raw fix**: when sitting skips foot IK bones, prev_foot_raw is now
  cleared to None. Previously it held the position from the last standing frame (could be
  hundreds of frames ago), causing a huge velocity spike on the first frame after standing.
- **Camera C disabled**: iPad via Camo was wifi-only with watermark, caused frame capture
  failures and crashes. CAM_C_INDEX set to -1. Need USB-connected device for viable 3rd camera.

### v5.4 — Neck pitch correction, foot velocity clamp, neck velocity limit
- **Neck pitch correction**: when torso pitch is clamped during sitting (e.g. raw -32°
  → clamp -20°), ear_mid still reflects the raw spine tilt. Without correction, the
  neck rotates 12°+ forward to compensate ("snake neck"). Fix: rotate neck_dir by the
  same pitch correction amount before computing neck FK. Neck now only rotates for
  actual head movement, not depth error already handled by the torso clamp.
- **Foot velocity clamping (FOOT_MAX_SPEED = 6.0 m/s)**: arms had ARM_MAX_SPEED=8 m/s
  but feet had zero velocity protection. Left foot hit 19.4 m/s from depth spikes
  (right foot only 3.0 m/s — camera geometry asymmetry). Now holds previous position
  when foot speed exceeds threshold. Mirrors the arm velocity clamp pattern.
- **Neck angular velocity limiting (NECK_MAX_ANGULAR_VEL = 8°/frame)**: caps neck
  rotation change to ~160°/s at 20fps. Prevents violent head-snap from depth spikes.
  The user's slow head turn was becoming an instant snap in Blender. Applied after
  NECK_ROT_MAX cap, before confidence slerp.

### v5.3 — Smooth sit_blend ramp
- **sit_blend ramp (SIT_BLEND_FRAMES = 8)**: replaced binary is_sitting flag with
  0→1 blend over 8 frames. All seated dampers (hip lateral, arm depth, leg lateral,
  torso pitch clamp) are multiplied by sit_blend instead of gated by a boolean.
  Prevents visible pops on sit/stand transitions.

### v4.9 — Frame sync fix, seated arm depth damping
- **Frame sync (capture pipeline)**: replaced threaded `.read()` with sequential
  `grab()/retrieve()` pattern in melodic_capture.py. Both cameras now grab frames
  microseconds apart on the main thread instead of 50-100ms apart in parallel
  threads. Also set `CAP_PROP_BUFFERSIZE=1` to minimize USB frame buffer latency.
  This is the OpenCV-recommended approach for multi-camera synchronization without
  hardware triggers.
- **Seated arm depth damping (SEATED_ARM_DEPTH_DAMP = 0.30)**: the depth axis
  (Y in Blender/armature space) is the noisiest axis in stereo triangulation.
  During sitting, upper arm Y component showed +0.70 (elbow 16cm forward of
  shoulder — physically wrong for arms on armrests). Y is damped to 30% during
  sitting, keeping arms at the body's sides in depth. Only affects upper arm FK
  when `is_sitting` is true.

### v4.7 — Direction stability, 4th-order Butterworth, pipeline tightening
- **Context-aware splay clamp (FAILED, replaced in v4.8)**: elbow height approach
  didn't work for lateral raises or seated poses.
- **Direction stability confidence boost**: arm_fk_conf no longer drops to 0 when
  wrist-shoulder distance is short. If shoulder→elbow direction is stable frame-
  to-frame (dot product > 0.95), confidence gets up to +0.5 boost. Fixes arms
  snapping to sides when sitting on armrests.
- **4th-order Butterworth filter**: upgraded from 2nd-order to two cascaded biquad
  sections. Professional standard (matches Vicon/Pose2Sim).
- **Capture pipeline tightened**:
  - OUTLIER_MAX_VELOCITY: 2.0 → 0.3 m/frame (~6.3 m/s at 21fps)
  - OUTLIER_MAX_VELOCITY_FEET: 0.5 → 0.15 m/frame (~3.15 m/s at 21fps)
  - Bone length tolerances: 0.4 → 0.20-0.25 (limbs vs arms)
  - Floor calibration confidence uses config threshold (not hardcoded 0.3)
- **recorder.py fixes**:
  - keypoint_format now reflects actual mode (coco_body_17 vs coco_wholebody_133)
  - Per-keypoint confidence stored: min(conf_a, conf_b) per triangulated point

## RESOLVED — ARM_SPLAY_MAX = 0.10 Destroyed Arm Raises (v4.4-v4.7)

**This was the #1 regression from v3.x.** The ARM_SPLAY_MAX clamp was added in
v4.4 to fix chicken-wing (elbows splaying outward during standing). It clamped
the outward X component of the upper arm FK direction to 0.10 in armature space.

**Problem**: Lateral arm raises produce X of 0.26+ in armature space. Even v4.7's
context-aware clamp using elbow height kept splay_lim at 0.15 because elbows
stay at/below shoulder height during lateral raises. Also made seated poses worse
by pushing elbows inside the torso.

**Root cause**: The original chicken-wing was an IK solver artifact (v3.x) —
when wrist is close to shoulder, IK solver bends elbow sideways. In FK mode,
this doesn't happen. Triangulation noise is small (X varies ~0.15-0.25), not
the 0.5-0.8 feared.

**Fixed in v4.8**: Single high fixed limit (ARM_SPLAY_LIMIT = 0.80). Acts as
safety net for extreme triangulation artifacts only. arm_fk_conf + velocity
clamp provide the real quality control for arm data.

## Other Known Issues (as of v4.9)

### Left leg bending during sitting (mitigated in v4.8)
- Left shin_fk has consistent X offset (-0.115 to -0.145) vs right (-0.063 to -0.119)
- Root cause: camera placement asymmetry (both X values negative = same direction bias)
- v4.8 adds SEATED_LEG_LATERAL_DAMP=0.25 to reduce X to 25% during sitting
- Residual asymmetry may remain if cameras are very unequally placed

### Head pitch_via_neck diagnostic shows 60°+ during sitting
- This is a RAW MOCAP diagnostic metric (nose-ear angle), NOT the applied rotation
- The actual applied neck_rot is 23-29° during sitting, which is reasonable
- The high raw value includes the torso lean in the measurement
- The depsgraph fix in v4.6 IS working for the applied rotation

### Capture pipeline (remaining issues)
- 3 independent 1D Kalman filters per keypoint (should be coupled 3D)
- ~~No camera synchronization mechanism~~ → fixed in v4.9 (grab/retrieve)
- 90° camera angle is geometrically suboptimal — ~45° is better (FreeMocap/Rokoko approach)
- Kalman filter treats all 3 axes equally — depth axis (Y) should get higher measurement noise
- Coordinate transform blender↔CV on stereo_calibration.py lines 528-531, 629
  — verified CORRECT despite looking suspicious

### Fixed in v4.7 (were Tier 1/2 audit items)
- ~~OUTLIER_MAX_VELOCITY~~ → tightened to 0.3 m/frame (was 2.0)
- ~~OUTLIER_MAX_VELOCITY_FEET~~ → tightened to 0.15 m/frame (was 0.5)
- ~~Bone tolerances 0.4~~ → 0.20-0.25 depending on limb
- ~~2nd-order Butterworth~~ → 4th-order (two cascaded biquads)
- ~~No per-keypoint confidence~~ → min(conf_a, conf_b) in JSON
- ~~keypoint_format lie~~ → reports actual mode (body_17 vs wholebody_133)
- ~~Floor calibration hardcoded 0.3~~ → uses config.MIN_KEYPOINT_CONFIDENCE

## Depth Axis — The Fundamental Limit of 2-Camera Stereo

All seated/sitting issues trace to **depth axis noise**. In stereo triangulation,
the axis perpendicular to the camera baseline has the worst resolution. At 90°
camera angle with 2.2m baseline, depth errors are 3-5x larger than lateral errors.

Symptoms of depth noise (all confirmed in real takes):
- Hip Z under-reported during sitting (0.245m vs real 0.4-0.5m)
- Spine backward tilt exaggerated (stereo sees -32° when real is ~-15°)
- Left/right foot asymmetric noise (camera geometry favors one side)
- Arm depth (Y) over-estimated during sitting (elbows appear 16cm forward)

The retargeter has 10+ dampers/clamps for depth noise (YAW_DEPTH_DAMP,
SEATED_ARM_DEPTH_DAMP, SEATED_HIP_LATERAL_DAMP, SEATED_PITCH_MIN/MAX, etc.).
Each fixes one symptom but can cause side effects. The real fix is better
camera geometry: 3 cameras at 30-45° angles with multi-pair triangulation.

## MediaPipe vs RTM Format — How to Tell

Pre-MelodicCapRTM takes (MelodicCapFresh era) used MediaPipe single-camera 3D.
These **cannot be retargeted** — MediaPipe normalizes to hip-centered space
(hip always at origin), making the Z axis scale-free.

How to identify monocular takes:
- `hip_raw: (0.000, -0.001, 0.000)` through all frames — hip never moves
- Leg length asymmetry >5% at frame 0
- No `format: "melodiccap_rtm_v1"` in JSON metadata
- `apply_solver.py` prints a format warning when processing these

RTM stereo takes have:
- `format: "melodiccap_rtm_v1"` and `keypoint_format: "coco_body_17"` in metadata
- hip_raw translates between frames (world-space coordinates)
- Leg asymmetry <3% at frame 0

## Frame 0 A-Pose Guide

Frame 0 is used for calibration (proportion measurement, rest pitch/yaw, hip Z
baseline, foot Z offset). A bad frame 0 poisons everything downstream.

Required pose:
- Stand naturally, feet shoulder-width apart
- Arms straight out at ~45° from body (A-pose), palms forward
- Face the camera-forward direction (perpendicular to baseline)
- Hold for 2 full seconds at the start of recording

The addon's `validate_frame0_pose()` checks: spine tilt <15°, arm asymmetry
<10%, wrists near hip height, hip Z reasonable. Warnings are logged but
currently don't block retargeting (planned enforcement in future version).

## Troubleshooting

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| Character stretches vertically | Check `hip_raw` in log — if stuck at (0,0,0), it's monocular input | Use RTM stereo takes only |
| Arms flap wildly | Check `arm_fk_conf` — should be >0.6 | Recapture with better lighting/angles |
| Character leans forever | `torso_rest_pitch` captured a lean at frame 0 | Re-record with clean A-pose |
| Neck stretches like snake | Torso pitch clamped + neck compensating | v5.4 pitch correction fixes this |
| Sitting too shallow | Depth axis under-reports hip Z drop | Use 3-camera multi-pair mode |
| Left foot pops but right doesn't | Camera geometry asymmetry + gradual Z drift | v5.6 velocity clamp (ground_clamp handles visual); move cameras closer together |
| Feet snappy, never plant | `foot_speed` reference mismatch (pre- vs post-clamp) | v5.6 fixes — uses raw_pos consistently |
| Head turns snap violently | No neck angular velocity limit | v5.4 adds 8°/frame cap |
| "HIP TOO LOW" at frame 0 | Person not standing upright, or monocular data | Clean A-pose, verify stereo format |
| `offline_processor` says "single pair" | AC/BC pairs failed quality gates (floor offset, RMS, baseline) | Calibrate all 3 pairs with floor propagation |
| `[HARD BLOCK] L/R bone lengths diverge` (forearm/shin >2x ratio) on solver-output | Position-averaging in addon's `smooth_frames` compresses bone lengths when forearm direction varies frame-to-frame; the addon was double-smoothing on top of the solver | v5.12 bypasses addon smoothing when solver_meta is present. Verify with `trace_forearm.py` |

## Working Well
- Frame 0 A-pose calibration and proportion measurement
- Global/per-chain scale factors
- Torso yaw rotation
- Torso pitch (rest-subtracted forward lean)
- Foot IK positioning and ground clamping
- Foot pin/unpin blending (smooth transitions)
- Walking-aware foot pinning (hip drift slide)
- Sit/stand detection with hysteresis
- Depsgraph flush ordering (fixed in v4.6)
- 4th-order Butterworth low-pass filter (custom implementation, no scipy) (v4.7)
- Arm splay: high fixed safety-net limit 0.80 (v4.8 — replaces broken context clamp)
- Arm FK direction stability confidence boost (v4.7)
- Per-keypoint confidence in data pipeline (v4.7)
- Arm velocity clamping (rejects triangulation spikes)
- Seated leg lateral damping (reduces camera placement bias in X) (v4.8)
- Seated arm depth damping (reduces depth axis noise in upper arm Y) (v4.9)
- Frame sync via sequential grab()/retrieve() (v4.9)
- Seated hip lateral stabilization (reduces triangulation X drift) (v5.2)
- Neck rotation cap at 50° (prevents physically impossible head tilt) (v5.2)
- Head yaw cap ±60° (allows natural head turns) (v5.2)
- Seated torso pitch clamp -20°/+35° (prevents extreme lean) (v5.2)
- Dense diagnostic logging near transitions
- Neck pitch correction (prevents snake neck from torso clamp compensation) (v5.5)
- Foot velocity clamping at 6 m/s (prevents leg pops from depth spikes) (v5.6)
- Pinning speed uses raw_pos for consistency with prev_foot_raw reference (v5.6)
- Neck angular velocity limiting at 8°/frame (prevents violent head snaps) (v5.4)
- Foot prev_foot_raw cleared during sitting (prevents stale reference spike on stand) (v5.5)
- Smooth sit_blend ramp over 8 frames (prevents binary sit/stand pops) (v5.3)
- Skeleton solver v2: direction-preserving chain fitting, soft spine/hip constraints

## Known Limitations
- Frame 0 must be a clean standing A-pose
- All bone names hardcoded to JaxRigify — no abstraction for other rigs
- Single-person detection only (takes first detected person)
- No finger tracking in body_fast mode (only wholebody detector)
- FK arm rotation cannot match IK arm POSITION accuracy
- 2-camera stereo has inherent depth axis noise — 3-camera multi-pair recommended
- Sitting depth under-reported without multi-pair triangulation
