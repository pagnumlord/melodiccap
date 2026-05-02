# MelodicCap - Multi-Camera Markerless Motion Capture

## Project Overview
Multi-camera markerless motion capture system using RTMPose for 2D pose detection,
stereo triangulation for 3D reconstruction, and a Blender addon to retarget captured
motion data onto Rigify character rigs. For a short film.

## Key Facts
- **User height**: 6'1" (1.856m)
- **Primary rig**: JaxRigify (1.87284m tall)
- **Other characters** (pending): Kai, Kiko, Dr White, Hiro, THE SHADOW
- **Cameras**: Samsung S25 via DroidCam (cam A) + Logitech C615 (cam B)
- **Previous 3rd camera (iPad via Camo)**: disabled — wifi-only, watermark, crashes.
  Need USB-connected Android phone or dedicated camera for viable 3rd camera.
- **Recommended geometry**: 30-45° angle spread, ~1.5m baseline, cameras ~2m from performer
- **3-camera multi-pair mode**: offline_processor picks best pair (AB/AC/BC) per frame
  by reprojection error. Set CAM_C_INDEX in melodic_capture.py to enable (currently -1).
- **Cameras move every session** — old calibrations are useless. Must recalibrate
  stereo cameras at the start of each capture session.
- **No hardware sync** — DroidCam is NOT hardware-synced with Sony ZV-1F.
  Frame sync via sequential grab()/retrieve() (v4.9) reduces delta to ~20ms.

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

## Version History (Retargeter)

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
