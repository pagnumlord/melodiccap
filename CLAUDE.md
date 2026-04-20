# MelodicCap - Multi-Camera Markerless Motion Capture

## Project Overview
Multi-camera markerless motion capture system using RTMPose for 2D pose detection,
stereo triangulation for 3D reconstruction, and a Blender addon to retarget captured
motion data onto Rigify character rigs. For a short film.

## Key Facts
- **User height**: 6'1" (1.856m)
- **Primary rig**: JaxRigify (1.87284m tall)
- **Other characters** (pending): Kai, Kiko, Dr White, Hiro, THE SHADOW
- **Cameras**: Sony ZV-1F (cam A) + Samsung S25 via DroidCam (cam B) + iPad (cam C)
- **Recommended geometry**: 30-45° angle spread, ~1.5m baseline, all cameras ~2m from performer
- **3-camera multi-pair mode**: offline_processor picks best pair (AB/AC/BC) per frame
  by reprojection error. Set CAM_C_INDEX in melodic_capture.py to enable.
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
  - `melodiccap_rtm_addon.py` — Main addon (v5.4): imports JSON takes, retargets to JaxRigify

## Blender Addon - Current State (v5.4)
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
| Left foot pops but right doesn't | Camera geometry asymmetry | v5.4 foot velocity clamp; also try 3 cameras |
| Head turns snap violently | No neck angular velocity limit | v5.4 adds 8°/frame cap |
| "HIP TOO LOW" at frame 0 | Person not standing upright, or monocular data | Clean A-pose, verify stereo format |
| `offline_processor` says "single pair" | AC/BC pairs failed quality gates (floor offset, RMS, baseline) | Calibrate all 3 pairs with floor propagation |

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
- Neck pitch correction (prevents snake neck from torso clamp compensation) (v5.4)
- Foot velocity clamping at 6 m/s (prevents triangulation spike leg pops) (v5.4)
- Neck angular velocity limiting at 8°/frame (prevents violent head snaps) (v5.4)
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
