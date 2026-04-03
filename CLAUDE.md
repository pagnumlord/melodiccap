# MelodicCap - Dual-Camera Markerless Motion Capture

## Project Overview
Dual-camera markerless motion capture system using RTMPose for 2D pose detection,
stereo triangulation for 3D reconstruction, and a Blender addon to retarget captured
motion data onto Rigify character rigs. For a short film.

## Key Facts
- **User height**: 6'1" (1.856m)
- **Primary rig**: JaxRigify (1.87284m tall)
- **Other characters** (pending): Kai, Kiko, Dr White, Hiro, THE SHADOW
- **Cameras**: Sony ZV-1F + Samsung S25 via DroidCam at ~90° angle, ~2m baseline
- **Cameras move every session** — old calibrations are useless. Must recalibrate
  stereo cameras at the start of each capture session.
- **No hardware sync** — DroidCam is NOT hardware-synced with Sony ZV-1F.
  ~50-100ms frame misalignment corrupts stereo depth.

## Architecture
- `MelodicCapRTM/` — Python capture pipeline
  - `melodic_capture.py` — Main capture loop (dual camera, parallel inference)
  - `stereo_calibration.py` — Stereo camera calibration (checkerboard + bone constraints)
  - `pose_detector.py` — RTMPose wrapper (single-person detection)
  - `kalman.py` — Kalman filter for 3D keypoint smoothing
  - `recorder.py` — JSON take recorder
  - `offline_processor.py` — Offline triangulation
- `MelodicCapRTM/blender_addon/` — Blender addon
  - `melodiccap_rtm_addon.py` — Main addon (v4.6): imports JSON takes, retargets to JaxRigify

## Blender Addon - Current State (v4.6)
- Proportional retargeting: measures mocap vs rig proportions from frame 0
- **Hybrid mode (default)**: Arms use FK rotations, legs use IK positioning
- Torso: yaw + pitch rotation (pitch = forward lean, rest-subtracted)
- Spine FK: single bone OR full 4-bone chain (distributes rotation 1/N per bone)
- Neck FK: parent-aware ('auto' mode) — relative to current torso/spine
- Head: yaw-only from ear line vs shoulder line
- Foot IK: positioned with per-chain scaling, speed-based pinning with smooth blend
- Foot pinning: walking-aware with hip-drift slide, smooth pin/unpin over 6-8 frames
- Ankle Z offset: precomputed from first 20 standing frames
- Sit/stand detection: hip Z drop with hysteresis (-0.15m sit, -0.10m stand)
- When sitting: legs switch to FK, foot pinning disabled
- Arm FK confidence: wrist-to-shoulder ratio < 0.7 = low confidence
- Arm splay clamp: ARM_SPLAY_MAX limits outward X in armature space
- Arm velocity clamp: rejects >8 m/s hand IK spikes
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

## CRITICAL BUG — ARM_SPLAY_MAX = 0.10 Destroys Arm Raises (v4.4+)

**This is the #1 regression from v3.x.** The ARM_SPLAY_MAX clamp was added in
v4.4 to fix chicken-wing (elbows splaying outward during standing). It clamps
the outward X component of the upper arm FK direction to 0.10 in armature space.

**Problem**: When the user raises their arms out to the side or up, the outward
X component in armature space can be 0.5-0.8+. Clamping to 0.10 crushes this,
and after renormalization the downward Z component dominates, making the arm
point nearly straight down. Example from v4.6 test at arm-raise peak:
- Mocap shoulder→elbow world direction: ~(0.83, -0.08, -0.55) — arm raised ~33°
- After armature transform + ARM_SPLAY_MAX clamp: (0.178, -0.134, -0.975) — CRUSHED

**Why it worked in v3.x**: Arms used IK, not FK. The wrist IK target was
positioned from mocap data without any splay clamp. IK solver pulled the hand
to the correct position regardless of elbow direction.

**Fix needed**: Either:
1. Increase ARM_SPLAY_MAX significantly (0.5+) and accept some chicken-wing
2. Make the clamp context-aware (only clamp when arms are near sides, not raised)
3. Use a soft clamp (reduce but don't crush) instead of hard clamp
4. Switch back to IK for arms when arm ratio > 0.7 (confident FK data)

## Other Known Issues (as of v4.6)

### Seated arm position
- Arm hold-pose fallback holds the last-good rotation from the sit transition,
  not the actual armrest position. The mocap data DOES show arms on armrests
  (elbows at reasonable positions), but arm_fk_conf drops because wrist-to-shoulder
  DISTANCE is short, even though the FK DIRECTION might be valid.
- Possible fix: base confidence on direction stability, not just distance ratio

### Left leg bending during sitting
- Left shin_fk has consistent X offset (-0.115 to -0.145) vs right (-0.063 to -0.119)
- Shin rotation angles are large (65-82°) — may be amplified by parent chain
- Could be triangulation asymmetry from camera placement (one camera closer to left side)

### Head pitch_via_neck diagnostic shows 60°+ during sitting
- This is a RAW MOCAP diagnostic metric (nose-ear angle), NOT the applied rotation
- The actual applied neck_rot is 23-29° during sitting, which is reasonable
- The high raw value includes the torso lean in the measurement
- The depsgraph fix in v4.6 IS working for the applied rotation

### Capture pipeline issues (from system audit)
- OUTLIER_MAX_VELOCITY = 2.0 m/frame is too permissive (42 m/s at 21fps)
- OUTLIER_MAX_VELOCITY_FEET = 0.5 m/frame also high
- Bone tolerances all 0.4 (40% variation — guesswork)
- 3 independent 1D Kalman filters per keypoint (should be coupled 3D)
- 2nd-order Butterworth at 4.0 Hz body / 3.5 Hz feet
  (professional systems use 4th-order at 6-10 Hz)
- No per-keypoint confidence in JSON data pipeline
- No camera synchronization mechanism
- recorder.py always writes "coco_wholebody_133" even in body-only mode (17 kp)
- Floor calibration ankle confidence 0.3 hardcoded (inconsistent)
- Coordinate transform blender↔CV on stereo_calibration.py lines 528-531, 629
  — verified CORRECT despite looking suspicious

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
- Butterworth low-pass filter (custom implementation, no scipy)
- Arm velocity clamping (rejects triangulation spikes)
- Dense diagnostic logging near transitions

## Known Limitations
- Frame 0 must be a clean standing A-pose
- All bone names hardcoded to JaxRigify — no abstraction for other rigs
- Single-person detection only (takes first detected person)
- No finger tracking in body_fast mode (only wholebody detector)
- FK arm rotation cannot match IK arm POSITION accuracy
