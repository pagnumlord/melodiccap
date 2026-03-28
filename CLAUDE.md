# MelodicCap - Dual-Camera Markerless Motion Capture

## Project Overview
Dual-camera markerless motion capture system using RTMPose for 2D pose detection,
stereo triangulation for 3D reconstruction, and a Blender addon to retarget captured
motion data onto Rigify character rigs.

## Key Facts
- **User height**: 6'1" (1.856m)
- **Primary rig**: JaxRigify (1.87284m tall)
- **Other characters** (pending): Kai, Kiko, Dr White, Hiro, THE SHADOW
- **Cameras move every session** — old calibrations are useless. Must recalibrate
  stereo cameras at the start of each capture session. Never assume a previous
  session's calibration is reusable.

## Architecture
- `MelodicCapRTM/` — Python capture pipeline
  - `melodic_capture.py` — Main capture loop (dual camera, parallel inference)
  - `stereo_calibration.py` — Stereo camera calibration (checkerboard + bone constraints)
  - `pose_detector.py` — RTMPose wrapper (single-person detection)
  - `kalman.py` — Kalman filter for 3D keypoint smoothing
  - `recorder.py` — JSON take recorder
- `MelodicCapRTM/blender_addon/` — Blender addon
  - `melodiccap_rtm_addon.py` — Main addon: imports JSON takes, retargets to JaxRigify

## Blender Addon (melodiccap_rtm_addon.py)
- Proportional retargeting: measures mocap vs rig proportions from frame 0
- IK mode: positions hand_ik/foot_ik targets with per-chain scaling
- FK mode: rotates spine, neck, head, limb bones
- Arms use dep-graph-evaluated rig shoulder position for IK targets
- Foot pinning: speed-based ground lock with smooth pin-IN/unpin blending
- Ankle Z offset: precomputed from first 20 standing frames
- Torso: yaw-only rotation (pitch/roll NOT yet implemented)
- Head: yaw-only rotation (pitch NOT yet implemented)
- Spine FK: only first bone rotated (full chain rotation NOT yet implemented)

## Known Limitations
- Frame 0 must be a clean standing A-pose — sitting/crouching in frame 0 gives
  wrong proportions and breaks all downstream scaling
- Spine FK using only the first bone cannot represent sitting/hunching postures
- Torso yaw-only makes character look stiff during complex movements
- Head yaw-only loses pitch (nodding) and roll (tilting)
- All bone names are hardcoded to JaxRigify — no abstraction for other rigs yet
- Single-person detection only (takes first detected person)
- recorder.py always writes "coco_wholebody_133" even in body-only mode (17 kp)
