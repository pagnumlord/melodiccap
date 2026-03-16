# MelodicCap — AI Context File
# Claude should read this at the start of every session.

## Project Overview
Motion capture pipeline for "Melodic Justice" short film.
2 webcams + MediaPipe → 3D triangulation → Blender Rigify retargeting.
Target character: JaxRigify with JaxBody5 mesh, Blender 4.4.3.

## Hardware Setup
- **Camera A (index 3)**: Sony ZV-1F via Imaging Edge Webcam (MSMF, 1280x720)
- **Camera B (index 0)**: Samsung S25 via DroidCam (MSMF, requests 1280x720)
- **GPU**: NVIDIA RTX 3060 Ti (NOT used by MediaPipe Python — CPU only)
- **OS**: Windows (DirectShow camera enumeration)

## Camera Placement (Current Recommendation)
- Side by side, 0.8-1.5m apart (baseline)
- Nearly parallel, 5-10° toe-in each (NOT 45°+ — causes depth errors)
- Subject 2-3m in front, both cameras at chest height
- Both cameras must see full body (head to feet)

## Calibration Boards
- **Main board**: 10x5 ChArUco dual-sheet (DICT_4X4_50)
  - Square size: 42.86mm (measured: 1 11/16")
  - Marker size: 30.9mm (72% of square)
  - 36 corners — sufficient for robust calibration
- **Floor board**: 4x3 ChArUco (DICT_4X4_50, same dictionary)
  - Square size: 63.5mm (2.5 inches)
  - Marker size: 47.6mm (75% of square)
  - IMPORTANT: Only have ONE board visible at a time (same dictionary = ID collision)

## Pipeline Steps (in order)
1. **Calibrate (C key)**: Collect 30+ frames of main board from varied angles, then press S
2. **Floor (F key)**: Lay floor board flat on ground, press F
3. **Verify (V key)**: Hold MAIN board (big board) in view of both cameras — triangulates corners and checks accuracy. Target: < 5mm avg error.
4. **Record (R key)**: Capture motion data (requires verified calibration + floor)
5. **Post-process**: Automatic on save — velocity filter, bone enforcement, joint constraints, Gaussian smooth
6. **Retarget in Blender**: Import JSON, addon retargets to JaxRigify

## Key File Locations
| File | Purpose |
|------|---------|
| `MelodicCapStudio/capture/melodic_capture.py` | Main capture + calibration + recording |
| `MelodicCapStudio/capture/robust_calibration.py` | Standalone calibration tool |
| `MelodicCapStudio/blender/melodiccap_retargeter.py` | Blender addon for import/retarget |
| `MelodicCapStudio/calibration/stereo_calibration.json` | Current calibration data |
| `MelodicCapStudio/calibration/generate_boards.py` | Board generator |
| `MelodicCapStudio/CAMERA_SETUP_GUIDE.md` | Camera angles, GPU, quality info |
| `PLAN_FORWARD.md` | Comprehensive diagnosis and fix plan |

## Known Issues & Constraints
- MediaPipe Python does NOT use CUDA — runs on CPU via TFLite
- DroidCam adds software latency vs real USB camera — problematic for fast movements
- Sony ZV-1F focal length ~780px vs DroidCam ~817px — ~5% mismatch, acceptable
- Current effective FPS: ~4-5 FPS at complexity=1 (live), unlimited in offline mode
- Offline mode (--offline) uses complexity=2 for best quality on pre-recorded video

## Quality Targets
- Stereo RMS: < 1.0 (hard limit for recording: < 1.5)
- Floor calibration std: < 0.01m (hard limit: < 0.03m)
- Verify board error: < 5mm average
- Bone-length CV: < 5% (ideally < 2%)
- Outlier rate: < 5 per second
- Take quality rating: GOOD or FAIR

## MediaPipe Landmarks (33 total)
Key landmarks used in triangulation and retargeting:
- 11/12: Left/Right shoulder
- 13/14: Left/Right elbow
- 15/16: Left/Right wrist
- 23/24: Left/Right hip
- 25/26: Left/Right knee
- 27/28: Left/Right ankle

## Pose Estimator Status
Currently using: MediaPipe BlazePose (Python, CPU-only)
Alternatives considered but NOT implemented:
- RTMPose (CUDA support, used by Pose2Sim) — significant code change
- ViTPose (CUDA support, state-of-art accuracy) — significant code change
Decision: Not worth switching yet. CPU performance is adequate. Offline mode eliminates FPS constraint.

## Previous Session History
- Camera indices changed multiple times. Current confirmed: Sony=3, DroidCam=0
- Original 4x3 board (6 corners) caused severe overfitting (k3=-17.1). Upgraded to 10x5 (36 corners).
- Post-processing upgraded to v3.0 with biomechanical constraints
- Camera angle was previously too steep (45°+). Recommended: 10-15° total convergence.
