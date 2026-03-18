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
- **Room**: Small room — cameras can't go far back, struggling to capture full body

## Camera Placement (Current Recommendation)
- Side by side, 0.8-1.5m apart (baseline)
- Nearly parallel, 5-10° toe-in each (NOT 45°+ — causes depth errors)
- Subject 2-3m in front, both cameras at chest height
- Both cameras must see full body (head to feet)
- NOTE: 90° setup (front + side) worked for Rokoko but requires different triangulation assumptions
- For small rooms: wide-angle lenses or closer cameras with wider FOV may help

## Calibration Boards
- **Main board**: 10x5 ChArUco dual-sheet (DICT_4X4_50)
  - Square size: 42.86mm (measured: 1 11/16")
  - Marker size: 30.9mm (72% of square)
  - 36 corners — sufficient for robust calibration
  - Printed on 2x letter paper (landscape), taped together, mounted on rigid board
  - Professional copier adds border — account for this in measurements
- **Floor board**: 4x3 ChArUco (**DICT_5X5_50** — DIFFERENT dictionary from main!)
  - Square size: 63.5mm (2.5 inches)
  - Marker size: 47.6mm (75% of square)
  - 6 interior corners
  - Print LANDSCAPE on letter paper for maximum size
  - High-contrast border for detection at distance
  - **MUST use charuco_floor_4x3_DICT5X5.png** (NOT an old DICT_4X4_50 board)
  - Both boards CAN coexist (different dictionaries) but keep only one visible during calibration

## Pipeline Steps (in order)
1. **Calibrate (C key)**: Collect 30+ frames of main board from varied angles, then press S
2. **Floor (F key)**: Lay floor board flat on ground, press F
3. **Verify (V key)**: Hold MAIN board (big board) in view of both cameras — triangulates corners and checks accuracy. Target: < 5mm avg error.
4. **Record (R key)**: Capture motion data (requires verified calibration + floor)
   - 10-second countdown (configurable via --countdown)
   - T-pose for ~15 frames to stop recording gracefully (no need to walk to keyboard)
   - Auto-stops after 120 seconds (configurable via --max-duration)
5. **Video Dump (D key)**: Record raw .mp4 from both cameras for offline re-processing
6. **Post-process**: Automatic on save — velocity filter, bone enforcement, joint constraints, Gaussian smooth
7. **Retarget in Blender**: Import JSON, addon retargets to JaxRigify

## Key File Locations
| File | Purpose |
|------|---------|
| `MelodicCapStudio/capture/melodic_capture.py` | Main capture + calibration + recording |
| `MelodicCapStudio/capture/robust_calibration.py` | Standalone calibration tool |
| `MelodicCapStudio/blender/melodiccap_retargeter.py` | Blender addon for import/retarget |
| `MelodicCapStudio/calibration/stereo_calibration.json` | Current calibration data |
| `MelodicCapStudio/calibration/generate_boards.py` | Board generator |
| `MelodicCapStudio/tools/skeleton_preview.py` | 3D visualization of take JSON |
| `MelodicCapStudio/tools/take_validator.py` | Outlier detection & cleaning tool |
| `MelodicCapStudio/CAMERA_SETUP_GUIDE.md` | Camera angles, GPU, quality info |
| `PLAN_FORWARD.md` | Comprehensive diagnosis and fix plan |
| `PLAN_GUI_APP.md` | Future GUI app architecture plan |

## CLI Arguments
```
python melodic_capture.py [cam_a] [cam_b]     # Live capture
python melodic_capture.py --offline vid_a vid_b  # Offline processing
python melodic_capture.py --lite               # Fast mode (complexity=0)
python melodic_capture.py --countdown 15       # Custom countdown
python melodic_capture.py --max-duration 60    # Max recording seconds
python melodic_capture.py --skip-frames 90     # Skip N frames in offline (buffer)
python melodic_capture.py --list-cameras       # Show available cameras
```

## Known Issues & Constraints
- MediaPipe Python does NOT use CUDA — runs on CPU via TFLite
- DroidCam adds software latency vs real USB camera — problematic for fast movements
- Sony ZV-1F focal length ~780px vs DroidCam ~817px — ~5% mismatch, acceptable
- Current effective FPS: ~4-5 FPS at complexity=1 (live), unlimited in offline mode
- Offline mode (--offline) uses complexity=2 for best quality on pre-recorded video
- Small room limits camera placement and full-body capture

## Bugs Fixed (2026-03-17 Session)
- **CRITICAL: Floor board dictionary mismatch** — `melodic_capture.py` was using DICT_4X4_50 for floor detection but `generate_boards.py` generates floor board with DICT_5X5_50. This made floor calibration nearly impossible with the correct board. Now fixed with separate `floor_detector` using DICT_5X5_50.
- **No graceful recording stop** — Added T-pose detection (hold T-pose ~15 frames) to stop recording without walking to keyboard. Also added auto-stop timer (120s default).
- **No offline buffer** — Added `--skip-frames` CLI arg to skip initial frames in offline processing so you can get into position.
- **Retargeter silent failures** — Added comprehensive take data validation: checks frame count, essential landmarks, coordinate space sanity (hip Z height), coverage stats, and calibration info. All logged to file.
- **Floor board hard to detect at distance** — Regenerated with landscape orientation, high-contrast border, and corner markers for better visibility.

## Quality Targets
- Stereo RMS: < 1.0 (hard limit for recording: < 1.5)
- Floor calibration std: < 0.01m (hard limit: < 0.03m)
- Verify board error: < 5mm average
- Bone-length CV: < 5% (ideally < 2%)
- Outlier rate: < 5 per second
- Take quality rating: GOOD or FAIR

## Coordinate System
- **Capture output**: Blender coordinates (Z-up, Y-forward)
  - `blender_x = opencv_x` (right)
  - `blender_y = opencv_z` (forward/depth)
  - `blender_z = -opencv_y + floor_offset` (up)
- **Retargeter expects**: Data already in Blender coords from capture script
- **Floor Z=0**: Established by floor calibration offset
- **NO mirroring**: Person's LEFT = Character's LEFT (both at +X)

## MediaPipe Landmarks (33 total)
Key landmarks used in triangulation and retargeting:
- 0: Nose (head tracking)
- 7/8: Left/Right ear (head tracking)
- 11/12: Left/Right shoulder
- 13/14: Left/Right elbow
- 15/16: Left/Right wrist
- 19/20: Left/Right index finger (hand rotation)
- 23/24: Left/Right hip
- 25/26: Left/Right knee
- 27/28: Left/Right ankle
- 31/32: Left/Right foot index (foot rotation)

## Retargeter Bone Mapping
- **IK targets**: hand_ik.L/R (wrists), foot_ik.L/R (ankles) — delta-from-reference in root space
- **FK chains**: upper_arm_fk, forearm_fk, thigh_fk, shin_fk (.L/.R) — V2R rotations
- **Pole targets**: upper_arm_ik_target, thigh_ik_target (.L/.R) — 3-point Keemap projection
- **Spine**: 4-segment (spine_fk, .001, .002, .003) — lean + twist distribution
- **Torso**: Location (hip delta) + rotation (hip orientation change)
- **Head**: V2R from nose-to-ear-midpoint direction
- **IK_parent=0**: CRITICAL — root space prevents double root motion
- **pole_vector=True**: CRITICAL — enables pole target bones

## Pose Estimator Status
Currently using: MediaPipe BlazePose (Python, CPU-only)
Alternatives considered but NOT implemented:
- RTMPose (CUDA support, used by Pose2Sim) — significant code change
- ViTPose (CUDA support, state-of-art accuracy) — significant code change
Decision: Not worth switching yet. CPU performance is adequate. Offline mode eliminates FPS constraint.

## DroidCam Alternatives (Samsung S25)
Options investigated for lower-latency phone streaming:
1. **Samsung built-in webcam mode** (Settings → Connected Devices → Camera) — native USB, lowest latency
2. **Iriun Webcam** (USB mode) — ~30-50ms latency, 1080p, free
3. **IP Webcam via USB tethering** — can achieve low latency over localhost
4. **scrcpy** — primarily screen mirroring, not ideal for webcam
5. **DroidCam (current)** — ~100-200ms latency, acceptable for offline mode

## Session History
- Camera indices changed multiple times. Current confirmed: Sony=3, DroidCam=0
- Original 4x3 board (6 corners) caused severe overfitting (k3=-17.1). Upgraded to 10x5 (36 corners).
- Post-processing upgraded to v3.0 with biomechanical constraints
- Camera angle was previously too steep (45°+). Recommended: 10-15° total convergence.
- 2026-03-17: Fixed floor board dictionary mismatch (DICT_4X4_50 → DICT_5X5_50)
- 2026-03-17: Added T-pose graceful stop, auto-stop timer, offline skip-frames
- 2026-03-17: Added retargeter take data validation with comprehensive logging
- 2026-03-17: Improved floor board generation (landscape, high-contrast border)
- 2026-03-17: Every Blender import failure has been with BAD TAKE DATA or FLOOR CALIBRATION ISSUES, not retargeter bugs
