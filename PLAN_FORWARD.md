# MelodicCap — Comprehensive Fix Plan

## ROOT CAUSE DIAGNOSIS

### Why the retargeted animation looks unusable

**The data is bad.** Every other problem is downstream of this. The numbers:

| Metric | Your Data | Professional Standard |
|--------|-----------|----------------------|
| Shoulder width variation | 0.197m - 1.165m (5.9x) | < 1% variation |
| Hip width variation | 0.193m - 1.909m (9.9x) | < 1% variation |
| Max wrist velocity | 27.2 m/s (impossible) | < 15 m/s |
| Max hip velocity | 21.8 m/s (impossible) | < 6 m/s |
| Bone-length CV (best take) | 21-43% | < 2% |

No retargeter can make a skeleton whose shoulders jump between 20cm and 116cm look good.

### The 3 root causes

1. **ChArUco board: 4x3 = only 6 corners** (need 24+ for stable calibration)
   - Camera A's k3 distortion = -17.1 (overfitting, should be < 1.0)
   - 6 corners can't constrain 9 unknowns per camera
   - File: `MelodicCapStudio/capture/melodic_capture.py` line 57-60

2. **Camera mismatch** (Sony ZV-1F fx=625px vs DroidCam fx=905px)
   - 1.45x focal length mismatch = different fields of view
   - DroidCam adds software latency (frames not truly simultaneous)
   - File: `MelodicCapStudio/calibration/stereo_calibration.json`

3. **Post-processing was band-aids** (smoothing garbage doesn't make gold)
   - 3-frame moving average on 1-meter jumps = smoothed bad data
   - 15% bone-length tolerance = still allows huge variation
   - No biomechanical joint constraints
   - File: `MelodicCapStudio/capture/melodic_capture.py` lines 168-412

### Why Rokoko and FreeMoCap work better

- **Rokoko**: Uses IMU sensors (accelerometers/gyroscopes), NOT cameras. Each body segment has a sensor. No triangulation noise. They export Mixamo-style rigs because that's the bone naming convention.
- **FreeMoCap**: Uses 2+ cameras with LARGE charuco boards (7x5 = 24 corners minimum), extensive post-processing with OpenSim biomechanical constraints, and skeleton solvers that enforce rigid bone lengths and joint limits.

Both enforce: rigid bone lengths, joint angle limits, temporal smoothing, and scale consistency.

---

## WHAT WAS FIXED

### 1. New ChArUco Boards Generated
- **Files created:**
  - `MelodicCapStudio/calibration/generate_boards.py` — board generator script
  - `MelodicCapStudio/calibration/charuco_board_5x7_single.png` — single 8.5x11" sheet, 5x7 = **24 corners**
  - `MelodicCapStudio/calibration/charuco_board_10x5_left.png` — dual sheet left half
  - `MelodicCapStudio/calibration/charuco_board_10x5_right.png` — dual sheet right half, 10x5 = **36 corners**

- **Old board**: 4x3 = 6 corners
- **New single sheet**: 5x7 = 24 corners (4x improvement)
- **New dual sheet**: 10x5 = 36 corners (6x improvement)

### 2. Calibration Pipeline Upgraded
- **File modified:** `MelodicCapStudio/capture/melodic_capture.py`
  - Board config updated: `CHARUCO_COLS=5, CHARUCO_ROWS=7, CHARUCO_SQUARE_M=0.035`
  - Added automatic distortion model selection: k3 fixed to 0 when < 200 total corner observations
  - Added extreme distortion coefficient warnings
  - Prevents the k3=-17.1 overfitting that was ruining Camera A's calibration

- **File modified:** `MelodicCapStudio/capture/robust_calibration.py`
  - Board config updated to match new board

### 3. Post-Processing Upgraded to v3.0
- **File modified:** `MelodicCapStudio/capture/melodic_capture.py` — `postprocess_take()`

  **Step 1: Adaptive velocity filtering** (was: fixed 0.5m/frame for all)
  - Per-landmark velocity limits based on biomechanics
  - Head: 5 m/s, shoulders: 8 m/s, wrists: 15 m/s, hips: 6 m/s
  - Catches the 27-1075 m/s spikes while allowing real fast movements

  **Step 2: Strict bone-length enforcement** (was: 15% tolerance)
  - NO tolerance. Bones don't change length. Period.
  - Processes in parent-before-child order (hips→torso→limbs)
  - Uses robust median from ALL frames (not just first 10)

  **Step 3: Biomechanical joint-angle constraints** (NEW)
  - Elbows: 25°-175° (no hyperextension, no over-bending)
  - Knees: 25°-178° (same)
  - Blends toward previous frame to fix violations

  **Step 4: 5-frame Gaussian-weighted smoothing** (was: 3-frame average)
  - Weights: [1, 4, 6, 4, 1] / 16
  - Preserves sharp movements better than simple average

  **Step 5: Re-enforce bone lengths** (NEW)
  - Smoothing can drift bone lengths; this pass re-locks them

  **Step 6-7: Trim + floor clamp** (unchanged)

  **Step 8: Data quality report** (NEW)
  - Reports final bone-length consistency (CV should be ~0%)
  - Outlier rate per second → GOOD/FAIR/POOR quality rating

### 4. Retargeter Outlier Filter Upgraded
- **File modified:** `MelodicCapStudio/blender/melodiccap_retargeter.py`
  - Per-landmark adaptive velocity limits (same as capture pipeline)
  - Reports per-landmark velocity limits in log
  - Velocity scale slider maps to per-landmark limits proportionally

---

## FILES REFERENCE

### Core Pipeline (what you run)
| File | Purpose |
|------|---------|
| `MelodicCapStudio/capture/melodic_capture.py` | Main capture + calibration + recording |
| `MelodicCapStudio/capture/robust_calibration.py` | Standalone calibration tool |
| `MelodicCapStudio/blender/melodiccap_retargeter.py` | Blender addon for import/retarget |
| `MelodicCapStudio/calibration/stereo_calibration.json` | Current calibration data |
| `MelodicCapStudio/calibration/generate_boards.py` | Board generator |

### Board Files (print these)
| File | What to do |
|------|------------|
| `charuco_board_5x7_single.png` | Print on ONE 8.5x11" sheet at 100% scale |
| `charuco_board_10x5_left.png` | Print LANDSCAPE, tape to right half |
| `charuco_board_10x5_right.png` | Print LANDSCAPE, tape to left half |

### Data Flow
```
[Camera A + Camera B]
     ↓ MediaPipe 2D landmarks (33 per camera)
[melodic_capture.py: triangulate_landmarks()]
     ↓ Stereo triangulation → 3D points (Blender coords)
     ↓ Kalman filter (per-landmark)
[melodic_capture.py: postprocess_take()]  ← UPGRADED v3.0
     ↓ Velocity filter → Bone enforcement → Joint constraints → Gaussian smooth
[Take JSON saved]
     ↓
[melodiccap_retargeter.py: import in Blender]
     ↓ prefilter_landmarks() → Adaptive velocity filter
     ↓ analyze() → Scale factor, reference frame
     ↓ animate() → IK targets, pole targets, FK rotations, spine
[Keyframes on JaxRigify armature]
```

### Bone Mapping (verified correct)
| Retargeter Bone | MediaPipe Landmark | Rokoko Equivalent |
|-----------------|-------------------|-------------------|
| `hand_ik.L` | 15 (left_wrist) | LeftHand |
| `hand_ik.R` | 16 (right_wrist) | RightHand |
| `foot_ik.L` | 27 (left_ankle) | LeftFoot |
| `foot_ik.R` | 28 (right_ankle) | RightFoot |
| `upper_arm_fk.L` | 11→13 (shoulder→elbow) | LeftUpperArm |
| `forearm_fk.L` | 13→15 (elbow→wrist) | LeftLowerArm |
| `thigh_fk.L` | 23→25 (hip→knee) | LeftUpperLeg |
| `shin_fk.L` | 25→27 (knee→ankle) | LeftLowerLeg |
| `upper_arm_ik_target.L` | pole from 11,13,15 | (IK pole) |
| `thigh_ik_target.L` | pole from 23,25,27 | (IK pole) |
| `spine_fk` through `.003` | virtual spine midpoints | Spine/Chest |
| `torso` | hip midpoint (23+24)/2 | Hips |

These ARE the Rigify control bones that Rokoko's custom entries target.

---

## WHAT YOU NEED TO DO

### Immediate (before next capture session)

1. **Print the new board**
   - RECOMMENDED: Print `charuco_board_10x5_left.png` + `charuco_board_10x5_right.png` LANDSCAPE
   - Tape together on the BACK along the red alignment lines
   - Mount on rigid flat surface (foam board, clipboard, cardboard)
   - OR: Print `charuco_board_5x7_single.png` portrait, mount flat

2. **Measure a printed square with a ruler**
   - If the "35mm" square measures 34.2mm, update `CHARUCO_SQUARE_M = 0.0342`
   - In `melodic_capture.py` lines 60-61
   - Also update `robust_calibration.py` lines 28-31

3. **Update board config in scripts if using dual-sheet board**
   - `CHARUCO_COLS = 10, CHARUCO_ROWS = 5, CHARUCO_SQUARE_M = 0.043, CHARUCO_MARKER_M = 0.031`

4. **Recalibrate**
   - Run capture, press C to collect 30+ frames from varied angles
   - Move board SLOWLY between positions
   - Cover: tilts, rotations, near/far, all areas of camera view
   - Press S to calibrate
   - Target: Stereo RMS < 1.0 (currently 0.87 which is OK but with bad k3)

5. **Floor calibrate**
   - Lay board flat on floor, press F

6. **Test capture**
   - Record a take, check the quality report in the log
   - Should see bone-length CV < 5% and quality rating "GOOD"

### Medium-term (if DroidCam is still causing issues)

- Consider using a second USB webcam instead of DroidCam
- Even a cheap Logitech C920 ($50) will have better frame sync than a phone app
- Matching cameras (same model) eliminates the focal length mismatch issue

### The math chain (no mismatches)

```
Board: 5x7 at 35mm = 175x245mm on 8.5x11"
  → (5-1)*(7-1) = 24 charuco corners per frame
  → 30 frames × 24 corners = 720 total observations
  → Enough to constrain 9 params/camera with k3=0 (8 unknowns)
  → k3 fixed to prevent overfitting
  → Stereo calibration with CALIB_FIX_INTRINSIC (enough data now)
  → Expected RMS: < 0.5 pixels

Capture: 1280×720 @ ~13 FPS
  → MediaPipe landmarks: ~3-5 pixel detection error (720p)
  → Triangulation at 2m distance, 1.4m baseline:
    → 5px error × (2²)/(1.4 × 765) = ~1.9cm depth error
  → Post-processing:
    → Velocity filter: catches >6 m/s hip jumps, >15 m/s wrist jumps
    → Bone enforcement: locks all bones to median length (0% variation)
    → Joint constraints: elbows 25-175°, knees 25-178°
    → Gaussian smooth: 5-frame kernel [1,4,6,4,1]/16
    → Re-enforce bones after smooth
  → Expected bone-length CV: < 2%

Retarget: Blender addon
  → Second velocity filter pass (catches any remaining spikes)
  → Scale: person height → character height
  → IK targets: delta-from-reference in root space
  → FK rotations: Vector-to-Rotation with actual bone rest axes
  → Pole targets: Keemap 3-point projection with delta-from-reference
  → Spine: 4-segment virtual midpoints
```
