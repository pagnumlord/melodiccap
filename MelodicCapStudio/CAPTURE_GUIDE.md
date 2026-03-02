# MelodicCap Studio - Capture Guide

## Prerequisites

### Hardware
- **Camera A**: Sony ZV-1F (USB-C streaming to PC)
- **Camera B**: Samsung S25 via DroidCam (USB or WiFi)
- **ChArUco Board**: 4x3 grid, 63.5mm squares (print and mount on flat surface)
- Both cameras should see the same area from different angles (~60-90 degrees apart)

### Software Dependencies
```
pip install opencv-python mediapipe numpy
```

If you already have these installed, you're good. Check with:
```
python -c "import cv2; import mediapipe; print('OK')"
```

---

## Step 0: Pull the Latest Files

```powershell
cd C:\Users\ninja\Documents\melodiccap
git pull origin claude/ai-motion-capture-blender-bOcVi
```

The capture scripts are in:
```
melodiccap/MelodicCapStudio/
  capture/melodic_capture.py      <-- Main script (run this)
  capture/robust_calibration.py   <-- Alternative calibration tool
  calibration/                    <-- Calibration data saved here
  takes/                          <-- Recorded takes saved here
  logs/                           <-- Debug logs saved here
  tools/                          <-- Validation/preview utilities
  blender/melodiccap_retargeter.py <-- Blender addon
```

---

## Step 1: Connect Your Cameras

1. **Sony ZV-1F**: Plug USB-C into PC. Set camera to USB streaming mode.
2. **Samsung S25**: Open DroidCam on phone, connect via USB or WiFi.

### Finding Camera Indices

The script defaults to `CAM_A=2` (Sony) and `CAM_B=0` (DroidCam). If these don't match your setup, you can either:

**Option A**: Pass indices as arguments:
```powershell
cd C:\Users\ninja\Documents\melodiccap\MelodicCapStudio
python capture\melodic_capture.py 2 0
```

**Option B**: Edit the top of `capture/melodic_capture.py`:
```python
CAM_A_INDEX = 2          # Sony ZV-1F
CAM_B_INDEX = 0          # DroidCam
```

If you're not sure which index is which, try different combinations. When the app opens, you'll see both camera feeds side by side - swap the numbers if they're backwards.

---

## Step 2: Run the Capture App

```powershell
cd C:\Users\ninja\Documents\melodiccap\MelodicCapStudio
python capture\melodic_capture.py
```

You'll see a split-screen window with both camera feeds and a status bar at the bottom.

**Controls:**
| Key | Action |
|-----|--------|
| **C** | Start/stop collecting calibration frames |
| **S** | Run stereo calibration (after collecting frames) |
| **F** | Floor calibration mode |
| **R** | Start/stop recording motion capture |
| **Q** | Quit |

---

## Step 3: Calibrate Cameras (Stereo Calibration)

This teaches the system how your two cameras relate to each other in 3D space.

1. Press **C** to start collecting calibration frames
2. Hold your ChArUco board so **both cameras can see it** at the same time
3. Move the board around slowly:
   - Different angles (tilt left, right, up, down)
   - Different distances (close, far)
   - Different positions (left, center, right of frame)
   - **KEY**: Try to cover the full area where you'll be performing
4. The status bar shows how many frames have been captured
   - Borders turn GREEN when both cameras see the board
   - Aim for **15-25 frames** with varied positions
5. Press **C** again to stop collecting
6. Press **S** to run calibration

**What to look for in the output:**
```
--- Iteration 1/5 ---
Active frames: 35
  Camera A RMS: 0.08    <-- Good (< 0.5)
  Camera B RMS: 0.10    <-- Good (< 0.5)
  Stereo RMS: 2.40      <-- Will improve with outlier rejection
  Removed frame 12 (error: 3.81)
  Removed frame 28 (error: 2.95)

--- Iteration 2/5 ---
Active frames: 33
  Stereo RMS: 0.85      <-- Good after removing outliers!
  Baseline: 1.44m       <-- Distance between cameras
```

The calibration now uses **iterative outlier rejection**: it calibrates, checks per-frame
reprojection error, removes the worst frames, and recalibrates. This dramatically
improves results compared to single-pass calibration.

If final Stereo RMS is > 2.0 after all iterations, try again with more frames and more varied board positions.

---

## Step 4: Floor Calibration

This tells the system where the ground plane is.

1. **Lay your ChArUco board flat on the floor** where you'll be standing
2. Make sure both cameras can see it clearly
3. Press **F**
4. The system will automatically detect the board and calculate the floor offset

**What to look for:**
```
Floor offset: 0.802m (std: 0.003m)
```
- The offset is how high the floor is in the camera's coordinate system
- Low std (< 0.01m) means the calibration is consistent

---

## Step 5: Record a Take

1. Stand in the capture area, visible to both cameras
2. Press **R** to start the countdown (10 seconds)
3. Get into A-pose (stand straight, arms slightly out) before countdown ends
4. Perform your motion
5. Press **R** again to stop recording

**For your test take, do this sequence:**
1. A-pose for ~3 seconds
2. Raise your LEFT arm only (tests L/R correctness)
3. Raise your RIGHT arm only
4. Raise both arms
5. Walk a few steps forward
6. Walk back
7. Return to A-pose

**During recording, watch for:**
- **Green "OK"** on both cameras = good tracking
- **Red "LOST"** = camera lost your pose (try to stay in frame)
- **Frame counter** in top-right shows recorded frames
- **"pred:" count** = frames where tracking was lost but Kalman predicted
- **"drop:" count** = frames completely lost (bad)

---

## Step 6: Get Debug Data Back to Me

After recording, the system saves:
1. **Take file**: `MelodicCapStudio/takes/take_YYYYMMDD_HHMMSS.json`
2. **Session log**: `MelodicCapStudio/logs/capture_session_YYYYMMDD_HHMMSS.log`
3. **Calibration**: `MelodicCapStudio/calibration/stereo_calibration.json`

### Push everything back via git:

```powershell
cd C:\Users\ninja\Documents\melodiccap
git add MelodicCapStudio/takes/ MelodicCapStudio/logs/ MelodicCapStudio/calibration/
git commit -m "New capture take with debug logs"
git push -u origin claude/ai-motion-capture-blender-bOcVi
```

### What I can analyze from the logs:
- **Per-landmark visibility scores** (which body parts had good/bad tracking)
- **Frame timing** (inter-camera sync quality)
- **Outlier counts** (how many bad triangulations were filtered)
- **Camera loss patterns** (which camera lost tracking and when)
- **Calibration quality** (RMS values, baseline, floor consistency)

---

## Step 7: Import in Blender

1. Copy the updated addon to Blender:
   - Source: `MelodicCapStudio/blender/melodiccap_retargeter.py`
   - Destination: Blender addons folder (or install via Edit > Preferences > Add-ons)

2. In Blender:
   - Select your JaxRigify armature
   - Open the **MelodicCap** panel in the sidebar (N panel)
   - Click **Import Take** and select your take JSON
   - Watch the console for analysis output

---

## Troubleshooting

### "FAILED to open Camera A/B"
- Check USB connections
- Try different camera indices: `python capture\melodic_capture.py 0 1` or `1 2` etc.
- Make sure DroidCam is running on the phone
- On Windows, close any other app using the cameras

### "Stereo RMS too high"
- Collect more calibration frames (20+)
- Vary board positions and angles more
- Make sure the board is rigid (not bent/curved)
- Both cameras must see the board simultaneously

### "Board not detected"
- Better lighting (avoid harsh shadows)
- Hold board steady
- Make sure the full board is visible
- Don't hold the board too far away

### Low frame rate during recording
- Close other applications
- MediaPipe model_complexity=1 is a good balance
- Reducing to 0 (lite) is faster but less accurate
- Make sure cameras are running at 720p, not higher

### Character looks wrong in Blender after import
- Make sure armature scale is (1, 1, 1) - apply scale if not
- Try the "Diagnostic Dump" button to save rig info
- Share the diagnostic log + take file + capture session log with me
