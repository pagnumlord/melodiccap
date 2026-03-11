# MelodicCap Studio - Full Pipeline Guide (v3.1)

## Quick Reference

```
PULL → CAMERAS → CALIBRATE → FLOOR → VERIFY → RECORD → BLENDER IMPORT
```

---

## Step 0: Pull Latest Code

```powershell
cd C:\Users\ninja\Documents\melodiccap
git pull origin claude/ai-motion-capture-blender-bOcVi
```

The files you need:
```
MelodicCapStudio/
  capture/melodic_capture.py       <-- Capture app (run this)
  calibration/stereo_calibration.json  <-- Saved calibration
  takes/                           <-- Recorded takes
  logs/                            <-- Debug logs
  blender/melodiccap_retargeter.py <-- Blender addon
```

---

## Step 1: Connect Cameras

1. **Sony ZV-1F**: Plug USB-C, set to USB streaming mode
2. **Samsung S25**: Open DroidCam, connect via USB or WiFi

### Run the capture app:

```powershell
cd C:\Users\ninja\Documents\melodiccap\MelodicCapStudio
python capture\melodic_capture.py 3 0
```

Replace `3 0` with your camera indices. If cameras appear swapped, swap the numbers.

To find available cameras:
```powershell
python ..\detect_cameras.py
```

---

## Step 2: Calibrate Stereo (Every Time Cameras Move)

**You must recalibrate every time a camera moves, even slightly.**

1. Press **C** to start collecting
2. Hold the 10x5 ChArUco board so **both** cameras see it
3. Move the board **slowly** through varied positions:
   - Tilt left/right/up/down
   - Near and far
   - Left, center, right of frame
   - Rotate the board
   - **30+ frames minimum**, 50+ is better
4. Press **C** to stop collecting
5. Press **S** to calibrate

**Good result:**
```
Stereo RMS: 0.75    <-- Under 1.0 is good
Baseline: 1.32m     <-- Should match your camera spacing
```

**If RMS > 1.5:** Try again. More frames, slower movement, more varied angles.

**If it says "KEPT OLD":** Your new calibration was worse than existing. This is a protection — try again with better technique.

---

## Step 3: Floor Calibration

After stereo calibration succeeds, the app **auto-enters floor mode**.

1. Lay a ChArUco board **flat on the floor** where you'll stand
2. Both cameras must see the board
3. The app detects it automatically and sets Z=0

**Good result:**
```
Floor offset: 0.838m (std: 0.002m)
```

Low std (< 0.01m) = good. If std > 0.05m, the board isn't flat or calibration is noisy.

If you need to redo floor later: press **F** to toggle floor mode.

---

## Step 4: Verify Calibration

1. Hold the ChArUco board visible to both cameras
2. Press **V**

The app triangulates the board corners and checks if the measured distances between corners match the real board dimensions.

**Good result:**
```
VERIFIED GOOD! Board error: 3.2mm avg, 5.1mm max (15 pairs) — GOOD
```

**Quality ratings:**
- **GOOD** (< 5mm error): Ready to record
- **OK** (5-10mm error): Acceptable
- **POOR** (> 10mm error): Recalibrate

The status bar also shows calibration health continuously when the board is visible.

**You cannot record until verification passes.**

---

## Step 5: Record a Take

1. Stand in the capture area, visible to both cameras
2. Press **R** — 10-second countdown starts
3. **Start in A-pose** (stand straight, arms slightly out from sides)
4. Hold A-pose for 2-3 seconds (this becomes your reference frame)
5. Perform your motion
6. Press **R** to stop

**Notes:**
- The last 1.5 seconds are **automatically trimmed** (removes your movement toward the keyboard)
- Watch for green "OK" on both cameras = good tracking
- Red "LOST" = pose lost (stay in frame)
- A quality report prints after each take

**For test takes, do this sequence:**
1. A-pose for 3 seconds
2. Raise LEFT arm only
3. Raise RIGHT arm only
4. Raise both arms
5. Walk forward a few steps
6. Walk back
7. Return to A-pose

---

## Step 6: Push Data for Debugging

```powershell
cd C:\Users\ninja\Documents\melodiccap
git add MelodicCapStudio/takes/ MelodicCapStudio/logs/ MelodicCapStudio/calibration/
git commit -m "New capture session"
git push -u origin claude/ai-motion-capture-blender-bOcVi
```

---

## Step 7: Import in Blender

### Install the Addon

1. Open Blender
2. Edit > Preferences > Add-ons > Install from Disk
3. Select: `MelodicCapStudio/blender/melodiccap_retargeter.py`
4. Enable the addon (check the box)

If already installed, **reload** it:
- In the addon list, disable then re-enable, OR
- Blender > Edit > Preferences > Add-ons > find "MelodicCap" > remove, then reinstall

### Import a Take

1. Select your **JaxRigify armature** in the viewport
2. Make sure armature scale is (1, 1, 1) — if not, Ctrl+A > Apply Scale
3. Open the **N panel** (press N) > **MelodicCap** tab
4. Settings to check:
   - **Filter Outliers**: ON (default)
   - **Ground Clamp Feet**: ON (default)
   - **Pole Targets**: ON (keeps elbows/knees bending correctly)
   - **IK Rotation**: ON (wrist/foot orientation)
   - **Spine Animation**: ON (torso lean + twist)
5. Click **Import Take**
6. Select your `.json` take file
7. Check the Blender console for the analysis log

### What to Look For

**In the console output:**
```
Scale factor: 1.0286          <-- Should be 0.5-2.0
Outlier filter: 0 values replaced  <-- Low is good
Frames processed: 269         <-- Should match your take
Keyframes created: 8070       <-- Many = good
```

**In the viewport:**
- Press Space to play animation
- Character should track your movements
- LEFT arm raise = character's LEFT arm raises (no mirroring)
- Feet should stay planted when you stand still
- Spine should lean/twist when you lean/twist

### Troubleshooting Retarget

| Problem | Fix |
|---------|-----|
| Arms fly off to sides | Check IK_parent=0 (addon sets this automatically) |
| Everything mirrored | This shouldn't happen (v2.0 has correct mapping) |
| Character too big/small | Check armature scale is (1,1,1), apply if not |
| Feet go through floor | Enable "Ground Clamp Feet" in settings |
| Jittery movement | Lower the "Max Landmark Speed" setting |
| Jerky ending | Last 1.5s is auto-trimmed; if still bad, trim the take JSON |

---

## Controls Reference (Capture App)

| Key | Action | Notes |
|-----|--------|-------|
| **C** | Collect/stop calibration frames | Hold board visible to both cameras |
| **S** | Run stereo calibration | Need 8+ frames (30+ recommended) |
| **F** | Toggle floor calibration mode | Lay board flat on floor |
| **V** | Verify calibration | Show board, checks triangulation accuracy |
| **R** | Start/stop recording | Blocked until verified + floor set |
| **Q** | Quit | |

### Pipeline Order (enforced by software):

```
C → S (calibrate) → F (floor) → V (verify) → R (record)
```

You cannot skip steps. Recording is blocked until calibration is verified and floor is set.

---

## File Locations

| File | What |
|------|------|
| `capture/melodic_capture.py` | Main capture + calibration app |
| `blender/melodiccap_retargeter.py` | Blender addon (v2.0) |
| `calibration/stereo_calibration.json` | Current stereo calibration |
| `calibration/generate_boards.py` | Print new ChArUco boards |
| `takes/take_*.json` | Recorded motion capture takes |
| `logs/capture_session_*.log` | Debug logs from each session |
