# MelodicCap Studio v1.0

A clean, simple motion capture system for **Melodic Justice**.

## Quick Start

### Step 1: Install Dependencies

Open Command Prompt and run:

```
pip install opencv-python opencv-contrib-python mediapipe numpy
```

### Step 2: Configure Cameras

Edit `melodic_capture_v1.py` and update these lines (around line 30):

```python
CAM_A_INDEX = 2  # Change to your Sony ZV-1F index
CAM_B_INDEX = 4  # Change to your Samsung S25 DroidCam index
```

To find your camera indices, run this in Python:
```python
import cv2
for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"Camera {i}: Available")
        cap.release()
```

### Step 3: Run the Application

Double-click `run_melodiccap.bat` or run:
```
python melodic_capture_v1.py
```

### Step 4: Calibrate Cameras

1. Print the ChArUco board (or display on iPad at 100% scale)
2. Press **C** to start collecting calibration frames
3. Move the board around, covering different angles and distances
4. When you have 20+ frames, press **S** to run calibration
5. Calibration saves automatically to `calibration/stereo_calibration.json`

### Step 5: Calibrate Floor

1. Place the ChArUco board flat on the ground
2. Press **F** to calibrate the floor plane
3. This sets Z=0 at floor level

### Step 6: Record Motion

1. Press **R** to start recording
2. Perform your motion
3. Press **R** again to stop
4. JSON file saves to `takes/` folder

---

## Blender Import

### Install Addon

1. Open Blender
2. Edit → Preferences → Add-ons
3. Click "Install..." 
4. Select `melodiccap_blender_addon.py`
5. Enable "MelodicCap Importer"

### Import Motion

1. Select your JaxRigify armature
2. Go to View3D sidebar → MelodicCap tab
3. Click "Import JSON Take"
4. Select your recorded JSON file
5. Adjust settings in the file browser:
   - **Use IK Targets**: Enable for smoother hands/feet
   - **Use FK Rotations**: Enable for limb angles
   - **Ground Clamp Feet**: Prevents feet going through floor
   - **Foot Pin Threshold**: Higher = stickier feet (0.02 recommended)

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| Q | Quit application |
| C | Start/stop collecting calibration frames |
| S | Run stereo calibration |
| F | Calibrate floor plane |
| R | Start/stop recording |

---

## Folder Structure

```
MelodicCapFresh/
├── melodic_capture_v1.py      # Main capture application
├── melodiccap_blender_addon.py # Blender import addon
├── run_melodiccap.bat         # Launch script
├── calibration/
│   └── stereo_calibration.json # Camera calibration
└── takes/
    └── take_YYYYMMDD_HHMMSS.json # Recorded takes
```

---

## Troubleshooting

### "Cameras not opening"
- Check camera indices in Config class
- Make sure DroidCam is running before starting
- Try unplugging/replugging USB cameras

### "Board not detected"
- Ensure good lighting
- Board should be fully visible in BOTH cameras
- Print at 100% scale or verify iPad display size

### "Calibration RMS too high"
- Collect more frames (30+)
- Move board to different distances and angles
- Ensure board is flat and not warped

### "3D positions look wrong"
- Re-run stereo calibration
- Check that both cameras can see your full body
- Calibrate floor again

### "Animation jittery in Blender"
- Increase smoothing in capture (Kalman filter)
- Use FK mode instead of IK
- Reduce foot pin threshold

---

## Technical Details

### Coordinate System

- **OpenCV**: X=right, Y=down, Z=forward
- **Blender**: X=right, Y=forward, Z=up
- Conversion: `blender = (cv_x, cv_z, -cv_y)`

### MediaPipe Landmarks

33 body landmarks tracked:
- 0-10: Face
- 11-12: Shoulders
- 13-14: Elbows
- 15-16: Wrists
- 17-22: Hands
- 23-24: Hips
- 25-26: Knees
- 27-28: Ankles
- 29-32: Feet

### JaxRigify Bone Mapping

| Body Part | FK Bone | IK Bone |
|-----------|---------|---------|
| Torso | torso | - |
| Spine | spine_fk, spine_fk.001-003 | - |
| L Arm | upper_arm_fk.L, forearm_fk.L | hand_ik.L |
| R Arm | upper_arm_fk.R, forearm_fk.R | hand_ik.R |
| L Leg | thigh_fk.L, shin_fk.L | foot_ik.L |
| R Leg | thigh_fk.R, shin_fk.R | foot_ik.R |

---

## Version History

- **v1.0** - Clean rewrite, verified with JaxRigify
