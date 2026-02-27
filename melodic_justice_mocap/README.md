# Melodic Justice Motion Capture System

A dual-camera markerless motion capture system designed specifically for retargeting to Rigify rigs in Blender, created for the short film "Melodic Justice".

## 🎬 Overview

This system provides:
- **Dual-camera recording** with proper stereo triangulation
- **ChArUco board calibration** for accurate 3D reconstruction
- **MediaPipe pose detection** for markerless tracking
- **Blender addon** for importing and retargeting to Rigify rigs
- **Full hand tracking** support

### Key Features
- ✅ Proper 3D triangulation from dual cameras
- ✅ Targets Rigify **control bones** (not DEF bones)
- ✅ IK-based retargeting for natural motion
- ✅ Coordinate system conversion (MediaPipe → Blender)
- ✅ A-pose recording support
- ✅ Hand and finger tracking

## 📁 Project Structure

```
melodic_justice_mocap/
├── main.py                 # Main application (GUI + CLI)
├── config.py               # Configuration and bone mappings
├── requirements.txt        # Python dependencies
├── calibration/
│   └── calibrate.py        # Camera calibration with ChArUco
├── capture/
│   └── dual_camera.py      # Dual camera recording + triangulation
├── blender_addon/
│   └── melodic_justice_mocap.py  # Blender import/retarget addon
└── data/                   # Output directory for takes
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd melodic_justice_mocap
pip install -r requirements.txt
```

### 2. Generate Calibration Board

```bash
python main.py --cli board --output charuco_board.png
```

Print the board at **100% scale** and verify the square sizes match (default: 4cm squares).

### 3. Calibrate Cameras

Position your two cameras at approximately **45-60 degrees** apart, facing your capture area.

```bash
python main.py --cli calibrate --left 0 --right 1 --frames 20 --output ./calibration_data
```

Controls during calibration:
- **SPACE**: Capture frame pair
- **Q**: Quit

### 4. Record Motion

```bash
python main.py --cli record --left 0 --right 1 --calibration ./calibration_data/stereo_calibration.json
```

Controls during recording:
- **R**: Start/Stop recording
- **Q**: Quit

### 5. Export for Blender

```bash
python main.py --cli export take_20240101_120000 --input-dir ./mocap_data
```

### 6. Import in Blender

1. Install the Blender addon: `blender_addon/melodic_justice_mocap.py`
2. Select your JaxRigify armature
3. Open the MoCap panel (N key → MoCap tab)
4. Load the exported JSON file
5. Click "Retarget Animation"

## 📷 Camera Setup

### Recommended Hardware
- **Two cameras**: Any combination of:
  - Webcams (Logitech, etc.)
  - Smartphones via DroidCam
  - Dedicated cameras (Sony ZV-1F, etc.)
- **Resolution**: 1280x720 minimum
- **FPS**: 30 FPS recommended

### Camera Positioning
```
        [Capture Area]
              |
         45-60°
        /     \
   [Cam L]   [Cam R]
       \     /
        \   /
     [Baseline: ~0.5m]
```

- Position cameras 0.5-1m apart (baseline)
- Angle each camera 45-60° toward the center
- Both cameras should see the full capture area
- Ensure similar lighting for both views

### Finding Camera IDs

```bash
python main.py --cli record  # Will attempt to detect cameras
```

Or use the GUI and click "Detect Cameras".

## 🎯 Rigify Bone Mapping

The system maps MediaPipe landmarks to these Rigify **control bones**:

| MediaPipe Landmark | Rigify Control Bone | Type |
|-------------------|---------------------|------|
| hip_center | torso | Location + Rotation |
| spine_center | chest | Location |
| nose | head | Rotation |
| left_wrist | hand_ik.L | Location |
| right_wrist | hand_ik.R | Location |
| left_ankle | foot_ik.L | Location |
| right_ankle | foot_ik.R | Location |
| left_elbow | upper_arm_ik_target.L | Location (pole) |
| right_elbow | upper_arm_ik_target.R | Location (pole) |
| left_knee | thigh_ik_target.L | Location (pole) |
| right_knee | thigh_ik_target.R | Location (pole) |
| left_shoulder | shoulder.L | Rotation |
| right_shoulder | shoulder.R | Rotation |

### Why IK Controls?

Previous attempts failed because they targeted **DEF (deformation) bones** instead of control bones. In Rigify:

- **DEF- bones**: Driven by constraints, shouldn't be keyframed directly
- **Control bones**: What animators actually control (torso, hand_ik, etc.)
- **MCH- bones**: Mechanism bones, internal use only
- **ORG- bones**: Original metarig bones, reference only

This system correctly targets the **control bones**.

## 🎭 Recording Tips

### A-Pose Recording
For best retargeting results, start each take in an **A-pose** matching JaxRigify's rest pose:
- Arms at ~45° angle from body
- Palms facing forward
- Feet shoulder-width apart

### Capture Area
- Ensure full body is visible to BOTH cameras
- Avoid crossing limbs in front of body (occlusion)
- Good lighting from multiple angles reduces shadows

### Performance Tips
- Close other applications during recording
- Use USB 3.0 ports for webcams
- Ensure consistent frame rates from both cameras

## 🔧 Troubleshooting

### "No cameras detected"
- Check USB connections
- Try different USB ports
- On Windows, check Device Manager
- Install camera drivers if needed

### "Triangulation looks flat"
- Cameras may be too close together (increase baseline)
- Camera angle may be too small (increase to 45-60°)
- Run stereo calibration if not done

### "Retargeting doesn't work"
- Verify armature is selected in Blender
- Check if it's a Rigify rig (look for torso, hand_ik bones)
- Ensure IK/FK switch is set to IK mode
- Try adjusting the scale factor

### "Pose detection fails"
- Improve lighting
- Ensure full body is visible
- Wear contrasting clothing (avoid green)
- Try reducing model_complexity setting

## 📚 References

- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)
- [OpenCV ArUco/ChArUco](https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html)
- [Blender Rigify Documentation](https://docs.blender.org/manual/en/latest/addons/rigging/rigify/index.html)
- [FreeMoCap Project](https://freemocap.org/)

## 📄 License

Created for Melodic Justice Production. For use in the short film "Melodic Justice".

## 🤝 Contributing

This is a production tool for a specific project. Feel free to adapt for your own use.

---

**Happy Motion Capturing! 🎬**
