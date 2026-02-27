# Motion Capture to Rigify Retargeting System

A complete pipeline for recording dual-camera motion capture and retargeting to Blender Rigify characters.

**Built for**: Melodic Justice Short Film  
**Target Character**: JaxRigify with JaxBody5 mesh  
**Blender Version**: 4.4.3

---

## 🎯 What This System Does

1. **Records** your performance using 2 webcams + MediaPipe
2. **Triangulates** 3D positions from dual camera angles
3. **Imports** motion data into Blender as an armature
4. **Retargets** animation to your Rigify character (JaxRigify)
5. **Outputs** clean, keyframed animation ready for refinement

---

## 📁 Files Overview

### Core System Files

| File | Purpose | Location |
|------|---------|----------|
| `mocap_recorder.py` | Dual-camera recording app | Run standalone |
| `mocap_rigify_retarget.py` | Blender addon for import | Install in Blender |
| `rigify_constraint_retarget.py` | Advanced retargeting script | Run in Blender Text Editor |
| `WORKFLOW_GUIDE.md` | Complete documentation | Read first! |
| `quick_start_test.py` | System testing script | Run before recording |
| `requirements.txt` | Python dependencies | `pip install -r` |

### Supporting Files

- `README.md` - This file
- `system_diagram.html` - Visual workflow diagram (open in browser)

---

## 🚀 Quick Start (5 Minutes)

### 1. Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt --break-system-packages

# Test your system
python quick_start_test.py
```

### 2. Install Blender Addon

1. Open Blender 4.4.3
2. Edit → Preferences → Add-ons → Install
3. Select `mocap_rigify_retarget.py`
4. Enable the addon

### 3. Record Your First Take

```bash
# Start recorder (adjust camera IDs if needed)
python mocap_recorder.py --camera_0 0 --camera_1 1

# In the window:
# - Position yourself visible in BOTH cameras
# - Press SPACE to start recording
# - Perform your action
# - Press SPACE to stop
# - Press Q to quit
```

Files saved in `./mocap_takes/`

### 4. Import to Blender

1. Open your scene with JaxRigify
2. File → Import → Mocap JSON
3. Select your take file
4. See animated mocap armature

### 5. Retarget to JaxRigify

**Option A: Using Addon Panel**
1. View3D → Sidebar → Mocap tab
2. Click "Retarget to Rigify"
3. Select source: MocapArmature
4. Select target: JaxRigify
5. Click OK

**Option B: Using Script (Recommended)**
1. Open `rigify_constraint_retarget.py` in Text Editor
2. Edit armature names (lines 133-134)
3. Run script
4. Preview animation
5. In console: `bake_animation()`
6. In console: `remove_constraints()`

Done! JaxRigify now has your animation.

---

## 📖 Documentation

### For First-Time Users

1. **Read FIRST**: `WORKFLOW_GUIDE.md` - Complete walkthrough
2. **Visual Guide**: Open `system_diagram.html` in browser
3. **Test System**: Run `quick_start_test.py`

### Troubleshooting Guides

See `WORKFLOW_GUIDE.md` section: [Troubleshooting](#troubleshooting)

Common issues:
- Cameras not found → Check USB connections, permissions
- Bones don't move → Check IK/FK switches, bone names
- Animation jittery → Apply smoothing, better lighting
- Feet sliding → Add floor constraints
- Wrong scale → Scale mocap rig to match character

---

## 🎓 Key Concepts

### Why Your Previous Attempts Failed

**Problem 1: You were targeting deformation bones**
- Rigify has 3 layers: Control → Mechanism → Deformation
- You must animate **control bones** (torso, hand_ik.L, etc.)
- Not DEF- bones (those are driven by controls)

**Problem 2: Flat triangulation**
- Single camera can't capture depth properly
- Dual cameras need proper calibration
- This system handles coordinate conversion correctly

**Problem 3: Wrong retargeting approach**
- Direct keyframe copying is unstable
- Constraint-based retargeting is more robust
- Allows preview before baking

### Rigify Control Bones (What You Animate)

**Body**:
- `torso` - Hips/root control
- `chest` - Upper torso
- `spine_fk` - Spine controls
- `head` - Head control

**Arms** (IK mode):
- `hand_ik.L` / `hand_ik.R` - Hand position
- `upper_arm_ik_target.L/R` - Elbow pole targets

**Arms** (FK mode):
- `upper_arm_fk.L/R` - Shoulder rotation
- `forearm_fk.L/R` - Elbow rotation  
- `hand_fk.L/R` - Hand rotation

**Legs** (IK mode):
- `foot_ik.L/R` - Foot position
- `thigh_ik_target.L/R` - Knee pole targets

**Legs** (FK mode):
- `thigh_fk.L/R` - Hip rotation
- `shin_fk.L/R` - Knee rotation
- `foot_fk.L/R` - Ankle rotation

**Use IK for mocap** - it's more natural!

---

## 🔧 System Requirements

### Hardware

**Minimum**:
- 2 USB webcams (720p)
- 8GB RAM
- Blender-compatible GPU

**Recommended**:
- 2 USB webcams (1080p, 30fps+)
- 16GB RAM
- NVIDIA GPU for faster MediaPipe

### Software

- Python 3.8+
- Blender 4.4.3
- OpenCV 4.8+
- MediaPipe 0.10+
- NumPy 1.24+

### Camera Setup

- **Position**: 45-90° apart
- **Distance**: 1.5-2m from subject
- **Height**: Chest level
- **Lighting**: Even, no harsh shadows
- **Background**: Plain, contrasting with subject

---

## 📊 Workflow Diagram

```
┌─────────────────┐
│  Real World     │
│  Performance    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  1. CAPTURE                 │
│  mocap_recorder.py          │
│  - Dual camera recording    │
│  - MediaPipe pose tracking  │
│  - 3D triangulation         │
│  Output: .json + .bvh       │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  2. IMPORT                  │
│  mocap_rigify_retarget.py   │
│  - Creates mocap armature   │
│  - Applies keyframes        │
│  Output: Animated armature  │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  3. RETARGET                │
│  rigify_constraint_retarget │
│  - Maps to control bones    │
│  - Uses constraints         │
│  - Bakes to keyframes       │
│  Output: JaxRigify animated │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  4. REFINE (Manual)         │
│  - Graph Editor timing      │
│  - Add face animation       │
│  - Polish hand/foot poses   │
│  - Add secondary motion     │
│  Output: Final animation!   │
└─────────────────────────────┘
```

Open `system_diagram.html` for interactive version.

---

## 🎬 Production Tips

### Recording Best Practices

✅ **DO**:
- Wear fitted clothing
- Start/end in neutral pose (A-pose)
- Move slower than final speed
- Keep full body in BOTH cameras
- Record in good lighting
- Hold poses 1 second at start/end

❌ **DON'T**:
- Wear baggy/reflective clothing
- Move too fast
- Occlude body parts
- Record with shadows/backlight
- Start recording immediately

### Animation Refinement

After retargeting:

1. **Timing** - Use Graph Editor
   - Speed up/slow down sections
   - Add ease in/out
   - Remove unwanted holds

2. **Face** - Manual keyframing
   - Jaw open/close
   - Eye movement/blinks
   - Eyebrow raises
   - Use shape keys for expressions

3. **Hands** - Refine finger poses
   - MediaPipe hand tracking is approximate
   - Key poses at important moments
   - Add personality to gestures

4. **Feet** - Lock to ground
   - Add Floor constraints
   - Key foot_ik positions at contact
   - Prevent sliding

5. **Secondary** - Add life
   - Shoulder overlap
   - Spine settle
   - Weight shifts
   - Anticipation/follow-through

### Building Your Mocap Library

Create reusable takes:

- **Walk cycles** (different speeds/styles)
- **Idles** (neutral, tired, alert)
- **Gestures** (pointing, waving, thinking)
- **Actions** (sitting, reaching, jumping)
- **Reactions** (surprise, laugh, recoil)

Mix and match in NLA Editor!

---

## 🔮 Future Enhancements

### Planned Features

- [ ] Camera calibration tool
- [ ] 3+ camera support
- [ ] Face tracking integration (MediaPipe Face Mesh)
- [ ] Hand tracking (MediaPipe Hands)
- [ ] Live retargeting preview
- [ ] Mocap data cleaning/smoothing tools
- [ ] Auto-scale mocap to character
- [ ] Batch processing multiple takes

### Integration with Other Tools

**FreeMoCap**:
- Better multi-camera support
- Automatic calibration
- GUI application
- Drop-in replacement for recorder

**Rokoko Studio** (Free tier):
- Motion cleanup
- Direct Blender integration
- Online processing

**Move.ai** (Research):
- Single camera 3D pose
- Paid service

---

## 🐛 Known Issues

1. **Triangulation accuracy**
   - Current implementation is simplified
   - Proper camera calibration improves this
   - See Advanced Topics in WORKFLOW_GUIDE.md

2. **Hand/finger tracking**
   - MediaPipe Pose has limited finger detail
   - Use MediaPipe Hands for better results
   - May need manual refinement

3. **Coordinate system**
   - MediaPipe uses different axes than Blender
   - Current conversion: X→X, Y→Z, Z→-Y
   - May need adjustment for your setup

4. **Performance**
   - Real-time tracking can drop frames
   - Lower camera resolution if needed
   - Consider recording at 24fps vs 30fps

---

## 📝 Project Structure

```
mocap_rigify_system/
├── README.md                          # This file
├── WORKFLOW_GUIDE.md                  # Complete documentation
├── requirements.txt                   # Python dependencies
├── quick_start_test.py               # System test script
│
├── mocap_recorder.py                 # Recording application
├── mocap_rigify_retarget.py         # Blender addon
├── rigify_constraint_retarget.py    # Advanced retargeting
│
├── system_diagram.html               # Visual workflow
│
└── mocap_takes/                      # Output directory
    ├── take_20260208_143022.json    # Motion data
    └── take_20260208_143022.bvh     # BVH format
```

---

## 🤝 Contributing

For "Melodic Justice" production:

1. Document any workflow improvements
2. Share successful camera setups
3. Create template takes (walks, idles, etc.)
4. Report bugs and fixes
5. Suggest feature additions

---

## 📜 License

Created for "Melodic Justice" Short Film production.

Free to use and modify for your project.

Credits appreciated but not required.

---

## 🎯 Success Criteria

Your system is working when:

- ✅ Both cameras track your full body
- ✅ Green skeleton appears in recorder
- ✅ JSON file imports to Blender
- ✅ Mocap armature animates smoothly
- ✅ JaxRigify moves naturally after retargeting
- ✅ Animation looks good at 24fps playback
- ✅ No major sliding/popping/jitter

---

## 🆘 Getting Help

1. **Read**: WORKFLOW_GUIDE.md
2. **Check**: Troubleshooting section
3. **Test**: Run quick_start_test.py
4. **Verify**: Camera placement, lighting
5. **Debug**: Check Blender console for errors

Common fixes solve 90% of issues!

---

## 🎬 Next Steps

1. ✅ Install system (`pip install -r requirements.txt`)
2. ✅ Test cameras (`python quick_start_test.py`)
3. ✅ Install Blender addon
4. ✅ Record test take (simple action)
5. ✅ Import and retarget
6. ✅ Verify complete pipeline works
7. ✅ Record production takes
8. ✅ Make Melodic Justice! 🎥

---

**Good luck with your short film!** 🎬✨
