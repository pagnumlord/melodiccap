# MelodicCap Studio - Bug Fixes Summary

## Files Fixed

| File | Status | Key Fixes |
|------|--------|-----------|
| `triangulation_engine.py` | ✅ FIXED | Added undistortion, proper rectification |
| `post_processor.py` | ✅ FIXED | Coordinate scaling (normalized → pixels) |
| `floor_calibrator.py` | ✅ CREATED | Matches main_ui.py import expectations |
| `melodiccap_addon_v3.py` | ✅ FIXED | Panel draw(), Quaternion(), Blender 4.4 |
| `antigrav_retargeter_v3.py` | ✅ FIXED | Panel draw(), bone rest axes |

---

## Critical Bug Fixes

### 1. triangulation_engine.py

**Before:** Points were triangulated without undistortion
```python
# OLD (WRONG):
pt1 = np.array([[lm_a.x * 1280, lm_a.y * 720]], dtype=np.float32)
point_4d = cv2.triangulatePoints(self.P1, self.P2, pt1.T, pt2.T)
```

**After:** Proper undistortion and rectification
```python
# NEW (CORRECT):
def undistort_and_rectify(self, pts, camera='left'):
    pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    rectified = cv2.undistortPoints(pts, K, D, R=R, P=P)
    return rectified.reshape(-1, 2)
```

### 2. post_processor.py

**Before:** Normalized coordinates (0-1) passed directly to triangulation
```python
# OLD (WRONG):
pts_a.append([lms_dict_a[s_i][0], lms_dict_a[s_i][1]])  # 0-1 range!
```

**After:** Proper pixel scaling
```python
# NEW (CORRECT):
x_a = lms_dict_a[s_i][0] * self.image_width   # 0-1280
y_a = lms_dict_a[s_i][1] * self.image_height  # 0-720
pts_a.append([x_a, y_a])
```

### 3. floor_calibrator.py

**Before:** File named `floor_calibration.py` with incompatible API

**After:** Created `floor_calibrator.py` with expected method:
```python
def detect_floor_from_frames(self, frame_a, frame_b, triangulation_engine):
    # Returns (result_dict, message)
    # result_dict has 'normal' and 'point' keys
```

### 4. Blender Addons (Both)

**Before:** Wrong panel draw signature
```python
def draw(self, layout):  # WRONG!
    col = layout.column(align=True)
```

**After:** Correct signature
```python
def draw(self, context):  # CORRECT
    layout = self.layout
    col = layout.column(align=True)
```

**Before:** Invalid Quaternion initialization
```python
bone.rotation_quaternion = Quaternion((0, 0, 0.785))  # WRONG!
```

**After:** Proper Euler → Quaternion conversion
```python
euler = Euler((0, 0, 0.785), 'XYZ')
bone.rotation_quaternion = euler.to_quaternion()
```

---

## Installation Instructions

### 1. Replace Capture Engine Files

Copy these files to `MelodicCapStudio/MelodicCapAntiGrav/capture/engine/`:

- `triangulation_engine.py` (REPLACE)
- `post_processor.py` (REPLACE)
- `floor_calibrator.py` (NEW - rename from floor_calibration.py if exists)

### 2. Replace Blender Addons

Copy to `MelodicCapStudio/MelodicCapAntiGrav/blender_addon/`:

- `melodiccap_addon_v3.py`
- `antigrav_retargeter_v3.py`

In Blender:
1. Edit > Preferences > Add-ons
2. Remove old versions
3. Install new versions
4. Enable them

### 3. Verify Calibration

Your stereo calibration quality:
- Left camera RMS: 0.054 ✅ Excellent
- Right camera RMS: 0.399 ⚠️ Consider recalibrating
- Stereo RMS: 1.168 ⚠️ Acceptable but could improve

**Recommendation:** Recalibrate with the right camera (DroidCam) at higher quality settings if possible.

---

## Testing Checklist

- [ ] Run `run_studio_app.bat` - should launch without errors
- [ ] Start preview with both cameras
- [ ] Calibrate floor - should show "FLOOR: OK" message
- [ ] Record a short take (5-10 seconds)
- [ ] Check output JSON has `landmarks_3d` with realistic values
- [ ] Import into Blender with AntiGrav addon
- [ ] Verify character animation looks correct

---

## What Changed in Each File

### triangulation_engine.py
- Added `K1`, `K2`, `D1`, `D2`, `R1`, `R2` loading from calibration
- Added `undistort_and_rectify()` method
- Updated `triangulate_points()` to use undistortion
- Updated `triangulate_pose()` to use undistortion
- Added `reset_filters()` for Kalman reset between takes

### post_processor.py
- Added `self.image_width`, `self.image_height` from calibration
- Fixed `_triangulate_landmarks()` to scale normalized → pixels
- Fixed `_triangulate_hands()` to scale normalized → pixels
- Added visibility threshold check
- Added gap filling before Butterworth filter
- Added optional foot locking

### floor_calibrator.py (NEW FILE)
- ChArUco board detection
- Stereo triangulation of floor points
- SVD plane fitting
- Integration with TriangulationEngine

### melodiccap_addon_v3.py
- Fixed `draw()` method signature
- Fixed `Quaternion()` to use `Euler().to_quaternion()`
- Updated `bl_info` for Blender 4.4.0
- Added operator options `{'REGISTER', 'UNDO'}`

### antigrav_retargeter_v3.py
- Fixed `draw()` method signature
- Fixed `Quaternion()` usage
- Added `bone_rest_axes` dictionary for proper V2R
- Improved Vector handling (check isinstance)
- Updated `bl_info` for Blender 4.4.0
