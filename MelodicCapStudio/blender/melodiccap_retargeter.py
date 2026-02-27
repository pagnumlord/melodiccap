"""
MelodicCap Retargeter v1.4
==========================
Clean retargeter combining:
- v4's proven delta-from-reference IK approach (mathematically correct)
- AntiGrav V3's V2R (Vector-to-Rotation) FK method
- Pole targets for correct elbow/knee direction (from Keemap/AntiGrav approach)
- IK target rotation for wrist/foot orientation
- Ground clamping (feet can't go through floor)
- Smart foot pinning (reduces sliding when feet should be planted)
- 4-segment spine animation via virtual midpoints
- Outlier filtering (catches MediaPipe landmark spikes)

For Blender 4.4+ with JaxRigify armature.

KEY DESIGN DECISIONS:
- NO mirroring: Person's LEFT = Character's LEFT (both at +X in capture/Blender coords)
  Person faces camera (-Y), character's .L bones are at +X — same side.
- NO X-axis negation (v5/v12.1 proved this double-mirrors)
- Data is already in Blender coordinates from the capture script
- IK targets use delta-from-reference (includes hip movement naturally)
- Pole targets use 3-point projection (Keemap algorithm) with delta-from-reference
- FK rest axes from ACTUAL JaxRigify bone dump (A-pose: arms hang DOWN, not T-pose)
- IK rotation rest axes from actual bone directions (hand_ik=-Z, foot_ik=+Y)

v1.4 OUTLIER FILTERING:
- Velocity-based pre-filter on raw landmarks before animation.
  MediaPipe sometimes outputs garbage positions (landmarks jumping 30-80m in a
  single frame). The pre-filter scans all frames, tracks per-landmark velocity,
  and replaces outlier positions with the last known good position.
  Threshold: configurable max velocity (default 10 m/s) adapted to capture FPS.
  Fixes IK targets, pole targets, FK rotations, and spine all at once since
  they all derive from the same landmark data.

v1.3 CRITICAL FIXES (from Rigify property diagnostics):
- IK_parent set to 0 (root space) during import. Default IK_parent=1 makes IK
  targets follow torso via parent chain, but our delta already includes hip
  displacement → double root motion (arms fly off during walking).
  Root space (0) means IK targets are independent of torso movement.
- pole_vector enabled (True) during import. Default is False, which means the
  IK solver ignores pole target bone positions entirely.
- pole_parent set to 0 (root space) for same reason as IK_parent.
"""

bl_info = {
    "name": "MelodicCap Retargeter",
    "author": "Karsten / MelodicCap Studio",
    "version": (1, 4, 0),
    "blender": (4, 4, 0),
    "location": "View3D > Sidebar > MelodicCap",
    "description": "Import MelodicCap motion capture data to JaxRigify armature",
    "category": "Animation",
}

import bpy
import json
import os
import datetime
from pathlib import Path
from mathutils import Vector, Matrix, Quaternion
from bpy.props import StringProperty, FloatProperty, BoolProperty, IntProperty
from bpy_extras.io_utils import ImportHelper

# =============================================================================
# LANDMARK DEFINITIONS (MediaPipe 33 body landmarks)
# =============================================================================

LANDMARKS = {
    0: "nose", 11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow", 15: "left_wrist", 16: "right_wrist",
    23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
}

# =============================================================================
# BONE MAPPING
# Person's LEFT = Character's LEFT (NO mirroring)
# Both the performer and JaxRigify have LEFT at +X in the shared coordinate space.
# (Person faces camera at -Y; character's .L bones verified at +X)
# =============================================================================

# IK targets for hand/foot position
IK_TARGETS = {
    'hand_ik.L': 15,   # Person's left wrist -> Character's left hand
    'hand_ik.R': 16,   # Person's right wrist -> Character's right hand
    'foot_ik.L': 27,   # Person's left ankle -> Character's left foot
    'foot_ik.R': 28,   # Person's right ankle -> Character's right foot
}

# FK bone chains for limb rotation (V2R: start landmark -> end landmark)
FK_CHAINS = {
    # Person's LEFT -> Character's LEFT
    'upper_arm_fk.L': (11, 13),   # left shoulder -> left elbow
    'forearm_fk.L':   (13, 15),   # left elbow -> left wrist
    'thigh_fk.L':     (23, 25),   # left hip -> left knee
    'shin_fk.L':      (25, 27),   # left knee -> left ankle
    # Person's RIGHT -> Character's RIGHT
    'upper_arm_fk.R': (12, 14),   # right shoulder -> right elbow
    'forearm_fk.R':   (14, 16),   # right elbow -> right wrist
    'thigh_fk.R':     (24, 26),   # right hip -> right knee
    'shin_fk.R':      (26, 28),   # right knee -> right ankle
}

# Pole targets for IK elbow/knee direction (3-point: root, mid, end)
# Bone names verified from JaxRigify diagnostic dump
POLE_TARGETS = {
    'upper_arm_ik_target.L': (11, 13, 15),  # Person's L shoulder→elbow→wrist
    'upper_arm_ik_target.R': (12, 14, 16),  # Person's R shoulder→elbow→wrist
    'thigh_ik_target.L': (23, 25, 27),       # Person's L hip→knee→ankle
    'thigh_ik_target.R': (24, 26, 28),       # Person's R hip→knee→ankle
}

# IK target rotation mapping (for wrist/foot orientation)
# Rest axes from actual JaxRigify bone dump:
#   hand_ik.L dir=( 0.147,-0.048,-0.988)  hand_ik.R dir=(-0.077,-0.128,-0.989)
#   foot_ik.L dir=( 0.000, 1.000, 0.000)  foot_ik.R dir=( 0.000, 1.000, 0.000)
IK_ROTATION = {
    'hand_ik.L': (13, 15, Vector(( 0.147, -0.048, -0.988))),  # L forearm dir
    'hand_ik.R': (14, 16, Vector((-0.077, -0.128, -0.989))),  # R forearm dir
    'foot_ik.L': (25, 27, Vector((0, 1, 0))),                  # L shin dir → feet point +Y
    'foot_ik.R': (26, 28, Vector((0, 1, 0))),                  # R shin dir → feet point +Y
}

# Spine V2R using virtual midpoints (4-segment spine)
SPINE_CHAINS = {
    'spine_fk':      ('hip_mid', 'spine_low'),
    'spine_fk.001':  ('spine_low', 'spine_mid'),
    'spine_fk.002':  ('spine_mid', 'neck_mid'),
    'spine_fk.003':  ('neck_mid', 'shoulder_mid'),
}

# Rigify IK/FK switch bones
IK_FK_SWITCHES = {
    'upper_arm_parent.L': 'IK_FK',
    'upper_arm_parent.R': 'IK_FK',
    'thigh_parent.L': 'IK_FK',
    'thigh_parent.R': 'IK_FK',
}

# Rest axes for V2R FK rotations — from ACTUAL JaxRigify bone dump (A-pose)
# These are the bone rest directions in ARMATURE SPACE for each FK bone.
# Arms hang DOWN in A-pose (not sideways like T-pose).
BONE_REST_AXES = {
    # Arms hang down and slightly outward (A-pose, from bone dump)
    'upper_arm_fk.L': Vector(( 0.287,  0.070, -0.955)).normalized(),
    'forearm_fk.L':   Vector(( 0.468, -0.156, -0.870)).normalized(),
    'upper_arm_fk.R': Vector((-0.265,  0.071, -0.962)).normalized(),
    'forearm_fk.R':   Vector((-0.453, -0.179, -0.873)).normalized(),
    # Legs point down (nearly straight, from bone dump)
    'thigh_fk.L': Vector(( 0.070, -0.040, -0.997)).normalized(),
    'shin_fk.L':  Vector((-0.010,  0.066, -0.998)).normalized(),
    'thigh_fk.R': Vector((-0.055, -0.041, -0.998)).normalized(),
    'shin_fk.R':  Vector((-0.036,  0.066, -0.997)).normalized(),
    # Spine points up (from bone dump)
    'spine_fk':      Vector(( 0.000, -0.095,  0.995)).normalized(),
    'spine_fk.001':  Vector((-0.000, -0.010,  1.000)).normalized(),
    'spine_fk.002':  Vector((-0.000,  0.060,  0.998)).normalized(),
    'spine_fk.003':  Vector(( 0.000,  0.001,  1.000)).normalized(),
}

# =============================================================================
# LOGGING — writes to both Blender console AND a log file
# =============================================================================

_log_file = None
_log_path = None

def log_init(tag="import"):
    """Open a log file in the logs/ directory next to this addon."""
    global _log_file, _log_path
    log_close()  # Close any previous log

    # Find a writable logs directory
    # Try addon directory first, fall back to temp
    addon_dir = Path(__file__).parent.parent  # MelodicCapStudio/
    logs_dir = addon_dir / "logs"
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        logs_dir = Path(bpy.app.tempdir) / "melodiccap_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_path = logs_dir / f"melodiccap_{tag}_{ts}.log"
    _log_file = open(_log_path, 'w', encoding='utf-8')
    log(f"Log file: {_log_path}")

def log_close():
    """Close the log file."""
    global _log_file, _log_path
    if _log_file:
        _log_file.close()
        _log_file = None

def log(msg, level="INFO"):
    """Log to both Blender console and file."""
    line = f"[{level}] MelodicCap: {msg}"
    print(line)
    if _log_file:
        _log_file.write(line + "\n")
        _log_file.flush()  # Flush immediately so we never lose data

def log_get_path():
    """Return the current log file path."""
    return _log_path

# =============================================================================
# UTILITIES
# =============================================================================

def track_range(ranges, bone_name, value, value_type='loc'):
    """Track min/max range for a bone's animated values."""
    if bone_name not in ranges:
        ranges[bone_name] = {
            'min': [float('inf')] * 3,
            'max': [float('-inf')] * 3,
            'type': value_type,
        }
    r = ranges[bone_name]
    for i in range(3):
        v = value[i]
        if v < r['min'][i]:
            r['min'][i] = v
        if v > r['max'][i]:
            r['max'][i] = v

def get_lm(landmarks, idx):
    """Get landmark as Vector. Handles both string and int keys."""
    for key in [str(idx), idx]:
        if key in landmarks:
            p = landmarks[key]
            return Vector((p[0], p[1], p[2]))
    return None

def get_mid(landmarks, i1, i2):
    """Get midpoint of two landmarks."""
    p1, p2 = get_lm(landmarks, i1), get_lm(landmarks, i2)
    if p1 and p2:
        return (p1 + p2) / 2
    return None

def compute_pole_position(v_root, v_mid, v_end, offset=0.3):
    """Compute IK pole target position from a 3-joint chain.

    Uses the Keemap algorithm: project the mid-joint perpendicular to the
    root→end line and place the pole target along that perpendicular direction.
    Returns the pole position in mocap world space, or None if the limb is
    too straight to determine a bend direction.

    Args:
        v_root: Start of chain (shoulder/hip) as Vector
        v_mid: Mid-joint (elbow/knee) as Vector
        v_end: End of chain (wrist/ankle) as Vector
        offset: Distance to place pole target from mid-joint (meters)
    """
    line = v_end - v_root
    if line.length < 0.001:
        return None

    line_norm = line.normalized()
    # Project mid-joint onto the root→end line
    proj_length = (v_mid - v_root).dot(line_norm)
    proj_point = v_root + line_norm * proj_length

    # Perpendicular from projected point to actual mid-joint = pole direction
    pole_dir = v_mid - proj_point
    if pole_dir.length < 0.005:
        return None  # Limb too straight; pole direction undefined

    return v_mid + pole_dir.normalized() * offset


def compute_virtual_spine(landmarks):
    """Calculate virtual midpoints for 4-segment spine animation.

    Creates 5 virtual landmarks from hips (23,24) and shoulders (11,12):
      hip_mid -> spine_low -> spine_mid -> neck_mid -> shoulder_mid
    """
    hip_l = get_lm(landmarks, 23)
    hip_r = get_lm(landmarks, 24)
    sh_l = get_lm(landmarks, 11)
    sh_r = get_lm(landmarks, 12)

    if not all([hip_l, hip_r, sh_l, sh_r]):
        return None

    hip_mid = (hip_l + hip_r) / 2
    shoulder_mid = (sh_l + sh_r) / 2
    spine_mid = (hip_mid + shoulder_mid) / 2
    spine_low = (hip_mid + spine_mid) / 2
    neck_mid = (spine_mid + shoulder_mid) / 2

    return {
        'hip_mid': hip_mid,
        'spine_low': spine_low,
        'spine_mid': spine_mid,
        'neck_mid': neck_mid,
        'shoulder_mid': shoulder_mid,
    }


# =============================================================================
# IMPORTER
# =============================================================================

class MelodicCapImporter:

    def __init__(self, armature, take_data, settings):
        self.armature = armature
        self.take_data = take_data
        self.settings = settings

        # Reference frame data
        self.ref_hip = None
        self.ref_landmarks = None

        # Scaling
        self.scale = 1.0
        self.char_height = 1.87

        # IK target rest positions (world space)
        self.ik_rest_positions = {}

        # Reference pole positions (mocap world space, for delta computation)
        self.ref_pole_positions = {}

        # Stats
        self.stats = {'frames': 0, 'keys': 0, 'bones': set()}

        # Per-bone range tracking for diagnostics
        self.ranges = {}  # bone_name -> {'min': Vector, 'max': Vector, 'type': 'loc'|'rot'}

    def analyze(self):
        """Analyze character and mocap data, compute scale factor."""
        log("=" * 60)
        log("MELODICCAP RETARGETER v1.4 - ANALYSIS")
        log("=" * 60)

        # --- Armature scale check ---
        arm_scale = self.armature.scale
        log(f"  Armature scale: ({arm_scale.x:.3f}, {arm_scale.y:.3f}, {arm_scale.z:.3f})")
        if abs(arm_scale.x - 1.0) > 0.01 or abs(arm_scale.y - 1.0) > 0.01 or abs(arm_scale.z - 1.0) > 0.01:
            log("  WARNING: Armature scale is not 1.0! Consider applying scale (Ctrl+A).", "WARN")

        # --- Character height from bone extents ---
        bones = self.armature.data.bones
        world = self.armature.matrix_world

        min_z, max_z = float('inf'), float('-inf')
        for bone in bones:
            h = (world @ bone.head_local).z
            t = (world @ bone.tail_local).z
            min_z = min(min_z, h, t)
            max_z = max(max_z, h, t)

        self.char_height = max_z - min_z if max_z > min_z else 1.87
        log(f"  Character height: {self.char_height:.3f}m")

        # --- IK target rest positions ---
        log(f"\n  IK target rest positions (world space):")
        for ik_bone in IK_TARGETS.keys():
            if ik_bone in bones:
                bone = bones[ik_bone]
                head_world = world @ bone.head_local
                self.ik_rest_positions[ik_bone] = head_world.copy()
                log(f"    {ik_bone}: ({head_world.x:.3f}, {head_world.y:.3f}, {head_world.z:.3f})")

        # --- IK/FK switch status ---
        log(f"\n  IK/FK switch status (BEFORE import configuration):")
        pose_bones = self.armature.pose.bones
        for switch_bone, prop_name in IK_FK_SWITCHES.items():
            if switch_bone in pose_bones:
                pb = pose_bones[switch_bone]
                props = []
                if prop_name in pb:
                    val = pb[prop_name]
                    mode = "FK" if val > 0.5 else "IK"
                    props.append(f"IK_FK={val:.1f}({mode})")
                if 'IK_parent' in pb:
                    props.append(f"IK_parent={pb['IK_parent']}")
                if 'pole_vector' in pb:
                    props.append(f"pole_vector={pb['pole_vector']}")
                if 'pole_parent' in pb:
                    props.append(f"pole_parent={pb['pole_parent']}")
                log(f"    {switch_bone}: {', '.join(props)}")

        # --- Mocap data ---
        frames = self.take_data.get('frames', [])
        if not frames:
            log("ERROR: No frames in take data!", "ERROR")
            return False

        log(f"\n  Mocap data: {len(frames)} frames, {self.take_data.get('duration_seconds', 0):.1f}s")

        # --- Calibration info ---
        calib = self.take_data.get('calibration', {})
        log(f"  Calibration: stereo RMS={calib.get('rms_stereo', 'N/A')}, "
            f"baseline={calib.get('baseline', 'N/A')}m, "
            f"floor_offset={calib.get('floor_offset', 0):.3f}m")

        # --- Reference frame (frame 0) ---
        self.ref_landmarks = frames[0].get('landmarks', {})
        self.ref_hip = get_mid(self.ref_landmarks, 23, 24)

        if not self.ref_hip:
            log("ERROR: No hip center in frame 0!", "ERROR")
            return False

        log(f"\n  Reference frame hip: ({self.ref_hip.x:.3f}, {self.ref_hip.y:.3f}, {self.ref_hip.z:.3f})")

        # --- Person height: nose to ankle midpoint + 0.15m ---
        nose = get_lm(self.ref_landmarks, 0)
        ankle_l = get_lm(self.ref_landmarks, 27)
        ankle_r = get_lm(self.ref_landmarks, 28)

        if nose and ankle_l and ankle_r:
            ankle_mid = (ankle_l + ankle_r) / 2
            mocap_height = (nose.z - ankle_mid.z) + 0.15
            self.scale = self.char_height / mocap_height
            log(f"  Person height: {mocap_height:.3f}m")
            log(f"  Scale factor: {self.scale:.4f}")
        else:
            log("  WARNING: Could not measure person height, using scale=1.0", "WARN")

        # --- Reference landmark positions ---
        log(f"\n  Reference landmarks (relative to hip):")
        for idx in [11, 12, 15, 16, 23, 24, 27, 28]:
            pos = get_lm(self.ref_landmarks, idx)
            if pos:
                rel = pos - self.ref_hip
                name = LANDMARKS.get(idx, str(idx))
                log(f"    [{idx:2d}] {name:15s}: ({rel.x:7.3f}, {rel.y:7.3f}, {rel.z:7.3f})")

        # --- Reference pole target positions ---
        log(f"\n  Reference pole positions:")
        for pole_bone, (lm_root, lm_mid, lm_end) in POLE_TARGETS.items():
            v_root = get_lm(self.ref_landmarks, lm_root)
            v_mid = get_lm(self.ref_landmarks, lm_mid)
            v_end = get_lm(self.ref_landmarks, lm_end)
            if v_root and v_mid and v_end:
                pole_pos = compute_pole_position(v_root, v_mid, v_end)
                if pole_pos:
                    self.ref_pole_positions[pole_bone] = pole_pos
                    log(f"    {pole_bone}: ({pole_pos.x:.3f}, {pole_pos.y:.3f}, {pole_pos.z:.3f})")
                else:
                    log(f"    {pole_bone}: limb too straight, skipping")

        return True

    def prefilter_landmarks(self):
        """Pre-filter landmark data to remove outlier spikes.

        Uses velocity-based filtering: if a landmark moves faster than
        max_velocity m/s between consecutive frames, the frame is replaced
        with the last known good position. This catches MediaPipe tracking
        glitches where landmarks jump to absurd positions (e.g. 75m in one frame).

        Modifies the frame landmark data in-place so all downstream consumers
        (IK targets, pole targets, FK rotations, spine) benefit automatically.
        """
        frames = self.take_data.get('frames', [])
        if len(frames) < 2:
            return

        max_velocity = self.settings.get('outlier_velocity', 10.0)

        # Compute frame duration from capture metadata
        duration = self.take_data.get('duration_seconds', len(frames) / 10.0)
        fps = len(frames) / max(duration, 0.1)
        max_jump = max_velocity / fps  # meters per frame

        log(f"\n  Outlier filter: max_velocity={max_velocity:.0f} m/s, "
            f"capture fps={fps:.1f}, max_jump={max_jump:.3f}m/frame")

        # Collect all landmark indices used by any mapping
        used_landmarks = set()
        for lm_idx in IK_TARGETS.values():
            used_landmarks.add(lm_idx)
        for i1, i2 in FK_CHAINS.values():
            used_landmarks.add(i1)
            used_landmarks.add(i2)
        for lm_root, lm_mid, lm_end in POLE_TARGETS.values():
            used_landmarks.add(lm_root)
            used_landmarks.add(lm_mid)
            used_landmarks.add(lm_end)
        for lm_start, lm_end, _ in IK_ROTATION.values():
            used_landmarks.add(lm_start)
            used_landmarks.add(lm_end)
        used_landmarks.update([0, 11, 12, 23, 24])  # spine/height landmarks

        # Track state per landmark
        prev_good = {}       # lm_idx -> Vector (last accepted position)
        filter_counts = {}   # lm_idx -> total filtered frames
        consecutive = {}     # lm_idx -> current consecutive hold count
        max_consecutive = {} # lm_idx -> worst consecutive hold run

        for fidx, fdata in enumerate(frames):
            lms = fdata.get('landmarks', {})

            for lm_idx in used_landmarks:
                pos = get_lm(lms, lm_idx)
                if pos is None:
                    continue

                # Determine the actual key type used in this dict
                key = str(lm_idx) if str(lm_idx) in lms else lm_idx

                if lm_idx in prev_good:
                    displacement = (pos - prev_good[lm_idx]).length
                    if displacement > max_jump:
                        # Outlier — replace with last good position
                        good = prev_good[lm_idx]
                        lms[key] = [good.x, good.y, good.z]
                        filter_counts[lm_idx] = filter_counts.get(lm_idx, 0) + 1
                        consecutive[lm_idx] = consecutive.get(lm_idx, 0) + 1
                        mc = max_consecutive.get(lm_idx, 0)
                        if consecutive[lm_idx] > mc:
                            max_consecutive[lm_idx] = consecutive[lm_idx]
                    else:
                        # Good frame — update baseline
                        prev_good[lm_idx] = pos
                        consecutive[lm_idx] = 0
                else:
                    # First frame — set baseline
                    prev_good[lm_idx] = pos
                    consecutive[lm_idx] = 0

        # Log results
        if filter_counts:
            total = sum(filter_counts.values())
            log(f"  Outlier filter: {total} landmark values replaced "
                f"across {len(filter_counts)} landmarks:")
            for lm_idx in sorted(filter_counts.keys()):
                name = LANDMARKS.get(lm_idx, f"landmark_{lm_idx}")
                count = filter_counts[lm_idx]
                pct = count / len(frames) * 100
                mc = max_consecutive.get(lm_idx, 0)
                log(f"    [{lm_idx:2d}] {name:15s}: {count:4d} frames ({pct:5.1f}%), "
                    f"max consecutive hold={mc}")
                if mc > fps * 2:  # More than 2 seconds of sustained holds
                    log(f"         WARNING: {mc} consecutive holds ({mc/fps:.1f}s) — "
                        f"possible sustained tracking loss", "WARN")
        else:
            log(f"  Outlier filter: no outliers detected "
                f"(all motion within {max_jump:.3f}m/frame)")

    def set_ik_fk_mode(self, use_ik=True):
        """Set IK/FK switches on the rig. 0.0=IK, 1.0=FK."""
        target_value = 0.0 if use_ik else 1.0
        pose_bones = self.armature.pose.bones

        for switch_bone, prop_name in IK_FK_SWITCHES.items():
            if switch_bone in pose_bones:
                pb = pose_bones[switch_bone]
                if prop_name in pb:
                    pb[prop_name] = target_value

        log(f"  Set {'IK' if use_ik else 'FK'} mode on all limbs")

    def apply_animation(self):
        """Apply animation using hybrid approach:
        - Torso: delta from reference hip (root motion)
        - IK targets: delta from reference positions (hands/feet)
        - IK rotation: wrist/foot orientation from forearm/shin direction
        - Pole targets: 3-point projection for elbow/knee bend direction
        - FK rotations: V2R with per-bone rest axes (visible in FK mode only)
        - Spine: V2R with virtual midpoints (body twist/bend)
        """
        log("\n" + "=" * 60)
        log("APPLYING ANIMATION")
        log("=" * 60)

        frames = self.take_data.get('frames', [])
        pose_bones = self.armature.pose.bones
        start = self.settings.get('start_frame', 1)
        pin_threshold = self.settings.get('pin_threshold', 0.02)
        animate_fk = self.settings.get('animate_fk', True)
        animate_spine = self.settings.get('animate_spine', True)
        ground_clamp = self.settings.get('ground_clamp', True)
        animate_poles = self.settings.get('animate_poles', True)
        animate_ik_rot = self.settings.get('animate_ik_rot', True)

        # Pre-filter outlier landmarks (modifies frame data in-place)
        if self.settings.get('filter_outliers', True):
            self.prefilter_landmarks()

        # Force IK mode
        self.set_ik_fk_mode(use_ik=True)

        # Configure Rigify properties for correct mocap retargeting
        log("  Configuring Rigify properties:")
        for switch_bone in IK_FK_SWITCHES:
            if switch_bone in pose_bones:
                pb = pose_bones[switch_bone]
                # IK_parent=0 (root space): CRITICAL for delta-from-reference.
                # Default IK_parent=1 makes IK targets follow torso via parent
                # chain, but our delta already includes hip displacement.
                # That causes double root motion (arms fly off during walking).
                # Root space (0) makes IK targets independent of torso.
                if 'IK_parent' in pb:
                    old = pb['IK_parent']
                    pb['IK_parent'] = 0
                    if old != 0:
                        log(f"    {switch_bone}: IK_parent {old}->0 (root space)")
                # pole_vector=True: enables pole target bones so our elbow/knee
                # direction animation actually affects the IK solver.
                # Default False means pole targets are ignored entirely.
                if 'pole_vector' in pb:
                    old = pb['pole_vector']
                    pb['pole_vector'] = True
                    if not old:
                        log(f"    {switch_bone}: pole_vector False->True")
                # pole_parent=0 (root space): same reasoning as IK_parent.
                # Prevents double-motion on pole target positions.
                if 'pole_parent' in pb:
                    old = pb['pole_parent']
                    pb['pole_parent'] = 0
                    if old != 0:
                        log(f"    {switch_bone}: pole_parent {old}->0 (root space)")

        # Set quaternion rotation mode on all FK bones
        for pb in pose_bones:
            pb.rotation_mode = 'QUATERNION'

        # Armature transforms
        world = self.armature.matrix_world
        world_inv = world.inverted()

        # Check available bones
        avail_ik = {}
        for bone, lm_idx in IK_TARGETS.items():
            if bone in pose_bones:
                avail_ik[bone] = lm_idx

        avail_fk = {}
        for bone, (i1, i2) in FK_CHAINS.items():
            if bone in pose_bones:
                avail_fk[bone] = (i1, i2)

        avail_spine = {}
        if animate_spine:
            for bone, (s, e) in SPINE_CHAINS.items():
                if bone in pose_bones:
                    avail_spine[bone] = (s, e)

        avail_poles = {}
        if animate_poles:
            for bone, (lm_root, lm_mid, lm_end) in POLE_TARGETS.items():
                if bone in pose_bones and bone in self.ref_pole_positions:
                    avail_poles[bone] = (lm_root, lm_mid, lm_end)

        avail_ik_rot = {}
        if animate_ik_rot:
            for bone, (lm_start, lm_end, rest_ax) in IK_ROTATION.items():
                if bone in pose_bones:
                    avail_ik_rot[bone] = (lm_start, lm_end, rest_ax)

        has_torso = 'torso' in pose_bones

        log(f"  Available: {len(avail_ik)} IK targets, {len(avail_poles)} pole targets, "
            f"{len(avail_fk)} FK chains, {len(avail_spine)} spine segments, "
            f"{len(avail_ik_rot)} IK rotations, torso={'yes' if has_torso else 'no'}")

        # Smart pinning state (for feet)
        prev_deltas = {}

        # How many frames get full detailed logging (first N)
        DETAIL_FRAMES = 5

        # Process each frame
        log(f"  Processing {len(frames)} frames (detailed log for first {DETAIL_FRAMES})...")

        for fidx, fdata in enumerate(frames):
            bf = start + fidx
            bpy.context.scene.frame_set(bf)
            detail = fidx < DETAIL_FRAMES  # Verbose logging for early frames

            lms = fdata.get('landmarks', {})
            hip = get_mid(lms, 23, 24)

            if not hip:
                if detail:
                    log(f"    Frame {fidx}: SKIPPED (no hip landmarks)")
                continue

            if detail:
                log(f"\n    --- Frame {fidx} (Blender frame {bf}) ---")
                log(f"    Hip center: ({hip.x:.4f}, {hip.y:.4f}, {hip.z:.4f})")
                log(f"    Hip delta from ref: ({hip.x - self.ref_hip.x:.4f}, "
                    f"{hip.y - self.ref_hip.y:.4f}, {hip.z - self.ref_hip.z:.4f})")

            # =================================================================
            # ROOT MOTION (TORSO)
            # =================================================================
            if has_torso:
                delta = (hip - self.ref_hip) * self.scale
                local_delta = world_inv.to_3x3() @ delta

                torso = pose_bones['torso']
                torso.location = local_delta
                torso.keyframe_insert(data_path="location", frame=bf)

                track_range(self.ranges, 'torso', local_delta, 'loc')
                self.stats['keys'] += 1
                self.stats['bones'].add('torso')

                if detail:
                    log(f"    TORSO loc: ({local_delta.x:.4f}, {local_delta.y:.4f}, {local_delta.z:.4f})")

            # =================================================================
            # IK TARGETS (HANDS AND FEET)
            # =================================================================
            for ik_bone, lm_idx in avail_ik.items():
                pos = get_lm(lms, lm_idx)
                if not pos:
                    continue

                ref_pos = get_lm(self.ref_landmarks, lm_idx)
                if not ref_pos:
                    continue

                mocap_delta = (pos - ref_pos) * self.scale

                # Ground clamp
                clamped = False
                if ground_clamp and 'foot' in ik_bone:
                    rest_pos = self.ik_rest_positions.get(ik_bone)
                    if rest_pos:
                        world_z = rest_pos.z + mocap_delta.z
                        if world_z < 0:
                            mocap_delta.z = -rest_pos.z
                            clamped = True

                local_delta = world_inv.to_3x3() @ mocap_delta

                # Smart pinning
                pinned = False
                if 'foot' in ik_bone and pin_threshold > 0:
                    prev = prev_deltas.get(ik_bone)
                    if prev is not None:
                        if (local_delta - prev).length < pin_threshold:
                            local_delta = prev.copy()
                            pinned = True
                    prev_deltas[ik_bone] = local_delta.copy()

                pb = pose_bones[ik_bone]
                pb.location = local_delta
                pb.keyframe_insert(data_path="location", frame=bf)

                track_range(self.ranges, ik_bone, local_delta, 'loc')
                self.stats['keys'] += 1
                self.stats['bones'].add(ik_bone)

                if detail:
                    flags = ""
                    if clamped:
                        flags += " [CLAMPED]"
                    if pinned:
                        flags += " [PINNED]"
                    lm_name = LANDMARKS.get(lm_idx, str(lm_idx))
                    log(f"    {ik_bone} loc: ({local_delta.x:.4f}, {local_delta.y:.4f}, {local_delta.z:.4f})"
                        f"  lm{lm_idx}({lm_name}){flags}")

            # =================================================================
            # IK TARGET ROTATION
            # =================================================================
            for ik_bone, (lm_start, lm_end, rest_ax) in avail_ik_rot.items():
                p1 = get_lm(lms, lm_start)
                p2 = get_lm(lms, lm_end)
                if not p1 or not p2:
                    continue

                limb_dir = (p2 - p1).normalized()
                dir_local = world_inv.to_quaternion() @ limb_dir
                quat = rest_ax.rotation_difference(dir_local)

                pb = pose_bones[ik_bone]
                pb.rotation_mode = 'QUATERNION'
                pb.rotation_quaternion = quat
                pb.keyframe_insert(data_path="rotation_quaternion", frame=bf)

                track_range(self.ranges, ik_bone + '_rot', [quat.w, quat.x, quat.y], 'rot')
                self.stats['keys'] += 1

                if detail:
                    log(f"    {ik_bone} rot: w={quat.w:.3f} ({quat.x:.3f}, {quat.y:.3f}, {quat.z:.3f})"
                        f"  dir_local=({dir_local.x:.3f}, {dir_local.y:.3f}, {dir_local.z:.3f})")

            # =================================================================
            # POLE TARGETS
            # =================================================================
            for pole_bone, (lm_root, lm_mid, lm_end) in avail_poles.items():
                v_root = get_lm(lms, lm_root)
                v_mid = get_lm(lms, lm_mid)
                v_end = get_lm(lms, lm_end)
                if not all([v_root, v_mid, v_end]):
                    continue

                pole_pos = compute_pole_position(v_root, v_mid, v_end)
                if not pole_pos:
                    if detail:
                        log(f"    {pole_bone}: limb too straight, skipped")
                    continue

                ref_pole = self.ref_pole_positions.get(pole_bone)
                if not ref_pole:
                    continue

                pole_delta = (pole_pos - ref_pole) * self.scale
                local_delta = world_inv.to_3x3() @ pole_delta

                pb = pose_bones[pole_bone]
                pb.location = local_delta
                pb.keyframe_insert(data_path="location", frame=bf)

                track_range(self.ranges, pole_bone, local_delta, 'loc')
                self.stats['keys'] += 1
                self.stats['bones'].add(pole_bone)

                if detail:
                    log(f"    {pole_bone} loc: ({local_delta.x:.4f}, {local_delta.y:.4f}, {local_delta.z:.4f})")

            # =================================================================
            # FK ROTATIONS (V2R METHOD)
            # =================================================================
            if animate_fk:
                for fk_bone, (i1, i2) in avail_fk.items():
                    p1, p2 = get_lm(lms, i1), get_lm(lms, i2)
                    if not p1 or not p2:
                        continue

                    target_dir = (p2 - p1).normalized()
                    target_dir_local = world_inv.to_quaternion() @ target_dir

                    rest_axis = BONE_REST_AXES.get(fk_bone)
                    if not rest_axis:
                        continue

                    quat = rest_axis.rotation_difference(target_dir_local)

                    pb = pose_bones[fk_bone]
                    pb.rotation_quaternion = quat
                    pb.keyframe_insert(data_path="rotation_quaternion", frame=bf)

                    track_range(self.ranges, fk_bone, [quat.w, quat.x, quat.y], 'rot')
                    self.stats['keys'] += 1
                    self.stats['bones'].add(fk_bone)

                    if detail:
                        log(f"    {fk_bone} rot: w={quat.w:.3f} ({quat.x:.3f}, {quat.y:.3f}, {quat.z:.3f})"
                            f"  dir=({target_dir_local.x:.3f}, {target_dir_local.y:.3f}, {target_dir_local.z:.3f})")

            # =================================================================
            # SPINE ANIMATION (V2R with virtual midpoints)
            # =================================================================
            if animate_spine and avail_spine:
                spine_pts = compute_virtual_spine(lms)
                if spine_pts:
                    for bone_name, (start_key, end_key) in avail_spine.items():
                        s_pos = spine_pts.get(start_key)
                        e_pos = spine_pts.get(end_key)
                        if not s_pos or not e_pos:
                            continue

                        target_dir = (e_pos - s_pos).normalized()
                        target_dir_local = world_inv.to_quaternion() @ target_dir

                        rest_axis = BONE_REST_AXES.get(bone_name, Vector((0, 0, 1)))
                        quat = rest_axis.rotation_difference(target_dir_local)

                        pb = pose_bones[bone_name]
                        pb.rotation_quaternion = quat
                        pb.keyframe_insert(data_path="rotation_quaternion", frame=bf)

                        track_range(self.ranges, bone_name, [quat.w, quat.x, quat.y], 'rot')
                        self.stats['keys'] += 1
                        self.stats['bones'].add(bone_name)

                        if detail:
                            log(f"    {bone_name} rot: w={quat.w:.3f} ({quat.x:.3f}, {quat.y:.3f}, {quat.z:.3f})"
                                f"  dir=({target_dir_local.x:.3f}, {target_dir_local.y:.3f}, {target_dir_local.z:.3f})")

            self.stats['frames'] += 1

            if fidx % 50 == 0 and fidx >= DETAIL_FRAMES:
                log(f"    Frame {fidx}/{len(frames)}")

        return True

    def summary(self):
        """Print import summary with range diagnostics."""
        log("\n" + "=" * 60)
        log("IMPORT SUMMARY")
        log("=" * 60)
        log(f"  Frames processed: {self.stats['frames']}")
        log(f"  Keyframes created: {self.stats['keys']}")
        log(f"  Bones animated ({len(self.stats['bones'])}):")

        ik_bones = sorted(b for b in self.stats['bones'] if '_ik' in b)
        fk_bones = sorted(b for b in self.stats['bones'] if '_fk' in b)
        spine_bones = sorted(b for b in self.stats['bones'] if 'spine' in b)
        other_bones = sorted(b for b in self.stats['bones'] if '_ik' not in b and '_fk' not in b and 'spine' not in b)

        if other_bones:
            log(f"    Root: {other_bones}")
        if ik_bones:
            log(f"    IK: {ik_bones}")
        if fk_bones:
            log(f"    FK: {fk_bones}")
        if spine_bones:
            log(f"    Spine: {spine_bones}")

        # === RANGE DIAGNOSTICS ===
        log("\n" + "=" * 60)
        log("RANGE DIAGNOSTICS (min → max per axis)")
        log("=" * 60)
        log("  Location ranges (meters from rest position):")
        for bone_name in sorted(self.ranges.keys()):
            r = self.ranges[bone_name]
            if r['type'] != 'loc':
                continue
            mn, mx = r['min'], r['max']
            log(f"    {bone_name:35s}  X:[{mn[0]:+.3f} → {mx[0]:+.3f}]  "
                f"Y:[{mn[1]:+.3f} → {mx[1]:+.3f}]  Z:[{mn[2]:+.3f} → {mx[2]:+.3f}]")
            # Flag suspicious ranges
            for i, axis in enumerate(['X', 'Y', 'Z']):
                span = mx[i] - mn[i]
                if span > 2.0:
                    log(f"      WARNING: {axis} span = {span:.3f}m (>2m — possible outlier?)", "WARN")

        log("\n  Rotation ranges (quaternion w,x,y components):")
        for bone_name in sorted(self.ranges.keys()):
            r = self.ranges[bone_name]
            if r['type'] != 'rot':
                continue
            mn, mx = r['min'], r['max']
            log(f"    {bone_name:35s}  w:[{mn[0]:+.3f}→{mx[0]:+.3f}]  "
                f"x:[{mn[1]:+.3f}→{mx[1]:+.3f}]  y:[{mn[2]:+.3f}→{mx[2]:+.3f}]")


# =============================================================================
# OPERATORS
# =============================================================================

class MELODICCAP_OT_import(bpy.types.Operator, ImportHelper):
    """Import MelodicCap take JSON and apply to selected armature"""
    bl_idname = "melodiccap.import_take"
    bl_label = "Import Take"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})

    def execute(self, context):
        arm = context.active_object
        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature first!")
            return {'CANCELLED'}

        # Initialize file logging
        log_init("import")

        log(f"\n  File: {self.filepath}")
        log(f"  Armature: {arm.name}")

        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load JSON: {e}")
            log_close()
            return {'CANCELLED'}

        settings = {
            'start_frame': context.scene.melodiccap_start_frame,
            'animate_fk': context.scene.melodiccap_animate_fk,
            'animate_spine': context.scene.melodiccap_animate_spine,
            'animate_poles': context.scene.melodiccap_animate_poles,
            'animate_ik_rot': context.scene.melodiccap_animate_ik_rot,
            'ground_clamp': context.scene.melodiccap_ground_clamp,
            'pin_threshold': context.scene.melodiccap_pin_threshold,
            'filter_outliers': context.scene.melodiccap_filter_outliers,
            'outlier_velocity': context.scene.melodiccap_outlier_velocity,
        }

        log(f"\n  Settings:")
        for k, v in settings.items():
            log(f"    {k}: {v}")

        imp = MelodicCapImporter(arm, data, settings)

        if not imp.analyze():
            self.report({'ERROR'}, "Analysis failed - check console")
            log_close()
            return {'CANCELLED'}

        bpy.ops.object.mode_set(mode='POSE')
        imp.apply_animation()
        imp.summary()

        log_path = log_get_path()
        log_close()

        msg = f"Imported {imp.stats['frames']} frames, {len(imp.stats['bones'])} bones"
        if log_path:
            msg += f" | Log: {log_path}"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class MELODICCAP_OT_clear(bpy.types.Operator):
    """Clear all animation data and reset pose"""
    bl_idname = "melodiccap.clear"
    bl_label = "Clear Animation"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm = context.active_object
        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature!")
            return {'CANCELLED'}

        if arm.animation_data:
            arm.animation_data_clear()

        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.pose.transforms_clear()

        log("Cleared animation and reset pose")
        self.report({'INFO'}, "Animation cleared")
        return {'FINISHED'}


class MELODICCAP_OT_set_ik_mode(bpy.types.Operator):
    """Set rig to IK mode (recommended for mocap)"""
    bl_idname = "melodiccap.set_ik_mode"
    bl_label = "Set IK Mode"

    def execute(self, context):
        arm = context.active_object
        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature!")
            return {'CANCELLED'}

        pose_bones = arm.pose.bones
        for switch_bone, prop_name in IK_FK_SWITCHES.items():
            if switch_bone in pose_bones:
                pb = pose_bones[switch_bone]
                if prop_name in pb:
                    pb[prop_name] = 0.0

        self.report({'INFO'}, "Set to IK mode (all limbs)")
        return {'FINISHED'}


class MELODICCAP_OT_set_fk_mode(bpy.types.Operator):
    """Set rig to FK mode"""
    bl_idname = "melodiccap.set_fk_mode"
    bl_label = "Set FK Mode"

    def execute(self, context):
        arm = context.active_object
        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature!")
            return {'CANCELLED'}

        pose_bones = arm.pose.bones
        for switch_bone, prop_name in IK_FK_SWITCHES.items():
            if switch_bone in pose_bones:
                pb = pose_bones[switch_bone]
                if prop_name in pb:
                    pb[prop_name] = 1.0

        self.report({'INFO'}, "Set to FK mode (all limbs)")
        return {'FINISHED'}


class MELODICCAP_OT_diagnostic(bpy.types.Operator):
    """Dump complete rig diagnostic to log file"""
    bl_idname = "melodiccap.diagnostic"
    bl_label = "Diagnostic Dump"

    def execute(self, context):
        arm = context.active_object
        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature!")
            return {'CANCELLED'}

        log_init("diagnostic")
        log("=" * 60)
        log("RIG DIAGNOSTIC DUMP")
        log("=" * 60)

        # Armature info
        log(f"  Armature: {arm.name}")
        log(f"  Location: ({arm.location.x:.4f}, {arm.location.y:.4f}, {arm.location.z:.4f})")
        log(f"  Scale: ({arm.scale.x:.4f}, {arm.scale.y:.4f}, {arm.scale.z:.4f})")
        log(f"  Rotation: ({arm.rotation_euler.x:.4f}, {arm.rotation_euler.y:.4f}, {arm.rotation_euler.z:.4f})")

        world = arm.matrix_world
        log(f"\n  World matrix:")
        for r in range(4):
            log(f"    [{world[r][0]:+.4f}  {world[r][1]:+.4f}  {world[r][2]:+.4f}  {world[r][3]:+.4f}]")

        # IK/FK switch state
        log(f"\n  IK/FK Switch Properties:")
        pose_bones = arm.pose.bones
        for switch_bone, prop_name in IK_FK_SWITCHES.items():
            if switch_bone in pose_bones:
                pb = pose_bones[switch_bone]
                props = {k: pb[k] for k in pb.keys() if not k.startswith('_')}
                log(f"    {switch_bone}: {props}")

        # All bones we care about — current pose state
        log(f"\n  Bone Pose State (current frame {context.scene.frame_current}):")
        all_bones = (
            list(IK_TARGETS.keys()) +
            list(POLE_TARGETS.keys()) +
            list(FK_CHAINS.keys()) +
            list(SPINE_CHAINS.keys()) +
            ['torso']
        )
        for bone_name in sorted(set(all_bones)):
            if bone_name not in pose_bones:
                log(f"    {bone_name:35s}  MISSING")
                continue
            pb = pose_bones[bone_name]
            loc = pb.location
            rot = pb.rotation_quaternion
            parent = pb.parent.name if pb.parent else "None"
            log(f"    {bone_name:35s}  loc=({loc.x:+.4f}, {loc.y:+.4f}, {loc.z:+.4f})  "
                f"rot=w{rot.w:+.3f}({rot.x:+.3f},{rot.y:+.3f},{rot.z:+.3f})  parent={parent}")

        # Bone rest positions
        log(f"\n  Bone Rest Positions (world space):")
        bones = arm.data.bones
        for bone_name in sorted(set(all_bones)):
            if bone_name not in bones:
                continue
            bone = bones[bone_name]
            head_world = world @ bone.head_local
            tail_world = world @ bone.tail_local
            direction = (tail_world - head_world).normalized()
            length = (tail_world - head_world).length
            log(f"    {bone_name:35s}  head=({head_world.x:+.4f},{head_world.y:+.4f},{head_world.z:+.4f})  "
                f"dir=({direction.x:+.3f},{direction.y:+.3f},{direction.z:+.3f})  len={length:.4f}")

        # Animation data
        log(f"\n  Animation Data:")
        if arm.animation_data and arm.animation_data.action:
            action = arm.animation_data.action
            log(f"    Action: {action.name}")
            log(f"    Frame range: {action.frame_range[0]:.0f} - {action.frame_range[1]:.0f}")
            log(f"    FCurves: {len(action.fcurves)}")
            # List unique bone data paths
            bone_paths = set()
            for fc in action.fcurves:
                parts = fc.data_path.split('"')
                if len(parts) >= 2:
                    bone_paths.add(parts[1])
            log(f"    Animated bones ({len(bone_paths)}): {sorted(bone_paths)}")
        else:
            log(f"    No animation data")

        path = log_get_path()
        log_close()

        self.report({'INFO'}, f"Diagnostic saved: {path}")
        return {'FINISHED'}


# =============================================================================
# PANEL
# =============================================================================

class MELODICCAP_PT_panel(bpy.types.Panel):
    bl_label = "MelodicCap Retargeter"
    bl_idname = "MELODICCAP_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'MelodicCap'

    def draw(self, context):
        layout = self.layout

        # Target armature
        box = layout.box()
        box.label(text="Target:", icon='ARMATURE_DATA')
        if context.active_object and context.active_object.type == 'ARMATURE':
            box.label(text=f"  {context.active_object.name}")
            scale = context.active_object.scale
            if abs(scale.x - 1.0) > 0.01 or abs(scale.y - 1.0) > 0.01 or abs(scale.z - 1.0) > 0.01:
                box.label(text="  Scale not 1.0!", icon='ERROR')
        else:
            box.label(text="  (Select armature)")

        # Settings
        box = layout.box()
        box.label(text="Settings:", icon='SETTINGS')
        box.prop(context.scene, "melodiccap_start_frame")
        box.prop(context.scene, "melodiccap_pin_threshold")
        box.separator()
        box.prop(context.scene, "melodiccap_filter_outliers")
        if context.scene.melodiccap_filter_outliers:
            box.prop(context.scene, "melodiccap_outlier_velocity")
        box.separator()
        box.prop(context.scene, "melodiccap_animate_poles")
        box.prop(context.scene, "melodiccap_animate_ik_rot")
        box.prop(context.scene, "melodiccap_animate_fk")
        box.prop(context.scene, "melodiccap_animate_spine")
        box.prop(context.scene, "melodiccap_ground_clamp")

        # Actions
        box = layout.box()
        box.label(text="Actions:", icon='ACTION')
        box.operator("melodiccap.import_take", icon='IMPORT')

        row = box.row(align=True)
        row.operator("melodiccap.set_ik_mode", icon='CON_KINEMATIC')
        row.operator("melodiccap.set_fk_mode", icon='BONE_DATA')

        box.operator("melodiccap.clear", icon='X')

        box.separator()
        box.operator("melodiccap.diagnostic", icon='FILE_TEXT')


# =============================================================================
# REGISTRATION
# =============================================================================

classes = [
    MELODICCAP_OT_import,
    MELODICCAP_OT_clear,
    MELODICCAP_OT_set_ik_mode,
    MELODICCAP_OT_set_fk_mode,
    MELODICCAP_OT_diagnostic,
    MELODICCAP_PT_panel,
]

def register():
    for c in classes:
        bpy.utils.register_class(c)

    bpy.types.Scene.melodiccap_start_frame = IntProperty(
        name="Start Frame",
        default=1,
        min=1,
        description="Frame to start animation"
    )
    bpy.types.Scene.melodiccap_animate_poles = BoolProperty(
        name="Pole Targets",
        default=True,
        description="Animate elbow/knee pole targets for correct bend direction (CRITICAL)"
    )
    bpy.types.Scene.melodiccap_animate_ik_rot = BoolProperty(
        name="IK Rotation",
        default=True,
        description="Animate wrist/foot orientation based on forearm/shin direction"
    )
    bpy.types.Scene.melodiccap_animate_fk = BoolProperty(
        name="FK Rotations",
        default=True,
        description="Animate FK bone rotations (invisible in IK mode; available if you switch to FK)"
    )
    bpy.types.Scene.melodiccap_animate_spine = BoolProperty(
        name="Spine Animation",
        default=True,
        description="Animate spine using virtual midpoints (4-segment V2R)"
    )
    bpy.types.Scene.melodiccap_filter_outliers = BoolProperty(
        name="Filter Outliers",
        default=True,
        description="Remove landmark spikes caused by MediaPipe tracking glitches"
    )
    bpy.types.Scene.melodiccap_outlier_velocity = FloatProperty(
        name="Max Landmark Speed (m/s)",
        default=10.0,
        min=1.0,
        max=50.0,
        description="Maximum plausible landmark velocity. Faster movement is treated as an outlier"
    )
    bpy.types.Scene.melodiccap_ground_clamp = BoolProperty(
        name="Ground Clamp Feet",
        default=True,
        description="Prevent feet from going below floor level"
    )
    bpy.types.Scene.melodiccap_pin_threshold = FloatProperty(
        name="Foot Pin Threshold",
        default=0.02,
        min=0.0,
        max=0.2,
        description="Higher = stickier feet (reduces sliding). 0 = disabled"
    )

    log("MelodicCap Retargeter v1.4 registered")

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

    del bpy.types.Scene.melodiccap_start_frame
    del bpy.types.Scene.melodiccap_filter_outliers
    del bpy.types.Scene.melodiccap_outlier_velocity
    del bpy.types.Scene.melodiccap_animate_poles
    del bpy.types.Scene.melodiccap_animate_ik_rot
    del bpy.types.Scene.melodiccap_animate_fk
    del bpy.types.Scene.melodiccap_animate_spine
    del bpy.types.Scene.melodiccap_ground_clamp
    del bpy.types.Scene.melodiccap_pin_threshold

if __name__ == "__main__":
    register()
