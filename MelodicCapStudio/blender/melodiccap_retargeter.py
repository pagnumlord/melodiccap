"""
MelodicCap Retargeter v1.1
==========================
Clean retargeter combining:
- v4's proven delta-from-reference IK approach (mathematically correct)
- AntiGrav V3's V2R (Vector-to-Rotation) FK method
- Pole targets for correct elbow/knee direction (from Keemap/AntiGrav approach)
- IK target rotation for wrist/foot orientation
- Ground clamping (feet can't go through floor)
- Smart foot pinning (reduces sliding when feet should be planted)
- 4-segment spine animation via virtual midpoints

For Blender 4.4+ with JaxRigify armature.

KEY DESIGN DECISIONS:
- NO X-axis negation (v5/v12.1 proved this double-mirrors)
- L/R mirroring is handled ONLY in the bone mapping table
- Data is already in Blender coordinates from the capture script
- IK targets use delta-from-reference (includes hip movement naturally)
- Pole targets use 3-point projection (Keemap algorithm) with delta-from-reference
- FK rotations use V2R with per-bone rest axes from JaxRigify (not generic +Y)
"""

bl_info = {
    "name": "MelodicCap Retargeter",
    "author": "Karsten / MelodicCap Studio",
    "version": (1, 1, 0),
    "blender": (4, 4, 0),
    "location": "View3D > Sidebar > MelodicCap",
    "description": "Import MelodicCap motion capture data to JaxRigify armature",
    "category": "Animation",
}

import bpy
import json
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
# MediaPipe person's LEFT = Blender's RIGHT (person faces camera)
# This is the ONLY place mirroring happens. NO coordinate negation anywhere.
# =============================================================================

# IK targets for hand/foot position
IK_TARGETS = {
    'hand_ik.R': 15,   # Person's left wrist -> Character's right hand
    'hand_ik.L': 16,   # Person's right wrist -> Character's left hand
    'foot_ik.R': 27,   # Person's left ankle -> Character's right foot
    'foot_ik.L': 28,   # Person's right ankle -> Character's left foot
}

# FK bone chains for limb rotation (V2R: start landmark -> end landmark)
FK_CHAINS = {
    # Person's LEFT -> Blender RIGHT
    'upper_arm_fk.R': (11, 13),   # left shoulder -> left elbow
    'forearm_fk.R':   (13, 15),   # left elbow -> left wrist
    'thigh_fk.R':     (23, 25),   # left hip -> left knee
    'shin_fk.R':      (25, 27),   # left knee -> left ankle
    # Person's RIGHT -> Blender LEFT
    'upper_arm_fk.L': (12, 14),   # right shoulder -> right elbow
    'forearm_fk.L':   (14, 16),   # right elbow -> right wrist
    'thigh_fk.L':     (24, 26),   # right hip -> right knee
    'shin_fk.L':      (26, 28),   # right knee -> right ankle
}

# Pole targets for IK elbow/knee direction (3-point: root, mid, end)
# Uses same L↔R mirroring as IK_TARGETS
# Bone names verified from JaxRigify diagnostic dump
POLE_TARGETS = {
    'upper_arm_ik_target.R': (11, 13, 15),  # Person's L shoulder→elbow→wrist
    'upper_arm_ik_target.L': (12, 14, 16),  # Person's R shoulder→elbow→wrist
    'thigh_ik_target.R': (23, 25, 27),       # Person's L hip→knee→ankle
    'thigh_ik_target.L': (24, 26, 28),       # Person's R hip→knee→ankle
}

# IK target rotation mapping (for wrist/foot orientation)
# (start_landmark, end_landmark, rest_axis_for_that_ik_bone)
IK_ROTATION = {
    'hand_ik.R': (13, 15, Vector((0, 1, 0))),   # L forearm dir → hands point +Y at rest
    'hand_ik.L': (14, 16, Vector((0, 1, 0))),   # R forearm dir
    'foot_ik.R': (25, 27, Vector((0, 0, 1))),   # L shin dir → feet point +Z at rest
    'foot_ik.L': (26, 28, Vector((0, 0, 1))),   # R shin dir
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

# Rest axes for V2R FK rotations (verified from JaxRigify bone structure)
# These are the bone rest directions in ARMATURE SPACE for each FK bone.
BONE_REST_AXES = {
    # Arms point outward from body
    'upper_arm_fk.L': Vector((1, 0, 0)),
    'forearm_fk.L': Vector((1, 0, 0)),
    'upper_arm_fk.R': Vector((-1, 0, 0)),
    'forearm_fk.R': Vector((-1, 0, 0)),
    # Legs point down
    'thigh_fk.L': Vector((0, 0, -1)),
    'shin_fk.L': Vector((0, 0, -1)),
    'thigh_fk.R': Vector((0, 0, -1)),
    'shin_fk.R': Vector((0, 0, -1)),
    # Spine points up
    'spine_fk': Vector((0, 0, 1)),
    'spine_fk.001': Vector((0, 0, 1)),
    'spine_fk.002': Vector((0, 0, 1)),
    'spine_fk.003': Vector((0, 0, 1)),
}

# =============================================================================
# UTILITIES
# =============================================================================

def log(msg, level="INFO"):
    print(f"[{level}] MelodicCap: {msg}")

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

    def analyze(self):
        """Analyze character and mocap data, compute scale factor."""
        log("=" * 60)
        log("MELODICCAP RETARGETER v1.0 - ANALYSIS")
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
        log(f"\n  IK/FK switch status:")
        pose_bones = self.armature.pose.bones
        for switch_bone, prop_name in IK_FK_SWITCHES.items():
            if switch_bone in pose_bones:
                pb = pose_bones[switch_bone]
                if prop_name in pb:
                    val = pb[prop_name]
                    mode = "FK" if val > 0.5 else "IK"
                    log(f"    {switch_bone}['{prop_name}'] = {val:.2f} ({mode} mode)")

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

        # Force IK mode
        self.set_ik_fk_mode(use_ik=True)

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

        # Process each frame
        log(f"  Processing {len(frames)} frames...")

        for fidx, fdata in enumerate(frames):
            bf = start + fidx
            bpy.context.scene.frame_set(bf)

            lms = fdata.get('landmarks', {})
            hip = get_mid(lms, 23, 24)

            if not hip:
                continue

            # =================================================================
            # ROOT MOTION (TORSO)
            # Delta from reference hip, scaled, in armature-local space.
            # Frame 0 delta = (0,0,0) so character starts at rest position.
            # =================================================================
            if has_torso:
                delta = (hip - self.ref_hip) * self.scale
                local_delta = world_inv.to_3x3() @ delta

                torso = pose_bones['torso']
                torso.location = local_delta
                torso.keyframe_insert(data_path="location", frame=bf)

                self.stats['keys'] += 1
                self.stats['bones'].add('torso')

                if fidx == 0:
                    log(f"    Frame 0 torso delta: ({local_delta.x:.4f}, {local_delta.y:.4f}, {local_delta.z:.4f})")

            # =================================================================
            # IK TARGETS (HANDS AND FEET)
            # Delta from reference position, scaled.
            # This naturally includes hip movement because the delta captures
            # the total displacement from reference, not just local limb motion.
            # =================================================================
            for ik_bone, lm_idx in avail_ik.items():
                pos = get_lm(lms, lm_idx)
                if not pos:
                    continue

                ref_pos = get_lm(self.ref_landmarks, lm_idx)
                if not ref_pos:
                    continue

                # Delta from reference, scaled (still in world/mocap space)
                mocap_delta = (pos - ref_pos) * self.scale

                # Ground clamp for feet: prevent going below floor
                # (done in world space BEFORE converting to local, for correctness)
                if ground_clamp and 'foot' in ik_bone:
                    rest_pos = self.ik_rest_positions.get(ik_bone)
                    if rest_pos:
                        world_z = rest_pos.z + mocap_delta.z
                        if world_z < 0:
                            mocap_delta.z = -rest_pos.z

                # Convert to armature-local space
                local_delta = world_inv.to_3x3() @ mocap_delta

                # Smart pinning for feet: reduce sliding
                if 'foot' in ik_bone and pin_threshold > 0:
                    prev = prev_deltas.get(ik_bone)
                    if prev is not None:
                        if (local_delta - prev).length < pin_threshold:
                            local_delta = prev.copy()
                    prev_deltas[ik_bone] = local_delta.copy()

                pb = pose_bones[ik_bone]
                pb.location = local_delta
                pb.keyframe_insert(data_path="location", frame=bf)

                self.stats['keys'] += 1
                self.stats['bones'].add(ik_bone)

                if fidx == 0:
                    log(f"    Frame 0 {ik_bone}: ({local_delta.x:.4f}, {local_delta.y:.4f}, {local_delta.z:.4f})")

            # =================================================================
            # IK TARGET ROTATION (WRIST/FOOT ORIENTATION)
            # Orient hands along forearm direction and feet along shin direction.
            # This gives proper wrist angle and foot tilt.
            # =================================================================
            for ik_bone, (lm_start, lm_end, rest_ax) in avail_ik_rot.items():
                p1 = get_lm(lms, lm_start)
                p2 = get_lm(lms, lm_end)
                if not p1 or not p2:
                    continue

                limb_dir = (p2 - p1).normalized()
                # Transform direction to armature-local space
                dir_local = world_inv.to_quaternion() @ limb_dir

                quat = rest_ax.rotation_difference(dir_local)

                pb = pose_bones[ik_bone]
                pb.rotation_mode = 'QUATERNION'
                pb.rotation_quaternion = quat
                pb.keyframe_insert(data_path="rotation_quaternion", frame=bf)

                self.stats['keys'] += 1

            # =================================================================
            # POLE TARGETS (ELBOW/KNEE DIRECTION FOR IK SOLVER)
            # Uses 3-point projection (Keemap algorithm) to compute where the
            # elbow/knee should point, then applies as delta-from-reference.
            # Without pole targets, the IK solver guesses bend direction.
            # =================================================================
            for pole_bone, (lm_root, lm_mid, lm_end) in avail_poles.items():
                v_root = get_lm(lms, lm_root)
                v_mid = get_lm(lms, lm_mid)
                v_end = get_lm(lms, lm_end)
                if not all([v_root, v_mid, v_end]):
                    continue

                pole_pos = compute_pole_position(v_root, v_mid, v_end)
                if not pole_pos:
                    continue  # Limb too straight; skip this frame

                ref_pole = self.ref_pole_positions.get(pole_bone)
                if not ref_pole:
                    continue

                # Delta from reference, scaled
                pole_delta = (pole_pos - ref_pole) * self.scale
                local_delta = world_inv.to_3x3() @ pole_delta

                pb = pose_bones[pole_bone]
                pb.location = local_delta
                pb.keyframe_insert(data_path="location", frame=bf)

                self.stats['keys'] += 1
                self.stats['bones'].add(pole_bone)

            # =================================================================
            # FK ROTATIONS (V2R METHOD)
            # For each bone: compute direction from start->end landmark,
            # transform to armature-local, then rotation_difference from
            # rest axis to target direction.
            # NOTE: FK rotations are invisible in IK mode (IK_FK=0.0).
            # They are keyframed so the user can switch to FK mode later.
            # Rest axes are per-bone from JaxRigify bone structure.
            # =================================================================
            if animate_fk:
                for fk_bone, (i1, i2) in avail_fk.items():
                    p1, p2 = get_lm(lms, i1), get_lm(lms, i2)
                    if not p1 or not p2:
                        continue

                    target_dir = (p2 - p1).normalized()

                    # Transform direction to armature-local space
                    target_dir_local = world_inv.to_quaternion() @ target_dir

                    # Per-bone rest axis from JaxRigify bone structure
                    rest_axis = BONE_REST_AXES.get(fk_bone)
                    if not rest_axis:
                        continue  # Skip bones without known rest axes

                    # Rotation from rest to target
                    quat = rest_axis.rotation_difference(target_dir_local)

                    pb = pose_bones[fk_bone]
                    pb.rotation_quaternion = quat
                    pb.keyframe_insert(data_path="rotation_quaternion", frame=bf)

                    self.stats['keys'] += 1
                    self.stats['bones'].add(fk_bone)

            # =================================================================
            # SPINE ANIMATION (V2R with virtual midpoints)
            # Creates 5 virtual points along the torso from hips to shoulders,
            # then applies V2R rotation to each spine segment.
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

                        # Spine bones point +Z at rest
                        rest_axis = Vector((0, 0, 1))
                        quat = rest_axis.rotation_difference(target_dir_local)

                        pb = pose_bones[bone_name]
                        pb.rotation_quaternion = quat
                        pb.keyframe_insert(data_path="rotation_quaternion", frame=bf)

                        self.stats['keys'] += 1
                        self.stats['bones'].add(bone_name)

            self.stats['frames'] += 1

            if fidx % 50 == 0:
                log(f"    Frame {fidx}/{len(frames)}")

        return True

    def summary(self):
        """Print import summary."""
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

        log(f"\n  File: {self.filepath}")
        log(f"  Armature: {arm.name}")

        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load JSON: {e}")
            return {'CANCELLED'}

        settings = {
            'start_frame': context.scene.melodiccap_start_frame,
            'animate_fk': context.scene.melodiccap_animate_fk,
            'animate_spine': context.scene.melodiccap_animate_spine,
            'animate_poles': context.scene.melodiccap_animate_poles,
            'animate_ik_rot': context.scene.melodiccap_animate_ik_rot,
            'ground_clamp': context.scene.melodiccap_ground_clamp,
            'pin_threshold': context.scene.melodiccap_pin_threshold,
        }

        imp = MelodicCapImporter(arm, data, settings)

        if not imp.analyze():
            self.report({'ERROR'}, "Analysis failed - check console")
            return {'CANCELLED'}

        bpy.ops.object.mode_set(mode='POSE')
        imp.apply_animation()
        imp.summary()

        self.report({'INFO'}, f"Imported {imp.stats['frames']} frames, {len(imp.stats['bones'])} bones animated")
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


# =============================================================================
# REGISTRATION
# =============================================================================

classes = [
    MELODICCAP_OT_import,
    MELODICCAP_OT_clear,
    MELODICCAP_OT_set_ik_mode,
    MELODICCAP_OT_set_fk_mode,
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

    log("MelodicCap Retargeter v1.1 registered")

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

    del bpy.types.Scene.melodiccap_start_frame
    del bpy.types.Scene.melodiccap_animate_poles
    del bpy.types.Scene.melodiccap_animate_ik_rot
    del bpy.types.Scene.melodiccap_animate_fk
    del bpy.types.Scene.melodiccap_animate_spine
    del bpy.types.Scene.melodiccap_ground_clamp
    del bpy.types.Scene.melodiccap_pin_threshold

if __name__ == "__main__":
    register()
