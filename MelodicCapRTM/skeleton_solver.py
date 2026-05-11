"""
Skeleton Solver — Constrained Articulated Fitting
===================================================
Fits an articulated rigid-body skeleton with fixed bone lengths and
joint angle limits to triangulated 3D keypoints. Produces anatomically
valid skeleton poses every frame.

Design: direction-preserving chain fitting. Triangulation errors mainly
corrupt distances (depth axis noisy), but directions between adjacent
joints are generally reliable. We trust the triangulated directions and
enforce the calibrated bone lengths.

Deterministic (no iterative convergence), numpy-only (no scipy), and
guarantees exact bone lengths by construction.

Usage (called by offline_processor.py):
    solver = SkeletonSolver(n_cal_frames=30)
    output_frames = solver.solve_sequence(output_frames)
"""

import numpy as np
from keypoint_map import LM


BONE_DEFS = [
    (LM.LEFT_SHOULDER, LM.LEFT_ELBOW, 'upper_arm_l'),
    (LM.LEFT_ELBOW, LM.LEFT_WRIST, 'forearm_l'),
    (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, 'upper_arm_r'),
    (LM.RIGHT_ELBOW, LM.RIGHT_WRIST, 'forearm_r'),
    (LM.LEFT_HIP, LM.LEFT_KNEE, 'thigh_l'),
    (LM.LEFT_KNEE, LM.LEFT_ANKLE, 'shin_l'),
    (LM.RIGHT_HIP, LM.RIGHT_KNEE, 'thigh_r'),
    (LM.RIGHT_KNEE, LM.RIGHT_ANKLE, 'shin_r'),
]

VIRTUAL_BONE_NAMES = ['spine', 'hip_width', 'shoulder_width']

ARM_CHAINS = [
    ('left_arm', LM.LEFT_SHOULDER, [
        (LM.LEFT_ELBOW, 'upper_arm_l'),
        (LM.LEFT_WRIST, 'forearm_l'),
    ]),
    ('right_arm', LM.RIGHT_SHOULDER, [
        (LM.RIGHT_ELBOW, 'upper_arm_r'),
        (LM.RIGHT_WRIST, 'forearm_r'),
    ]),
]

LEG_CHAINS = [
    ('left_leg', LM.LEFT_HIP, [
        (LM.LEFT_KNEE, 'thigh_l'),
        (LM.LEFT_ANKLE, 'shin_l'),
    ]),
    ('right_leg', LM.RIGHT_HIP, [
        (LM.RIGHT_KNEE, 'thigh_r'),
        (LM.RIGHT_ANKLE, 'shin_r'),
    ]),
]

JOINT_ROM = {
    'elbow_l': (5, 175),
    'elbow_r': (5, 175),
    'knee_l': (5, 175),
    'knee_r': (5, 175),
}

JOINT_ANGLE_DEFS = [
    (LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST, 'elbow_l'),
    (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST, 'elbow_r'),
    (LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE, 'knee_l'),
    (LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE, 'knee_r'),
]

# v5.16: Shoulder and hip ROM constraints. The parent reference is a virtual
# midpoint (hip_mid for shoulders, shoulder_mid for hips) which can't fit the
# JOINT_ANGLE_DEFS triple-of-landmarks pattern, so they're applied separately
# in solve_frame Step 7b.
#
# Bounds: humerus-to-spine angle in [25°, 178°] for shoulder. 25° lower bound
# means the upper arm cannot be within 25° of "parallel to spine" (which would
# physically be the arm pointing straight down past the hip in standing pose,
# anatomically fine, OR the arm passing through the rib cage when the body is
# rotated, anatomically impossible). 178° upper bound prevents math degenerate
# cases with the angle clamp's axis computation. Real anatomical max for arm
# overhead is 0° to spine, but at 0° the cross product axis is undefined; 178°
# (in our convention angle = 180° - actual_anatomical_angle) gives a tiny
# margin.
#
# Hip: thigh-to-spine angle in [15°, 178°]. 15° prevents the leg from
# crossing the body centerline upward (hip flexion past 165° physically
# impossible). 178° same degenerate-case margin.
SHOULDER_HIP_ROM = {
    'shoulder_l': (25, 178),
    'shoulder_r': (25, 178),
    'hip_l':      (15, 178),
    'hip_r':      (15, 178),
}

DIRECTION_SMOOTH_ALPHA = 0.5          # baseline EMA weight
DIRECTION_SMOOTH_ALPHA_LOWCONF = 0.15  # trust previous heavily when a joint is noisy
DIRECTION_SMOOTH_ALPHA_JUMP = 0.85     # let large, high-conf direction changes through
DIRECTION_LOWCONF_THRESHOLD = 0.4      # joint conf below this -> use LOWCONF α
DIRECTION_GOODCONF_THRESHOLD = 0.5     # joint conf above this -> allow JUMP α
DIRECTION_JUMP_DOT = 0.7               # cos(45°) — dir change beyond this counts as "jump"
# v5.16: very-low-confidence band. The detector's per-keypoint confidence
# (min(cam_a, cam_b) per triangulated keypoint, in JSON since v4.7) gives us
# a real signal for "this frame's geometry is mostly noise". Average of the
# bone's two endpoints below VERY_LOWCONF_THRESHOLD = use α = 0.08 — almost
# entirely trust the previous frame, drop the current one. This is the gate
# the existing min-based check was missing: a single low-conf endpoint hits
# 0.15 (LOWCONF) but a bone where BOTH endpoints are weak should be even
# more cautious.
DIRECTION_VERY_LOWCONF_THRESHOLD = 0.35
DIRECTION_SMOOTH_ALPHA_VERY_LOWCONF = 0.08
LOG_INTERVAL = 50


def _safe_normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.array([0.0, 0.0, 1.0])
    return v / n


def _angle_between(v1, v2):
    c = np.clip(np.dot(_safe_normalize(v1), _safe_normalize(v2)), -1.0, 1.0)
    return np.degrees(np.arccos(c))


def _clamp_angle(parent_pos, joint_pos, child_pos, min_deg, max_deg, bone_length):
    """Clamp the angle at joint_pos between parent-joint and joint-child.

    Returns the corrected child position with the original bone length preserved.
    If clamped, rotates the child direction around the bend axis to the
    nearest legal angle.
    """
    v_parent = parent_pos - joint_pos
    v_child = child_pos - joint_pos

    angle = _angle_between(v_parent, v_child)

    if min_deg <= angle <= max_deg:
        return child_pos, angle, False

    target_angle = np.clip(angle, min_deg, max_deg)
    target_rad = np.radians(target_angle)

    axis = np.cross(v_parent, v_child)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        perp = np.array([1, 0, 0]) if abs(v_parent[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(v_parent, perp)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            return child_pos, angle, False

    axis = axis / axis_norm

    parent_dir = _safe_normalize(v_parent)
    # Rotate parent_dir by target_rad around axis (Rodrigues). The result is
    # the child direction at angle target_rad from the parent direction.
    # Previous code rotated by (pi - target_rad) instead, which clamped to
    # the *complement* angle (e.g. 178° clamped to 5° instead of 175°),
    # collapsing wrists onto shoulders whenever elbow noise pushed past 175°.
    c = np.cos(target_rad)
    s = np.sin(target_rad)
    new_child_dir = parent_dir * c + np.cross(axis, parent_dir) * s + axis * np.dot(axis, parent_dir) * (1 - c)
    new_child_dir = _safe_normalize(new_child_dir)

    corrected = joint_pos + new_child_dir * bone_length
    return corrected, angle, True


class SkeletonSolver:
    def __init__(self, n_cal_frames=30, direction_smooth_alpha=DIRECTION_SMOOTH_ALPHA):
        self.n_cal_frames = n_cal_frames
        self.direction_smooth_alpha = float(direction_smooth_alpha)
        self.bone_lengths = {}
        self.calibrated = False
        self._stats = {
            'angle_clamps': 0,
            'angle_clamp_details': {},
            'max_length_delta': 0.0,
            'length_deltas': [],
            'frames_solved': 0,
            'frames_skipped': 0,
            'smooth_alpha_lowconf_frames': 0,
            'smooth_alpha_jump_frames': 0,
            'spine_rate_clamps': 0,
        }
        # v5.16: spine direction from previous frame, used by Step 2b's pitch
        # rate limit. None on first frame (no comparison possible).
        self._prev_spine_dir = None

    def _get_point(self, landmarks, idx):
        key = str(idx)
        if key in landmarks:
            return np.array(landmarks[key], dtype=np.float64)
        if idx in landmarks:
            return np.array(landmarks[idx], dtype=np.float64)
        return None

    def _all_body_present(self, landmarks):
        required = [
            LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
            LM.LEFT_HIP, LM.RIGHT_HIP,
        ]
        return all(self._get_point(landmarks, idx) is not None for idx in required)

    def calibrate(self, frames):
        """
        Collect bone-length samples from clean-pose frames only.

        v5.7: each candidate frame must pass quality gates BEFORE its samples
        are admitted. Gates reject frames where:
          - L/R bone pairs differ by >12% (triangulation asymmetry)
          - Hip Z deviates from running median by >3σ (sitting/crouching
            contaminating an otherwise-standing calibration window)
          - Shoulder midpoint above hip midpoint by <0.25m (slouched)
        If fewer than MIN_GOOD_FRAMES survive, the solve aborts with an error
        rather than silently calibrating on bad data.
        """
        MIN_GOOD_FRAMES = 20
        MAX_LR_ASYM = 0.12  # 12% between L/R bone lengths
        MAX_HIP_Z_SIGMA = 3.0
        MIN_SHOULDER_HIP_DZ = 0.25

        samples = {name: [] for _, _, name in BONE_DEFS}
        for vname in VIRTUAL_BONE_NAMES:
            samples[vname] = []

        # Pass 1: compute per-frame bone lengths for all candidates, also
        # track hip Z so we can compute running median + sigma.
        candidates = []
        for frame in frames:
            lm = frame.get('landmarks_3d', {})
            if not lm or not self._all_body_present(lm):
                continue

            per_frame = {}
            for parent_idx, child_idx, name in BONE_DEFS:
                p = self._get_point(lm, parent_idx)
                c = self._get_point(lm, child_idx)
                if p is None or c is None:
                    continue
                length = float(np.linalg.norm(c - p))
                if 0.01 < length < 1.5:
                    per_frame[name] = length

            ls = self._get_point(lm, LM.LEFT_SHOULDER)
            rs = self._get_point(lm, LM.RIGHT_SHOULDER)
            lh = self._get_point(lm, LM.LEFT_HIP)
            rh = self._get_point(lm, LM.RIGHT_HIP)
            if ls is None or rs is None or lh is None or rh is None:
                continue

            hip_mid = (lh + rh) / 2
            shoulder_mid = (ls + rs) / 2
            spine_len = float(np.linalg.norm(shoulder_mid - hip_mid))
            if 0.1 < spine_len < 1.0:
                per_frame['spine'] = spine_len
            hw = float(np.linalg.norm(lh - rh))
            if 0.05 < hw < 0.6:
                per_frame['hip_width'] = hw
            sw = float(np.linalg.norm(ls - rs))
            if 0.10 < sw < 0.7:
                per_frame['shoulder_width'] = sw

            candidates.append({
                'bones': per_frame,
                'hip_z': float(hip_mid[2]),
                'shoulder_hip_dz': float(shoulder_mid[2] - hip_mid[2]),
            })

        # Pass 2: derive hip-Z statistics across all candidates, then gate
        hip_zs = np.array([c['hip_z'] for c in candidates]) if candidates else np.array([])
        if len(hip_zs) > 0:
            hip_z_med = float(np.median(hip_zs))
            hip_z_sigma = float(np.std(hip_zs))
        else:
            hip_z_med, hip_z_sigma = 0.0, 0.0

        good_frames = 0
        rejected = {'asymmetry': 0, 'hip_z': 0, 'slouch': 0}

        def _lr_pair_ok(pf, l_name, r_name):
            l = pf.get(l_name)
            r = pf.get(r_name)
            if l is None or r is None:
                return True  # can't check → don't reject
            if max(l, r) <= 0:
                return True
            return abs(l - r) / max(l, r) <= MAX_LR_ASYM

        for cand in candidates:
            if good_frames >= self.n_cal_frames:
                break
            pf = cand['bones']

            # Gate: L/R asymmetry
            pairs_ok = all([
                _lr_pair_ok(pf, 'upper_arm_l', 'upper_arm_r'),
                _lr_pair_ok(pf, 'forearm_l', 'forearm_r'),
                _lr_pair_ok(pf, 'thigh_l', 'thigh_r'),
                _lr_pair_ok(pf, 'shin_l', 'shin_r'),
            ])
            if not pairs_ok:
                rejected['asymmetry'] += 1
                continue

            # Gate: hip Z far from median (sitting/crouching contamination)
            if (hip_z_sigma > 0 and
                    abs(cand['hip_z'] - hip_z_med) > MAX_HIP_Z_SIGMA * hip_z_sigma):
                rejected['hip_z'] += 1
                continue

            # Gate: slouched pose
            if cand['shoulder_hip_dz'] < MIN_SHOULDER_HIP_DZ:
                rejected['slouch'] += 1
                continue

            for name, length in pf.items():
                samples[name].append(length)
            good_frames += 1

        print(f"\n[SOLVER] ═══════════════════════════════════════════════════")
        print(f"[SOLVER] Bone length calibration ({good_frames} clean frames, "
              f"{sum(rejected.values())} rejected: "
              f"asym={rejected['asymmetry']} hipZ={rejected['hip_z']} "
              f"slouch={rejected['slouch']})")
        print(f"[SOLVER] ───────────────────────────────────────────────────")

        if good_frames < MIN_GOOD_FRAMES:
            print(f"[SOLVER] ✗ Calibration FAILED — only {good_frames} clean frames "
                  f"survived (need ≥{MIN_GOOD_FRAMES}). The capture does not "
                  f"contain enough clean standing frames to calibrate bone lengths.")
            print(f"[SOLVER] ═══════════════════════════════════════════════════\n")
            self.calibrated = False
            return False

        min_samples = max(5, good_frames // 3)
        for name in [n for _, _, n in BONE_DEFS] + VIRTUAL_BONE_NAMES:
            s = samples[name]
            if len(s) >= min_samples:
                median = float(np.median(s))
                std = float(np.std(s))
                self.bone_lengths[name] = median
                print(f"[SOLVER]   {name:20s}: {median:.4f}m  (σ={std:.4f}m, n={len(s)})")
            else:
                print(f"[SOLVER]   {name:20s}: INSUFFICIENT DATA (n={len(s)}, need {min_samples})")

        self.calibrated = len(self.bone_lengths) >= 8
        if self.calibrated:
            print(f"[SOLVER] ✓ Calibrated {len(self.bone_lengths)} bones")
        else:
            print(f"[SOLVER] ✗ Calibration FAILED — only {len(self.bone_lengths)} bones learned (need ≥8)")
        print(f"[SOLVER] ═══════════════════════════════════════════════════\n")
        return self.calibrated

    def solve_frame(self, landmarks, frame_idx=-1, do_log=False):
        if not self.calibrated:
            return landmarks

        if not landmarks or not self._all_body_present(landmarks):
            self._stats['frames_skipped'] += 1
            return landmarks

        pts = {}
        for key, val in landmarks.items():
            idx = int(key) if isinstance(key, str) else key
            pts[idx] = np.array(val, dtype=np.float64)

        corrected = dict(pts)
        frame_deltas = []

        lh = pts.get(LM.LEFT_HIP)
        rh = pts.get(LM.RIGHT_HIP)
        ls = pts.get(LM.LEFT_SHOULDER)
        rs = pts.get(LM.RIGHT_SHOULDER)

        if lh is None or rh is None or ls is None or rs is None:
            self._stats['frames_skipped'] += 1
            return landmarks

        hip_mid = (lh + rh) / 2.0
        shoulder_mid_raw = (ls + rs) / 2.0

        # Step 1b: spine pitch RATE limit (v5.16). If the spine direction
        # changed by more than SPINE_PITCH_MAX_RATE_DEG per frame, rotate
        # the current direction toward the previous one to absorb the
        # excess change. This kills the per-frame discontinuity at
        # sit-down transitions (the user described as "rough sit motion")
        # and also damps depth-axis noise that drives spurious 8-15° pitch
        # spikes during normal motion.
        #
        # Cap chosen to match the addon's NECK_MAX_ANGULAR_VEL after the
        # v5.16 tightening (5°/frame at 8fps = 40°/sec). Spine flexes
        # slower than the neck — 8°/frame is generous and still cuts the
        # depth-spike contribution to the pitch trajectory.
        SPINE_PITCH_MAX_RATE_DEG = 8.0
        cur_spine_v = shoulder_mid_raw - hip_mid
        spine_len_raw = float(np.linalg.norm(cur_spine_v))
        if self._prev_spine_dir is not None and spine_len_raw > 1e-6:
            cur_spine_dir = cur_spine_v / spine_len_raw
            cos_a = float(np.clip(np.dot(cur_spine_dir, self._prev_spine_dir), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(cos_a)))
            if angle_deg > SPINE_PITCH_MAX_RATE_DEG:
                excess_rad = float(np.radians(angle_deg - SPINE_PITCH_MAX_RATE_DEG))
                # Rodrigues rotation around (cur × prev) toward prev
                axis = np.cross(cur_spine_dir, self._prev_spine_dir)
                axis_n = float(np.linalg.norm(axis))
                if axis_n > 1e-8:
                    axis = axis / axis_n
                    cos_t = float(np.cos(excess_rad))
                    sin_t = float(np.sin(excess_rad))
                    new_dir = (cur_spine_dir * cos_t
                               + np.cross(axis, cur_spine_dir) * sin_t
                               + axis * np.dot(axis, cur_spine_dir) * (1.0 - cos_t))
                    new_dir = _safe_normalize(new_dir)
                    shoulder_mid_raw = hip_mid + new_dir * spine_len_raw
                    self._stats['spine_rate_clamps'] += 1
                    if do_log:
                        print(f"[SOLVER] frame {frame_idx}: spine_rate clamped "
                              f"{angle_deg:.1f}° → {SPINE_PITCH_MAX_RATE_DEG:.1f}° "
                              f"per frame")
            # Store the (possibly-clamped) direction for next frame
            self._prev_spine_dir = _safe_normalize(shoulder_mid_raw - hip_mid)
        else:
            self._prev_spine_dir = _safe_normalize(cur_spine_v) if spine_len_raw > 1e-6 else None

        # Step 2: spine (soft constraint — only correct if >30% deviation)
        if 'spine' in self.bone_lengths:
            raw_spine = np.linalg.norm(shoulder_mid_raw - hip_mid)
            spine_cal = self.bone_lengths['spine']
            spine_ratio = raw_spine / spine_cal if spine_cal > 1e-6 else 1.0
            if abs(spine_ratio - 1.0) > 0.30:
                sign = 1.0 if spine_ratio > 1.0 else -1.0
                target = spine_cal * (1.0 + 0.30 * sign)
                spine_dir = _safe_normalize(shoulder_mid_raw - hip_mid)
                shoulder_mid = hip_mid + spine_dir * target
                delta = abs(raw_spine - target)
                frame_deltas.append(('spine', raw_spine, target, delta))
                if do_log:
                    print(f"[SOLVER] frame {frame_idx}: spine CORRECTED "
                          f"raw={raw_spine:.3f}m → capped={target:.3f}m "
                          f"(ratio={spine_ratio:.2f}, cal={spine_cal:.3f}m)")
            else:
                shoulder_mid = shoulder_mid_raw
        else:
            shoulder_mid = shoulder_mid_raw

        # Step 3: shoulders — anchor to calibrated shoulder_width (v5.15).
        #
        # Diagnostic on take 204628 showed shoulder width collapsing 44%
        # during forward bends (0.349m → 0.195m), pure depth-axis triangulation
        # noise — anatomy is fixed, the cameras just lose correspondence as
        # both shoulders move toward them. Previously this step kept the raw
        # half-offsets and let the collapse propagate to every downstream
        # consumer (torso yaw/pitch, FK arms, retargeting). Now: when the
        # raw L↔R distance deviates >10% from calibration, project both
        # shoulders symmetrically onto the calibrated separation along the
        # raw L→R axis. ±10% allows real shoulder-girdle flex (shrugs,
        # scapula rotation in arm raises) but kills the >20% camera-driven
        # collapse.
        half_offset_l = ls - shoulder_mid_raw
        half_offset_r = rs - shoulder_mid_raw
        if 'shoulder_width' in self.bone_lengths:
            sw_cal = self.bone_lengths['shoulder_width']
            raw_sw = float(np.linalg.norm(ls - rs))
            sw_ratio = raw_sw / sw_cal if sw_cal > 1e-6 else 1.0
            if abs(sw_ratio - 1.0) > 0.10 and raw_sw > 1e-6:
                sign = 1.0 if sw_ratio > 1.0 else -1.0
                target_sw = sw_cal * (1.0 + 0.10 * sign)
                shoulder_axis = _safe_normalize(ls - rs)
                # Project around the corrected shoulder_mid so the spine
                # constraint (Step 2) still anchors them vertically.
                corrected[LM.LEFT_SHOULDER] = shoulder_mid + shoulder_axis * (target_sw / 2)
                corrected[LM.RIGHT_SHOULDER] = shoulder_mid - shoulder_axis * (target_sw / 2)
                frame_deltas.append(('shoulder_width', raw_sw, target_sw, abs(raw_sw - target_sw)))
                if do_log:
                    print(f"[SOLVER] frame {frame_idx}: shoulder_width CORRECTED "
                          f"raw={raw_sw:.3f}m → capped={target_sw:.3f}m "
                          f"(ratio={sw_ratio:.2f}, cal={sw_cal:.3f}m)")
            else:
                corrected[LM.LEFT_SHOULDER] = shoulder_mid + half_offset_l
                corrected[LM.RIGHT_SHOULDER] = shoulder_mid + half_offset_r
        else:
            corrected[LM.LEFT_SHOULDER] = shoulder_mid + half_offset_l
            corrected[LM.RIGHT_SHOULDER] = shoulder_mid + half_offset_r

        # Step 3b: hip width — same anatomical anchoring as shoulders.
        # v5.15: tightened threshold 20% → 10%. Old 20% let the diagnostic
        # range 0.16-0.24m through (calibration mean 0.199m), so 9% of frames
        # had >10% hip-width error post-solver.
        if 'hip_width' in self.bone_lengths:
            hw_cal = self.bone_lengths['hip_width']
            raw_hw = float(np.linalg.norm(lh - rh))
            hw_ratio = raw_hw / hw_cal if hw_cal > 1e-6 else 1.0
            if abs(hw_ratio - 1.0) > 0.10 and raw_hw > 1e-6:
                sign = 1.0 if hw_ratio > 1.0 else -1.0
                target_hw = hw_cal * (1.0 + 0.10 * sign)
                hip_axis = _safe_normalize(lh - rh)
                corrected[LM.LEFT_HIP] = hip_mid + hip_axis * (target_hw / 2)
                corrected[LM.RIGHT_HIP] = hip_mid - hip_axis * (target_hw / 2)
                frame_deltas.append(('hip_width', raw_hw, target_hw, abs(raw_hw - target_hw)))
                if do_log:
                    print(f"[SOLVER] frame {frame_idx}: hip_width CORRECTED "
                          f"raw={raw_hw:.3f}m → capped={target_hw:.3f}m "
                          f"(ratio={hw_ratio:.2f}, cal={hw_cal:.3f}m)")

        # Step 5: arm chains
        for chain_name, root_idx, segments in ARM_CHAINS:
            parent_pos = corrected.get(root_idx)
            if parent_pos is None:
                continue
            for child_idx, bone_name in segments:
                child_raw = pts.get(child_idx)
                if child_raw is None:
                    break
                if bone_name not in self.bone_lengths:
                    parent_pos = child_raw
                    continue
                bone_len = self.bone_lengths[bone_name]
                direction = _safe_normalize(child_raw - parent_pos)
                corrected_child = parent_pos + direction * bone_len
                raw_len = np.linalg.norm(child_raw - parent_pos)
                delta = abs(raw_len - bone_len)
                frame_deltas.append((bone_name, raw_len, bone_len, delta))
                corrected[child_idx] = corrected_child
                parent_pos = corrected_child

        # Step 6: leg chains
        for chain_name, root_idx, segments in LEG_CHAINS:
            parent_pos = corrected.get(root_idx)
            if parent_pos is None:
                continue
            for child_idx, bone_name in segments:
                child_raw = pts.get(child_idx)
                if child_raw is None:
                    break
                if bone_name not in self.bone_lengths:
                    parent_pos = child_raw
                    continue
                bone_len = self.bone_lengths[bone_name]
                direction = _safe_normalize(child_raw - parent_pos)
                corrected_child = parent_pos + direction * bone_len
                raw_len = np.linalg.norm(child_raw - parent_pos)
                delta = abs(raw_len - bone_len)
                frame_deltas.append((bone_name, raw_len, bone_len, delta))
                corrected[child_idx] = corrected_child
                parent_pos = corrected_child

        # Step 7: joint angle limits
        for parent_idx, joint_idx, child_idx, joint_name in JOINT_ANGLE_DEFS:
            if joint_name not in JOINT_ROM:
                continue
            p = corrected.get(parent_idx)
            j = corrected.get(joint_idx)
            c = corrected.get(child_idx)
            if p is None or j is None or c is None:
                continue

            min_deg, max_deg = JOINT_ROM[joint_name]
            bone_name = None
            for _, ci, bn in BONE_DEFS:
                if ci == child_idx:
                    bone_name = bn
                    break
            bone_len = self.bone_lengths.get(bone_name, np.linalg.norm(c - j))

            new_child, raw_angle, was_clamped = _clamp_angle(p, j, c, min_deg, max_deg, bone_len)
            if was_clamped:
                corrected[child_idx] = new_child
                clamped_angle = np.clip(raw_angle, min_deg, max_deg)
                self._stats['angle_clamps'] += 1
                if joint_name not in self._stats['angle_clamp_details']:
                    self._stats['angle_clamp_details'][joint_name] = 0
                self._stats['angle_clamp_details'][joint_name] += 1
                if do_log:
                    print(f"[SOLVER] frame {frame_idx}: {joint_name} clamped {raw_angle:.1f}° → {clamped_angle:.1f}°")

        # Step 7b: shoulder and hip ROM (v5.16). The "parent" for these is a
        # virtual spine midpoint, not a single landmark, so this is separate
        # from the JOINT_ANGLE_DEFS loop above.
        #
        # For shoulders, the parent reference is hip_mid (so the angle measures
        # how far the upper arm deviates from the spine direction).
        # For hips, the parent reference is shoulder_mid (same idea, opposite
        # end of the spine).
        sh_hip_constraints = [
            ('shoulder_l', hip_mid, LM.LEFT_SHOULDER, LM.LEFT_ELBOW, 'upper_arm_l'),
            ('shoulder_r', hip_mid, LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, 'upper_arm_r'),
            ('hip_l', shoulder_mid, LM.LEFT_HIP, LM.LEFT_KNEE, 'thigh_l'),
            ('hip_r', shoulder_mid, LM.RIGHT_HIP, LM.RIGHT_KNEE, 'thigh_r'),
        ]
        for joint_name, parent_pos, joint_idx, child_idx, bone_name in sh_hip_constraints:
            if joint_name not in SHOULDER_HIP_ROM:
                continue
            j = corrected.get(joint_idx)
            c = corrected.get(child_idx)
            if parent_pos is None or j is None or c is None:
                continue
            min_deg, max_deg = SHOULDER_HIP_ROM[joint_name]
            bone_len = self.bone_lengths.get(bone_name, np.linalg.norm(c - j))
            new_child, raw_angle, was_clamped = _clamp_angle(
                parent_pos, j, c, min_deg, max_deg, bone_len)
            if was_clamped:
                corrected[child_idx] = new_child
                self._stats['angle_clamps'] += 1
                if joint_name not in self._stats['angle_clamp_details']:
                    self._stats['angle_clamp_details'][joint_name] = 0
                self._stats['angle_clamp_details'][joint_name] += 1
                if do_log:
                    clamped_angle = np.clip(raw_angle, min_deg, max_deg)
                    print(f"[SOLVER] frame {frame_idx}: {joint_name} clamped "
                          f"{raw_angle:.1f}° → {clamped_angle:.1f}°")

        # Track length correction stats
        if frame_deltas:
            max_delta = max(d[3] for d in frame_deltas)
            self._stats['max_length_delta'] = max(self._stats['max_length_delta'], max_delta)
            self._stats['length_deltas'].append(max_delta)

        if do_log and frame_deltas:
            worst = max(frame_deltas, key=lambda x: x[3])
            print(f"[SOLVER] frame {frame_idx}: worst correction {worst[0]} "
                  f"raw={worst[1]:.3f}m → fixed={worst[2]:.3f}m (Δ={worst[3]:.3f}m, "
                  f"{worst[3]/worst[2]*100:.1f}%)")

        self._stats['frames_solved'] += 1

        result = dict(landmarks)
        for idx, pos in corrected.items():
            result[str(idx)] = pos.tolist()

        return result

    def _temporal_smooth_directions(self, all_solved_frames):
        """Confidence-weighted EMA on bone direction vectors.

        v5.7: α varies per-frame-per-bone based on joint confidence:
          - Either endpoint conf < DIRECTION_LOWCONF_THRESHOLD → α = LOWCONF
            (trust previous heavily; don't let a bad frame propagate).
          - Both endpoints conf ≥ DIRECTION_GOODCONF_THRESHOLD AND
            current direction jumps sharply from previous (dot < JUMP_DOT)
            → α = JUMP (let legitimate sit/stand transitions through).
          - Otherwise → baseline α (from __init__).

        Fixes the pre-v5.7 symptom where a fixed α=0.5 would blend one bad
        low-confidence frame into several subsequent frames, producing the
        "arm freeze at armrest" pattern.
        """
        n = len(all_solved_frames)
        if n < 3:
            return all_solved_frames

        all_chains = ARM_CHAINS + LEG_CHAINS
        baseline_alpha = self.direction_smooth_alpha

        # Per-frame confidence lookup (per keypoint)
        confidences = []
        for frame in all_solved_frames:
            conf_dict = frame.get('confidence') or {}
            if conf_dict:
                parsed = {}
                for k, v in conf_dict.items():
                    try:
                        parsed[int(k)] = float(v)
                    except (TypeError, ValueError):
                        continue
                confidences.append(parsed)
            else:
                confidences.append(None)

        def _joint_conf(i, idx):
            if confidences[i] is None:
                return 1.0  # no confidence info → don't downweight
            return confidences[i].get(int(idx), 1.0)

        bone_directions = {}
        bone_keys_ordered = []
        for chain_name, root_idx, segments in all_chains:
            prev = root_idx
            for child_idx, bone_name in segments:
                key = (prev, child_idx, bone_name)
                bone_directions[key] = []
                bone_keys_ordered.append(key)
                prev = child_idx

        for frame in all_solved_frames:
            lm = frame.get('landmarks_3d', {})
            if not lm:
                for key in bone_keys_ordered:
                    bone_directions[key].append(None)
                continue

            for parent_idx, child_idx, bone_name in bone_keys_ordered:
                p = self._get_point(lm, parent_idx)
                c = self._get_point(lm, child_idx)
                key = (parent_idx, child_idx, bone_name)
                if p is not None and c is not None:
                    bone_directions[key].append(_safe_normalize(c - p))
                else:
                    bone_directions[key].append(None)

        def _ema_smooth(dirs, parent_idx, child_idx):
            smoothed = [None] * len(dirs)
            running = None
            for i, d in enumerate(dirs):
                if d is None:
                    smoothed[i] = running
                    continue
                if running is None:
                    running = d.copy()
                    smoothed[i] = running.copy()
                    continue

                # Confidence of parent and child on this frame
                parent_conf = _joint_conf(i, parent_idx)
                child_conf = _joint_conf(i, child_idx)
                min_conf = min(parent_conf, child_conf)
                avg_conf = (parent_conf + child_conf) / 2.0

                # v5.16: VERY_LOWCONF tier — both endpoints weak on average.
                # Use heaviest smoothing so the bad frame barely propagates.
                if avg_conf < DIRECTION_VERY_LOWCONF_THRESHOLD:
                    a = DIRECTION_SMOOTH_ALPHA_VERY_LOWCONF
                    self._stats['smooth_alpha_lowconf_frames'] += 1
                elif min_conf < DIRECTION_LOWCONF_THRESHOLD:
                    a = DIRECTION_SMOOTH_ALPHA_LOWCONF
                    self._stats['smooth_alpha_lowconf_frames'] += 1
                else:
                    dot = float(np.dot(d, running))
                    if (min_conf >= DIRECTION_GOODCONF_THRESHOLD
                            and dot < DIRECTION_JUMP_DOT):
                        a = DIRECTION_SMOOTH_ALPHA_JUMP
                        self._stats['smooth_alpha_jump_frames'] += 1
                    else:
                        a = baseline_alpha

                running = a * d + (1 - a) * running
                running = _safe_normalize(running)
                smoothed[i] = running.copy()
            return smoothed

        smoothed_dirs = {}
        for key, dirs in bone_directions.items():
            parent_idx, child_idx, _ = key
            smoothed_dirs[key] = _ema_smooth(dirs, parent_idx, child_idx)

        result_frames = []
        for i, frame in enumerate(all_solved_frames):
            lm = frame.get('landmarks_3d', {})
            if not lm:
                result_frames.append(frame)
                continue

            new_lm = dict(lm)

            for chain_name, root_idx, segments in all_chains:
                parent_pos = self._get_point(new_lm, root_idx)
                if parent_pos is None:
                    continue
                prev = root_idx
                for child_idx, bone_name in segments:
                    key = (prev, child_idx, bone_name)
                    sd = smoothed_dirs.get(key, [None] * n)
                    if i < len(sd) and sd[i] is not None and bone_name in self.bone_lengths:
                        new_child = parent_pos + sd[i] * self.bone_lengths[bone_name]
                        new_lm[str(child_idx)] = new_child.tolist()
                        parent_pos = new_child
                    else:
                        cp = self._get_point(new_lm, child_idx)
                        if cp is not None:
                            parent_pos = cp
                        break
                    prev = child_idx

            new_frame = dict(frame)
            new_frame['landmarks_3d'] = new_lm
            result_frames.append(new_frame)

        return result_frames

    def solve_sequence(self, frames):
        if not frames:
            return frames

        print(f"\n[SOLVER] Starting skeleton solver on {len(frames)} frames...")

        frames_with_data = [f for f in frames if f.get('landmarks_3d')]
        if not frames_with_data:
            print("[SOLVER] No frames with landmark data — skipping solver")
            return frames

        if not self.calibrate(frames_with_data):
            print("[SOLVER] Calibration failed — returning original frames")
            return frames

        print(f"[SOLVER] Solving {len(frames)} frames...")
        solved = []
        for i, frame in enumerate(frames):
            lm = frame.get('landmarks_3d')
            if not lm:
                solved.append(frame)
                continue

            do_log = (i % LOG_INTERVAL == 0) or (i == len(frames) - 1)
            new_lm = self.solve_frame(lm, frame_idx=i, do_log=do_log)
            new_frame = dict(frame)
            new_frame['landmarks_3d'] = new_lm
            solved.append(new_frame)

        print(f"\n[SOLVER] Applying temporal direction smoothing "
              f"(α baseline={self.direction_smooth_alpha}, "
              f"low-conf={DIRECTION_SMOOTH_ALPHA_LOWCONF}, "
              f"jump={DIRECTION_SMOOTH_ALPHA_JUMP})...")
        smoothed = self._temporal_smooth_directions(solved)

        self._print_summary(len(frames))
        return smoothed

    def _print_summary(self, total_frames):
        s = self._stats
        print(f"\n[SOLVER] ═══════════════════════════════════════════════════")
        print(f"[SOLVER] SUMMARY")
        print(f"[SOLVER] ───────────────────────────────────────────────────")
        print(f"[SOLVER]   Frames total:       {total_frames}")
        print(f"[SOLVER]   Frames solved:       {s['frames_solved']}")
        print(f"[SOLVER]   Frames skipped:      {s['frames_skipped']}")
        print(f"[SOLVER]   Angle clamps total:  {s['angle_clamps']}")
        if s['angle_clamp_details']:
            for joint, count in sorted(s['angle_clamp_details'].items()):
                pct = count / max(1, s['frames_solved']) * 100
                print(f"[SOLVER]     {joint}: {count} ({pct:.1f}%)")
        if s['length_deltas']:
            mean_delta = np.mean(s['length_deltas'])
            print(f"[SOLVER]   Max bone correction: {s['max_length_delta']:.4f}m")
            print(f"[SOLVER]   Mean worst-per-frame: {mean_delta:.4f}m")
        lc = s.get('smooth_alpha_lowconf_frames', 0)
        jm = s.get('smooth_alpha_jump_frames', 0)
        if lc or jm:
            print(f"[SOLVER]   Low-conf smoothing: {lc} bone-frames")
            print(f"[SOLVER]   Jump-mode smoothing: {jm} bone-frames")
        print(f"[SOLVER] ═══════════════════════════════════════════════════\n")

    def get_metadata(self):
        return {
            'enabled': True,
            'bone_lengths': dict(self.bone_lengths),
            'angle_clamps_total': self._stats['angle_clamps'],
            'max_length_correction_m': round(self._stats['max_length_delta'], 4),
            'frames_solved': self._stats['frames_solved'],
            'frames_skipped': self._stats['frames_skipped'],
            'direction_smooth_alpha': self.direction_smooth_alpha,
            'smooth_alpha_lowconf_frames': self._stats.get('smooth_alpha_lowconf_frames', 0),
            'smooth_alpha_jump_frames': self._stats.get('smooth_alpha_jump_frames', 0),
        }
