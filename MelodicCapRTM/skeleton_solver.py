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

VIRTUAL_BONE_NAMES = ['spine', 'hip_width']

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

DIRECTION_SMOOTH_ALPHA = 0.5
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
    c = np.cos(np.pi - target_rad)
    s = np.sin(np.pi - target_rad)
    new_child_dir = parent_dir * c + np.cross(axis, parent_dir) * s + axis * np.dot(axis, parent_dir) * (1 - c)
    new_child_dir = _safe_normalize(new_child_dir)

    corrected = joint_pos + new_child_dir * bone_length
    return corrected, angle, True


class SkeletonSolver:
    def __init__(self, n_cal_frames=30):
        self.n_cal_frames = n_cal_frames
        self.bone_lengths = {}
        self.calibrated = False
        self._stats = {
            'angle_clamps': 0,
            'angle_clamp_details': {},
            'max_length_delta': 0.0,
            'length_deltas': [],
            'frames_solved': 0,
            'frames_skipped': 0,
        }

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
        samples = {name: [] for _, _, name in BONE_DEFS}
        for vname in VIRTUAL_BONE_NAMES:
            samples[vname] = []

        good_frames = 0
        for frame in frames:
            lm = frame.get('landmarks_3d', {})
            if not lm or not self._all_body_present(lm):
                continue

            for parent_idx, child_idx, name in BONE_DEFS:
                p = self._get_point(lm, parent_idx)
                c = self._get_point(lm, child_idx)
                if p is not None and c is not None:
                    length = np.linalg.norm(c - p)
                    if 0.01 < length < 1.5:
                        samples[name].append(length)

            ls = self._get_point(lm, LM.LEFT_SHOULDER)
            rs = self._get_point(lm, LM.RIGHT_SHOULDER)
            lh = self._get_point(lm, LM.LEFT_HIP)
            rh = self._get_point(lm, LM.RIGHT_HIP)

            if ls is not None and rs is not None and lh is not None and rh is not None:
                hip_mid = (lh + rh) / 2
                shoulder_mid = (ls + rs) / 2
                spine_len = np.linalg.norm(shoulder_mid - hip_mid)
                if 0.1 < spine_len < 1.0:
                    samples['spine'].append(spine_len)

                hw = np.linalg.norm(lh - rh)
                if 0.05 < hw < 0.6:
                    samples['hip_width'].append(hw)

            good_frames += 1
            if good_frames >= self.n_cal_frames:
                break

        print(f"\n[SOLVER] ═══════════════════════════════════════════════════")
        print(f"[SOLVER] Bone length calibration ({good_frames} frames sampled)")
        print(f"[SOLVER] ───────────────────────────────────────────────────")

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

        # Step 3: shoulders (raw offsets from corrected shoulder_mid, no width enforcement)
        half_offset_l = ls - shoulder_mid_raw
        half_offset_r = rs - shoulder_mid_raw
        corrected[LM.LEFT_SHOULDER] = shoulder_mid + half_offset_l
        corrected[LM.RIGHT_SHOULDER] = shoulder_mid + half_offset_r

        # Step 3b: hip width (soft constraint — only correct if >20% deviation)
        if 'hip_width' in self.bone_lengths:
            hw_cal = self.bone_lengths['hip_width']
            raw_hw = np.linalg.norm(lh - rh)
            hw_ratio = raw_hw / hw_cal if hw_cal > 1e-6 else 1.0
            if abs(hw_ratio - 1.0) > 0.20:
                sign = 1.0 if hw_ratio > 1.0 else -1.0
                target_hw = hw_cal * (1.0 + 0.20 * sign)
                hip_axis = _safe_normalize(lh - rh)
                corrected[LM.LEFT_HIP] = hip_mid + hip_axis * (target_hw / 2)
                corrected[LM.RIGHT_HIP] = hip_mid - hip_axis * (target_hw / 2)

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
        """EMA on bone direction vectors, then reconstruct positions.

        Smooths joint angles without corrupting bone lengths.
        """
        n = len(all_solved_frames)
        if n < 3:
            return all_solved_frames

        all_chains = ARM_CHAINS + LEG_CHAINS
        alpha = DIRECTION_SMOOTH_ALPHA

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

        def _ema_smooth(dirs):
            smoothed = [None] * len(dirs)
            running = None
            for i, d in enumerate(dirs):
                if d is None:
                    smoothed[i] = running
                    continue
                if running is None:
                    running = d.copy()
                else:
                    running = alpha * d + (1 - alpha) * running
                    running = _safe_normalize(running)
                smoothed[i] = running.copy()
            return smoothed

        smoothed_dirs = {}
        for key, dirs in bone_directions.items():
            smoothed_dirs[key] = _ema_smooth(dirs)

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

        print(f"\n[SOLVER] Applying temporal direction smoothing (α={DIRECTION_SMOOTH_ALPHA})...")
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
        print(f"[SOLVER] ═══════════════════════════════════════════════════\n")

    def get_metadata(self):
        return {
            'enabled': True,
            'bone_lengths': dict(self.bone_lengths),
            'angle_clamps_total': self._stats['angle_clamps'],
            'max_length_correction_m': round(self._stats['max_length_delta'], 4),
            'frames_solved': self._stats['frames_solved'],
            'frames_skipped': self._stats['frames_skipped'],
        }
