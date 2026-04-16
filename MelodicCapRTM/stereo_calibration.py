"""
Stereo Camera Calibration & Triangulation
==========================================
Extracted from MelodicCapFresh/melodic_capture_v1.py.
Same proven logic — ChArUco calibration, stereo rectification, triangulation.

Now backend-agnostic: accepts generic {idx: (x, y, score)} dicts instead
of MediaPipe landmark objects.
"""

import cv2
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from kalman import SimpleKalman


# Reasonable limits for a human body in a room
OUTLIER_MAX_DISTANCE = 3.0      # meters from body centroid
OUTLIER_MAX_VELOCITY = 0.3      # meters per frame (~6.3 m/s at 21fps — fast arm swing)
# Tighter limits for feet — they move slower than hands and jitter is most visible
OUTLIER_MAX_VELOCITY_FEET = 0.15 # meters per frame (~3.15 m/s at 21fps — fast step)
FOOT_INDICES = {15, 16, 17, 18, 19, 20, 21, 22}  # ankles + foot keypoints
BONE_PAIRS_FOR_SCALE = [
    # (parent_idx, child_idx, expected_length_meters, tolerance_ratio)
    # COCO-WholeBody indices
    # Tolerance = max deviation from learned length. 0.25 = 25% — reasonable for
    # stereo noise. Lower limbs are more stable, upper limbs noisier.
    (5, 7, 0.28, 0.25),   # left shoulder → left elbow
    (7, 9, 0.25, 0.25),   # left elbow → left wrist
    (6, 8, 0.28, 0.25),   # right shoulder → right elbow
    (8, 10, 0.25, 0.25),  # right elbow → right wrist
    (11, 13, 0.42, 0.20), # left hip → left knee
    (13, 15, 0.40, 0.20), # left knee → left ankle
    (12, 14, 0.42, 0.20), # right hip → right knee
    (14, 16, 0.40, 0.20), # right knee → right ankle
    (5, 6, 0.36, 0.20),   # shoulder width
    (11, 12, 0.28, 0.20), # hip width
]


# =============================================================================
# Per-camera intrinsics file I/O (stand-alone, no stereo pair required)
# =============================================================================
#
# Splitting intrinsics capture from extrinsics capture is what every
# professional CV calibration tool does (MATLAB Camera Calibrator, Kalibr,
# OpenCV's own sample calibrator). Intrinsics (K, D) are stable across
# sessions because they depend on the lens and sensor, not camera pose.
# Stereo extrinsics (R, T) change every time you move the cameras.
#
# The "one board, both cameras must see it" flow forces the intrinsics
# solver to use only frames that happen to be in the overlap volume, which
# is a narrow cluster in a small room. That's how you end up with a 9.9%
# Camera B K-matrix mismatch between AB and BC calibrations.

def save_intrinsics(filepath, K, D, rms, image_size, coverage_map=None,
                    n_frames=0, camera_label=None):
    """Save per-camera intrinsics to a stand-alone JSON file.

    This is deliberately separate from StereoCalibration.save() — an
    intrinsics file has NO R/T and can be produced from a single camera.

    Args:
        filepath: Path to write (e.g. calibration/intrinsics_a.json)
        K: 3x3 camera matrix
        D: 1xN distortion coefficients
        rms: reprojection RMS from calibrateCameraCharuco
        image_size: (width, height) the calibration was captured at
        coverage_map: optional 3x3 int array of per-cell capture counts
        n_frames: total number of capture frames used
        camera_label: optional human tag ('A', 'B', 'C')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'version': 'melodiccap_intrinsics_v1',
        'timestamp': datetime.now().isoformat(),
        'camera_label': camera_label,
        'image_size': list(image_size),
        'K': np.asarray(K).tolist(),
        'D': np.asarray(D).tolist(),
        'rms': float(rms),
        'n_frames': int(n_frames),
    }
    if coverage_map is not None:
        data['coverage_map'] = np.asarray(coverage_map, dtype=int).tolist()

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_intrinsics(filepath):
    """Load per-camera intrinsics. Returns dict or None if missing/invalid.

    Returned dict has: K (3x3 np.ndarray), D (1xN np.ndarray),
    rms (float), image_size (list), n_frames (int), coverage_map (optional).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return None
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return {
            'K': np.array(data['K'], dtype=np.float64),
            'D': np.array(data['D'], dtype=np.float64),
            'rms': float(data.get('rms', 0.0)),
            'image_size': list(data.get('image_size', [0, 0])),
            'n_frames': int(data.get('n_frames', 0)),
            'coverage_map': data.get('coverage_map'),
            'camera_label': data.get('camera_label'),
        }
    except Exception as e:
        print(f"[WARN] Failed to load intrinsics from {filepath}: {e}")
        return None


class StereoCalibration:
    """Handles stereo camera calibration and 3D triangulation."""

    def __init__(self, config):
        """
        Args:
            config: Config object with calibration parameters
        """
        self.config = config
        self.is_calibrated = False

        # Calibration matrices
        self.K1 = None  # Camera A intrinsic
        self.K2 = None  # Camera B intrinsic
        self.D1 = None  # Camera A distortion
        self.D2 = None  # Camera B distortion
        self.R = None   # Rotation between cameras
        self.T = None   # Translation between cameras
        self.R1 = None  # Rectification rotation A
        self.R2 = None  # Rectification rotation B
        self.P1 = None  # Projection matrix A
        self.P2 = None  # Projection matrix B

        # Floor calibration
        self.floor_z_offset = 0.0

        # Reprojection error from last triangulate_pose() call
        # (mean_err_cam_a, mean_err_cam_b, max_err) in pixels
        self.last_reproj_error = (0.0, 0.0, 0.0)

        # Kalman filters for each landmark
        self.filters = {}

        # Previous frame 3D points for velocity-based outlier rejection
        self._prev_points = {}

        # Calibration image size (for runtime validation)
        self._cal_image_size = None

        # Bone-length calibration (computed from first N good frames)
        self._bone_lengths = {}        # {(parent, child): median_length}
        self._bone_samples = {}        # {(parent, child): [lengths...]}
        self._bone_cal_frames = 0
        self._bone_cal_target = 30     # Collect this many frames then lock

        # ChArUco board setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(config.ARUCO_DICT)
        self.charuco_board = cv2.aruco.CharucoBoard(
            (config.CHARUCO_SQUARES_X, config.CHARUCO_SQUARES_Y),
            config.CHARUCO_SQUARE_SIZE,
            config.CHARUCO_MARKER_SIZE,
            self.aruco_dict
        )
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_detector = cv2.aruco.ArucoDetector(
            self.aruco_dict, self.detector_params
        )

    def validate_frame_resolution(self, frame_a, frame_b):
        """
        Check that actual frame resolution matches calibration.
        Returns (ok, message). Call this once after cameras init.
        """
        issues = []
        expected = (self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT)

        for label, frame in [("A", frame_a), ("B", frame_b)]:
            h, w = frame.shape[:2]
            actual = (w, h)
            if actual != expected:
                issues.append(
                    f"Camera {label}: getting {w}x{h}, expected {expected[0]}x{expected[1]}"
                )

        if self._cal_image_size and self._cal_image_size != list(expected):
            issues.append(
                f"Calibration was done at {self._cal_image_size[0]}x{self._cal_image_size[1]}, "
                f"config says {expected[0]}x{expected[1]}"
            )

        if issues:
            return False, "RESOLUTION MISMATCH:\n  " + "\n  ".join(issues)
        return True, "OK"

    def load(self, filepath):
        """Load calibration from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.K1 = np.array(data['K1'])
            self.K2 = np.array(data['K2'])
            self.D1 = np.array(data['D1'])
            self.D2 = np.array(data['D2'])
            self.R = np.array(data['R']) if data.get('R') is not None else np.eye(3)
            self.T = np.array(data['T']) if data.get('T') is not None else np.zeros((3, 1))
            self.R1 = np.array(data.get('R1', np.eye(3).tolist()))
            self.R2 = np.array(data.get('R2', np.eye(3).tolist()))
            # P1/P2 may be absent in very old calibration files. Use a zero
            # 3x4 identity-ish fallback that will produce wrong results but
            # at least won't crash — the user will recalibrate anyway.
            _P_default = np.hstack([np.eye(3), np.zeros((3, 1))]).tolist()
            self.P1 = np.array(data.get('P1', _P_default))
            self.P2 = np.array(data.get('P2', _P_default))
            self.floor_z_offset = data.get('floor_z_offset', 0.0)

            # Restore quality metrics. save() writes these but load() used to
            # drop them, forcing offline_processor.py and chain_calibration.py
            # to re-read the JSON themselves to get RMS/baseline.
            self._stereo_rms = data.get('stereo_rms')
            self._cam_a_rms = data.get('cam_a_rms')
            self._cam_b_rms = data.get('cam_b_rms')
            self.baseline_meters = data.get('baseline_meters')
            self.derivation_method = data.get('derivation_method')

            self._cal_image_size = data.get('image_size', None)
            self.is_calibrated = True
            print(f"[OK] Loaded calibration from {filepath}")
            print(f"     Baseline: {data.get('baseline_meters', 'unknown')}m")
            print(f"     Calibrated at: {self._cal_image_size[0]}x{self._cal_image_size[1]}" if self._cal_image_size else "     Calibrated at: unknown resolution")
            if self.floor_z_offset != 0.0:
                print(f"     Floor offset: {self.floor_z_offset:.3f}m")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to load calibration: {e}")
            return False

    def save(self, filepath):
        """Save calibration to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'version': 'melodiccap_rtm_v1',
            'timestamp': datetime.now().isoformat(),
            'image_size': [self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT],
            'K1': self.K1.tolist(),
            'K2': self.K2.tolist(),
            'D1': self.D1.tolist(),
            'D2': self.D2.tolist(),
            'R': self.R.tolist() if self.R is not None else None,
            'T': self.T.tolist() if self.T is not None else None,
            'R1': self.R1.tolist(),
            'R2': self.R2.tolist(),
            'P1': self.P1.tolist(),
            'P2': self.P2.tolist(),
            'baseline_meters': float(np.linalg.norm(self.T)) if self.T is not None else 0,
            'stereo_rms': getattr(self, '_stereo_rms', None),
            'cam_a_rms': getattr(self, '_cam_a_rms', None),
            'cam_b_rms': getattr(self, '_cam_b_rms', None),
            'floor_z_offset': float(self.floor_z_offset),
        }
        # Preserve chain-derived flag when re-saving an already-loaded calibration.
        derivation = getattr(self, 'derivation_method', None)
        if derivation:
            data['derivation_method'] = derivation

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[OK] Saved calibration to {filepath}")

    def detect_charuco(self, image):
        """Detect ChArUco board in image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is None or len(ids) < 2:
            return None, None

        num, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, self.charuco_board
        )

        if num < 3:
            return None, None

        return charuco_corners, charuco_ids

    def _check_spatial_coverage(self, all_corners, img_size, label):
        """
        Check if board positions cover enough of the frame.
        Returns (ok, coverage_pct, message).
        """
        w, h = img_size
        # Divide frame into a 3x3 grid, check which cells have board corners
        grid = np.zeros((3, 3), dtype=bool)
        for corners in all_corners:
            for pt in corners.reshape(-1, 2):
                col = min(int(pt[0] / (w / 3)), 2)
                row = min(int(pt[1] / (h / 3)), 2)
                grid[row, col] = True

        coverage = grid.sum() / 9.0
        if coverage < 0.33:
            return False, coverage, (
                f"Camera {label}: Board only covers {coverage:.0%} of frame. "
                f"Move the board to more positions — top, bottom, left, right, center."
            )
        return True, coverage, f"Camera {label}: {coverage:.0%} spatial coverage"

    def calibrate_stereo(self, frames_a, frames_b,
                         K1_fixed=None, D1_fixed=None,
                         K2_fixed=None, D2_fixed=None):
        """
        Calibrate stereo cameras from synchronized frame pairs.

        Args:
            frames_a: List of frames from camera A
            frames_b: List of frames from camera B
            K1_fixed, D1_fixed: pre-computed intrinsics for Camera A. If
                provided, monocular estimation for Camera A is skipped and
                these matrices are passed straight into stereoCalibrate with
                CALIB_FIX_INTRINSIC. This is the fast path you get when
                per-camera intrinsics were already captured via the `I` key.
            K2_fixed, D2_fixed: same, for Camera B.

        When fixed intrinsics are supplied for both cameras, the required
        per-camera spatial coverage gate is relaxed (intrinsics are already
        trusted, so extrinsics only need ~8 shared views).

        Returns:
            True if successful
        """
        extrinsics_only = (K1_fixed is not None and D1_fixed is not None
                           and K2_fixed is not None and D2_fixed is not None)

        if extrinsics_only:
            print("\n[CALIBRATING STEREO EXTRINSICS ONLY "
                  "— using saved per-camera intrinsics]")
        else:
            print("\n[CALIBRATING STEREO CAMERAS]")

        all_corners_a = []
        all_corners_b = []
        all_ids_a = []
        all_ids_b = []

        MIN_CORNERS = 6   # per camera — lowered for 5x7 board (24 corners total)

        for i, (fa, fb) in enumerate(zip(frames_a, frames_b)):
            ca, ia = self.detect_charuco(fa)
            cb, ib = self.detect_charuco(fb)

            if ca is not None and cb is not None and len(ca) >= MIN_CORNERS and len(cb) >= MIN_CORNERS:
                # Also check common corners — need enough shared points for stereo
                common = set(ia.flatten()) & set(ib.flatten())
                if len(common) >= 8:
                    all_corners_a.append(ca)
                    all_ids_a.append(ia)
                    all_corners_b.append(cb)
                    all_ids_b.append(ib)
                    print(f"  Frame {i+1}: OK ({len(ca)}/{len(cb)} corners, {len(common)} common)")
                else:
                    print(f"  Frame {i+1}: Skipped (only {len(common)} common corners, need 8+)")
            elif ca is not None and cb is not None:
                print(f"  Frame {i+1}: Skipped (too few corners: A={len(ca)}, B={len(cb)}, need {MIN_CORNERS}+)")
            else:
                detected = []
                if ca is not None: detected.append(f"A={len(ca)}")
                if cb is not None: detected.append(f"B={len(cb)}")
                print(f"  Frame {i+1}: Board not detected in {'both cameras' if not detected else 'one camera'}")

        # Extrinsics-only needs ~8 shared views; legacy joint path still needs 10.
        min_required = 8 if extrinsics_only else 10
        if len(all_corners_a) < min_required:
            print(f"[ERROR] Not enough valid frames ({len(all_corners_a)}). "
                  f"Need at least {min_required}.")
            return False

        img_size = (self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT)

        # Spatial coverage gate. Only matters when we're solving intrinsics
        # here — if the caller already passed fixed K/D (from per-camera
        # intrinsics capture), we just need a handful of matched views for R/T.
        if not extrinsics_only:
            ok_a, cov_a, msg_a = self._check_spatial_coverage(all_corners_a, img_size, "A")
            ok_b, cov_b, msg_b = self._check_spatial_coverage(all_corners_b, img_size, "B")
            print(f"\n  {msg_a}")
            print(f"  {msg_b}")

            if not ok_a or not ok_b:
                print(f"\n[ERROR] Insufficient spatial coverage!")
                print(f"  The board needs to appear in different parts of the frame.")
                print(f"  Move it: left/right, up/down, near/far, and tilt it.")
                print(f"  This is especially important for Camera B (DroidCam).")
                return False
        else:
            print(f"\n  [OK] Skipping spatial coverage check "
                  f"(intrinsics are pre-computed; only need {min_required}+ shared views)")

        if extrinsics_only:
            # Use the caller-supplied matrices as both the seed and the fixed
            # intrinsics for stereoCalibrate. CALIB_FIX_INTRINSIC at line ~400
            # makes stereoCalibrate only solve for R and T.
            K1 = np.array(K1_fixed, dtype=np.float64)
            D1 = np.array(D1_fixed, dtype=np.float64)
            K2 = np.array(K2_fixed, dtype=np.float64)
            D2 = np.array(D2_fixed, dtype=np.float64)
            ret_a = 0.0  # unknown — came from a separate intrinsics session
            ret_b = 0.0
            print("  Using pre-computed Camera A intrinsics")
            print("  Using pre-computed Camera B intrinsics")
        else:
            # Calibrate camera A
            print("\n  Calibrating Camera A...")
            try:
                ret_a, K1, D1, _, _ = cv2.aruco.calibrateCameraCharuco(
                    all_corners_a, all_ids_a, self.charuco_board, img_size, None, None
                )
                print(f"    RMS Error: {ret_a:.4f}")
            except cv2.error as e:
                print(f"    [FAILED] Camera A calibration error: {e}")
                print(f"    Board positions likely too similar. Move board more.")
                return False

            # Calibrate camera B
            print("  Calibrating Camera B...")
            try:
                ret_b, K2, D2, _, _ = cv2.aruco.calibrateCameraCharuco(
                    all_corners_b, all_ids_b, self.charuco_board, img_size, None, None
                )
                print(f"    RMS Error: {ret_b:.4f}")
            except cv2.error as e:
                print(f"    [FAILED] Camera B calibration error: {e}")
                print(f"    Board positions from DroidCam likely too similar.")
                print(f"    Make sure the board appears at different positions and angles")
                print(f"    from DroidCam's perspective — not just the Sony's.")
                return False

        # Build stereo correspondences
        obj_points = []
        img_points_a = []
        img_points_b = []

        for ca, ia, cb, ib in zip(all_corners_a, all_ids_a, all_corners_b, all_ids_b):
            ids_a_set = set(ia.flatten())
            ids_b_set = set(ib.flatten())
            common = ids_a_set & ids_b_set

            if len(common) < 8:
                continue

            obj_pts = []
            img_a = []
            img_b = []

            for cid in sorted(common):
                obj_pt = self.charuco_board.getChessboardCorners()[cid]
                obj_pts.append(obj_pt)

                idx_a = np.where(ia.flatten() == cid)[0][0]
                idx_b = np.where(ib.flatten() == cid)[0][0]
                img_a.append(ca[idx_a].flatten())
                img_b.append(cb[idx_b].flatten())

            obj_points.append(np.array(obj_pts, dtype=np.float32))
            img_points_a.append(np.array(img_a, dtype=np.float32))
            img_points_b.append(np.array(img_b, dtype=np.float32))

        # ── Iterative Stereo Calibration ─────────────────────────────
        # Run stereo calibration, compute per-frame reprojection errors,
        # drop worst frames, repeat until RMS stabilizes or we run out of frames.
        # This is similar to what Pose2Sim does with its calibration refinement.
        MAX_ITERATIONS = 5
        cur_obj = list(obj_points)
        cur_img_a = list(img_points_a)
        cur_img_b = list(img_points_b)

        for iteration in range(MAX_ITERATIONS):
            label = f"(pass {iteration + 1})" if iteration > 0 else ""
            print(f"\n  Running stereo calibration {label}... ({len(cur_obj)} frames)")

            if len(cur_obj) < 8:
                print(f"  [ERROR] Too few frames remaining ({len(cur_obj)}). Cannot calibrate.")
                return False

            ret_stereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
                cur_obj, cur_img_a, cur_img_b,
                K1, D1, K2, D2, img_size,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-7),
                flags=cv2.CALIB_FIX_INTRINSIC
            )
            print(f"    Stereo RMS: {ret_stereo:.4f}")

            # Compute per-frame epipolar error — measures stereo consistency
            # Points in camera A should lie on epipolar lines in camera B
            per_frame_errors = []
            for i in range(len(cur_obj)):
                pts_a = cur_img_a[i].reshape(-1, 2)
                pts_b = cur_img_b[i].reshape(-1, 2)
                n_pts = len(pts_a)
                if n_pts < 4:
                    per_frame_errors.append((i, 999.0))
                    continue
                # Undistort points
                pts_a_ud = cv2.undistortPoints(pts_a.reshape(-1, 1, 2), K1, D1, P=K1).reshape(-1, 2)
                pts_b_ud = cv2.undistortPoints(pts_b.reshape(-1, 1, 2), K2, D2, P=K2).reshape(-1, 2)
                # Compute epipolar lines: l = F * p_a (in homogeneous coords)
                pts_a_h = np.hstack([pts_a_ud, np.ones((n_pts, 1))])
                pts_b_h = np.hstack([pts_b_ud, np.ones((n_pts, 1))])
                # Distance from pts_b to epipolar lines from pts_a
                lines_b = (F @ pts_a_h.T).T  # epipolar lines in image B
                dists_b = np.abs(np.sum(lines_b * pts_b_h, axis=1)) / np.linalg.norm(lines_b[:, :2], axis=1)
                # Distance from pts_a to epipolar lines from pts_b
                lines_a = (F.T @ pts_b_h.T).T
                dists_a = np.abs(np.sum(lines_a * pts_a_h, axis=1)) / np.linalg.norm(lines_a[:, :2], axis=1)
                avg_epi_err = float(np.mean(dists_a) + np.mean(dists_b)) / 2.0
                per_frame_errors.append((i, avg_epi_err))

            # Sort by error, find frames above threshold
            per_frame_errors.sort(key=lambda x: x[1], reverse=True)
            worst_error = per_frame_errors[0][1]

            # If already good enough or no bad frames, stop iterating
            if worst_error <= 1.5 or ret_stereo < 0.4:
                break

            # Drop frames with high epipolar error (>2px) or top 20% worst
            threshold = max(2.0, np.percentile([e for _, e in per_frame_errors], 80))
            keep_indices = [i for i, err in per_frame_errors if err <= threshold]

            if len(keep_indices) < 8:
                # Can't drop more, keep what we have
                break

            dropped = len(cur_obj) - len(keep_indices)
            print(f"    Dropping {dropped} worst frames (threshold: {threshold:.1f}px, worst: {worst_error:.1f}px)")

            cur_obj = [cur_obj[i] for i in sorted(keep_indices)]
            cur_img_a = [cur_img_a[i] for i in sorted(keep_indices)]
            cur_img_b = [cur_img_b[i] for i in sorted(keep_indices)]

        baseline = np.linalg.norm(T)

        # ── Quality Gate ──────────────────────────────────────────────
        quality_issues = []

        # Only check per-camera intrinsic RMS when we actually computed it.
        # Extrinsics-only path trusts the caller-supplied intrinsics.
        if not extrinsics_only:
            if ret_a > 1.0:
                quality_issues.append(f"Camera A intrinsic RMS {ret_a:.3f} > 1.0 (bad)")
            elif ret_a > 0.5:
                quality_issues.append(f"Camera A intrinsic RMS {ret_a:.3f} > 0.5 (mediocre)")

            if ret_b > 1.0:
                quality_issues.append(f"Camera B intrinsic RMS {ret_b:.3f} > 1.0 (bad)")
            elif ret_b > 0.5:
                quality_issues.append(f"Camera B intrinsic RMS {ret_b:.3f} > 0.5 (mediocre)")

        if ret_stereo > 1.5:
            quality_issues.append(f"Stereo RMS {ret_stereo:.3f} > 1.5 (BAD — will produce stretched/jittery results)")
        elif ret_stereo > 1.0:
            quality_issues.append(f"Stereo RMS {ret_stereo:.3f} > 1.0 (high — may have some jitter)")

        if baseline < 0.05:
            quality_issues.append(f"Baseline {baseline:.3f}m is very small — poor depth resolution")
        elif baseline > 2.0:
            quality_issues.append(f"Baseline {baseline:.3f}m is very large — may lose close subjects")

        if quality_issues:
            print(f"\n  ⚠ CALIBRATION QUALITY WARNINGS:")
            for issue in quality_issues:
                print(f"    • {issue}")

        # Reject truly terrible calibrations
        if ret_stereo > 3.0:
            print(f"\n  [REJECTED] Stereo RMS {ret_stereo:.3f} is too high.")
            print(f"  Tips: Use fewer, higher-quality frames. Hold board steady.")
            print(f"  Move slowly. Ensure both cameras see the board clearly.")
            return False

        if quality_issues and ret_stereo > 1.5:
            print(f"\n  [WARNING] Calibration saved but quality is poor (RMS {ret_stereo:.3f}).")
            print(f"  Consider recalibrating with 20-30 slow, deliberate frames.")

        # Stereo rectification
        print("  Computing rectification...")
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K1, D1, K2, D2, img_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )

        # Store results
        self.K1, self.D1 = K1, D1
        self.K2, self.D2 = K2, D2
        self.R, self.T = R, T
        self.R1, self.R2 = R1, R2
        self.P1, self.P2 = P1, P2
        self.is_calibrated = True
        self._cal_image_size = list(img_size)
        self._stereo_rms = float(ret_stereo)
        # In extrinsics-only mode the per-camera RMS was never recomputed
        # (we trusted the caller's intrinsics). Store None so offline tools
        # don't mis-read a placeholder zero as "excellent".
        self._cam_a_rms = None if extrinsics_only else float(ret_a)
        self._cam_b_rms = None if extrinsics_only else float(ret_b)

        print(f"\n[OK] Stereo calibration complete!")
        print(f"     Baseline: {baseline:.3f}m")
        print(f"     Stereo RMS: {ret_stereo:.4f}")
        if ret_stereo < 0.5:
            print(f"     Quality: EXCELLENT")
        elif ret_stereo < 1.0:
            print(f"     Quality: GOOD")
        elif ret_stereo < 1.5:
            print(f"     Quality: ACCEPTABLE")
        else:
            print(f"     Quality: POOR — recalibrate for better results")

        return True

    def triangulate(self, pts_a, pts_b):
        """
        Triangulate 3D points from stereo pixel correspondences.

        Args:
            pts_a: Nx2 array of pixel coordinates from camera A
            pts_b: Nx2 array of pixel coordinates from camera B

        Returns:
            Nx3 array in Blender space (X=right, Y=forward, Z=up)
        """
        if not self.is_calibrated:
            return None

        pts_a = np.array(pts_a, dtype=np.float32).reshape(-1, 1, 2)
        pts_b = np.array(pts_b, dtype=np.float32).reshape(-1, 1, 2)

        # Undistort and rectify
        pts_a_rect = cv2.undistortPoints(pts_a, self.K1, self.D1, R=self.R1, P=self.P1)
        pts_b_rect = cv2.undistortPoints(pts_b, self.K2, self.D2, R=self.R2, P=self.P2)

        # Triangulate
        points_4d = cv2.triangulatePoints(
            self.P1, self.P2,
            pts_a_rect.reshape(-1, 2).T,
            pts_b_rect.reshape(-1, 2).T
        )

        # Convert from homogeneous
        cv_points = (points_4d[:3] / points_4d[3]).T

        # Convert OpenCV to Blender coordinates
        # OpenCV: X=right, Y=down, Z=forward
        # Blender: X=right, Y=forward, Z=up
        blender_points = np.zeros_like(cv_points)
        blender_points[:, 0] = cv_points[:, 0]      # X stays X
        blender_points[:, 1] = cv_points[:, 2]      # Y = CV's Z (depth/forward)
        blender_points[:, 2] = -cv_points[:, 1]     # Z = -CV's Y (up)

        # Apply floor offset
        blender_points[:, 2] += self.floor_z_offset

        return blender_points

    def compute_reprojection_error(self, pts_2d_a, pts_2d_b, pts_3d_cv):
        """
        Compute reprojection error: project 3D points back to both cameras
        and measure pixel distance from original detections.

        Points from triangulatePoints(P1, P2) are in the rectified coordinate
        system. To reproject back to original pixel space:
        1. Un-rectify: transform from rectified frame back to original camera A
        2. Project to camera A with K1/D1
        3. Transform to camera B with R/T, project with K2/D2

        Args:
            pts_2d_a: Nx2 original pixel coords from camera A
            pts_2d_b: Nx2 original pixel coords from camera B
            pts_3d_cv: Nx3 points in rectified OpenCV space

        Returns:
            (mean_error_a, mean_error_b, max_error) in pixels
        """
        if len(pts_3d_cv) == 0:
            return 0.0, 0.0, 0.0

        pts_3d_rect = np.array(pts_3d_cv, dtype=np.float64)

        # Un-rectify: transform from rectified frame to original camera A frame
        # Rectified = R1 @ Original_A, so Original_A = R1^T @ Rectified
        R1_inv = self.R1.T
        pts_3d_origA = (R1_inv @ pts_3d_rect.T).T

        # Project to camera A (identity R, zero T — points are now in camera A frame)
        rvec_a = np.zeros(3)
        tvec_a = np.zeros(3)
        proj_a, _ = cv2.projectPoints(pts_3d_origA, rvec_a, tvec_a, self.K1, self.D1)
        proj_a = proj_a.reshape(-1, 2)

        # Project to camera B: transform from camera A to camera B
        rvec_b, _ = cv2.Rodrigues(self.R)
        proj_b, _ = cv2.projectPoints(pts_3d_origA, rvec_b, self.T, self.K2, self.D2)
        proj_b = proj_b.reshape(-1, 2)

        err_a = np.linalg.norm(proj_a - np.array(pts_2d_a).reshape(-1, 2), axis=1)
        err_b = np.linalg.norm(proj_b - np.array(pts_2d_b).reshape(-1, 2), axis=1)

        return float(np.mean(err_a)), float(np.mean(err_b)), float(max(np.max(err_a), np.max(err_b)))

    def triangulate_pose(self, detections_a, detections_b, smooth=True):
        """
        Triangulate 3D pose from two sets of 2D detections.

        Args:
            detections_a: dict of {keypoint_idx: (pixel_x, pixel_y, confidence)}
            detections_b: dict of {keypoint_idx: (pixel_x, pixel_y, confidence)}
            smooth: whether to apply Kalman filtering

        Returns:
            dict of {keypoint_idx: [x, y, z]} in Blender space
        """
        if not self.is_calibrated:
            return None

        min_conf = self.config.MIN_KEYPOINT_CONFIDENCE
        points_3d = {}

        # Find keypoints visible in both cameras
        common_indices = set(detections_a.keys()) & set(detections_b.keys())

        # Optionally skip face (23-90) and hand (91-132) keypoints
        # to reduce triangulation cost without losing body accuracy
        skip_face_hands = getattr(self.config, 'SKIP_FACE_HANDS', False)
        if skip_face_hands:
            common_indices = {idx for idx in common_indices if idx <= 22}

        # First pass: triangulate all valid keypoints
        raw_points = {}
        _reproj_2d_a = []
        _reproj_2d_b = []
        _reproj_3d_cv = []
        for idx in common_indices:
            px_a, py_a, conf_a = detections_a[idx]
            px_b, py_b, conf_b = detections_b[idx]

            # Skip low confidence
            if conf_a < min_conf or conf_b < min_conf:
                continue

            # Triangulate single point
            result = self.triangulate([[px_a, py_a]], [[px_b, py_b]])
            if result is not None:
                raw_points[idx] = result[0].tolist()
                # Collect for reprojection error (convert Blender back to CV)
                bpt = result[0]
                cv_pt = [bpt[0], -bpt[2] + self.floor_z_offset, bpt[1]]  # reverse Blender→CV
                _reproj_2d_a.append([px_a, py_a])
                _reproj_2d_b.append([px_b, py_b])
                _reproj_3d_cv.append(cv_pt)

        # ── Outlier Rejection ─────────────────────────────────────
        # Compute body centroid from core keypoints (hips + shoulders)
        core_indices = {5, 6, 11, 12}  # COCO shoulders + hips
        core_pts = [raw_points[i] for i in core_indices if i in raw_points]
        if core_pts:
            centroid = np.mean(core_pts, axis=0)
        elif raw_points:
            centroid = np.mean(list(raw_points.values()), axis=0)
        else:
            centroid = np.array([0.0, 0.0, 1.0])

        for idx, pt_3d in raw_points.items():
            pt = np.array(pt_3d)

            # Reject points too far from body centroid
            dist = np.linalg.norm(pt - centroid)
            if dist > OUTLIER_MAX_DISTANCE:
                # Use previous frame's value if available
                if idx in self._prev_points:
                    pt_3d = self._prev_points[idx]
                    pt = np.array(pt_3d)  # update pt so velocity check below uses the fallback
                else:
                    continue  # Skip entirely if no history

            # Reject impossible frame-to-frame velocity
            # Feet get a tighter threshold — they're noisier and jitter is most visible
            if idx in self._prev_points:
                prev = np.array(self._prev_points[idx])
                velocity = np.linalg.norm(pt - prev)
                max_vel = OUTLIER_MAX_VELOCITY_FEET if idx in FOOT_INDICES else OUTLIER_MAX_VELOCITY
                if velocity > max_vel:
                    # Use previous value instead of teleporting
                    pt_3d = self._prev_points[idx]

            # Apply Kalman smoothing
            # Feet get higher measurement noise (trust prediction more, reject jitter)
            if smooth:
                if idx not in self.filters:
                    q = self.config.KALMAN_PROCESS_NOISE     # 1e-4
                    r = self.config.KALMAN_MEASUREMENT_NOISE  # 1e-2
                    if idx in FOOT_INDICES:
                        r = r * 5  # 5e-2: feet are noisier, trust measurement less
                    self.filters[idx] = [
                        SimpleKalman(q, r)
                        for _ in range(3)
                    ]

                pt_3d = [
                    self.filters[idx][0].update(pt_3d[0]),
                    self.filters[idx][1].update(pt_3d[1]),
                    self.filters[idx][2].update(pt_3d[2]),
                ]

            points_3d[idx] = pt_3d

        # ── Bone-Length Constraint ─────────────────────────────────
        # Like FreeMoCap/Pose2Sim: learn body proportions from first
        # N frames, then enforce them to prevent stretching.
        if self._bone_cal_frames < self._bone_cal_target:
            # Learning phase: collect bone length samples
            for parent, child, _, _ in BONE_PAIRS_FOR_SCALE:
                if parent in points_3d and child in points_3d:
                    length = np.linalg.norm(
                        np.array(points_3d[parent]) - np.array(points_3d[child])
                    )
                    if 0.01 < length < 1.5:  # Only plausible lengths
                        key = (parent, child)
                        if key not in self._bone_samples:
                            self._bone_samples[key] = []
                        self._bone_samples[key].append(length)
            self._bone_cal_frames += 1

            if self._bone_cal_frames == self._bone_cal_target:
                # Lock in median bone lengths
                for key, samples in self._bone_samples.items():
                    if len(samples) >= 10:
                        self._bone_lengths[key] = float(np.median(samples))
                if self._bone_lengths:
                    print(f"[BONE CAL] Learned {len(self._bone_lengths)} bone lengths from {self._bone_cal_target} frames")
                    for (p, c), length in sorted(self._bone_lengths.items()):
                        print(f"    {p}->{c}: {length:.3f}m")
        else:
            # Enforcement phase: clamp bones that are too long/short
            for parent, child, _, tolerance in BONE_PAIRS_FOR_SCALE:
                key = (parent, child)
                if key not in self._bone_lengths:
                    continue
                if parent not in points_3d or child not in points_3d:
                    continue

                p_pt = np.array(points_3d[parent])
                c_pt = np.array(points_3d[child])
                current_len = np.linalg.norm(c_pt - p_pt)
                expected_len = self._bone_lengths[key]

                if current_len < 0.001:
                    continue

                ratio = current_len / expected_len
                if abs(ratio - 1.0) > tolerance:
                    # Clamp: move child point to correct distance along same direction
                    direction = (c_pt - p_pt) / current_len
                    corrected = p_pt + direction * expected_len
                    points_3d[child] = corrected.tolist()

        # Store for next frame's velocity check
        self._prev_points = dict(points_3d)

        # Compute reprojection error for diagnostics
        if _reproj_3d_cv:
            self.last_reproj_error = self.compute_reprojection_error(
                _reproj_2d_a, _reproj_2d_b, _reproj_3d_cv
            )
        else:
            self.last_reproj_error = (0.0, 0.0, 0.0)

        return points_3d

    def detect_floor_debug(self, frame_a, frame_b):
        """
        Detect ChArUco board in both frames and return detailed debug info.

        Returns:
            dict with keys: corners_a, ids_a, corners_b, ids_b,
                            markers_a_count, markers_b_count,
                            charuco_a_count, charuco_b_count
        """
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers first (raw markers, before ChArUco interpolation)
        markers_a, marker_ids_a, _ = self.aruco_detector.detectMarkers(gray_a)
        markers_b, marker_ids_b, _ = self.aruco_detector.detectMarkers(gray_b)

        m_count_a = 0 if marker_ids_a is None else len(marker_ids_a)
        m_count_b = 0 if marker_ids_b is None else len(marker_ids_b)

        # Interpolate ChArUco corners
        corners_a, ids_a = None, None
        corners_b, ids_b = None, None

        if marker_ids_a is not None and len(marker_ids_a) >= 2:
            num_a, corners_a, ids_a = cv2.aruco.interpolateCornersCharuco(
                markers_a, marker_ids_a, gray_a, self.charuco_board
            )
            if num_a < 3:
                corners_a, ids_a = None, None

        if marker_ids_b is not None and len(marker_ids_b) >= 2:
            num_b, corners_b, ids_b = cv2.aruco.interpolateCornersCharuco(
                markers_b, marker_ids_b, gray_b, self.charuco_board
            )
            if num_b < 3:
                corners_b, ids_b = None, None

        return {
            'corners_a': corners_a, 'ids_a': ids_a,
            'corners_b': corners_b, 'ids_b': ids_b,
            'markers_a': markers_a, 'marker_ids_a': marker_ids_a,
            'markers_b': markers_b, 'marker_ids_b': marker_ids_b,
            'markers_a_count': m_count_a,
            'markers_b_count': m_count_b,
            'charuco_a_count': 0 if corners_a is None else len(corners_a),
            'charuco_b_count': 0 if corners_b is None else len(corners_b),
        }

    def draw_floor_debug(self, display_a, display_b, debug_info):
        """Draw floor calibration debug visuals on display frames."""
        # Draw detected ArUco markers
        if debug_info['marker_ids_a'] is not None:
            cv2.aruco.drawDetectedMarkers(display_a, debug_info['markers_a'], debug_info['marker_ids_a'])
        if debug_info['marker_ids_b'] is not None:
            cv2.aruco.drawDetectedMarkers(display_b, debug_info['markers_b'], debug_info['marker_ids_b'])

        # Draw ChArUco corners
        if debug_info['corners_a'] is not None:
            cv2.aruco.drawDetectedCornersCharuco(display_a, debug_info['corners_a'], debug_info['ids_a'])
        if debug_info['corners_b'] is not None:
            cv2.aruco.drawDetectedCornersCharuco(display_b, debug_info['corners_b'], debug_info['ids_b'])

        # Status text per camera
        ma = debug_info['markers_a_count']
        ca = debug_info['charuco_a_count']
        mb = debug_info['markers_b_count']
        cb = debug_info['charuco_b_count']

        color_a = (0, 255, 0) if ca >= 3 else (0, 0, 255)
        color_b = (0, 255, 0) if cb >= 3 else (0, 0, 255)

        cv2.putText(display_a, f"FLOOR CAL: {ma} markers, {ca} corners",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_a, 2)
        cv2.putText(display_b, f"FLOOR CAL: {mb} markers, {cb} corners",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_b, 2)

    def calibrate_floor(self, frame_a, frame_b):
        """
        Calibrate floor plane from ChArUco board placed on ground.
        Uses stereo triangulation if both cameras see the board,
        falls back to single-camera solvePnP if only one does.

        Returns:
            (success, message)
        """
        debug = self.detect_floor_debug(frame_a, frame_b)
        corners_a, ids_a = debug['corners_a'], debug['ids_a']
        corners_b, ids_b = debug['corners_b'], debug['ids_b']

        both_detected = corners_a is not None and corners_b is not None

        if both_detected:
            # Stereo triangulation path (best accuracy)
            common_ids = set(ids_a.flatten()) & set(ids_b.flatten())
            if len(common_ids) < 3:
                return False, f"Not enough common corners ({len(common_ids)})"

            pts_a = []
            pts_b = []
            ids_a_flat = ids_a.flatten()
            ids_b_flat = ids_b.flatten()

            for cid in sorted(common_ids):
                idx_a = np.where(ids_a_flat == cid)[0][0]
                idx_b = np.where(ids_b_flat == cid)[0][0]
                pts_a.append(corners_a[idx_a].flatten())
                pts_b.append(corners_b[idx_b].flatten())

            floor_points = self.triangulate(pts_a, pts_b)

            if floor_points is None:
                return False, "Triangulation failed"

            centroid = np.mean(floor_points, axis=0)
            centered = floor_points - centroid
            _, _, vh = np.linalg.svd(centered)
            normal = vh[2]
            if normal[2] < 0:
                normal = -normal

            self.floor_z_offset = -centroid[2]
            return True, f"Floor set via stereo (offset: {self.floor_z_offset:.3f}m, {len(common_ids)} corners)"

        # Single-camera fallback using solvePnP
        single_corners = corners_a if corners_a is not None else corners_b
        single_ids = ids_a if corners_a is not None else ids_b
        cam_label = "A" if corners_a is not None else "B"
        K = self.K1 if corners_a is not None else self.K2
        D = self.D1 if corners_a is not None else self.D2

        if single_corners is None:
            a_info = f"A: {debug['markers_a_count']} markers"
            b_info = f"B: {debug['markers_b_count']} markers"
            return False, f"Board not detected ({a_info}, {b_info}). Need 2+ markers per camera."

        if len(single_corners) < 4:
            return False, f"Only {len(single_corners)} corners in Camera {cam_label}, need 4+"

        # Get 3D object points for detected corners
        obj_pts = []
        img_pts = []
        board_corners = self.charuco_board.getChessboardCorners()
        for i, cid in enumerate(single_ids.flatten()):
            obj_pts.append(board_corners[cid])
            img_pts.append(single_corners[i].flatten())

        obj_pts = np.array(obj_pts, dtype=np.float32)
        img_pts = np.array(img_pts, dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D)
        if not success:
            return False, "solvePnP failed"

        # solvePnP gives board pose in THIS camera's coordinate frame.
        # But triangulate() outputs in the STEREO system's frame (camera A).
        # We must transform to camera A's frame first.
        R_board, _ = cv2.Rodrigues(rvec)
        board_origin_cam = tvec.flatten()

        if corners_a is not None:
            # This IS camera A — board_origin is already in stereo frame
            board_origin_stereo = board_origin_cam
        else:
            # This is camera B — transform to camera A's frame
            # OpenCV stereo: P_B = R @ P_A + T → so P_A = R^T @ (P_B - T)
            if self.R is not None and self.T is not None:
                board_origin_stereo = (self.R.T @ (board_origin_cam.reshape(3, 1) - self.T)).flatten()
            else:
                # No stereo extrinsics available, fall back to raw (will be approximate)
                board_origin_stereo = board_origin_cam

        # Convert to Blender coords: Z_blender = -Y_opencv
        # The floor is where the board is, so offset = -(-Y) = Y
        # (negate because triangulate() does Z = -Y_opencv, we want floor at Z=0)
        self.floor_z_offset = -(-board_origin_stereo[1])  # = board_origin_stereo[1]

        return True, f"Floor set via Camera {cam_label} solvePnP (offset: {self.floor_z_offset:.3f}m, {len(single_corners)} corners)"

    def calibrate_floor_from_ankles(self, detections_a, detections_b):
        """
        Auto-detect floor level from ankle keypoints.
        Stand still with feet flat on ground. Uses the lowest ankle Z
        as floor reference (like FreeMoCap does).

        Call this with several frames of standing data for best results.
        Returns (success, message).
        """
        if not self.is_calibrated:
            return False, "Not calibrated"

        # COCO-WholeBody ankle indices
        ANKLE_INDICES = [15, 16]  # left ankle, right ankle

        ankle_points = []
        for idx in ANKLE_INDICES:
            if idx in detections_a and idx in detections_b:
                px_a, py_a, conf_a = detections_a[idx]
                px_b, py_b, conf_b = detections_b[idx]

                if conf_a < self.config.MIN_KEYPOINT_CONFIDENCE or conf_b < self.config.MIN_KEYPOINT_CONFIDENCE:
                    continue

                result = self.triangulate([[px_a, py_a]], [[px_b, py_b]])
                if result is not None:
                    ankle_points.append(result[0])

        if not ankle_points:
            return False, "No ankle keypoints detected"

        # Floor is at the lowest ankle Z (before floor offset is applied,
        # but triangulate() already applies current offset — so the Z values
        # include whatever offset is currently set)
        ankle_z_values = [pt[2] for pt in ankle_points]
        min_z = min(ankle_z_values)

        # Adjust offset so ankles sit at ~0.05m (ankle bone is ~5cm above floor)
        adjustment = -(min_z - 0.05)
        self.floor_z_offset += adjustment

        return True, f"Floor set from ankles (adjustment: {adjustment:+.3f}m, new offset: {self.floor_z_offset:.3f}m)"

    def reset_filters(self):
        """Reset all Kalman filters and tracking state (call at start of new take)."""
        for idx in self.filters:
            for f in self.filters[idx]:
                f.reset()
        self._prev_points = {}
        self._bone_lengths = {}
        self._bone_samples = {}
        self._bone_cal_frames = 0
