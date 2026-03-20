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

        # Kalman filters for each landmark
        self.filters = {}

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

    def load(self, filepath):
        """Load calibration from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.K1 = np.array(data['K1'])
            self.K2 = np.array(data['K2'])
            self.D1 = np.array(data['D1'])
            self.D2 = np.array(data['D2'])
            self.R1 = np.array(data.get('R1', np.eye(3).tolist()))
            self.R2 = np.array(data.get('R2', np.eye(3).tolist()))
            self.P1 = np.array(data['P1'])
            self.P2 = np.array(data['P2'])
            self.floor_z_offset = data.get('floor_z_offset', 0.0)

            self.is_calibrated = True
            print(f"[OK] Loaded calibration from {filepath}")
            print(f"     Baseline: {data.get('baseline_meters', 'unknown')}m")
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
            'floor_z_offset': self.floor_z_offset,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[OK] Saved calibration to {filepath}")

    def detect_charuco(self, image):
        """Detect ChArUco board in image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is None or len(ids) < 4:
            return None, None

        num, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, self.charuco_board
        )

        if num < 4:
            return None, None

        return charuco_corners, charuco_ids

    def calibrate_stereo(self, frames_a, frames_b):
        """
        Calibrate stereo cameras from synchronized frame pairs.

        Args:
            frames_a: List of frames from camera A
            frames_b: List of frames from camera B

        Returns:
            True if successful
        """
        print("\n[CALIBRATING STEREO CAMERAS]")

        all_corners_a = []
        all_corners_b = []
        all_ids_a = []
        all_ids_b = []

        for i, (fa, fb) in enumerate(zip(frames_a, frames_b)):
            ca, ia = self.detect_charuco(fa)
            cb, ib = self.detect_charuco(fb)

            if ca is not None and cb is not None:
                all_corners_a.append(ca)
                all_ids_a.append(ia)
                all_corners_b.append(cb)
                all_ids_b.append(ib)
                print(f"  Frame {i+1}: OK ({len(ca)} corners)")
            else:
                print(f"  Frame {i+1}: Board not detected in both cameras")

        if len(all_corners_a) < 10:
            print(f"[ERROR] Not enough valid frames ({len(all_corners_a)}). Need at least 10.")
            return False

        img_size = (self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT)

        # Calibrate camera A
        print("\n  Calibrating Camera A...")
        ret_a, K1, D1, _, _ = cv2.aruco.calibrateCameraCharuco(
            all_corners_a, all_ids_a, self.charuco_board, img_size, None, None
        )
        print(f"    RMS Error: {ret_a:.4f}")

        # Calibrate camera B
        print("  Calibrating Camera B...")
        ret_b, K2, D2, _, _ = cv2.aruco.calibrateCameraCharuco(
            all_corners_b, all_ids_b, self.charuco_board, img_size, None, None
        )
        print(f"    RMS Error: {ret_b:.4f}")

        # Build stereo correspondences
        obj_points = []
        img_points_a = []
        img_points_b = []

        for ca, ia, cb, ib in zip(all_corners_a, all_ids_a, all_corners_b, all_ids_b):
            ids_a_set = set(ia.flatten())
            ids_b_set = set(ib.flatten())
            common = ids_a_set & ids_b_set

            if len(common) < 4:
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

        # Stereo calibration
        print("\n  Running stereo calibration...")
        ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            obj_points, img_points_a, img_points_b,
            K1, D1, K2, D2, img_size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
            flags=cv2.CALIB_FIX_INTRINSIC
        )
        print(f"    Stereo RMS Error: {ret_stereo:.4f}")

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

        baseline = np.linalg.norm(T)
        print(f"\n[OK] Stereo calibration complete!")
        print(f"     Baseline: {baseline:.3f}m")
        print(f"     Stereo RMS: {ret_stereo:.4f}")

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

        for idx in common_indices:
            px_a, py_a, conf_a = detections_a[idx]
            px_b, py_b, conf_b = detections_b[idx]

            # Skip low confidence
            if conf_a < min_conf or conf_b < min_conf:
                continue

            # Triangulate single point
            result = self.triangulate([[px_a, py_a]], [[px_b, py_b]])

            if result is None:
                continue

            pt_3d = result[0].tolist()

            # Apply Kalman smoothing
            if smooth:
                if idx not in self.filters:
                    self.filters[idx] = [
                        SimpleKalman(
                            self.config.KALMAN_PROCESS_NOISE,
                            self.config.KALMAN_MEASUREMENT_NOISE
                        )
                        for _ in range(3)
                    ]

                pt_3d = [
                    self.filters[idx][0].update(pt_3d[0]),
                    self.filters[idx][1].update(pt_3d[1]),
                    self.filters[idx][2].update(pt_3d[2]),
                ]

            points_3d[idx] = pt_3d

        return points_3d

    def calibrate_floor(self, frame_a, frame_b):
        """
        Calibrate floor plane from ChArUco board placed on ground.

        Returns:
            (success, message)
        """
        corners_a, ids_a = self.detect_charuco(frame_a)
        corners_b, ids_b = self.detect_charuco(frame_b)

        if corners_a is None or corners_b is None:
            return False, "Board not detected in both cameras"

        common_ids = set(ids_a.flatten()) & set(ids_b.flatten())
        if len(common_ids) < 4:
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

        # Fit plane using SVD
        centroid = np.mean(floor_points, axis=0)
        centered = floor_points - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2]

        if normal[2] < 0:
            normal = -normal

        self.floor_z_offset = -centroid[2]

        return True, f"Floor set (offset: {self.floor_z_offset:.3f}m)"

    def reset_filters(self):
        """Reset all Kalman filters (call at start of new take)."""
        for idx in self.filters:
            for f in self.filters[idx]:
                f.reset()
