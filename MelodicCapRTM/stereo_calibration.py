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
            'floor_z_offset': float(self.floor_z_offset),
        }

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

            if ca is not None and cb is not None and len(ca) >= 4 and len(cb) >= 4:
                all_corners_a.append(ca)
                all_ids_a.append(ia)
                all_corners_b.append(cb)
                all_ids_b.append(ib)
                print(f"  Frame {i+1}: OK ({len(ca)} corners)")
            elif ca is not None and cb is not None:
                print(f"  Frame {i+1}: Skipped (too few corners: A={len(ca)}, B={len(cb)})")
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

        # The board is on the floor, so the board's Z=0 plane IS the floor.
        # tvec gives translation from camera to board origin.
        # We need the floor height in our stereo coordinate system.
        R_cam, _ = cv2.Rodrigues(rvec)
        # Board origin in camera coords
        board_origin_cam = tvec.flatten()
        # Board normal in camera coords (board Z-axis)
        board_normal_cam = R_cam[:, 2]

        # Convert to Blender coords: Z_blender = -Y_opencv + offset
        # The floor height in blender coords is -board_origin_cam[1]
        # (since board is at floor level)
        floor_z_cv = board_origin_cam[1]  # OpenCV Y (downward)
        self.floor_z_offset = floor_z_cv  # This negates the -Y in triangulate

        return True, f"Floor set via Camera {cam_label} solvePnP (offset: {self.floor_z_offset:.3f}m, {len(single_corners)} corners)"

    def reset_filters(self):
        """Reset all Kalman filters (call at start of new take)."""
        for idx in self.filters:
            for f in self.filters[idx]:
                f.reset()
