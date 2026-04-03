import numpy as np
import cv2
import json

class TriangulationEngine:
    def __init__(self, calibration_path=None):
        self.P1 = None
        self.P2 = None
        self.is_calibrated = False
        self.image_size = (1280, 720)

        # Kalman Filters for 33 landmarks
        self.filters = {} # idx -> [filter_x, filter_y, filter_z]

        # Floor calibration
        self.floor_offset_z = 0.0

        if calibration_path:
            self.load_calibration(calibration_path)

    def _init_filter(self, idx):
        # Tuned Kalman: q/r ratio ~0.1 gives responsive tracking without excessive jitter
        # Previous q=1e-5, r=1e-2 (ratio 0.001) was far too sluggish
        class SimpleKalman:
            def __init__(self, q=1e-2, r=1e-1):
                self.q = q # process noise covariance
                self.r = r # measurement noise covariance
                self.x = 0 # value
                self.p = 1 # estimation error covariance
                self.initialized = False

            def update(self, measurement):
                if not self.initialized:
                    self.x = measurement
                    self.initialized = True
                    return self.x

                # Prediction
                self.p = self.p + self.q
                # Measurement update
                k = self.p / (self.p + self.r)
                self.x = self.x + k * (measurement - self.x)
                self.p = (1 - k) * self.p
                return self.x

        self.filters[idx] = [SimpleKalman() for _ in range(3)]

    def load_calibration(self, path):
        """Load stereo calibration from JSON file"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            # Projection matrices (rectified)
            self.P1 = np.array(data['P1'])
            self.P2 = np.array(data['P2'])

            # Intrinsic matrices
            self.K1 = np.array(data['K1'])
            self.K2 = np.array(data['K2'])

            # Distortion coefficients
            self.D1 = np.array(data['D1'])
            self.D2 = np.array(data['D2'])

            # Rectification transforms
            self.R1 = np.array(data.get('R1', np.eye(3).tolist()))
            self.R2 = np.array(data.get('R2', np.eye(3).tolist()))

            # Extrinsics (for re-rectification)
            if 'R' in data:
                self.R = np.array(data['R'])
            if 'T' in data:
                self.T = np.array(data['T'])

            # Image size from calibration (don't hardcode!)
            if 'image_size' in data:
                self.image_size = tuple(data['image_size'])

            # Floor offset
            self.floor_offset_z = data.get('floor_z_offset', 0.0)
            self.floor_calibrated = data.get('floor_calibrated', False)

            self.is_calibrated = True
            print(f"Loaded calibration from {path}")
            print(f"  Image size: {self.image_size}")
            print(f"  Floor offset: {self.floor_offset_z:.3f}m (calibrated={self.floor_calibrated})")

        except Exception as e:
            print(f"Failed to load calibration: {e}")
            self.is_calibrated = False

    def undistort_and_rectify(self, pts, camera='left'):
        """
        Undistort and rectify 2D points for triangulation.
        """
        if camera == 'left':
            K, D, R, P = self.K1, self.D1, self.R1, self.P1
        else:
            K, D, R, P = self.K2, self.D2, self.R2, self.P2
        
        pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        
        # Undistort and rectify in one step
        rectified = cv2.undistortPoints(pts, K, D, R=R, P=P)
        
        return rectified.reshape(-1, 2)

    def triangulate_points(self, pts1, pts2):
        """
        pts1, pts2: numpy arrays of shape (N, 1, 2) in pixels
        returns: Nx3 array in Blender Space (+Z Up, +Y Forward)
        """
        if not self.is_calibrated: return None
        
        pts1 = self.undistort_and_rectify(pts1, 'left')
        pts2 = self.undistort_and_rectify(pts2, 'right')
        
        points_4d = cv2.triangulatePoints(self.P1, self.P2, pts1.T, pts2.T)
        cv_pts = (points_4d[:3] / points_4d[3]).T
        
        # Transform to Blender Space: CV(x, y, z) -> Blender(x, z, -y)
        blender_pts = np.zeros_like(cv_pts)
        blender_pts[:, 0] = cv_pts[:, 0]     # X is Right
        blender_pts[:, 1] = cv_pts[:, 2]     # Y is Forward (CV Depth)
        blender_pts[:, 2] = -cv_pts[:, 1]    # Z is Up (CV -Vertical)
        return blender_pts

    def triangulate_pose(self, landmarks_a, landmarks_b, smooth=True):
        """
        landmarks_a, landmarks_b: MediaPipe landmark objects
        returns: Dictionary of landmark_idx -> [X, Y, Z] (Blender Space, meters)
        """
        if not self.is_calibrated:
            return None

        w, h = self.image_size
        points_3d = {}
        for i in range(33):
            lm_a = landmarks_a.landmark[i]
            lm_b = landmarks_b.landmark[i]

            # Skip low-confidence landmarks
            if lm_a.visibility < 0.3 or lm_b.visibility < 0.3:
                continue

            # Convert normalized (0-1) to pixel coordinates using actual image size
            pt1 = np.array([[lm_a.x * w, lm_a.y * h]], dtype=np.float32)
            pt2 = np.array([[lm_b.x * w, lm_b.y * h]], dtype=np.float32)

            # Undistort and rectify BEFORE triangulation
            pt1_rect = self.undistort_and_rectify(pt1, 'left')
            pt2_rect = self.undistort_and_rectify(pt2, 'right')

            point_4d = cv2.triangulatePoints(self.P1, self.P2, pt1_rect.T, pt2_rect.T)
            cv_pt = (point_4d[:3] / point_4d[3]).flatten()

            # CV(x, y, z) -> Blender(x, z, -y) + floor offset
            blender_pt = [cv_pt[0], cv_pt[2], -cv_pt[1] + self.floor_offset_z]

            if smooth:
                if i not in self.filters:
                    self._init_filter(i)

                smoothed_pt = [
                    self.filters[i][0].update(blender_pt[0]),
                    self.filters[i][1].update(blender_pt[1]),
                    self.filters[i][2].update(blender_pt[2])
                ]
                points_3d[i] = smoothed_pt
            else:
                points_3d[i] = blender_pt

        return points_3d
    def triangulate_hands(self, hands_a, hands_b):
        """
        Triangulates multiple hand landmarks (2D normalized -> 3D Blender Space)
        """
        if not self.is_calibrated or not hands_a or not hands_b:
            return None

        w, h = self.image_size
        hands_3d = []
        # Multi-hand matching (simplistic: match index 0 to 0, 1 to 1)
        for h_a, h_b in zip(hands_a, hands_b):
            pts_a, pts_b = [], []
            for i in range(21):
                pts_a.append([h_a.landmark[i].x * w, h_a.landmark[i].y * h])
                pts_b.append([h_b.landmark[i].x * w, h_b.landmark[i].y * h])

            pts_a = np.array(pts_a).reshape(-1, 1, 2)
            pts_b = np.array(pts_b).reshape(-1, 1, 2)

            h_3d_res = self.triangulate_points(pts_a, pts_b)
            hands_3d.append(h_3d_res.tolist())

        return hands_3d

    def set_floor_plane(self, floor_normal, floor_point):
        """Sets the ground plane for Z-clamping and alignment"""
        self.floor_normal = np.array(floor_normal)
        self.floor_point = np.array(floor_point)
        # Calculate offset to bring floor to Z=0
        # floor_point[2] is the Z coordinate of the floor in Blender space
        # We want floor at Z=0, so offset = -floor_z
        self.floor_offset_z = -self.floor_point[2]
        self.floor_calibrated = True
        print(f"Floor calibration set: Z-Offset = {self.floor_offset_z:.4f}m")

    def calibrate_extrinsic_from_samples(self, samples, board):
        """
        Refines extrinsic calibration (R, T) using multiple ChArUco samples.
        samples: list of (frame1, frame2)
        board: cv2.aruco.CharucoBoard
        """
        all_obj_pts = []
        all_img_pts1 = []
        all_img_pts2 = []
        
        detector = cv2.aruco.CharucoDetector(board)
        
        for f1, f2 in samples:
            # Convert to gray if needed
            gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY) if len(f1.shape) == 3 else f1
            gray2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY) if len(f2.shape) == 3 else f2
            
            res1 = detector.detectBoard(gray1)
            res2 = detector.detectBoard(gray2)
            
            if res1[0] is not None and res2[0] is not None:
                # Match IDs
                ids1 = res1[1].flatten()
                ids2 = res2[1].flatten()
                common_ids = np.intersect1d(ids1, ids2)
                
                if len(common_ids) >= 8:
                    # Get 3D board corners for matched IDs
                    # CharucoBoard.getChessboardCorners() is what we need
                    # But for version compatibility, we'll use board.getChessboardCorners()
                    obj_pts = board.getChessboardCorners()[common_ids]
                    
                    # Get 2D image corners for matched IDs
                    img_pts1 = []
                    img_pts2 = []
                    for cid in common_ids:
                        idx1 = np.where(ids1 == cid)[0][0]
                        idx2 = np.where(ids2 == cid)[0][0]
                        img_pts1.append(res1[0][idx1])
                        img_pts2.append(res2[0][idx2])
                    
                    all_obj_pts.append(obj_pts.astype(np.float32))
                    all_img_pts1.append(np.array(img_pts1).astype(np.float32))
                    all_img_pts2.append(np.array(img_pts2).astype(np.float32))

        if len(all_obj_pts) < 1:
            return False, "Not enough valid shared samples (need at least 1 with 8+ corners)"

        # Perform stereo calibration refinement
        # Note: We keep Intrinsic (K, D) fixed
        flags = cv2.CALIB_FIX_INTRINSIC
        
        # We need an initial R and T if possible, or just identity
        R_init = getattr(self, 'R', np.eye(3))
        T_init = getattr(self, 'T', np.array([[-1.7], [0], [0]])) # Baseline heuristic if missing
        
        ret, self.K1, self.D1, self.K2, self.D2, self.R, self.T, E, F = cv2.stereoCalibrate(
            all_obj_pts, all_img_pts1, all_img_pts2,
            self.K1, self.D1, self.K2, self.D2,
            self.image_size, R=R_init, T=T_init, flags=flags
        )
        
        # Rectify again with new R, T
        self.R1, self.R2, self.P1, self.P2, Q, _, _ = cv2.stereoRectify(
            self.K1, self.D1, self.K2, self.D2,
            self.image_size, self.R, self.T
        )
        
        self.is_calibrated = True
        return True, f"Calibration refined! RMS Error: {ret:.4f}"

    def save_calibration(self, path):
        """Save current calibration to JSON"""
        data = {
            'K1': self.K1.tolist(), 'D1': self.D1.tolist(), 'P1': self.P1.tolist(), 'R1': self.R1.tolist(),
            'K2': self.K2.tolist(), 'D2': self.D2.tolist(), 'P2': self.P2.tolist(), 'R2': self.R2.tolist(),
            'R': self.R.tolist(), 'T': self.T.tolist(),
            'image_size': [self.image_size[0], self.image_size[1]],
            'floor_z_offset': float(self.floor_offset_z),
            'floor_calibrated': getattr(self, 'floor_calibrated', False),
            'baseline_meters': float(np.linalg.norm(self.T)) if hasattr(self, 'T') else 0.0,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Calibration saved to {path}")
