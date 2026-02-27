import numpy as np
import cv2
import json

class TriangulationEngine:
    def __init__(self, calibration_path=None):
        self.P1 = None
        self.P2 = None
        self.K1 = None
        self.K2 = None
        self.D1 = None
        self.D2 = None
        self.R1 = None
        self.R2 = None
        self.is_calibrated = False
        self.image_size = (1280, 720)
        
        # Kalman Filters for 33 landmarks
        self.filters = {}  # idx -> [filter_x, filter_y, filter_z]
        
        # Floor calibration
        self.floor_offset_z = 0.0
        self.floor_normal = None
        self.floor_point = None
        
        if calibration_path:
            self.load_calibration(calibration_path)

    def _init_filter(self, idx):
        """Initialize Kalman filters for a landmark"""
        class SimpleKalman:
            def __init__(self, q=1e-5, r=1e-2):
                self.q = q  # process noise covariance
                self.r = r  # measurement noise covariance
                self.x = 0  # value
                self.p = 1  # estimation error covariance
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
            
            def reset(self):
                self.initialized = False
                self.x = 0
                self.p = 1

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
            self.R1 = np.array(data.get('R1', np.eye(3)))
            self.R2 = np.array(data.get('R2', np.eye(3)))
            
            # Image size
            if 'image_size' in data:
                self.image_size = tuple(data['image_size'])
            
            self.is_calibrated = True
            print(f"Loaded calibration from {path}")
            print(f"  Image size: {self.image_size}")
            print(f"  Baseline: {data.get('baseline_meters', 'unknown')}m")
            
        except Exception as e:
            print(f"Failed to load calibration: {e}")
            self.is_calibrated = False

    def undistort_and_rectify(self, pts, camera='left'):
        """
        Undistort and rectify 2D points for triangulation.
        
        Args:
            pts: Nx2 array of pixel coordinates
            camera: 'left' or 'right'
            
        Returns:
            Nx2 array of rectified coordinates
        """
        if camera == 'left':
            K, D, R, P = self.K1, self.D1, self.R1, self.P1
        else:
            K, D, R, P = self.K2, self.D2, self.R2, self.P2
        
        pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        
        # Undistort and rectify in one step
        # cv2.undistortPoints with R and P parameters does both
        rectified = cv2.undistortPoints(pts, K, D, R=R, P=P)
        
        return rectified.reshape(-1, 2)

    def triangulate_points(self, pts1, pts2, already_rectified=False):
        """
        Triangulate 3D points from stereo correspondences.
        
        Args:
            pts1, pts2: Nx1x2 or Nx2 arrays in pixels (unless already_rectified)
            already_rectified: If True, skip undistortion/rectification
            
        Returns:
            Nx3 array in Blender Space (+Z Up, +Y Forward)
        """
        if not self.is_calibrated:
            return None
        
        pts1 = np.array(pts1, dtype=np.float32).reshape(-1, 2)
        pts2 = np.array(pts2, dtype=np.float32).reshape(-1, 2)
        
        if not already_rectified:
            pts1 = self.undistort_and_rectify(pts1, 'left')
            pts2 = self.undistort_and_rectify(pts2, 'right')
        
        # Triangulate
        points_4d = cv2.triangulatePoints(
            self.P1, self.P2, 
            pts1.T, pts2.T
        )
        
        # Convert from homogeneous coordinates
        cv_pts = (points_4d[:3] / points_4d[3]).T
        
        # Transform to Blender Space: CV(x, y, z) -> Blender(x, z, -y)
        blender_pts = np.zeros_like(cv_pts)
        blender_pts[:, 0] = cv_pts[:, 0]      # X is Right
        blender_pts[:, 1] = cv_pts[:, 2]      # Y is Forward (CV Depth)
        blender_pts[:, 2] = -cv_pts[:, 1]     # Z is Up (CV -Vertical)
        
        return blender_pts

    def triangulate_pose(self, landmarks_a, landmarks_b, smooth=True):
        """
        Triangulate all pose landmarks from MediaPipe detections.
        
        Args:
            landmarks_a, landmarks_b: MediaPipe landmark objects
            smooth: Apply Kalman filtering
            
        Returns:
            Dictionary of landmark_idx -> [X, Y, Z] (Blender Space)
        """
        if not self.is_calibrated:
            return None
        
        w, h = self.image_size
        points_3d = {}
        
        for i in range(33):
            lm_a = landmarks_a.landmark[i]
            lm_b = landmarks_b.landmark[i]
            
            # Skip low-confidence detections
            if lm_a.visibility < 0.3 or lm_b.visibility < 0.3:
                continue
            
            # Convert normalized coords to pixels
            pt1 = np.array([[lm_a.x * w, lm_a.y * h]], dtype=np.float32)
            pt2 = np.array([[lm_b.x * w, lm_b.y * h]], dtype=np.float32)
            
            # Undistort and rectify
            pt1_rect = self.undistort_and_rectify(pt1, 'left')
            pt2_rect = self.undistort_and_rectify(pt2, 'right')
            
            # Triangulate single point
            point_4d = cv2.triangulatePoints(
                self.P1, self.P2, 
                pt1_rect.T, pt2_rect.T
            )
            cv_pt = (point_4d[:3] / point_4d[3]).flatten()
            
            # CV(x, y, z) -> Blender(x, z, -y)
            blender_pt = [cv_pt[0], cv_pt[2], -cv_pt[1]]
            
            # Apply floor offset
            blender_pt[2] += self.floor_offset_z
            
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
        Triangulate hand landmarks from MediaPipe detections.
        
        Args:
            hands_a, hands_b: Lists of MediaPipe hand landmark objects
            
        Returns:
            List of Nx3 arrays (one per matched hand pair)
        """
        if not self.is_calibrated or not hands_a or not hands_b:
            return None
        
        w, h = self.image_size
        hands_3d = []
        
        # Simple matching: pair by index (could improve with position matching)
        for h_a, h_b in zip(hands_a, hands_b):
            pts_a = []
            pts_b = []
            
            for i in range(21):
                lm_a = h_a.landmark[i]
                lm_b = h_b.landmark[i]
                pts_a.append([lm_a.x * w, lm_a.y * h])
                pts_b.append([lm_b.x * w, lm_b.y * h])
            
            pts_a = np.array(pts_a, dtype=np.float32)
            pts_b = np.array(pts_b, dtype=np.float32)
            
            h_3d = self.triangulate_points(pts_a, pts_b)
            
            # Apply floor offset
            if h_3d is not None:
                h_3d[:, 2] += self.floor_offset_z
                hands_3d.append(h_3d.tolist())
        
        return hands_3d if hands_3d else None

    def set_floor_plane(self, floor_normal, floor_point):
        """
        Set the ground plane for Z-alignment.
        
        Args:
            floor_normal: [nx, ny, nz] unit normal vector
            floor_point: [x, y, z] point on the floor
        """
        self.floor_normal = np.array(floor_normal)
        self.floor_point = np.array(floor_point)
        
        # Calculate offset to bring floor to Z=0
        # In Blender space, Z is up
        self.floor_offset_z = -self.floor_point[2]
        print(f"Floor calibration set: Z-Offset = {self.floor_offset_z:.4f}m")

    def apply_floor_offset(self, point_3d):
        """
        Apply floor offset to a single point.
        
        Args:
            point_3d: [x, y, z] or list
            
        Returns:
            Adjusted point with floor at Z=0
        """
        if isinstance(point_3d, list):
            point_3d = point_3d.copy()
            point_3d[2] += self.floor_offset_z
            return point_3d
        else:
            result = point_3d.copy()
            result[2] += self.floor_offset_z
            return result

    def reset_filters(self):
        """Reset all Kalman filters (call when starting new take)"""
        for idx in self.filters:
            for f in self.filters[idx]:
                f.reset()
        print("Kalman filters reset")
