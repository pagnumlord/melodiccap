import cv2
import numpy as np
import json
import os

class FloorCalibrator:
    """
    Detects the calibration board on the ground to establish Z=0
    and the forward direction (+Y).
    """
    def __init__(self, board_config):
        self.squares_x = board_config.get('squares_x', 4)
        self.squares_y = board_config.get('squares_y', 3)
        self.square_length = board_config.get('square_length', 0.040)
        self.marker_length = board_config.get('marker_length', 0.030)
        
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.board = cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), self.square_length, self.marker_length, self.dictionary)
        self.detector = cv2.aruco.CharucoDetector(self.board)

    def detect_floor_from_frames(self, frame_a, frame_b, triangulation_engine):
        """
        Calculates the floor plane from triangulated board corners
        """
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
        
        c_a, ids_a, _, _ = self.detector.detectBoard(gray_a)
        c_b, ids_b, _, _ = self.detector.detectBoard(gray_b)
        
        if c_a is None or c_b is None:
            return None, "Board not detected in both cameras"
            
        # Match IDs
        shared_ids = np.intersect1d(ids_a.flatten(), ids_b.flatten())
        if len(shared_ids) < 4:
            return None, f"Too few shared corners ({len(shared_ids)})"
            
        pts_a, pts_b = [], []
        for sid in shared_ids:
            idx_a = np.where(ids_a == sid)[0][0]
            idx_b = np.where(ids_b == sid)[0][0]
            pts_a.append(c_a[idx_a])
            pts_b.append(c_b[idx_b])
            
        pts_a = np.array(pts_a).reshape(-1, 1, 2)
        pts_b = np.array(pts_b).reshape(-1, 1, 2)
        
        # Triangulate corners
        corners_3d = triangulation_engine.triangulate_points(pts_a, pts_b)
        
        # Fit plane (ax + by + cz + d = 0)
        # Using SVD for plane fitting
        centroid = np.mean(corners_3d, axis=0)
        centered = corners_3d - centroid
        u, s, vh = np.linalg.svd(centered)
        normal = vh[2, :] # Normal is the last row of Vh
        
        # Ensure normal points "Up" (positive Z in Blender space)
        if normal[2] < 0:
            normal = -normal
            
        return {
            "normal": normal.tolist(),
            "point": centroid.tolist(),
            "corners_3d": corners_3d.tolist()
        }, "Floor Detected Successfully"
