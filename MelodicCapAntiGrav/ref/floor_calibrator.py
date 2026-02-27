"""
Floor Calibrator for MelodicCap Studio

Uses ChArUco board detection to establish ground plane.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict


class FloorCalibrator:
    """
    Detects ChArUco board on floor to establish world coordinate system.
    
    The board's position defines:
    - World origin (center of board)
    - Ground plane (Z=0)
    - Forward direction (board's Y axis = world Y)
    """
    
    def __init__(self, board_config: dict = None):
        """
        Initialize floor calibrator.
        
        Args:
            board_config: Dict with 'squares_x', 'squares_y', 'square_length', 'marker_length'
        """
        # Default to 4x3 board with 40mm squares (iPad display board)
        if board_config is None:
            board_config = {
                'squares_x': 4,
                'squares_y': 3,
                'square_length': 0.040,  # 40mm
                'marker_length': 0.030   # 30mm
            }
        
        self.board_config = board_config
        
        # Create ArUco dictionary and board
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.charuco_board = cv2.aruco.CharucoBoard(
            (board_config['squares_x'], board_config['squares_y']),
            board_config['square_length'],
            board_config['marker_length'],
            self.aruco_dict
        )
        
        # Detection parameters
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        
        self.aruco_detector = cv2.aruco.ArucoDetector(
            self.aruco_dict,
            self.detector_params
        )
        
        # Results
        self.floor_normal = None
        self.floor_point = None
        self.floor_rotation = None

    def detect_board(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect ChArUco board in image.
        
        Args:
            image: BGR image
            
        Returns:
            (corners, ids) or (None, None) if detection failed
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect ArUco markers
        marker_corners, marker_ids, _ = self.aruco_detector.detectMarkers(gray)
        
        if marker_ids is None or len(marker_ids) < 4:
            return None, None
        
        # Interpolate ChArUco corners
        num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            self.charuco_board
        )
        
        if num_corners < 4:
            return None, None
        
        return charuco_corners, charuco_ids

    def detect_floor_from_frames(
        self, 
        frame_a: np.ndarray, 
        frame_b: np.ndarray,
        triangulation_engine
    ) -> Tuple[Optional[Dict], str]:
        """
        Detect floor plane from stereo camera frames.
        
        Args:
            frame_a: BGR image from camera A (left)
            frame_b: BGR image from camera B (right)
            triangulation_engine: TriangulationEngine instance for 3D reconstruction
            
        Returns:
            (result_dict, message) - result_dict has 'normal' and 'point', or None on failure
        """
        # Detect board in both cameras
        corners_a, ids_a = self.detect_board(frame_a)
        corners_b, ids_b = self.detect_board(frame_b)
        
        if corners_a is None:
            return None, "Board not detected in Camera A"
        
        if corners_b is None:
            return None, "Board not detected in Camera B"
        
        # Find common corner IDs
        ids_a_set = set(ids_a.flatten())
        ids_b_set = set(ids_b.flatten())
        common_ids = ids_a_set & ids_b_set
        
        if len(common_ids) < 4:
            return None, f"Not enough common corners ({len(common_ids)})"
        
        # Build corresponding point arrays
        pts_a = []
        pts_b = []
        
        ids_a_flat = ids_a.flatten()
        ids_b_flat = ids_b.flatten()
        
        for cid in sorted(common_ids):
            idx_a = np.where(ids_a_flat == cid)[0][0]
            idx_b = np.where(ids_b_flat == cid)[0][0]
            
            pts_a.append(corners_a[idx_a].flatten())
            pts_b.append(corners_b[idx_b].flatten())
        
        pts_a = np.array(pts_a, dtype=np.float32)
        pts_b = np.array(pts_b, dtype=np.float32)
        
        # Triangulate floor points
        if not triangulation_engine.is_calibrated:
            return None, "Triangulation engine not calibrated"
        
        floor_points_3d = triangulation_engine.triangulate_points(pts_a, pts_b)
        
        if floor_points_3d is None or len(floor_points_3d) < 3:
            return None, "Triangulation failed"
        
        # Fit plane to floor points using SVD
        centroid = np.mean(floor_points_3d, axis=0)
        centered = floor_points_3d - centroid
        
        _, _, vh = np.linalg.svd(centered)
        
        # Normal is the last row of V (smallest singular value direction)
        normal = vh[2, :]
        
        # Ensure normal points up (positive Z in Blender space)
        if normal[2] < 0:
            normal = -normal
        
        # Store results
        self.floor_normal = normal
        self.floor_point = centroid
        
        # Calculate some statistics
        # Distance of points to plane
        distances = np.abs(np.dot(floor_points_3d - centroid, normal))
        mean_dist = np.mean(distances)
        max_dist = np.max(distances)
        
        result = {
            'normal': normal.tolist(),
            'point': centroid.tolist(),
            'fit_error_mean': float(mean_dist),
            'fit_error_max': float(max_dist),
            'num_points': len(floor_points_3d)
        }
        
        message = f"OK ({len(floor_points_3d)} pts, err={mean_dist*1000:.1f}mm)"
        
        return result, message

    def draw_detection(self, image: np.ndarray, 
                       corners: np.ndarray = None, 
                       ids: np.ndarray = None) -> np.ndarray:
        """
        Draw detected board corners on image.
        
        Args:
            image: Image to draw on (will be copied)
            corners: ChArUco corners (or auto-detect if None)
            ids: Corner IDs
            
        Returns:
            Image with detection drawn
        """
        output = image.copy()
        
        if corners is None:
            corners, ids = self.detect_board(image)
        
        if corners is not None:
            cv2.aruco.drawDetectedCornersCharuco(
                output, corners, ids,
                cornerColor=(0, 255, 0)
            )
            
            # Draw coordinate axes at first corner
            if len(corners) > 0:
                pt = tuple(corners[0].flatten().astype(int))
                cv2.circle(output, pt, 10, (0, 0, 255), -1)
                cv2.putText(output, "FLOOR", (pt[0]+15, pt[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(output, "NO BOARD DETECTED", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return output


def create_floor_board_image(
    squares_x: int = 4,
    squares_y: int = 3,
    square_size_px: int = 200,
    margin_px: int = 50
) -> np.ndarray:
    """
    Create a printable ChArUco board image for floor calibration.
    
    Args:
        squares_x, squares_y: Board dimensions
        square_size_px: Size of each square in pixels
        margin_px: White margin around board
        
    Returns:
        Board image (grayscale)
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        0.04,  # Square size (doesn't matter for image)
        0.03,  # Marker size
        aruco_dict
    )
    
    board_size = (squares_x * square_size_px + 2 * margin_px,
                  squares_y * square_size_px + 2 * margin_px)
    
    board_image = board.generateImage(board_size, marginSize=margin_px)
    
    return board_image


if __name__ == "__main__":
    # Generate printable board
    board_img = create_floor_board_image()
    cv2.imwrite("floor_calibration_board.png", board_img)
    print("Generated floor_calibration_board.png")
    print("Print at 100% scale and measure squares to verify size.")
