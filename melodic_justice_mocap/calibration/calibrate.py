"""
Melodic Justice Motion Capture - Camera Calibration
===================================================
ChArUco board generation and stereo camera calibration.
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime
from typing import List, Tuple, Optional, Dict
from dataclasses import asdict
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from config import CharucoBoardConfig, CameraConfig, StereoConfig, MocapConfig
except ImportError:
    # Fallback for standalone use
    from dataclasses import dataclass, field
    
    @dataclass
    class CharucoBoardConfig:
        squares_x: int = 10
        squares_y: int = 5
        square_length_meters: float = 0.043
        marker_length_meters: float = 0.031
        dictionary_id: int = 0


# ============================================================================
# CHARUCO BOARD GENERATION
# ============================================================================

class CharucoBoardGenerator:
    """Generates and manages ChArUco calibration boards."""
    
    # ArUco dictionary types
    DICTIONARIES = {
        0: cv2.aruco.DICT_4X4_50,
        1: cv2.aruco.DICT_4X4_100,
        2: cv2.aruco.DICT_4X4_250,
        3: cv2.aruco.DICT_5X5_50,
        4: cv2.aruco.DICT_5X5_100,
        5: cv2.aruco.DICT_6X6_50,
    }
    
    def __init__(self, config: CharucoBoardConfig = None):
        self.config = config or CharucoBoardConfig()
        self._setup_board()
    
    def _setup_board(self):
        """Initialize the ChArUco board."""
        dict_type = self.DICTIONARIES.get(
            self.config.dictionary_id, 
            cv2.aruco.DICT_4X4_50
        )
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        
        self.charuco_board = cv2.aruco.CharucoBoard(
            (self.config.squares_x, self.config.squares_y),
            self.config.square_length_meters,
            self.config.marker_length_meters,
            self.aruco_dict
        )
        
        # Create detector
        self.detector_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(
            self.aruco_dict, 
            self.detector_params
        )
        self.charuco_detector = cv2.aruco.CharucoDetector(self.charuco_board)
    
    def generate_board_image(
        self, 
        output_path: str = "charuco_board.png",
        pixels_per_meter: int = 2000,
        margin_pixels: int = 50
    ) -> np.ndarray:
        """
        Generate a printable ChArUco board image.
        
        Args:
            output_path: Where to save the image
            pixels_per_meter: Resolution for printing
            margin_pixels: White border around the board
            
        Returns:
            Board image as numpy array
        """
        # Calculate image size
        board_width_px = int(
            self.config.squares_x * self.config.square_length_meters * pixels_per_meter
        )
        board_height_px = int(
            self.config.squares_y * self.config.square_length_meters * pixels_per_meter
        )
        
        # Generate the board
        board_image = self.charuco_board.generateImage(
            (board_width_px, board_height_px),
            marginSize=margin_pixels
        )
        
        # Save with print info
        if output_path:
            cv2.imwrite(output_path, board_image)
            
            # Generate info file
            info_path = output_path.replace('.png', '_info.txt')
            with open(info_path, 'w') as f:
                f.write("ChArUco Board Calibration Pattern\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Squares: {self.config.squares_x} x {self.config.squares_y}\n")
                f.write(f"Square size: {self.config.square_length_meters * 100:.1f} cm\n")
                f.write(f"Marker size: {self.config.marker_length_meters * 100:.1f} cm\n")
                f.write(f"Total size: {self.config.squares_x * self.config.square_length_meters * 100:.1f} x "
                       f"{self.config.squares_y * self.config.square_length_meters * 100:.1f} cm\n\n")
                f.write("IMPORTANT: Print at 100% scale!\n")
                f.write("Measure the squares to verify they match the sizes above.\n")
            
            print(f"Board saved to: {output_path}")
            print(f"Info saved to: {info_path}")
        
        return board_image
    
    def detect_board(
        self, 
        image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect ChArUco board in an image.
        
        Returns:
            Tuple of (charuco_corners, charuco_ids, aruco_corners)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect ArUco markers
        marker_corners, marker_ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        if marker_ids is None or len(marker_ids) < 4:
            return None, None, None
        
        # Interpolate ChArUco corners
        charuco_corners, charuco_ids, _, _ = self.charuco_detector.detectBoard(
            gray, 
            markerCorners=marker_corners,
            markerIds=marker_ids
        )
        
        return charuco_corners, charuco_ids, marker_corners
    
    def draw_detected_board(
        self, 
        image: np.ndarray,
        charuco_corners: np.ndarray,
        charuco_ids: np.ndarray,
        marker_corners: np.ndarray = None
    ) -> np.ndarray:
        """Draw detected board on image for visualization."""
        output = image.copy()
        
        if marker_corners is not None:
            cv2.aruco.drawDetectedMarkers(output, marker_corners)
        
        if charuco_corners is not None and charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(
                output, 
                charuco_corners, 
                charuco_ids,
                cornerColor=(0, 255, 0)
            )
        
        return output


# ============================================================================
# CAMERA CALIBRATION
# ============================================================================

class CameraCalibrator:
    """Calibrates individual cameras and stereo pairs."""
    
    def __init__(self, board_generator: CharucoBoardGenerator):
        self.board_gen = board_generator
        self.calibration_images_left: List[np.ndarray] = []
        self.calibration_images_right: List[np.ndarray] = []
        self.all_corners_left: List[np.ndarray] = []
        self.all_corners_right: List[np.ndarray] = []
        self.all_ids_left: List[np.ndarray] = []
        self.all_ids_right: List[np.ndarray] = []
        self.image_size: Tuple[int, int] = None
    
    def add_calibration_frame(
        self, 
        image_left: np.ndarray, 
        image_right: np.ndarray = None
    ) -> Tuple[bool, bool]:
        """
        Add a calibration frame pair.
        
        Returns:
            Tuple of (left_detected, right_detected)
        """
        left_ok = False
        right_ok = False
        
        # Store image size
        if self.image_size is None:
            self.image_size = (image_left.shape[1], image_left.shape[0])
        
        # Detect in left image
        corners_l, ids_l, _ = self.board_gen.detect_board(image_left)
        if corners_l is not None and len(corners_l) >= 6:
            self.calibration_images_left.append(image_left.copy())
            self.all_corners_left.append(corners_l)
            self.all_ids_left.append(ids_l)
            left_ok = True
        
        # Detect in right image (if provided)
        if image_right is not None:
            corners_r, ids_r, _ = self.board_gen.detect_board(image_right)
            if corners_r is not None and len(corners_r) >= 6:
                self.calibration_images_right.append(image_right.copy())
                self.all_corners_right.append(corners_r)
                self.all_ids_right.append(ids_r)
                right_ok = True
        
        return left_ok, right_ok
    
    def calibrate_single_camera(
        self, 
        corners_list: List[np.ndarray],
        ids_list: List[np.ndarray],
        image_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calibrate a single camera.
        
        Returns:
            Tuple of (camera_matrix, dist_coeffs, reprojection_error)
        """
        if len(corners_list) < 10:
            raise ValueError(f"Need at least 10 calibration frames, got {len(corners_list)}")
        
        # Prepare object points
        obj_points = []
        img_points = []
        
        for corners, ids in zip(corners_list, ids_list):
            obj_pts, img_pts = self.board_gen.charuco_board.matchImagePoints(corners, ids)
            if obj_pts is not None and len(obj_pts) > 0:
                obj_points.append(obj_pts)
                img_points.append(img_pts)
        
        # Calibrate
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points,
            img_points,
            image_size,
            None,
            None,
            flags=cv2.CALIB_RATIONAL_MODEL
        )
        
        return camera_matrix, dist_coeffs, ret
    
    def calibrate_left_camera(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Calibrate the left camera."""
        return self.calibrate_single_camera(
            self.all_corners_left,
            self.all_ids_left,
            self.image_size
        )
    
    def calibrate_right_camera(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Calibrate the right camera."""
        return self.calibrate_single_camera(
            self.all_corners_right,
            self.all_ids_right,
            self.image_size
        )
    
    def calibrate_stereo(
        self,
        camera_matrix_left: np.ndarray,
        dist_coeffs_left: np.ndarray,
        camera_matrix_right: np.ndarray,
        dist_coeffs_right: np.ndarray
    ) -> Dict:
        """
        Perform stereo calibration.
        
        Returns:
            Dictionary with stereo calibration results
        """
        if len(self.all_corners_left) != len(self.all_corners_right):
            raise ValueError("Mismatched number of calibration frames for stereo")
        
        # Find matching corners between left and right
        obj_points = []
        img_points_left = []
        img_points_right = []
        
        for i in range(len(self.all_corners_left)):
            corners_l = self.all_corners_left[i]
            corners_r = self.all_corners_right[i]
            ids_l = self.all_ids_left[i]
            ids_r = self.all_ids_right[i]
            
            # Find common IDs
            common_ids = np.intersect1d(ids_l.flatten(), ids_r.flatten())
            
            if len(common_ids) < 6:
                continue
            
            # Get corresponding corners
            obj_pts_l, img_pts_l = self.board_gen.charuco_board.matchImagePoints(corners_l, ids_l)
            obj_pts_r, img_pts_r = self.board_gen.charuco_board.matchImagePoints(corners_r, ids_r)
            
            if obj_pts_l is not None and obj_pts_r is not None:
                # Use left object points as reference
                obj_points.append(obj_pts_l)
                img_points_left.append(img_pts_l)
                img_points_right.append(img_pts_r)
        
        if len(obj_points) < 10:
            raise ValueError(f"Need at least 10 stereo frame pairs, got {len(obj_points)}")
        
        # Stereo calibration
        flags = (
            cv2.CALIB_FIX_INTRINSIC |
            cv2.CALIB_RATIONAL_MODEL
        )
        
        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            obj_points,
            img_points_left,
            img_points_right,
            camera_matrix_left,
            dist_coeffs_left,
            camera_matrix_right,
            dist_coeffs_right,
            self.image_size,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        )
        
        # Stereo rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            camera_matrix_left,
            dist_coeffs_left,
            camera_matrix_right,
            dist_coeffs_right,
            self.image_size,
            R,
            T,
            alpha=0
        )
        
        return {
            'reprojection_error': ret,
            'R': R,
            'T': T,
            'E': E,
            'F': F,
            'R1': R1,
            'R2': R2,
            'P1': P1,
            'P2': P2,
            'Q': Q,
            'roi_left': roi1,
            'roi_right': roi2
        }
    
    def clear(self):
        """Clear all calibration data."""
        self.calibration_images_left.clear()
        self.calibration_images_right.clear()
        self.all_corners_left.clear()
        self.all_corners_right.clear()
        self.all_ids_left.clear()
        self.all_ids_right.clear()
        self.image_size = None


# ============================================================================
# CALIBRATION SESSION
# ============================================================================

class CalibrationSession:
    """Manages a complete calibration session."""
    
    def __init__(self, output_dir: str = "./calibration_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.board_gen = CharucoBoardGenerator()
        self.calibrator = CameraCalibrator(self.board_gen)
        
        self.results = {}
    
    def run_interactive_calibration(
        self,
        camera_left_id: int = 0,
        camera_right_id: int = 1,
        num_frames: int = 20,
        frame_delay: float = 1.0
    ):
        """
        Run interactive stereo calibration session.
        
        Args:
            camera_left_id: Left camera device ID
            camera_right_id: Right camera device ID
            num_frames: Number of calibration frames to capture
            frame_delay: Minimum delay between captures
        """
        # Open cameras
        cap_left = self._open_camera(camera_left_id)
        cap_right = self._open_camera(camera_right_id)
        
        if cap_left is None or cap_right is None:
            print("Failed to open cameras!")
            return
        
        print("\n" + "=" * 50)
        print("STEREO CALIBRATION SESSION")
        print("=" * 50)
        print(f"\nCapturing {num_frames} frame pairs...")
        print("Hold the ChArUco board visible to BOTH cameras.")
        print("Move the board to different positions and angles.")
        print("\nControls:")
        print("  SPACE - Capture frame pair")
        print("  Q     - Quit calibration")
        print("=" * 50 + "\n")
        
        frames_captured = 0
        last_capture_time = 0
        
        while frames_captured < num_frames:
            # Read frames
            ret_l, frame_left = cap_left.read()
            ret_r, frame_right = cap_right.read()
            
            if not ret_l or not ret_r:
                continue
            
            # Detect board in both frames
            corners_l, ids_l, markers_l = self.board_gen.detect_board(frame_left)
            corners_r, ids_r, markers_r = self.board_gen.detect_board(frame_right)
            
            # Draw detection
            display_left = self.board_gen.draw_detected_board(
                frame_left, corners_l, ids_l, markers_l
            )
            display_right = self.board_gen.draw_detected_board(
                frame_right, corners_r, ids_r, markers_r
            )
            
            # Detection status
            left_ok = corners_l is not None and len(corners_l) >= 6
            right_ok = corners_r is not None and len(corners_r) >= 6
            
            # Add status text
            status_l = "DETECTED" if left_ok else "NOT DETECTED"
            status_r = "DETECTED" if right_ok else "NOT DETECTED"
            color_l = (0, 255, 0) if left_ok else (0, 0, 255)
            color_r = (0, 255, 0) if right_ok else (0, 0, 255)
            
            cv2.putText(display_left, f"LEFT: {status_l}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color_l, 2)
            cv2.putText(display_right, f"RIGHT: {status_r}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color_r, 2)
            cv2.putText(display_left, f"Frames: {frames_captured}/{num_frames}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show combined view
            combined = np.hstack([display_left, display_right])
            cv2.imshow("Stereo Calibration", combined)
            
            key = cv2.waitKey(1) & 0xFF
            
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            
            if key == ord(' ') and current_time - last_capture_time > frame_delay:
                # Capture
                if left_ok and right_ok:
                    self.calibrator.add_calibration_frame(frame_left, frame_right)
                    frames_captured += 1
                    last_capture_time = current_time
                    print(f"Captured frame {frames_captured}/{num_frames}")
                else:
                    print("Board not detected in both views!")
            
            elif key == ord('q'):
                print("Calibration cancelled.")
                break
        
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()
        
        if frames_captured >= 10:
            print("\nCalibrating cameras...")
            self._run_calibration()
        else:
            print(f"\nNot enough frames captured ({frames_captured}). Need at least 10.")
    
    def _open_camera(self, device_id: int) -> Optional[cv2.VideoCapture]:
        """Open a camera with fallback backends."""
        backends = [
            cv2.CAP_DSHOW,      # Windows DirectShow
            cv2.CAP_MSMF,       # Windows Media Foundation
            cv2.CAP_V4L2,       # Linux Video4Linux2
            cv2.CAP_ANY,        # Auto-select
        ]
        
        for backend in backends:
            cap = cv2.VideoCapture(device_id, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                return cap
            cap.release()
        
        return None
    
    def _run_calibration(self):
        """Run the calibration computation."""
        try:
            # Calibrate individual cameras
            print("Calibrating left camera...")
            cam_matrix_l, dist_l, error_l = self.calibrator.calibrate_left_camera()
            print(f"  Reprojection error: {error_l:.4f}")
            
            print("Calibrating right camera...")
            cam_matrix_r, dist_r, error_r = self.calibrator.calibrate_right_camera()
            print(f"  Reprojection error: {error_r:.4f}")
            
            # Stereo calibration
            print("Performing stereo calibration...")
            stereo_results = self.calibrator.calibrate_stereo(
                cam_matrix_l, dist_l, cam_matrix_r, dist_r
            )
            print(f"  Stereo reprojection error: {stereo_results['reprojection_error']:.4f}")
            
            # Store results
            self.results = {
                'timestamp': datetime.now().isoformat(),
                'image_size': list(self.calibrator.image_size),
                'camera_left': {
                    'camera_matrix': cam_matrix_l.tolist(),
                    'dist_coeffs': dist_l.flatten().tolist(),
                    'reprojection_error': error_l
                },
                'camera_right': {
                    'camera_matrix': cam_matrix_r.tolist(),
                    'dist_coeffs': dist_r.flatten().tolist(),
                    'reprojection_error': error_r
                },
                'stereo': {
                    'R': stereo_results['R'].tolist(),
                    'T': stereo_results['T'].flatten().tolist(),
                    'E': stereo_results['E'].tolist(),
                    'F': stereo_results['F'].tolist(),
                    'R1': stereo_results['R1'].tolist(),
                    'R2': stereo_results['R2'].tolist(),
                    'P1': stereo_results['P1'].tolist(),
                    'P2': stereo_results['P2'].tolist(),
                    'Q': stereo_results['Q'].tolist(),
                    'reprojection_error': stereo_results['reprojection_error']
                }
            }
            
            # Save results
            output_path = os.path.join(self.output_dir, "stereo_calibration.json")
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            print(f"\nCalibration saved to: {output_path}")
            print("\nCalibration complete!")
            
        except Exception as e:
            print(f"Calibration failed: {e}")
            raise


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for calibration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Camera Calibration Tool")
    parser.add_argument('--generate-board', action='store_true',
                       help="Generate ChArUco board image")
    parser.add_argument('--calibrate', action='store_true',
                       help="Run interactive calibration")
    parser.add_argument('--left-camera', type=int, default=0,
                       help="Left camera device ID")
    parser.add_argument('--right-camera', type=int, default=1,
                       help="Right camera device ID")
    parser.add_argument('--output-dir', type=str, default="./calibration_data",
                       help="Output directory")
    parser.add_argument('--num-frames', type=int, default=20,
                       help="Number of calibration frames")
    
    args = parser.parse_args()
    
    if args.generate_board:
        board_gen = CharucoBoardGenerator()
        board_gen.generate_board_image(
            os.path.join(args.output_dir, "charuco_board.png")
        )
    
    if args.calibrate:
        session = CalibrationSession(args.output_dir)
        session.run_interactive_calibration(
            args.left_camera,
            args.right_camera,
            args.num_frames
        )
    
    if not args.generate_board and not args.calibrate:
        parser.print_help()


if __name__ == "__main__":
    main()
