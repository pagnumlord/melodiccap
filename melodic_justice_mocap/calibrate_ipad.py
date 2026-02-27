#!/usr/bin/env python3
"""
Stereo Camera Calibration for iPad ChArUco Board
=================================================

Configured for iPad Pro 11" board:
- 5x4 squares
- 30mm square size
- 22.5mm marker size

Usage:
    python calibrate_ipad.py --left 2 --right 4
"""

import cv2
import numpy as np
import json
import argparse
import os
import time
from datetime import datetime
from pathlib import Path

# ============================================================================
# iPad Pro 11" ChArUco Board Configuration
# ============================================================================
SQUARES_X = 5
SQUARES_Y = 4
SQUARE_SIZE = 0.030  # 30mm in meters
MARKER_SIZE = 0.0225  # 22.5mm in meters (75% of square)


class StereoCalibrator:
    """Stereo camera calibration using ChArUco board."""
    
    def __init__(
        self,
        left_camera_id: int = 0,
        right_camera_id: int = 1,
        squares_x: int = SQUARES_X,
        squares_y: int = SQUARES_Y,
        square_size: float = SQUARE_SIZE,
        marker_size: float = MARKER_SIZE
    ):
        self.left_id = left_camera_id
        self.right_id = right_camera_id
        
        # Board configuration
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_size = square_size
        self.marker_size = marker_size
        
        # Create ArUco dictionary and ChArUco board
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.charuco_board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_size,
            marker_size,
            self.aruco_dict
        )
        
        # Create detectors
        self.aruco_detector = cv2.aruco.ArucoDetector(
            self.aruco_dict,
            cv2.aruco.DetectorParameters()
        )
        self.charuco_detector = cv2.aruco.CharucoDetector(self.charuco_board)
        
        # Storage for calibration data
        self.all_charuco_corners_left = []
        self.all_charuco_ids_left = []
        self.all_charuco_corners_right = []
        self.all_charuco_ids_right = []
        self.image_size = None
        
        # Cameras
        self.cap_left = None
        self.cap_right = None
        
        print(f"\n{'='*60}")
        print("STEREO CALIBRATION - iPad ChArUco Board")
        print(f"{'='*60}")
        print(f"Board: {squares_x}x{squares_y} squares")
        print(f"Square size: {square_size*1000:.1f}mm")
        print(f"Marker size: {marker_size*1000:.1f}mm")
        print(f"Left camera: {left_camera_id}")
        print(f"Right camera: {right_camera_id}")
        print(f"{'='*60}\n")
    
    def initialize_cameras(self) -> bool:
        """Initialize both cameras."""
        print("Initializing cameras...")
        
        # Try different backends
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Auto"),
        ]
        
        # Initialize left camera
        for backend, name in backends:
            self.cap_left = cv2.VideoCapture(self.left_id, backend)
            if self.cap_left.isOpened():
                self.cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                ret, frame = self.cap_left.read()
                if ret:
                    print(f"✓ Left camera (ID {self.left_id}) opened with {name}")
                    break
                self.cap_left.release()
        
        if not self.cap_left or not self.cap_left.isOpened():
            print(f"✗ Failed to open left camera (ID {self.left_id})")
            return False
        
        # Initialize right camera
        for backend, name in backends:
            self.cap_right = cv2.VideoCapture(self.right_id, backend)
            if self.cap_right.isOpened():
                self.cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                ret, frame = self.cap_right.read()
                if ret:
                    print(f"✓ Right camera (ID {self.right_id}) opened with {name}")
                    break
                self.cap_right.release()
        
        if not self.cap_right or not self.cap_right.isOpened():
            print(f"✗ Failed to open right camera (ID {self.right_id})")
            return False
        
        return True
    
    def detect_charuco(self, image: np.ndarray):
        """
        Detect ChArUco corners in an image.
        
        Returns:
            (charuco_corners, charuco_ids, annotated_image, num_corners)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        annotated = image.copy()
        
        # Detect ArUco markers first
        marker_corners, marker_ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        if marker_ids is None or len(marker_ids) == 0:
            # No markers detected
            cv2.putText(annotated, "No markers detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return None, None, annotated, 0
        
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(annotated, marker_corners, marker_ids)
        
        # Interpolate ChArUco corners
        num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, self.charuco_board
        )
        
        if charuco_corners is not None and num_corners >= 4:
            # Draw ChArUco corners
            cv2.aruco.drawDetectedCornersCharuco(annotated, charuco_corners, charuco_ids)
            
            # Add corner count
            color = (0, 255, 0) if num_corners >= 6 else (0, 255, 255)
            cv2.putText(annotated, f"Corners: {num_corners}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            return charuco_corners, charuco_ids, annotated, num_corners
        else:
            cv2.putText(annotated, f"Markers: {len(marker_ids)}, need more corners", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return None, None, annotated, 0
    
    def capture_frame(self) -> bool:
        """Capture a calibration frame from both cameras."""
        ret_l, frame_l = self.cap_left.read()
        ret_r, frame_r = self.cap_right.read()
        
        if not ret_l or not ret_r:
            print("Failed to read from cameras")
            return False
        
        # Store image size
        if self.image_size is None:
            self.image_size = (frame_l.shape[1], frame_l.shape[0])
        
        # Detect ChArUco in both images
        corners_l, ids_l, _, num_l = self.detect_charuco(frame_l)
        corners_r, ids_r, _, num_r = self.detect_charuco(frame_r)
        
        # Need detection in both cameras with enough corners
        if corners_l is None or corners_r is None:
            print("  ✗ Board not detected in both cameras")
            return False
        
        if num_l < 6 or num_r < 6:
            print(f"  ✗ Not enough corners (left: {num_l}, right: {num_r}, need 6+)")
            return False
        
        # Store the corners
        self.all_charuco_corners_left.append(corners_l)
        self.all_charuco_ids_left.append(ids_l)
        self.all_charuco_corners_right.append(corners_r)
        self.all_charuco_ids_right.append(ids_r)
        
        print(f"  ✓ Captured! Left: {num_l} corners, Right: {num_r} corners")
        return True
    
    def calibrate(self, output_path: str = "./calibration_data"):
        """Run full stereo calibration."""
        num_frames = len(self.all_charuco_corners_left)
        
        if num_frames < 5:
            print(f"\n✗ Not enough frames for calibration ({num_frames}, need at least 5)")
            return None
        
        print(f"\nRunning calibration with {num_frames} frames...")
        
        # Calibrate left camera
        print("  Calibrating left camera...")
        ret_l, K1, D1, rvecs_l, tvecs_l = cv2.aruco.calibrateCameraCharuco(
            self.all_charuco_corners_left,
            self.all_charuco_ids_left,
            self.charuco_board,
            self.image_size,
            None, None
        )
        print(f"    RMS error: {ret_l:.4f}")
        
        # Calibrate right camera
        print("  Calibrating right camera...")
        ret_r, K2, D2, rvecs_r, tvecs_r = cv2.aruco.calibrateCameraCharuco(
            self.all_charuco_corners_right,
            self.all_charuco_ids_right,
            self.charuco_board,
            self.image_size,
            None, None
        )
        print(f"    RMS error: {ret_r:.4f}")
        
        # Prepare object points for stereo calibration
        # We need matching corner IDs between left and right
        obj_points = []
        img_points_l = []
        img_points_r = []
        
        for i in range(num_frames):
            corners_l = self.all_charuco_corners_left[i]
            ids_l = self.all_charuco_ids_left[i]
            corners_r = self.all_charuco_corners_right[i]
            ids_r = self.all_charuco_ids_right[i]
            
            # Find common IDs
            ids_l_flat = ids_l.flatten()
            ids_r_flat = ids_r.flatten()
            common_ids = np.intersect1d(ids_l_flat, ids_r_flat)
            
            if len(common_ids) < 4:
                continue
            
            # Get corresponding corners
            obj_pts = []
            img_pts_l = []
            img_pts_r = []
            
            for cid in common_ids:
                # Get object point (3D position on board)
                obj_pt = self.charuco_board.getChessboardCorners()[cid]
                obj_pts.append(obj_pt)
                
                # Get image points
                idx_l = np.where(ids_l_flat == cid)[0][0]
                idx_r = np.where(ids_r_flat == cid)[0][0]
                img_pts_l.append(corners_l[idx_l])
                img_pts_r.append(corners_r[idx_r])
            
            obj_points.append(np.array(obj_pts, dtype=np.float32))
            img_points_l.append(np.array(img_pts_l, dtype=np.float32))
            img_points_r.append(np.array(img_pts_r, dtype=np.float32))
        
        if len(obj_points) < 5:
            print(f"\n✗ Not enough matching frames ({len(obj_points)})")
            return None
        
        # Stereo calibration
        print("  Running stereo calibration...")
        flags = cv2.CALIB_FIX_INTRINSIC
        
        ret_stereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            obj_points,
            img_points_l,
            img_points_r,
            K1, D1, K2, D2,
            self.image_size,
            flags=flags
        )
        print(f"    Stereo RMS error: {ret_stereo:.4f}")
        
        # Compute rectification
        print("  Computing rectification...")
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, D1, K2, D2, self.image_size, R, T, alpha=0
        )
        
        # Save calibration
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, "stereo_calibration.json")
        
        calibration_data = {
            "timestamp": datetime.now().isoformat(),
            "image_size": list(self.image_size),
            "board_config": {
                "squares_x": self.squares_x,
                "squares_y": self.squares_y,
                "square_size": self.square_size,
                "marker_size": self.marker_size
            },
            "num_frames": num_frames,
            "rms_error": {
                "left": ret_l,
                "right": ret_r,
                "stereo": ret_stereo
            },
            "K1": K1.tolist(),
            "D1": D1.tolist(),
            "K2": K2.tolist(),
            "D2": D2.tolist(),
            "R": R.tolist(),
            "T": T.tolist(),
            "E": E.tolist(),
            "F": F.tolist(),
            "R1": R1.tolist(),
            "R2": R2.tolist(),
            "P1": P1.tolist(),
            "P2": P2.tolist(),
            "Q": Q.tolist(),
            "baseline_meters": float(np.linalg.norm(T))
        }
        
        with open(output_file, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"\n{'='*60}")
        print("CALIBRATION COMPLETE!")
        print(f"{'='*60}")
        print(f"Saved to: {output_file}")
        print(f"Baseline: {np.linalg.norm(T)*100:.1f} cm")
        print(f"Stereo RMS: {ret_stereo:.4f}")
        print(f"{'='*60}\n")
        
        return calibration_data
    
    def run_interactive(self, target_frames: int = 15):
        """Run interactive calibration session."""
        if not self.initialize_cameras():
            return None
        
        print("\n" + "="*60)
        print("INTERACTIVE CALIBRATION")
        print("="*60)
        print("\nControls:")
        print("  SPACE - Capture frame (when board visible in BOTH cameras)")
        print("  C     - Run calibration with captured frames")
        print("  R     - Reset (clear all captured frames)")
        print("  Q     - Quit")
        print(f"\nTarget: {target_frames} frames")
        print("Move the board to different positions and angles!")
        print("="*60 + "\n")
        
        try:
            while True:
                # Read from both cameras
                ret_l, frame_l = self.cap_left.read()
                ret_r, frame_r = self.cap_right.read()
                
                if not ret_l or not ret_r:
                    continue
                
                # Detect ChArUco
                _, _, annotated_l, num_l = self.detect_charuco(frame_l)
                _, _, annotated_r, num_r = self.detect_charuco(frame_r)
                
                # Add status info
                status = f"Frames: {len(self.all_charuco_corners_left)}/{target_frames}"
                cv2.putText(annotated_l, status, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_l, "LEFT (ZV-1F)", (10, annotated_l.shape[0]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(annotated_r, "RIGHT (DroidCam)", (10, annotated_r.shape[0]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Show capture readiness
                if num_l >= 6 and num_r >= 6:
                    cv2.putText(annotated_l, "READY - Press SPACE", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_r, "READY - Press SPACE", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Resize for display
                scale = 0.7
                disp_l = cv2.resize(annotated_l, None, fx=scale, fy=scale)
                disp_r = cv2.resize(annotated_r, None, fx=scale, fy=scale)
                
                # Stack horizontally
                combined = np.hstack([disp_l, disp_r])
                cv2.imshow("Stereo Calibration", combined)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                
                elif key == ord(' '):
                    print(f"\nCapturing frame {len(self.all_charuco_corners_left)+1}...")
                    if self.capture_frame():
                        if len(self.all_charuco_corners_left) >= target_frames:
                            print(f"\n✓ Reached target of {target_frames} frames!")
                            print("Press 'C' to calibrate or continue capturing.")
                
                elif key == ord('c'):
                    if len(self.all_charuco_corners_left) >= 5:
                        result = self.calibrate()
                        if result:
                            break
                    else:
                        print(f"\nNeed at least 5 frames (have {len(self.all_charuco_corners_left)})")
                
                elif key == ord('r'):
                    print("\nResetting - cleared all captured frames")
                    self.all_charuco_corners_left = []
                    self.all_charuco_ids_left = []
                    self.all_charuco_corners_right = []
                    self.all_charuco_ids_right = []
        
        finally:
            self.cap_left.release()
            self.cap_right.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Stereo calibration for iPad ChArUco board",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This calibration script is configured for the iPad Pro 11" board:
  - 5x4 squares
  - 30mm square size
  - 22.5mm marker size

Example:
  python calibrate_ipad.py --left 2 --right 4 --frames 15
        """
    )
    
    parser.add_argument("--left", type=int, default=0,
                       help="Left camera ID (default: 0)")
    parser.add_argument("--right", type=int, default=1,
                       help="Right camera ID (default: 1)")
    parser.add_argument("--frames", type=int, default=15,
                       help="Target number of frames (default: 15)")
    parser.add_argument("--output", type=str, default="./calibration_data",
                       help="Output directory for calibration data")
    
    # Allow custom board sizes
    parser.add_argument("--squares_x", type=int, default=SQUARES_X,
                       help=f"Board squares X (default: {SQUARES_X})")
    parser.add_argument("--squares_y", type=int, default=SQUARES_Y,
                       help=f"Board squares Y (default: {SQUARES_Y})")
    parser.add_argument("--square_size", type=float, default=SQUARE_SIZE,
                       help=f"Square size in meters (default: {SQUARE_SIZE})")
    parser.add_argument("--marker_size", type=float, default=MARKER_SIZE,
                       help=f"Marker size in meters (default: {MARKER_SIZE})")
    
    args = parser.parse_args()
    
    calibrator = StereoCalibrator(
        left_camera_id=args.left,
        right_camera_id=args.right,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_size=args.square_size,
        marker_size=args.marker_size
    )
    
    calibrator.run_interactive(target_frames=args.frames)


if __name__ == "__main__":
    main()
