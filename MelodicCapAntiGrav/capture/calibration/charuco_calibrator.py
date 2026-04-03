import cv2
import numpy as np
import sys
import os

# Import canonical board config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from engine.camera_thread import BOARD_COLS, BOARD_ROWS, BOARD_SQUARE_M, BOARD_MARKER_M, BOARD_DICT


class CharucoCalibrator:
    def __init__(self, squares_x=BOARD_COLS, squares_y=BOARD_ROWS,
                 square_length=BOARD_SQUARE_M, marker_length=BOARD_MARKER_M,
                 dictionary=BOARD_DICT):
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary)
        self.board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, self.dictionary)
        self.detector = cv2.aruco.CharucoDetector(self.board)

        self.all_corners = []
        self.all_ids = []
        self.image_size = None

    def process_frame(self, frame):
        """
        Detects Charuco corners in a frame
        """
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.image_size = grayscale.shape[::-1]
        
        charuco_corners, charuco_ids, marker_corners, marker_ids = self.detector.detectBoard(grayscale)
        
        if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 4:
            self.all_corners.append(charuco_corners)
            self.all_ids.append(charuco_ids)
            return True, frame
        return False, frame

    def calibrate(self):
        """
        Computes camera intrinsics
        """
        if len(self.all_corners) < 10:
            return None
            
        ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=self.all_corners,
            charucoIds=self.all_ids,
            board=self.board,
            imageSize=self.image_size,
            cameraMatrix=None,
            distCoeffs=None
        )
        return {"mtx": mtx, "dist": dist}

if __name__ == "__main__":
    print("Charuco Calibration Module Ready.")
