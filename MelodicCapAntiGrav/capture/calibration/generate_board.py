"""
Generate ChArUco calibration board image.
Uses the SAME board config as the rest of the pipeline.

IMPORTANT: Print this at 100% scale (no fit-to-page) and measure
the printed squares to verify they match BOARD_SQUARE_M.
"""
import cv2
import cv2.aruco as aruco
import sys
import os

# Import canonical config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from engine.camera_thread import BOARD_COLS, BOARD_ROWS, BOARD_SQUARE_M, BOARD_MARKER_M, BOARD_DICT


def generate_charuco_board(output_path=None):
    dictionary = aruco.getPredefinedDictionary(BOARD_DICT)
    board = aruco.CharucoBoard(
        (BOARD_COLS, BOARD_ROWS), BOARD_SQUARE_M, BOARD_MARKER_M, dictionary
    )

    # Generate at 300 DPI for A4 (210x297mm)
    img = board.generateImage((2100, 2970), marginSize=100, borderBits=1)

    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "charuco_board.png")

    cv2.imwrite(output_path, img)
    print(f"Board generated: {output_path}")
    print(f"  Grid: {BOARD_COLS}x{BOARD_ROWS}")
    print(f"  Square: {BOARD_SQUARE_M*1000:.1f}mm")
    print(f"  Marker: {BOARD_MARKER_M*1000:.1f}mm")
    print(f"  IMPORTANT: Print at 100% scale and verify square size with ruler!")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else None
    generate_charuco_board(output)
