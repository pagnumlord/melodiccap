"""Generate a 7x5 ChArUco board for stereo calibration. Print at 100% scale."""
import cv2
import cv2.aruco as aruco
from pathlib import Path


def generate_charuco_board():
    # Must match Config in melodic_capture.py
    squares_x = 7
    squares_y = 5
    square_length = 0.035  # 35mm
    marker_length = 0.026  # 26mm
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    board = aruco.CharucoBoard(
        (squares_x, squares_y), square_length, marker_length, dictionary
    )

    # A4 at 300 DPI = 2480 x 3508 pixels
    img = board.generateImage((2480, 3508), marginSize=100, borderBits=1)

    output_dir = Path(__file__).parent / "calibration"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "charuco_7x5_A4.png"

    cv2.imwrite(str(output_path), img)
    print(f"Board saved to: {output_path}")
    print("Print at 100% scale (no fit-to-page). Measure a square - should be 35mm.")


if __name__ == "__main__":
    generate_charuco_board()
