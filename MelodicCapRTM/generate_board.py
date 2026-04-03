"""Generate a 10x5 ChArUco board for stereo calibration. Print at 100% scale."""
import cv2
import cv2.aruco as aruco
from pathlib import Path


def generate_charuco_board():
    # Must match Config in melodic_capture.py EXACTLY
    squares_x = 10
    squares_y = 5
    square_length = 0.04286  # 1 and 11/16 inches = 42.86mm
    marker_length = 0.03016  # 1 and 3/16 inches = 30.16mm
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    board = aruco.CharucoBoard(
        (squares_x, squares_y), square_length, marker_length, dictionary
    )

    # Two letter pages side by side at 300 DPI = 5100 x 3300 pixels
    # (11" x 8.5" landscape × 2 pages wide = 22" x 8.5")
    img = board.generateImage((5100, 3300), marginSize=100, borderBits=1)

    output_dir = Path(__file__).parent / "calibration"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "charuco_10x5.png"

    cv2.imwrite(str(output_path), img)
    print(f"Board saved to: {output_path}")
    print(f"Board config: {squares_x}x{squares_y}, square={square_length*1000:.1f}mm, marker={marker_length*1000:.1f}mm")
    print("Print at 100% scale across two letter pages. Measure a square — should be ~43mm (1 11/16\").")


if __name__ == "__main__":
    generate_charuco_board()
