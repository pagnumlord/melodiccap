"""
Generate a ChArUco board for stereo calibration.

Default: 5x7 board, fits on a single letter page (8.5" x 11") with margins.
Print at 100% scale (no fit-to-page). Measure actual square size with ruler.

Previous: 10x5 board required two taped pages — edge truncation caused ~3mm
systematic corner error. The single-sheet 5x7 eliminates this.
"""
import cv2
import cv2.aruco as aruco
from pathlib import Path


def generate_charuco_board():
    # Must match Config in melodic_capture.py EXACTLY
    squares_x = 5
    squares_y = 7
    square_length = 0.035    # 35mm — fits 5 columns in 175mm (6.89") on 8.5" page
    marker_length = 0.025    # 25mm — ~71% of square size (good detection ratio)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    board = aruco.CharucoBoard(
        (squares_x, squares_y), square_length, marker_length, dictionary
    )

    n_corners = (squares_x - 1) * (squares_y - 1)

    # Single letter page at 300 DPI = 2550 x 3300 pixels (8.5" x 11" portrait)
    # Board: 175mm x 245mm = 6.89" x 9.65" — fits with ~0.8" margins
    img = board.generateImage((2550, 3300), marginSize=150, borderBits=1)

    output_dir = Path(__file__).parent / "calibration"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "charuco_5x7.png"

    cv2.imwrite(str(output_path), img)
    print(f"Board saved to: {output_path}")
    print(f"Board: {squares_x}x{squares_y}, square={square_length*1000:.0f}mm, "
          f"marker={marker_length*1000:.0f}mm, corners={n_corners}")
    print(f"Dictionary: DICT_4X4_50")
    print()
    print("PRINTING INSTRUCTIONS:")
    print("  1. Print on letter paper (8.5\" x 11\") in PORTRAIT orientation")
    print("  2. Set scale to 100% / Actual Size (NOT fit-to-page)")
    print("  3. Measure a printed square with a ruler — should be ~35mm (1 3/8\")")
    print("  4. If it's different, update CHARUCO_SQUARE_SIZE in melodic_capture.py")
    print("  5. Mount on RIGID FLAT surface (cardboard/clipboard). Board must be FLAT.")


if __name__ == "__main__":
    generate_charuco_board()
