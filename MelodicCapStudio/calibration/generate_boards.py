"""
MelodicCap ChArUco Board Generator
===================================
Generates optimized ChArUco boards for stereo calibration.

OPTION A: Single 8.5x11" sheet — 5x7 squares at 35mm (24 corners)
OPTION B: Two 8.5x11" sheets taped together — 10x5 at 45mm (36 corners)

The old 4x3 board had only 6 corners. That's why calibration was bad.
Minimum for decent stereo calibration is ~20 corners. 24-36 is good.

HOW TO USE THE 2-SHEET BOARD:
1. Print both _left.png and _right.png on separate 8.5x11" sheets
2. Cut along the RIGHT edge of the left sheet (trim to the alignment marks)
3. Align the sheets using the alignment marks (arrows + lines)
4. Tape them together on the BACK with clear tape
5. Mount on a rigid flat surface (cardboard, foam board, clipboard)
   CRITICAL: The board MUST be flat. Any warping = bad calibration.

SQUARE SIZE MATTERS:
- The square size printed on paper must be MEASURED with a ruler
- Your printer may scale. Measure an actual printed square.
- Enter the MEASURED size into the capture script's CHARUCO_SQUARE_M
- If a "35mm" square prints at 34.2mm, use 0.0342, not 0.035

Usage:
    python generate_boards.py
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_single_sheet_board():
    """Generate a 5x7 ChArUco board that fits on one 8.5x11" sheet.

    5 columns x 7 rows of squares at 35mm each
    Board area: 175mm x 245mm (fits in 215.9mm x 279.4mm with margins)
    Interior corners: (5-1) * (7-1) = 24
    ArUco markers: 12 (in the black squares)
    """
    squares_x = 5
    squares_y = 7
    square_length_m = 0.035   # 35mm
    marker_length_m = 0.025   # 25mm (71% of square, good for detection)

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    board = aruco.CharucoBoard(
        (squares_x, squares_y), square_length_m, marker_length_m, dictionary
    )

    # Generate at 300 DPI for 8.5x11"
    # 8.5" x 300 = 2550px, 11" x 300 = 3300px
    page_w_px = 2550
    page_h_px = 3300

    # Board dimensions at 300 DPI
    # 35mm = 1.378" -> 413px per square
    # Board: 5*413 = 2067px wide, 7*413 = 2893px tall
    board_w_px = int(squares_x * square_length_m * 1000 / 25.4 * 300)
    board_h_px = int(squares_y * square_length_m * 1000 / 25.4 * 300)

    # Generate board image
    img = board.generateImage((board_w_px, board_h_px), marginSize=0, borderBits=1)

    # Place on white page with centering
    page = np.ones((page_h_px, page_w_px), dtype=np.uint8) * 255
    offset_x = (page_w_px - board_w_px) // 2
    offset_y = (page_h_px - board_h_px) // 2
    page[offset_y:offset_y + board_h_px, offset_x:offset_x + board_w_px] = img

    # Add labels
    page_color = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)

    # Title at top
    cv2.putText(page_color, "MelodicCap ChArUco Board - SINGLE SHEET",
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.putText(page_color, f"5x7 squares | 35mm squares | 25mm markers | DICT_4X4_50 | 24 corners",
                (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Instructions at bottom
    y_bottom = page_h_px - 120
    cv2.putText(page_color, "IMPORTANT: Measure a printed square with a ruler. Enter ACTUAL size in capture script.",
                (50, y_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 180), 2)
    cv2.putText(page_color, "Mount on rigid flat surface (cardboard/clipboard). Board must be FLAT.",
                (50, y_bottom + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 180), 2)
    cv2.putText(page_color, f"Print at 100% scale (no fit-to-page). Target: 35mm = ~1.378 inches per square.",
                (50, y_bottom + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Ruler marks on edges (10mm increments) for measurement verification
    dpi = 300
    mm_per_px = 25.4 / dpi
    for mm in range(0, 200, 10):
        px = int(mm / mm_per_px)
        x_start = offset_x
        # Top ruler
        tick_len = 30 if mm % 50 == 0 else 15
        cv2.line(page_color, (x_start + px, offset_y - 5), (x_start + px, offset_y - 5 - tick_len), (0, 0, 0), 2)
        if mm % 50 == 0:
            cv2.putText(page_color, f"{mm}mm", (x_start + px - 20, offset_y - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    path = os.path.join(OUTPUT_DIR, "charuco_board_5x7_single.png")
    cv2.imwrite(path, page_color)
    print(f"Single-sheet board saved: {path}")
    print(f"  5x7 squares, 35mm each, 24 interior corners")
    print(f"  Board pixel size: {board_w_px}x{board_h_px}")

    return squares_x, squares_y, square_length_m, marker_length_m


def generate_dual_sheet_board():
    """Generate a 10x5 ChArUco board split across two 8.5x11" sheets (landscape).

    10 columns x 5 rows of squares at 45mm each
    Board area: 450mm x 225mm
    Interior corners: (10-1) * (5-1) = 36

    Each sheet is landscape (11" x 8.5" = 279.4mm x 215.9mm)
    Left half: 5 columns = 225mm (fits in 279.4mm)
    Right half: 5 columns = 225mm (fits in 279.4mm)
    Height: 5 rows x 45mm = 225mm (fits in 215.9mm... tight, use 43mm)

    Actually, let's use 43mm squares to safely fit height:
    5 x 43 = 215mm, fits in 215.9mm usable
    10 x 43 = 430mm across two sheets
    Corners: (10-1)*(5-1) = 36
    """
    squares_x = 10
    squares_y = 5
    square_length_m = 0.043   # 43mm (fits 5 rows in 8.5" height)
    marker_length_m = 0.031   # 31mm (72% of square)

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    board = aruco.CharucoBoard(
        (squares_x, squares_y), square_length_m, marker_length_m, dictionary
    )

    # Generate the full board
    dpi = 300
    sq_px = int(square_length_m * 1000 / 25.4 * dpi)  # ~508px per square
    board_w_px = squares_x * sq_px   # ~5080px
    board_h_px = squares_y * sq_px   # ~2540px

    img = board.generateImage((board_w_px, board_h_px), marginSize=0, borderBits=1)

    # Split into left and right halves
    half_cols = squares_x // 2  # 5 columns each
    half_w_px = half_cols * sq_px

    img_left = img[:, :half_w_px]
    img_right = img[:, half_w_px:]

    # Page dimensions (landscape 8.5x11")
    page_w_px = 3300  # 11" at 300dpi
    page_h_px = 2550  # 8.5" at 300dpi

    for side, half_img, label in [("left", img_left, "LEFT HALF"), ("right", img_right, "RIGHT HALF")]:
        page = np.ones((page_h_px, page_w_px), dtype=np.uint8) * 255

        # Center vertically, align to edge for joining
        offset_y = (page_h_px - board_h_px) // 2

        if side == "left":
            # Board aligned to RIGHT edge (where sheets join)
            offset_x = page_w_px - half_w_px - 50  # 50px margin from right edge
        else:
            # Board aligned to LEFT edge (where sheets join)
            offset_x = 50  # 50px margin from left edge

        page[offset_y:offset_y + board_h_px, offset_x:offset_x + half_w_px] = half_img

        page_color = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)

        # Alignment marks
        mark_y_top = offset_y - 30
        mark_y_bot = offset_y + board_h_px + 30

        if side == "left":
            # Alignment marks on right edge
            mark_x = offset_x + half_w_px
            # Arrow pointing right
            cv2.arrowedLine(page_color, (mark_x - 80, mark_y_top), (mark_x, mark_y_top), (0, 0, 255), 3)
            cv2.arrowedLine(page_color, (mark_x - 80, mark_y_bot), (mark_x, mark_y_bot), (0, 0, 255), 3)
            # Vertical alignment line
            cv2.line(page_color, (mark_x, offset_y - 60), (mark_x, offset_y + board_h_px + 60), (0, 0, 255), 2)
            cv2.putText(page_color, "ALIGN HERE ->", (mark_x - 250, mark_y_top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Alignment marks on left edge
            mark_x = offset_x
            cv2.arrowedLine(page_color, (mark_x + 80, mark_y_top), (mark_x, mark_y_top), (0, 0, 255), 3)
            cv2.arrowedLine(page_color, (mark_x + 80, mark_y_bot), (mark_x, mark_y_bot), (0, 0, 255), 3)
            cv2.line(page_color, (mark_x, offset_y - 60), (mark_x, offset_y + board_h_px + 60), (0, 0, 255), 2)
            cv2.putText(page_color, "<- ALIGN HERE", (mark_x + 10, mark_y_top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Labels
        cv2.putText(page_color, f"MelodicCap ChArUco - {label} (tape sheets together)",
                    (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(page_color, f"10x5 squares | 43mm squares | 31mm markers | DICT_4X4_50 | 36 corners",
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Bottom instructions
        y_b = page_h_px - 80
        cv2.putText(page_color, "Print LANDSCAPE at 100% scale. Trim right/left edge at red line. Tape halves together on BACK.",
                    (50, y_b), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 180), 2)
        cv2.putText(page_color, "Mount on RIGID FLAT surface. Measure actual square size with ruler.",
                    (50, y_b + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 180), 2)

        path = os.path.join(OUTPUT_DIR, f"charuco_board_10x5_{side}.png")
        cv2.imwrite(path, page_color)
        print(f"Dual-sheet {side} saved: {path}")

    print(f"  10x5 squares, 43mm each, 36 interior corners")
    print(f"  Print both sheets LANDSCAPE, trim, tape together")

    return squares_x, squares_y, square_length_m, marker_length_m


if __name__ == "__main__":
    print("=" * 60)
    print("MELODICCAP CHARUCO BOARD GENERATOR")
    print("=" * 60)

    print("\n--- Option A: Single 8.5x11\" sheet ---")
    s1 = generate_single_sheet_board()

    print("\n--- Option B: Two 8.5x11\" sheets (landscape, taped) ---")
    s2 = generate_dual_sheet_board()

    print("\n" + "=" * 60)
    print("CONFIGURATION VALUES FOR CAPTURE SCRIPT:")
    print("=" * 60)
    print(f"\nOption A (single sheet):")
    print(f"  CHARUCO_COLS = {s1[0]}")
    print(f"  CHARUCO_ROWS = {s1[1]}")
    print(f"  CHARUCO_SQUARE_M = {s1[2]}  # MEASURE AND UPDATE!")
    print(f"  CHARUCO_MARKER_M = {s1[3]}  # scale proportionally")

    print(f"\nOption B (two sheets, RECOMMENDED):")
    print(f"  CHARUCO_COLS = {s2[0]}")
    print(f"  CHARUCO_ROWS = {s2[1]}")
    print(f"  CHARUCO_SQUARE_M = {s2[2]}  # MEASURE AND UPDATE!")
    print(f"  CHARUCO_MARKER_M = {s2[3]}  # scale proportionally")

    print("\nREMINDER: Measure a printed square with a ruler!")
    print("If 35mm prints as 34.1mm, use 0.0341 not 0.035")
