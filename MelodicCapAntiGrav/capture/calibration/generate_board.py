import cv2
import cv2.aruco as aruco

def generate_charuco_board():
    # Board parameters (matches our calibrator)
    squares_x = 5
    squares_y = 7
    square_length = 0.04 # 40mm
    marker_length = 0.02 # 20mm
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    
    board = aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)
    
    # Generate image (A4 size roughly at 300 DPI)
    # A4 is 210 x 297 mm
    img = board.generateImage((2100, 2970), marginSize=100, borderBits=1)
    
    cv2.imwrite("C:/Users/ninja/Documents/MelodicCapStudio/MelodicCapStudio_App/calibration/charuco_board_A4.png", img)
    print("Charuco Board generated at: C:/Users/ninja/Documents/MelodicCapStudio/MelodicCapStudio_App/calibration/charuco_board_A4.png")

if __name__ == "__main__":
    generate_charuco_board()
