"""
MelodicCap Studio v6 - BUG FIXES
=================================
Fixed:
- Kalman filter .tolist() error
- Better error handling for lost tracking
- Graceful handling when one camera loses sight

Controls:
  C = Collect calibration frames
  S = Run stereo calibration
  F = Floor calibration
  R = Record motion capture
  Q = Quit
"""

import cv2
import numpy as np
import mediapipe as mp
import json
import time
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

CAM_A_INDEX = 2          # Sony ZV-1F (use DSHOW)
CAM_B_INDEX = 0          # DroidCam (use CAP_ANY)

# ChArUco board (your 63.5mm board)
CHARUCO_COLS = 4
CHARUCO_ROWS = 3
CHARUCO_SQUARE_M = 0.0635
CHARUCO_MARKER_M = 0.0476

# Paths
BASE_DIR = Path(r"C:\Users\ninja\Documents\MelodicCapStudio\MelodicCapFresh")
CALIBRATION_FILE = BASE_DIR / "calibration" / "stereo_calibration.json"
TAKES_DIR = BASE_DIR / "takes"

# Recording
COUNTDOWN_SECONDS = 3

# =============================================================================
# DETECTOR SETUP
# =============================================================================

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard(
    (CHARUCO_COLS, CHARUCO_ROWS),
    CHARUCO_SQUARE_M,
    CHARUCO_MARKER_M,
    aruco_dict
)
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(aruco_dict, params)
board_corners_3d = board.getChessboardCorners()

# =============================================================================
# CALIBRATION FUNCTIONS
# =============================================================================

def detect_charuco(frame):
    """Detect ChArUco corners in frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is None or len(ids) < 2:
        return None, None, 0
    
    num, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    
    if num is None or num < 4:
        return None, None, len(ids)
    
    return charuco_corners, charuco_ids, len(ids)


def draw_detection(frame, charuco_corners, charuco_ids, num_markers):
    """Draw detection results on frame"""
    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(display, corners, ids)
    
    if charuco_corners is not None:
        cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids,
                                              cornerColor=(0, 255, 0))
    
    return display


def run_stereo_calibration(frames_a, frames_b):
    """Run stereo calibration"""
    print("\n" + "="*50)
    print("STEREO CALIBRATION")
    print("="*50)
    
    all_corners_a, all_ids_a = [], []
    all_corners_b, all_ids_b = [], []
    valid_pairs = []
    
    for fa, fb in zip(frames_a, frames_b):
        cc_a, ci_a, _ = detect_charuco(fa)
        cc_b, ci_b, _ = detect_charuco(fb)
        
        if cc_a is None or cc_b is None:
            continue
        
        common = set(ci_a.flatten()) & set(ci_b.flatten())
        if len(common) < 4:
            continue
        
        all_corners_a.append(cc_a)
        all_ids_a.append(ci_a)
        all_corners_b.append(cc_b)
        all_ids_b.append(ci_b)
        valid_pairs.append((cc_a, ci_a, cc_b, ci_b, common))
    
    print(f"Valid pairs: {len(valid_pairs)}/{len(frames_a)}")
    
    if len(valid_pairs) < 8:
        return None, f"Only {len(valid_pairs)} valid pairs (need 8+)"
    
    img_size = (frames_a[0].shape[1], frames_a[0].shape[0])
    
    print("\nCalibrating Camera A...")
    ret_a, K1, D1, _, _ = cv2.aruco.calibrateCameraCharuco(
        all_corners_a, all_ids_a, board, img_size, None, None
    )
    print(f"  RMS: {ret_a:.4f}")
    
    print("Calibrating Camera B...")
    ret_b, K2, D2, _, _ = cv2.aruco.calibrateCameraCharuco(
        all_corners_b, all_ids_b, board, img_size, None, None
    )
    print(f"  RMS: {ret_b:.4f}")
    
    obj_points = []
    img_points_a = []
    img_points_b = []
    
    for cc_a, ci_a, cc_b, ci_b, common in valid_pairs:
        obj_pts = []
        pts_a = []
        pts_b = []
        
        ci_a_flat = ci_a.flatten()
        ci_b_flat = ci_b.flatten()
        
        for cid in sorted(common):
            obj_pts.append(board_corners_3d[cid])
            idx_a = np.where(ci_a_flat == cid)[0][0]
            idx_b = np.where(ci_b_flat == cid)[0][0]
            pts_a.append(cc_a[idx_a].flatten())
            pts_b.append(cc_b[idx_b].flatten())
        
        obj_points.append(np.array(obj_pts, dtype=np.float32))
        img_points_a.append(np.array(pts_a, dtype=np.float32))
        img_points_b.append(np.array(pts_b, dtype=np.float32))
    
    print("\nStereo calibration...")
    ret_stereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_a, img_points_b,
        K1, D1, K2, D2, img_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    
    baseline = float(np.linalg.norm(T))
    print(f"  Stereo RMS: {ret_stereo:.4f}")
    print(f"  Baseline: {baseline:.3f}m")
    
    if ret_stereo > 10.0:
        return None, f"Stereo RMS {ret_stereo:.1f} too high!"
    
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'version': '6.0',
        'timestamp': datetime.now().isoformat(),
        'image_size': list(img_size),
        'K1': K1.tolist(), 'D1': D1.tolist(),
        'K2': K2.tolist(), 'D2': D2.tolist(),
        'R1': R1.tolist(), 'R2': R2.tolist(),
        'P1': P1.tolist(), 'P2': P2.tolist(),
        'R': R.tolist(), 'T': T.tolist(),
        'rms_cam_a': float(ret_a),
        'rms_cam_b': float(ret_b),
        'rms_stereo': float(ret_stereo),
        'baseline_meters': baseline,
        'floor_z_offset': 0.0,
        'floor_calibrated': False,
    }
    
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved: {CALIBRATION_FILE}")
    
    return data, f"CamA:{ret_a:.2f} CamB:{ret_b:.2f} Stereo:{ret_stereo:.2f} Base:{baseline:.2f}m"


def calibrate_floor(frame_a, frame_b, calib_data):
    """Calibrate floor from board lying flat on ground"""
    cc_a, ci_a, _ = detect_charuco(frame_a)
    cc_b, ci_b, _ = detect_charuco(frame_b)
    
    if cc_a is None:
        return False, "Board not detected in Camera A"
    if cc_b is None:
        return False, "Board not detected in Camera B"
    
    common = set(ci_a.flatten()) & set(ci_b.flatten())
    if len(common) < 4:
        return False, f"Only {len(common)} common corners"
    
    K1 = np.array(calib_data['K1'])
    D1 = np.array(calib_data['D1'])
    K2 = np.array(calib_data['K2'])
    D2 = np.array(calib_data['D2'])
    R1 = np.array(calib_data['R1'])
    R2 = np.array(calib_data['R2'])
    P1 = np.array(calib_data['P1'])
    P2 = np.array(calib_data['P2'])
    
    pts_a = []
    pts_b = []
    ci_a_flat = ci_a.flatten()
    ci_b_flat = ci_b.flatten()
    
    for cid in sorted(common):
        idx_a = np.where(ci_a_flat == cid)[0][0]
        idx_b = np.where(ci_b_flat == cid)[0][0]
        pts_a.append(cc_a[idx_a].flatten())
        pts_b.append(cc_b[idx_b].flatten())
    
    pts_a = np.array(pts_a, dtype=np.float32).reshape(-1, 1, 2)
    pts_b = np.array(pts_b, dtype=np.float32).reshape(-1, 1, 2)
    
    pts_a_rect = cv2.undistortPoints(pts_a, K1, D1, R=R1, P=P1)
    pts_b_rect = cv2.undistortPoints(pts_b, K2, D2, R=R2, P=P2)
    
    points_4d = cv2.triangulatePoints(P1, P2,
                                       pts_a_rect.reshape(-1, 2).T,
                                       pts_b_rect.reshape(-1, 2).T)
    points_3d = (points_4d[:3] / points_4d[3]).T
    
    blender_z = -points_3d[:, 1]
    floor_offset = -np.mean(blender_z)
    
    calib_data['floor_z_offset'] = float(floor_offset)
    calib_data['floor_calibrated'] = True
    
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(calib_data, f, indent=2)
    
    return True, f"Floor offset: {floor_offset:.3f}m"


# =============================================================================
# MOTION CAPTURE FUNCTIONS
# =============================================================================

class KalmanFilter3D:
    """Simple Kalman filter for 3D point smoothing - FIXED VERSION"""
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        self.kf = cv2.KalmanFilter(6, 3)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        self.initialized = False
    
    def update(self, measurement):
        """Update filter with measurement. Always returns a list of 3 floats."""
        if measurement is None:
            if self.initialized:
                pred = self.kf.predict()
                return [float(pred[0,0]), float(pred[1,0]), float(pred[2,0])]
            return None
        
        # Convert to numpy array - handle both list and array inputs
        if isinstance(measurement, (list, tuple)):
            m = np.array([[measurement[0]], [measurement[1]], [measurement[2]]], dtype=np.float32)
        else:
            m = np.array(measurement, dtype=np.float32).reshape(3, 1)
        
        if not self.initialized:
            self.kf.statePost = np.array([[m[0,0]], [m[1,0]], [m[2,0]], [0], [0], [0]], dtype=np.float32)
            self.initialized = True
            return [float(m[0,0]), float(m[1,0]), float(m[2,0])]
        
        self.kf.predict()
        corrected = self.kf.correct(m)
        return [float(corrected[0,0]), float(corrected[1,0]), float(corrected[2,0])]


def triangulate_landmarks(landmarks_a, landmarks_b, calib_data, img_size):
    """Triangulate MediaPipe landmarks to 3D"""
    K1 = np.array(calib_data['K1'])
    D1 = np.array(calib_data['D1'])
    K2 = np.array(calib_data['K2'])
    D2 = np.array(calib_data['D2'])
    R1 = np.array(calib_data['R1'])
    R2 = np.array(calib_data['R2'])
    P1 = np.array(calib_data['P1'])
    P2 = np.array(calib_data['P2'])
    floor_offset = calib_data.get('floor_z_offset', 0.0)
    
    w, h = img_size
    points_3d = {}
    
    for idx in range(33):
        lm_a = landmarks_a.landmark[idx]
        lm_b = landmarks_b.landmark[idx]
        
        # Skip low confidence landmarks
        if lm_a.visibility < 0.5 or lm_b.visibility < 0.5:
            continue
        
        pt_a = np.array([[lm_a.x * w, lm_a.y * h]], dtype=np.float32).reshape(-1, 1, 2)
        pt_b = np.array([[lm_b.x * w, lm_b.y * h]], dtype=np.float32).reshape(-1, 1, 2)
        
        pt_a_rect = cv2.undistortPoints(pt_a, K1, D1, R=R1, P=P1)
        pt_b_rect = cv2.undistortPoints(pt_b, K2, D2, R=R2, P=P2)
        
        point_4d = cv2.triangulatePoints(P1, P2,
                                          pt_a_rect.reshape(2, 1),
                                          pt_b_rect.reshape(2, 1))
        point_3d = (point_4d[:3] / point_4d[3]).flatten()
        
        # Convert OpenCV to Blender coordinates
        blender_x = point_3d[0]
        blender_y = point_3d[2]
        blender_z = -point_3d[1] + floor_offset
        
        points_3d[idx] = [float(blender_x), float(blender_y), float(blender_z)]
    
    return points_3d


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    print("\n" + "="*60)
    print("MELODICCAP STUDIO v6 - BUG FIXES")
    print("="*60)
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    # Open cameras
    print("\n[CAMERAS]")
    
    print(f"  Camera A (index {CAM_A_INDEX})...")
    cap_a = cv2.VideoCapture(CAM_A_INDEX, cv2.CAP_DSHOW)
    cap_a.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap_a.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap_a.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if cap_a.isOpened():
        print(f"    OK ({int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))})")
    else:
        print("    FAILED!")
        return
    
    print(f"  Camera B (index {CAM_B_INDEX})...")
    cap_b = cv2.VideoCapture(CAM_B_INDEX, cv2.CAP_ANY)
    cap_b.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap_b.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap_b.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if cap_b.isOpened():
        print(f"    OK ({int(cap_b.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap_b.get(cv2.CAP_PROP_FRAME_HEIGHT))})")
    else:
        print("    FAILED!")
        return
    
    # Load existing calibration
    calib_data = None
    if CALIBRATION_FILE.exists():
        try:
            with open(CALIBRATION_FILE) as f:
                calib_data = json.load(f)
            print(f"\n[CALIBRATION] Loaded (Stereo RMS: {calib_data.get('rms_stereo', '?'):.2f})")
            if calib_data.get('floor_calibrated'):
                print(f"[FLOOR] Offset: {calib_data.get('floor_z_offset', 0):.3f}m")
        except:
            pass
    
    print("\n[CONTROLS]")
    print("  C = Collect calibration frames")
    print("  S = Run stereo calibration")
    print("  F = Floor calibration")
    print("  R = Record motion capture")
    print("  Q = Quit")
    
    # State
    cal_frames_a = []
    cal_frames_b = []
    collecting_cal = False
    floor_mode = False
    recording = False
    countdown_active = False
    countdown_start = 0
    last_capture = 0
    status = "Ready" if calib_data else "Press C to calibrate"
    recorded_frames = []
    record_start_time = 0
    
    # Kalman filters
    kalman_filters = {}
    
    # Frame counter for debug
    frame_count = 0
    dropped_frames = 0
    
    while True:
        ret_a, frame_a = cap_a.read()
        ret_b, frame_b = cap_b.read()
        
        if not ret_a or not ret_b:
            continue
        
        frame_count += 1
        
        # Detect ChArUco (for calibration modes)
        cc_a, ci_a, markers_a = detect_charuco(frame_a)
        cc_b, ci_b, markers_b = detect_charuco(frame_b)
        
        board_a = cc_a is not None
        board_b = cc_b is not None
        
        # Create display frames
        if collecting_cal or floor_mode:
            display_a = draw_detection(frame_a, cc_a, ci_a, markers_a)
            display_b = draw_detection(frame_b, cc_b, ci_b, markers_b)
        else:
            display_a = frame_a.copy()
            display_b = frame_b.copy()
        
        # Process poses (only when not collecting calibration)
        pose_a_valid = False
        pose_b_valid = False
        
        if not collecting_cal and not floor_mode:
            results_a = pose.process(cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB))
            results_b = pose.process(cv2.cvtColor(frame_b, cv2.COLOR_BGR2RGB))
            
            pose_a_valid = results_a.pose_landmarks is not None
            pose_b_valid = results_b.pose_landmarks is not None
            
            # Draw skeletons
            if pose_a_valid:
                mp_draw.draw_landmarks(display_a, results_a.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if pose_b_valid:
                mp_draw.draw_landmarks(display_b, results_b.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Recording - only record when BOTH cameras have pose
            if recording and pose_a_valid and pose_b_valid and calib_data:
                try:
                    img_size = (frame_a.shape[1], frame_a.shape[0])
                    points_3d = triangulate_landmarks(
                        results_a.pose_landmarks,
                        results_b.pose_landmarks,
                        calib_data,
                        img_size
                    )
                    
                    # Apply Kalman filtering - FIXED: now always returns list
                    smoothed = {}
                    for idx, pt in points_3d.items():
                        if idx not in kalman_filters:
                            kalman_filters[idx] = KalmanFilter3D()
                        
                        filtered = kalman_filters[idx].update(pt)
                        if filtered is not None:
                            smoothed[idx] = filtered  # Already a list
                    
                    if len(smoothed) > 0:
                        recorded_frames.append({
                            'timestamp': time.time() - record_start_time,
                            'landmarks': smoothed,
                            'num_landmarks': len(smoothed)
                        })
                except Exception as e:
                    dropped_frames += 1
                    print(f"[WARN] Frame {frame_count}: {e}")
            
            elif recording and (not pose_a_valid or not pose_b_valid):
                dropped_frames += 1
        
        # Draw borders for calibration modes
        h_a, w_a = display_a.shape[:2]
        h_b, w_b = display_b.shape[:2]
        
        if collecting_cal or floor_mode:
            color_a = (0, 255, 0) if board_a else (0, 0, 255)
            color_b = (0, 255, 0) if board_b else (0, 0, 255)
            cv2.rectangle(display_a, (0, 0), (w_a-1, h_a-1), color_a, 8)
            cv2.rectangle(display_b, (0, 0), (w_b-1, h_b-1), color_b, 8)
        
        # Camera labels with tracking status
        cv2.rectangle(display_a, (0, 0), (200, 40), (0, 0, 0), -1)
        if collecting_cal or floor_mode:
            cv2.putText(display_a, f"CAM A ({markers_a}m)", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            color = (0, 255, 0) if pose_a_valid else (0, 0, 255)
            cv2.putText(display_a, f"CAM A {'OK' if pose_a_valid else 'LOST'}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.rectangle(display_b, (0, 0), (200, 40), (0, 0, 0), -1)
        if collecting_cal or floor_mode:
            cv2.putText(display_b, f"CAM B ({markers_b}m)", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            color = (0, 255, 0) if pose_b_valid else (0, 0, 255)
            cv2.putText(display_b, f"CAM B {'OK' if pose_b_valid else 'LOST'}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Collect calibration frames
        if collecting_cal and board_a and board_b:
            common = set(ci_a.flatten()) & set(ci_b.flatten())
            now = time.time()
            if len(common) >= 4 and now - last_capture > 0.4:
                cal_frames_a.append(frame_a.copy())
                cal_frames_b.append(frame_b.copy())
                last_capture = now
                status = f"Captured {len(cal_frames_a)} frames ({len(common)} common)"
                print(f"[CAL] Frame {len(cal_frames_a)}")
        
        # Floor calibration
        if floor_mode and board_a and board_b and calib_data:
            success, msg = calibrate_floor(frame_a, frame_b, calib_data)
            if success:
                floor_mode = False
                status = f"Floor OK: {msg}"
                print(f"[FLOOR] {msg}")
        
        # Countdown handling
        if countdown_active:
            remaining = COUNTDOWN_SECONDS - (time.time() - countdown_start)
            if remaining <= 0:
                countdown_active = False
                recording = True
                record_start_time = time.time()
                recorded_frames = []
                kalman_filters = {}
                dropped_frames = 0
                status = "RECORDING..."
                print("[REC] Started")
        
        # Resize and combine
        scale = 0.5
        disp_a = cv2.resize(display_a, None, fx=scale, fy=scale)
        disp_b = cv2.resize(display_b, None, fx=scale, fy=scale)
        combined = np.hstack([disp_a, disp_b])
        
        # Draw countdown
        if countdown_active:
            remaining = COUNTDOWN_SECONDS - (time.time() - countdown_start)
            count = max(1, int(remaining) + 1)
            h, w = combined.shape[:2]
            cv2.rectangle(combined, (w//2-60, h//2-60), (w//2+60, h//2+60), (0, 0, 0), -1)
            cv2.putText(combined, str(count), (w//2-25, h//2+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 4)
        
        # Recording indicator
        if recording:
            h, w = combined.shape[:2]
            cv2.circle(combined, (w-30, 30), 15, (0, 0, 255), -1)
            elapsed = time.time() - record_start_time
            cv2.putText(combined, f"{len(recorded_frames)}f {elapsed:.1f}s", (w-150, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if dropped_frames > 0:
                cv2.putText(combined, f"drop:{dropped_frames}", (w-150, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        
        # Status bar
        bar = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(bar, f"Cal frames: {len(cal_frames_a)}", (combined.shape[1]-180, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        combined = np.vstack([combined, bar])
        cv2.imshow("MelodicCap Studio v6", combined)
        
        # Key handling
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('c'):
            if collecting_cal:
                collecting_cal = False
                status = f"Collected {len(cal_frames_a)} frames. Press S to calibrate."
            else:
                cal_frames_a = []
                cal_frames_b = []
                collecting_cal = True
                floor_mode = False
                status = "CALIBRATING - Show board to BOTH cameras"
        
        elif key == ord('s'):
            if len(cal_frames_a) < 8:
                status = f"Need 8+ frames (have {len(cal_frames_a)})"
            else:
                collecting_cal = False
                status = "Running calibration..."
                cv2.waitKey(100)
                
                result, msg = run_stereo_calibration(cal_frames_a, cal_frames_b)
                if result:
                    calib_data = result
                    status = f"SUCCESS! {msg}"
                else:
                    status = f"FAILED: {msg}"
        
        elif key == ord('f'):
            if calib_data is None:
                status = "Calibrate cameras first!"
            else:
                floor_mode = not floor_mode
                status = "FLOOR MODE - Lay board flat" if floor_mode else "Floor mode off"
        
        elif key == ord('r'):
            if calib_data is None:
                status = "Calibrate cameras first!"
            elif recording:
                # Stop recording and save
                recording = False
                
                if len(recorded_frames) > 0:
                    TAKES_DIR.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = TAKES_DIR / f"take_{timestamp}.json"
                    
                    take_data = {
                        'version': '6.0',
                        'timestamp': datetime.now().isoformat(),
                        'duration_seconds': time.time() - record_start_time,
                        'frame_count': len(recorded_frames),
                        'dropped_frames': dropped_frames,
                        'calibration': {
                            'rms_stereo': calib_data.get('rms_stereo'),
                            'baseline': calib_data.get('baseline_meters'),
                            'floor_offset': calib_data.get('floor_z_offset', 0)
                        },
                        'frames': recorded_frames
                    }
                    
                    with open(filename, 'w') as f:
                        json.dump(take_data, f)
                    
                    status = f"Saved: {filename.name} ({len(recorded_frames)} frames)"
                    print(f"[SAVED] {filename}")
                    print(f"        {len(recorded_frames)} frames, {dropped_frames} dropped")
                else:
                    status = "No frames recorded"
            else:
                # Start countdown
                countdown_active = True
                countdown_start = time.time()
                status = "GET READY..."
    
    cap_a.release()
    cap_b.release()
    cv2.destroyAllWindows()
    print("\n[DONE]")


if __name__ == "__main__":
    main()
