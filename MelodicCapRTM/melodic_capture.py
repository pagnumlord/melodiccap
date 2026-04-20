"""
MelodicCap RTM - Motion Capture Studio
========================================
Dual-camera markerless motion capture using RTMPose/RTMW.

Replaces MediaPipe with rtmlib for significantly better pose estimation.
Everything else (calibration, triangulation, recording) is the same proven pipeline.

Hardware:
- Camera A (Sony ZV-1F): default index 2
- Camera B (Samsung S25 via DroidCam): default index 0
- GPU: RTX 3060 Ti (CUDA acceleration for RTMW)

Usage (first-run / hardware changed):
1. Run this script
2. Press 'I' to capture per-camera intrinsics for Camera A. Fill the 3x3
   grid overlay by holding the ChArUco board in each quadrant at varied
   distances and tilts. Press 'S' to solve + save, auto-advances to the
   next camera. Repeat until all cameras are done.
3. Press 'C' to collect paired calibration frames (show the board to both
   cameras at once at various positions).
4. Press 'S' to run stereo calibration. If intrinsics JSONs exist, only
   extrinsics are solved (much faster, much more accurate).
5. Press 'F' (or 'G') to calibrate floor.
6. Press 'R' to start/stop recording.
7. Press 'Q' to quit.

Subsequent sessions where only the cameras moved (intrinsics unchanged):
skip step 2 — the saved intrinsics get reused automatically.

Output: JSON files in takes/ directory, ready for Blender import.
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

from keypoint_map import LM, BODY_SKELETON
from pose_detector import PoseDetector
from stereo_calibration import StereoCalibration, save_intrinsics, load_intrinsics
from recorder import MocapRecorder


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """All settings in one place."""

    # Camera indices (change these to match your system)
    # A + B are REQUIRED, C is optional. Put most reliable cameras as A/B.
    CAM_A_INDEX = 0  # Samsung S25 DroidCam
    CAM_B_INDEX = 1  # Logitech C615
    CAM_C_INDEX = -1  # Sony ZV-1F (set to -1 to disable third camera)

    # Resolution
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FPS = 30

    # Paths — auto-detects based on where this script lives
    # On your PC this resolves to C:\Users\ninja\Documents\MelodicCapStudio\MelodicCapRTM
    # (assuming you cloned/copied this folder there)
    BASE_DIR = Path(__file__).resolve().parent
    CALIBRATION_FILE = BASE_DIR / "calibration" / "stereo_calibration.json"
    # Multi-pair calibration files (3-camera mode)
    CALIBRATION_FILE_AB = BASE_DIR / "calibration" / "stereo_calibration_ab.json"
    CALIBRATION_FILE_AC = BASE_DIR / "calibration" / "stereo_calibration_ac.json"
    CALIBRATION_FILE_BC = BASE_DIR / "calibration" / "stereo_calibration_bc.json"
    # Per-camera intrinsics files (captured via the `I` key — stand-alone,
    # not tied to any stereo pair). Loaded by the `S` key to short-circuit
    # the monocular step of stereo calibration.
    INTRINSICS_FILE_A = BASE_DIR / "calibration" / "intrinsics_a.json"
    INTRINSICS_FILE_B = BASE_DIR / "calibration" / "intrinsics_b.json"
    INTRINSICS_FILE_C = BASE_DIR / "calibration" / "intrinsics_c.json"
    TAKES_DIR = BASE_DIR / "takes"

    # Per-camera intrinsics capture gates
    INTRINSICS_MIN_CELLS = 8      # 3x3 grid; require at least 8 of 9 filled
    INTRINSICS_MIN_FRAMES = 25    # enough views to solve K+D reliably
    INTRINSICS_MAX_RMS = 1.0      # reject if solver reports worse than this

    # ChArUco board (must match your printed board)
    # Defaults are the small A4 5x7 board at 35mm squares. If you print a
    # larger board (recommended: A2/A1 foamcore), drop a board_config.json
    # next to this file and the loader below will override these at startup.
    # IMPORTANT: measure your actual printed square size with calipers.
    CHARUCO_SQUARES_X = 5
    CHARUCO_SQUARES_Y = 7
    CHARUCO_SQUARE_SIZE = 0.035    # 35mm target — MEASURE YOUR PRINT
    CHARUCO_MARKER_SIZE = 0.025    # 25mm target — MEASURE YOUR PRINT
    ARUCO_DICT = cv2.aruco.DICT_4X4_50
    BOARD_CONFIG_FILE = None  # populated in __post_init__ below

    # Recording
    RECORDING_COUNTDOWN = 5  # seconds before recording starts
    RECORDING_TRIM_END = 2.0  # seconds to trim from end of recording
    OFFLINE_MODE = True   # Save raw 2D only, triangulate later with offline_processor.py

    # Pose detection
    # 'body' = RTMPose 17 keypoints (fast, all we need for IK)
    # 'wholebody' = RTMW 133 keypoints (slower, adds fingers/face we don't use)
    # Body mode gives 2-3x better FPS. Switch to wholebody only if you
    # need finger data and have the GPU headroom.
    POSE_MODE = 'body'
    POSE_QUALITY = 'fast'       # 'fast' (lightweight, best FPS), 'balanced', or 'accurate'
    POSE_DEVICE = 'cuda'        # 'cuda' or 'cpu'
    MIN_KEYPOINT_CONFIDENCE = 0.3

    # Skip face (23-90) and hand (91-132) keypoints during triangulation.
    # Only relevant in wholebody mode. In body mode there's nothing to skip.
    SKIP_FACE_HANDS = True

    # Smoothing
    KALMAN_PROCESS_NOISE = 1e-4
    KALMAN_MEASUREMENT_NOISE = 1e-2


def _load_board_config(config):
    """Override ChArUco board geometry from calibration/board_config.json.

    Called once from MelodicCapApp.__init__. Lets the user swap to a bigger
    printed board (A2/A1 foamcore) without editing source or restarting
    through an IDE — just edit the JSON and relaunch.

    JSON schema (all keys optional, fall back to Config defaults):
        {
          "squares_x": 5,
          "squares_y": 7,
          "square_size_m": 0.080,
          "marker_size_m": 0.060,
          "aruco_dict": "DICT_4X4_50",
          "notes": "Printed on A1 foamcore at Staples, measured 80.1mm"
        }
    """
    path = config.BASE_DIR / "calibration" / "board_config.json"
    if not path.exists():
        print(f"[BOARD] Using hardcoded defaults: "
              f"{config.CHARUCO_SQUARES_X}x{config.CHARUCO_SQUARES_Y} @ "
              f"{config.CHARUCO_SQUARE_SIZE*1000:.0f}mm")
        print(f"        (drop a board_config.json in calibration/ to override)")
        return

    try:
        import json
        with open(path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[BOARD] Failed to read {path}: {e}")
        return

    if 'squares_x' in data:
        config.CHARUCO_SQUARES_X = int(data['squares_x'])
    if 'squares_y' in data:
        config.CHARUCO_SQUARES_Y = int(data['squares_y'])
    if 'square_size_m' in data:
        config.CHARUCO_SQUARE_SIZE = float(data['square_size_m'])
    if 'marker_size_m' in data:
        config.CHARUCO_MARKER_SIZE = float(data['marker_size_m'])
    if 'aruco_dict' in data:
        dict_name = str(data['aruco_dict'])
        dict_attr = getattr(cv2.aruco, dict_name, None)
        if dict_attr is not None:
            config.ARUCO_DICT = dict_attr
        else:
            print(f"[BOARD] Unknown aruco_dict '{dict_name}' — keeping default")

    config.BOARD_CONFIG_FILE = str(path)
    print(f"[BOARD] Loaded {path.name}: "
          f"{config.CHARUCO_SQUARES_X}x{config.CHARUCO_SQUARES_Y} @ "
          f"{config.CHARUCO_SQUARE_SIZE*1000:.1f}mm squares, "
          f"{config.CHARUCO_MARKER_SIZE*1000:.1f}mm markers")
    if 'notes' in data:
        print(f"        Notes: {data['notes']}")


# =============================================================================
# DRAWING UTILITIES
# =============================================================================

# Colors (BGR)
COLOR_SKELETON = (0, 255, 0)
COLOR_JOINT = (0, 200, 255)
COLOR_LOW_CONF = (100, 100, 100)
COLOR_HAND = (255, 180, 0)
COLOR_FOOT = (0, 255, 255)


def draw_detections(frame, detections, min_conf=0.3):
    """
    Draw detected keypoints and skeleton on frame.

    Args:
        frame: BGR image (modified in place)
        detections: dict of {idx: (px, py, conf)}
        min_conf: minimum confidence to draw
    """
    if detections is None:
        return frame

    h, w = frame.shape[:2]

    # Draw skeleton connections
    for idx_a, idx_b in BODY_SKELETON:
        if idx_a in detections and idx_b in detections:
            pa = detections[idx_a]
            pb = detections[idx_b]
            if pa[2] >= min_conf and pb[2] >= min_conf:
                pt_a = (int(pa[0]), int(pa[1]))
                pt_b = (int(pb[0]), int(pb[1]))
                # Feet get a different color
                if idx_a >= LM.LEFT_BIG_TOE:
                    color = COLOR_FOOT
                else:
                    color = COLOR_SKELETON
                cv2.line(frame, pt_a, pt_b, color, 2)

    # Draw keypoints
    for idx, (px, py, conf) in detections.items():
        if conf < min_conf:
            continue
        pt = (int(px), int(py))
        # Color by body part
        if 91 <= idx <= 132:
            color = COLOR_HAND
            radius = 2
        elif 17 <= idx <= 22:
            color = COLOR_FOOT
            radius = 3
        elif 23 <= idx <= 90:
            continue  # Skip face keypoints in display (too cluttered)
        else:
            color = COLOR_JOINT
            radius = 3
        cv2.circle(frame, pt, radius, color, -1)

    return frame


def draw_status(frame, text, color=(0, 255, 0)):
    """Draw status text on frame."""
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2)


# =============================================================================
# NON-BLOCKING CAMERA READER
# =============================================================================

class _CamCReader:
    """Background frame reader for optional Camera C.

    Continuously reads frames in a daemon thread. The main loop
    retrieves the latest frame without blocking. Each frame is only
    returned once (consumed on read), so a 1fps camera doesn't flood
    the pipeline with duplicate frames.
    """

    def __init__(self, cap):
        self.cap = cap
        self._frame = None
        self._new = False
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
                    self._new = True

    def read(self):
        """Get latest NEW frame. Non-blocking.

        Returns (True, frame) if a new frame is available.
        Returns (False, None) if no new frame since last read.
        """
        with self._lock:
            if self._new and self._frame is not None:
                self._new = False
                return True, self._frame.copy()
            return False, None

    def stop(self):
        """Stop background thread and release camera."""
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class MelodicCapApp:
    """Main motion capture application."""

    def __init__(self):
        self.config = Config()

        # Apply board_config.json overrides BEFORE any StereoCalibration is
        # constructed, because StereoCalibration.__init__ builds the ChArUco
        # board from these values.
        _load_board_config(self.config)

        # Track if user wanted CUDA (to warn if we fell back)
        self._cuda_was_requested = (self.config.POSE_DEVICE == 'cuda')

        # Thread pool for parallel inference on both cameras
        self._infer_pool = ThreadPoolExecutor(max_workers=3)

        # Initialize pose detector
        print("\n[INITIALIZING POSE DETECTOR]")
        try:
            self.detector = PoseDetector(
                mode=self.config.POSE_MODE,
                quality=self.config.POSE_QUALITY,
                device=self.config.POSE_DEVICE,
            )
        except Exception as e:
            print(f"[WARNING] CUDA init failed ({e}), falling back to CPU")
            self.config.POSE_DEVICE = 'cpu'
            self.detector = PoseDetector(
                mode=self.config.POSE_MODE,
                quality=self.config.POSE_QUALITY,
                device='cpu',
            )

        # Check if CUDA is actually being used (onnxruntime may silently fall back)
        if self.config.POSE_DEVICE == 'cuda':
            try:
                import onnxruntime as ort
                sess_providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' not in sess_providers:
                    print("[WARNING] CUDA provider not available, running on CPU")
                    self.config.POSE_DEVICE = 'cpu'
                else:
                    # Test if CUDA actually loads (provider may be listed but fail)
                    try:
                        test_sess = ort.InferenceSession(
                            str(next(Path.home().glob('.cache/rtmlib/**/*.onnx'))),
                            providers=['CUDAExecutionProvider']
                        )
                        actual = test_sess.get_providers()
                        del test_sess
                        if 'CUDAExecutionProvider' not in actual:
                            print("[WARNING] CUDA provider listed but failed to load — running on CPU")
                            self.config.POSE_DEVICE = 'cpu'
                    except Exception:
                        print("[WARNING] CUDA provider failed to initialize — running on CPU")
                        self.config.POSE_DEVICE = 'cpu'
            except ImportError:
                pass

        # Cameras
        self.cap_a = None
        self.cap_b = None
        self.cap_c = None
        self.has_cam_c = False  # True if third camera opened successfully
        self._cam_c_reader = None  # Background thread reader for Camera C

        # Systems — primary calibration (AB pair, used for real-time preview)
        self.calibration = StereoCalibration(self.config)
        # Additional calibrations for multi-pair offline triangulation
        self.calibration_ac = StereoCalibration(self.config)
        self.calibration_bc = StereoCalibration(self.config)
        self.recorder = MocapRecorder(self.config.TAKES_DIR)

        # Calibration frame collection — per-pair for 3-camera support
        # Each pair collects frames independently (board may not be visible
        # to all 3 cameras simultaneously at every angle)
        self.cal_frames_ab = ([], [])  # (cam_a_frames, cam_b_frames)
        self.cal_frames_ac = ([], [])  # (cam_a_frames, cam_c_frames)
        self.cal_frames_bc = ([], [])  # (cam_b_frames, cam_c_frames)
        self.collecting_cal_frames = False
        self._last_cal_time = 0
        self._last_cal_time_ac = 0
        self._last_cal_time_bc = 0
        # Stillness detection state per pair (set by _check_pair_stillness)
        self._prev_corners_ab = None
        self._prev_corners_ac = None
        self._prev_corners_bc = None

        # Floor calibration mode
        self.floor_cal_active = False

        # Auto floor calibration countdown
        self._floor_auto_countdown = False
        self._floor_auto_start = 0

        # Per-camera intrinsics capture mode (I key).
        # None = off, 'A'/'B'/'C' = that camera's capture is active.
        # Intrinsics are stored as lists of (corners, ids) so we can call
        # cv2.aruco.calibrateCameraCharuco directly when the user presses S.
        self._intrinsics_mode = None
        self._intrinsics_corners = {'A': [], 'B': [], 'C': []}  # list of charuco corners arrays
        self._intrinsics_ids = {'A': [], 'B': [], 'C': []}      # list of charuco ids arrays
        self._intrinsics_coverage = {
            'A': np.zeros((3, 3), dtype=int),
            'B': np.zeros((3, 3), dtype=int),
            'C': np.zeros((3, 3), dtype=int),
        }
        self._intrinsics_last_cell = {'A': None, 'B': None, 'C': None}
        self._intrinsics_prev_corners = {'A': None, 'B': None, 'C': None}
        self._intrinsics_last_cap_time = {'A': 0.0, 'B': 0.0, 'C': 0.0}

        # State
        self.running = True
        self._countdown_start = 0
        self._countdown_active = False

    def _open_camera(self, index, label):
        """Open a single camera by index, trying multiple backends.

        Tries DirectShow first, then Media Foundation, then default.
        Picks the backend that delivers frames fastest at the right resolution.
        Returns (cap, success).
        """
        import time as _time
        print(f"  Opening Camera {label} (index {index})...")

        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (None, "default"),
        ]

        best_cap = None
        best_info = None
        best_score = -1  # Higher is better

        for backend_id, backend_name in backends:
            try:
                if backend_id is not None:
                    cap = cv2.VideoCapture(index, backend_id)
                else:
                    cap = cv2.VideoCapture(index)
            except Exception:
                continue

            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, self.config.FPS)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            # Test: read one frame and check resolution + speed
            t0 = _time.time()
            ret, frame = cap.read()
            read_ms = (_time.time() - t0) * 1000

            if not ret or frame is None:
                cap.release()
                continue

            actual_w = frame.shape[1]
            actual_h = frame.shape[0]
            res_match = (actual_w == self.config.FRAME_WIDTH and
                         actual_h == self.config.FRAME_HEIGHT)

            # Score: prefer correct resolution + fast reads
            score = 0
            if res_match:
                score += 100
            if read_ms < 200:
                score += 50
            elif read_ms < 500:
                score += 20

            info = f"{backend_name}: {actual_w}x{actual_h}, {read_ms:.0f}ms"
            print(f"    [{backend_name}] {actual_w}x{actual_h}, {read_ms:.0f}ms"
                  f" {'OK' if res_match else 'res mismatch'}")

            if score > best_score:
                # Release previous best
                if best_cap is not None:
                    best_cap.release()
                best_cap = cap
                best_info = info
                best_score = score
            else:
                cap.release()

            # If we got correct resolution, stop trying — don't risk
            # other backends hanging (MSMF can hang 30+ seconds on some cameras)
            if res_match:
                break

        if best_cap is not None and best_cap.isOpened():
            print(f"    [OK] Camera {label} opened ({best_info})")
            return best_cap, True
        else:
            print(f"    [ERROR] Failed to open Camera {label} on any backend")
            return cv2.VideoCapture(), False

    def init_cameras(self):
        """Initialize cameras (A + B required, C optional)."""
        print("\n[INITIALIZING CAMERAS]")

        self.cap_a, ok_a = self._open_camera(self.config.CAM_A_INDEX, "A (DroidCam)")
        if not ok_a:
            return False

        self.cap_b, ok_b = self._open_camera(self.config.CAM_B_INDEX, "B (Logitech)")
        if not ok_b:
            return False

        # Camera C is optional — skip if index is -1 or fails to open
        if self.config.CAM_C_INDEX >= 0:
            self.cap_c, ok_c = self._open_camera(self.config.CAM_C_INDEX, "C (Sony ZV-1F)")
            if ok_c:
                ok_c = self._validate_optional_camera(self.cap_c, "C")
            self.has_cam_c = ok_c
            if not ok_c:
                print(f"    [WARNING] Camera C disabled — continuing with 2 cameras")
                if self.cap_c is not None:
                    self.cap_c.release()
                    self.cap_c = None
        else:
            print(f"  Camera C disabled (CAM_C_INDEX = -1)")

        return True

    def _validate_optional_camera(self, cap, label):
        """Validate an optional camera delivers correct resolution at acceptable speed.

        Returns False (camera should be disabled) if:
        - Resolution doesn't match expected FRAME_WIDTH x FRAME_HEIGHT
        - Average frame delivery > 300ms (would block main loop)
        """
        # Check resolution
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"    [FAIL] Camera {label}: can't deliver frames")
            return False

        actual_w, actual_h = frame.shape[1], frame.shape[0]
        if actual_w != self.config.FRAME_WIDTH or actual_h != self.config.FRAME_HEIGHT:
            print(f"    [FAIL] Camera {label}: resolution {actual_w}x{actual_h} "
                  f"!= expected {self.config.FRAME_WIDTH}x{self.config.FRAME_HEIGHT}")
            print(f"    Check camera mode/settings. Sony ZV-1F: use 'i sq' mode for 720p.")
            return False

        # Check frame delivery speed (3 frames, skip first for warmup)
        times = []
        for _ in range(4):
            t0 = time.time()
            ret, _ = cap.read()
            if ret:
                times.append(time.time() - t0)

        if len(times) < 2:
            print(f"    [FAIL] Camera {label}: stopped delivering frames during speed test")
            return False

        avg_ms = np.mean(times[1:]) * 1000  # Skip first (cold start)
        if avg_ms > 300:
            print(f"    [FAIL] Camera {label}: too slow ({avg_ms:.0f}ms/frame avg)")
            print(f"    Would block main loop. Check USB cable/port and camera settings.")
            return False

        print(f"    [OK] Camera {label} validated: {actual_w}x{actual_h}, {avg_ms:.0f}ms/frame avg")
        return True

    def _check_pair_stillness(self, corners_1, ids_1, corners_2, ids_2,
                                  prev_attr, threshold=2.0):
        """Check if board is held still in a camera pair.

        Compares current corners with previous frame's corners. Returns True
        if max corner movement is below threshold for both cameras.

        Args:
            corners_1, ids_1: ChArUco corners/ids from camera 1
            corners_2, ids_2: ChArUco corners/ids from camera 2
            prev_attr: attribute name on self to store/retrieve previous corners
            threshold: max pixel movement for "still"
        """
        prev = getattr(self, prev_attr, None)
        setattr(self, prev_attr, (corners_1, ids_1, corners_2, ids_2))

        if prev is None or len(corners_1) < 6 or len(corners_2) < 6:
            return False

        prev_c1, prev_id1, prev_c2, prev_id2 = prev
        if prev_id1 is None or prev_id2 is None:
            return False

        # Check stillness in camera 1
        common_1 = set(ids_1.flatten()) & set(prev_id1.flatten())
        common_2 = set(ids_2.flatten()) & set(prev_id2.flatten())
        if len(common_1) < 4 or len(common_2) < 4:
            return False

        move_1 = 0
        for cid in common_1:
            idx_cur = np.where(ids_1.flatten() == cid)[0][0]
            idx_prev = np.where(prev_id1.flatten() == cid)[0][0]
            move_1 = max(move_1, np.linalg.norm(
                corners_1[idx_cur].flatten() - prev_c1[idx_prev].flatten()))

        move_2 = 0
        for cid in common_2:
            idx_cur = np.where(ids_2.flatten() == cid)[0][0]
            idx_prev = np.where(prev_id2.flatten() == cid)[0][0]
            move_2 = max(move_2, np.linalg.norm(
                corners_2[idx_cur].flatten() - prev_c2[idx_prev].flatten()))

        return move_1 < threshold and move_2 < threshold

    # ─────────────────────────────────────────────────────────────────
    # Per-camera intrinsics capture (I key)
    # ─────────────────────────────────────────────────────────────────

    def _intrinsics_cycle(self):
        """Advance `I` key through off → A → B → C → off.

        Skips Camera C when it's unavailable.
        """
        order = [None, 'A', 'B']
        if self.has_cam_c:
            order.append('C')
        try:
            i = order.index(self._intrinsics_mode)
        except ValueError:
            i = 0
        self._intrinsics_mode = order[(i + 1) % len(order)]

        if self._intrinsics_mode is None:
            print("\n[INTRINSICS] Capture OFF")
        else:
            label = self._intrinsics_mode
            n_frames = len(self._intrinsics_corners[label])
            n_cells = int(np.count_nonzero(self._intrinsics_coverage[label]))
            print(f"\n[INTRINSICS] Capturing Camera {label}")
            print(f"  Fill the 3x3 grid by moving the board into every quadrant,")
            print(f"  at different distances and tilts. Hold still between moves.")
            print(f"  Goal: {self.config.INTRINSICS_MIN_CELLS}/9 cells, "
                  f"{self.config.INTRINSICS_MIN_FRAMES}+ frames.")
            print(f"  Current progress: {n_cells}/9 cells, {n_frames} frames.")
            print(f"  Press S to solve + save, or I to advance to the next camera.")

    def _intrinsics_grid_cell(self, corners, img_size):
        """Return (row, col) cell index 0..2 for the board centroid."""
        pts = corners.reshape(-1, 2)
        if len(pts) == 0:
            return None
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        w, h = img_size
        col = max(0, min(2, int(cx / (w / 3))))
        row = max(0, min(2, int(cy / (h / 3))))
        return (row, col)

    def _intrinsics_still(self, label, corners, ids, threshold=2.0):
        """Single-camera stillness check adapted from `_check_pair_stillness`."""
        prev = self._intrinsics_prev_corners[label]
        self._intrinsics_prev_corners[label] = (corners, ids)
        if prev is None:
            return False
        prev_c, prev_id = prev
        if prev_id is None or ids is None:
            return False
        common = set(ids.flatten()) & set(prev_id.flatten())
        if len(common) < 4:
            return False
        max_move = 0.0
        for cid in common:
            idx_cur = np.where(ids.flatten() == cid)[0][0]
            idx_prev = np.where(prev_id.flatten() == cid)[0][0]
            max_move = max(max_move, float(np.linalg.norm(
                corners[idx_cur].flatten() - prev_c[idx_prev].flatten())))
        return max_move < threshold

    def _intrinsics_maybe_capture(self, frame):
        """Detect board on the active intrinsics camera and maybe store it.

        Returns the same frame with a 3x3 grid overlay and HUD text drawn
        on it, ready for display. No-op if mode is off.
        """
        label = self._intrinsics_mode
        if label is None:
            return frame

        img_size = (self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT)
        display = frame.copy()

        # Draw the 3x3 grid with green fills for cells already covered.
        cov = self._intrinsics_coverage[label]
        w, h = img_size
        for r in range(3):
            for c in range(3):
                x0 = int(c * w / 3)
                y0 = int(r * h / 3)
                x1 = int((c + 1) * w / 3)
                y1 = int((r + 1) * h / 3)
                if cov[r, c] >= 2:
                    overlay = display.copy()
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 180, 0), -1)
                    cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
                cv2.rectangle(display, (x0, y0), (x1 - 1, y1 - 1),
                              (80, 80, 80), 1)

        corners, ids = self.calibration.detect_charuco(frame)
        if corners is not None:
            cv2.aruco.drawDetectedCornersCharuco(display, corners)
            cell = self._intrinsics_grid_cell(corners, img_size)
            still = self._intrinsics_still(label, corners, ids)
            now = time.time()
            cooldown_ok = (now - self._intrinsics_last_cap_time[label]) > 0.5
            new_cell = (cell is not None and cell != self._intrinsics_last_cell[label])
            cell_wants_more = (cell is not None and cov[cell[0], cell[1]] < 3)

            if still and cooldown_ok and (new_cell or cell_wants_more):
                # Only store frames with at least 6 corners — same bar used
                # in calibrate_stereo's pair capture.
                if len(corners) >= 8:
                    self._intrinsics_corners[label].append(corners.copy())
                    self._intrinsics_ids[label].append(ids.copy())
                    cov[cell[0], cell[1]] += 1
                    self._intrinsics_last_cell[label] = cell
                    self._intrinsics_last_cap_time[label] = now
                    n_frames = len(self._intrinsics_corners[label])
                    n_cells = int(np.count_nonzero(cov))
                    print(f"  [I/{label}] Captured at cell {cell} "
                          f"({n_frames} frames, {n_cells}/9 cells)")

        # HUD
        n_frames = len(self._intrinsics_corners[label])
        n_cells = int(np.count_nonzero(cov))
        hud = (f"INTRINSICS {label}: {n_cells}/9 cells, {n_frames} frames  "
               f"(I=next cam  S=solve+save)")
        draw_status(display, hud, (0, 255, 255))
        return display

    def _intrinsics_solve_and_save(self):
        """Solve the monocular intrinsics for the active camera and save to JSON."""
        label = self._intrinsics_mode
        if label is None:
            return

        n_frames = len(self._intrinsics_corners[label])
        n_cells = int(np.count_nonzero(self._intrinsics_coverage[label]))
        if n_frames < self.config.INTRINSICS_MIN_FRAMES:
            print(f"\n[INTRINSICS] Camera {label}: only {n_frames} frames "
                  f"(need {self.config.INTRINSICS_MIN_FRAMES}+). Keep capturing.")
            return
        if n_cells < self.config.INTRINSICS_MIN_CELLS:
            print(f"\n[INTRINSICS] Camera {label}: only {n_cells}/9 grid cells "
                  f"filled (need {self.config.INTRINSICS_MIN_CELLS}+). "
                  f"Move the board into the empty quadrants.")
            return

        img_size = (self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT)

        # Filter out frames where the detected corners are nearly collinear
        # on the board grid.  OpenCV's initIntrinsicParams2D needs a valid
        # homography from each frame, and that requires corners spanning at
        # least 2 board rows AND 2 board columns.
        cols_per_row = self.config.CHARUCO_SQUARES_X - 1  # inner corners per row
        good_corners = []
        good_ids = []
        for corners_i, ids_i in zip(self._intrinsics_corners[label],
                                    self._intrinsics_ids[label]):
            flat_ids = ids_i.ravel()
            rows_seen = set(int(cid) // cols_per_row for cid in flat_ids)
            cols_seen = set(int(cid) % cols_per_row for cid in flat_ids)
            if len(rows_seen) >= 2 and len(cols_seen) >= 2:
                good_corners.append(corners_i)
                good_ids.append(ids_i)

        n_dropped = n_frames - len(good_corners)
        if n_dropped:
            print(f"\n[INTRINSICS] Dropped {n_dropped}/{n_frames} frames "
                  f"with nearly-collinear corners.")
        if len(good_corners) < self.config.INTRINSICS_MIN_FRAMES:
            print(f"\n[INTRINSICS] Camera {label}: only {len(good_corners)} "
                  f"usable frames after filtering (need "
                  f"{self.config.INTRINSICS_MIN_FRAMES}+). "
                  f"Capture more frames with the full board visible.")
            return

        print(f"\n[INTRINSICS] Solving Camera {label} from "
              f"{len(good_corners)} frames ({n_dropped} dropped)...")
        try:
            rms, K, D, _, _ = cv2.aruco.calibrateCameraCharuco(
                good_corners,
                good_ids,
                self.calibration.charuco_board,
                img_size, None, None,
            )
        except cv2.error as e:
            print(f"  [FAILED] {e}")
            print(f"  Try capturing more varied tilts/distances and retry.")
            return

        print(f"  RMS: {rms:.4f}")
        if rms > self.config.INTRINSICS_MAX_RMS:
            print(f"  [REJECTED] RMS {rms:.3f} > {self.config.INTRINSICS_MAX_RMS:.2f}. "
                  f"Re-capture with slower, steadier board positions.")
            return

        out_path = {
            'A': self.config.INTRINSICS_FILE_A,
            'B': self.config.INTRINSICS_FILE_B,
            'C': self.config.INTRINSICS_FILE_C,
        }[label]
        save_intrinsics(
            out_path, K, D, rms, img_size,
            coverage_map=self._intrinsics_coverage[label],
            n_frames=n_frames,
            camera_label=label,
        )
        print(f"  [OK] Saved {out_path.name}")

        # Clear the buffer so the user can recapture this camera if they
        # want, and auto-advance to the next camera so a single S press
        # keeps the flow going.
        self._intrinsics_corners[label] = []
        self._intrinsics_ids[label] = []
        self._intrinsics_coverage[label][:] = 0
        self._intrinsics_last_cell[label] = None
        self._intrinsics_prev_corners[label] = None

        # Advance to the next camera that still needs capture.
        self._intrinsics_cycle()

    def _propagate_floor_offset(self, offset):
        """Write the new floor offset into every pair calibration file.

        The F/G keys used to only update `stereo_calibration.json` (the
        "primary" file) and the in-memory self.calibration. Chain derivation
        reads `stereo_calibration_ab.json`, so it saw stale floor=0 even
        after a fresh F press — the user then had to manually patch the
        AB JSON before re-running chain_calibration.py. Propagating to
        every pair file closes that gap.
        """
        pair_files = [
            self.config.CALIBRATION_FILE_AB,
            self.config.CALIBRATION_FILE_AC,
            self.config.CALIBRATION_FILE_BC,
        ]
        touched = []
        import json
        for p in pair_files:
            if not p.exists():
                continue
            try:
                with open(p, 'r') as f:
                    data = json.load(f)
                data['floor_z_offset'] = float(offset)
                with open(p, 'w') as f:
                    json.dump(data, f, indent=2)
                touched.append(p.name)
            except Exception as e:
                print(f"  [WARN] Failed to propagate floor to {p.name}: {e}")
        if touched:
            print(f"  [OK] Floor offset {offset:.3f}m propagated to "
                  f"{', '.join(touched)}")

        # If a chain-derived AC now needs refreshing (its rectification
        # matrices were computed with the old floor), re-derive so the
        # updated floor flows through end-to-end. Best-effort — if it
        # fails, the user can press X manually.
        if (self.has_cam_c
                and self.config.CALIBRATION_FILE_AB.exists()
                and self.config.CALIBRATION_FILE_BC.exists()):
            try:
                from chain_calibration import derive_ac
                print("  [CHAIN] Re-deriving AC to pick up the new floor...")
                derive_ac(self.config.CALIBRATION_FILE_AB,
                          self.config.CALIBRATION_FILE_BC,
                          self.config.CALIBRATION_FILE_AC)
            except Exception as e:
                print(f"  [CHAIN] Re-derivation failed ({e}); "
                      f"run X manually if AC is stale.")

    def _run_camera_diagnostics(self):
        """Run startup diagnostics on both cameras. Reports actual resolution and timing."""
        print("\n[CAMERA DIAGNOSTICS]")

        cam_list = [("A (DroidCam)", self.cap_a), ("B (Logitech)", self.cap_b)]
        if self.has_cam_c:
            cam_list.append(("C (Sony ZV-1F)", self.cap_c))
        for label, cap in cam_list:
            # Read actual resolution (what the camera is ACTUALLY delivering)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)

            expected_w = self.config.FRAME_WIDTH
            expected_h = self.config.FRAME_HEIGHT

            res_ok = (actual_w == expected_w and actual_h == expected_h)
            res_status = "OK" if res_ok else "MISMATCH"

            print(f"  Camera {label}:")
            print(f"    Resolution: {actual_w}x{actual_h} (expected {expected_w}x{expected_h}) [{res_status}]")
            print(f"    Reported FPS: {actual_fps:.1f}")

            if not res_ok:
                print(f"    !!! RESOLUTION MISMATCH — calibration will be INVALID")
                print(f"    !!! Fix DroidCam settings or change FRAME_WIDTH/HEIGHT in config")

            # Measure actual frame delivery timing (5 frames)
            times = []
            for _ in range(6):
                t0 = time.time()
                ret, _ = cap.read()
                if ret:
                    times.append(time.time() - t0)

            if len(times) >= 2:
                avg_ms = np.mean(times[1:]) * 1000  # Skip first (cold start)
                max_ms = np.max(times[1:]) * 1000
                print(f"    Frame read time: avg {avg_ms:.0f}ms, max {max_ms:.0f}ms")
                if max_ms > 200:
                    print(f"    !!! Slow frame delivery — may cause sync issues")

        # Quick sync test: read both cameras and compare timing
        t_a = time.time()
        self.cap_a.read()
        t_a = time.time() - t_a

        t_b = time.time()
        self.cap_b.read()
        t_b = time.time() - t_b

        sync_delta = abs(t_a - t_b) * 1000
        print(f"  Frame sync delta: {sync_delta:.0f}ms {'(OK)' if sync_delta < 50 else '(HIGH — may cause triangulation errors)'}")

    def load_existing_calibration(self):
        """Try to load existing calibration (all pairs)."""
        # Primary AB pair
        loaded_ab = False
        if self.config.CALIBRATION_FILE.exists():
            loaded_ab = self.calibration.load(self.config.CALIBRATION_FILE)
        elif self.config.CALIBRATION_FILE_AB.exists():
            loaded_ab = self.calibration.load(self.config.CALIBRATION_FILE_AB)
        else:
            # Also try MelodicCapFresh calibration as fallback
            fresh_cal = self.config.BASE_DIR.parent / "MelodicCapFresh" / "calibration" / "stereo_calibration.json"
            if fresh_cal.exists():
                print(f"[INFO] Found MelodicCapFresh calibration, loading...")
                loaded_ab = self.calibration.load(fresh_cal)

        # Load additional pairs if available (for offline multi-pair triangulation)
        if self.config.CALIBRATION_FILE_AC.exists():
            self.calibration_ac.load(self.config.CALIBRATION_FILE_AC)
        if self.config.CALIBRATION_FILE_BC.exists():
            self.calibration_bc.load(self.config.CALIBRATION_FILE_BC)

        return loaded_ab

    def run(self):
        """Main loop."""
        print("\n" + "=" * 60)
        print("MELODICCAP RTM STUDIO")
        print(f"  Detector: {self.config.POSE_MODE} ({self.config.POSE_QUALITY})")
        print(f"  Device: {self.config.POSE_DEVICE}")
        if self._cuda_was_requested and self.config.POSE_DEVICE == 'cpu':
            print(f"  !! WARNING: CUDA was requested but FAILED — running on CPU !!")
            print(f"  !! Expect 10-15 FPS instead of 30+ !!")
        skip_fh = getattr(self.config, 'SKIP_FACE_HANDS', False)
        print(f"  Skip face/hands: {skip_fh}")
        print("=" * 60)

        # Create directories
        self.config.BASE_DIR.mkdir(parents=True, exist_ok=True)
        (self.config.BASE_DIR / "calibration").mkdir(exist_ok=True)
        self.config.TAKES_DIR.mkdir(exist_ok=True)

        # Initialize cameras
        if not self.init_cameras():
            print("\n[FATAL] Could not initialize cameras. Exiting.")
            return

        # ── Camera Diagnostics ────────────────────────────────────
        self._run_camera_diagnostics()

        # Start non-blocking background reader for Camera C.
        # Must be AFTER diagnostics (which reads from cap_c directly).
        # The reader continuously grabs frames on a daemon thread so
        # Camera C never blocks the main loop, even if it's slow.
        if self.has_cam_c and self.cap_c is not None:
            self._cam_c_reader = _CamCReader(self.cap_c)
            print("[INFO] Camera C background reader started")

        # Try to load existing calibration
        self.load_existing_calibration()

        # Validate resolution matches calibration
        if self.calibration.is_calibrated:
            ret_a, test_a = self.cap_a.read()
            ret_b, test_b = self.cap_b.read()
            if ret_a and ret_b:
                ok, msg = self.calibration.validate_frame_resolution(test_a, test_b)
                if not ok:
                    print(f"\n  !!! {msg}")
                    print(f"  !!! Calibration will produce WRONG 3D results.")
                    print(f"  !!! Recalibrate at current resolution, or fix camera settings.")

        print("\n[CONTROLS]")
        print("  I = Per-camera intrinsics capture (cycles A/B" +
              ("/C" if self.has_cam_c else "") +
              "); grid coverage UI")
        print("  C = Collect paired calibration frames (hold board steady)")
        print("  S = In I mode: solve + save intrinsics for the active camera")
        print("      Otherwise: run stereo calibration on collected pair frames")
        print("  F = Calibrate floor with ChArUco board")
        print("  G = Auto floor calibration from ankles (stand still, feet flat)")
        if self.has_cam_c:
            print("  X = Derive AC from AB + BC (chain calibration, when A/C too far apart)")
        print("  R = Start/Stop recording")
        print("  Q = Quit")
        print()
        print("  Recommended flow (first run after any hardware change):")
        print("    1) I → fill Camera A's 3x3 grid → S → advance → same for B (and C)")
        print("    2) C → hold board in positions both cameras see → S → solve extrinsics")
        print("    3) F or G → calibrate floor")
        print("    4) R → record")
        print()

        frame_count = 0
        fps_timer = time.time()
        display_fps = 0
        timing_accum = {'cam': 0, 'infer': 0, 'tri': 0, 'total': 0, 'count': 0}

        while self.running:
            t_loop = time.time()

            # v4.9: Sequential grab/retrieve for frame synchronization.
            # A + B: sequential grab() on main thread (microseconds apart).
            # Camera C: non-blocking read from background thread.
            t0 = time.time()
            grab_a = self.cap_a.grab()
            grab_b = self.cap_b.grab()
            ret_a, frame_a = self.cap_a.retrieve() if grab_a else (False, None)
            ret_b, frame_b = self.cap_b.retrieve() if grab_b else (False, None)
            # Camera C: get latest frame from background reader (never blocks)
            if self._cam_c_reader:
                ret_c, frame_c = self._cam_c_reader.read()
            else:
                ret_c, frame_c = False, None
            t_cam = time.time() - t0

            if not ret_a or not ret_b:
                # Cameras may need flushing after idle periods (calibration, etc.)
                # Try a few more grabs before giving up on this frame
                for _ in range(5):
                    self.cap_a.grab()
                    self.cap_b.grab()
                grab_a = self.cap_a.grab()
                grab_b = self.cap_b.grab()
                ret_a, frame_a = self.cap_a.retrieve() if grab_a else (False, None)
                ret_b, frame_b = self.cap_b.retrieve() if grab_b else (False, None)
                if not ret_a or not ret_b:
                    print("[WARNING] Frame capture failed")
                    continue

            # Only run pose detection when NOT in calibration modes
            # (pose inference is expensive, skip it during calibration)
            det_a = None
            det_b = None
            det_c = None
            t_infer = 0
            in_intrinsics = self._intrinsics_mode is not None
            if (not self.collecting_cal_frames and not self.floor_cal_active
                    and not in_intrinsics):
                t0 = time.time()
                min_conf = self.config.MIN_KEYPOINT_CONFIDENCE
                # Run A+B in parallel (2 threads), then C sequentially.
                # Running all 3 in parallel overloads the GPU (10fps → ~15fps).
                future_a = self._infer_pool.submit(
                    self.detector.detect_single, frame_a, min_conf
                )
                future_b = self._infer_pool.submit(
                    self.detector.detect_single, frame_b, min_conf
                )
                det_a = future_a.result()
                det_b = future_b.result()
                # Camera C runs after A+B finish (sequential to avoid GPU contention)
                if self.has_cam_c and ret_c:
                    det_c = self.detector.detect_single(frame_c, min_conf)
                t_infer = time.time() - t0

            # Draw detections
            display_a = frame_a.copy()
            display_b = frame_b.copy()
            display_c = frame_c.copy() if (self.has_cam_c and ret_c) else None
            if det_a is not None or det_b is not None:
                draw_detections(display_a, det_a, self.config.MIN_KEYPOINT_CONFIDENCE)
                draw_detections(display_b, det_b, self.config.MIN_KEYPOINT_CONFIDENCE)
            if det_c is not None and display_c is not None:
                draw_detections(display_c, det_c, self.config.MIN_KEYPOINT_CONFIDENCE)

            # Intrinsics mode: overlay grid on the active camera's display.
            # The capture-grid overlay is drawn *on top* of the raw frame,
            # not on the already-decorated display, so pose detection icons
            # (there won't be any, because inference is gated off in this
            # mode) never interfere with corner detection.
            if in_intrinsics:
                active_frame = {'A': frame_a, 'B': frame_b,
                                'C': frame_c if (self.has_cam_c and ret_c) else None
                                }[self._intrinsics_mode]
                if active_frame is not None:
                    active_display = self._intrinsics_maybe_capture(active_frame)
                    if self._intrinsics_mode == 'A':
                        display_a = active_display
                    elif self._intrinsics_mode == 'B':
                        display_b = active_display
                    elif self._intrinsics_mode == 'C':
                        display_c = active_display

            # Handle countdown → start recording transition
            if self._countdown_active:
                remaining = self.config.RECORDING_COUNTDOWN - (time.time() - self._countdown_start)
                if remaining <= 0:
                    self._countdown_active = False
                    self.calibration.reset_filters()
                    self.recorder.start()

            # Handle auto floor calibration countdown → collect ankles
            if self._floor_auto_countdown:
                remaining = self.config.RECORDING_COUNTDOWN - (time.time() - self._floor_auto_start)
                if remaining <= 0:
                    self._floor_auto_countdown = False
                    print("[FLOOR-AUTO] Collecting ankle samples...")
                    ankle_samples = []
                    for _ in range(15):
                        grab_sa = self.cap_a.grab()
                        grab_sb = self.cap_b.grab()
                        # Camera C reader handles its own flushing in background
                        ret_sa, sample_a = self.cap_a.retrieve() if grab_sa else (False, None)
                        ret_sb, sample_b = self.cap_b.retrieve() if grab_sb else (False, None)
                        if ret_sa and ret_sb:
                            sa = self.detector.detect_single(sample_a, min_confidence=self.config.MIN_KEYPOINT_CONFIDENCE)
                            sb = self.detector.detect_single(sample_b, min_confidence=self.config.MIN_KEYPOINT_CONFIDENCE)
                            if sa and sb:
                                success, msg = self.calibration.calibrate_floor_from_ankles(sa, sb)
                                if success:
                                    ankle_samples.append(self.calibration.floor_z_offset)
                    if ankle_samples:
                        median_offset = float(np.median(ankle_samples))
                        self.calibration.floor_z_offset = median_offset
                        self.calibration.save(self.config.CALIBRATION_FILE)
                        print(f"  [OK] Floor set from ankles (offset: {median_offset:.3f}m, {len(ankle_samples)} samples)")
                        self._propagate_floor_offset(median_offset)
                    else:
                        print("  [FAILED] Ankles not detected. Make sure both cameras can see your feet.")

            # Floor calibration mode - show debug and auto-accept
            if self.floor_cal_active:
                debug = self.calibration.detect_floor_debug(frame_a, frame_b)
                self.calibration.draw_floor_debug(display_a, display_b, debug)

                # Auto-attempt calibration when we have enough corners
                has_a = debug['charuco_a_count'] >= 3
                has_b = debug['charuco_b_count'] >= 3
                if has_a or has_b:
                    success, msg = self.calibration.calibrate_floor(frame_a, frame_b)
                    if success:
                        self.calibration.save(self.config.CALIBRATION_FILE)
                        print(f"  [OK] {msg}")
                        self._propagate_floor_offset(self.calibration.floor_z_offset)
                        self.floor_cal_active = False
                    else:
                        # Show status but keep trying
                        draw_status(display_a, f"FLOOR: {msg}", (0, 165, 255))
                        draw_status(display_b, f"FLOOR: {msg}", (0, 165, 255))
                else:
                    draw_status(display_a, "FLOOR: Place board on ground", (0, 255, 255))
                    draw_status(display_b, "FLOOR: Place board on ground", (0, 255, 255))

            # Status text
            elif self._floor_auto_countdown:
                remaining = self.config.RECORDING_COUNTDOWN - (time.time() - self._floor_auto_start)
                secs = int(remaining) + 1
                draw_status(display_a, f"FLOOR CAL IN {secs}... stand still", (0, 255, 128))
                draw_status(display_b, f"FLOOR CAL IN {secs}... stand still", (0, 255, 128))
                for disp in [display_a, display_b]:
                    h, w = disp.shape[:2]
                    cv2.putText(disp, str(secs), (w // 2 - 40, h // 2 + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 128), 8)
            elif self.collecting_cal_frames:
                n_ab_s = len(self.cal_frames_ab[0])
                cal_parts = [f"AB:{n_ab_s}"]
                if self.has_cam_c:
                    cal_parts.append(f"AC:{len(self.cal_frames_ac[0])}")
                    cal_parts.append(f"BC:{len(self.cal_frames_bc[0])}")
                cal_text = f"CALIBRATING: {' | '.join(cal_parts)} frames"
                draw_status(display_a, cal_text, (0, 255, 255))
                draw_status(display_b, cal_text, (0, 255, 255))
            elif self._countdown_active:
                remaining = self.config.RECORDING_COUNTDOWN - (time.time() - self._countdown_start)
                secs = int(remaining) + 1
                draw_status(display_a, f"STARTING IN {secs}...", (0, 255, 255))
                draw_status(display_b, f"STARTING IN {secs}...", (0, 255, 255))
                # Big countdown number in center
                for disp in [display_a, display_b]:
                    h, w = disp.shape[:2]
                    cv2.putText(disp, str(secs), (w // 2 - 40, h // 2 + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 8)
            elif self.recorder.is_recording:
                rec_label = "REC-RAW" if self.config.OFFLINE_MODE else "REC"
                draw_status(display_a, f"{rec_label}: {len(self.recorder.frames)}f", (0, 0, 255))
                draw_status(display_b, f"{rec_label}: {len(self.recorder.frames)}f", (0, 0, 255))
            elif self.calibration.is_calibrated:
                draw_status(display_a, f"READY [{display_fps:.0f} fps]")
                draw_status(display_b, f"READY [{display_fps:.0f} fps]")
            else:
                draw_status(display_a, "NOT CALIBRATED (Press C)", (0, 165, 255))
                draw_status(display_b, "NOT CALIBRATED (Press C)", (0, 165, 255))

            # If calibrated and both poses detected, do triangulation
            t_tri = 0
            if (self.calibration.is_calibrated and
                    det_a is not None and det_b is not None):

                if self.config.OFFLINE_MODE:
                    # Offline: skip triangulation, save raw 2D only
                    if self.recorder.is_recording:
                        self.recorder.add_frame(None, det_a, det_b, det_c)
                else:
                    t0 = time.time()
                    points_3d = self.calibration.triangulate_pose(
                        det_a, det_b, smooth=True
                    )
                    t_tri = time.time() - t0

                    if points_3d and self.recorder.is_recording:
                        self.recorder.add_frame(points_3d, det_a, det_b, det_c)

            # Collecting calibration frames — multi-pair with stillness detection
            if self.collecting_cal_frames:
                ca, ida = self.calibration.detect_charuco(frame_a)
                cb, idb = self.calibration.detect_charuco(frame_b)
                cc, idc = (None, None)
                if self.has_cam_c and ret_c:
                    cc, idc = self.calibration.detect_charuco(frame_c)

                # Draw detected corners on all displays
                if ca is not None:
                    cv2.aruco.drawDetectedCornersCharuco(display_a, ca)
                if cb is not None:
                    cv2.aruco.drawDetectedCornersCharuco(display_b, cb)
                if cc is not None and display_c is not None:
                    cv2.aruco.drawDetectedCornersCharuco(display_c, cc)

                MIN_COMMON_FOR_CAPTURE = 8
                STILL_THRESH = 2.0
                now = time.time()

                # Check each camera pair for common corners + stillness
                # AB pair (primary)
                captured_any = False
                if ca is not None and cb is not None:
                    common_ab = set(ida.flatten()) & set(idb.flatten())
                    n_ab = len(common_ab)
                    if n_ab >= MIN_COMMON_FOR_CAPTURE:
                        still_ab = self._check_pair_stillness(
                            ca, ida, cb, idb, '_prev_corners_ab', STILL_THRESH)
                        if still_ab and now - self._last_cal_time > 0.5:
                            self.cal_frames_ab[0].append(frame_a.copy())
                            self.cal_frames_ab[1].append(frame_b.copy())
                            self._last_cal_time = now
                            captured_any = True
                            print(f"  [AB] Captured frame {len(self.cal_frames_ab[0])} ({n_ab} common)")

                # AC pair
                if self.has_cam_c and ca is not None and cc is not None:
                    common_ac = set(ida.flatten()) & set(idc.flatten())
                    n_ac = len(common_ac)
                    if n_ac >= MIN_COMMON_FOR_CAPTURE:
                        still_ac = self._check_pair_stillness(
                            ca, ida, cc, idc, '_prev_corners_ac', STILL_THRESH)
                        if still_ac and now - self._last_cal_time_ac > 0.5:
                            self.cal_frames_ac[0].append(frame_a.copy())
                            self.cal_frames_ac[1].append(frame_c.copy())
                            self._last_cal_time_ac = now
                            captured_any = True
                            print(f"  [AC] Captured frame {len(self.cal_frames_ac[0])} ({n_ac} common)")

                # BC pair
                if self.has_cam_c and cb is not None and cc is not None:
                    common_bc = set(idb.flatten()) & set(idc.flatten())
                    n_bc = len(common_bc)
                    if n_bc >= MIN_COMMON_FOR_CAPTURE:
                        still_bc = self._check_pair_stillness(
                            cb, idb, cc, idc, '_prev_corners_bc', STILL_THRESH)
                        if still_bc and now - self._last_cal_time_bc > 0.5:
                            self.cal_frames_bc[0].append(frame_b.copy())
                            self.cal_frames_bc[1].append(frame_c.copy())
                            self._last_cal_time_bc = now
                            captured_any = True
                            print(f"  [BC] Captured frame {len(self.cal_frames_bc[0])} ({n_bc} common)")

                # Status display
                n_ab_frames = len(self.cal_frames_ab[0])
                status_parts = [f"AB:{n_ab_frames}"]
                if self.has_cam_c:
                    status_parts.append(f"AC:{len(self.cal_frames_ac[0])}")
                    status_parts.append(f"BC:{len(self.cal_frames_bc[0])}")
                cal_status = f"CALIBRATING: {' | '.join(status_parts)} frames"
                draw_status(display_a, cal_status, (0, 255, 255))
                draw_status(display_b, cal_status, (0, 255, 255))

                if not captured_any and now - self._last_cal_time > 2.0:
                    if ca is None or cb is None:
                        hint = "Board not seen by cameras"
                    else:
                        hint = "HOLD STILL to capture"
                    draw_status(display_a, hint, (0, 165, 255))
                    draw_status(display_b, hint, (0, 165, 255))

            # FPS counter + timing breakdown
            t_total = time.time() - t_loop
            timing_accum['cam'] += t_cam
            timing_accum['infer'] += t_infer
            timing_accum['tri'] += t_tri
            timing_accum['total'] += t_total
            timing_accum['count'] += 1
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                display_fps = frame_count / elapsed
                n = timing_accum['count'] or 1
                avg_cam = timing_accum['cam'] / n * 1000
                avg_infer = timing_accum['infer'] / n * 1000
                avg_tri = timing_accum['tri'] / n * 1000
                avg_total = timing_accum['total'] / n * 1000
                reproj = self.calibration.last_reproj_error
                reproj_str = f" reproj:{reproj[0]:.1f}/{reproj[1]:.1f}px" if reproj[2] > 0 else ""
                print(f"[TIMING] {display_fps:.1f} fps | cam:{avg_cam:.0f}ms infer:{avg_infer:.0f}ms tri:{avg_tri:.0f}ms total:{avg_total:.0f}ms{reproj_str}")
                frame_count = 0
                fps_timer = time.time()
                timing_accum = {'cam': 0, 'infer': 0, 'tri': 0, 'total': 0, 'count': 0}

            # Display — 2 or 3 camera layout
            if self.has_cam_c and display_c is not None:
                # 3-camera: top row = A + B (640x360 each), bottom row = C centered
                row_top = np.hstack([
                    cv2.resize(display_a, (640, 360)),
                    cv2.resize(display_b, (640, 360))
                ])
                cam_c_resized = cv2.resize(display_c, (640, 360))
                # Pad C to same width as top row (center it)
                pad_w = (row_top.shape[1] - 640) // 2
                pad_left = np.zeros((360, pad_w, 3), dtype=np.uint8)
                pad_right = np.zeros((360, row_top.shape[1] - 640 - pad_w, 3), dtype=np.uint8)
                row_bottom = np.hstack([pad_left, cam_c_resized, pad_right])
                combined = np.vstack([row_top, row_bottom])
            else:
                combined = np.hstack([
                    cv2.resize(display_a, (640, 360)),
                    cv2.resize(display_b, (640, 360))
                ])

            # FPS bar at bottom — shows model, device, FPS, and warnings
            bar = np.zeros((30, combined.shape[1], 3), dtype=np.uint8)
            mode_str = f"RTMW {self.config.POSE_QUALITY}" if self.config.POSE_MODE == 'wholebody' else f"RTMPose {self.config.POSE_QUALITY}"
            device_str = self.config.POSE_DEVICE.upper()
            skip_str = " | body-only" if getattr(self.config, 'SKIP_FACE_HANDS', False) else ""

            # Warn if CUDA was requested but we fell back to CPU
            if self._cuda_was_requested and self.config.POSE_DEVICE == 'cpu':
                bar[:] = (0, 0, 80)  # dark red background
                cv2.putText(bar, f"{mode_str} | !! CPU FALLBACK (CUDA FAILED) !! | {display_fps:.0f} fps{skip_str}",
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            elif display_fps < 15 and not self.collecting_cal_frames:
                bar[:] = (0, 40, 80)  # dark orange background
                cv2.putText(bar, f"{mode_str} | {device_str} | {display_fps:.0f} fps LOW{skip_str}",
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            else:
                cv2.putText(bar, f"{mode_str} | {device_str} | {display_fps:.0f} fps{skip_str}",
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            combined = np.vstack([combined, bar])

            cv2.imshow("MelodicCap RTM", combined)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n[QUIT]")
                self.running = False

            elif key == ord('i'):
                # Intrinsics capture: cycle through Camera A, B, (C).
                # Entering intrinsics mode stops paired-capture mode so the
                # two states don't compete for the same key.
                if self.collecting_cal_frames:
                    self.collecting_cal_frames = False
                    print("\n[CALIBRATION] Pair-capture stopped (switching to I mode).")
                self._intrinsics_cycle()

            elif key == ord('c'):
                # Entering pair-capture mode stops intrinsics mode for the
                # same reason.
                if self._intrinsics_mode is not None:
                    self._intrinsics_mode = None
                    print("\n[INTRINSICS] Capture OFF (switching to C mode).")
                if not self.collecting_cal_frames:
                    print("\n[CALIBRATION] Starting frame collection...")
                    print("  Hold the board STILL at each position — captures when stationary.")
                    print("  Move to a new position, hold still, repeat. Cover the full frame.")
                    if self.has_cam_c:
                        print("  3-CAMERA MODE: collecting frames for all pairs (AB, AC, BC)")
                        print("  Try to show the board where 2+ cameras can see it at once.")
                    print("  Press 'S' when you have 10+ frames per pair to run calibration.")
                    self.cal_frames_ab = ([], [])
                    self.cal_frames_ac = ([], [])
                    self.cal_frames_bc = ([], [])
                    self._prev_corners_ab = None
                    self._prev_corners_ac = None
                    self._prev_corners_bc = None
                    self.collecting_cal_frames = True
                else:
                    print("\n[CALIBRATION] Stopped collecting.")
                    self.collecting_cal_frames = False

            elif key == ord('s'):
                # Polymorphic S: in intrinsics mode it saves the current
                # camera's K/D; otherwise it runs stereo calibration on
                # whatever pair frames were collected via C.
                if self._intrinsics_mode is not None:
                    self._intrinsics_solve_and_save()
                    continue

                n_ab = len(self.cal_frames_ab[0])
                n_ac = len(self.cal_frames_ac[0])
                n_bc = len(self.cal_frames_bc[0])

                if n_ab < 8:
                    print(f"\n[ERROR] Need at least 8 AB frames (have {n_ab})")
                else:
                    self.collecting_cal_frames = False

                    # Try to load saved per-camera intrinsics. When both
                    # cameras of a pair have saved intrinsics, stereo
                    # calibration skips the monocular pass and only solves
                    # R/T — much faster and far more accurate because each
                    # camera's K was measured against a full 3x3 grid
                    # coverage instead of whatever positions happened to be
                    # shared during pair capture.
                    intr_a = load_intrinsics(self.config.INTRINSICS_FILE_A)
                    intr_b = load_intrinsics(self.config.INTRINSICS_FILE_B)
                    intr_c = (load_intrinsics(self.config.INTRINSICS_FILE_C)
                              if self.has_cam_c else None)
                    if intr_a is not None:
                        print(f"[S] Found saved intrinsics for Camera A "
                              f"(rms {intr_a['rms']:.3f}, "
                              f"{intr_a['n_frames']} frames)")
                    else:
                        print(f"[WARN] No saved intrinsics for Camera A; "
                              f"using legacy joint estimation")
                    if intr_b is not None:
                        print(f"[S] Found saved intrinsics for Camera B "
                              f"(rms {intr_b['rms']:.3f}, "
                              f"{intr_b['n_frames']} frames)")
                    else:
                        print(f"[WARN] No saved intrinsics for Camera B; "
                              f"using legacy joint estimation")
                    if self.has_cam_c:
                        if intr_c is not None:
                            print(f"[S] Found saved intrinsics for Camera C "
                                  f"(rms {intr_c['rms']:.3f}, "
                                  f"{intr_c['n_frames']} frames)")
                        else:
                            print(f"[WARN] No saved intrinsics for Camera C; "
                                  f"using legacy joint estimation")

                    def _kd(intr):
                        if intr is None:
                            return (None, None)
                        return (intr['K'], intr['D'])

                    K_a, D_a = _kd(intr_a)
                    K_b, D_b = _kd(intr_b)
                    K_c, D_c = _kd(intr_c)

                    ab_ok = False
                    ac_ok = False
                    bc_ok = False

                    # Calibrate AB pair (primary — required)
                    print(f"\n[CALIBRATION] AB pair: {n_ab} frames...")
                    if self.calibration.calibrate_stereo(
                            self.cal_frames_ab[0], self.cal_frames_ab[1],
                            K1_fixed=K_a, D1_fixed=D_a,
                            K2_fixed=K_b, D2_fixed=D_b):
                        self.calibration.save(self.config.CALIBRATION_FILE)
                        self.calibration.save(self.config.CALIBRATION_FILE_AB)
                        print(f"  [OK] AB calibration saved")
                        ab_ok = True

                    # Calibrate AC pair (optional)
                    if n_ac >= 8:
                        print(f"\n[CALIBRATION] AC pair: {n_ac} frames...")
                        if self.calibration_ac.calibrate_stereo(
                                self.cal_frames_ac[0], self.cal_frames_ac[1],
                                K1_fixed=K_a, D1_fixed=D_a,
                                K2_fixed=K_c, D2_fixed=D_c):
                            self.calibration_ac.save(self.config.CALIBRATION_FILE_AC)
                            print(f"  [OK] AC calibration saved")
                            ac_ok = True
                    elif self.has_cam_c:
                        print(f"  [SKIP] AC pair: only {n_ac} frames (need 8)")

                    # Calibrate BC pair (optional)
                    if n_bc >= 8:
                        print(f"\n[CALIBRATION] BC pair: {n_bc} frames...")
                        if self.calibration_bc.calibrate_stereo(
                                self.cal_frames_bc[0], self.cal_frames_bc[1],
                                K1_fixed=K_b, D1_fixed=D_b,
                                K2_fixed=K_c, D2_fixed=D_c):
                            self.calibration_bc.save(self.config.CALIBRATION_FILE_BC)
                            print(f"  [OK] BC calibration saved")
                            bc_ok = True
                    elif self.has_cam_c:
                        print(f"  [SKIP] BC pair: only {n_bc} frames (need 8)")

                    # Auto-chain: if AB + BC both calibrated but AC didn't,
                    # derive AC from the chain. Saves the user a round-trip to
                    # a separate terminal and the chain_calibration.py CLI.
                    if self.has_cam_c and ab_ok and bc_ok and not ac_ok:
                        print("\n[CHAIN] AC not directly calibrated — "
                              "deriving from AB + BC...")
                        try:
                            from chain_calibration import derive_ac
                            if derive_ac(self.config.CALIBRATION_FILE_AB,
                                         self.config.CALIBRATION_FILE_BC,
                                         self.config.CALIBRATION_FILE_AC):
                                print("[CHAIN] Auto-derived AC saved. "
                                      "(Press 'X' to re-derive after floor cal.)")
                            else:
                                print("[CHAIN] Auto-derivation failed — see "
                                      "error above. Press 'X' to retry after "
                                      "floor calibration.")
                        except Exception as e:
                            print(f"[CHAIN] Auto-derivation crashed: {e}")

                    # Flush camera buffers after long calibration processing
                    # Camera C reader handles its own flushing in background
                    for _ in range(10):
                        self.cap_a.grab()
                        self.cap_b.grab()

                    self.cal_frames_ab = ([], [])
                    self.cal_frames_ac = ([], [])
                    self.cal_frames_bc = ([], [])

            elif key == ord('f'):
                if not self.calibration.is_calibrated:
                    print("\n[ERROR] Calibrate cameras first (press C)")
                elif self.floor_cal_active:
                    self.floor_cal_active = False
                    print("\n[FLOOR] Cancelled.")
                else:
                    self.floor_cal_active = True
                    print("\n[FLOOR] Floor calibration mode ON")
                    print("  Place the ChArUco board flat on the ground.")
                    print("  Debug markers will show on screen. Auto-accepts when detected.")
                    print("  Press 'F' again to cancel.")

            elif key == ord('g'):
                # Auto floor calibration from ankles — with countdown
                if not self.calibration.is_calibrated:
                    print("\n[ERROR] Calibrate cameras first (press C)")
                elif self._floor_auto_countdown:
                    # Cancel
                    self._floor_auto_countdown = False
                    print("\n[FLOOR-AUTO] Cancelled.")
                else:
                    self._floor_auto_countdown = True
                    self._floor_auto_start = time.time()
                    print(f"\n[FLOOR-AUTO] Stand still with feet flat in {self.config.RECORDING_COUNTDOWN}s... (G to cancel)")

            elif key == ord('x'):
                # Chain calibration: derive AC from the AB and BC files on
                # disk. Useful when Camera A and Camera C are too far apart
                # for the ChArUco board to be detectable in both at once,
                # or after re-running floor calibration (so the derived AC
                # picks up the new floor offset).
                if not self.has_cam_c:
                    print("\n[CHAIN] Chain calibration requires Camera C. "
                          "Disabled in 2-camera mode.")
                else:
                    print("\n[CHAIN] Deriving AC from AB + BC on disk...")
                    try:
                        from chain_calibration import derive_ac
                        ok = derive_ac(self.config.CALIBRATION_FILE_AB,
                                       self.config.CALIBRATION_FILE_BC,
                                       self.config.CALIBRATION_FILE_AC)
                        if not ok:
                            print("[CHAIN] Derivation failed — see error above.")
                    except Exception as e:
                        print(f"[CHAIN] Derivation crashed: {e}")

            elif key == ord('r'):
                if not self.calibration.is_calibrated:
                    print("\n[ERROR] Calibrate cameras first (press C)")
                elif self._countdown_active:
                    # Cancel countdown
                    self._countdown_active = False
                    print("\n[RECORDING] Countdown cancelled.")
                elif self.recorder.is_recording:
                    detector_info = f"{self.config.POSE_MODE}_{self.config.POSE_QUALITY}"
                    trim_secs = self.config.RECORDING_TRIM_END
                    kp_count = 133 if self.config.POSE_MODE == 'wholebody' else 17
                    self.recorder.stop(detector_info=detector_info, trim_end=trim_secs,
                                       offline_mode=self.config.OFFLINE_MODE,
                                       keypoint_count=kp_count)
                elif (abs(self.calibration.floor_z_offset) < 0.01
                      and not getattr(self, '_floor_missing_ack', False)):
                    # Pre-flight: floor not calibrated. The offline processor
                    # will reject every pair without a floor offset, so the
                    # take would be useless. Warn and require a confirm press.
                    print("\n[WARNING] No floor offset calibrated — the "
                          "offline processor will reject every pair and the "
                          "take will be unusable.")
                    print("          Press 'F' (or 'G') to calibrate the floor "
                          "first, or press 'R' again to record anyway.")
                    self._floor_missing_ack = True
                else:
                    # Start countdown
                    self._floor_missing_ack = False
                    self._countdown_active = True
                    self._countdown_start = time.time()
                    print(f"\n[RECORDING] Starting in {self.config.RECORDING_COUNTDOWN} seconds... (R to cancel)")

        # Cleanup
        if self.recorder.is_recording:
            kp_count = 133 if self.config.POSE_MODE == 'wholebody' else 17
            self.recorder.stop(offline_mode=self.config.OFFLINE_MODE,
                               keypoint_count=kp_count)

        self._infer_pool.shutdown(wait=False)
        self.cap_a.release()
        self.cap_b.release()
        if self._cam_c_reader is not None:
            self._cam_c_reader.stop()
        cv2.destroyAllWindows()

        print("\n[DONE]")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    app = MelodicCapApp()
    app.run()
