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

Usage:
1. Run this script
2. Press 'C' to start collecting calibration frames (show ChArUco board to both)
3. Press 'S' to run stereo calibration (after collecting 20+ frames)
4. Press 'F' to calibrate floor (place board on ground)
5. Press 'R' to start/stop recording
6. Press 'Q' to quit

Output: JSON files in takes/ directory, ready for Blender import.
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from keypoint_map import LM, BODY_SKELETON
from pose_detector import PoseDetector
from stereo_calibration import StereoCalibration
from recorder import MocapRecorder


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """All settings in one place."""

    # Camera indices (change these to match your system)
    CAM_A_INDEX = 2  # Sony ZV-1F
    CAM_B_INDEX = 0  # Samsung S25 DroidCam

    # Resolution
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FPS = 30

    # Paths — auto-detects based on where this script lives
    # On your PC this resolves to C:\Users\ninja\Documents\MelodicCapStudio\MelodicCapRTM
    # (assuming you cloned/copied this folder there)
    BASE_DIR = Path(__file__).resolve().parent
    CALIBRATION_FILE = BASE_DIR / "calibration" / "stereo_calibration.json"
    TAKES_DIR = BASE_DIR / "takes"

    # ChArUco board (must match your printed board)
    # 5x7 board, single letter page — no edge truncation
    # IMPORTANT: measure actual printed square with ruler, update if different
    CHARUCO_SQUARES_X = 5
    CHARUCO_SQUARES_Y = 7
    CHARUCO_SQUARE_SIZE = 0.035    # 35mm target — MEASURE YOUR PRINT
    CHARUCO_MARKER_SIZE = 0.025    # 25mm target — MEASURE YOUR PRINT
    ARUCO_DICT = cv2.aruco.DICT_4X4_50

    # Recording
    RECORDING_COUNTDOWN = 5  # seconds before recording starts
    RECORDING_TRIM_END = 2.0  # seconds to trim from end of recording
    OFFLINE_MODE = False  # True = save raw 2D only, triangulate later with offline_processor.py

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
# MAIN APPLICATION
# =============================================================================

class MelodicCapApp:
    """Main motion capture application."""

    def __init__(self):
        self.config = Config()

        # Track if user wanted CUDA (to warn if we fell back)
        self._cuda_was_requested = (self.config.POSE_DEVICE == 'cuda')

        # Thread pool for parallel inference on both cameras
        self._infer_pool = ThreadPoolExecutor(max_workers=2)

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

        # Systems
        self.calibration = StereoCalibration(self.config)
        self.recorder = MocapRecorder(self.config.TAKES_DIR)

        # Calibration frame collection
        self.cal_frames_a = []
        self.cal_frames_b = []
        self.collecting_cal_frames = False
        self._last_cal_time = 0
        self._prev_cal_corners_a = None  # For stillness detection
        self._prev_cal_corners_b = None

        # Floor calibration mode
        self.floor_cal_active = False

        # Auto floor calibration countdown
        self._floor_auto_countdown = False
        self._floor_auto_start = 0

        # State
        self.running = True
        self._countdown_start = 0
        self._countdown_active = False

    def init_cameras(self):
        """Initialize both cameras."""
        print("\n[INITIALIZING CAMERAS]")

        # Camera A
        print(f"  Opening Camera A (index {self.config.CAM_A_INDEX})...")
        self.cap_a = cv2.VideoCapture(self.config.CAM_A_INDEX, cv2.CAP_DSHOW)
        if not self.cap_a.isOpened():
            self.cap_a = cv2.VideoCapture(self.config.CAM_A_INDEX)

        if self.cap_a.isOpened():
            self.cap_a.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.FRAME_WIDTH)
            self.cap_a.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.FRAME_HEIGHT)
            self.cap_a.set(cv2.CAP_PROP_FPS, self.config.FPS)
            self.cap_a.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # v4.9: minimize frame latency
            print(f"    [OK] Camera A opened")
        else:
            print(f"    [ERROR] Failed to open Camera A")
            return False

        # Camera B
        print(f"  Opening Camera B (index {self.config.CAM_B_INDEX})...")
        self.cap_b = cv2.VideoCapture(self.config.CAM_B_INDEX, cv2.CAP_DSHOW)
        if not self.cap_b.isOpened():
            self.cap_b = cv2.VideoCapture(self.config.CAM_B_INDEX)

        if self.cap_b.isOpened():
            self.cap_b.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.FRAME_WIDTH)
            self.cap_b.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.FRAME_HEIGHT)
            self.cap_b.set(cv2.CAP_PROP_FPS, self.config.FPS)
            self.cap_b.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # v4.9: minimize frame latency
            print(f"    [OK] Camera B opened")
        else:
            print(f"    [ERROR] Failed to open Camera B")
            return False

        return True

    def _run_camera_diagnostics(self):
        """Run startup diagnostics on both cameras. Reports actual resolution and timing."""
        print("\n[CAMERA DIAGNOSTICS]")

        for label, cap in [("A (Sony ZV-1F)", self.cap_a), ("B (DroidCam)", self.cap_b)]:
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
        """Try to load existing calibration."""
        if self.config.CALIBRATION_FILE.exists():
            return self.calibration.load(self.config.CALIBRATION_FILE)
        # Also try MelodicCapFresh calibration as fallback
        fresh_cal = self.config.BASE_DIR.parent / "MelodicCapFresh" / "calibration" / "stereo_calibration.json"
        if fresh_cal.exists():
            print(f"[INFO] Found MelodicCapFresh calibration, loading...")
            return self.calibration.load(fresh_cal)
        return False

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
        print("  C = Collect calibration frames (hold board steady)")
        print("  S = Run stereo calibration (after collecting frames)")
        print("  F = Calibrate floor with ChArUco board")
        print("  G = Auto floor calibration from ankles (stand still, feet flat)")
        print("  R = Start/Stop recording")
        print("  Q = Quit")
        print()

        frame_count = 0
        fps_timer = time.time()
        display_fps = 0
        timing_accum = {'cam': 0, 'infer': 0, 'tri': 0, 'total': 0, 'count': 0}

        while self.running:
            t_loop = time.time()

            # v4.9: Sequential grab/retrieve for frame synchronization.
            # .read() = .grab() + .retrieve() combined. Using threaded .read()
            # means each camera grabs its frame at different times (50-100ms apart
            # with USB cameras). Sequential grab() calls happen microseconds apart
            # on the main thread, so both cameras capture the same moment.
            t0 = time.time()
            grab_a = self.cap_a.grab()
            grab_b = self.cap_b.grab()
            ret_a, frame_a = self.cap_a.retrieve() if grab_a else (False, None)
            ret_b, frame_b = self.cap_b.retrieve() if grab_b else (False, None)
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
            t_infer = 0
            if not self.collecting_cal_frames and not self.floor_cal_active:
                t0 = time.time()
                # Run both cameras' inference in parallel — overlaps CPU
                # preprocessing of camera B with GPU inference of camera A
                min_conf = self.config.MIN_KEYPOINT_CONFIDENCE
                future_a = self._infer_pool.submit(
                    self.detector.detect_single, frame_a, min_conf
                )
                future_b = self._infer_pool.submit(
                    self.detector.detect_single, frame_b, min_conf
                )
                det_a = future_a.result()
                det_b = future_b.result()
                t_infer = time.time() - t0

            # Draw detections
            display_a = frame_a.copy()
            display_b = frame_b.copy()
            if det_a is not None or det_b is not None:
                draw_detections(display_a, det_a, self.config.MIN_KEYPOINT_CONFIDENCE)
                draw_detections(display_b, det_b, self.config.MIN_KEYPOINT_CONFIDENCE)

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
                draw_status(display_a, f"CALIBRATING: {len(self.cal_frames_a)} frames", (0, 255, 255))
                draw_status(display_b, f"CALIBRATING: {len(self.cal_frames_b)} frames", (0, 255, 255))
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
                        self.recorder.add_frame(None, det_a, det_b)
                else:
                    t0 = time.time()
                    points_3d = self.calibration.triangulate_pose(
                        det_a, det_b, smooth=True
                    )
                    t_tri = time.time() - t0

                    if points_3d and self.recorder.is_recording:
                        self.recorder.add_frame(points_3d, det_a, det_b)

            # Collecting calibration frames — with stillness detection
            if self.collecting_cal_frames:
                ca, ida = self.calibration.detect_charuco(frame_a)
                cb, idb = self.calibration.detect_charuco(frame_b)

                if ca is not None and cb is not None:
                    cv2.aruco.drawDetectedCornersCharuco(display_a, ca)
                    cv2.aruco.drawDetectedCornersCharuco(display_b, cb)

                    # v4.9: Check common corners between cameras BEFORE capture.
                    # Previously only checked per-camera count (len >= 6), but
                    # cameras can each see 6+ different corner IDs with no overlap.
                    # This wastes the user's time capturing useless frames.
                    MIN_COMMON_FOR_CAPTURE = 8
                    common_ab = set(ida.flatten()) & set(idb.flatten())
                    n_common = len(common_ab)

                    # Show live corner count on both displays
                    corner_color = (0, 255, 0) if n_common >= MIN_COMMON_FOR_CAPTURE else (0, 0, 255)
                    corner_text = f"Common: {n_common} (A:{len(ca)} B:{len(cb)})"
                    cv2.putText(display_a, corner_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, corner_color, 2)
                    cv2.putText(display_b, corner_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, corner_color, 2)

                    if n_common < MIN_COMMON_FOR_CAPTURE:
                        # Not enough common corners — show guidance
                        if n_common == 0:
                            hint = "Board not seen by both cameras"
                        else:
                            hint = f"Angle board so BOTH cameras see it ({n_common}/8 common)"
                        draw_status(display_a, hint, (0, 0, 255))
                        draw_status(display_b, hint, (0, 0, 255))
                        self._prev_cal_corners_a = None
                        self._prev_cal_corners_b = None
                    else:
                        # Enough common corners — check stillness
                        board_still = False
                        STILL_THRESH = 2.0  # pixels — must move less than this
                        if (self._prev_cal_corners_a is not None and
                                self._prev_cal_corners_b is not None and
                                len(ca) >= 6 and len(cb) >= 6):
                            prev_a, prev_ida = self._prev_cal_corners_a
                            prev_b, prev_idb = self._prev_cal_corners_b
                            if prev_ida is not None and prev_idb is not None:
                                common_a = set(ida.flatten()) & set(prev_ida.flatten())
                                common_b = set(idb.flatten()) & set(prev_idb.flatten())
                                if len(common_a) >= 4 and len(common_b) >= 4:
                                    move_a = 0
                                    for cid in common_a:
                                        idx_cur = np.where(ida.flatten() == cid)[0][0]
                                        idx_prev = np.where(prev_ida.flatten() == cid)[0][0]
                                        move_a = max(move_a, np.linalg.norm(ca[idx_cur].flatten() - prev_a[idx_prev].flatten()))
                                    move_b = 0
                                    for cid in common_b:
                                        idx_cur = np.where(idb.flatten() == cid)[0][0]
                                        idx_prev = np.where(prev_idb.flatten() == cid)[0][0]
                                        move_b = max(move_b, np.linalg.norm(cb[idx_cur].flatten() - prev_b[idx_prev].flatten()))
                                    board_still = (move_a < STILL_THRESH and move_b < STILL_THRESH)

                        self._prev_cal_corners_a = (ca, ida)
                        self._prev_cal_corners_b = (cb, idb)

                        if board_still and time.time() - self._last_cal_time > 0.5:
                            self.cal_frames_a.append(frame_a.copy())
                            self.cal_frames_b.append(frame_b.copy())
                            self._last_cal_time = time.time()
                            print(f"  Captured frame {len(self.cal_frames_a)} ({n_common} common corners)")
                        elif not board_still and time.time() - self._last_cal_time > 2.0:
                            draw_status(display_a, "HOLD STILL to capture", (0, 165, 255))
                            draw_status(display_b, "HOLD STILL to capture", (0, 165, 255))
                else:
                    self._prev_cal_corners_a = None
                    self._prev_cal_corners_b = None

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

            # Display
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

            elif key == ord('c'):
                if not self.collecting_cal_frames:
                    print("\n[CALIBRATION] Starting frame collection...")
                    print("  Hold the board STILL at each position — captures when stationary.")
                    print("  Move to a new position, hold still, repeat. Cover the full frame.")
                    print("  Press 'S' when you have 20+ frames to run calibration.")
                    self.cal_frames_a = []
                    self.cal_frames_b = []
                    self._prev_cal_corners_a = None
                    self._prev_cal_corners_b = None
                    self.collecting_cal_frames = True
                else:
                    print("\n[CALIBRATION] Stopped collecting.")
                    self.collecting_cal_frames = False

            elif key == ord('s'):
                if len(self.cal_frames_a) >= 10:
                    print(f"\n[CALIBRATION] Running with {len(self.cal_frames_a)} frames...")
                    self.collecting_cal_frames = False

                    if self.calibration.calibrate_stereo(self.cal_frames_a, self.cal_frames_b):
                        self.calibration.save(self.config.CALIBRATION_FILE)

                    # v4.9: Flush camera buffers after long calibration processing.
                    # grab()/retrieve() fails if cameras sit idle for seconds —
                    # the USB buffer fills with stale frames that can't be grabbed.
                    for _ in range(10):
                        self.cap_a.grab()
                        self.cap_b.grab()

                    self.cal_frames_a = []
                    self.cal_frames_b = []
                else:
                    print(f"\n[ERROR] Need at least 10 frames (have {len(self.cal_frames_a)})")

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
                else:
                    # Start countdown
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
        cv2.destroyAllWindows()

        print("\n[DONE]")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    app = MelodicCapApp()
    app.run()
