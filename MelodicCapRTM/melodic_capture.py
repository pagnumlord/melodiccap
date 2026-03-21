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
    CHARUCO_SQUARES_X = 4
    CHARUCO_SQUARES_Y = 3
    CHARUCO_SQUARE_SIZE = 0.0635  # 63.5mm in meters
    CHARUCO_MARKER_SIZE = 0.0476  # 47.6mm in meters
    ARUCO_DICT = cv2.aruco.DICT_4X4_50

    # Pose detection
    POSE_MODE = 'wholebody'     # 'body' (17 kps, fast) or 'wholebody' (133 kps)
    POSE_QUALITY = 'balanced'   # 'fast', 'balanced', or 'accurate'
    POSE_DEVICE = 'cuda'        # 'cuda' or 'cpu'
    MIN_KEYPOINT_CONFIDENCE = 0.3

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

        # State
        self.running = True

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
            print(f"    [OK] Camera B opened")
        else:
            print(f"    [ERROR] Failed to open Camera B")
            return False

        return True

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
        print("=" * 60)

        # Create directories
        self.config.BASE_DIR.mkdir(parents=True, exist_ok=True)
        (self.config.BASE_DIR / "calibration").mkdir(exist_ok=True)
        self.config.TAKES_DIR.mkdir(exist_ok=True)

        # Initialize cameras
        if not self.init_cameras():
            print("\n[FATAL] Could not initialize cameras. Exiting.")
            return

        # Try to load existing calibration
        self.load_existing_calibration()

        print("\n[CONTROLS]")
        print("  C = Collect calibration frames (hold board steady)")
        print("  S = Run stereo calibration (after collecting frames)")
        print("  F = Calibrate floor (place board on ground)")
        print("  R = Start/Stop recording")
        print("  Q = Quit")
        print()

        frame_count = 0
        fps_timer = time.time()
        display_fps = 0

        while self.running:
            # Capture frames
            ret_a, frame_a = self.cap_a.read()
            ret_b, frame_b = self.cap_b.read()

            if not ret_a or not ret_b:
                print("[WARNING] Frame capture failed")
                continue

            # Only run pose detection when NOT collecting calibration frames
            # (pose inference is expensive, skip it during calibration)
            det_a = None
            det_b = None
            if not self.collecting_cal_frames:
                det_a = self.detector.detect_single(
                    frame_a, min_confidence=self.config.MIN_KEYPOINT_CONFIDENCE
                )
                det_b = self.detector.detect_single(
                    frame_b, min_confidence=self.config.MIN_KEYPOINT_CONFIDENCE
                )

            # Draw detections
            display_a = frame_a.copy()
            display_b = frame_b.copy()
            if det_a is not None or det_b is not None:
                draw_detections(display_a, det_a, self.config.MIN_KEYPOINT_CONFIDENCE)
                draw_detections(display_b, det_b, self.config.MIN_KEYPOINT_CONFIDENCE)

            # Status text
            if self.collecting_cal_frames:
                draw_status(display_a, f"CALIBRATING: {len(self.cal_frames_a)} frames", (0, 255, 255))
                draw_status(display_b, f"CALIBRATING: {len(self.cal_frames_b)} frames", (0, 255, 255))
            elif self.recorder.is_recording:
                draw_status(display_a, f"REC: {len(self.recorder.frames)}f", (0, 0, 255))
                draw_status(display_b, f"REC: {len(self.recorder.frames)}f", (0, 0, 255))
            elif self.calibration.is_calibrated:
                draw_status(display_a, f"READY [{display_fps:.0f} fps]")
                draw_status(display_b, f"READY [{display_fps:.0f} fps]")
            else:
                draw_status(display_a, "NOT CALIBRATED (Press C)", (0, 165, 255))
                draw_status(display_b, "NOT CALIBRATED (Press C)", (0, 165, 255))

            # If calibrated and both poses detected, do triangulation
            if (self.calibration.is_calibrated and
                    det_a is not None and det_b is not None):

                points_3d = self.calibration.triangulate_pose(
                    det_a, det_b, smooth=True
                )

                if points_3d and self.recorder.is_recording:
                    self.recorder.add_frame(points_3d, det_a, det_b)

            # Collecting calibration frames
            if self.collecting_cal_frames:
                ca, _ = self.calibration.detect_charuco(frame_a)
                cb, _ = self.calibration.detect_charuco(frame_b)

                if ca is not None and cb is not None:
                    cv2.aruco.drawDetectedCornersCharuco(display_a, ca)
                    cv2.aruco.drawDetectedCornersCharuco(display_b, cb)

                    if time.time() - self._last_cal_time > 0.5:
                        self.cal_frames_a.append(frame_a.copy())
                        self.cal_frames_b.append(frame_b.copy())
                        self._last_cal_time = time.time()
                        print(f"  Captured frame {len(self.cal_frames_a)}")

            # FPS counter
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                display_fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.time()

            # Display
            combined = np.hstack([
                cv2.resize(display_a, (640, 360)),
                cv2.resize(display_b, (640, 360))
            ])

            # FPS bar at bottom
            bar = np.zeros((30, combined.shape[1], 3), dtype=np.uint8)
            mode_str = f"RTMW {self.config.POSE_QUALITY}" if self.config.POSE_MODE == 'wholebody' else f"RTMPose {self.config.POSE_QUALITY}"
            cv2.putText(bar, f"{mode_str} | {self.config.POSE_DEVICE.upper()} | {display_fps:.0f} fps",
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
                    print("  Move the ChArUco board around. Collecting every 0.5s when visible.")
                    print("  Press 'S' when you have 20+ frames to run calibration.")
                    self.cal_frames_a = []
                    self.cal_frames_b = []
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

                    self.cal_frames_a = []
                    self.cal_frames_b = []
                else:
                    print(f"\n[ERROR] Need at least 10 frames (have {len(self.cal_frames_a)})")

            elif key == ord('f'):
                if not self.calibration.is_calibrated:
                    print("\n[ERROR] Calibrate cameras first (press C)")
                else:
                    print("\n[FLOOR] Calibrating...")
                    success, msg = self.calibration.calibrate_floor(frame_a, frame_b)
                    if success:
                        self.calibration.save(self.config.CALIBRATION_FILE)
                        print(f"  [OK] {msg}")
                    else:
                        print(f"  [ERROR] {msg}")

            elif key == ord('r'):
                if not self.calibration.is_calibrated:
                    print("\n[ERROR] Calibrate cameras first (press C)")
                elif self.recorder.is_recording:
                    detector_info = f"{self.config.POSE_MODE}_{self.config.POSE_QUALITY}"
                    self.recorder.stop(detector_info=detector_info)
                else:
                    self.calibration.reset_filters()
                    self.recorder.start()

        # Cleanup
        if self.recorder.is_recording:
            self.recorder.stop()

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
