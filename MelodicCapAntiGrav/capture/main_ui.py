import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import threading
import tkinter as tk
import os
import time
import numpy as np

# Internal Imports
from engine.camera_thread import CameraThread
from engine.view_3d import Skeleton3DView
from engine.triangulation_engine import TriangulationEngine
from engine.recorder_engine import TakeRecorder
from engine.post_processor import ScientificRefiner

# Use paths relative to this script's location for portability
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)  # MelodicCapAntiGrav
_DEFAULT_CAL_PATH = os.path.join(_PROJECT_DIR, "data", "calibration_data", "stereo_calibration.json")
_DEFAULT_TAKES_DIR = os.path.join(_PROJECT_DIR, "mocap_takes")


class MelodicCapStudioApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("MelodicCap Studio V3")
        self.geometry("1400x900")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Custom Fonts
        self.header_font = ctk.CTkFont(family="Inter", size=28, weight="bold")
        self.subhead_font = ctk.CTkFont(family="Inter", size=14, weight="bold")

        # Camera Management
        self.cam_a_thread = None
        self.cam_b_thread = None
        self.available_cams = self.detect_available_cameras()

        # Engines - use portable paths
        self.cal_path = _DEFAULT_CAL_PATH
        os.makedirs(os.path.dirname(self.cal_path), exist_ok=True)
        os.makedirs(_DEFAULT_TAKES_DIR, exist_ok=True)

        self.triangulation_engine = TriangulationEngine(self.cal_path if os.path.exists(self.cal_path) else None)
        self.recorder = TakeRecorder(_DEFAULT_TAKES_DIR)
        self.refiner = ScientificRefiner(self.cal_path) if os.path.exists(self.cal_path) else None
        self.is_recording = False
        self.last_take_path = None

        self.setup_ui()
        self.after(100, self.update_previews)

    def detect_available_cameras(self):
        available = []
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available.append(str(i))
                cap.release()
        return available if available else ["0", "1", "2"]

    def log_to_console(self, msg):
        if hasattr(self, 'debug_console'):
            self.debug_console.insert("end", f"[{time.strftime('%H:%M:%S')}] {msg}\n")
            self.debug_console.see("end")

    def on_cam_change(self, value):
        self.start_previews()

    def setup_ui(self):
        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0, fg_color="#1a1a1a")
        self.sidebar.pack(side="left", fill="y")

        self.logo_label = ctk.CTkLabel(self.sidebar, text="MelodicCap", font=self.header_font, text_color="#3a86ff")
        self.logo_label.pack(pady=30)

        self.btn_capture = ctk.CTkButton(self.sidebar, text="Capture View", command=self.show_capture)
        self.btn_capture.pack(pady=10, padx=20, fill="x")

        self.btn_calibrate = ctk.CTkButton(self.sidebar, text="Calibration", fg_color="transparent", border_width=1, command=self.show_calibration)
        self.btn_calibrate.pack(pady=10, padx=20, fill="x")

        self.btn_assets = ctk.CTkButton(self.sidebar, text="Characters", fg_color="transparent", border_width=1, command=self.show_characters)
        self.btn_assets.pack(pady=10, padx=20, fill="x")

        self.btn_record = ctk.CTkButton(self.sidebar, text="START RECORDING", fg_color="#ff4d4d", hover_color="#cc0000", height=50, font=self.subhead_font, command=self.toggle_recording)
        self.btn_record.pack(pady=30, padx=20, side="bottom")

        self.status_label = ctk.CTkLabel(self.sidebar, text="STATUS: IDLE", font=ctk.CTkFont(size=12))
        self.status_label.pack(pady=10, side="bottom")

        # Refine Button (Production Tools)
        self.btn_refine = ctk.CTkButton(self.sidebar, text="REFINE LATEST TAKE", command=self.manual_refine, fg_color="#A51D2D")
        self.btn_refine.pack(pady=10, padx=20, fill="x", side="bottom")

        # Main Content Area
        self.main_container = ctk.CTkFrame(self, fg_color="#121212")
        self.main_container.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Initialize Views
        self.views = {}
        self._init_capture_view()
        self._init_calibration_view()
        self._init_characters_view()

        # Show initial view
        self.show_capture()

    def _init_capture_view(self):
        view = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.views["capture"] = view
        
        # Upper: Cameras
        self.feed_grid = ctk.CTkFrame(view, fg_color="transparent")
        self.feed_grid.pack(fill="both", expand=True)

        self.cam0_view = ctk.CTkFrame(self.feed_grid, corner_radius=15, border_width=1, border_color="#333")
        self.cam0_view.place(relx=0, rely=0, relwidth=0.49, relheight=0.49)
        self.cam0_label = ctk.CTkLabel(self.cam0_view, text="[ SELECT CAM A ]")
        self.cam0_label.pack(expand=True, fill="both")

        self.cam1_view = ctk.CTkFrame(self.feed_grid, corner_radius=15, border_width=1, border_color="#333")
        self.cam1_view.place(relx=0.51, rely=0, relwidth=0.49, relheight=0.49)
        self.cam1_label = ctk.CTkLabel(self.cam1_view, text="[ SELECT CAM B ]")
        self.cam1_label.pack(expand=True, fill="both")

        # Lower: Debug Console
        self.viz_view = ctk.CTkFrame(self.feed_grid, corner_radius=15, border_width=1, border_color="#333", fg_color="#000")
        self.viz_view.place(relx=0, rely=0.51, relwidth=1.0, relheight=0.49)
        
        self.debug_console = ctk.CTkTextbox(self.viz_view, fg_color="black", text_color="#00ff00", font=ctk.CTkFont(family="Consolas", size=12))
        self.debug_console.pack(fill="both", expand=True, padx=10, pady=10)
        self.log_to_console("MELODICCAP PRODUCTION DEBUGGER READY")
        self.log_to_console("-------------------------------------")

        # Bottom Config
        self.config_box = ctk.CTkFrame(view, height=80, corner_radius=15)
        self.config_box.pack(fill="x", side="bottom", pady=10)

        ctk.CTkLabel(self.config_box, text="Cam A (Sony):").pack(side="left", padx=15)
        self.cam0_menu = ctk.CTkOptionMenu(self.config_box, values=self.available_cams, width=100, command=self.on_cam_change)
        self.cam0_menu.set(self.available_cams[0])
        self.cam0_menu.pack(side="left", padx=5)

        ctk.CTkLabel(self.config_box, text="Cam B (S25):").pack(side="left", padx=15)
        self.cam1_menu = ctk.CTkOptionMenu(self.config_box, values=self.available_cams, width=100, command=self.on_cam_change)
        self.cam1_menu.set(self.available_cams[min(len(self.available_cams)-1, 1)])
        self.cam1_menu.pack(side="left", padx=5)

        self.hand_mode_var = ctk.BooleanVar(value=False)
        self.cb_hand_mode = ctk.CTkCheckBox(self.config_box, text="Hand Close-up Mode", variable=self.hand_mode_var)
        self.cb_hand_mode.pack(side="left", padx=20)

        self.btn_start = ctk.CTkButton(self.config_box, text="START PREVIEW", width=200, command=self.start_previews)
        self.btn_start.pack(side="right", padx=20)

    def _init_calibration_view(self):
        view = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.views["calibration"] = view
        
        header = ctk.CTkLabel(view, text="INTERACTIVE 9-SECTION CALIBRATION", font=self.header_font)
        header.pack(pady=10)
        
        # Camera Views for Calibration with Overlays
        self.cal_feeds_frame = ctk.CTkFrame(view, fg_color="transparent")
        self.cal_feeds_frame.pack(fill="both", expand=True, padx=10)
        
        self.cal_cam0_view = ctk.CTkFrame(self.cal_feeds_frame, corner_radius=15, border_width=1, border_color="#333")
        self.cal_cam0_view.place(relx=0, rely=0, relwidth=0.49, relheight=0.6)
        self.cal_cam0_label = ctk.CTkLabel(self.cal_cam0_view, text="[ CAM A FEED ]")
        self.cal_cam0_label.pack(expand=True, fill="both")
        
        self.cal_cam1_view = ctk.CTkFrame(self.cal_feeds_frame, corner_radius=15, border_width=1, border_color="#333")
        self.cal_cam1_view.place(relx=0.51, rely=0, relwidth=0.49, relheight=0.6)
        self.cal_cam1_label = ctk.CTkLabel(self.cal_cam1_view, text="[ CAM B FEED ]")
        self.cal_cam1_label.pack(expand=True, fill="both")

        # Control Panel
        self.cal_controls = ctk.CTkFrame(view, height=200, fg_color="#1a1a1a", border_width=1, border_color="#444")
        self.cal_controls.pack(fill="x", side="bottom", padx=10, pady=10)
        
        # Section Progress
        self.prog_frame = ctk.CTkFrame(self.cal_controls, fg_color="transparent")
        self.prog_frame.pack(pady=10)
        
        self.cam0_prog_label = ctk.CTkLabel(self.prog_frame, text="CAM A SECTIONS (0/9)", font=self.subhead_font)
        self.cam0_prog_label.pack(side="left", padx=50)
        
        self.cam1_prog_label = ctk.CTkLabel(self.prog_frame, text="CAM B SECTIONS (0/9)", font=self.subhead_font)
        self.cam1_prog_label.pack(side="left", padx=50)
        
        # Steps
        self.btn_solve_ext = ctk.CTkButton(self.cal_controls, text="SOLVE EXTRINSICS", command=self.solve_extrinsics, state="disabled", fg_color="#ffd700", text_color="black")
        self.btn_solve_ext.pack(side="left", padx=20, pady=20)
        
        self.step3_btn = ctk.CTkButton(self.cal_controls, text="ALIGN FLOOR (SET Z=0)", command=self.calibrate_floor, state="disabled")
        self.step3_btn.pack(side="left", padx=20, pady=20)

        self.btn_lock_cal = ctk.CTkButton(self.cal_controls, text="LOCK & ENABLE RECORDING", command=self.lock_calibration, state="disabled", fg_color="#00ff00", text_color="black")
        self.btn_lock_cal.pack(side="right", padx=20, pady=20)

        self.cal_log = ctk.CTkTextbox(view, height=100, fg_color="black", text_color="#00ff00", font=ctk.CTkFont(family="Consolas", size=10))
        self.cal_log.pack(fill="x", padx=10, pady=5, side="bottom")
        
        # State tracking
        self.cal_samples = []
        self.cam0_sections = [False] * 9  # 3x3 grid
        self.cam1_sections = [False] * 9
        self.is_cal_locked = False

    def _init_characters_view(self):
        view = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.views["characters"] = view
        ctk.CTkLabel(view, text="CHARACTER SELECTION (COMING SOON)", font=self.header_font).pack(pady=50)

    def show_capture(self):
        self._set_active_view("capture")
        self.btn_capture.configure(fg_color="#3a86ff")
        self.btn_calibrate.configure(fg_color="transparent")
        self.btn_assets.configure(fg_color="transparent")
        if self.cam_a_thread: self.cam_a_thread.calibration_mode = False
        if self.cam_b_thread: self.cam_b_thread.calibration_mode = False

    def show_calibration(self):
        self._set_active_view("calibration")
        self.btn_capture.configure(fg_color="transparent")
        self.btn_calibrate.configure(fg_color="#3a86ff")
        self.btn_assets.configure(fg_color="transparent")
        if self.cam_a_thread: self.cam_a_thread.calibration_mode = True
        if self.cam_b_thread: self.cam_b_thread.calibration_mode = True
        self.cal_log.insert("end", "Calibration Mode Active: Move board to red squares.\n")

    def show_characters(self):
        self._set_active_view("characters")
        self.btn_capture.configure(fg_color="transparent")
        self.btn_calibrate.configure(fg_color="transparent")
        self.btn_assets.configure(fg_color="#3a86ff")
        if self.cam_a_thread: self.cam_a_thread.calibration_mode = False
        if self.cam_b_thread: self.cam_b_thread.calibration_mode = False

    def _check_camera_drift(self):
        """Watchdog: Detects if camera has moved significantly after lock"""
        if not self.is_cal_locked: return
        
        # Throttled check (every 2 seconds)
        if hasattr(self, '_last_drift_check') and time.time() - self._last_drift_check < 2:
            return
        self._last_drift_check = time.time()

        if self.cam_a_thread and self.cam_b_thread:
            # Check if board is visible on the floor
            if self.cam_a_thread.board_detected_section is not None and self.cam_b_thread.board_detected_section is not None:
                # Run a quick floor verification
                board_config = None  # FloorCalibrator now uses canonical config
                from engine.floor_calibrator import FloorCalibrator
                fc = FloorCalibrator()
                
                res, _ = fc.detect_floor_from_frames(
                    cv2.cvtColor(self.cam_a_thread.latest_frame, cv2.COLOR_RGB2BGR),
                    cv2.cvtColor(self.cam_b_thread.latest_frame, cv2.COLOR_RGB2BGR),
                    self.triangulation_engine
                )
                
                if res:
                    # Compare new normal with existing engine floor normal
                    curr_normal = np.array(self.triangulation_engine.floor_normal)
                    new_normal = np.array(res['normal'])
                    dot = np.dot(curr_normal, new_normal)
                    
                    # If angle > 5 degrees or height diff > 2cm
                    if dot < 0.996: # cos(5deg) approx 0.996
                        self.log_to_console("WARNING: CAMERA DRIFT DETECTED! Recalibrate Floor.")
                        self.status_label.configure(text="STATUS: CALIBRATION DRIFT!", text_color="#ffcc00")
                        self.is_cal_locked = False # Force unlock
                        self.btn_lock_cal.configure(state="normal", text="RE-LOCK CALIBRATION", fg_color="#ffcc00")

    def _set_active_view(self, view_name):
        for name, frame in self.views.items():
            if name == view_name:
                frame.pack(fill="both", expand=True)
            else:
                frame.pack_forget()

    def capture_cal_sample(self):
        """Captures a frame sample for extrinsic verification"""
        if not self.cam_a_thread or not self.cam_b_thread:
            return
        
        self.cal_samples.append((self.cam_a_thread.latest_frame, self.cam_b_thread.latest_frame))
        count = len(self.cal_samples)
        self.sample_count_label.configure(text=f"Samples: {count}/5")
        self.cal_log.insert("end", f"Sample {count} captured.\n")
        
        if count >= 3:
            self.btn_solve_ext.configure(state="normal")

    def solve_extrinsics(self):
        """Processes collected samples to refine extrinsic calibration"""
        if not self.cal_samples: return
        
        self.cal_log.insert("end", "Solving extrinsics (Fixing Intrinsics)...\n")
        self.btn_solve_ext.configure(state="disabled", text="SOLVING...")
        
        from engine.floor_calibrator import FloorCalibrator
        fc = FloorCalibrator()

        success, msg = self.triangulation_engine.calibrate_extrinsic_from_samples(self.cal_samples, fc.board)
        
        if success:
            self.cal_log.insert("end", f"SUCCESS: {msg}\n")
            self.step3_btn.configure(state="normal")
            # Save refined calibration
            self.triangulation_engine.save_calibration(self.cal_path)
            self.btn_solve_ext.configure(text="SOLVED", fg_color="#00ff00")
        else:
            self.cal_log.insert("end", f"FAILED: {msg}\n")
            self.btn_solve_ext.configure(state="normal", text="RETRY SOLVE")

    def calibrate_floor(self):
        """Triggers the FloorCalibrator using current frames"""
        if not self.cam_a_thread or not self.cam_b_thread:
            print("Start previews before calibrating floor.")
            return
            
        from engine.floor_calibrator import FloorCalibrator
        # Use board config from iPad 40mm board
        board_config = None  # FloorCalibrator now uses canonical config
        fc = FloorCalibrator()
        
        frame_a = self.cam_a_thread.latest_frame
        frame_b = self.cam_b_thread.latest_frame
        
        if frame_a is not None and frame_b is not None:
            # Convert back to BGR for OpenCV
            fa_bgr = cv2.cvtColor(frame_a, cv2.COLOR_RGB2BGR)
            fb_bgr = cv2.cvtColor(frame_b, cv2.COLOR_RGB2BGR)
            
            res, msg = fc.detect_floor_from_frames(fa_bgr, fb_bgr, self.triangulation_engine)
            if res:
                self.triangulation_engine.set_floor_plane(res['normal'], res['point'])
                self.status_label.configure(text=f"FLOOR: {msg}", text_color="#00ff00")
                self.cal_log.insert("end", f"FLOOR ALIGNED: {msg}\n")
                self.btn_lock_cal.configure(state="normal")
            else:
                self.status_label.configure(text=f"ERROR: {msg}", text_color="#ff4d4d")

    def start_previews(self):
        if self.cam_a_thread: self.cam_a_thread.stop()
        if self.cam_b_thread: self.cam_b_thread.stop()
        def _reinit():
            self.cam_a_thread = CameraThread(int(self.cam0_menu.get()), None)
            self.cam_b_thread = CameraThread(int(self.cam1_menu.get()), None)
            self.cam_a_thread.start()
            self.cam_b_thread.start()
        self.after(600, _reinit)

    def manual_refine(self):
        """Manually trigger refinement on the last recorded take"""
        if not self.last_take_path or not os.path.exists(self.last_take_path):
            self.log_to_console("Error: No take found to refine.")
            return
            
        self.btn_refine.configure(state="disabled", text="REFINING...")
        self.log_to_console(f"Manually refining: {os.path.basename(self.last_take_path)}")
        
        def run_refine():
            self.refiner.refine_take(self.last_take_path)
            self.log_to_console("Manual Refinement Complete.")
            self.after(0, lambda: self.btn_refine.configure(state="normal", text="REFINE LATEST TAKE"))
            
        threading.Thread(target=run_refine, daemon=True).start()

    def update_previews(self):
        # Determine active view to optimize preview
        is_cal_view = self.main_container.winfo_children() and self.main_container.winfo_children()[0] == self.views.get("calibration")
        
        # 1. Update Camera A
        if self.cam_a_thread and self.cam_a_thread.latest_frame is not None:
            raw_frame_a = self.cam_a_thread.latest_frame.copy()
            if is_cal_view:
                raw_frame_a = self._process_cal_frame(raw_frame_a, 0)
            
            img = Image.fromarray(raw_frame_a)
            ctk_img = ctk.CTkImage(img, size=(640, 360) if not is_cal_view else (None, None)) # Scaling handled by label if None
            
            if is_cal_view:
                self.cal_cam0_label.configure(image=ctk_img, text="")
            else:
                self.cam0_label.configure(image=ctk_img, text="")

        # 2. Update Camera B
        if self.cam_b_thread and self.cam_b_thread.latest_frame is not None:
            raw_frame_b = self.cam_b_thread.latest_frame.copy()
            if is_cal_view:
                raw_frame_b = self._process_cal_frame(raw_frame_b, 1)
            
            img = Image.fromarray(raw_frame_b)
            ctk_img = ctk.CTkImage(img, size=(640, 360) if not is_cal_view else (None, None))
            
            if is_cal_view:
                self.cal_cam1_label.configure(image=ctk_img, text="")
            else:
                self.cam1_label.configure(image=ctk_img, text="")

        # 3. Process 3D Triangulation for Capture/Recording (Only if not in calibration view)
        points_3d = None
        lms_a, lms_b = None, None
        hands_a, hands_b = None, None
        
        if not is_cal_view and self.cam_a_thread and self.cam_b_thread:
            lms_a = self.cam_a_thread.landmarks.get("pose") if self.cam_a_thread.landmarks else None
            lms_b = self.cam_b_thread.landmarks.get("pose") if self.cam_b_thread.landmarks else None
            hands_a = self.cam_a_thread.landmarks.get("hands") if self.cam_a_thread.landmarks else None
            hands_b = self.cam_b_thread.landmarks.get("hands") if self.cam_b_thread.landmarks else None
            
            if lms_a and lms_b and self.triangulation_engine.is_calibrated:
                # Check frame sync: warn if cameras are >50ms apart
                if self.cam_a_thread.frame_timestamp and self.cam_b_thread.frame_timestamp:
                    sync_delta = abs(self.cam_a_thread.frame_timestamp - self.cam_b_thread.frame_timestamp)
                    if sync_delta > 0.05 and time.time() % 5 < 0.1:
                        self.log_to_console(f"WARNING: Frame sync delta {sync_delta*1000:.0f}ms")

                points_3d = self.triangulation_engine.triangulate_pose(lms_a, lms_b, smooth=True)
                if self.hand_mode_var.get() and hands_a and hands_b:
                    self.log_to_console(f"Triangulating Fingers... (Lms: {len(lms_a.landmark)})")
            elif self.cam_a_thread.landmarks and "pose_world" in self.cam_a_thread.landmarks:
                pw = self.cam_a_thread.landmarks["pose_world"]
                if pw:
                    # pose_world is in meters centered on hip; scale is reasonable as-is
                    points_3d = {i: [lm.x, lm.z, -lm.y] for i, lm in enumerate(pw.landmark)}
                    if time.time() % 2 < 0.1:
                        self.log_to_console("Single-Cam Fallback (Pose World)")

        if points_3d:
            if self.is_recording:
                self.recorder.add_frame(points_3d, lms_a, lms_b, hands_a, hands_b)

        # 4. Drift Watchdog
        self._check_camera_drift()

        self.after(33, self.update_previews)

    def _process_cal_frame(self, frame, cam_idx):
        """Uses background thread results to draw grid overlay"""
        h, w = frame.shape[:2]
        thread = self.cam_a_thread if cam_idx == 0 else self.cam_b_thread
        sections = self.cam0_sections if cam_idx == 0 else self.cam1_sections
        
        # Get result from thread
        sec_idx = thread.board_detected_section if thread else None
        
        if sec_idx is not None and 0 <= sec_idx < 9:
            if not sections[sec_idx]:
                sections[sec_idx] = True
                self.cal_log.insert("end", f"Cam {chr(65+cam_idx)}: Section {sec_idx+1} captured!\n")
                self._update_cal_progress()
                
                # Store sample for solving
                f1 = self.cam_a_thread.latest_frame.copy() if self.cam_a_thread else None
                f2 = self.cam_b_thread.latest_frame.copy() if self.cam_b_thread else None
                if f1 is not None and f2 is not None:
                    self.cal_samples.append((f1, f2))

        # DRAW GRID OVERLAY (Fast)
        overlay = frame.copy()
        for i in range(9):
            sx, sy = i % 3, i // 3
            x1, y1 = int(sx * (w/3)), int(sy * (h/3))
            x2, y2 = int((sx+1) * (w/3)), int((sy+1) * (h/3))
            
            color = (0, 255, 0) if sections[i] else (255, 0, 0)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        return frame

    def _update_cal_progress(self):
        c0 = sum(self.cam0_sections)
        c1 = sum(self.cam1_sections)
        self.cam0_prog_label.configure(text=f"CAM A SECTIONS ({c0}/9)")
        self.cam1_prog_label.configure(text=f"CAM B SECTIONS ({c1}/9)")
        
        # Need coverage in both to solve extrinsics (e.g. at least 3 sections each)
        if c0 >= 3 and c1 >= 3:
            self.btn_solve_ext.configure(state="normal")

    def lock_calibration(self):
        self.is_cal_locked = True
        self.btn_lock_cal.configure(text="CALIBRATION LOCKED", fg_color="#333", state="disabled")
        self.cal_log.insert("end", "STUDIO READY: Calibration Locked. Capture View Enabled.\n")
        self.show_capture()

    def toggle_recording(self):
        if not self.is_cal_locked:
            self.status_label.configure(text="ERROR: PERFORM CALIBRATION FIRST", text_color="#ff4d4d")
            self.show_calibration()
            return
            
        if not self.is_recording:
            if not self.cam_a_thread: return
            self.is_recording = True
            self.recorder.start()
            self.btn_record.configure(text="STOP RECORDING", fg_color="#454545")
            self.status_label.configure(text="STATUS: RECORDING...", text_color="#ff4d4d")
        else:
            self.is_recording = False
            self.status_label.configure(text="STATUS: REFINING...", text_color="#ffd700")
            file_path = self.recorder.stop()
            if file_path:
                refined = self.refiner.refine_take(file_path)
                self.status_label.configure(text=f"SAVED: {os.path.basename(refined)}", text_color="#3a86ff")
                self.last_take_path = file_path
            self.btn_record.configure(text="START RECORDING", fg_color="#1f538d")

if __name__ == "__main__":
    app = MelodicCapStudioApp()
    app.mainloop()
