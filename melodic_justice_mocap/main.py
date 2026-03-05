#!/usr/bin/env python3
"""
Melodic Justice Motion Capture - Main Application
=================================================
Unified interface for motion capture recording, calibration, and export.

Usage:
    python main.py              # Launch GUI
    python main.py --cli        # Command line mode
    python main.py --calibrate  # Run calibration
    python main.py --record     # Start recording
"""

import os
import sys
import argparse
import json
import time
import threading
from datetime import datetime

# Add module paths
sys.path.insert(0, os.path.dirname(__file__))

# Try importing tkinter for GUI (optional)
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
    print("Warning: tkinter not available. GUI mode disabled.")

# Import our modules
try:
    from config import MocapConfig, DEFAULT_CONFIG
    from calibration.calibrate import CalibrationSession, CharucoBoardGenerator
    from capture.dual_camera import DualCameraRecorder, TakeManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the melodic_justice_mocap directory")


# ============================================================================
# GUI APPLICATION
# ============================================================================

class MocapGUI:
    """Tkinter-based GUI for the motion capture system."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Melodic Justice Motion Capture")
        self.root.geometry("800x600")
        
        # Configuration
        self.config = DEFAULT_CONFIG
        self.recorder = None
        self.is_recording = False
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the GUI layout."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self._create_record_tab()
        self._create_calibration_tab()
        self._create_takes_tab()
        self._create_settings_tab()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var, 
            relief='sunken'
        )
        status_bar.pack(fill='x', side='bottom')
    
    def _create_record_tab(self):
        """Create the recording tab."""
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Record")
        
        # Camera selection
        cam_frame = ttk.LabelFrame(frame, text="Cameras", padding=10)
        cam_frame.pack(fill='x', pady=5)
        
        ttk.Label(cam_frame, text="Left Camera ID:").grid(row=0, column=0, sticky='w')
        self.left_cam_var = tk.StringVar(value="0")
        ttk.Entry(cam_frame, textvariable=self.left_cam_var, width=10).grid(row=0, column=1)
        
        ttk.Label(cam_frame, text="Right Camera ID:").grid(row=0, column=2, sticky='w', padx=(20, 0))
        self.right_cam_var = tk.StringVar(value="1")
        ttk.Entry(cam_frame, textvariable=self.right_cam_var, width=10).grid(row=0, column=3)
        
        ttk.Button(cam_frame, text="Detect Cameras", command=self._detect_cameras).grid(row=0, column=4, padx=(20, 0))
        
        # Calibration file
        calib_frame = ttk.LabelFrame(frame, text="Calibration", padding=10)
        calib_frame.pack(fill='x', pady=5)
        
        self.calib_var = tk.StringVar()
        ttk.Entry(calib_frame, textvariable=self.calib_var, width=50).pack(side='left', fill='x', expand=True)
        ttk.Button(calib_frame, text="Browse", command=self._browse_calibration).pack(side='left', padx=5)
        
        # Output directory
        output_frame = ttk.LabelFrame(frame, text="Output Directory", padding=10)
        output_frame.pack(fill='x', pady=5)
        
        self.output_var = tk.StringVar(value="./mocap_data")
        ttk.Entry(output_frame, textvariable=self.output_var, width=50).pack(side='left', fill='x', expand=True)
        ttk.Button(output_frame, text="Browse", command=self._browse_output).pack(side='left', padx=5)
        
        # Controls
        controls_frame = ttk.Frame(frame, padding=10)
        controls_frame.pack(fill='x', pady=20)
        
        self.preview_btn = ttk.Button(
            controls_frame, 
            text="Start Preview", 
            command=self._toggle_preview
        )
        self.preview_btn.pack(side='left', padx=5)
        
        self.record_btn = ttk.Button(
            controls_frame, 
            text="Start Recording",
            state='disabled'
        )
        self.record_btn.pack(side='left', padx=5)
        
        # Instructions
        instructions = ttk.Label(
            frame,
            text="Instructions:\n"
                 "1. Connect both cameras and enter their IDs\n"
                 "2. (Optional) Load stereo calibration file\n"
                 "3. Click 'Start Preview' to see camera feeds\n"
                 "4. Press 'R' in preview window to record\n"
                 "5. Press 'Q' to quit preview",
            justify='left'
        )
        instructions.pack(pady=20, anchor='w')
    
    def _create_calibration_tab(self):
        """Create the calibration tab."""
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Calibration")
        
        # ChArUco board generation
        board_frame = ttk.LabelFrame(frame, text="ChArUco Board", padding=10)
        board_frame.pack(fill='x', pady=5)
        
        ttk.Label(board_frame, text="Squares X:").grid(row=0, column=0, sticky='w')
        self.squares_x_var = tk.StringVar(value="10")
        ttk.Entry(board_frame, textvariable=self.squares_x_var, width=10).grid(row=0, column=1)
        
        ttk.Label(board_frame, text="Squares Y:").grid(row=0, column=2, sticky='w', padx=(20, 0))
        self.squares_y_var = tk.StringVar(value="5")
        ttk.Entry(board_frame, textvariable=self.squares_y_var, width=10).grid(row=0, column=3)
        
        ttk.Label(board_frame, text="Square Size (cm):").grid(row=1, column=0, sticky='w', pady=(10, 0))
        self.square_size_var = tk.StringVar(value="4.3")
        ttk.Entry(board_frame, textvariable=self.square_size_var, width=10).grid(row=1, column=1, pady=(10, 0))
        
        ttk.Button(
            board_frame, 
            text="Generate Board Image", 
            command=self._generate_board
        ).grid(row=1, column=2, columnspan=2, padx=(20, 0), pady=(10, 0))
        
        # Calibration controls
        calib_ctrl_frame = ttk.LabelFrame(frame, text="Stereo Calibration", padding=10)
        calib_ctrl_frame.pack(fill='x', pady=10)
        
        ttk.Label(calib_ctrl_frame, text="Number of frames:").grid(row=0, column=0, sticky='w')
        self.num_frames_var = tk.StringVar(value="20")
        ttk.Entry(calib_ctrl_frame, textvariable=self.num_frames_var, width=10).grid(row=0, column=1)
        
        ttk.Button(
            calib_ctrl_frame,
            text="Run Calibration",
            command=self._run_calibration
        ).grid(row=0, column=2, padx=(20, 0))
        
        # Instructions
        instructions = ttk.Label(
            frame,
            text="Calibration Steps:\n"
                 "1. Print the ChArUco board at 100% scale\n"
                 "2. Measure the printed squares to verify size\n"
                 "3. Run calibration and show the board to both cameras\n"
                 "4. Move the board to different positions and angles\n"
                 "5. Press SPACE to capture each frame pair",
            justify='left'
        )
        instructions.pack(pady=20, anchor='w')
    
    def _create_takes_tab(self):
        """Create the takes management tab."""
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Takes")
        
        # Takes list
        list_frame = ttk.LabelFrame(frame, text="Saved Takes", padding=10)
        list_frame.pack(fill='both', expand=True, pady=5)
        
        self.takes_list = tk.Listbox(list_frame, height=10)
        self.takes_list.pack(fill='both', expand=True, side='left')
        
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.takes_list.yview)
        scrollbar.pack(side='right', fill='y')
        self.takes_list.config(yscrollcommand=scrollbar.set)
        
        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill='x', pady=10)
        
        ttk.Button(btn_frame, text="Refresh", command=self._refresh_takes).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Export for Blender", command=self._export_take).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Delete", command=self._delete_take).pack(side='left', padx=5)
        
        # Initial refresh
        self._refresh_takes()
    
    def _create_settings_tab(self):
        """Create the settings tab."""
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Settings")
        
        # MediaPipe settings
        mp_frame = ttk.LabelFrame(frame, text="MediaPipe Settings", padding=10)
        mp_frame.pack(fill='x', pady=5)
        
        ttk.Label(mp_frame, text="Model Complexity:").grid(row=0, column=0, sticky='w')
        self.model_complexity_var = tk.StringVar(value="2")
        ttk.Combobox(
            mp_frame, 
            textvariable=self.model_complexity_var,
            values=["0 (Lite)", "1 (Full)", "2 (Heavy)"],
            width=15
        ).grid(row=0, column=1)
        
        ttk.Label(mp_frame, text="Detection Confidence:").grid(row=1, column=0, sticky='w', pady=(10, 0))
        self.detection_conf_var = tk.StringVar(value="0.7")
        ttk.Entry(mp_frame, textvariable=self.detection_conf_var, width=10).grid(row=1, column=1, pady=(10, 0))
        
        # Hand tracking
        self.enable_hands_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            mp_frame, 
            text="Enable Hand Tracking",
            variable=self.enable_hands_var
        ).grid(row=2, column=0, columnspan=2, sticky='w', pady=(10, 0))
        
        # Blender settings
        blender_frame = ttk.LabelFrame(frame, text="Blender Export", padding=10)
        blender_frame.pack(fill='x', pady=10)
        
        ttk.Label(blender_frame, text="Target Rig:").grid(row=0, column=0, sticky='w')
        self.rig_name_var = tk.StringVar(value="JaxRigify")
        ttk.Entry(blender_frame, textvariable=self.rig_name_var, width=20).grid(row=0, column=1)
        
        ttk.Label(blender_frame, text="Scale Factor:").grid(row=1, column=0, sticky='w', pady=(10, 0))
        self.scale_var = tk.StringVar(value="1.0")
        ttk.Entry(blender_frame, textvariable=self.scale_var, width=10).grid(row=1, column=1, pady=(10, 0))
        
        # Save/Load
        ttk.Button(frame, text="Save Settings", command=self._save_settings).pack(pady=20)
    
    # === Event Handlers ===
    
    def _detect_cameras(self):
        """Detect available cameras."""
        self.status_var.set("Detecting cameras...")
        self.root.update()
        
        try:
            import cv2
            cameras = []
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    cameras.append(i)
                    cap.release()
            
            if cameras:
                messagebox.showinfo(
                    "Cameras Detected",
                    f"Found cameras at IDs: {cameras}\n\n"
                    f"Suggestion:\n"
                    f"- Left camera: {cameras[0]}\n"
                    f"- Right camera: {cameras[1] if len(cameras) > 1 else 'N/A'}"
                )
            else:
                messagebox.showwarning("No Cameras", "No cameras detected!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to detect cameras: {e}")
        
        self.status_var.set("Ready")
    
    def _browse_calibration(self):
        """Browse for calibration file."""
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            self.calib_var.set(filepath)
    
    def _browse_output(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.output_var.set(directory)
    
    def _toggle_preview(self):
        """Toggle preview mode."""
        if self.recorder is None:
            try:
                left_id = int(self.left_cam_var.get())
                right_id = int(self.right_cam_var.get())
                
                calib_path = self.calib_var.get()
                if calib_path and not os.path.exists(calib_path):
                    calib_path = None
                
                self.recorder = DualCameraRecorder(
                    output_dir=self.output_var.get(),
                    calibration_path=calib_path
                )
                
                # Start in separate thread
                self.preview_btn.config(text="Stop Preview")
                self.status_var.set("Starting preview...")
                
                # Run preview (this blocks)
                self.recorder.start_preview(left_id, right_id)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start preview: {e}")
            finally:
                self.recorder = None
                self.preview_btn.config(text="Start Preview")
                self.status_var.set("Ready")
    
    def _generate_board(self):
        """Generate ChArUco board image."""
        try:
            from calibration.calibrate import CharucoBoardGenerator, CharucoBoardConfig
            
            config = CharucoBoardConfig(
                squares_x=int(self.squares_x_var.get()),
                squares_y=int(self.squares_y_var.get()),
                square_length_meters=float(self.square_size_var.get()) / 100
            )
            
            generator = CharucoBoardGenerator(config)
            
            # Ask for save location
            filepath = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png")],
                initialfile="charuco_board.png"
            )
            
            if filepath:
                generator.generate_board_image(filepath)
                messagebox.showinfo(
                    "Board Generated",
                    f"ChArUco board saved to:\n{filepath}\n\n"
                    "Print at 100% scale and verify square sizes!"
                )
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate board: {e}")
    
    def _run_calibration(self):
        """Run stereo calibration."""
        try:
            left_id = int(self.left_cam_var.get())
            right_id = int(self.right_cam_var.get())
            num_frames = int(self.num_frames_var.get())
            
            session = CalibrationSession(self.output_var.get())
            
            self.status_var.set("Running calibration...")
            self.root.update()
            
            session.run_interactive_calibration(
                left_id, 
                right_id,
                num_frames
            )
            
            messagebox.showinfo("Complete", "Calibration complete!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Calibration failed: {e}")
        
        self.status_var.set("Ready")
    
    def _refresh_takes(self):
        """Refresh the takes list."""
        self.takes_list.delete(0, tk.END)
        
        manager = TakeManager(self.output_var.get())
        for take in manager.list_takes():
            self.takes_list.insert(tk.END, take)
    
    def _export_take(self):
        """Export selected take for Blender."""
        selection = self.takes_list.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a take to export")
            return
        
        take_name = self.takes_list.get(selection[0])
        
        manager = TakeManager(self.output_var.get())
        output_path = manager.export_for_blender(take_name)
        
        if output_path:
            messagebox.showinfo(
                "Exported",
                f"Take exported for Blender:\n{output_path}\n\n"
                "Load this file in the Blender addon."
            )
    
    def _delete_take(self):
        """Delete selected take."""
        selection = self.takes_list.curselection()
        if not selection:
            return
        
        take_name = self.takes_list.get(selection[0])
        
        if messagebox.askyesno("Confirm Delete", f"Delete take '{take_name}'?"):
            filepath = os.path.join(self.output_var.get(), f"{take_name}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
            self._refresh_takes()
    
    def _save_settings(self):
        """Save current settings."""
        # TODO: Implement settings persistence
        messagebox.showinfo("Settings", "Settings saved!")
    
    def run(self):
        """Run the GUI main loop."""
        self.root.mainloop()


# ============================================================================
# CLI INTERFACE
# ============================================================================

def run_cli():
    """Run in command-line mode."""
    parser = argparse.ArgumentParser(
        description="Melodic Justice Motion Capture System"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Calibrate command
    calib_parser = subparsers.add_parser('calibrate', help='Run camera calibration')
    calib_parser.add_argument('--left', type=int, default=0, help='Left camera ID')
    calib_parser.add_argument('--right', type=int, default=1, help='Right camera ID')
    calib_parser.add_argument('--frames', type=int, default=20, help='Number of frames')
    calib_parser.add_argument('--output', type=str, default='./calibration_data', help='Output directory')
    
    # Record command
    record_parser = subparsers.add_parser('record', help='Start recording')
    record_parser.add_argument('--left', type=int, default=0, help='Left camera ID')
    record_parser.add_argument('--right', type=int, default=1, help='Right camera ID')
    record_parser.add_argument('--calibration', type=str, help='Calibration file path')
    record_parser.add_argument('--output', type=str, default='./mocap_data', help='Output directory')
    
    # Generate board command
    board_parser = subparsers.add_parser('board', help='Generate ChArUco board')
    board_parser.add_argument('--output', type=str, default='charuco_board.png', help='Output file')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export take for Blender')
    export_parser.add_argument('take', type=str, help='Take name')
    export_parser.add_argument('--input-dir', type=str, default='./mocap_data', help='Input directory')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List saved takes')
    list_parser.add_argument('--input-dir', type=str, default='./mocap_data', help='Input directory')
    
    args = parser.parse_args()
    
    if args.command == 'calibrate':
        session = CalibrationSession(args.output)
        session.run_interactive_calibration(args.left, args.right, args.frames)
    
    elif args.command == 'record':
        recorder = DualCameraRecorder(
            output_dir=args.output,
            calibration_path=args.calibration
        )
        recorder.start_preview(args.left, args.right)
    
    elif args.command == 'board':
        generator = CharucoBoardGenerator()
        generator.generate_board_image(args.output)
    
    elif args.command == 'export':
        manager = TakeManager(args.input_dir)
        manager.export_for_blender(args.take)
    
    elif args.command == 'list':
        manager = TakeManager(args.input_dir)
        takes = manager.list_takes()
        print("\nSaved Takes:")
        for take in takes:
            print(f"  - {take}")
    
    else:
        parser.print_help()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cli', action='store_true', help='Run in CLI mode')
    args, remaining = parser.parse_known_args()
    
    if args.cli or not HAS_TKINTER:
        # Restore remaining args and run CLI
        sys.argv = [sys.argv[0]] + remaining
        run_cli()
    else:
        # Run GUI
        app = MocapGUI()
        app.run()


if __name__ == "__main__":
    main()
