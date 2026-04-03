"""
Motion Capture Recorder
========================
Records triangulated 3D pose data to JSON files.

Output format includes a 'format' field so the Blender addon
can auto-detect the keypoint scheme.
"""

import json
import time
from datetime import datetime
from pathlib import Path


class MocapRecorder:
    """Records motion capture data to JSON."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.frames = []
        self.is_recording = False
        self.start_time = 0
        self.take_name = ""

    def start(self):
        """Start a new recording."""
        self.frames = []
        self.is_recording = True
        self.start_time = time.time()
        self.take_name = f"take_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\n[RECORDING] Started: {self.take_name}")

    def add_frame(self, points_3d=None, detections_a=None, detections_b=None):
        """
        Add a frame to the recording.

        Args:
            points_3d: dict of {keypoint_idx: [x, y, z]}, or None for offline mode
            detections_a: optional dict of {idx: (px, py, conf)} from camera A
            detections_b: optional dict of {idx: (px, py, conf)} from camera B
        """
        if not self.is_recording:
            return

        timestamp = time.time() - self.start_time

        frame_data = {
            "timestamp": timestamp,
        }

        if points_3d is not None:
            # Store [x, y, z] for each keypoint, plus per-keypoint confidence
            frame_data["landmarks_3d"] = {str(k): v for k, v in points_3d.items()}

            # Per-keypoint confidence: min(conf_a, conf_b) for the 2D detections
            # that produced this 3D point. Enables the retargeter to weight or
            # reject low-confidence keypoints.
            if detections_a is not None and detections_b is not None:
                confidences = {}
                for k in points_3d:
                    conf_a = detections_a[k][2] if k in detections_a else 0.0
                    conf_b = detections_b[k][2] if k in detections_b else 0.0
                    confidences[str(k)] = round(min(conf_a, conf_b), 3)
                frame_data["confidence"] = confidences

        # Store raw 2D for potential re-processing
        if detections_a is not None:
            frame_data["raw_2d_a"] = {
                str(k): [v[0], v[1], v[2]]
                for k, v in detections_a.items()
            }

        if detections_b is not None:
            frame_data["raw_2d_b"] = {
                str(k): [v[0], v[1], v[2]]
                for k, v in detections_b.items()
            }

        self.frames.append(frame_data)

    def stop(self, detector_info="rtmw-l", trim_end=0.0, offline_mode=False,
             keypoint_count=17):
        """
        Stop recording and save to file.

        Args:
            detector_info: string describing the pose detector used
            trim_end: seconds to trim from the end of the recording
            offline_mode: if True, saves as raw_detections format for post-processing
            keypoint_count: actual number of keypoints detected (17=body, 133=wholebody)
        """
        if not self.is_recording:
            return None

        self.is_recording = False
        duration = time.time() - self.start_time

        # Trim end frames (removes walking-to-keyboard frames)
        if trim_end > 0 and self.frames:
            cutoff = duration - trim_end
            original_count = len(self.frames)
            self.frames = [f for f in self.frames if f["timestamp"] <= cutoff]
            trimmed = original_count - len(self.frames)
            if trimmed > 0:
                print(f"  Trimmed {trimmed} frames ({trim_end:.1f}s) from end")
            duration = cutoff

        fps = len(self.frames) / duration if duration > 0 else 0

        if offline_mode:
            fmt = "melodiccap_raw_v1"
            suffix = "_raw"
        else:
            fmt = "melodiccap_rtm_v1"
            suffix = ""

        # Report actual keypoint format based on detector output
        if keypoint_count >= 133:
            kp_format = "coco_wholebody_133"
        else:
            kp_format = "coco_body_17"

        data = {
            "format": fmt,
            "keypoint_format": kp_format,
            "detector": detector_info,
            "take_name": self.take_name,
            "duration": duration,
            "fps": fps,
            "frame_count": len(self.frames),
            "created": datetime.now().isoformat(),
            "frames": self.frames
        }

        filepath = self.output_dir / f"{self.take_name}{suffix}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        mode_str = "RAW" if offline_mode else "SAVED"
        print(f"\n[{mode_str}] {filepath}")
        print(f"        {len(self.frames)} frames, {fps:.1f} fps, {duration:.1f}s")
        if offline_mode:
            print(f"        Run offline_processor.py to triangulate this take")

        return str(filepath)
