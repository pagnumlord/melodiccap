"""
RTMPose / RTMW Pose Detector
==============================
Wraps rtmlib for 2D pose estimation.

Supports two modes:
- 'body': RTMPose body-only (17 COCO keypoints) — fastest
- 'wholebody': RTMW wholebody (133 keypoints) — body + hands + face + feet

Output is always in COCO-WholeBody index format, matching LM indices directly.

Hardware note:
- RTX 3060 Ti with onnxruntime-gpu: RTMW-l runs 60+ FPS easily
- CPU fallback (Ryzen 5 3600): RTMPose-m body-only ~30 FPS
"""

import numpy as np

# Model configs: (model_name, input_size)
# These are downloaded automatically by rtmlib on first use (~50-200MB)
MODELS = {
    'body': {
        'fast': ('rtmpose-s', (256, 192)),
        'balanced': ('rtmpose-m', (256, 192)),
        'accurate': ('rtmpose-l', (256, 192)),
    },
    'wholebody': {
        'fast': ('rtmw-l', (256, 192)),
        'balanced': ('rtmw-l', (384, 288)),
        'accurate': ('rtmw-x', (384, 288)),
    },
}


class PoseDetector:
    """
    2D pose estimation using RTMPose/RTMW via rtmlib.

    Returns keypoints and confidence scores indexed by COCO-WholeBody format.
    """

    def __init__(self, mode='wholebody', quality='balanced', device='cuda',
                 backend='onnxruntime'):
        """
        Args:
            mode: 'body' (17 keypoints) or 'wholebody' (133 keypoints)
            quality: 'fast', 'balanced', or 'accurate'
            device: 'cuda' or 'cpu'
            backend: 'onnxruntime' (recommended), 'opencv', or 'openvino'
        """
        self.mode = mode
        self.quality = quality
        self.num_keypoints = 133 if mode == 'wholebody' else 17

        model_name, input_size = MODELS[mode][quality]

        if mode == 'wholebody':
            from rtmlib import Wholebody
            self.detector = Wholebody(
                pose=model_name,
                pose_input_size=input_size,
                backend=backend,
                device=device,
            )
        else:
            from rtmlib import Body
            self.detector = Body(
                pose=model_name,
                pose_input_size=input_size,
                backend=backend,
                device=device,
            )

        print(f"[PoseDetector] mode={mode}, model={model_name}, "
              f"input={input_size}, device={device}")

    def detect(self, frame):
        """
        Run pose detection on a single BGR frame.

        Args:
            frame: BGR numpy array (H, W, 3)

        Returns:
            keypoints: (N, K, 2) array of (x, y) pixel coordinates per person
            scores: (N, K) array of confidence scores per keypoint

        Where N = number of detected people, K = 17 or 133.
        For single-person mocap, use index [0] for the primary person.
        Returns (None, None) if no person detected.
        """
        keypoints, scores = self.detector(frame)

        if keypoints is None or len(keypoints) == 0:
            return None, None

        return keypoints, scores

    def detect_single(self, frame, min_confidence=0.3):
        """
        Detect the primary person in frame.

        Returns:
            detections: dict of {keypoint_idx: (pixel_x, pixel_y, confidence)}
                        Only includes keypoints above min_confidence.
            Returns None if no person detected.
        """
        keypoints, scores = self.detect(frame)

        if keypoints is None:
            return None

        # Take the first (most confident) person
        kps = keypoints[0]   # (K, 2)
        scrs = scores[0]     # (K,)

        detections = {}
        for i in range(len(kps)):
            conf = float(scrs[i])
            if conf >= min_confidence:
                detections[i] = (float(kps[i][0]), float(kps[i][1]), conf)

        return detections

    def get_num_keypoints(self):
        return self.num_keypoints

    def get_mode(self):
        return self.mode
