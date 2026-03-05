"""
Melodic Justice Motion Capture - Configuration
==============================================
Central configuration for the dual-camera mocap system.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json

# ============================================================================
# MEDIAPIPE LANDMARK DEFINITIONS
# ============================================================================

# MediaPipe Pose landmarks (33 points)
POSE_LANDMARKS = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer", 
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}

# MediaPipe Hand landmarks (21 points per hand)
HAND_LANDMARKS = {
    0: "wrist",
    1: "thumb_cmc",
    2: "thumb_mcp",
    3: "thumb_ip",
    4: "thumb_tip",
    5: "index_mcp",
    6: "index_pip",
    7: "index_dip",
    8: "index_tip",
    9: "middle_mcp",
    10: "middle_pip",
    11: "middle_dip",
    12: "middle_tip",
    13: "ring_mcp",
    14: "ring_pip",
    15: "ring_dip",
    16: "ring_tip",
    17: "pinky_mcp",
    18: "pinky_pip",
    19: "pinky_dip",
    20: "pinky_tip",
}

# ============================================================================
# RIGIFY BONE MAPPING
# ============================================================================

# MediaPipe to Rigify control bone mapping
# This maps MediaPipe landmarks to Rigify CONTROL bones (not DEF bones!)
MEDIAPIPE_TO_RIGIFY_MAPPING = {
    # === TORSO AND ROOT ===
    "hip_center": "torso",  # Computed from left_hip + right_hip
    "spine_center": "chest",  # Computed from shoulders + hips
    
    # === HEAD AND NECK ===
    "nose": "head",
    "neck_base": "neck",  # Computed from shoulders midpoint
    
    # === LEFT ARM (IK) ===
    "left_shoulder": "shoulder.L",
    "left_wrist": "hand_ik.L",
    "left_elbow": "upper_arm_ik_target.L",  # Pole target
    
    # === RIGHT ARM (IK) ===
    "right_shoulder": "shoulder.R", 
    "right_wrist": "hand_ik.R",
    "right_elbow": "upper_arm_ik_target.R",  # Pole target
    
    # === LEFT LEG (IK) ===
    "left_hip": "thigh_parent.L",
    "left_ankle": "foot_ik.L",
    "left_knee": "thigh_ik_target.L",  # Pole target
    "left_foot_index": "toe_ik.L",
    
    # === RIGHT LEG (IK) ===
    "right_hip": "thigh_parent.R",
    "right_ankle": "foot_ik.R",
    "right_knee": "thigh_ik_target.R",  # Pole target
    "right_foot_index": "toe_ik.R",
    
    # === FINGERS (LEFT) ===
    "left_thumb_tip": "thumb.03.L",
    "left_index_tip": "f_index.03.L",
    "left_middle_tip": "f_middle.03.L",
    "left_ring_tip": "f_ring.03.L",
    "left_pinky_tip": "f_pinky.03.L",
    
    # === FINGERS (RIGHT) ===
    "right_thumb_tip": "thumb.03.R",
    "right_index_tip": "f_index.03.R",
    "right_middle_tip": "f_middle.03.R",
    "right_ring_tip": "f_ring.03.R",
    "right_pinky_tip": "f_pinky.03.R",
}

# FK alternative mapping (if IK causes issues)
MEDIAPIPE_TO_RIGIFY_FK_MAPPING = {
    # Left arm FK
    "left_shoulder": "upper_arm_fk.L",
    "left_elbow": "forearm_fk.L",
    "left_wrist": "hand_fk.L",
    
    # Right arm FK
    "right_shoulder": "upper_arm_fk.R",
    "right_elbow": "forearm_fk.R",
    "right_wrist": "hand_fk.R",
    
    # Left leg FK
    "left_hip": "thigh_fk.L",
    "left_knee": "shin_fk.L",
    "left_ankle": "foot_fk.L",
    
    # Right leg FK
    "right_hip": "thigh_fk.R",
    "right_knee": "shin_fk.R",
    "right_ankle": "foot_fk.R",
}

# ============================================================================
# CAMERA CONFIGURATION
# ============================================================================

@dataclass
class CameraConfig:
    """Configuration for a single camera."""
    name: str
    device_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    backend: str = "auto"  # auto, dshow, msmf, v4l2
    
    # Intrinsic parameters (from calibration)
    camera_matrix: Optional[List[List[float]]] = None
    dist_coeffs: Optional[List[float]] = None
    
    # Extrinsic parameters (from stereo calibration)
    rotation: Optional[List[List[float]]] = None
    translation: Optional[List[float]] = None


@dataclass  
class StereoConfig:
    """Configuration for stereo camera setup."""
    camera_left: CameraConfig = field(default_factory=lambda: CameraConfig("left", 0))
    camera_right: CameraConfig = field(default_factory=lambda: CameraConfig("right", 1))
    
    # Stereo baseline and angle
    baseline_meters: float = 0.5  # Distance between cameras
    angle_degrees: float = 50.0  # Angle between cameras (45-60 optimal)
    
    # Stereo calibration results
    R: Optional[List[List[float]]] = None  # Rotation between cameras
    T: Optional[List[float]] = None  # Translation between cameras
    E: Optional[List[List[float]]] = None  # Essential matrix
    F: Optional[List[List[float]]] = None  # Fundamental matrix
    
    # Rectification parameters
    R1: Optional[List[List[float]]] = None
    R2: Optional[List[List[float]]] = None
    P1: Optional[List[List[float]]] = None
    P2: Optional[List[List[float]]] = None
    Q: Optional[List[List[float]]] = None


# ============================================================================
# CHARUCO BOARD CONFIGURATION
# ============================================================================

@dataclass
class CharucoBoardConfig:
    """Configuration for ChArUco calibration board."""
    squares_x: int = 10
    squares_y: int = 5
    square_length_meters: float = 0.043  # 43mm squares (measured from printed 10x5 board)
    marker_length_meters: float = 0.031  # 31mm markers (72% of square)
    dictionary_id: int = 0  # DICT_4X4_50
    
    def get_board_size_meters(self) -> Tuple[float, float]:
        """Get physical board dimensions."""
        return (
            self.squares_x * self.square_length_meters,
            self.squares_y * self.square_length_meters
        )


# ============================================================================
# RECORDING CONFIGURATION
# ============================================================================

@dataclass
class RecordingConfig:
    """Configuration for motion capture recording."""
    output_dir: str = "./mocap_data"
    
    # Recording settings
    countdown_seconds: int = 3
    max_duration_seconds: int = 300  # 5 minutes max
    
    # MediaPipe settings
    model_complexity: int = 2  # 0=lite, 1=full, 2=heavy
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.7
    enable_segmentation: bool = False
    
    # Hand tracking
    enable_hands: bool = True
    max_num_hands: int = 2
    
    # Output format
    save_video: bool = True
    save_landmarks: bool = True
    video_codec: str = "mp4v"
    
    # Smoothing
    apply_smoothing: bool = True
    smoothing_window: int = 5


# ============================================================================
# BLENDER EXPORT CONFIGURATION  
# ============================================================================

@dataclass
class BlenderExportConfig:
    """Configuration for Blender export."""
    rig_name: str = "JaxRigify"
    mesh_name: str = "JaxBody5"
    
    # Coordinate system conversion
    # MediaPipe: Y-up, Z-forward
    # Blender: Z-up, -Y-forward
    apply_coordinate_transform: bool = True
    
    # Scale factor (MediaPipe normalized coords to Blender units)
    scale_factor: float = 1.0
    
    # Character rest pose
    rest_pose: str = "A_POSE"  # A_POSE or T_POSE
    
    # Animation settings
    frame_rate: int = 30
    start_frame: int = 1
    
    # Retargeting method
    use_ik: bool = True  # Use IK controls (recommended)
    use_constraints: bool = True  # Use bone constraints
    
    # Bone mapping
    bone_mapping: Dict[str, str] = field(
        default_factory=lambda: MEDIAPIPE_TO_RIGIFY_MAPPING.copy()
    )


# ============================================================================
# GLOBAL CONFIG
# ============================================================================

@dataclass
class MocapConfig:
    """Master configuration for the entire mocap system."""
    # Sub-configs
    stereo: StereoConfig = field(default_factory=StereoConfig)
    charuco: CharucoBoardConfig = field(default_factory=CharucoBoardConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    blender: BlenderExportConfig = field(default_factory=BlenderExportConfig)
    
    # Paths
    project_root: str = field(default_factory=lambda: os.path.dirname(__file__))
    calibration_dir: str = "calibration_data"
    takes_dir: str = "takes"
    
    def save(self, filepath: str):
        """Save configuration to JSON."""
        import dataclasses
        
        def serialize(obj):
            if dataclasses.is_dataclass(obj):
                return {k: serialize(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, list):
                return [serialize(v) for v in obj]
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            return obj
        
        with open(filepath, 'w') as f:
            json.dump(serialize(self), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'MocapConfig':
        """Load configuration from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct nested dataclasses
        stereo_data = data.get('stereo', {})
        stereo_data['camera_left'] = CameraConfig(**stereo_data.get('camera_left', {}))
        stereo_data['camera_right'] = CameraConfig(**stereo_data.get('camera_right', {}))
        
        return cls(
            stereo=StereoConfig(**stereo_data),
            charuco=CharucoBoardConfig(**data.get('charuco', {})),
            recording=RecordingConfig(**data.get('recording', {})),
            blender=BlenderExportConfig(**data.get('blender', {})),
            project_root=data.get('project_root', os.path.dirname(__file__)),
            calibration_dir=data.get('calibration_dir', 'calibration_data'),
            takes_dir=data.get('takes_dir', 'takes'),
        )


# Create default global config
DEFAULT_CONFIG = MocapConfig()


if __name__ == "__main__":
    # Test config serialization
    config = MocapConfig()
    print("Mocap Configuration:")
    print(f"  Stereo angle: {config.stereo.angle_degrees} degrees")
    print(f"  Target rig: {config.blender.rig_name}")
    print(f"  Use IK: {config.blender.use_ik}")
