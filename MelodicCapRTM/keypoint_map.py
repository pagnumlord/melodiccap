"""
MelodicCap Keypoint Mapping Layer
==================================
Defines a stable internal landmark index scheme and provides
mapping functions between different pose estimation backends.

Supported formats:
- COCO-WholeBody (133 keypoints) — used by RTMW/DWPose via rtmlib
- COCO-17 (17 body keypoints) — used by RTMPose body-only
- MediaPipe (33 landmarks) — legacy format from MelodicCapFresh

The internal format (LM) matches COCO-WholeBody indices directly,
since that's what RTMW outputs. Old MediaPipe takes get converted
on import in the Blender addon.
"""


class LM:
    """
    MelodicCap internal landmark indices.
    Matches COCO-WholeBody format (0-132).

    RTMW output maps 1:1 to these indices — no conversion needed.
    """

    # === Body (0-16) ===
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    # === Feet (17-22) ===
    LEFT_BIG_TOE = 17
    LEFT_SMALL_TOE = 18
    LEFT_HEEL = 19
    RIGHT_BIG_TOE = 20
    RIGHT_SMALL_TOE = 21
    RIGHT_HEEL = 22

    # === Face (23-90) ===
    # 68 face landmarks following iBUG/300W convention
    FACE_START = 23
    FACE_END = 90

    # === Left Hand (91-111) ===
    # 21 keypoints: wrist + 4 per finger (thumb, index, middle, ring, pinky)
    LEFT_HAND_WRIST = 91
    LEFT_THUMB_CMC = 92
    LEFT_THUMB_MCP = 93
    LEFT_THUMB_IP = 94
    LEFT_THUMB_TIP = 95
    LEFT_INDEX_MCP = 96
    LEFT_INDEX_PIP = 97
    LEFT_INDEX_DIP = 98
    LEFT_INDEX_TIP = 99
    LEFT_MIDDLE_MCP = 100
    LEFT_MIDDLE_PIP = 101
    LEFT_MIDDLE_DIP = 102
    LEFT_MIDDLE_TIP = 103
    LEFT_RING_MCP = 104
    LEFT_RING_PIP = 105
    LEFT_RING_DIP = 106
    LEFT_RING_TIP = 107
    LEFT_PINKY_MCP = 108
    LEFT_PINKY_PIP = 109
    LEFT_PINKY_DIP = 110
    LEFT_PINKY_TIP = 111

    # === Right Hand (112-132) ===
    RIGHT_HAND_WRIST = 112
    RIGHT_THUMB_CMC = 113
    RIGHT_THUMB_MCP = 114
    RIGHT_THUMB_IP = 115
    RIGHT_THUMB_TIP = 116
    RIGHT_INDEX_MCP = 117
    RIGHT_INDEX_PIP = 118
    RIGHT_INDEX_DIP = 119
    RIGHT_INDEX_TIP = 120
    RIGHT_MIDDLE_MCP = 121
    RIGHT_MIDDLE_PIP = 122
    RIGHT_MIDDLE_DIP = 123
    RIGHT_MIDDLE_TIP = 124
    RIGHT_RING_MCP = 125
    RIGHT_RING_PIP = 126
    RIGHT_RING_DIP = 127
    RIGHT_RING_TIP = 128
    RIGHT_PINKY_MCP = 129
    RIGHT_PINKY_PIP = 130
    RIGHT_PINKY_DIP = 131
    RIGHT_PINKY_TIP = 132

    # Total keypoints per format
    BODY_COUNT = 17       # indices 0-16
    FOOT_COUNT = 6        # indices 17-22
    FACE_COUNT = 68       # indices 23-90
    LEFT_HAND_COUNT = 21  # indices 91-111
    RIGHT_HAND_COUNT = 21 # indices 112-132
    WHOLEBODY_COUNT = 133 # total


# Skeleton connections for drawing (pairs of LM indices)
BODY_SKELETON = [
    # Torso
    (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
    (LM.LEFT_HIP, LM.RIGHT_HIP),
    (LM.LEFT_SHOULDER, LM.LEFT_HIP),
    (LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
    # Left arm
    (LM.LEFT_SHOULDER, LM.LEFT_ELBOW),
    (LM.LEFT_ELBOW, LM.LEFT_WRIST),
    # Right arm
    (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW),
    (LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
    # Left leg
    (LM.LEFT_HIP, LM.LEFT_KNEE),
    (LM.LEFT_KNEE, LM.LEFT_ANKLE),
    # Right leg
    (LM.RIGHT_HIP, LM.RIGHT_KNEE),
    (LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
    # Head
    (LM.NOSE, LM.LEFT_EYE),
    (LM.NOSE, LM.RIGHT_EYE),
    (LM.LEFT_EYE, LM.LEFT_EAR),
    (LM.RIGHT_EYE, LM.RIGHT_EAR),
    # Feet
    (LM.LEFT_ANKLE, LM.LEFT_HEEL),
    (LM.LEFT_ANKLE, LM.LEFT_BIG_TOE),
    (LM.LEFT_HEEL, LM.LEFT_BIG_TOE),
    (LM.RIGHT_ANKLE, LM.RIGHT_HEEL),
    (LM.RIGHT_ANKLE, LM.RIGHT_BIG_TOE),
    (LM.RIGHT_HEEL, LM.RIGHT_BIG_TOE),
]

# Hand skeleton connections (per-hand, offset from hand wrist index)
HAND_SKELETON_OFFSETS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
]


# =============================================================================
# MEDIAPIPE -> LM CONVERSION (for importing old takes)
# =============================================================================

# MediaPipe pose landmark indices -> LM indices
# Only maps landmarks that have a direct equivalent
MEDIAPIPE_TO_LM = {
    0: LM.NOSE,
    # MediaPipe has 6 eye points (inner/center/outer per side)
    # We map the center eye points
    2: LM.LEFT_EYE,
    5: LM.RIGHT_EYE,
    7: LM.LEFT_EAR,
    8: LM.RIGHT_EAR,
    # MediaPipe 9,10 = mouth corners — no LM equivalent in body keypoints
    11: LM.LEFT_SHOULDER,
    12: LM.RIGHT_SHOULDER,
    13: LM.LEFT_ELBOW,
    14: LM.RIGHT_ELBOW,
    15: LM.LEFT_WRIST,
    16: LM.RIGHT_WRIST,
    # MediaPipe 17-22 = pinky/index/thumb tips — approximate hand points
    # Map them to the closest hand keypoint
    17: LM.LEFT_PINKY_TIP,
    18: LM.RIGHT_PINKY_TIP,
    19: LM.LEFT_INDEX_TIP,
    20: LM.RIGHT_INDEX_TIP,
    21: LM.LEFT_THUMB_TIP,
    22: LM.RIGHT_THUMB_TIP,
    23: LM.LEFT_HIP,
    24: LM.RIGHT_HIP,
    25: LM.LEFT_KNEE,
    26: LM.RIGHT_KNEE,
    27: LM.LEFT_ANKLE,
    28: LM.RIGHT_ANKLE,
    29: LM.LEFT_HEEL,
    30: LM.RIGHT_HEEL,
    31: LM.LEFT_BIG_TOE,
    32: LM.RIGHT_BIG_TOE,
}

# Reverse mapping: LM -> MediaPipe (for backward compatibility)
LM_TO_MEDIAPIPE = {v: k for k, v in MEDIAPIPE_TO_LM.items()}


def convert_mediapipe_take(landmarks_3d_mp):
    """
    Convert a single frame's landmarks from MediaPipe indices to LM indices.

    Args:
        landmarks_3d_mp: dict with string MediaPipe indices as keys,
                         e.g. {"11": [x, y, z], "12": [x, y, z], ...}

    Returns:
        dict with string LM indices as keys,
        e.g. {"5": [x, y, z], "6": [x, y, z], ...}
    """
    converted = {}
    for mp_key, coords in landmarks_3d_mp.items():
        mp_idx = int(mp_key)
        if mp_idx in MEDIAPIPE_TO_LM:
            lm_idx = MEDIAPIPE_TO_LM[mp_idx]
            converted[str(lm_idx)] = coords
    return converted


# Indices that are most important for body mocap
# (used for visibility filtering — skip triangulation if these are missing)
CRITICAL_BODY_INDICES = [
    LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
    LM.LEFT_HIP, LM.RIGHT_HIP,
    LM.LEFT_ELBOW, LM.RIGHT_ELBOW,
    LM.LEFT_WRIST, LM.RIGHT_WRIST,
    LM.LEFT_KNEE, LM.RIGHT_KNEE,
    LM.LEFT_ANKLE, LM.RIGHT_ANKLE,
]
