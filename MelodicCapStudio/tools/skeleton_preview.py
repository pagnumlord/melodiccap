"""
MelodicCap Skeleton Preview
============================
Standalone matplotlib script for quick 3D skeleton visualization.
Works outside of Blender for fast verification of take data.

Usage:
    python skeleton_preview.py <take_file.json> [frame_number]
    python skeleton_preview.py takes/take_20260221_010702.json
    python skeleton_preview.py takes/take_20260221_010702_clean.json 50

Features:
- 3D skeleton wireframe
- Coordinate axes for orientation verification
- Frame slider to scrub through takes
- Prints coordinate ranges and person height
"""

import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('TkAgg')
except Exception:
    pass

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

# MediaPipe body connections for skeleton wireframe
SKELETON_CONNECTIONS = [
    # Torso
    (11, 12),  # shoulders
    (11, 23),  # left shoulder -> left hip
    (12, 24),  # right shoulder -> right hip
    (23, 24),  # hips
    # Left arm
    (11, 13),  # left shoulder -> left elbow
    (13, 15),  # left elbow -> left wrist
    # Right arm
    (12, 14),  # right shoulder -> right elbow
    (14, 16),  # right elbow -> right wrist
    # Left leg
    (23, 25),  # left hip -> left knee
    (25, 27),  # left knee -> left ankle
    # Right leg
    (24, 26),  # right hip -> right knee
    (26, 28),  # right knee -> right ankle
    # Head
    (0, 11),   # nose -> left shoulder (approximate)
    (0, 12),   # nose -> right shoulder (approximate)
]

LANDMARK_NAMES = {
    0: "nose", 11: "L_shoulder", 12: "R_shoulder",
    13: "L_elbow", 14: "R_elbow", 15: "L_wrist", 16: "R_wrist",
    23: "L_hip", 24: "R_hip", 25: "L_knee", 26: "R_knee",
    27: "L_ankle", 28: "R_ankle",
}


def get_lm(landmarks, idx):
    """Get landmark position [x, y, z] from landmarks dict."""
    for key in [str(idx), idx]:
        if key in landmarks:
            return landmarks[key]
    return None


def draw_skeleton(ax, landmarks, frame_idx):
    """Draw 3D skeleton wireframe for a single frame."""
    ax.cla()

    # Draw connections
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        p1 = get_lm(landmarks, start_idx)
        p2 = get_lm(landmarks, end_idx)
        if p1 and p2:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    'b-', linewidth=2, alpha=0.7)

    # Draw joints
    xs, ys, zs = [], [], []
    for key, pos in landmarks.items():
        idx = int(key) if isinstance(key, str) else key
        xs.append(pos[0])
        ys.append(pos[1])
        zs.append(pos[2])

    ax.scatter(xs, ys, zs, c='red', s=30, alpha=0.8)

    # Label key landmarks
    for idx, name in LANDMARK_NAMES.items():
        pos = get_lm(landmarks, idx)
        if pos:
            ax.text(pos[0], pos[1], pos[2], f" {name}", fontsize=6, alpha=0.6)

    # Draw floor plane (Z=0)
    ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], [0, 0, 0, 0, 0],
            'g--', alpha=0.3, linewidth=1)

    # Draw coordinate axes at origin
    axis_len = 0.3
    ax.quiver(0, 0, 0, axis_len, 0, 0, color='r', arrow_length_ratio=0.2, label='X (right)')
    ax.quiver(0, 0, 0, 0, axis_len, 0, color='g', arrow_length_ratio=0.2, label='Y (forward)')
    ax.quiver(0, 0, 0, 0, 0, axis_len, color='b', arrow_length_ratio=0.2, label='Z (up)')

    # Set reasonable axis limits based on data
    if xs:
        cx, cy, cz = sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)
        r = 1.5  # 1.5m radius from center
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
        ax.set_zlim(0, cz + r)

    ax.set_xlabel('X (right)')
    ax.set_ylabel('Y (forward)')
    ax.set_zlabel('Z (up)')
    ax.set_title(f'Frame {frame_idx} ({len(landmarks)} landmarks)')


def preview_take(filepath, start_frame=0):
    """Interactive 3D skeleton preview with frame slider."""
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"ERROR: File not found: {filepath}")
        return

    with open(filepath) as f:
        data = json.load(f)

    frames = data.get('frames', [])
    if not frames:
        print("ERROR: No frames in take!")
        return

    # Print take info
    print("=" * 60)
    print(f"SKELETON PREVIEW: {filepath.name}")
    print("=" * 60)
    print(f"  Frames: {len(frames)}")
    print(f"  Duration: {data.get('duration_seconds', 0):.1f}s")

    # Coordinate ranges
    all_x, all_y, all_z = [], [], []
    for fdata in frames:
        for pos in fdata.get('landmarks', {}).values():
            all_x.append(pos[0])
            all_y.append(pos[1])
            all_z.append(pos[2])

    if all_x:
        print(f"\n  Coordinate ranges:")
        print(f"    X: [{min(all_x):.2f}, {max(all_x):.2f}]")
        print(f"    Y: [{min(all_y):.2f}, {max(all_y):.2f}]")
        print(f"    Z: [{min(all_z):.2f}, {max(all_z):.2f}]")

    # Person height from frame 0
    ref_lms = frames[0].get('landmarks', {})
    nose = get_lm(ref_lms, 0)
    ankle_l = get_lm(ref_lms, 27)
    ankle_r = get_lm(ref_lms, 28)

    if nose and ankle_l and ankle_r:
        ankle_mid_z = (ankle_l[2] + ankle_r[2]) / 2
        person_height = (nose[2] - ankle_mid_z) + 0.15
        print(f"  Person height: {person_height:.3f}m")

    # Hip center frame 0
    hip_l = get_lm(ref_lms, 23)
    hip_r = get_lm(ref_lms, 24)
    if hip_l and hip_r:
        hip = [(hip_l[i] + hip_r[i]) / 2 for i in range(3)]
        print(f"  Hip center (frame 0): ({hip[0]:.3f}, {hip[1]:.3f}, {hip[2]:.3f})")

    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.15)

    # Initial draw
    current_frame = min(start_frame, len(frames) - 1)
    draw_skeleton(ax, frames[current_frame].get('landmarks', {}), current_frame)

    # Frame slider
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, len(frames) - 1,
                    valinit=current_frame, valstep=1)

    def update(val):
        frame_idx = int(slider.val)
        lms = frames[frame_idx].get('landmarks', {})
        draw_skeleton(ax, lms, frame_idx)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    print(f"\n  Use the slider to scrub through frames.")
    print(f"  Close the window to exit.")
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python skeleton_preview.py <take_file.json> [frame_number]")
        print("Example: python skeleton_preview.py takes/take_20260221_010702.json")
        sys.exit(1)

    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    preview_take(sys.argv[1], start)
