#!/usr/bin/env python3
"""
MelodicCap JSON Diagnostic
Analyzes coordinate ranges to help fix import issues
"""

import json
import sys
from pathlib import Path

def analyze(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print("=" * 70)
    print("MELODICCAP JSON DIAGNOSTIC")
    print("=" * 70)
    
    # Metadata
    print("\n📋 METADATA")
    for key, value in data.get('metadata', {}).items():
        print(f"   {key}: {value}")
    
    # Find valid frames
    valid_frames = []
    for frame in data['data']:
        pose = frame.get('pose_landmarks')
        if pose and len(pose) >= 33:
            valid_frames.append(frame)
    
    print(f"\n📊 FRAME STATISTICS")
    print(f"   Total frames: {len(data['data'])}")
    print(f"   Valid frames (with pose): {len(valid_frames)}")
    
    if not valid_frames:
        print("\n❌ ERROR: No valid pose data found!")
        return
    
    # Collect all coordinates
    all_x, all_y, all_z = [], [], []
    
    for frame in valid_frames:
        for lm in frame['pose_landmarks']:
            all_x.append(lm['x'])
            all_y.append(lm['y'])
            all_z.append(lm['z'])
    
    print(f"\n📏 COORDINATE RANGES (across all frames)")
    print(f"   X: {min(all_x):.4f} to {max(all_x):.4f} (range: {max(all_x)-min(all_x):.4f})")
    print(f"   Y: {min(all_y):.4f} to {max(all_y):.4f} (range: {max(all_y)-min(all_y):.4f})")
    print(f"   Z: {min(all_z):.4f} to {max(all_z):.4f} (range: {max(all_z)-min(all_z):.4f})")
    
    # Analyze first valid frame (T-pose reference)
    first_frame = valid_frames[0]
    landmarks = {lm['id']: lm for lm in first_frame['pose_landmarks']}
    
    print(f"\n🦴 FIRST VALID FRAME (Frame {first_frame['frame']}) - Key Landmarks")
    key_ids = {
        0: "Nose", 
        11: "L_Shoulder", 12: "R_Shoulder",
        13: "L_Elbow", 14: "R_Elbow",
        15: "L_Wrist", 16: "R_Wrist",
        23: "L_Hip", 24: "R_Hip",
        25: "L_Knee", 26: "R_Knee",
        27: "L_Ankle", 28: "R_Ankle"
    }
    
    for id, name in key_ids.items():
        if id in landmarks:
            lm = landmarks[id]
            print(f"   {name:12} ({id:2}): X={lm['x']:7.4f}  Y={lm['y']:7.4f}  Z={lm['z']:7.4f}")
    
    # Calculate body measurements from T-pose
    if 23 in landmarks and 24 in landmarks:
        l_hip = landmarks[23]
        r_hip = landmarks[24]
        hip_center_x = (l_hip['x'] + r_hip['x']) / 2
        hip_center_y = (l_hip['y'] + r_hip['y']) / 2
        hip_spread = abs(r_hip['x'] - l_hip['x'])
        print(f"\n📐 CALCULATED BODY METRICS (Frame {first_frame['frame']})")
        print(f"   Hip center X: {hip_center_x:.4f}")
        print(f"   Hip center Y: {hip_center_y:.4f}")
        print(f"   Hip spread:   {hip_spread:.4f}")
        
        if 11 in landmarks and 12 in landmarks:
            shoulder_spread = abs(landmarks[12]['x'] - landmarks[11]['x'])
            print(f"   Shoulder spread: {shoulder_spread:.4f}")
        
        if 15 in landmarks and 16 in landmarks:
            arm_span = abs(landmarks[16]['x'] - landmarks[15]['x'])
            print(f"   Arm span (wrist-wrist): {arm_span:.4f}")
        
        if 0 in landmarks and 27 in landmarks:
            body_height_y = abs(landmarks[0]['y'] - landmarks[27]['y'])
            print(f"   Body height (Y): {body_height_y:.4f}")
    
    # Detect coordinate system type
    print(f"\n🔍 COORDINATE SYSTEM ANALYSIS")
    
    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)
    z_range = max(all_z) - min(all_z)
    
    if max(all_x) <= 1.5 and min(all_x) >= -0.5:
        print("   X: Appears NORMALIZED (0-1 typical MediaPipe)")
    elif x_range > 100:
        print("   X: Appears to be PIXEL coordinates")
    else:
        print(f"   X: CUSTOM format (range {min(all_x):.2f} to {max(all_x):.2f})")
    
    if max(all_y) <= 1.5 and min(all_y) >= -0.5:
        print("   Y: Appears NORMALIZED (0-1 typical MediaPipe)")
    elif y_range > 100:
        print("   Y: Appears to be PIXEL coordinates")
    else:
        print(f"   Y: CUSTOM format (range {min(all_y):.2f} to {max(all_y):.2f})")
    
    if z_range < 2:
        print("   Z: Appears to be DEPTH estimate (normalized)")
    elif z_range > 10:
        print("   Z: Appears to be WORLD SPACE depth")
    else:
        print(f"   Z: CUSTOM format (range {min(all_z):.2f} to {max(all_z):.2f})")
    
    # Recommendations
    print(f"\n💡 RECOMMENDED IMPORT SETTINGS")
    
    if x_range < 2:  # Normalized
        scale_x = 2.0 / x_range if x_range > 0 else 2.0
        print(f"   base_scale_x ≈ {scale_x:.2f}")
    
    if y_range < 2:  # Normalized  
        scale_z = 2.0 / y_range if y_range > 0 else 2.0
        print(f"   base_scale_z ≈ {scale_z:.2f}")
    
    if z_range < 2:  # Normalized depth
        scale_y = 1.0  # For dual camera, use full depth
        print(f"   base_scale_y ≈ {scale_y:.2f}")
    
    print("\n" + "=" * 70)
    print("Run this output when asking Claude for help fixing the import!")
    print("=" * 70)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Try to find a dual JSON in common locations
        paths = [
            "data/takes/dual_take_name_20260121_201847.json",
            "data/takes"
        ]
        print("Usage: python diagnose_mocap_json.py <path_to_json>")
        print("\nExample: python diagnose_mocap_json.py data/takes/dual_take_name_20260121_201847.json")
        sys.exit(1)
    
    analyze(sys.argv[1])
