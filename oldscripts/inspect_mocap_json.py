"""
Mocap JSON Inspector
Checks what data is actually in your mocap JSON files

Usage:
    python inspect_mocap_json.py take_20260208_154156.json
"""

import json
import sys
from pathlib import Path

def inspect_mocap_json(filepath):
    """Inspect the structure of a mocap JSON file"""
    
    print("\n" + "="*70)
    print(f"INSPECTING: {filepath}")
    print("="*70 + "\n")
    
    # Load JSON
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load JSON: {e}")
        return
    
    # Check top-level structure
    print("Top-level keys:")
    for key in data.keys():
        print(f"  • {key}")
    
    print(f"\nTake info:")
    print(f"  Name: {data.get('take_name', 'N/A')}")
    print(f"  Timestamp: {data.get('timestamp', 'N/A')}")
    print(f"  Num frames: {data.get('num_frames', 'N/A')}")
    print(f"  FPS: {data.get('fps', 'N/A')}")
    
    # Check frames
    if 'frames' not in data:
        print("\nERROR: No 'frames' key in JSON!")
        return
    
    frames = data['frames']
    print(f"\nTotal frames in file: {len(frames)}")
    
    if not frames:
        print("ERROR: Frames array is empty!")
        return
    
    # Inspect first frame
    print("\n" + "-"*70)
    print("FIRST FRAME STRUCTURE:")
    print("-"*70)
    
    first_frame = frames[0]
    print(f"\nKeys in first frame:")
    for key in first_frame.keys():
        print(f"  • {key}")
    
    # Check landmarks_2d
    if 'landmarks_2d' in first_frame:
        landmarks_2d = first_frame['landmarks_2d']
        print(f"\nlandmarks_2d type: {type(landmarks_2d)}")
        
        if isinstance(landmarks_2d, dict):
            print(f"  Keys: {list(landmarks_2d.keys())}")
            for cam_name, lm_list in landmarks_2d.items():
                print(f"  {cam_name}: {len(lm_list)} landmarks")
                if lm_list:
                    print(f"    First landmark: {lm_list[0]}")
        else:
            print(f"  Length: {len(landmarks_2d)}")
            if landmarks_2d:
                print(f"  First item: {landmarks_2d[0]}")
    else:
        print("\nWARNING: No 'landmarks_2d' in first frame")
    
    # Check landmarks_3d
    if 'landmarks_3d' in first_frame:
        landmarks_3d = first_frame['landmarks_3d']
        print(f"\nlandmarks_3d type: {type(landmarks_3d)}")
        
        if isinstance(landmarks_3d, list):
            print(f"  Length: {len(landmarks_3d)}")
            if landmarks_3d:
                print(f"  First landmark: {landmarks_3d[0]}")
                print(f"  Sample landmarks:")
                for i in [0, 11, 12, 23, 24]:  # nose, shoulders, hips
                    if i < len(landmarks_3d):
                        lm = landmarks_3d[i]
                        print(f"    Landmark {i}: {lm}")
            else:
                print("  EMPTY! This is the problem!")
        elif isinstance(landmarks_3d, dict):
            print(f"  Keys: {list(landmarks_3d.keys())}")
            print(f"  Content: {landmarks_3d}")
        else:
            print(f"  Unexpected type!")
    else:
        print("\nERROR: No 'landmarks_3d' in first frame!")
    
    # Check a middle frame too
    if len(frames) > 10:
        print("\n" + "-"*70)
        print("MIDDLE FRAME (frame 10):")
        print("-"*70)
        
        mid_frame = frames[10]
        if 'landmarks_3d' in mid_frame:
            landmarks_3d = mid_frame['landmarks_3d']
            print(f"landmarks_3d length: {len(landmarks_3d)}")
            if landmarks_3d:
                print(f"Sample: {landmarks_3d[0] if len(landmarks_3d) > 0 else 'empty'}")
        else:
            print("No landmarks_3d!")
    
    print("\n" + "="*70)
    print("DIAGNOSIS:")
    print("="*70 + "\n")
    
    # Diagnose issues
    has_frames = 'frames' in data and len(data['frames']) > 0
    has_2d = 'landmarks_2d' in first_frame
    has_3d = 'landmarks_3d' in first_frame
    has_3d_data = has_3d and isinstance(first_frame['landmarks_3d'], list) and len(first_frame['landmarks_3d']) > 0
    
    if not has_frames:
        print("❌ No frames in file")
    else:
        print("✓ Frames exist")
    
    if not has_2d:
        print("❌ No 2D landmarks")
    else:
        print("✓ 2D landmarks exist")
    
    if not has_3d:
        print("❌ No 3D landmarks key")
    elif not has_3d_data:
        print("❌ 3D landmarks array is EMPTY - this is the problem!")
        print("\nPossible causes:")
        print("  1. Only one camera had tracking (need both cameras)")
        print("  2. Triangulation failed")
        print("  3. Recording stopped before 3D processing")
    else:
        print("✓ 3D landmarks populated correctly")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_mocap_json.py <path_to_json>")
        print("\nExample:")
        print("  python inspect_mocap_json.py mocap_takes/take_20260208_154156.json")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    if not Path(filepath).exists():
        # Try adding mocap_takes directory
        alt_path = Path("mocap_takes") / filepath
        if alt_path.exists():
            filepath = str(alt_path)
        else:
            print(f"ERROR: File not found: {filepath}")
            sys.exit(1)
    
    inspect_mocap_json(filepath)
