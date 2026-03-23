#!/usr/bin/env python3
"""
Camera Detection Script for Melodic Justice MoCap
Finds and identifies connected cameras for dual-camera setup.
"""

import cv2
import sys

def detect_cameras(max_cameras=10):
    """Detect all available cameras and show preview."""
    print("=" * 60)
    print("MELODIC JUSTICE MOCAP - CAMERA DETECTION")
    print("=" * 60)
    print("\nSearching for cameras...\n")
    
    available_cameras = []
    
    # Try different backends on Windows
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
    ]
    
    for idx in range(max_cameras):
        for backend, backend_name in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Try to get a frame
                ret, frame = cap.read()
                if ret:
                    available_cameras.append({
                        'id': idx,
                        'backend': backend,
                        'backend_name': backend_name,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'frame': frame
                    })
                    print(f"✓ Camera {idx} ({backend_name}): {width}x{height} @ {fps:.0f}fps")
                cap.release()
                break  # Found camera on this backend, move to next index
            cap.release()
    
    if not available_cameras:
        print("\n❌ No cameras detected!")
        print("\nTroubleshooting:")
        print("1. Make sure DroidCam is running on your S25 phone")
        print("2. Connect DroidCam Client on PC to the phone")
        print("3. Connect Sony ZV-1F via USB and enable webcam mode")
        print("4. Try unplugging and replugging cameras")
        return []
    
    print(f"\n{'=' * 60}")
    print(f"Found {len(available_cameras)} camera(s)")
    print(f"{'=' * 60}")
    
    return available_cameras


def preview_cameras(cameras):
    """Show preview windows for all detected cameras."""
    if not cameras:
        return
    
    print("\nOpening preview windows...")
    print("Press 'Q' to close all previews")
    print("Press '1', '2', etc. to select LEFT camera")
    print("Press 'L' to mark current window as LEFT")
    print("Press 'R' to mark current window as RIGHT")
    print()
    
    # Open capture objects
    caps = []
    for cam in cameras:
        cap = cv2.VideoCapture(cam['id'], cam['backend'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        caps.append(cap)
    
    left_cam = None
    right_cam = None
    
    while True:
        for i, (cap, cam) in enumerate(zip(caps, cameras)):
            ret, frame = cap.read()
            if ret:
                # Add label
                label = f"Camera {cam['id']} ({cam['backend_name']})"
                if cam['id'] == left_cam:
                    label += " [LEFT]"
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 10)
                elif cam['id'] == right_cam:
                    label += " [RIGHT]"
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
                
                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow(f"Camera {cam['id']}", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('1') and len(cameras) >= 1:
            left_cam = cameras[0]['id']
            print(f"LEFT camera set to {left_cam}")
        elif key == ord('2') and len(cameras) >= 2:
            left_cam = cameras[1]['id']
            print(f"LEFT camera set to {left_cam}")
        elif key == ord('3') and len(cameras) >= 3:
            right_cam = cameras[0]['id']
            print(f"RIGHT camera set to {right_cam}")
        elif key == ord('4') and len(cameras) >= 2:
            right_cam = cameras[1]['id']
            print(f"RIGHT camera set to {right_cam}")
    
    # Cleanup
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
    
    return left_cam, right_cam


def main():
    print("\n" + "=" * 60)
    print("  STEP 1: CAMERA DETECTION")
    print("=" * 60)
    print("\nMake sure both cameras are connected:")
    print("  • S25 phone: DroidCam app running, connected to PC client")
    print("  • Sony ZV-1F: USB connected, set to webcam/streaming mode")
    print()
    
    cameras = detect_cameras()
    
    if len(cameras) < 2:
        print("\n⚠️  Need at least 2 cameras for stereo capture!")
        print("Currently detected:", len(cameras))
        if len(cameras) == 1:
            print("\nYou can still test with one camera.")
        return
    
    print("\n" + "=" * 60)
    print("  STEP 2: CAMERA IDENTIFICATION")
    print("=" * 60)
    
    response = input("\nOpen preview windows to identify cameras? (y/n): ")
    if response.lower() == 'y':
        left, right = preview_cameras(cameras)
        
        if left is not None and right is not None:
            print(f"\n✓ Camera configuration:")
            print(f"  LEFT camera:  {left}")
            print(f"  RIGHT camera: {right}")
            print(f"\nTo calibrate, run:")
            print(f"  python main.py --cli calibrate --left {left} --right {right} --frames 20")
    else:
        print("\nDetected cameras:")
        for i, cam in enumerate(cameras):
            print(f"  {i}. Camera ID {cam['id']} - {cam['width']}x{cam['height']}")
        
        print("\nTypical setup:")
        print("  --left 0 --right 1")
        print("\nIf DroidCam appears as a different ID, adjust accordingly.")


if __name__ == "__main__":
    main()
