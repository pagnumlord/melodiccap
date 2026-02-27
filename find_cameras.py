"""
Advanced Camera Detection for Windows
Finds all cameras including virtual cameras (OBS, DroidCam, etc.)

This script tries multiple OpenCV backends to find all cameras
"""

import cv2
import sys

def test_camera_with_backend(camera_id, backend_name, backend_code):
    """Test a specific camera with a specific backend"""
    try:
        cap = cv2.VideoCapture(camera_id, backend_code)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                height, width = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                return {
                    'id': camera_id,
                    'backend': backend_name,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'working': True
                }
        cap.release()
    except Exception as e:
        pass
    return None


def find_all_cameras():
    """Scan for cameras using multiple backends"""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE CAMERA DETECTION")
    print("="*70 + "\n")
    
    # Different backends to try (Windows-specific)
    backends = [
        ("DirectShow (DSHOW)", cv2.CAP_DSHOW),      # Most compatible on Windows
        ("Media Foundation (MSMF)", cv2.CAP_MSMF),   # Windows 10+ default
        ("Auto-detect (ANY)", cv2.CAP_ANY),          # Let OpenCV choose
    ]
    
    all_cameras = {}
    
    # Try each backend
    for backend_name, backend_code in backends:
        print(f"\n🔍 Scanning with {backend_name}...")
        print("-" * 70)
        
        found_with_backend = []
        
        # Try camera IDs 0-10 (covers most systems)
        for cam_id in range(11):
            result = test_camera_with_backend(cam_id, backend_name, backend_code)
            if result:
                found_with_backend.append(result)
                
                # Add to master list
                key = (cam_id, backend_name)
                all_cameras[key] = result
                
                print(f"  ✓ Camera {cam_id}: {result['width']}x{result['height']} @ {result['fps']} FPS")
        
        if not found_with_backend:
            print("  ✗ No cameras found with this backend")
    
    return all_cameras


def show_detailed_camera_info(camera_id, backend_code):
    """Show detailed properties of a specific camera"""
    cap = cv2.VideoCapture(camera_id, backend_code)
    
    if not cap.isOpened():
        print(f"  ✗ Could not open camera {camera_id}")
        return
    
    print(f"\n📹 Detailed info for Camera {camera_id}:")
    print("-" * 70)
    
    properties = {
        'Width': cv2.CAP_PROP_FRAME_WIDTH,
        'Height': cv2.CAP_PROP_FRAME_HEIGHT,
        'FPS': cv2.CAP_PROP_FPS,
        'Brightness': cv2.CAP_PROP_BRIGHTNESS,
        'Contrast': cv2.CAP_PROP_CONTRAST,
        'Saturation': cv2.CAP_PROP_SATURATION,
        'Backend': cv2.CAP_PROP_BACKEND,
    }
    
    for name, prop in properties.items():
        value = cap.get(prop)
        print(f"  {name}: {value}")
    
    cap.release()


def preview_camera(camera_id, backend_code, backend_name):
    """Show live preview of a specific camera"""
    print(f"\n🎥 Opening Camera {camera_id} with {backend_name}...")
    print("Press 'Q' to close preview\n")
    
    cap = cv2.VideoCapture(camera_id, backend_code)
    
    if not cap.isOpened():
        print(f"✗ Failed to open Camera {camera_id}")
        return False
    
    # Try to set higher resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print(f"✗ Failed to read frame from Camera {camera_id}")
            break
        
        frame_count += 1
        
        # Add info overlay
        text = f"Camera {camera_id} - {backend_name} - Frame {frame_count}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        
        cv2.imshow(f'Camera {camera_id} Preview', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"✓ Camera {camera_id} preview closed\n")
    return True


def main():
    """Main camera detection workflow"""
    
    # Step 1: Find all cameras
    cameras = find_all_cameras()
    
    if not cameras:
        print("\n❌ NO CAMERAS FOUND!")
        print("\nTroubleshooting:")
        print("1. Make sure cameras are connected")
        print("2. For OBS Virtual Camera: Tools → Start Virtual Camera")
        print("3. For DroidCam: Make sure app is running on phone AND PC")
        print("4. Check Windows Privacy Settings → Camera → Allow apps to access camera")
        print("5. Try restarting OBS or DroidCam")
        return
    
    # Step 2: Show summary
    print("\n" + "="*70)
    print("CAMERA SUMMARY")
    print("="*70 + "\n")
    
    # Group by camera ID
    by_id = {}
    for (cam_id, backend), info in cameras.items():
        if cam_id not in by_id:
            by_id[cam_id] = []
        by_id[cam_id].append((backend, info))
    
    for cam_id in sorted(by_id.keys()):
        print(f"\n📹 Camera {cam_id}:")
        for backend, info in by_id[cam_id]:
            print(f"   • {backend}: {info['width']}x{info['height']} @ {info['fps']} FPS")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70 + "\n")
    
    # Find best working cameras
    working_ids = sorted(by_id.keys())
    
    if len(working_ids) >= 2:
        print(f"✓ Found {len(working_ids)} cameras!")
        print(f"\nFor mocap recording, I recommend using:")
        print(f"  • Camera {working_ids[-2]} (higher quality)")
        print(f"  • Camera {working_ids[-1]} (higher quality)")
        
        # Determine best backend
        best_backend = cv2.CAP_DSHOW  # Most compatible on Windows
        best_backend_name = "DirectShow (DSHOW)"
        
        print(f"\nUse this command:")
        print(f"  python mocap_recorder.py --camera_0 {working_ids[-2]} --camera_1 {working_ids[-1]}")
        
        # Step 3: Offer to preview
        print("\n" + "="*70)
        response = input("\nWould you like to preview cameras? (y/n): ")
        
        if response.lower() in ['y', 'yes']:
            print("\n📹 Camera Preview Mode")
            print("-" * 70)
            
            for cam_id in working_ids:
                response = input(f"\nPreview Camera {cam_id}? (y/n/q to quit): ")
                
                if response.lower() == 'q':
                    break
                elif response.lower() in ['y', 'yes']:
                    # Use DirectShow backend (most compatible)
                    preview_camera(cam_id, cv2.CAP_DSHOW, "DirectShow")
        
    else:
        print(f"⚠ Only found {len(working_ids)} camera(s)")
        print("You need at least 2 cameras for dual-camera mocap")
    
    print("\n" + "="*70)
    print("CAMERA DETECTION COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
