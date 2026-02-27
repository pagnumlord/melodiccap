#!/usr/bin/env python3
"""
Quick Start Test Script
Tests your camera setup and mocap system before recording

Usage:
    python quick_start_test.py
"""

import cv2
import sys

def test_cameras():
    """Test if cameras are accessible"""
    print("\n=== Testing Cameras ===\n")
    
    available_cameras = []
    
    # Test camera indices 0-4
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
                available_cameras.append({
                    'id': i,
                    'width': width,
                    'height': height,
                    'fps': fps
                })
                print(f"✓ Camera {i}: {width}x{height} @ {fps} FPS")
            cap.release()
    
    if len(available_cameras) < 2:
        print(f"\n⚠ Warning: Only {len(available_cameras)} camera(s) found")
        print("   You need at least 2 cameras for dual-camera mocap")
        print("   Check USB connections and permissions")
        return False
    else:
        print(f"\n✓ Found {len(available_cameras)} cameras - ready for dual-camera mocap!")
        return True

def test_mediapipe():
    """Test if MediaPipe is installed and working"""
    print("\n=== Testing MediaPipe ===\n")
    
    try:
        import mediapipe as mp
        print(f"✓ MediaPipe installed: version {mp.__version__}")
        
        # Test pose detection
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        print("✓ MediaPipe Pose initialized successfully")
        pose.close()
        
        return True
    except ImportError:
        print("✗ MediaPipe not installed!")
        print("  Install with: pip install mediapipe --break-system-packages")
        return False
    except Exception as e:
        print(f"✗ MediaPipe error: {e}")
        return False

def test_opencv():
    """Test if OpenCV is installed"""
    print("\n=== Testing OpenCV ===\n")
    
    try:
        import cv2
        print(f"✓ OpenCV installed: version {cv2.__version__}")
        return True
    except ImportError:
        print("✗ OpenCV not installed!")
        print("  Install with: pip install opencv-python --break-system-packages")
        return False

def test_system():
    """Run all system tests"""
    print("\n" + "="*50)
    print("Motion Capture System - Quick Start Test")
    print("="*50)
    
    results = {
        'opencv': test_opencv(),
        'mediapipe': test_mediapipe(),
        'cameras': test_cameras()
    }
    
    print("\n" + "="*50)
    print("Test Results Summary")
    print("="*50 + "\n")
    
    all_passed = all(results.values())
    
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test.upper():.<20} {status}")
    
    print()
    
    if all_passed:
        print("🎉 All tests passed! You're ready to record motion capture!")
        print("\nNext steps:")
        print("1. Position your cameras (45-90° apart)")
        print("2. Run: python mocap_recorder.py")
        print("3. Press SPACE to start/stop recording")
        print("\nSee WORKFLOW_GUIDE.md for detailed instructions")
        return True
    else:
        print("⚠ Some tests failed. Please fix the issues above.")
        print("\nQuick fixes:")
        if not results['opencv'] or not results['mediapipe']:
            print("- Install requirements: pip install -r requirements.txt --break-system-packages")
        if not results['cameras']:
            print("- Connect USB cameras")
            print("- Check camera permissions")
            print("- Try different USB ports")
        return False

def preview_camera():
    """Show camera preview for positioning"""
    print("\n=== Camera Preview ===")
    print("Opening camera feeds...")
    print("Press 'Q' to quit preview\n")
    
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)
    
    if not cap0.isOpened() or not cap1.isOpened():
        print("Error: Could not open cameras")
        return
    
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        
        pose0 = mp_pose.Pose(min_detection_confidence=0.5)
        pose1 = mp_pose.Pose(min_detection_confidence=0.5)
        
        print("Preview started. Position yourself so both cameras can see your full body.")
        print("Green skeleton should appear if tracking is working.")
        
        while True:
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()
            
            if not ret0 or not ret1:
                break
            
            # Process with MediaPipe
            rgb0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
            rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            
            results0 = pose0.process(rgb0)
            results1 = pose1.process(rgb1)
            
            # Draw landmarks
            if results0.pose_landmarks:
                mp_drawing.draw_landmarks(frame0, results0.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            else:
                cv2.putText(frame0, "No body detected", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if results1.pose_landmarks:
                mp_drawing.draw_landmarks(frame1, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            else:
                cv2.putText(frame1, "No body detected", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add labels
            cv2.putText(frame0, "Camera 0", (10, frame0.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame1, "Camera 1", (10, frame1.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Stack horizontally
            combined = cv2.hconcat([frame0, frame1])
            
            cv2.imshow('Camera Preview - Press Q to quit', combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        pose0.close()
        pose1.close()
        
    except ImportError:
        print("MediaPipe not available, showing raw camera feed...")
        
        while True:
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()
            
            if not ret0 or not ret1:
                break
            
            cv2.putText(frame0, "Camera 0 (no MediaPipe)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame1, "Camera 1 (no MediaPipe)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            combined = cv2.hconcat([frame0, frame1])
            cv2.imshow('Camera Preview - Press Q to quit', combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap0.release()
        cap1.release()
        cv2.destroyAllWindows()
        print("\nPreview closed")

if __name__ == "__main__":
    # Run system tests
    success = test_system()
    
    if success:
        print("\n" + "="*50)
        response = input("\nWould you like to preview cameras for positioning? (y/n): ")
        if response.lower() in ['y', 'yes']:
            preview_camera()
    
    sys.exit(0 if success else 1)
