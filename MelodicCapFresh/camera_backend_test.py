"""
Camera Backend Test - Find the right backend for DroidCam
"""
import cv2
import time

print("="*50)
print("CAMERA BACKEND TEST")
print("="*50)

# List of backends to try
backends = [
    (cv2.CAP_ANY, "CAP_ANY (auto)"),
    (cv2.CAP_DSHOW, "CAP_DSHOW (DirectShow)"),
    (cv2.CAP_MSMF, "CAP_MSMF (Media Foundation)"),
    (cv2.CAP_VFW, "CAP_VFW (Video for Windows)"),
]

def test_camera(index, name):
    print(f"\n{'='*50}")
    print(f"Testing {name} (index {index})")
    print("="*50)
    
    working_backends = []
    
    for backend_id, backend_name in backends:
        print(f"\n  [{backend_name}]", end=" ")
        try:
            cap = cv2.VideoCapture(index, backend_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✓ OK! Frame: {frame.shape}")
                    working_backends.append((backend_id, backend_name))
                else:
                    print("✗ Opened but no frames")
            else:
                print("✗ Failed to open")
            cap.release()
        except Exception as e:
            print(f"✗ Exception: {e}")
    
    return working_backends

# Test Camera A (Sony)
working_a = test_camera(2, "Sony ZV-1F")

# Test Camera B (DroidCam)
working_b = test_camera(0, "DroidCam")

# Summary
print("\n" + "="*50)
print("SUMMARY")
print("="*50)

print(f"\nCamera A (Sony, index 2):")
if working_a:
    for bid, bname in working_a:
        print(f"  ✓ {bname}")
else:
    print("  ✗ No working backend found!")

print(f"\nCamera B (DroidCam, index 0):")
if working_b:
    for bid, bname in working_b:
        print(f"  ✓ {bname}")
else:
    print("  ✗ No working backend found!")
    print("\n  TROUBLESHOOTING:")
    print("  1. Is DroidCam Client running on PC?")
    print("  2. Is phone connected (USB or WiFi)?")
    print("  3. Is DroidCam showing video in its own window?")
    print("  4. Try closing ALL other apps that might use the camera")
    print("  5. Try restarting DroidCam Client")

# If both have working backends, test them together
if working_a and working_b:
    print("\n" + "="*50)
    print("TESTING BOTH CAMERAS TOGETHER")
    print("="*50)
    
    # Use the first working backend for each
    backend_a = working_a[0][0]
    backend_b = working_b[0][0]
    
    print(f"\nUsing {working_a[0][1]} for Camera A")
    print(f"Using {working_b[0][1]} for Camera B")
    
    cap_a = cv2.VideoCapture(2, backend_a)
    cap_b = cv2.VideoCapture(0, backend_b)
    
    if cap_a.isOpened() and cap_b.isOpened():
        print("\n✓ Both cameras opened!")
        
        # Quick FPS test
        success = 0
        start = time.time()
        for i in range(30):
            ret_a, _ = cap_a.read()
            ret_b, _ = cap_b.read()
            if ret_a and ret_b:
                success += 1
        elapsed = time.time() - start
        fps = 30 / elapsed if elapsed > 0 else 0
        
        print(f"  Successful frame pairs: {success}/30")
        print(f"  Effective FPS: {fps:.1f}")
        
        print(f"\n{'='*50}")
        print("RECOMMENDED SETTINGS FOR melodic_capture_v3.py:")
        print("="*50)
        print(f"\n# In the init_cameras() function, change:")
        print(f"# self.cap_a = cv2.VideoCapture(Config.CAM_A_INDEX, cv2.CAP_DSHOW)")
        print(f"# to:")
        if backend_a == cv2.CAP_DSHOW:
            print(f"self.cap_a = cv2.VideoCapture(2, cv2.CAP_DSHOW)  # Keep as-is")
        else:
            print(f"self.cap_a = cv2.VideoCapture(2, cv2.{working_a[0][1].split()[0]})")
        
        if backend_b == cv2.CAP_DSHOW:
            print(f"self.cap_b = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Keep as-is")
        else:
            backend_const = working_b[0][1].split()[0]
            print(f"self.cap_b = cv2.VideoCapture(0, cv2.{backend_const})")
    else:
        print(f"\n✗ Failed to open both together")
        print(f"  Camera A: {cap_a.isOpened()}")
        print(f"  Camera B: {cap_b.isOpened()}")
    
    cap_a.release()
    cap_b.release()

print("\n[DONE]")
input("Press Enter to exit...")
