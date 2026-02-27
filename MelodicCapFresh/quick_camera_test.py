"""
Quick Camera Test - Debug why capture isn't working
"""
import cv2
import time

print("="*50)
print("QUICK CAMERA TEST")
print("="*50)

# Test Camera A (Sony - index 2)
print("\n[1] Testing Camera A (index 2)...")
cap_a = cv2.VideoCapture(2, cv2.CAP_DSHOW)
if cap_a.isOpened():
    ret, frame = cap_a.read()
    if ret:
        print(f"    OK! Frame size: {frame.shape}")
    else:
        print("    OPENED but can't read frames!")
else:
    print("    FAILED to open")
cap_a.release()

# Test Camera B (DroidCam - index 0)
print("\n[2] Testing Camera B (index 0)...")
cap_b = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if cap_b.isOpened():
    ret, frame = cap_b.read()
    if ret:
        print(f"    OK! Frame size: {frame.shape}")
    else:
        print("    OPENED but can't read frames!")
else:
    print("    FAILED to open")
cap_b.release()

# Test both together
print("\n[3] Testing BOTH cameras together...")
cap_a = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cap_b = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if cap_a.isOpened() and cap_b.isOpened():
    print("    Both opened!")
    
    # Try reading 10 frames
    success_count = 0
    for i in range(10):
        ret_a, frame_a = cap_a.read()
        ret_b, frame_b = cap_b.read()
        if ret_a and ret_b:
            success_count += 1
        time.sleep(0.033)  # ~30fps
    
    print(f"    Read {success_count}/10 frame pairs successfully")
else:
    print(f"    Camera A open: {cap_a.isOpened()}")
    print(f"    Camera B open: {cap_b.isOpened()}")

cap_a.release()
cap_b.release()

# Live test with window
print("\n[4] Live window test (press Q to quit)...")
print("    If you don't see a window, OpenCV display might be broken")

cap_a = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cap_b = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap_a.isOpened() or not cap_b.isOpened():
    print("    ERROR: Couldn't open cameras!")
else:
    print("    Showing live view... Press Q in the window to quit")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret_a, frame_a = cap_a.read()
        ret_b, frame_b = cap_b.read()
        
        if not ret_a or not ret_b:
            print(f"    Frame drop! A:{ret_a} B:{ret_b}")
            continue
        
        frame_count += 1
        
        # Resize for display
        frame_a = cv2.resize(frame_a, (640, 360))
        frame_b = cv2.resize(frame_b, (640, 360))
        
        # Add labels
        cv2.putText(frame_a, "Camera A (2) Sony", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_b, "Camera B (0) DroidCam", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # FPS
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame_a, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Combine
        combined = cv2.hstack([frame_a, frame_b])
        
        cv2.imshow("Camera Test - Press Q to quit", combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        # Auto-quit after 10 seconds
        if elapsed > 10:
            print("    Auto-quit after 10 seconds")
            break
    
    print(f"    Final FPS: {fps:.1f}")

cap_a.release()
cap_b.release()
cv2.destroyAllWindows()

print("\n[DONE]")
input("Press Enter to exit...")
