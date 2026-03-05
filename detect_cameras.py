"""Quick camera detection - lists all available camera devices."""
import cv2

print("Scanning camera IDs 0-9...\n")

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        backend = cap.getBackendName()
        ret, frame = cap.read()
        status = "readable" if ret else "opened but can't read"
        print(f"  Camera {i}: {w}x{h} @ {fps:.0f}fps ({backend}) - {status}")
        cap.release()
    else:
        cap.release()

print("\nDone. Use the IDs above for --left and --right in calibration.")
