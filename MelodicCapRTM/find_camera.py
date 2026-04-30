"""Find which camera indices actually deliver frames (not just register as devices)."""
import cv2
import time

print("Probing camera indices 0-4 (testing actual frame delivery)...\n")

for i in range(5):
    for backend_name, backend in [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]:
        cap = cv2.VideoCapture(i, backend)
        if not cap.isOpened():
            cap.release()
            continue

        # Try to actually read a frame
        t0 = time.time()
        ret, frame = cap.read()
        elapsed = (time.time() - t0) * 1000

        if ret and frame is not None:
            h, w = frame.shape[:2]
            status = f"WORKING — {w}x{h}, {elapsed:.0f}ms"
        else:
            status = f"PHANTOM — opens but no frames (read took {elapsed:.0f}ms)"

        print(f"  idx {i} via {backend_name}: {status}")
        cap.release()
    print()
