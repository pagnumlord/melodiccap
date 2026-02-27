"""
Mirror Consistency Checker
==========================
Checks if both cameras have the same mirror state.
For stereo to work, both must be consistent (both mirrored OR both normal).
"""

import cv2
import numpy as np

print("="*50)
print("MIRROR CONSISTENCY CHECKER")
print("="*50)

# Open cameras
cap_a = cv2.VideoCapture(2, cv2.CAP_DSHOW)  # Sony
cap_b = cv2.VideoCapture(0, cv2.CAP_ANY)    # DroidCam

if not cap_a.isOpened() or not cap_b.isOpened():
    print("ERROR: Could not open cameras!")
    exit(1)

cap_a.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_a.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap_b.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_b.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n" + "="*50)
print("INSTRUCTIONS")
print("="*50)
print("1. Stand where BOTH cameras can see you")
print("2. Hold up your RIGHT hand")
print("3. Look at which side of EACH frame your hand appears")
print("")
print("If your RIGHT hand appears on the LEFT side = MIRRORED")
print("If your RIGHT hand appears on the RIGHT side = NORMAL")
print("")
print("Press Q to quit after checking")
print("="*50)

while True:
    ret_a, frame_a = cap_a.read()
    ret_b, frame_b = cap_b.read()
    
    if not ret_a or not ret_b:
        continue
    
    # Resize for display
    display_a = cv2.resize(frame_a, (640, 360))
    display_b = cv2.resize(frame_b, (640, 360))
    
    h, w = display_a.shape[:2]
    
    # Draw center lines and labels
    cv2.line(display_a, (w//2, 0), (w//2, h), (0, 255, 255), 2)
    cv2.line(display_b, (w//2, 0), (w//2, h), (0, 255, 255), 2)
    
    # Left/Right labels
    cv2.putText(display_a, "LEFT", (30, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(display_a, "RIGHT", (w-120, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(display_b, "LEFT", (30, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(display_b, "RIGHT", (w-120, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Camera labels
    cv2.rectangle(display_a, (0, 0), (300, 40), (0, 0, 0), -1)
    cv2.putText(display_a, "Camera A (Sony, idx 2)", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.rectangle(display_b, (0, 0), (320, 40), (0, 0, 0), -1)
    cv2.putText(display_b, "Camera B (DroidCam, idx 0)", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Instructions at bottom
    cv2.rectangle(display_a, (0, h-50), (w, h), (0, 0, 0), -1)
    cv2.putText(display_a, "Hold up RIGHT hand - which side is it on?", (10, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.rectangle(display_b, (0, h-50), (w, h), (0, 0, 0), -1)
    cv2.putText(display_b, "Hold up RIGHT hand - which side is it on?", (10, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    combined = np.hstack([display_a, display_b])
    
    # Top instruction bar
    top_bar = np.zeros((60, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(top_bar, "Both cameras must be CONSISTENT: both mirrored OR both normal", 
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(top_bar, "If different, we need to flip ONE camera. Press Q when done checking.",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    display = np.vstack([top_bar, combined])
    
    cv2.imshow("Mirror Check - Press Q when done", display)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_a.release()
cap_b.release()
cv2.destroyAllWindows()

print("\n" + "="*50)
print("WHAT DID YOU SEE?")
print("="*50)
print("")
print("If BOTH cameras showed your RIGHT hand on the LEFT side:")
print("  -> Both are MIRRORED (consistent) - this is OK")
print("")
print("If BOTH cameras showed your RIGHT hand on the RIGHT side:")
print("  -> Both are NORMAL (consistent) - this is OK")
print("")
print("If cameras were DIFFERENT (one LEFT, one RIGHT):")
print("  -> INCONSISTENT! This breaks stereo calibration!")
print("  -> We need to flip ONE camera to match the other")
print("")
print("="*50)
