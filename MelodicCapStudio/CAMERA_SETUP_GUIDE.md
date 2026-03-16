# MelodicCap Studio - Camera Setup & Quality Guide

## Camera Angle (CRITICAL)

Stereo triangulation quality depends heavily on the angle between cameras.
The convergence angle is the angle each camera is rotated inward.

| Total Angle Between Cameras | Quality | Notes |
|-----------------------------|---------|-------|
| 0° (parallel) | Best depth accuracy | Smallest overlap, subject must be centered |
| 10-30° (5-15° per camera) | **BEST OVERALL** | Good overlap + good depth resolution |
| 40-60° | Poor | Depth error 2-3x worse, landmarks lost |
| 90° | Unusable | Minimal camera overlap |

### Recommended Layout

```
          [Subject]
          2-3m away
             |
   __________|__________
  |                      |
  |   capture area       |
  |______________________|
 /                        \
Cam A  (~5-10° in)      Cam B  (~5-10° in)
 |------- 0.8-1.5m -------|
```

- Cameras side by side, **0.8-1.5m apart** (baseline)
- Both pointing **nearly straight ahead**, 5-10° toe-in each
- Subject **2-3m** in front
- Both cameras at **same height** (chest-to-head level)
- **Both cameras must see the full body** — verify in both previews before calibrating

### Why Wider Angles Fail

Triangulation depth error ~ `d² / (b × f)` where:
- d = distance to subject
- b = **effective** baseline (shrinks at steep angles)
- f = focal length in pixels

At 45° convergence, the effective baseline is cos(22.5°) × physical baseline ≈ 0.92×,
BUT the real problem is that the epipolar geometry becomes ill-conditioned.
Small 2D detection errors (3-5 pixels from MediaPipe) become 5-15cm 3D errors
instead of the 1-2cm you'd get with near-parallel cameras.

At 45°+, one camera often can't see body parts the other can (e.g., left hip
is occluded from one camera's perspective), causing systematic landmark dropout.

---

## MediaPipe Model Complexity

| Mode | CLI Flag | Complexity | Speed vs Default | Accuracy |
|------|----------|-----------|-----------------|----------|
| Lite | `--lite` | 0 | ~2x faster | Reduced: worse on occlusion, edge cases |
| Full | (default) | 1 | Baseline | Good general-purpose |
| Heavy | `--offline` | 2 | ~2x slower | Best: handles occlusion, unusual poses |

All three detect the same 33 landmarks. The difference is the neural network size:
- **Lite (0)**: Smaller network, may jitter on partially-occluded limbs
- **Full (1)**: Good balance for real-time capture
- **Heavy (2)**: Largest network, best at hard poses — only practical in offline mode

### Recommendation

Use **Full (1)** for live capture. Use **Offline mode** (Heavy/2) for final takes
when you need maximum quality. Don't use Lite unless you're specifically testing
frame rate improvements.

---

## GPU Support

**MediaPipe Pose (Python API) does NOT use CUDA.** It runs on CPU via TensorFlow Lite.
An RTX 3060 Ti (or any GPU) sits idle during MediaPipe inference.

GPU acceleration for MediaPipe requires the C++ API or mobile platforms,
not the `mp.solutions.pose` Python bindings.

### What uses the GPU
- OpenCV display/drawing (minimal)
- Video decoding (if using hardware decode)

### If you need GPU-accelerated pose estimation
Alternatives that support CUDA:
- **RTMPose** (MMPose) — fast, accurate, CUDA support
- **ViTPose** — state-of-art accuracy, CUDA support
- These require code changes to replace the MediaPipe detector

For now, CPU performance at complexity=1 is adequate for 10-15 FPS per camera.

---

## FPS Targets

| FPS | Quality | Use Case |
|-----|---------|----------|
| 5-10 | Minimum | Slow movements, gestures |
| 15-20 | Good | Walking, arm movements |
| 24-30 | Professional | Fast actions, dancing |
| 60+ | Premium | Sports, martial arts |

Current system produces ~4-5 FPS effective output (at complexity=1).
With fixed camera angles, more landmarks will survive filtering,
increasing effective FPS. Offline mode removes the FPS constraint entirely.

---

## Quality Checklist (Pre-Capture)

Before recording, verify:

- [ ] Both cameras see full body (head to feet) in their previews
- [ ] Cameras are 0.8-1.5m apart, nearly parallel (5-15° toe-in)
- [ ] Stereo RMS < 1.0 after calibration
- [ ] Floor calibration std < 0.01m
- [ ] Verification passes with < 5mm avg error
- [ ] Capture area is well-lit (even lighting, no harsh backlighting)
- [ ] Subject wearing fitted clothing (loose/baggy clothes confuse pose detection)

## Quality Checklist (Post-Capture)

After recording, check the take quality report:

- [ ] Most frames have 25+ landmarks (out of 33)
- [ ] Left hip (landmark 23) is present in majority of frames
- [ ] Bone-length CV < 5%
- [ ] Quality rating: GOOD or FAIR
- [ ] Outlier rate < 5 per second

---

## Camera Hardware Notes

### Matched vs Mismatched Cameras

Ideal: two identical cameras (same focal length, same sensor, same latency).

Your setup: Sony ZV-1F (fx≈780px) + DroidCam/Samsung S25 (fx≈817px).
- ~5% focal length mismatch — acceptable
- DroidCam adds software latency — problematic for fast movements
- For slow-to-moderate motion, this works

If upgrading: a second USB webcam (even a $50 Logitech C920) will have
better frame sync than a phone app. Matching cameras eliminates all
focal-length mismatch concerns.

### Resolution

Both cameras process at the same resolution (determined by the smaller camera).
Higher resolution = better MediaPipe accuracy but slower processing.
1280x720 is the current target — good balance.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Most landmarks filtered as outliers | Camera angle too steep | Reduce convergence to 10-15° |
| Left/right body parts always missing | Occlusion from camera angle | Make cameras more parallel |
| Bone lengths vary wildly | Poor calibration or bad angle | Recalibrate with better board coverage |
| Very low FPS (< 5) | CPU bottleneck | Use `--lite` or `--offline` |
| Good 2D detection but bad 3D | Calibration error or angle | Verify calibration (V key), check RMS |
| Floor offset wrong | Board wasn't flat during floor cal | Redo floor calibration on flat surface |
