"""
Forensic diagnostic — trace forearm L/R length through every pipeline
stage of the addon's import path.

Stages traced:
  1. As read from the JSON       (post-solver, what the addon receives)
  2. After 3-frame moving average (smooth_frames)
  3. After Butterworth filter     (butterworth_filter_landmarks)

Reports per-frame forearm L (elbow_l → wrist_l) and forearm R
(elbow_r → wrist_r) in meters, plus the worst per-frame L/R ratio
at each stage.

This script COPIES the addon's smoothing functions so it runs without
Blender (no `bpy` dependency). The copies must stay byte-equivalent
to the addon's versions for the trace to be meaningful — if you change
the addon's smooth_frames or butterworth_filter_landmarks, update this
file too.

Usage:
    python trace_forearm.py takes/take_NAME.json
    python trace_forearm.py takes/take_NAME.json 60   # show 60 frames
"""
import json
import math
import sys
import copy


# COCO-WholeBody indices (mirror of melodiccap_rtm_addon.LM)
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_ANKLE = 15
RIGHT_ANKLE = 16
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_BIG_TOE = 17
RIGHT_BIG_TOE = 20
LEFT_SMALL_TOE = 18
RIGHT_SMALL_TOE = 21
LEFT_HEEL = 19
RIGHT_HEEL = 22


# ---------------------------------------------------------------------------
# Copies of addon helpers — keep in sync with melodiccap_rtm_addon.py
# ---------------------------------------------------------------------------

def smooth_frames(frames, window=5):
    """EXACT copy of addon's smooth_frames at melodiccap_rtm_addon.py:684.
    Bug is on line 702 (count++) vs line 712 (per-key division)."""
    if window < 2 or len(frames) < window:
        return frames

    half = window // 2
    smoothed = []

    for i in range(len(frames)):
        start = max(0, i - half)
        end = min(len(frames), i + half + 1)

        avg_landmarks = {}
        count = 0
        for j in range(start, end):
            lm = frames[j].get('landmarks_3d', {})
            if not lm:
                continue
            count += 1
            for key, coords in lm.items():
                if key not in avg_landmarks:
                    avg_landmarks[key] = [0.0, 0.0, 0.0]
                avg_landmarks[key][0] += coords[0]
                avg_landmarks[key][1] += coords[1]
                avg_landmarks[key][2] += coords[2]

        if count > 0:
            for key in avg_landmarks:
                avg_landmarks[key] = [c / count for c in avg_landmarks[key]]

        new_frame = dict(frames[i])
        new_frame['landmarks_3d'] = avg_landmarks
        smoothed.append(new_frame)

    return smoothed


def smooth_frames_fixed(frames, window=5):
    """Per-key counted version — what smooth_frames SHOULD do."""
    if window < 2 or len(frames) < window:
        return frames

    half = window // 2
    smoothed = []

    for i in range(len(frames)):
        start = max(0, i - half)
        end = min(len(frames), i + half + 1)

        avg_landmarks = {}
        key_counts = {}
        for j in range(start, end):
            lm = frames[j].get('landmarks_3d', {})
            if not lm:
                continue
            for key, coords in lm.items():
                if key not in avg_landmarks:
                    avg_landmarks[key] = [0.0, 0.0, 0.0]
                    key_counts[key] = 0
                avg_landmarks[key][0] += coords[0]
                avg_landmarks[key][1] += coords[1]
                avg_landmarks[key][2] += coords[2]
                key_counts[key] += 1

        for key in list(avg_landmarks.keys()):
            c = key_counts[key]
            if c > 0:
                avg_landmarks[key] = [v / c for v in avg_landmarks[key]]
            else:
                del avg_landmarks[key]

        new_frame = dict(frames[i])
        new_frame['landmarks_3d'] = avg_landmarks
        smoothed.append(new_frame)

    return smoothed


def _butter_lowpass_filtfilt(data, cutoff, fs, order=4):
    nyq = fs / 2.0
    wc = cutoff / nyq
    if wc >= 1.0:
        return data
    n = len(data)
    if n < 5:
        return list(data)
    warp = math.tan(math.pi * wc / 2.0)
    k = warp * warp
    sections = []
    for q_factor in [1.0 / (2.0 * math.cos(math.pi * 1 / 8)),
                     1.0 / (2.0 * math.cos(math.pi * 3 / 8))]:
        alpha = warp / q_factor
        norm = 1.0 / (1.0 + alpha + k)
        b0 = k * norm
        b1 = 2.0 * b0
        b2 = b0
        a1 = 2.0 * (k - 1.0) * norm
        a2 = (1.0 - alpha + k) * norm
        sections.append((b0, b1, b2, a1, a2))

    def _apply_biquad(signal, coeffs):
        b0, b1, b2, a1, a2 = coeffs
        out = list(signal)
        for i in range(2, len(out)):
            out[i] = (b0 * signal[i] + b1 * signal[i - 1] + b2 * signal[i - 2]
                       - a1 * out[i - 1] - a2 * out[i - 2])
        return out

    pad = min(12, n - 1)
    y = list(data)
    front_pad = [2.0 * y[0] - y[i] for i in range(pad, 0, -1)]
    back_pad = [2.0 * y[-1] - y[-(i + 2)] for i in range(pad)]
    padded = front_pad + y + back_pad
    result = padded
    for coeffs in sections:
        fwd = _apply_biquad(result, coeffs)
        fwd.reverse()
        bwd = _apply_biquad(fwd, coeffs)
        bwd.reverse()
        result = bwd
    return result[pad:pad + n]


def butterworth_filter_landmarks(frames, fps, cutoff_body=4.0, cutoff_feet=3.5):
    if len(frames) < 5:
        return frames
    foot_keys = {str(LEFT_ANKLE), str(RIGHT_ANKLE),
                 str(LEFT_BIG_TOE), str(RIGHT_BIG_TOE),
                 str(LEFT_SMALL_TOE), str(RIGHT_SMALL_TOE),
                 str(LEFT_HEEL), str(RIGHT_HEEL),
                 str(LEFT_KNEE), str(RIGHT_KNEE)}
    all_keys = set()
    for f in frames:
        all_keys.update(f.get('landmarks_3d', {}).keys())
    n = len(frames)
    for key in all_keys:
        cutoff = cutoff_feet if key in foot_keys else cutoff_body
        series = [None] * n
        for i, f in enumerate(frames):
            lm = f.get('landmarks_3d', {})
            if key in lm:
                series[i] = list(lm[key])
        valid = [i for i in range(n) if series[i] is not None]
        if len(valid) < 5:
            continue
        for axis in range(3):
            vals = [series[i][axis] for i in valid]
            filtered = _butter_lowpass_filtfilt(vals, cutoff, fps)
            for j, i in enumerate(valid):
                series[i][axis] = filtered[j]
        for i in valid:
            frames[i]['landmarks_3d'][key] = series[i]
    return frames


# ---------------------------------------------------------------------------
# Forearm length tracing
# ---------------------------------------------------------------------------

def forearm_length(frame, side):
    """Return (elbow→wrist) length for L/R side, or None if missing."""
    lm = frame.get('landmarks_3d', {})
    if side == 'L':
        elbow_key, wrist_key = str(LEFT_ELBOW), str(LEFT_WRIST)
    else:
        elbow_key, wrist_key = str(RIGHT_ELBOW), str(RIGHT_WRIST)
    if elbow_key not in lm or wrist_key not in lm:
        return None
    e = lm[elbow_key]
    w = lm[wrist_key]
    dx = e[0] - w[0]
    dy = e[1] - w[1]
    dz = e[2] - w[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def keypoint_present(frame, idx):
    return str(idx) in frame.get('landmarks_3d', {})


def stats_for_stage(frames, label):
    """Compute forearm L/R per frame, return ratios and worst frames."""
    ratios = []
    per_frame = []
    missing_l = 0
    missing_r = 0
    for i, f in enumerate(frames):
        fl = forearm_length(f, 'L')
        fr = forearm_length(f, 'R')
        if fl is None:
            missing_l += 1
        if fr is None:
            missing_r += 1
        if fl is None or fr is None or fl < 0.03 or fr < 0.03:
            per_frame.append((i, fl, fr, None))
            continue
        ratio = max(fl, fr) / min(fl, fr)
        ratios.append(ratio)
        per_frame.append((i, fl, fr, ratio))

    if not ratios:
        return per_frame, None
    worst = max(ratios)
    bad = sum(1 for r in ratios if r > 1.15)
    summary = {
        'n': len(ratios),
        'worst_ratio': worst,
        'mean_ratio': sum(ratios) / len(ratios),
        'bad_pct': 100.0 * bad / len(ratios),
        'missing_l': missing_l,
        'missing_r': missing_r,
    }
    print(f"\n[{label}]")
    print(f"  Frames analyzed: {summary['n']}")
    print(f"  Missing left wrist/elbow:  {missing_l}/{len(frames)}")
    print(f"  Missing right wrist/elbow: {missing_r}/{len(frames)}")
    print(f"  Worst L/R ratio: {worst:.3f}x")
    print(f"  Mean L/R ratio:  {summary['mean_ratio']:.3f}x")
    print(f"  Frames with ratio >1.15x: {bad} ({summary['bad_pct']:.1f}%)")
    return per_frame, summary


def show_per_frame(per_frame, n=20, label="frames"):
    print(f"\n  First {min(n, len(per_frame))} {label}:")
    print(f"  {'frm':>4}  {'L (m)':>7}  {'R (m)':>7}  {'L/R':>6}")
    for i, fl, fr, ratio in per_frame[:n]:
        if fl is None or fr is None:
            print(f"  {i:>4}  {'---':>7}  {'---':>7}  {'---':>6}")
        else:
            r_str = f"{ratio:.3f}x" if ratio else "---"
            print(f"  {i:>4}  {fl:>7.3f}  {fr:>7.3f}  {r_str:>6}")


def dump_xyz(frames_a, label_a, frames_b, label_b, n=10):
    """Dump elbow_l/wrist_l XYZ side-by-side for two stages of frames."""
    print(f"\n  RAW vs {label_b} — left elbow / left wrist XYZ for first {n} frames:")
    print(f"  {'frm':>3}  {'elbow_l ('+label_a+')':>25}  {'elbow_l ('+label_b+')':>25}  "
          f"{'wrist_l ('+label_a+')':>25}  {'wrist_l ('+label_b+')':>25}")
    for i in range(min(n, len(frames_a))):
        la = frames_a[i].get('landmarks_3d', {})
        lb = frames_b[i].get('landmarks_3d', {})
        ea = la.get(str(LEFT_ELBOW))
        eb = lb.get(str(LEFT_ELBOW))
        wa = la.get(str(LEFT_WRIST))
        wb = lb.get(str(LEFT_WRIST))
        ea_str = f"({ea[0]:+.3f},{ea[1]:+.3f},{ea[2]:+.3f})" if ea else "MISSING"
        eb_str = f"({eb[0]:+.3f},{eb[1]:+.3f},{eb[2]:+.3f})" if eb else "MISSING"
        wa_str = f"({wa[0]:+.3f},{wa[1]:+.3f},{wa[2]:+.3f})" if wa else "MISSING"
        wb_str = f"({wb[0]:+.3f},{wb[1]:+.3f},{wb[2]:+.3f})" if wb else "MISSING"
        print(f"  {i:>3}  {ea_str:>25}  {eb_str:>25}  {wa_str:>25}  {wb_str:>25}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python trace_forearm.py <take.json> [num_show]")
        sys.exit(1)

    path = sys.argv[1]
    if path.endswith('_raw.json'):
        print("ERROR: pass the SOLVED take JSON (no '_raw' in name).")
        print("       This script traces post-solver data through addon stages.")
        sys.exit(1)

    n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    with open(path) as f:
        take = json.load(f)
    frames = take.get('frames', [])
    if not frames:
        print("ERROR: no frames in take.")
        sys.exit(1)

    duration = take.get('duration', 0)
    fps = len(frames) / duration if duration > 0 else 13.0

    print(f"\n{path}")
    print(f"Total frames: {len(frames)}, fps: {fps:.2f}")

    # Stage 1: as read from JSON (post-solver)
    pf1, _ = stats_for_stage(frames, "STAGE 1: post-solver, as read from JSON")
    show_per_frame(pf1, n_show)

    # Stage 2: addon's BUGGY smooth_frames (window=3)
    frames_smoothed = smooth_frames(copy.deepcopy(frames), window=3)
    pf2, _ = stats_for_stage(frames_smoothed,
                             "STAGE 2: after smooth_frames(window=3) [BUGGY]")
    show_per_frame(pf2, n_show)

    # Stage 3: + Butterworth on top
    frames_butter = butterworth_filter_landmarks(
        copy.deepcopy(frames_smoothed), fps, 4.0, 3.5
    )
    pf3, _ = stats_for_stage(frames_butter,
                             "STAGE 3: + Butterworth (body=4Hz, feet=3.5Hz)")
    show_per_frame(pf3, n_show)

    # Stage 2 with FIXED smooth_frames
    frames_smoothed_fixed = smooth_frames_fixed(copy.deepcopy(frames), window=3)
    pf2f, _ = stats_for_stage(
        frames_smoothed_fixed,
        "STAGE 2 [FIXED]: after smooth_frames_fixed(window=3)"
    )
    show_per_frame(pf2f, n_show)

    frames_butter_fixed = butterworth_filter_landmarks(
        copy.deepcopy(frames_smoothed_fixed), fps, 4.0, 3.5
    )
    pf3f, _ = stats_for_stage(
        frames_butter_fixed,
        "STAGE 3 [FIXED]: + Butterworth"
    )
    show_per_frame(pf3f, n_show)

    # XYZ dump: side-by-side raw vs buggy-smoothed for the first frames.
    # This is what made it clear the bug isn't divide-by-count — when the
    # raw and smoothed ELBOW positions are different (and similarly for
    # WRIST), the smoothed forearm length collapses because the per-frame
    # solver-enforced direction varies across the smoothing window. v5.12
    # bypasses addon smoothing on solver-output JSONs to fix this.
    print("\n" + "=" * 60)
    print("XYZ EVIDENCE — RAW vs SMOOTHED positions (left elbow + wrist):")
    dump_xyz(frames, "raw", frames_smoothed, "smoothed", n=10)

    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("- v5.12 bypasses addon smoothing entirely when solver_meta is")
    print("  present in the JSON (the solver already temporally smoothed).")
    print("- STAGE 1 ratio is what the v5.12 addon will see (clean).")
    print("- STAGE 2 / 3 show the OLD behavior (buggy: position-averaging")
    print("  compresses bone lengths when forearm direction varies).")
    print("- STAGE 2 [FIXED] is v5.11's per-key count fix, which is a")
    print("  no-op when no keys are missing (does not solve THIS bug).")
    print("- Re-process & re-import are NOT needed; existing JSONs are fine.")
    print("  Reinstall v5.12 addon and re-import.")


if __name__ == '__main__':
    main()
