"""Smoke test for the automated retarget path (Phases 1 + 4 + Phase 3 batch).

Layered, CI-safe:

  A. Every characters/*.json (non-.local) validates against the schema
     the orchestrator + headless script depend on.            [stdlib]
  B. process_take.py, footlock.py, batch.py, headless_retarget.py,
     and _smpl_fk.py all byte-compile.                         [stdlib]
  C. `process_take --dry-run` wires end-to-end against the wave
     fixture (config resolve + stage plan + Blender cmd).      [numpy]
  D. OPT-IN real headless run: if env MELODICCAP_TEST_BLENDER and
     MELODICCAP_TEST_BASEBLEND are set, actually run Blender and assert
     the OK marker. Otherwise SKIP with a message.             [Blender]
  E. Footlock detection on a synthetic walking pose JSON.       [numpy]
  F. batch.py --dry-run on a 2-row synthetic manifest; verify the
     batch_report.json totals.                                  [stdlib]

Exit 0 if all non-skipped checks pass, 1 on any failure.

    python scripts/test_headless_retarget_smoke.py
"""

from __future__ import annotations

import json
import os
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Section E imports MelodicCapMono directly; make the repo root resolvable
# regardless of cwd. Other sections subprocess `python -m ...` with cwd=REPO.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
CHARS = REPO / "MelodicCapMono" / "characters"
WAVE = REPO / "MelodicCapMono" / "fixtures" / "wave.pose.json"
VALID_TYPES = {"COPY_ROTATION", "COPY_LOCATION"}
VALID_ROOT = {"zero", "trans"}

fails: list[str] = []


def ok(m): print(f"  PASS  {m}")
def bad(m): fails.append(m); print(f"  FAIL  {m}")
def skip(m): print(f"  SKIP  {m}")


def check_config(p: Path):
    try:
        c = json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        bad(f"{p.name}: not valid JSON ({e})"); return
    for k in ("name", "armature", "bvh_export", "ik_fk_one", "bone_map"):
        if k not in c:
            bad(f"{p.name}: missing required key {k!r}"); return
    be = c["bvh_export"]
    if be.get("root_motion") not in VALID_ROOT:
        bad(f"{p.name}: bvh_export.root_motion must be one of {VALID_ROOT}")
    if len(be.get("global_rot_euler_deg", [])) != 3:
        bad(f"{p.name}: bvh_export.global_rot_euler_deg must be [x,y,z]")
    if not isinstance(c["ik_fk_one"], list) or not c["ik_fk_one"]:
        bad(f"{p.name}: ik_fk_one must be a non-empty list")
    if not c["bone_map"]:
        bad(f"{p.name}: bone_map is empty"); return
    for src, spec in c["bone_map"].items():
        if "bone" not in spec:
            bad(f"{p.name}: bone_map[{src!r}] missing 'bone'")
        if spec.get("type", "COPY_ROTATION") not in VALID_TYPES:
            bad(f"{p.name}: bone_map[{src!r}].type invalid")
    ok(f"{p.name}: schema valid ({len(c['bone_map'])} bones, "
       f"{len(c['ik_fk_one'])} IK_FK)")


def main() -> int:
    print("MelodicCap headless-retarget smoke\n")

    print("A. character configs")
    cfgs = [p for p in sorted(CHARS.glob("*.json"))
            if not p.stem.endswith(".local")]
    if not cfgs:
        bad("no character configs found")
    for p in cfgs:
        check_config(p)

    print("\nB. byte-compile")
    for rel in ("MelodicCapMono/orchestrate/process_take.py",
                "MelodicCapMono/orchestrate/footlock.py",
                "MelodicCapMono/orchestrate/batch.py",
                "MelodicCapMono/blender_addon/headless_retarget.py",
                "MelodicCapMono/wham/_smpl_fk.py"):
        try:
            py_compile.compile(str(REPO / rel), doraise=True)
            ok(rel)
        except py_compile.PyCompileError as e:
            bad(f"{rel}: {e}")

    print("\nC. process_take --dry-run wiring")
    with tempfile.TemporaryDirectory() as td:
        cmd = [sys.executable, "-m",
               "MelodicCapMono.orchestrate.process_take",
               "--from-pose-json", str(WAVE), "--character", "jax",
               "--base-blend", "X.blend", "--blender-exe", "blender",
               "--out", str(Path(td) / "x.blend"), "--dry-run"]
        r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
        out = r.stdout + r.stderr
        if r.returncode == 0 and "BVH: convert" in out \
                and "headless_retarget.py" in out:
            ok("dry-run reached BVH + Blender stages, exit 0")
        else:
            bad(f"dry-run failed (rc={r.returncode}):\n{out[-600:]}")

    print("\nD. real headless run (opt-in)")
    bexe = os.environ.get("MELODICCAP_TEST_BLENDER")
    bbase = os.environ.get("MELODICCAP_TEST_BASEBLEND")
    if not (bexe and bbase):
        skip("set MELODICCAP_TEST_BLENDER and MELODICCAP_TEST_BASEBLEND "
             "to exercise the real Blender retarget")
    else:
        with tempfile.TemporaryDirectory() as td:
            outb = Path(td) / "wave.blend"
            cmd = [sys.executable, "-m",
                   "MelodicCapMono.orchestrate.process_take",
                   "--from-pose-json", str(WAVE), "--character", "jax",
                   "--base-blend", bbase, "--blender-exe", bexe,
                   "--out", str(outb)]
            r = subprocess.run(cmd, cwd=str(REPO),
                               capture_output=True, text=True)
            out = r.stdout + r.stderr
            if r.returncode == 0 and "[headless_retarget] OK" in out \
                    and outb.exists():
                ok("real headless retarget produced a .blend")
            else:
                bad(f"real headless failed (rc={r.returncode}):\n{out[-800:]}")

    print("\nE. footlock detection on synthetic walking pose")
    try:
        import numpy as np
        from MelodicCapMono.orchestrate import footlock
        # Build a synthetic walking pose: hips bob sinusoidally, ankles
        # alternate dipping through the floor. Pose is all-zero rotations
        # for limbs (SMPL T-pose) -- we drive contact through smpl_trans's
        # Y component AND alternating leg lift via simple hip_y rotations.
        # To make it actually exercise the detector, give the left and
        # right ankles distinct vertical motion via the hip joint pose.
        n = 60
        fps = 30.0
        frames = []
        import math as _m
        for f in range(n):
            t = f / fps
            # smpl_pose all zero; alter left_hip (joint 1) and right_hip
            # (joint 2) X-rotation in alternation to lift the corresponding
            # foot off the floor by rotating the leg backward, then forward.
            pose = [0.0] * 72
            # left foot lifts on first half-cycle; right on second half
            phase = _m.sin(2 * _m.pi * 1.0 * t)         # period 1.0s
            pose[1 * 3 + 0] = -0.6 * max(0.0, phase)    # left_hip X
            pose[2 * 3 + 0] = -0.6 * max(0.0, -phase)   # right_hip X
            frames.append({
                "frame": f, "timestamp": float(t),
                "smpl_pose": pose, "smpl_trans": [0.0, 0.0, 0.0],
            })
        walk = {"format": "melodiccap_mono_v1", "source_model": "fixture",
                "source_video": "smoke_walk", "character": "jax",
                "fps": fps, "smpl_betas": [0.0] * 10, "frames": frames}
        res = footlock.detect_foot_contacts(walk)
        # Floor calibration should be near the ankle rest Y (~-0.886)
        floor_l = res["floor_z"]["L"]
        floor_r = res["floor_z"]["R"]
        if abs(floor_l - floor_r) > 0.05:
            bad(f"floor_z asymmetry too high: L={floor_l:.3f} R={floor_r:.3f}")
        else:
            ok(f"floor_z calibrated L={floor_l:+.3f} R={floor_r:+.3f}")
        # per_frame arrays length matches frame count
        if (len(res["per_frame"]["foot_contact_l"]) != n
                or len(res["per_frame"]["foot_contact_r"]) != n):
            bad("per_frame contact array length mismatch")
        else:
            ok(f"per_frame length = {n}")
        # Both feet should be detected as planted for a meaningful chunk
        fic_l = res["frames_in_contact"]["L"]
        fic_r = res["frames_in_contact"]["R"]
        if fic_l < 5 or fic_r < 5:
            bad(f"frames_in_contact too low: L={fic_l} R={fic_r}")
        else:
            ok(f"frames_in_contact L={fic_l} R={fic_r} of {n}")
        # At least one interval per side (alternating walking pattern)
        ivs = res["intervals"]
        sides = {iv["side"] for iv in ivs}
        if sides != {"L", "R"} or len(ivs) < 2:
            bad(f"expected ≥2 intervals covering both sides, got {ivs}")
        else:
            ok(f"intervals={len(ivs)} covering both sides")
    except Exception as e:  # noqa: BLE001
        bad(f"footlock smoke: {type(e).__name__}: {e}")

    print("\nF. batch.py --dry-run on synthetic 2-row manifest")
    try:
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            manifest = td_p / "test.csv"
            manifest.write_text(
                "video,from_pose_json,character,fps,out_name\n"
                f",{WAVE},jax,30,wave_a\n"
                f",{WAVE},jax,,wave_b\n",
                encoding="utf-8",
            )
            cmd = [sys.executable, "-m",
                   "MelodicCapMono.orchestrate.batch", str(manifest),
                   "--dry-run", "--base-blend", "X.blend",
                   "--blender-exe", "blender"]
            r = subprocess.run(cmd, cwd=str(REPO),
                               capture_output=True, text=True)
            out = r.stdout + r.stderr
            report = td_p / "batch_report.json"
            if r.returncode != 0 or "[batch] DONE" not in out:
                bad(f"batch dry-run failed (rc={r.returncode}):\n"
                    f"{out[-600:]}")
            elif not report.exists():
                bad("batch dry-run produced no batch_report.json")
            else:
                rep = json.loads(report.read_text())
                if rep["total"] == 2 and rep["dry_run"] == 2 and rep["fail"] == 0:
                    ok(f"batch dry-run: total={rep['total']} dry={rep['dry_run']} fail={rep['fail']}")
                else:
                    bad(f"batch report unexpected: total={rep['total']} "
                        f"dry={rep['dry_run']} fail={rep['fail']}")
    except Exception as e:  # noqa: BLE001
        bad(f"batch dry-run smoke: {type(e).__name__}: {e}")

    print()
    if fails:
        print(f"SMOKE FAILED — {len(fails)} issue(s)")
        return 1
    print("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
