"""Smoke test for the automated retarget path (Phase 1).

Layered, CI-safe:

  A. Every characters/*.json (non-.local) validates against the schema
     the orchestrator + headless script depend on.            [stdlib]
  B. process_take.py and headless_retarget.py byte-compile.    [stdlib]
  C. `process_take --dry-run` wires end-to-end against the wave
     fixture (config resolve + stage plan + Blender cmd).      [numpy]
  D. OPT-IN real headless run: if env MELODICCAP_TEST_BLENDER and
     MELODICCAP_TEST_BASEBLEND are set, actually run Blender and assert
     the OK marker. Otherwise SKIP with a message.             [Blender]

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
                "MelodicCapMono/blender_addon/headless_retarget.py"):
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

    print()
    if fails:
        print(f"SMOKE FAILED — {len(fails)} issue(s)")
        return 1
    print("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
