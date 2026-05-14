# MelodicCapMono

Forward-looking SMPL-based motion capture pipeline for Melodic Justice.
Replaces the v5.x custom-solver path (`MelodicCapRTM/`, frozen). Same
Blender + hand-keyframing workflow at the end; massively cleaner middle.

## What's here today

**Phase C** (shared infrastructure) and **Phase D** (Path A, WHAM
wrapper) are in. Path B (EasyMocap, multi-camera SMPL fitting on the
existing `MelodicCapRTM/_raw.json` takes) is the next phase — see
`CLAUDE.md` for status.

You can use the Blender side to validate the retargeter on a fixture
**without installing any model**. That's intentional — debugging the
addon and debugging the model install should never be the same task.
Once that's working, Path A (`wham/`) takes a phone MP4 and produces
a pose JSON ready for the same addon.

## Two paths, one schema

```
                          ┌────────────────────┐
   Phone video (mp4) ───▶ │  Path A: WHAM      │ ─┐
                          │  (Phase D)         │  │
                          └────────────────────┘  ├──▶  pose.json  ──▶  Blender addon  ──▶  JaxRigify
                                                  │     (melodiccap_      (this dir,           animating
   Existing _raw.json ──▶ ┌────────────────────┐  │      mono_v1)         blender_addon/)
   (2-camera RTMPose      │  Path B: EasyMocap │ ─┘
   take from              │  (Phase E)         │
   MelodicCapRTM)         └────────────────────┘
```

Both paths emit the same pose JSON schema (see [SCHEMA.md](SCHEMA.md)).
The Blender addon doesn't know or care which path produced its input.

## Which path for which shot

From the script (see `CLAUDE.md` § Short film: Melodic Justice):

- **Performance shots (~80% of the film)** — dialogue, instrument play,
  walking, gestures, reactions. Use **Path A**. Phone in landscape,
  decent light, ~30-second takes.
- **Light action (~15%)** — running, dodging, snap-awake from nightmares.
  **Path A** still fine.
- **Heavy action with prop contact (~5%)** — concert climax fight, Hiro
  vs Dr. White staff combat. **Path B** preferred for precise spatial
  accuracy; hand-keyframe prop contact on top.

## Repo layout

```
MelodicCapMono/
├── README.md              <- you are here
├── SCHEMA.md              <- pose JSON contract (both paths emit this)
├── requirements.txt       <- glue deps only (numpy, opencv-python)
├── __init__.py            <- package marker
├── characters/            <- per-character SMPL→rig mappings
│   └── jax.json
├── fixtures/              <- pose JSONs for testing the Blender side
│   │                         WITHOUT any model installed
│   ├── rest.pose.json     <-   5 frames of all-zero SMPL pose
│   └── wave.pose.json     <-  30 frames, left arm raise
├── blender_addon/         <- the new Blender 4.4 addon
│   └── melodiccap_smpl_addon.py
├── wham/                  <- Phase D (Path A): WHAM wrapper, video2pose CLI
│   ├── README.md            (install + usage)
│   ├── video2pose.py        (entry point: mp4 -> pose.json)
│   └── wham_to_pose_json.py (pkl -> schema converter)
├── easymocap/             <- Phase E (Path B) — not yet
└── scripts/               <- helper scripts (compare_poses.py in Phase F)
```

## Quick start (Phase C only — validate the Blender side)

1. **Open Blender 4.4** with a JaxRigify-armatured scene loaded.
2. **Install the addon**: Edit → Preferences → Add-ons → Install →
   point at `MelodicCapMono/blender_addon/melodiccap_smpl_addon.py`.
   Enable it.
3. **Import the fixture**: File → Import → SMPL Pose JSON. Pick:
   - Pose file: `MelodicCapMono/fixtures/rest.pose.json`
   - Character config: `MelodicCapMono/characters/jax.json`
4. **Verify**: Jax should stand in his rest pose for 5 frames. If
   anything is rotated wrong (legs splayed, arms vertical, body
   horizontal), the fix is in `characters/jax.json` — per-bone
   `rot_offset_euler_deg` values. No code changes.
5. **Repeat with `wave.pose.json`**: Jax should raise his left arm
   over 30 frames. Validates per-frame keyframing works.

## What the schema validator checks

```bash
python scripts/test_smpl_addon_smoke.py MelodicCapMono/fixtures/rest.pose.json
python scripts/test_smpl_addon_smoke.py MelodicCapMono/fixtures/wave.pose.json
```

Validates `format == "melodiccap_mono_v1"`, frame count, SMPL pose array
shape (24 joints × 3 floats per frame), betas shape, root translation
shape. Exits 0 on success.

No Blender, no model installs needed for this check. Just stdlib.

## Path A — WHAM (in this PR)

See [wham/README.md](wham/README.md) for full install + usage. Quick version:

```bash
# One-time install (~1 hour):
git clone https://github.com/yohanshin/WHAM.git
conda env create -f WHAM/environment.yml
conda activate wham
cd WHAM && bash fetch_demo_data.sh
export WHAM_DIR=/abs/path/to/WHAM   # or set WHAM_DIR=... on Windows

# Per take (after recording <take>.mp4):
python -m MelodicCapMono.wham.video2pose <take>.mp4 <take>.pose.json --character jax

# Then File -> Import -> SMPL Pose JSON in Blender, pick <take>.pose.json.
```

Schema-only test (no WHAM install required):

```bash
python scripts/test_wham_converter_smoke.py
```

Builds synthetic WHAM-style pkl dicts in memory and verifies the
converter handles every shape variant we expect (rotation matrices vs
axis-angle, single vs multi-person, with/without `global_orient`,
etc.). Useful as a regression test when WHAM's pkl layout drifts.

## Coming next

- **Phase E**: `easymocap/` directory — wraps EasyMocap (multi-view SMPL
  fitter). Reuses the existing `MelodicCapRTM/calibration/*.json` and
  `_raw.json` takes. No re-recording needed for testing.
- **Phase F**: `scripts/compare_poses.py` — diff two pose JSONs frame
  by frame for A/B comparison of the same recorded take through both
  paths.

## License

MIT — see top-level [LICENSE](../LICENSE).
