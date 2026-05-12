# Contributing to MelodicCap

Thanks for showing up. This is a solo-filmmaker mocap pipeline transitioning
from a custom 2-camera stereo solver (v5.19, shipped a music video) toward a
learned-prior monocular pipeline (WHAM/GVHMR → SMPL → Rigify). The "What's
needed" section of [README.md](README.md) lists the high-value contribution
areas.

## Before you write code

1. **Read [`CLAUDE.md`](CLAUDE.md) end-to-end.** It's the source of truth for
   project state, the v5.19 patch history, and the pipeline direction. The
   v3.x→v5.19 version log is intentionally kept — it explains why every
   constant in the v5.x retargeter is the value it is.
2. **Read [`README.md`](README.md).** Specifically the *What's needed*
   section and the diagram of where the pipeline is heading.
3. **Don't iterate the v5.x custom solver.** That code (`skeleton_solver.py`
   and `MelodicCapRTM/blender_addon/melodiccap_rtm_addon.py`) is frozen.
   Patching it further is a known dead-end — see the *Status* section of
   CLAUDE.md.

## Where new work goes

| Area | Directory | Status |
|---|---|---|
| Monocular SMPL capture (WHAM / GVHMR) | `MelodicCapMono/` (to be created) | the priority |
| SMPL → Rigify retargeter | `MelodicCapMono/blender_addon/` | the priority |
| Character configs (Kai, Kiko, Dr White, Hiro, THE SHADOW) | `MelodicCapMono/characters/` | needed |
| 2-camera SMPL fitting via EasyMocap | `MelodicCapMono/` (specialty path) | deferred |
| Smoke tests, CI | `scripts/` and `.github/workflows/` | needed |

## Branch + PR conventions

- **Branch naming**:
  - `claude/<topic>-<slug>` — AI-assisted work via Claude Code.
  - `<github-username>/<topic>` — direct human contributions.
- **One concern per PR.** Docs-only PRs are great. Mixing docs + behavior
  changes makes review slower.
- **PR description** should answer: what changed, why now, how to test.

## Setup

```bash
git clone https://github.com/pagnumlord/melodiccap.git
cd melodiccap
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

Confirm the install works:

```bash
python scripts/smoke_test.py
```

The smoke test loads a committed sample take from `MelodicCapRTM/takes/`,
runs the v5.16 offline solver, and checks the output for bone-length /
hip-stability invariants. It needs no cameras, no GPU, no calibration
session — just the dependencies.

## Style

- Match the surrounding code. Most of v5.x is procedural numpy. The
  Blender addon is a single ~3000-line file by design (Blender's "install
  add-on" UI wants one file).
- Keep comments to "why" not "what". The v5.x code documents non-obvious
  invariants (depth-axis noise, FK parent matrices, IK/FK convention
  inversion); follow that pattern in new code.
- New Python deps go in the relevant `requirements.txt`. Root
  `requirements.txt` is the v5.x runtime; `MelodicCapMono/requirements.txt`
  is the WHAM/SMPL stack (kept separate because torch + CUDA is heavy).

## Questions before contributing

Open a GitHub issue with the `question` label. The repo just went public
in May 2026 so there's no community yet — your question being the first
one is fine.

## License

By contributing, you agree your code is released under the MIT license
in [`LICENSE`](LICENSE).
