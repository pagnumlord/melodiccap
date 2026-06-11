# Production plan — Melodic Justice (short film)

Shoot ordering, character rigging priority, and per-scene pipeline routing
for a solo filmmaker (one performer, costume changes) using the
WHAM → BVH → Rokoko-headless → foot-lock pipeline documented in
[`CLAUDE.md`](../CLAUDE.md) and `MelodicCapMono/wham/README.md`. Derived
from the working draft of the Act 1 + Act 2 + epilogue script; **no
screenplay text appears here** (the script is the writer's IP and the
repo is public).

Update this file when the script changes or a shoot session reveals a
new constraint.

## Pipeline status (this branch, May 2026)

Engine end-to-end works on a real take. One command per take + batch
runner for a whole shoot:

```
phone clip → WHAM (SMPL) → BVH → Rokoko-headless retarget
          → foot-contact + IK lock → baked per-take .blend
          × N takes in a manifest CSV → overnight batch
```

What stays manual on top of the baked base layer (by design):

| Manual pass | Why it stays manual |
|---|---|
| Character path across the set (root XY translation over time) | Monocular trans drifts; we pin and keyframe the path. |
| Foot-contact judgement on close-up shots | Automated detector catches majority; artist confirms hero shots. |
| Fingers on instruments | SMPL has no fingers; WHAM doesn't model them. Hand-keyframe per take. |
| Facial expression / lipsync | Separate pipeline (face mocap or Rhubarb-style lipsync), not the body pipeline. |
| Hand-body contact / occlusion artifacts | Monocular wrist orientation is under-constrained; clean per shot. |

## Characters — rigging priority

After Jax (rigged, calibrated, shipping), order by frequency × motion
variety in the current draft:

| # | Character | Approx scenes | Motion footprint | Rig complexity | Sequence |
|---|---|---:|---|---|---|
| 1 | **Kai** | ~9 | Drumming, drumstick/stick combat, tackled, climbing, dialogue | Standard human (Rigify) | **Next** |
| 2 | **Kiko** | ~9 | Keytar playing, keytar-as-bat, sneaking, expressive dialogue | Standard human | **Next** (parallel with Kai) |
| 3 | **Hiro** | ~4 | Standing/seated dialogue + brief staff combat (arm in sling later) | Standard, optional sling-arm pose constraint | After band |
| 4 | **Dr. White** | ~5 | Cane combat, formal posture, falls, energy throws via cane | Standard + cane prop | After band |
| 5 | **THE SHADOW** | ~3 (montage + climax) | Slow imposing stance, energy throws, arm slams, knockdowns | Standard — face mask is VFX, no face rig needed | After Dr. White |
| 6 | **Jett** | 2 (flashbacks) | Piano playing, dialogue, choked/pinned (combat hits) | Standard adult | Late |
| 7 | **Young Jax** | 2 (flashbacks) | Piano, slammed to ground, child proportions | **Smaller skeleton — different `height_m`, may need a `source_height_m` override added to the pipeline's height-fit math** | Late |
| 8 | **Umbrals (incl. Alex)** | Concert + montage | Choreographed group attacks, debris strike, lunges | One shared "umbral" rig with cosmetic variants | Late |
| 9 | **Yori Takahashi** | 1 (after-credits) | Doorway reveal, brief | Standard | Last |

Rationale for *Kai + Kiko first*: they share every garage scene and the
concert sequence with Jax — completing them unlocks the bulk of
shootable material in one step. Rig both in parallel; they're both
standard humanoid Rigify and the rigging effort is near-identical.

## Per-character setup checklist (one-time, ~30 min each)

For each new character after the rig is weight-painted:

1. Confirm the **armature object name** in `Characters.blend` (or a per-character `.blend`) is clean (e.g. `KaiRigify`).
2. Copy `MelodicCapMono/characters/jax.json` → `kai.json`. Edit `name`, `armature`, `height_m`. Leave `bone_map` and `ik_fk_one` untouched if it's a stock Rigify rig (bone names are shared).
3. (Untracked) `MelodicCapMono/characters/kai.local.json` with `base_blend` and `blender_exe` paths (machine-specific; gitignored).
4. **First-shot ground-truth test**: record one short clip, run through `process_take --character kai`, open the baked `.blend`. If a bone is rotated wrong, add a `calibration` entry (e.g. `"forearm_fk.L": { "invert": [false, false, true] }`) and re-run. Lock the config.

Young Jax is the one exception — child proportions need a source-height override the pipeline doesn't expose yet. Flag it when we get to that rig; ~30 min code change.

## Shoot ordering — clusters, not script order

Cluster by (character × location) to minimize costume/prop changeover
overhead per session.

| Wave | Cluster | Rigs needed by this wave | Why this order |
|---|---|---|---|
| 1 | **A — Jax solo** (bedroom, rooftop, streets, sneaking) | Jax | Validates the production pipeline on real-volume takes (10-15 clips) before multi-character commitments. First batch overnight run. |
| 2 | **B — Garage practice scenes** (multiple per script) | + Kai + Kiko | Shoot *all* Jax garage parts in one session, then all Kai parts, then all Kiko parts. NLA composite the three. Heavy instrument-playing scenes dominate runtime here. |
| 3 | **C — Music shop** + **D — Stress Relay HQ** | + Hiro + Dr. White | Two-person dialogue clusters; lower complexity than concert. Bring a stand-in cane for Dr. White scenes. |
| 4 | **G — Black Sun / Shadow atmospheric shots** | + THE SHADOW | Slow, imposing, atmospheric. Establishes the Shadow performance vocabulary before the climax sequence. |
| 5 | **F — Flashback scenes** (opening + memory) | + Jett + Young Jax | Piano + violent attack beats. Foam stand-in for the choke/slam beats; hand-keyframe contact. |
| 6 | **E — Concert venue (the boss fight)** | All rigs | Stage performance + Umbral attack + climax fight + rhythmic finale. **The hardest sequence — budget 2-3 sessions.** Lots of hand-polish. |
| 7 | **H — Epilogue + Yori teaser** | + Yori | Easy cooldown after Cluster E. |

## Pipeline routing by scene type

| Scene type | Path A (WHAM auto) | Hand polish on top | Pure VFX (no mocap) |
|---|---|---|---|
| Dialogue (most of Acts 1-2) | Body + limbs ✓ | Fingers on relays/Loops | Cybernetic glows, holograms |
| Instrument playing | Body + arms ✓ | **Fingers on guitar/keytar/drums/piano (always)** | Pickup glow, holo overlays |
| Walking / running / sitting / lying | All ✓ | Foot contact polish on top of auto IK lock | — |
| Light action (necklace attack, sneaking, dodging) | Most ✓ | Hit reactions, prop contact | Floating necklace, dark tendrils |
| Heavy action (concert melee, climax fight) | Body base layer ✓ | **Heavy**: prop contact, leap from rig, weapon swings, weapon transformation, hit reactions | Laser bolts, energy beams, mask cracks, guitar→guntar morph |
| Flashback attacks (opening, memory) | Body ✓ | Choke/slam beats, locked-eye moments | Dark tendrils, shadow figures |
| All face / lipsync | — | — | Separate pipeline (face mocap or Rhubarb), composited on top |

Estimate (per CLAUDE.md): ~80% performance / ~15% light action / ~5%
heavy action with prop contact. The 5% (the climax) takes
disproportionate polish time per second of screen time.

## Per-session shoot checklist

For each filming session:

1. One clean phone clip per take. 5-60 s. Full body in frame, single performer, decent lighting, neutral background where practical.
2. Name files at record time so the manifest builds itself:
   - `jax_walk_01.mp4`, `jax_walk_02.mp4`, `kai_drum_intro_01.mp4`, `kiko_keytar_swing_03.mp4`, etc.
   - The naming convention is `<character>_<scene-tag>_<NN>.mp4`. Stable across the project.
3. Drop clips in `<shoot-date>/` next to a `manifest.csv`:
   ```csv
   video,from_pose_json,character,fps,out_name
   C:/shoots/2026-MM-DD/jax_walk_01.mp4,,jax,30,jax_walk_01
   C:/shoots/2026-MM-DD/kai_drum_intro_01.mp4,,kai,30,kai_drum_intro_01
   ```
4. `python -m MelodicCapMono.orchestrate.batch C:/shoots/2026-MM-DD/manifest.csv --continue-on-error`
5. Next morning: open baked `.blend`s, judge what's good vs needs reshoot vs needs polish. Re-shoot is cheap (phone clip + one manifest row).

## NLA composite workflow — multi-person scenes

The script's most NLA-heavy scenes are the garage practices, the
backstage hallway confrontation, and the concert climax.

Pattern that scales:

1. **Block the scene physically once** — mark spots on the floor with tape so each character take aligns spatially.
2. **Shoot each character's part as a separate take**, in the same physical space, with the same blocking marks. Record yourself reacting to imagined off-camera cues so timing stays consistent across takes.
3. **Process each take separately** through the batch runner — output is one `.blend` per character with that character's baked action.
4. **Composite in Blender** — open a master scene, append each per-character armature + its action, drop the actions into the NLA editor on their respective rigs. Adjust frame offsets so beats line up across takes.
5. **Keyframe a shared camera + lighting** in the master scene; characters all play their NLA strips simultaneously.

This is the workflow CLAUDE.md already endorses ("current pipeline is
single-person. Capture each performer separately, composite in Blender's
NLA editor"). The batch runner makes step 3 trivial.

## Known polish budget (estimates, per shot type)

Rough effort multipliers vs. the mocap base layer time:

| Shot type | Hand-polish multiplier | What you're polishing |
|---|---|---|
| Static dialogue | 1.0× (essentially nothing) | Maybe an idle hand fidget |
| Walking dialogue | 1.2× | Root path keyframes (5-10 keys per shot) |
| Instrument-playing dialogue | 2-4× | Finger keyframes per song (large but reusable across takes of the same song) |
| Light action (sneak, dodge) | 1.5-2× | Hit reactions, prop contact moments |
| Concert melee | 5-10× | Choreographed multi-person, weapon contact, NLA timing |
| Climax fight | 10-20× | Laser combat, leap from rig, weapon transform, rhythmic dodging on beat |

Plan the schedule against these multipliers: 30 dialogue clips cost
much less time than one minute of climax footage even though the
runtimes are similar.

## Specific scenes worth pre-thinking before they're shot

- **Opening flashback** (piano + attack): paired performance (Young Jax + Jett at piano) means two takes recorded in sequence in the same room. The attack beats (slammed, choked) are hand-polish-heavy; record the body falling with a stand-in/foam prop and clean the contact.
- **Garage first practice**: 3 instruments, 3 performers. Plan the song first (so finger keyframing is reusable across the multiple times the song appears), then shoot each performer's part to a click track.
- **Necklace attack** (Kai strangled, slammed to wall): solo physical performance; the necklace is pure VFX, contact is sold via Kai's recoil. Record Kai's reaction; comp the necklace after.
- **Climax: weapon transformation**: pure VFX — Jax's mocap is just "holding the guitar in a stance". The transformation animation is a separate animated VFX layer over the held-guitar mocap.
- **Climax: leap from lighting rig**: heavy action. Record the body intent; hand-polish the airtime and the slam-down on beat. Probably the single most time-intensive polish job in the project.
- **Rhythmic combat finale**: shot on a click track / drum loop so the body motion lands on beat. Capture with audio playback during recording; align beat markers in the NLA strip.

## What this doc is and isn't

- **Is**: a versioned shoot-ordering and rigging-priority artifact that
  evolves with the script. Update when Act 3 lands or a shoot session
  changes a constraint.
- **Isn't**: the screenplay. The script is the writer's IP and lives
  outside this public repo. This file derives only the production
  facts (character × scene × motion type) needed to plan the pipeline
  use.

When Act 3 lands, add its scenes to the cluster table and re-check
whether any new characters / rigs / motion types appear.
