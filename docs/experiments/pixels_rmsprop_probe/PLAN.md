# Exploration: Atari-style pixels + RMSProp, across board sizes, against MFEC

**Status: exploratory, not pre-registered.** Branch `explore/pixels-rmsprop`.
It follows the user's hunch from `docs/DIAGNOSIS_AND_NEXT_STEPS.md` (E9, then
E2, then a size sweep with the probe and MFEC on the grayscale board). It is
not the document's step order, and nothing here replaces that plan: if this
fails, the document's E0–E8 still stand.

## What changed in the code (all off by default, RNG-neutral)
- `--render pixels`: the board as an 84×84 grayscale image, 4-frame stack;
  `*_naturecnn` agents use DQN's exact network.
- `--opt rmsprop --lr F`: RMSProp (rho 0.95, eps 1e-2, global-norm clip 10),
  run settings stored and tagged.

## Design
Every run is one seed of one agent, with the original ε schedule (1 → 0.02 over 5k) on open boards and the held
curriculum with ε 1 → 0.05 over 50k on 25×25. Seeds 1 and 2. Representation probe every 10k steps (open) or 25k (25×25).

| Phase | Board | Steps | Agents (all on pixels unless noted) |
|---|---|---|---|
| A | open 7×7, 10×10, 13×13 | 60k | NEC·Nature with Adam 5e-4 / RMSProp 2.5e-4 / RMSProp 1e-3; DQN·Nature with Adam 5e-4 / RMSProp 2.5e-4; frozen-EC·Nature; MFEC on pixels; MFEC on the grid (reference) |
| B | 25×25 `rooms`, bonus 5, curriculum `2:250000:600000` + relocation | 100k (held phase) | same eight |

## Read-outs
- **Learning:** fruit per episode over the second half (open boards); training
  fruit per 100 steps in 100k–… (25×25).
- **Encoder health** (telemetry): key norm at the end against its start, dead
  ReLUs.
- **Encoder content** (probe, last read): `steer`, `decode`, `knn_offset`,
  `food_ratio` (see `probe.py`).
- **MFEC:** does the grayscale board help or hurt it compared with the grid?

## Predictions (written before the runs)
1. RMSProp keeps NEC·Nature's key norm within 10× of its start at every size;
   Adam may not (the walls channel is gone in the pixel render, where walls are
   just gray pixels, so Adam may also survive).
2. On open boards NEC with a healthy optimizer is at least as good as frozen EC
   at 7×7 and 10×10 (rung 3 ≥ rung 2).
3. MFEC on pixels is no better than on the grid on 7×7 (the frame stack makes
   exact repeats rarer), and may be worse as the board grows.
4. On 25×25, probe `decode` stays near chance for every agent unless the Nature
   stack carries the position-independence the document predicts (E5); this is
   the number that matters for the overview's open question.

*Erratum (after the runs, prediction 1 left as written):* "the walls channel is gone" was inaccurate. The pixel render draws the
walls into the image at a dim gray (level 60, repeated in all 4 frames); only the separate channel is gone. See REPORT.md section 3.

Everything below the line is written after the runs, in `REPORT.md`.
