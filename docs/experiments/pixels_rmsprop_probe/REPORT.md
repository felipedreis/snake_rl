# Report: Atari-style pixels + RMSProp, across board sizes, against MFEC

**Status: complete, exploratory, not pre-registered.** Branch `explore/pixels-rmsprop`. Two seeds per cell in the main sweep
and one in the validation runs, so read patterns, not small differences. Plan and predictions: [PLAN.md](PLAN.md), written
before the runs. Tables: [results.md](results.md) (main sweep), [results_validate.md](results_validate.md) and
[steering_check.txt](steering_check.txt) (validation).

## In short

1. **Pixels + DQN's exact network do not make steering position-general.** At 25×25 the best on-policy steering is still an
   island around the start cell (NEC 0.86 within 2 cells, 0.43 beyond 6; chance about 0.37), the same shape as NEC·MLP's in the
   encoder-probe report. Probe `decode` of the food offset stays at 0.08–0.16 (chance 0.10) at every size.
2. **RMSProp fixes NEC's learning, not its representation.** NEC does better with RMSProp 2.5e-4 than with Adam at every size
   (7×7: 1.00 vs 0.28 fruit per episode; 25×25: 14.7 vs 9.7 per 100 steps). But a frozen random encoder does as well or better
   from 10×10 up, so the learned embedding adds nothing measurable there. DQN does the opposite: Adam beats RMSProp.
3. **No embedding divergence appeared in any pixel run**, with Adam or RMSProp. The walls are still drawn
   into the image (dimmer), so this does not confirm the walls-channel explanation either; it only says the divergence does not occur in this setting.
4. **MFEC is not better on the grayscale board.** The grid beats pixels at all four sizes (1.87 vs 1.59, 0.70 vs 0.63,
   0.40 vs 0.37, 12.5 vs 10.8). The differences are small against the seed spread.
5. **My first pixel probe was wrong, and the fix matters.** It showed four identical frames. With a moving snake, DQN's probe
   steering on 7×7 goes from 0.42 to 0.55, and now agrees with its steering on its own states (0.53).

## 1. What was run

- **Main sweep** (72 runs: 64 agent runs + 8 Random): open 7×7, 10×10, 13×13 (60k steps), and 25×25 `rooms` + bonus 5 with
  the held food curriculum (100k steps). Agents on the pixel render: NEC·Nature and DQN·Nature (Adam 5e-4; RMSProp 2.5e-4;
  NEC also RMSProp 1e-3), frozen EC·Nature, MFEC on pixels, and MFEC on the grid as the reference.
- **Validation runs** (6 runs, seed 1, saved agents): DQN·Nature Adam and NEC·Nature RMSProp 2.5e-4 at 7×7, 10×10 and 25×25,
  with the corrected probe, plus an on-policy steering check (`steering_check.py`).
- The validation runs reproduce the main sweep's seed-1 numbers exactly (for example DQN 7×7 1.63, 25×25 16.35), as expected
  from bit-for-bit reproducibility with the probe on or off.

## 2. Learning

Fruit per episode over the second half of training on open boards; fruit per 100 training steps over steps 50k–100k at 25×25
(not the 100k–250k window of earlier reports, so not comparable with them). Mean of 2 seeds.

| Agent | 7×7 | 10×10 | 13×13 | 25×25 rooms |
|---|---|---|---|---|
| Random | 0.20 | 0.15 | 0.17 | 6.6 |
| MFEC, grid | 1.87 | 0.70 | 0.40 | 12.5 |
| MFEC, pixels | 1.59 | 0.63 | 0.37 | 10.8 |
| EC frozen, Nature | 0.80 | 0.42 | 0.25 | 15.1 |
| DQN·Nature, Adam 5e-4 | **2.08** | **0.64** | **0.47** | **16.0** |
| DQN·Nature, RMSProp 2.5e-4 | 1.03 | 0.54 | 0.24 | 12.1 |
| NEC·Nature, Adam 5e-4 | 0.28 | 0.13 | 0.11 | 9.7 |
| NEC·Nature, RMSProp 2.5e-4 | 1.00 | 0.23 | 0.15 | 14.7 |
| NEC·Nature, RMSProp 1e-3 | 0.79 | 0.19 | 0.09 | 13.2 |

**What this shows:**
- **DQN·Nature with Adam is the best agent at every size.** On 7×7 it doubles the MLP DQN of the earlier study (2.08 vs 1.04;
  60k vs 40k steps), so the CNN on pixels is a real gain for DQN.
- **Everything beats Random on 25×25 by about 1.5–2.5×.** That is the held-phase task (food within 2 steps), which also rewards
  survival and wandering near the start.
- **RMSProp helps NEC at every size** (about 3.5× at 7×7), and it removes the dead-ReLU problem (0–3% dead at the end, against
  11–35% with Adam). NEC with RMSProp is the best NEC variant, but the learning rate matters less than the choice of optimizer
  (2.5e-4 vs 1e-3: similar).
- **A learned NEC beats a frozen encoder only at 7×7** (1.00 vs 0.80). At 10×10 and 13×13 the frozen encoder is clearly ahead (0.42
  vs 0.23; 0.25 vs 0.15), and at 25×25 they tie (15.1 vs 14.7). So rung 3 does not beat rung 2 beyond the smallest board, whatever
  the optimizer, which echoes the 10×10 result with Adam in the earlier study.
- **DQN with RMSProp 2.5e-4 learns slower than with Adam** (update-to-weight ratio 1–4 × 10⁻⁴, against about 1.5 × 10⁻³ with Adam; 1–3% dead ReLUs). At
  this step budget the lr is probably too small for DQN. I tried one value, so this is a statement about this setting.
- **MFEC on the grid is still hard to beat on small boards.** At 7×7 only DQN·Nature (Adam) is above it.

## 3. Encoder health

- **Key norm** (NEC; first → last 1,000-step window): RMSProp stays at 0.8–1.6 → 0.6–1.4 at all four sizes. Adam is also bounded
  in every pixel run (0.6–2.2 → 0.5–5.1), except that seed 2 at 25×25 with Adam is already at 22 in its first window and ends at 17.
  Nothing diverges. In the earlier grid runs, the same Adam setting reached 10⁶–10⁹ on the CNN encoders.
- **Why not diverging does not isolate the walls channel.** The pixel render does not remove the walls: they are drawn into the
  one grayscale image at level 60 (0.24 on the 0-1 scale), repeated in all 4 stacked frames, and the margin is drawn the same way.
  So a constant wall pattern is still in the input, only dimmer and no longer its own channel. The render also changed the
  network (valid convolutions, strides, 84×84 input) and added the frame stack. So these runs show that NEC with Adam is stable
  *in this setting*, which weakens the claim that a constant walls input alone explains the grid divergence (at full strength,
  as its own channel, zeroing it fixed that case; `docs/DIAGNOSIS_AND_NEXT_STEPS.md` section 2). The factor that matters may be the
  scale of the constant input, or the network, or both, and I have not tested which.
- **Dead ReLUs:** NEC with Adam ends with 11–35% of its last hidden layer dead, with RMSProp 0–3%.

## 4. What the encoder represents (probe), and a probe bug

**The bug.** In the main sweep, the pixel probe built each probe state as four identical frames (a snake that has never
moved). The agents never see that in training, so those probe columns in [results.md](results.md) are out of distribution and
are superseded. The fix (commit `70818c7`) builds the stack as a snake that has moved straight in, as far as the board allows.

**Validation (seed 1, corrected probe, and steering on the agent's own states).** Steering is P(greedy action lies on a shortest
path to the food); `decode` and `knn_offset` are as in `probe.py`.

| Run | probe steer (chance) | probe decode (chance 0.10) | own-state steer (chance) | greedy fruit / 100 steps |
|---|---|---|---|---|
| DQN 7×7 | 0.55 (0.39) | 0.14 | 0.53 (0.46) | 1.1 |
| NEC 7×7 | 0.41 (0.39) | 0.11 | 0.56 (0.46) | 2.1 |
| DQN 10×10 | 0.46 (0.40) | 0.13 | 0.51 (0.48) | 0.2 |
| NEC 10×10 | 0.41 (0.40) | 0.10 | 0.51 (0.46) | 0.1 |
| DQN 25×25 held | 0.41 (0.40) | 0.11 | 0.54 (0.34) | 13.9 |
| NEC 25×25 held | 0.41 (0.40) | 0.11 | 0.59 (0.37) | 19.1 |

**By head distance from the start cell, on the agent's own states (25×25 held phase):**

| Distance | 0–2 | 3–5 | 6–9 | 10+ |
|---|---|---|---|---|
| NEC·Nature, RMSProp | **0.86** | 0.60 | 0.43 | 0.43 |
| DQN·Nature, Adam | 0.64 | 0.56 | 0.34 | 0.42 |

**Reading:**
- **The corrected probe agrees with on-policy steering where there is signal** (DQN 7×7: 0.55 vs 0.53). It is not simply broken,
  and I would use it as it is now. NEC's probe steering is at chance even where its own-state steering is above chance (7×7:
  0.41 vs 0.56), so it still misses NEC's local skill, as the encoder-probe report found.
- **Steering is modest everywhere and local.** On open boards the agents barely steer: at 10×10, greedy play eats 0.1–0.2 fruit
  per 100 steps (they loop until the idle limit), so the 0.5–0.6 training scores come from ε-exploration and survival, not from
  heading for food.
- **The 25×25 shape is the old one.** NEC·Nature's island (0.86 at 0–2 cells, 0.43 beyond 6) matches NEC·MLP's (0.91, 0.42). The
  strided 7×7 map and 4-frame stack did not make the skill position-general.
- **The embedding does not encode the food offset at any size:** `decode` 0.08–0.16 and `knn_offset` about 0.1 or less across all runs, even where agents learn.

## 5. MFEC on the grayscale board

| | 7×7 | 10×10 | 13×13 | 25×25 |
|---|---|---|---|---|
| MFEC, grid | 1.87 (1.75, 1.99) | 0.70 (0.69, 0.70) | 0.40 (0.59, 0.21) | 12.5 (13.8, 11.2) |
| MFEC, pixels | 1.59 (1.68, 1.50) | 0.63 (0.67, 0.58) | 0.37 (0.37, 0.37) | 10.8 (8.9, 12.6) |

The grid is ahead or equal at all four sizes, so the hunch that grayscale might help MFEC is not supported here. The 13×13 and
25×25 grid seeds overlap the pixel seeds, so the gap is only clear at 7×7. A plausible reason, not tested, is that the 4-frame
stack makes exact repeats of a state rarer, and exact repeats are where MFEC does best; MFEC's random projection of 28,224
numbers also weights the board by pixel area rather than by cell.

## 6. Predictions in PLAN.md

1. *Key norm within 10× of start with RMSProp, Adam possibly fine:* **held** (see section 3, one outlier window for Adam).
2. *NEC at least as good as frozen EC at 7×7 and 10×10:* **half held**: 7×7 yes (1.00 vs 0.80), 10×10 no (0.23 vs 0.42).
3. *MFEC on pixels no better than on the grid:* **held**.
4. *Probe `decode` near chance on 25×25:* **held**, and also at the smaller sizes.

## 7. Limitations

- Two seeds per cell, one in the validation. NEC's 13×13 values range 0.07–0.22 across seeds.
- Two RMSProp learning rates for NEC and one for DQN; RMSProp's ε is added outside the root (PyTorch's convention), whereas
  TensorFlow's adds it inside, which would make steps even smaller for the same lr. Whether the NEC paper's setup matches is unknown.
- The pixel render has no action repeat and no max over frames (nothing flickers), so only the image size, the grayscale
  levels, the frame stack and the network follow Atari. The key size (32) is ours.
- Learning scores on open boards are mostly set by exploration (greedy play barely eats), so differences of a few tenths between
  agents say little about steering.
- The 25×25 window (steps 50k–100k, food within 2 steps) is not the one used in earlier reports.

> **Correction (2026-10-09, after `../grid_ablation_rmsprop_control/REPORT.md`):** two statements in this report are withdrawn or narrowed.
> (1) "RMSProp is the first setting where NEC's encoder trains healthily on a CNN" and the claim that RMSProp fixes the explosion: on the grid with
> the walls at 1.0, RMSProp diverged on 1 of 3 seeds. The pixel runs were stable because the pixel render draws walls at a dim gray (60/255); a
> wall gray of 255 diverges 3 of 3 even with the Nature CNN. (2) The RMSProp gain for NEC (about 3.5× at 7×7) was mostly pixel-NEC-with-Adam being
> unusually bad; on the grid it is 0.72 → 0.99 at 7×7 and absent at 13×13. Pixels also do not help DQN: the grid CNN is better on open boards.

## 8. Where this leaves the diagnosis document

- **E2 (RMSProp): done.** It is the right optimizer for NEC here (a learning rate of 2.5e-4), and it is the first setting in this
  project where NEC's encoder trains healthily on a CNN.
- **E9 (pixels + Nature network): done.** It is a stronger DQN, not a fix for position-bound skill.
- **E3/E5 are now mostly answered for this encoder:** NEC does not beat a frozen encoder past 7×7, and the strided CNN does not
  carry steering across positions. What remains untested from the document is what removes the position dependence: E6 (absolute
  actions), E7 (an egocentric control arm) and E8 (one factor at a time).
- **Suggested next steps, in order:** (1) E0 properly (distance split inside `probe.py`, on-policy), since this report needed it
  by hand; (2) E7, the egocentric control arm on pixels, which is now cheap; (3) a small learning-rate sweep for DQN with RMSProp
  (one value is not enough to say RMSProp is worse for DQN).
