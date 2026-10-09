# Report: what triggers NEC's embedding explosion, and is the render responsible for the pixel results?

**Status: complete, exploratory, not pre-registered** (see PLAN.md, committed before the runs; the analysis script was written after
they started). Branch `explore/pixels-rmsprop`. 62 runs: 30 in Part 1 (3 seeds per variant, 20k steps), 32 in Part 2 (2 seeds).
Tables: [results.md](results.md). Follows [../pixels_rmsprop_probe/REPORT.md](../pixels_rmsprop_probe/REPORT.md), and corrects it (section 5).

## In short

1. **The explosion is set by the size of the constant wall input, not by the optimizer, the frame stack, or the render.** On the same
   network and board, the walls at 1.0 diverge on 3 of 3 seeds, at 0.235 on 0 of 3, and hidden on 0 of 3. The pixel render shows the same
   dose-response on a different network: wall gray 255 diverges 3 of 3, the default 60 and 0 do not.
2. **RMSProp does not reliably prevent it.** With the walls at 1.0, RMSProp 2.5e-4 diverged on 1 of 3 seeds (key norm 27 → 8×10⁵ by 20k,
   2×10⁷ by 100k). My earlier "RMSProp stops it" came from a single seed and is withdrawn.
3. **The render is not what helped.** With RMSProp, NEC on the grid matches NEC on pixels at 7×7 and 13×13, and is a little ahead at 10×10.
   DQN with Adam is *better* on the grid CNN than on pixels at 7×7 and 10×10 (3.86 vs 2.08; 1.67 vs 0.64).
4. **The big RMSProp gain for NEC on pixels was mostly pixel-NEC-with-Adam being unusually bad.** On the grid the gain is smaller (7×7: 0.72 →
   0.99; 10×10: 0.12 → 0.31) and absent at 13×13 (0.11 → 0.12).

## 1. Part 1: what triggers the explosion

NEC, 25×25 `rooms` + bonus 5, held-phase settings, Adam 5e-4 unless noted, 20k steps, seeds 1–3. Diverged = key norm above 100 in any
1,000-step window (it starts near 1). Median key norm over seeds at 20k [min, max]:

| Variant | Diverged | Key norm at 20k |
|---|---|---|
| grid, walls 1.0 (baseline) | **3 of 3** | 1.4×10⁸ [3.6×10⁷, 4.6×10⁸] |
| grid, walls 0.235 | 0 of 3 | 0.47 [0.35, 6.4] |
| grid, walls hidden (0) | 0 of 3 | 0.25 [0.22, 0.30] |
| grid, open map (no walls channel) | 0 of 3 | 0.25 [0.21, 0.25] |
| grid, 4-frame stack | **3 of 3** | 9.1×10⁷ [4.5×10⁷, 2.6×10⁸] |
| grid, walls 0.235 + 4-frame stack | 0 of 3 | 0.30 [0.26, 3.2] |
| grid, RMSProp 2.5e-4, walls 1.0 | **1 of 3** | 1.5 [1.0, 8×10⁵] |
| pixels, wall gray 0 | 0 of 3 | 0.52 [0.44, 0.54] |
| pixels, wall gray 60 (default) | 0 of 3 | 1.7 [0.77, 15] |
| pixels, wall gray 255 | **3 of 3** | 2.4×10⁶ [53, 2.9×10⁷] |

**What this shows:**
- **The magnitude of the constant wall input is the factor.** On the grid, 1.0 diverges and 0.235 does not; in pixels, 255 diverges and 60
  does not. Hiding the walls entirely also works. This is two different networks (DQN-shaped, and the exact Nature CNN) agreeing.
- **The frame stack is not a factor:** stack alone diverges as the baseline does, and the dim walls with the stack are stable.
- **The pixel render's default gray (60 = 0.235) is why the pixel runs were stable,** not the image, the stride, or the network. That
  answers the open question from the pixels report.
- **Not fully clean:** one seed of the dim-walls grid variant and one of the default-pixel variant start with a high key norm (4–22 in the
  first window) and stay bounded. The protection is a margin, not a guarantee. I tested three wall levels per network, so I do not know where the
  threshold lies.
- **Why (my explanation, consistent with the table, not tested directly):** the constant input is the same in every state, so the
  encoder gains a large direction that moves all keys together; NEC's loss only sees distances between keys, so nothing pulls that direction
  back, and Adam's full-size steps keep growing it.

## 2. RMSProp is not a reliable fix on the grid

In the ablation (seeds 1–3), the grid with RMSProp diverged on seed 2 only. In Part 2 (seeds 1, 2) the 25×25 runs show the same seed 2 trajectory
(27 → 5×10⁵ → 2×10⁷ at 100k), as expected from reproducibility. Seed 1 was fine (1.0 throughout) and scored 19.0 fruit per 100 steps, the best NEC result of these sweeps, against 3.5 for the diverged seed. So RMSProp avoided the explosion in 2 of 3 seeds, and the failure is seed-dependent.
The pixel runs with RMSProp and with Adam, at the default dim walls, were bounded in every run.

## 3. Part 2: grid control against pixels

Score = fruit per episode (open boards, second half) or fruit per 100 steps (25×25, steps 50k–100k). Mean of 2 seeds.

| Agent | 7×7 | 10×10 | 13×13 | 25×25 |
|---|---|---|---|---|
| DQN grid, Adam | **3.86** | **1.67** | **0.59** | 14.7 |
| DQN pixels, Adam | 2.08 | 0.64 | 0.47 | **16.0** |
| DQN grid, RMSProp 2.5e-4 | 0.38 | 0.19 | 0.19 | 11.5 |
| DQN pixels, RMSProp 2.5e-4 | 1.03 | 0.54 | 0.24 | 12.1 |
| NEC grid, Adam | 0.72 | 0.12 | 0.11 | 3.3 (diverged) |
| NEC pixels, Adam | 0.28 | 0.13 | 0.11 | 9.7 |
| NEC grid, RMSProp 2.5e-4 | 0.99 | 0.31 | 0.12 | 11.2 (one seed diverged: 19.0, 3.5) |
| NEC pixels, RMSProp 2.5e-4 | 1.00 | 0.23 | 0.15 | 14.7 |

**What this shows:**
- **NEC with RMSProp:** grid ≈ pixels at 7×7 and 13×13; the grid is ahead at 10×10 (0.31 vs 0.23, seed ranges 0.23–0.38 vs 0.22–0.25). The render
  does not explain the NEC results.
- **DQN with Adam is better on the grid CNN than on pixels** on the open boards, by about 2×, and about equal at 25×25. The two networks also
  differ (same-padding strides on a cell grid, against the Nature CNN's valid convolutions on an 84×84 image), so this is render and network together.
  But it says the pixel route did not buy anything for DQN either. In the grid probe, DQN at 7×7 scores 0.64 steering (chance 0.39), the highest
  probe steering of any run so far.
- **RMSProp 2.5e-4 is a poor setting for DQN on the grid** (7×7: 0.38 against 3.86 with Adam). I tried one learning rate, so this says the
  paper-style lr I picked is too small for DQN here, not that RMSProp cannot work.
- **The probe still reads chance for NEC everywhere,** and `decode` stays 0.09–0.16 for all agents on both renders.

## 4. Predictions in PLAN.md

1. *g_base diverges on 3; g_ws0, g_rms and g_open do not:* **mostly held; g_rms failed** (1 of 3 diverged).
2. *g_fs4 diverges like g_base:* **held** (3 of 3).
3. *g_ws235 does not diverge:* **held** (0 of 3).
4. *px_ws425 diverges:* **held** (3 of 3).
5. *Part 2: grid ≈ pixels for RMSProp NEC and Adam DQN:* **half held**: NEC RMSProp yes (7×7, 13×13), Adam DQN no (the grid is clearly ahead on open boards).

## 5. Corrections to earlier statements

- **Pixels report, section 3 and 8, and the diagnosis document, section 2:** "RMSProp stops the explosion" and "the first setting where NEC's encoder trains healthily on a
  CNN" are wrong as stated. RMSProp avoided it in the pixel runs and in 2 of 3 grid seeds. The reliable remedy found so far is a dim or hidden constant
  input. Notes are added to both documents.
- **"RMSProp improves NEC at every size" (my summary after the pixels sweep):** true for 7×7 and 10×10 on both renders, but absent at 13×13 on the grid.
- **Pixels report, "DQN·Nature is a real gain over the MLP":** it is a CNN gain; the grid CNN gains more.

## 6. Limitations

- Three seeds in the ablation, two in the control; one board for the ablation (25×25 `rooms`), 20k steps. The 20k-step key norm shows the explosion but says nothing about
  later behaviour (the control runs to 100k and agree).
- Three wall levels per network; no threshold.
- The grid and pixel agents differ in network as well as in render, so Part 2 cannot separate the two.
- DQN with RMSProp uses a single learning rate.

## 7. What this suggests next (not done)

1. **Fix the input, not the optimizer:** feed NEC a centred or dimmed walls channel (0.235 is enough here), or subtract the constant map. It is one line in the
   env and reproduces the paper's setting better than hiding information. Then rerun the NEC·CNN arm of the encoder-probe experiment, which the divergence invalidated.
2. **Find the threshold:** a sweep of wall level (0.25, 0.5, 0.75, 1.0) on the grid, 3 seeds each. It would show whether the cutoff is smooth.
3. **Test RMSProp's failing seed:** why does seed 2 start at key norm 27? The first-window norm differs between seeds before any divergence, and may predict it.
4. The earlier E-steps still stand: E0 (distance split in the probe), E7 (egocentric control arm), E6 (absolute actions).
