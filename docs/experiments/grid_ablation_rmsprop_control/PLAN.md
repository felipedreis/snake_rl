# Exploration: which factor stops NEC's embedding explosion, and is the render responsible for the pixel results?

**Status: exploratory, not pre-registered.** Branch `explore/pixels-rmsprop`. It follows `../pixels_rmsprop_probe/REPORT.md`, whose pixel runs
changed several things at once (render, network, 4-frame stack, dimmer walls) and had no grid control. Two parts.

## Part 1: what triggers the explosion? (NEC, 25×25 `rooms` + bonus 5, held-phase settings, 20k steps, seeds 1–3, Adam 5e-4 unless noted)
Divergence = the training key norm exceeds 100 in any 1,000-step window (it starts near 1; the grid baseline reached 10³–10⁹).

| Variant | Agent / render | Changed from the baseline |
|---|---|---|
| g_base | `nec_dqncnn`, grid | nothing (the known diverging case) |
| g_ws235 | grid | walls channel at 0.235 instead of 1.0 (the pixel render's wall level) |
| g_ws0 | grid | walls channel at 0 (hidden; still lethal) |
| g_fs4 | grid | last 4 observations stacked |
| g_ws235_fs4 | grid | both of the above |
| g_rms | grid | RMSProp 2.5e-4 |
| g_open | grid | `open` map (no walls channel at all), bonus 5 |
| px_ws0 / px_ws1 / px_ws425 | `nec_naturecnn`, pixels | wall gray level 0 / 60 (default) / 255 |

## Part 2: RMSProp control (grid vs pixels)
`nec_dqncnn` and `dqn_dqncnn` on the grid render, each with Adam 5e-4 and RMSProp 2.5e-4, on the same boards, steps and seeds (1, 2) as the
pixel sweep (open 7/10/13 at 60k; 25×25 held phase at 100k), probe included. Compared with the pixel agents of the sweep.

## Predictions (written before the runs)
1. g_base diverges on all 3 seeds; g_ws0, g_rms and g_open do not.
2. g_fs4 diverges like g_base (stacking is not the cause).
3. g_ws235 does not diverge (the lower wall level is the factor that protected the pixel runs). Moderate confidence.
4. px_ws425 diverges: a bright constant input is enough, even with the Nature network. Low confidence: the network also differs.
5. Part 2: grid RMSProp NEC is within the seed spread of pixel RMSProp NEC at every size, and grid Adam DQN matches pixel Adam DQN. That
   would mean the render is not responsible for the pixel results. If the grid runs are clearly better or worse, the render matters.

Everything below the line is written after the runs, in `REPORT.md`.
