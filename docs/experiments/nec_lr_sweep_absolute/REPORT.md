# Report: NEC learning-rate sweep with absolute actions, on pixels (E3, adapted)

**Status: complete, exploratory, not pre-registered** (the design is the spec `ansible/experiments/nec_lr_sweep_absolute.yml`, written
before the runs). Branch `explore/pixels-rmsprop`. Follows [../absolute_actions_e6/REPORT.md](../absolute_actions_e6/REPORT.md) and E3 in
`docs/DIAGNOSIS_AND_NEXT_STEPS.md`. 24 runs on CCAD (job 1394), 60k steps each, 3 seeds (1-3); the longest took 1.7 h.
Raw results: `results/exp_nec_lr_sweep_absolute/` (on Hugging Face).

## In short

1. **Learning the embedding adds nothing at this budget.** The best NEC (RMSProp 1e-4, 0.83 fruit per episode) is inside the seed spread of NEC with a frozen embedding (0.78; 0.60-0.93).
2. **Absolute actions help NEC as they helped DQN:** 0.64 against 0.26 at RMSProp 2.5e-4, on every seed.
3. **NEC does not steer in any setting.** Probe steering is 0.40-0.43 (chance 0.39), and the probe's `decode` is 0.09 everywhere. DQN with absolute actions steers at 0.58 on this board.
4. **No embedding explosion in any run** (largest key norm 1.02-1.07). Expected: open board, dim pixel walls.

## Setup

- Board 10x10 `open`, `--render pixels` (wall gray 60, 4-frame stack), agents `nec_naturecnn` / `ec_frozen_naturecnn`, 60k steps, default epsilon schedule, `--probe-every 10000`.
- NEC with absolute actions: RMSProp lr 7.92e-6 (the Atari NEC value), 3e-5, 1e-4, 2.5e-4, 5e-4, and the Adam default (5e-4).
- `ec_frozen` with absolute actions, once per seed (its learning rate has no effect: nothing is trained).
- NEC with relative actions at RMSProp 2.5e-4, as the action-space reference.

## Results

Fruit per episode in the second half of training (steps above 30k); seeds 1, 2, 3 in brackets. Steering is the mean over seeds at the last probe
(chance 0.39-0.40).

| Agent / setting | Fruit | Seeds | Steering |
|---|---|---|---|
| `ec_frozen`, absolute | 0.78 | 0.82, 0.60, 0.93 | 0.39 |
| NEC RMSProp 7.92e-6, absolute | 0.77 | 0.83, 0.69, 0.80 | 0.41 |
| NEC RMSProp 3e-5, absolute | 0.69 | 0.67, 0.71, 0.68 | 0.43 |
| NEC RMSProp 1e-4, absolute | **0.83** | 0.73, 0.99, 0.77 | 0.42 |
| NEC RMSProp 2.5e-4, absolute | 0.64 | 0.70, 0.52, 0.71 | 0.40 |
| NEC RMSProp 5e-4, absolute | 0.63 | 0.64, 0.55, 0.70 | 0.41 |
| NEC Adam 5e-4 (default), absolute | 0.51 | 0.44, 0.68, 0.41 | 0.40 |
| NEC RMSProp 2.5e-4, relative | 0.26 | 0.18, 0.30, 0.29 | 0.41 |

For comparison, DQN with the Nature CNN and absolute actions scored 1.83 on this board (E6; 0.72 with relative actions).

**What this shows:**
- **No learning rate makes NEC beat the frozen embedding.** The best setting overlaps `ec_frozen`'s seed range, and the lowest learning rates (7.92e-6, 3e-5) behave like it, as they should when little is learned. This is the outcome E3 named as "itself a finding".
- **Higher learning rates look somewhat worse** (Adam and RMSProp >= 2.5e-4: 0.51-0.64) than lower ones (0.69-0.83), but the seed ranges overlap, and three seeds cannot rank the settings finely.
- **Absolute actions help NEC too,** by about 2.5x at the one setting tested with both. Even so, the best NEC is less than half of DQN with the same actions.
- **The representation is not learning what the probe measures.** Steering stays at chance and `decode` at 0.09 for all NEC variants, as in the earlier pixel and grid reports.

## Limits

- Three seeds, one board, 60k steps, one DND capacity. Differences between learning rates are within seed noise.
- **No random-agent baseline was run here,** so I cannot say how far above random these NEC scores are. Earlier notes put NEC near random at 10x10.
- The relative-actions reference is one setting only (RMSProp 2.5e-4), not a sweep.
- No explosion appeared, but this board has no walls. It says nothing about the 25x25 `rooms` case.

## Next

A random-agent baseline on this board; the same sweep at 25x25 (held phase) and with more seeds; the map curriculum with a dim wall level; E7 (egocentric
observation) as the control arm, since the probe says the embedding does not separate the food offset in any variant.
