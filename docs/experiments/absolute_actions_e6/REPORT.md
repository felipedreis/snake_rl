# Report: absolute actions on the pixel render (E6)

**Status: complete, exploratory.** Branch `explore/pixels-rmsprop`. Plan: [PLAN.md](PLAN.md), written before the runs. 12 runs of
`dqn_naturecnn` (`--render pixels`, Adam defaults), 60k steps, boards 7x7 and 10x10 `open`, seeds 1-3, `--actions relative` against
`--actions absolute`. Raw results: `results/exp_absolute_actions/`.

## In short

1. **Absolute actions roughly double DQN's score on both boards, on every seed.** The worst absolute seed beats the best relative seed on each board.
2. **Probe steering agrees:** relative DQN on 10x10 steers at about chance; absolute is clearly above it.
3. **Only DQN, two small boards, 3 seeds.** NEC and 25x25 are not tested here.

## Results

Fruit per episode, second half of training (steps above 30k). Seeds 1, 2, 3 in brackets.

| Board | Relative | Absolute | Ratio |
|---|---|---|---|
| 7x7 | 2.15 (1.63, 2.52, 2.29) | 4.35 (5.10, 3.60, 4.36) | 2.0x |
| 10x10 | 0.72 (0.59, 0.69, 0.87) | 1.83 (2.56, 1.53, 1.40) | 2.5x |

Greedy steering on the fixed probe set at the last probe (mean over seeds; chance rate in brackets):

| Board | Relative | Absolute |
|---|---|---|
| 7x7 | 0.57 (0.39) | 0.63 (0.39) |
| 10x10 | 0.46 (0.40) | 0.58 (0.39) |

## Predictions in PLAN.md

1. *Absolute ahead on both boards, with the larger gap at 10x10:* **half held.** Ahead on both, on every seed. The gap is larger in ratio at 10x10 (2.5x against 2.0x) and smaller in fruit (+1.1 against +2.2).
2. *The gap is smaller at 7x7:* **not clearly held**, for the same reason: it depends on whether the gap is read as a ratio or a difference.
3. *Not predicted:* whether absolute actions make the frame stack unnecessary. Not tested.

## Limits

- DQN only, with 3 seeds, 60k steps, and two open boards. 25x25 (held phase, `rooms`, bonus) was not run.
- The grid-CNN DQN with relative actions scored 3.86 (7x7) and 1.67 (10x10) in the grid control, close to absolute-action pixels here (4.35, 1.83). That mixes network and render, so it is not a clean comparison.
- Absolute actions are a departure from the relative actions used so far, and from the snake's own heading. A reversal keeps going straight.

## Next

NEC under absolute actions; the RMSProp learning-rate sweep under the chosen action space; a 25x25 held-phase run; the map curriculum.
