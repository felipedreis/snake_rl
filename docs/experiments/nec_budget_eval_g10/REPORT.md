# Report: does a paper-sized budget let NEC steer? (pre-registered)

**Status: complete.** Pre-registered in [PROTOCOL.md](PROTOCOL.md), frozen at commit `71a0625` before the first run (hashes in `FROZEN.sha256`; all four files matched
after the runs; later edits, such as the planned cleanup, will change the working-tree files, so verify against that commit with
`git show 71a0625:<path> | shasum -a 256`). Branch `explore/pixels-rmsprop`. 60 runs on CCAD (job 1475, from 10 Oct 12:44 to 11 Oct 01:18 cluster time, 60 of 60 completed, none failed, 12.6 h of wall clock, 230 CPU-hours).
Generated outputs: [results.md](results.md), [per_run.csv](per_run.csv), [learning_curves.png](learning_curves.png), [steering_by_distance.png](steering_by_distance.png).
Raw results: `results/exp_nec_budget_eval_g10/` (checkpoints still downloading, see Deviations).

## In short

1. **All three confirmatory hypotheses are not supported, as predicted.** At 250k steps no NEC arm beats `ec_frozen`, NEC is far below DQN, and NEC's near-probe steering is 0.43 (chance 0.39).
2. **The budget helps DQN a great deal and NEC not at all.** DQN's evaluation score rises from 2.08 (60k) to 9.03 (250k); NEC stays at 0.7-0.96 throughout.
3. **The learned embedding makes NEC worse, not better:** every NEC arm scores 0.22-0.31 below `ec_frozen` (intervals exclude 0), and its evaluation episodes are half as long (53-63 steps against 129).
4. **Tied Q-values do not explain the chance-level steering:** the all-tied share is 0.00 in every NEC and `ec_frozen` probe. The memory gives distinct values, and they do not point to the food.
5. **No early NEC advantage on this board:** NEC is level with DQN at 10k-20k steps and DQN is ahead from 30k.

## Setup (from the protocol)

10x10 `open`, pixels (wall gray 60, 4-frame stack), absolute actions, 250k steps, seeds 11-20, default epsilon schedule. Evaluation: 20 episodes at epsilon 0.05 every 10k steps on a
separate environment. Conditions: `random`, `dqn_naturecnn` (Adam 5e-4), `ec_frozen_naturecnn`, `nec_naturecnn` with RMSProp 7.92e-6, 3e-5 and 1e-4. Primary outcome O1: the mean evaluation
score over the evaluations at 230k, 240k and 250k, per run. Agents were saved every 60k steps.

## Results

**O1** (mean over 10 seeds; s.d. in brackets; the evaluation episode length is at 250k):

| Condition | O1 | Evaluation episode length | Near-probe steering at 250k |
|---|---|---|---|
| random | 0.15 (0.06) | 15 | - |
| DQN | **8.82** (0.73) | 108 | **0.77** |
| `ec_frozen` | 1.09 (0.18) | 128 | 0.42 |
| NEC 7.92e-6 | 0.82 (0.10) | 62 | 0.43 |
| NEC 3e-5 | 0.78 (0.13) | 54 | 0.41 |
| NEC 1e-4 | 0.88 (0.18) | 51 | 0.43 |

**Evaluation score by training step** (mean over seeds):

| Step | random | DQN | `ec_frozen` | NEC 7.92e-6 | NEC 3e-5 | NEC 1e-4 |
|---|---|---|---|---|---|---|
| 10k | 0.17 | 0.50 | 0.52 | 0.46 | 0.52 | 0.39 |
| 20k | 0.15 | 0.64 | 0.58 | 0.61 | 0.53 | 0.57 |
| 30k | 0.14 | 0.88 | 0.51 | 0.65 | 0.47 | 0.60 |
| 60k | 0.11 | 2.08 | 0.70 | 0.84 | 0.82 | 0.69 |
| 120k | 0.17 | 6.66 | 0.79 | 0.81 | 0.83 | 0.85 |
| 180k | 0.12 | 8.35 | 0.85 | 0.85 | 0.75 | 0.96 |
| 250k | 0.17 | 9.03 | 1.10 | 0.83 | 0.84 | 0.92 |

Near-probe steering by step for DQN: 0.57 (60k), 0.72 (120k), 0.76 (180k), 0.77 (250k); for every NEC arm and `ec_frozen` it stays at 0.40-0.43 throughout.

**Steering by walkable distance to the food, end of training** (chance in brackets):

| Condition | d=1 | d=2 | d=3 | d=4 | d=5 | d=6 |
|---|---|---|---|---|---|---|
| DQN | 0.80 (0.33) | 0.76 (0.43) | 0.75 (0.44) | 0.75 (0.45) | 0.73 (0.46) | 0.76 (0.46) |
| `ec_frozen` | 0.35 (0.33) | 0.45 (0.43) | 0.50 (0.44) | 0.53 (0.45) | 0.56 (0.46) | 0.58 (0.46) |
| NEC 1e-4 | 0.36 (0.33) | 0.45 (0.43) | 0.49 (0.44) | 0.50 (0.45) | 0.52 (0.46) | 0.51 (0.46) |

Embedding geometry on the near probe at 250k, seed 11 (`knn_config` / `knn_offset` / `decode`; chance of `decode` about 0.10): DQN 0.29 / 0.16 / 0.25; NEC 1e-4 0.54 / 0.06 / 0.11.
Maximum key norm in NEC runs: 1.29-1.30 (no explosion). Maximum share of DND appends taken by one action: 0.25-0.29 (no single-action collapse).

## Hypotheses against the protocol

| | Definition (protocol section 1) | Result | Verdict | Predicted |
|---|---|---|---|---|
| H1, NEC 7.92e-6 | difference to `ec_frozen` >= +0.25, 98.3% interval excludes 0 | -0.28 [-0.42, -0.14] | not supported | not supported |
| H1, NEC 3e-5 | same | -0.31 [-0.47, -0.16] | not supported | not supported |
| H1, NEC 1e-4 | same | -0.22 [-0.40, -0.04] | not supported | not supported |
| H2 | NEC 1e-4 minus DQN, 95% interval above 0 | -7.95 [-8.39, -7.51] | not supported | not supported |
| H3 | NEC 1e-4 near-probe steering >= 0.50 | 0.43 (0.41-0.44 per seed) | not supported | not supported |

H1 was one-sided, so "not supported" is the verdict, but the intervals show the differences are negative, not merely missing.

## What this means

**Measured**
- DQN learns steadily with the budget on this task (about 58 times the random score at 250k, steering at 0.73-0.80 at every distance from 1 to 6 steps). The earlier worry that DQN's 0.60 steering at 60k was marginal was about an agent still early in training.
- NEC does not improve with the budget, up to 250k steps, which is the paper's first reported point (1M frames) in agent steps. It also does not react to adjacent food: steering at distance 1 is 0.34-0.37 (chance 0.33).
- Training the embedding made NEC slightly worse than leaving it random, and its episodes end sooner.
- The Q-values are never tied. Whatever the memory returns is distinct but not aligned with the food.

**Inferred, not tested here**
- The budget is not the limit on NEC here. This makes the representation, or the signal that trains it, the likelier bottleneck. The oracle-key test (milestone issue 8) is the direct check.
- Why the learned embedding is worse than the frozen one, and why its episodes are shorter, is not explained by these data.

## Deviations from the protocol

1. **Collection:** the `collect.yml` pull of the checkpoints (about 12.5 GB over the VPN) hit its time limit after about 1.1 GB. The 60 result JSON and `.train.jsonl` files were fetched with a direct `rsync`, which is all that `analyze.py` reads. No number in this report depends on a checkpoint. The checkpoint download is being restarted.
2. **Nothing else changed.** The spec, the analysis script and the probe are the frozen versions (hashes verified after the runs); no run failed or was rerun; the random runs saved no agents, as planned.
3. The random baseline under absolute actions is now measured (0.15), replacing the earlier relative-action figure (0.17) from another experiment.

## Limits

- One board (10x10 `open`), one render, absolute actions. The paper's comparisons are at much larger scale (its 10M-frame table is 2.5M agent steps).
- Kept at defaults, as pre-registered: DND capacity 2e4 (paper 5e5), N = 50 (paper 100), key size 32 (a secondary source says 128), replay 2e4, exploration 1 to 0.02 over 5k steps with training from the start. They may matter; this experiment does not say.
- Three NEC learning rates; none above 1e-4.
- Near-probe numbers are for one probe set (food within 2 steps); the far probe covers up to 6.

## Next

See `docs/milestones/nec-where-it-breaks.md`: the oracle-key test, the board-size scaling study, the complexity ramp, the egocentric control arm, and the paper-faithfulness factorial.
