# Post-hoc geometry of the saved agents (exploratory)

**Status: exploratory, computed after the runs from checkpoints** (PROTOCOL.md section 3.5). Not part of the pre-registered analysis. Tables: [geometry.md](geometry.md).
Code: `src/snake_rl/geometry.py` (per checkpoint), `analyze_geometry.py` (aggregation). 250 checkpoints (50 runs x steps 60k, 120k, 180k, 240k, 250k), analysed on CCAD (job 1541, 50 of 50 completed).
**Validation:** the probe recomputed from each saved agent equals the run's own recorded probe at that step in 250 of 250 checkpoints.

## In short

1. **Memory values are not regional.** The 50 nearest stored values are almost as dispersed as the whole memory (coherence 0.9-1.0 across checkpoints, where 1 = no more alike than random rows). The kernel average is therefore close to the memory-wide mean for any state, which flattens Q across actions. This is now measured; before it was only my inference.
2. **"Organised by snake position" does not by itself separate NEC from DQN.** On the agents' own states, nearest embeddings share the head cell more than 10 times as often as chance for **all** agents, DQN included (DQN 0.21, NEC 0.25-0.30, `ec_frozen` 0.56 at 250k; chance 0.01-0.03). My earlier explanation (position-organised embeddings are why NEC fails) is therefore incomplete.
3. **A refined, still untested reading:** NEC's Q is computed from the nearest neighbours, so a geometry dominated by position makes it blind to the food. DQN reads Q out through a learned linear layer, which can use directions of its hidden layer that the nearest-neighbour structure ignores. Consistent with this, the food offset is weakly linearly readable in DQN (`decode` 0.24) and not at all in NEC (0.10, chance).
4. **The near probe is out of distribution for the learned NEC.** Probe states lie 3-4 times farther from memory than stored keys lie from each other (`probe_vs_memory` 3.1-4.1 for the learned arms against 1.5 for `ec_frozen`). The near-probe steering of NEC (chance level) is therefore measured on states it has not stored. The behavioural result stands on its own: NEC's evaluation score (0.8-0.9) is below the frozen embedding's (1.1).

## Measured (means over 10 seeds; see geometry.md for every table)

| Quantity | DQN | `ec_frozen` | NEC 1e-4 |
|---|---|---|---|
| Near-probe steering, 60k to 250k (chance 0.39) | 0.57 to 0.77 | 0.40 to 0.42 | 0.41 to 0.43 |
| Neighbours share the snake placement (`knn_config`), 60k to 250k | 0.40 to 0.33 | 0.60 to 0.60 | 0.59 to 0.55 |
| `decode` of the food offset (chance 0.10) | 0.14 to 0.24 | 0.10 | 0.10 |
| Q within a snake placement over overall Q (`qfood` ratio) | 0.30 to 0.20 | 0.40 to 0.44 | 0.54 to 0.61 |
| Q spread across actions (max - min) | 0.85 to 2.14 | 0.32 to 0.35 | 0.32 to 0.33 |
| Own-state neighbours share the head cell (chance 0.01-0.03), 60k to 250k | 0.77 to 0.21 | 0.66 to 0.56 | 0.47 to 0.25 |
| Own-state neighbours share the food offset, board frame (chance 0.02) | 0.15 to 0.07 | 0.04 to 0.07 | 0.04 to 0.04 |
| Memory coherence, 60k to 250k (1 = no regional structure) | - | 0.90 to 0.94 | 1.00 to 0.93 |
| `probe_vs_memory` (1 = in distribution) | - | 1.5 | 3.4 to 3.1 |

- NEC's Q-values barely separate actions (spread 0.31-0.35, flat over training) while DQN's spread grows about 2.5 times.
- NEC's Q varies with the food within a snake placement relatively more than DQN's does (0.54-0.61 against 0.20-0.30), yet it does not steer: the dependence is there but not aligned with the right action. This fits the earlier result that Q-values are never tied.
- Training moves the learned NEC embedding away from grouping by head cell (0.47 to 0.25 for 1e-4) without making it group by food offset (about 0.04, chance 0.02).

## Inferred, not tested

- That the nearest-neighbour readout (NEC) versus a learned linear readout (DQN) explains why a position-dominated geometry hurts NEC and not DQN. A direct test would swap NEC's readout, or give DQN a nearest-neighbour readout on its own hidden layer.
- That the oracle key (milestone issue #12) would make NEC steer.

## Limits

- Exploratory: nothing here was specified before the runs. 10 seeds per condition, one board (10x10), one render, absolute actions.
- On-policy statistics come from 20 episodes at epsilon 0.05 per checkpoint (up to 3,000 states) and exclude same-episode neighbours; the chance rate is computed over other-episode states.
- Purity is not correctness. A label can be grouped by the embedding without the value being right, and a nearest-neighbour structure that ignores a label can coexist with a linear readout that uses it.
