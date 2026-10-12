# Post-hoc: what the encoders learned on Ms. Pac-Man (exploratory)

**Status: post-hoc, exploratory, not pre-registered.** Follows [../REPORT.md](../REPORT.md). Tool: `python -m snake_rl.geometry_pacman` (new; the Pac-Man twin of
`geometry.py`), aggregation `analyze_geometry.py`, table [geometry.md](geometry.md). Runs: the 15 final checkpoints (100k steps) of `dqn_naturecnn`,
`ec_frozen_naturecnn` and `nec_naturecnn` (RMSProp 1e-4), seeds 1-5, 3000 states each from 100 episodes at epsilon 0.05 on the agent's own policy.

## Question
The ladder showed NEC (learned encoder) scoring no better than `ec_frozen` (random frozen encoder). Does the learned encoder differ at all in its geometry?

## Measures (all on states the agent visits; neighbours always come from OTHER episodes, since consecutive frames are near-duplicates)
- **Label purity:** the share of a state's 10 nearest neighbours in the embedding that share a label: Pac-Man's cell, the cell of the nearest pellet, the offset to the nearest ghost, whether a power pellet is active. Chance in brackets.
- **Action purity:** the same for "is this action good", with good = not ending next to a live ghost and closer to a pellet (50 neighbours).
- **Greedy steering:** how often the greedy action is good. **Reference:** the same purities with the raw 4x84x84 pixels as the "embedding".
- **Memory** (NEC, `ec_frozen`): coherence = s.d. of the 50 neighbours' stored values over s.d. of all stored values (1 = neighbours no more alike in value than random rows; lower = values vary smoothly); probe vs memory = distance from visited states to the nearest stored key over the keys' own nearest-neighbour distance (1 = visited states look like stored ones).

## Results (mean over 5 seeds, s.d. in brackets, chance in square brackets)

| Agent | Space | Pac-Man cell | nearest pellet | nearest ghost | power on | action purity | greedy steering |
|---|---|---|---|---|---|---|---|
| DQN | embedding (512) | 0.58 (0.04) [0.08] | 0.51 (0.03) [0.07] | 0.24 (0.03) [0.03] | 0.75 [0.51] | 0.81 (0.02) [0.63] | 0.38 [0.36] |
| `ec_frozen` | embedding (32) | 0.36 (0.02) [0.09] | 0.31 (0.02) [0.07] | 0.19 (0.01) [0.03] | 0.63 [0.50] | 0.75 (0.01) [0.62] | 0.40 [0.33] |
| NEC | embedding (32) | 0.43 (0.05) [0.10] | 0.39 (0.04) [0.08] | 0.21 (0.02) [0.04] | 0.65 [0.50] | 0.78 (0.03) [0.62] | 0.37 [0.33] |
| (reference) | raw pixels | 0.78 (0.02) [0.09] | 0.68 (0.02) [0.07] | 0.27 (0.01) [0.03] | 0.73 [0.50] | 0.87 (0.01) [0.62] | - |

| Agent | memory coherence | probe vs memory |
|---|---|---|
| `ec_frozen` | 0.88 (0.03) | 0.98 (0.03) |
| NEC | 0.70 (0.05) | 1.26 (0.09) |

## What this shows
1. **The learned encoder does change the geometry, a little.** Against the frozen encoder, NEC's neighbours agree more on Pac-Man's cell (0.43 vs 0.36) and the nearest pellet (0.39 vs 0.31). The seed ranges barely overlap (NEC 0.34-0.49 vs 0.34-0.38 for position), so it is a small, real shift, not a difference in score.
2. **The change is far from what DQN's encoder does** (0.58 and 0.51), though DQN's embedding has 512 dimensions against NEC's 32, so the size confounds that comparison.
3. **Every encoder is a worse neighbour space than the raw pixels** (0.78 for position). Compressing 28k pixels to 32 numbers loses the information that nearest-neighbour memory needs; learning recovers only a fraction of what the frozen random projection discards. This fits NEC not beating `ec_frozen`: a lookup in a worse space than pixels cannot do better than the pixel-space lookup.
4. **NEC's memory is organised by value but looks stale.** Coherence drops from 0.88 to 0.70: neighbours in NEC's keys have more similar values than in the frozen one, so the training does shape the memory. But probe vs memory rises to 1.26, i.e. visited states sit about 26% farther from the stored keys than the keys sit from each other, while the frozen encoder gives 0.98. A plausible reading is that keys written under earlier versions of the encoder are not re-embedded as it moves (`nec_refresh` exists for this), so new states no longer land near the old keys. I did not test that here.
5. **Action-level measures are uninformative.** Greedy steering is at chance for all agents including DQN (0.38 against 0.36), although DQN scores ~7x random. My "good action" label (safe, closer to the nearest pellet) is evidently not what DQN's policy follows (it can be right to move away from the nearest pellet), so I do not read anything from the near-chance steering, and action purity is high mostly because neighbours share position.

## Limits
- Five seeds, one checkpoint (100k steps), one NEC learning rate; no earlier checkpoints, so no trajectory of the geometry over training.
- 32 vs 512 dimensions; distances in different dimensions are not directly comparable.
- Labels are my own definitions (ties broken by reading order); the "good action" proxy is crude, see point 5.
- Correlational. Item 4 is a hypothesis, not a finding.

## Next
Re-run NEC with `nec_refresh` (keys re-embedded) on Pac-Man and check whether probe-vs-memory returns to ~1 and the score moves; save checkpoints every 20k steps to see when the geometry diverges from `ec_frozen`; a decodability read-out (ridge) of position and ghost offset from each embedding, which is less sensitive than neighbour purity.
