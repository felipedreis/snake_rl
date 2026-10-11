# Milestone: where and why NEC breaks on Snake

Status: **2026-10-11. Posted to GitHub as milestone 1 (issues #7-#18). Decisions so far: food curriculum and relocation are kept; cleanup (issue 8) comes after the oracle key and egocentric option are designed and before the scaling study.** Written by Claude from the work done on branch `explore/pixels-rmsprop`.
Each finding is marked **measured** (a number from a run or a probe), **inferred** (reasoning from measurements, not directly tested) or **unverified**.

## Goal

The project began as a reproduction of Neural Episodic Control (Pritzel et al. 2017) to check that its mechanisms are understood. NEC's central claim is data
efficiency: early in training it beats DQN. The milestone's aim is to understand **every part of NEC, how it differs from DQN and MFEC, why it does not learn to
play Snake well on larger or more complex boards, and exactly how it breaks.** It is not an attempt to make NEC win.

## Questions

1. Which NEC components matter (encoder, per-action memory, kernel lookup, N-step returns, replay-trained embedding), and how do DQN and MFEC differ from it?
2. At what board size does NEC stop beating DQN, and what fails first (memory, kernel, embedding, exploration)?
3. How does added complexity (pillars, walls, rooms, bonus food) change that?
4. Does an egocentric input fix the representation, and does NEC then beat DQN early in training, as the paper says it should?
5. Is the geometry explanation (below) correct?

## What we did

| Step | Where | Outcome |
|---|---|---|
| MFEC vs DQN vs NEC, 7x7 and 10x10, 10 seeds (pre-registered) | `docs/experiments/mfec_dqn_nec/` | NEC is on par with DQN at 7x7 (1.01 vs 1.04) and ahead early (0.83 vs 0.53 at 5k-10k steps); at 10x10 NEC is near random (0.23, random 0.17) while MFEC scores highest (0.66, not significantly above DQN's 0.60). |
| NEC ladder pilots v1-v4 on 25x25 `rooms` + bonus, up to 1M steps (pre-registered) | `docs/experiments/nec_ladder_g25*/` | All four pilots failed the gate (0 of 4 agents above random). |
| Encoder probe on 25x25 | `docs/experiments/encoder_food_probe_g25/` | What NEC learns is tied to the board position where it learned it; the CNN as built did not fix it. |
| Pixel render, RMSProp, and the embedding explosion | `docs/experiments/pixels_rmsprop_probe/`, `grid_ablation_rmsprop_control/` | The size of the constant wall input sets whether NEC's embedding explodes (walls 1.0: 3 of 3 diverge; 0.235: 0 of 3). RMSProp is not a reliable fix (1 of 3 diverged). |
| E6: absolute against relative actions | `docs/experiments/absolute_actions_e6/` | Absolute actions about double DQN's score on every seed (7x7: 4.35 vs 2.15; 10x10: 1.83 vs 0.72). |
| NEC learning-rate sweep with absolute actions | `docs/experiments/nec_lr_sweep_absolute/` | No learning rate makes NEC beat the frozen embedding (best 0.83 vs 0.78, seed ranges overlap); steering at chance for every setting; absolute actions help NEC about 2.5x (0.64 vs 0.26). |
| Budget and evaluation experiment, 60 runs at 250k steps (pre-registered, frozen) | `docs/experiments/nec_budget_eval_g10/` | H1, H2, H3 all **not supported**, as predicted. DQN improves strongly with budget (eval 2.08 at 60k to 9.03 at 250k); NEC stays at 0.7-0.96; every NEC arm is 0.22-0.31 *below* `ec_frozen`; NEC steering 0.43 (chance 0.39). |
| Infrastructure | `ansible/`, Hugging Face dataset `felipedreis/snake-rl-results` | Reproducible cluster runs; cluster and local results match up to float noise; raw results and checkpoints off git. |

## What we learned

**Measured**
- At 250k steps on 10x10 pixels with absolute actions (10 seeds): DQN 8.82 fruit per episode and steering 0.77 (0.73-0.80 at every distance from 1 to 6 steps); `ec_frozen` 1.09; NEC 0.78-0.88; random 0.15. NEC does not improve between 60k and 250k.
- NEC's Q-values are never tied on the probe (all-tied share 0.00), so chance-level steering is not missing information; NEC does not even react to adjacent food (steering at distance 1 is 0.34-0.37, chance 0.33).
- Learning the embedding made NEC worse than leaving it random (-0.22 to -0.31, intervals exclude 0), and its evaluation episodes are half as long (53-63 steps against 129).
- NEC is level with DQN at 10k-20k steps and DQN is ahead from 30k: no early NEC advantage on this board.
- On the 10x10 pixel board, DQN with absolute actions steers at 0.60 on the probe (chance 0.39) and NEC at 0.40, for every learning rate and optimizer tried at 60k steps.
- The embedding's neighbourhood structure follows pixel geometry. Mean pixel distance is 14.2 when only the food moves and 29.5 when the same food offset occurs with the snake elsewhere; the untrained Nature CNN keeps the same ratio (0.50 vs 0.48). Neighbours share the snake placement (`knn_config` 0.63) and not the food offset (`knn_offset` 0.04). (One seed; raw-pixel neighbour metrics use a 512-d random projection.)
- NEC with RMSProp barely changes this geometry in 60k steps (`knn_config` 0.63 to 0.58 at the highest rate); with Adam it changes a lot (to 0.16) without improving steering and with 18% dead units; DQN's hidden layer reorganises (0.73 to 0.39).
- The random baseline on 10x10 is 0.17 fruit per episode (relative actions, grid).

**Inferred, not directly tested**
- That this geometry is why NEC's Q-values barely separate actions: the 50 nearest memories of a state would mostly be states with the same snake and different food, whose best actions disagree, so their average is nearly flat. The neighbours measured were among probe states, not the agent's memory.
- That more budget would fix it: the budget experiment found that it does not, up to 250k steps. The representation, or the signal that trains it, is the likelier bottleneck (the oracle-key test is the direct check).
- That a position-organised embedding works in Ms. Pac-Man because the maze is fixed, actions are absolute and scoring needs no target-directed steering (the paper itself says small items are plentiful).

**Unverified or corrected**
- The NEC paper does **not** report its learning rate; the commonly cited 7.92e-6 comes from Agostinelli et al. 2019, Table 5. Other values from that table (key size 128, N = 100, replay 1e5, exploration 1 to 0.001 over steps 5k-25k, training from 50k steps) are also secondary-source.
- Labyrinth results are in the MFEC paper (random-projection or VAE embeddings, first-person view), not in NEC's.
- Whether a minimal egocentric creature (as in `dl2l`) beats these agents on Snake: **not measured here**.
- DQN has had defaults only, while NEC has had sweeps, so the comparison is not symmetric.

## Open questions behind the issues
- How good can a trivial policy be? (A scripted greedy agent calibrates "good".)
- How does the failure scale with board size and with complexity?
- Is the representation the bottleneck? (egocentric arm, oracle key)
- How much of the gap to the paper is the known departures?

## Issues, in order of implementation

Posted to GitHub as [milestone 1](https://github.com/felipedreis/snake_rl/milestone/1); issue number = order + 6, because the repo already had six pull requests.

| Order | GitHub | Issue | Deliverable | Depends on | Pre-register? |
|---|---|---|---|---|---|
| 1 | #7 | Analyse job 1475 and report (**done**) | `docs/experiments/nec_budget_eval_g10/REPORT.md` | - | frozen already |
| 2 | #8 | NEC explained: components and comparison with DQN and MFEC | document with a table per component and the code location of each | - | no |
| 3 | #9 | Make frozen hashes verifiable | each `FROZEN.sha256` records its commit; a script that checks the files at that commit | - | no |
| 4 | #10 | Reproducible script for the geometry measurements | script under `docs/` | - | no |
| 5 | #11 | Scripted greedy baseline | agent + a score for each board | - | no |
| 6 | #12 | Oracle-key agent (hand-made key: food offset, heading, blocked-neighbour flags) | agent + report; DQN on the same features as the fair comparison | 5 | yes |
| 7 | #13 | Egocentric observation option | env option + tests | - | no |
| 8 | #14 | Simplify the code base (inventory below) | smaller `env.py`, `run.py`, `nn.py`, `agents/`, tests; tag of the pre-cleanup commit; bit-for-bit check of the kept path | 3, 6, 7 | no |
| 9 | #15 | Board-size scaling study (DQN, MFEC, `ec_frozen`, NEC; a DQN learning-rate check) | protocol, runs, report: where it breaks and how | 2, 5, 8 | yes |
| 10 | #16 | Complexity ramp at fixed size (open, pillars, walls, rooms) | protocol, runs, report | 9 | yes |
| 11 | #17 | Egocentric control arm | protocol, runs, report | 7, 9 | yes |
| 12 | #18 | Paper-faithfulness factorial (key size, DND capacity, N, exploration, replay, delayed training) | protocol, runs, report | 1 | yes |

## Code simplification (issue 8): inventory and proposal

Source is about 2,300 lines (`env.py` 301, `nn.py` 277, `agents/nec.py` 275, `run.py` 227, `probe.py` 208, `watch.py` 260, `agents/dqn.py` 145, the rest smaller) plus a 702-line test file with 52 tests.
Of the 18 registered agents, the current work uses `random`, `mfec`, `dqn_naturecnn`, `dqn_nstep_naturecnn`, `ec_frozen_naturecnn` and `nec_naturecnn`.

| Candidate to remove | What it touches | Risk and note |
|---|---|---|
| Relative actions (`--actions relative`, 3-action path, `heading_after`'s relative branch, probe and watch labels) | `env.py`, `probe.py`, `watch.py`, `run.py`, tests | Old reports compare against relative actions (E6, the sweeps). Keep them reproducible with a git tag taken before the cleanup. |
| Grid-input agents: the 10 `*_cnn`, `*_dqncnn` and MLP registry entries, the `cnn`/`dqncnn`/MLP encoder branches in `NECAgent` and `DQNAgent` | `agents/`, `nn.py` | Most of the 52 tests use these because they are fast. A tiny encoder (or a reduced pixel setup) must stay for tests. |
| The grid observation itself | `env.py` | **Keep until the oracle key and the egocentric arm are designed:** both are easiest to build from the grid state. Decide in issues 6 and 7. |
| Grid-only options: distractor noise channels, grid frame stack, `wall_scale` grid branch | `env.py`, `run.py`, `probe.py` | Used by the 7x7 noise experiment and the explosion ablation only. |
| NEC extensions not in the paper: `nec_refresh`, `nec_bonus` | `agents/nec.py`, `memory.py` | Unused since the 25x25 pilots. |
| Food curriculum and relocation | `env.py`, `run.py` | **Kept** (decided by the experimenter): the complexity ramp may reuse them. |
| Readers' fallbacks for old JSON fields | `plot.py`, `watch.py` | Keep readers tolerant of old files even after the writers stop producing the fields. |

Proposed method: (1) tag the commit before starting; (2) remove one item per commit, tests green each time; (3) before and after, run `dqn_naturecnn`, `nec_naturecnn`, `ec_frozen_naturecnn`, `mfec` and `random` for a few thousand steps with the same seed and require identical episodes, evaluations and probes; (4) keep the results JSON schema readable for old runs.

**Frozen hashes (issue 3).** The older pre-registrations' `FROZEN.sha256` files already fail at HEAD (`encoder_food_probe_g25`: `run.sh` and `probe.py`; `mfec_dqn_nec`: `PROTOCOL.md` and `run.sh`),
because the files were edited after freezing; this was already the case before the 2026-10 work in this milestone. The cleanup will make the budget experiment's check fail too. Recording the commit in each
`FROZEN.sha256` and checking the files at that commit fixes this for good.

## Risks
- Scope creep: the project has already branched several times. Each issue should finish with a report before the next starts.
- Compute (measured on job 1475, 60 runs of 250k steps at 10x10 pixels, 25 at a time): 12.6 h of wall clock and 230 CPU-hours in total; per run, NEC 3.9-6.8 h
  on average depending on the learning rate (the 7.92e-6 arm was slowest, unexplained), DQN 5.4 h, `ec_frozen` 1.1 h, random a few seconds. Peak memory was about
  1.6 GB per task (8 GB was requested). The pixel render is always 84x84, so per-step cost should not grow with the board size (not yet measured on other boards).
  The earlier estimate of about a day was about 2x too pessimistic.
- Speed: in a 3,000-step profile, 86-90% of the time is in the NumPy convolution code (`nn.py`: forward, backward, im2col, optimizer step); the kNN memory is 2% at that
  size and the environment under 1%.
- Unequal tuning: NEC has had sweeps, DQN has not. Issue 4 includes a learning-rate check for DQN.
