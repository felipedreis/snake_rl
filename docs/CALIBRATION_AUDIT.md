# Calibration audit: our DQN / NEC / MFEC set-up against the papers (2026-10-12)

**Status: audit of existing runs plus two cheap local checks; no new training runs.** Prompted by the Ms. Pac-Man ladder
([REPORT](experiments/pacman_ladder/REPORT.md)). Extensions that are not in the papers (`nec_refresh`, `nec_bonus`) are left out on purpose: the aim is to
reproduce the papers' agents, not to improve them.

## Sources, and how far each can be trusted
- **NEC paper** (`nec_pritzel2017.pdf`, read in full for this audit): N, p, delta, gamma, capacities, optimizer family, update rate, frame skip. It does
  **not** report the learning rate, alpha, embedding size or epsilon: it says it swept them on six games.
- **DQN paper** (`dqn_mnih2015.pdf`, Extended Data Table 1, read from the page image).
- **Secondary:** Agostinelli et al. 2019, Table 5 ("as in the original papers"), as quoted in `docs/DIAGNOSIS_AND_NEXT_STEPS.md` and the nec_budget PROTOCOL.
  I did not re-open that paper; treat its numbers (lr 7.92e-6, alpha 0.1, key size 128, epsilon 1 to 0.001 over steps 5k-25k, training start 5e4 steps) as unverified here.

## Parameter table
Pac-Man runs use frame skip 4 and 84x84 pixels, so "agent step" = 4 frames there. In the Snake runs there is no frame skip.

| Item | Paper | Ours | Verdict |
|---|---|---|---|
| gamma | 0.99 (NEC, DQN) | 0.99 | match |
| p (neighbours), delta, kernel | 50, 1e-3, 1/(‖h-hi‖²+delta) | same | match |
| Minibatch | 32 | 32 | match |
| Updates per frames | 1 per 16 frames (NEC), 1 per 4 agent steps (DQN) | every 4 agent steps | **match on Pac-Man; Snake does 4x more updates per frame** (no frame skip) |
| Encoder | DQN's conv stack | `naturecnn`: 32 8x8/4, 64 4x4/2, 64 3x3/1, FC 512, valid padding, then key | match (pixel agents); MLP/`dqncnn` on the grid are not |
| Optimizer family | RMSProp (NEC); RMSProp 2.5e-4, momentum 0.95, min sq. grad 0.01 (DQN) | NEC RMSProp rho 0.95 eps 1e-2 (lr swept); **DQN default is Adam 5e-4** | **DQN deviates** (see C1) |
| NEC lr | not reported (secondary 7.92e-6) | 1e-4 (best of sweep) | adapted: 7.92e-6 left the encoder frozen on our reward scale |
| Loss | L2 (NEC); error clipped to [-1,1] (DQN) | L2 (NEC); Huber (DQN) | match; we also clip gradient norm at 10 (not in papers) |
| Reward clipping | none for NEC, [-1,1] for DQN | none (rewards 0.1-2 per event) | match in form; **scale differs from Atari**, see C2 |
| N-step | N = 100 (NEC) | N = 50 | deviation; and on Pac-Man **irrelevant**: training episodes last 31 agent steps, so no return ever bootstraps (pure Monte Carlo) |
| DND capacity | 5e5 per action | 2e4 | **deviation that bites**: by 100k steps actions 0/1 had 12k-26k evictions (about half of what they wrote); the paper's regime almost never evicts |
| Replay (NEC / DQN) | 1e5 / 1e6 | 2e4 / 5e4 | scaled down (run length) |
| Target network (DQN) | sync every 10,000 updates (= 40k agent steps) | every 1000 agent steps (250 updates) | scaled down; 100k-step runs would see 2-3 syncs at the paper's value |
| Training start | 5e4 frames (DQN), 5e4 steps (secondary, NEC) | DQN 1000 rows, NEC 32 rows | deviation (early training on very few, correlated rows) |
| Epsilon | DQN 1 to 0.1 over 1M frames; NEC unreported (secondary 1 to 0.001, 5k-25k steps); MFEC 0.005 | 1 to 0.02 over 5k agent steps, all agents | adapted; one schedule for all agents, not each paper's |
| Evaluation epsilon | 0.05 (DQN) | 0.05 | match |
| Key dimension | unreported (secondary 128) | 32 | unknown, differs from secondary |
| alpha (fast update) | unreported (secondary 0.1) | 0.1 | consistent with secondary; **fast path almost never fires on pixels** (3-7 exact-key updates per action in 100k steps) |
| mem_lr (keys/values by SGD) | "lower than alpha" | 1e-2 | consistent with the statement |
| kNN | approximate, kd-tree | exact | not a deviation in kind |
| Budget | 10M frames (Table 1 compares at 10M) | Pac-Man 100k agent steps = 0.4M frames | 4% of the paper's budget |
| MFEC | random projection, k = 11, gamma 1 on Atari (0.99 on Labyrinth), eps 0.005, 1e6 memories | random projection 64-d, k = 11, **gamma 0.99**, shared eps schedule, 2e4 memories | deviates on gamma, eps and capacity; our "MFEC" is MFEC-like, not the paper's exact set-up |

## Two measurements on the Pac-Man checkpoints (100k steps, seeds 1-5, 60 episodes at epsilon 0.05 each)
Mean of max-Q over visited states, next to the mean discounted return actually collected from those states:

| Agent | mean max Q | mean MC return-to-go |
|---|---|---|
| DQN | **7.25** | 0.95 |
| NEC (RMSProp 1e-4) | 0.69 | 0.62 |
| `ec_frozen` | 0.69 | 0.53 |

1. **DQN's values are inflated about 7x** over what its own policy collects, while NEC's memory is close to calibrated. DQN still scores best, so the
   policy ordering can be right while the values are wrong. Not diagnosed: candidates are no target-network lag as in the paper (sync 1000 steps vs 40k), Adam instead of RMSProp,
   and plain max-bootstrapping with N = 1. This is the largest calibration problem found.
2. **NEC's encoder is training at about DQN's pace:** update/weight ratio 0.0007-0.0009 against 0.0012 for DQN (last telemetry window). It is not frozen at 1e-4.
3. Memory use: exact-key updates 0-7 per action out of 20,000 writes; evictions up to 26k per action. The tabular fast path plays no part on pixels, and the
   capacity limit does.

## Checks that stay inside the papers (proposed, none run)
Ordered by how cheaply they settle a deviation.

| # | Check | Settles | Cost |
|---|---|---|---|
| C1 | DQN with RMSProp 2.5e-4 (paper) against Adam 5e-4, then re-measure max Q against MC return | whether DQN's inflation and its lead come from the optimizer | 5 seeds x ~2 h on CCAD; no code change (`--opt rmsprop --lr 2.5e-4`) |
| C2 | *Only if C1 does not fix DQN's inflated Q:* DQN target sync at the paper's ratio (every 10k updates). Weak at our scale: that is 2-3 syncs in a 100k-step run | the other candidate cause of the inflation | small code change (`target_every` run setting) |
| C3 | NEC with the paper's N = 100 and capacity above the run length (1e5 per action), nothing else changed | whether eviction matters (the geometry result "visited states far from keys" has eviction as a competing explanation) | needs run-level overrides for `N` and `dnd_cap` |
| C4 | Atari reward units (pellet 10, power 50, ghost 200, death -1 or none) with two arms: NEC and MFEC unclipped, DQN with rewards clipped to [-1, 1] as in the paper (the logged score stays raw). NEC also at the Atari lr 7.92e-6. **Our current DQN is unclipped and keeps the 20x ghost-to-pellet gap, so it is easier than the paper's DQN.** Also count ghosts eaten per episode for each arm | the DIAGNOSIS hypothesis that 7.92e-6 only moves the encoder when returns are in the hundreds, and whether clipping is what keeps DQN from eating ghosts (a hypothesis, not yet shown) | needs a reward-scale setting in `MsPacman` and a reward-clip option applied agent-side in `run.py` |
| C5 | Longer budget: 1M frames (250k agent steps), 10 seeds | how much of the Snake/Pac-Man gap is only budget | ~4-8 h per NEC run |

## What this means for the earlier conclusions
- Our DQN also sees unclipped, magnitude-preserving rewards, unlike the paper's clipped DQN, which makes it easier than the paper's.
- The Pac-Man ordering (DQN 426, MFEC 390, `ec_frozen` 389, NEC 357) should be read as **provisional**: DQN is the arm furthest from its paper set-up
  (optimizer, target sync, training start), and its values are badly calibrated.
- "NEC's learned encoder adds nothing" is not affected by C1/C2 (both concern DQN), but it is open to C3 (eviction) and C4 (reward scale).
- The Snake results sit on the same set-up (Adam DQN, N = 50, capacity 2e4); I have not re-measured their Q calibration.
