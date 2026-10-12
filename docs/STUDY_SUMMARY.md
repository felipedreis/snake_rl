# Episodic control on Snake and Ms. Pac-Man: closing summary

*Status: **closed on 2026-10-12.** This document summarises the study from the experiment reports in `docs/`. Every number
comes from a linked report. It adds no new runs. Where a statement is our interpretation and not a measurement, it says
so.*

## In short

- **Why we started.** To understand the episodic control proposed by MFEC (Blundell et al., 2016) and NEC (Pritzel et
  al., 2017) well enough to choose one for `dl2l`. NEC's headline claim is data efficiency: early in training it beats
  DQN.
- **What we did.** We wrote a pure-NumPy implementation of DQN, N-step DQN, MFEC, a frozen-embedding EC and NEC, with
  hand-derived and gradient-checked gradients through the kNN memory. We tested them on our own Snake (7×7 to 25×25, with
  obstacles, bonus food, noise channels, Atari-style 84×84 pixels and DQN's exact network) and on our own Ms. Pac-Man.
  There were 9 experiments with reports, 4 of them pre-registered.
- **What we found.**
  1. **We did not reproduce NEC's advantage over DQN on Snake, at any size, render or budget we tried.** At 10×10 with
     the paper's encoder and 250k steps, DQN scores 8.82 fruit per episode and NEC 0.78–0.88. NEC's learned embedding
     scored *below* a frozen random one.
  2. **MFEC reproduced its paper's claim where states repeat.** It was the best and fastest agent on the clean 7×7 board.
     It fails as soon as observations stop repeating (noise channels).
  3. **NEC is very sensitive to magnitudes that DQN ignores:**
     - the size of a constant input (walls) decides whether its embedding explodes;
     - the scale of the reward decides whether its learned encoder helps at all.
  4. **On Ms. Pac-Man we found one configuration where NEC is ahead of the paper-style DQN, but only weakly.** NEC scored
     519 against 478, with overlapping seed ranges, and only once rewards were in Atari units. Tuning one DQN setting
     (target sync) then put DQN at 668, the best arm of the study.
  5. **Most of what we saw is consistent with a tuning asymmetry in the original comparison** (section 5).
- **Why we stop.** Each step towards the paper's setting moved the answer a little, and none produced a robust NEC
  advantage. More configurations would mostly test how far NEC can be tuned, not what episodic control offers. Two more
  recent directions are a better use of time (section 7).

## 1. The path, in one table

| # | Experiment | Setting | Main result | Report |
|---|---|---|---|---|
| 1 | MFEC vs DQN vs NEC (pre-registered, 120 runs) | 7×7 and 10×10 grid, MLP, 40–60k steps | MFEC best on clean 7×7 (1.58 vs DQN 1.04, NEC 1.01); NEC ahead of DQN only early on 7×7 (0.83 vs 0.53); NEC near random at 10×10 (0.23) | [mfec_dqn_nec](experiments/mfec_dqn_nec/REPORT.md) |
| 2 | Ablation ladder pilots v1–v4 (pre-registered) | 25×25 `rooms` + bonus food, up to 1M steps; paper ε schedules, food curriculum, food relocation | all four failed their gate (0 of 4 agents above random); v4 fixed the reward signal, but no agent steered | [v1](experiments/nec_ladder_g25/REPORT.md), [v2](experiments/nec_ladder_g25_paper_eps/REPORT.md), [v3](experiments/nec_ladder_g25_food_curriculum/REPORT.md), [v4](experiments/nec_ladder_g25_food_relocation/REPORT.md) |
| 3 | Encoder food probe (pre-registered by hash) | 25×25, {NEC, DQN} × {MLP, CNN} | skills are tied to board position: NEC·MLP steers at 0.91 near the start and at chance 10+ cells away; the NEC·CNN arm diverged | [encoder_food_probe_g25](experiments/encoder_food_probe_g25/REPORT.md) |
| 4 | Divergence diagnosis | 25×25, one seed, ablations | trigger: the constant walls input; enabler: a loss with no restoring force on the embedding's scale | [DIAGNOSIS_AND_NEXT_STEPS](DIAGNOSIS_AND_NEXT_STEPS.md) |
| 5 | Atari-style pixels + RMSProp | 84×84 gray, 4-frame stack, Nature CNN, 7×7 to 25×25 | the paper's input and network do not make steering position-general; learned NEC beats frozen EC only at 7×7 | [pixels_rmsprop_probe](experiments/pixels_rmsprop_probe/REPORT.md) |
| 6 | Wall-magnitude ablation | grid and pixels, 3 seeds | walls at 1.0 (or gray 255) diverge 3 of 3; 0.235 (gray 60) or hidden 0 of 3; RMSProp is no reliable fix (1 of 3 diverged) | [grid_ablation_rmsprop_control](experiments/grid_ablation_rmsprop_control/REPORT.md) |
| 7 | Absolute actions (E6) | DQN, pixels, 7×7 and 10×10 | absolute (Atari-like) actions double DQN's score on every seed | [absolute_actions_e6](experiments/absolute_actions_e6/REPORT.md) |
| 8 | NEC learning-rate sweep | 10×10 pixels, absolute actions, 6 settings | no learning rate makes NEC beat the frozen embedding (best 0.83 vs 0.78) | [nec_lr_sweep_absolute](experiments/nec_lr_sweep_absolute/REPORT.md) |
| 9 | Budget and evaluation (pre-registered, 60 runs, 250k steps) | 10×10 pixels, absolute actions | DQN 2.08 → 9.03 with budget, NEC flat at 0.7–0.96 and 0.22–0.31 *below* frozen EC; all three hypotheses not supported, as predicted | [nec_budget_eval_g10](experiments/nec_budget_eval_g10/REPORT.md), [posthoc](experiments/nec_budget_eval_g10/posthoc/REPORT.md) |
| 10 | Ms. Pac-Man ladder | our 14×14 maze, 2 ghosts, 84×84, frame skip 4, 100k steps | dense reward: every agent 6–8× random; order DQN 426, MFEC 390, frozen EC 389, NEC 357 (overlapping) | [pacman_ladder](experiments/pacman_ladder/REPORT.md), [posthoc](experiments/pacman_ladder/posthoc/REPORT.md) |
| 11 | Calibration C1, C4 | Pac-Man; paper optimizer for DQN; Atari reward units | reward units move NEC 357 → 519; its learned encoder now beats frozen EC by 120–155; paper-style clipped DQN 478 | [pacman_calibration](experiments/pacman_calibration/REPORT.md), [CALIBRATION_AUDIT](CALIBRATION_AUDIT.md) |
| 12 | Calibration C2, C3 | Pac-Man; DQN target sync; NEC memory size and N | DQN with a 4000-step sync: 668, the best arm of the study; NEC without eviction 553 | [pacman_c2c3](experiments/pacman_c2c3/REPORT.md) |

## 2. What we built that outlives the study

- **A hand-written, gradient-checked NEC.** It includes the gradient through the kernel-weighted kNN lookup into the
  encoder, keys and values (worst relative error 3 × 10⁻⁷ against finite differences). It also has MLP, two grid CNNs and
  DQN's exact Nature CNN, plus Adam and RMSProp, all in NumPy.
- **Bit-for-bit reproducible runs** with one BLAS thread, checked across refactors and on the CCAD cluster (job 1382: all
  12 runs gave identical episodes).
- **Diagnostics that found real problems:**
  - training telemetry (`telemetry.py`);
  - a representation probe (`probe.py`), whose steering split by distance is what exposed position-bound skills;
  - embedding-geometry tools (`geometry.py`, `geometry_pacman.py`);
  - value calibration (max Q against the return actually collected).
- **Pre-registration practice.** Protocols, frozen hashes and confirmatory tests with Holm correction. Several "not
  supported" verdicts were predicted as such.
- **Raw results** on Hugging Face (`felipedreis/snake-rl-results`).

## 3. Findings about the algorithms

### 3.1 MFEC does what its paper says, under the conditions its paper relies on

- On clean 7×7 Snake, MFEC's score at 5k–10k steps (1.00) already matched where DQN and NEC ended (≈ 1.0). It ended
  about 0.55 fruit per episode ahead of both (Holm-corrected p = 7.6e-5).
- **The mechanism:** about a quarter of its lookups hit an exactly stored state, and two thirds of its writes update an
  existing row. It is a lookup table of the best return ever seen, which suits deterministic dynamics.
- **When states stop repeating, it fails completely.** With 4 noise channels, MFEC had 0% exact hits and scored 0.23
  against random's 0.21. At 10×10, its early lead over DQN shrank to an inconclusive +0.06, as the paper reports for
  Atari.
- **It is cheap and robust to reward scale.** On Pac-Man, a run took 1,380 s against NEC's 5,278 s and DQN's 6,733 s.
  Its score did not move between small and Atari reward units (390 vs 385).

### 3.2 NEC's learned embedding rarely helped

- **Snake:** a learned embedding beat a frozen random one only on the 7×7 pixel board (1.00 vs 0.80). From 10×10 up it
  tied or lost. At 250k steps every NEC arm was 0.22–0.31 *below* frozen EC (intervals exclude 0).
- **NEC did not steer even to adjacent food.** At distance 1, steering was 0.34–0.37 (chance 0.33). Its Q-values were
  never tied, so the memory gives distinct values; they just do not point to the food.
- **Why (measured on 250 saved agents, the causal link inferred):** the 50 neighbours' values are almost as dispersed as
  the whole memory (coherence 0.9–1.0). So the kernel average sits near the memory-wide mean and barely separates
  actions (spread 0.31–0.35, flat over training; DQN's grows from 0.85 to 2.14).
  - Embeddings are organised by snake position for every agent, including DQN.
  - DQN reads Q through a learned linear layer that can use other directions. NEC reads it from neighbours, so a
    position-dominated geometry hides the food from it.
- **Budget was not the limit.** DQN improved roughly fourfold between 60k and 250k steps, while NEC stayed flat.
- **"The learned encoder adds nothing" was, on Pac-Man, a property of our small reward scale** (section 3.4).

### 3.3 Sensitivity to input magnitude: the embedding explosion

- On the 25×25 `rooms` board, NEC with a CNN encoder diverged: key norm 1 → 10⁶–10⁹ within a few thousand steps, 80%
  dead ReLUs. The gradient was verified correct, and the same CNN trained stably under DQN's loss.
- **The trigger is the magnitude of a constant, information-free input** (the walls, 44 of ~47 non-zero inputs):

  | Wall level | Diverged |
  |---|---|
  | grid 1.0, or pixel gray 255 | 3 of 3 |
  | grid 0.235, or pixel gray 60 | 0 of 3 |
  | hidden | 0 of 3 |

  Two different networks agree. The frame stack is not a factor.
- **The enabler is NEC's loss.** It sees only distances between keys, and kernel weights do not change under a common
  shift or rescaling. Nothing pins the embedding's scale, and Adam's normalised steps grow a direction that moves all keys
  together. DQN is spared because its outputs must match the targets.
- **Ruled out:** the memory SGD and stale keys. RMSProp lowered the risk but did not remove it (1 of 3 seeds still
  diverged). L2-normalised keys hid the problem without fixing it.

### 3.4 Sensitivity to reward magnitude

- **The same NEC (RMSProp 1e-4) on Ms. Pac-Man:**

  | Rewards | NEC score | Learned encoder vs frozen EC |
  |---|---|---|
  | small units | 357 | no gain (frozen EC 389) |
  | Atari units (10 / 50 / 200) | 519 | +155 |

  The controls behaved: frozen EC and MFEC, which train no network, did not move.
- **With salient rewards, NEC's memory values were well calibrated** (max Q / return 1.04–1.10). DQN's were inflated
  2–9× until the target sync was fixed.
- **The often-quoted Atari NEC learning rate (7.92e-6, from a secondary source; the paper does not report it) depends on
  the reward scale.** It froze the encoder on our ±1 Snake rewards (distance correlation with initialisation 0.999), but
  it worked in Atari units on Pac-Man (483).
- **Caveat:** the Atari-unit arms also dropped the death penalty and the clear bonus. Our gradient clip (10, not in the
  paper) cut 100% of NEC's updates in those arms. Neither was separated.
- **Not tested:** whether reward scale also closes the gap on Snake.

### 3.5 Things that helped every agent, and that the papers take for granted

- **Absolute actions,** as on an Atari joystick, doubled DQN's score (7×7: 4.35 vs 2.15; 10×10: 1.83 vs 0.72). They also
  helped NEC about 2.5×.
- **Food curriculum and relocation** fixed the sparse-reward problem on 25×25. Every agent ate thousands of fruit early.
- **DQN's target sync** was the largest single DQN effect (478 → 668 on Pac-Man). The best value (4000 agent steps) was
  neither our default (1000) nor the paper's ratio (40,000, which scored 364).

## 4. Where NEC came out ahead, and how fragile it was

| Comparison | NEC | DQN | Read |
|---|---|---|---|
| Snake 7×7 grid, early (5k–10k steps) | 0.83 | 0.53 | ahead early, level by the end (1.01 vs 1.04) |
| Snake 25×25 held phase, matched ε (fruit per 100 steps) | 25.6 | 7.2–8.2 | more fruit, but only near the start cell (83% of its states within 5 cells) |
| Pac-Man, Atari units, vs paper-style clipped DQN | 519 | 478 | ahead, but the seed ranges overlap (417–565 vs 434–539) |
| …same, after one DQN setting was tuned | 553 (no eviction) | **668** | DQN ahead by 115; ranges overlap; 668 is the best of three selected values |

So the one configuration where NEC led needed salient rewards, the paper's units and an untuned DQN. Once DQN got the
same amount of tuning attention, the lead reversed.

## 5. On the original comparison (our interpretation)

This section is interpretation, informed by our data but not tested by it.

- **Unequal tuning.** Nature DQN's settings were tuned by Mnih et al. for a 50M-frame regime. The NEC paper tuned NEC (its
  learning rate, α, embedding size, N and ε, swept on six games) for the early, data-efficient regime. It then compared
  at 10M frames and below against DQN with its long-horizon settings.
- **Our data show the same direction of effect.** Our NEC was swept and our DQN was not, and NEC looked competitive. The
  one DQN setting we did tune (target sync, which only matters relative to the budget) was enough to move DQN from below
  NEC to the best arm we ran. The milestone review had flagged this asymmetry already ("DQN has had defaults only, while
  NEC has had sweeps").
- **The pattern is documented elsewhere.** van Hasselt, Hessel and Aslanides (2019, *When to use parametric models in
  reinforcement learning?*) showed that a Rainbow DQN tuned for the low-data regime beats a model-based agent's published
  Atari 100k scores, with less experience and computation. Baselines tuned for a different data regime make
  data-efficiency claims look larger than they are.
- **What we cannot say:** whether a fully paper-faithful NEC (10M frames, capacity 5 × 10⁵, real Atari) would hold its
  published lead over a regime-tuned DQN. We never ran Atari, and our budgets were 0.4M frames on Pac-Man.

## 6. Answer for `dl2l`: MFEC or NEC?

From this evidence, **MFEC-style episodic control, not NEC.**

| | MFEC | NEC |
|---|---|---|
| Worked as its paper says | yes, where states repeat | not reproduced on Snake; weakly, and under conditions, on Pac-Man |
| Sensitive to input magnitude | no network to explode | constant inputs can blow up the embedding |
| Sensitive to reward magnitude | no (390 vs 385) | yes (357 vs 519) |
| Cost per Pac-Man run | 1,380 s | 5,278 s |
| Moving parts to tune | k, capacity, projection | lr, optimizer, α, key size, N, p, δ, capacity, mem_lr, clip |
| Main weakness | needs a representation in which similar states are near each other | its learned representation was rarely better than a random one |

**What decides MFEC's success is the representation it looks up in, not the memory.**
- Random projections of raw input failed with noise channels.
- On Pac-Man, raw pixels were a better neighbour space than any encoder we trained (position purity 0.78, against 0.43
  for NEC and 0.58 for DQN).
- If `dl2l` gives the memory a representation learned by some other objective, a simple MFEC-style table on top is the
  cheaper and more predictable choice. That objective could be self-supervised, a world model, or the egocentric state
  of an embodied agent.
- NEC's end-to-end training of that representation through the kNN was the fragile part.

## 7. What we would study instead

- **Episodic memory of failures:** FEMA, from Miao, *Learning From Failures: Efficient Reinforcement Learning Control
  with Episodic Memory* ([arXiv:2603.07110](https://arxiv.org/abs/2603.07110)).
  - It stores short-horizon failure trajectories (collisions, falls) in an episodic memory and uses retrieved similar
    failures to steer learning towards longer trajectories.
  - It reports a 33% gain in sample efficiency on standard benchmarks and runs on a real bipedal robot.
  - Early Snake training is also dominated by short failures (random dies in about 11 steps), so this setting fits.
  - The memory sits beside the learner and does not replace its value function. That avoids the NEC failure we kept
    hitting.
- **A JEPA-style encoder from pixels.** Train a joint-embedding predictive model on Snake frames, then put an MFEC-style
  memory or a small value head on top of the frozen embedding.
  - It tests, from the other side, our main finding that the representation decides everything.
  - Our tools carry over: the pixel render, the probes (does the embedding separate the egocentric food offset?) and the
    geometry measures.

## 8. Limits of our own conclusions

- **Our games, not Atari.** Snake and our Ms. Pac-Man are our own designs, so scores cannot be compared with the papers'.
- **Small budgets.** At most 1M agent steps on Snake, and 100k agent steps (0.4M frames, 4% of the NEC paper's 10M) on
  Pac-Man.
- **Few seeds.** 5 seeds on Pac-Man and 2–3 in several exploratory Snake sweeps. Only four experiments were
  pre-registered, and the Pac-Man line was all exploratory.
- **Deviations from the paper that we kept:**
  - DND capacity 2 × 10⁴ (paper 5 × 10⁵);
  - key size 32 (secondary source 128);
  - N = 50 (paper 100);
  - our gradient clip at 10;
  - training from the start instead of after 5 × 10⁴ steps;
  - one shared ε schedule.

  Removing eviction on Pac-Man changed the score within noise (519 → 553). The others were not tested.
- **Selection.** DQN's 668 is the best of three sync values chosen on the same scores. NEC's learning rate was also
  chosen on earlier runs. A fair, held-out sweep for both was planned and not run.

## 9. Open items left as they are

These were planned in [the milestone](milestones/nec-where-it-breaks.md) and the audit, and are now dropped:

- the oracle-key agent;
- the egocentric control arm;
- the board-size scaling study and the complexity ramp;
- the paper-faithfulness factorial;
- NEC without the gradient clip;
- reward scale on Snake;
- a fair DQN/NEC sweep with fresh seeds;
- a 1M-frame confirmatory run.

The [project overview](PROJECT_OVERVIEW.md), the [diagnosis](DIAGNOSIS_AND_NEXT_STEPS.md) and the
[calibration audit](CALIBRATION_AUDIT.md) record the state of each line when it stopped.

## Sources

- Pritzel et al. 2017, *Neural Episodic Control*, [arXiv:1703.01988](https://arxiv.org/abs/1703.01988)
- Blundell et al. 2016, *Model-Free Episodic Control*, [arXiv:1606.04460](https://arxiv.org/abs/1606.04460)
- Mnih et al. 2015, *Human-level control through deep reinforcement learning*, Nature 518
- van Hasselt, Hessel and Aslanides 2019, *When to use parametric models in reinforcement learning?*, NeurIPS,
  [arXiv:1906.05243](https://arxiv.org/abs/1906.05243)
- Agostinelli et al. 2019, *Memory-Efficient Episodic Control Reinforcement Learning with Dynamic Online k-means*,
  [arXiv:1911.09560](https://arxiv.org/abs/1911.09560) (secondary source for NEC's unreported hyperparameters)
- Miao 2026, *Learning From Failures: Efficient Reinforcement Learning Control with Episodic Memory*,
  [arXiv:2603.07110](https://arxiv.org/abs/2603.07110)
