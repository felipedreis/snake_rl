# Experiment protocol: MFEC vs DQN vs NEC on Snake

Status: **pre-registered**. This document, `run.sh` and `analyze.py` were written and frozen before any
confirmatory run started. Their SHA-256 hashes are recorded in `REPORT.md`. Any departure from this plan
is listed under "Deviations" in the report.

## 1. Question

How does Model-Free Episodic Control (MFEC, Blundell et al., 2016) compare with DQN (Mnih et al.,
2015) and Neural Episodic Control (NEC, Pritzel et al., 2017) on our Snake environment? The comparison
covers:
- how well each agent plays by the end of a fixed training budget;
- how quickly it learns (data efficiency, MFEC's central claim);
- how this changes when the observation has many irrelevant inputs, and when the board is larger.

## 2. Background: the task in plain words

**The game.** A snake moves on an n×n board. Each step it goes straight, turns right, or turns left.
- Eating the food gives +1 and grows the snake.
- Hitting a wall or its own body gives −1 and ends the episode.
- Everything else gives 0.
- An episode is also cut off ("truncated") if the snake goes 2·n² steps without eating.

The **score** of an episode is the number of fruit eaten.

**What the agent sees.** The observation is a stack of n×n binary images ("channels"), flattened into
one vector:
- channel 1: where the head is;
- channel 2: where the body is;
- channel 3: where the food is.

These three channels fully describe the game.

**Noise channels (called `distractors` in the code).** As an option, we add D extra n×n channels.
Each pixel in them is an independent coin flip (1 with probability 0.5), redrawn at every step. They
carry no information about the game. They exist to test robustness. With D = 4 on a 7×7 board:
- the observation has 7 × 49 = 343 inputs, and 196 of them are pure noise;
- two observations of the *same* game situation look very different (about 98 noise pixels differ on
  average);
- so the exact same observation never appears twice.

An agent that compares raw observations (as MFEC does) is expected to suffer. An agent that learns
which inputs matter (DQN, NEC) can in principle learn to ignore the noise.

## 3. Agents (as implemented in this repo; no hyperparameter tuning)

| Name in code | What it is                                                                                                                                                             | Key settings                                                                                            |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `dqn`        | DQN: neural network Q(s,a), 1-step bootstrapped targets, replay, target network                                                                                        | MLP 64-64, Adam 5e-4, replay 5e4, batch 32, train every 4 steps, target sync 1000, Huber loss, γ = 0.99 |
| `nec`        | NEC: learned encoder + one differentiable memory (DND) per action, kernel-weighted kNN                                                                                 | key 32, encoder MLP 64, p = 50 neighbours, N = 50-step targets, DND 2e4 per action, γ = 0.99            |
| `mfec`       | MFEC: fixed random projection + one memory per action; exact match → stored value, else mean of k nearest; at episode end, backward Monte Carlo return with max update | key 64, k = 11, memory 2e4 per action, γ = 0.99                                                         |
| `random`     | Uniform random actions (reference floor only, not a hypothesis subject)                                                                                                | —                                                                                                       |

All agents share run.py's ε-greedy exploration schedule. ε decays linearly from 1 to 0.02 over the
first 5,000 steps, then stays at 0.02.

## 4. Design

Three environment configurations:

| ID | Board | Noise channels D | Training steps per run | Results directory |
|---|---|---|---|---|
| C1 "small, clean" | 7×7 | 0 | 40,000 | `results/exp_mfec/d0` |
| C2 "small, noisy" | 7×7 | 4 | 40,000 | `results/exp_mfec/d4` |
| C3 "large, clean" | 10×10 | 0 | 60,000 | `results/exp_mfec/g10_d0` |

- **Factors:** agent ∈ {dqn, nec, mfec, random} × configuration ∈ {C1, C2, C3}.
- **Replication:** 10 independent runs per cell, seeds 101–110. The seed sets the agent's random
  stream (`seed`) and the environment's (`seed + 1000`).
- **Total:** 120 runs, executed by `run.sh`.

New seeds are used on purpose. Seeds 1–5 (and 1–3 for C3) were used in an earlier exploratory pilot,
and that pilot shaped the hypotheses below. Reusing them would test the hypotheses on the data that
generated them.

## 5. Data collected

For every run, `snake-run` writes one JSON file containing:
- `episodes`: a list of (environment step at which the episode ended, score), one entry per finished
  episode. An episode still unfinished when the budget runs out is not recorded.
- `diagnostics`: every 6,000 steps, per action, the memory counters (memory agents only):
  - rows in use, appends, exact-match updates, evictions;
  - mean and largest absolute stored value;
  - for MFEC also the number of lookups and of exact-match lookup hits.
- `wallclock_s`: run time. It is recorded but **not analysed**: runs share the CPU (8 in parallel), so
  timings are not comparable.
- The run settings (agent, seed, steps, D, size, ε floor).

## 6. Outcome measures (computed per run; the run is the unit of analysis)

- **Primary, LATE:** the mean score of the episodes that ended in the second half of training,
  i.e. steps (S/2, S]. This measures how well the agent plays once it has trained.
- **Primary, EARLY:** the mean score of the episodes that ended in steps (5,000, 10,000]. This is
  the first 5k steps after exploration has dropped to its floor, and it measures data efficiency.
- **Secondary (descriptive only):**
  - fruit per 1,000 steps in the second half, which, unlike score per episode, rewards staying alive
    without eating less;
  - learning curves (mean ± s.e. over seeds, 20 bins);
  - for MFEC, the share of lookups that hit an exact stored state and the share of writes that
    updated an existing row, from the last diagnostics entry.

## 7. Hypotheses (confirmatory) and why

Direction is stated in advance. "A > B" means the population mean of the outcome is higher for A.

| ID  | Config | Outcome | Hypothesis | Rationale                                                                                                                                                                |
| --- | ------ | ------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| H1a | C1     | LATE    | MFEC > DQN | Snake is deterministic, and in a clean 7×7 board states repeat often. That is the regime where the paper says an episodic table excels. Pilot: 1.54 vs 0.97.             |
| H1b | C1     | LATE    | MFEC > NEC | MFEC's max update latches onto the best return seen; NEC averages and bootstraps. Pilot: 1.54 vs 0.87.                                                                   |
| H2  | C2     | LATE    | DQN > MFEC | The random projection keeps the noise, so nearest neighbours are close to random and exact matches never happen. DQN can learn to ignore the noise. Pilot: 0.48 vs 0.23. |
| H3a | C3     | LATE    | MFEC > NEC | Same mechanism as H1b. Pilot (n=3): 0.64 vs 0.23.                                                                                                                        |
| H3b | C3     | LATE    | MFEC > DQN | Same mechanism as H1a, on a board with more distinct states (weaker expectation). Pilot (n=3): 0.64 vs 0.53.                                                             |
| H4a | C1     | EARLY   | MFEC > DQN | The paper's main claim: episodic control is more data-efficient early on. Pilot: 1.07 vs 0.55.                                                                           |
| H4b | C3     | EARLY   | MFEC > DQN | Same claim on the larger board. Pilot (n=3): 0.57 vs 0.49.                                                                                                               |

Other pairwise comparisons, comparisons against `random`, and all secondary measures are
**exploratory**. They are reported descriptively, with uncorrected p-values marked as such, and they
do not confirm or refute anything.

## 8. Statistical analysis (implemented in `analyze.py`)

- **Test:** for each confirmatory hypothesis, a two-sided **exact permutation test** on the difference
  in means between the two groups of 10 runs. All C(20,10) = 184,756 relabelings are enumerated. The
  test assumes only that, under the null, the runs are exchangeable between the two agents; it does not
  assume normality. (A two-sided test is used even though directions are stated, to be conservative.)
- **Multiple comparisons:** Holm–Bonferroni correction across the 7 confirmatory tests, family-wise
  α = 0.05.
- **Decision rule:**
  - **Supported:** the Holm-adjusted p < 0.05 *and* the observed difference has the predicted sign.
  - **Contradicted:** p < 0.05 with the opposite sign.
  - **Not supported:** otherwise. This is not evidence of "no difference".
- **Effect size:** the difference in means, with a 95% percentile bootstrap CI (10,000 resamples
  within each group, RNG seed 0), and Hedges' g.
- **Descriptives:** per cell, the mean, s.d., and 95% bootstrap CI of the mean, for both primary
  outcomes and fruit per 1,000 steps.
- **Missing data:** if a run crashes, it is rerun once with the same seed. If it fails again, it is
  excluded and reported, and the test uses the remaining runs (still exact).
- **Power (rough, from pilot):** the pilot standard deviations are about 0.05–0.1 and the predicted
  differences 0.1–0.6. With n = 10 per group and a two-sided test at the strictest Holm level
  α = 0.05/7:
  - power is about 0.9 for a standardised effect g = 2 and about 0.6 for g = 1.5;
  - H1a, H1b, H2, H3a and H4a have pilot effects well above g = 2;
  - H3b (pilot g ≈ 1.0) and H4b (pilot g ≈ 0.6) are likely underpowered. A "not supported" verdict
    there is weak evidence either way.
  - The smallest attainable exact p is 2/184,756 ≈ 1e-5, so the test itself never limits significance.

## 9. Assumptions

1. **Independence:** runs with different seeds are independent samples of each agent's behaviour.
   The same seed under different agents shares the environment's food RNG, but the trajectories
   diverge immediately, so we treat groups as independent, not paired.
2. **Score measure:** mean score per episode measures playing ability. It matches the "score" in the
   MFEC and DQN papers. Episodes are weighted equally within a run, so a run with many short episodes
   is not over-weighted against one with few long ones at the run level.
3. **Windows:** the second half of training approximates the performance reached within the budget,
   not asymptotic performance. Conclusions apply to these budgets (40k / 60k steps) only.
4. **Defaults, not tuned optima:** all hyperparameters are the repo's defaults or the papers' values,
   untuned for Snake. Results compare these specific configurations, not the best achievable version
   of each algorithm.
5. **Faithfulness of MFEC:** it deviates from the paper in γ (0.99; the paper used 1 on Atari and
   0.99 on Labyrinth), memory size (2e4 vs 1e6 per action), ε floor (0.02 vs 0.005) and features (a
   random projection only, no VAE). NEC's deviations are listed in the README.
6. **Numerics:** runs execute with `OMP_NUM_THREADS=1` (and the equivalent BLAS variables), so parallel
   jobs do not oversubscribe the CPU. This can change floating-point summation order compared with
   earlier runs, so results are not bit-identical to the pilot. That is irrelevant here because all
   confirmatory runs are fresh.

## 10. Procedure

0. (Done before freezing.) `analyze.py` was dry-run on the pilot results (seeds 1–5) only to debug
   it. Those numbers are the "pilot" figures quoted above and are not part of the results.

1. Freeze this protocol, `run.sh` and `analyze.py`, and record their hashes.
2. `bash docs/experiments/mfec_dqn_nec/run.sh`: runs all 120 runs, 8 in parallel, logging to
   `logs/exp_mfec.log`.
3. Check that 120 result files exist, and rerun failures per section 8.
4. `python docs/experiments/mfec_dqn_nec/analyze.py`: writes `results.md`, `per_run.csv` and the
   learning-curve figure into this folder.
5. Write `REPORT.md`: a recap of this protocol, the results, hypothesis decisions, exploratory
   observations, deviations, and limitations.
