# Experiment protocol: which part of NEC makes it data-efficient? (25×25 rooms, bonus food)

Status: **DRAFT for review, not frozen.** Once approved, this document, `run.sh` and `analyze.py` are
frozen *after* the pilot gate (section 10) and before any confirmatory run. Their SHA-256 hashes go in
`FROZEN.sha256`, and any departure from this plan is listed under "Deviations" in the report.

## 1. Question

Neural Episodic Control (NEC, Pritzel et al., 2017) is claimed to learn from far fewer samples than DQN.
NEC differs from DQN in three ways at once:
- **N-step returns:** targets use 50 real rewards before bootstrapping, not 1.
- **Episodic memory:** Q is read from a per-action table of past (situation, return) pairs by
  kernel-weighted nearest neighbours, and a good return is usable as soon as it is written.
- **A learned embedding:** the "situation" key is produced by an encoder trained by gradient descent,
  so similar situations get nearby keys.

**Which of these three components produces NEC's gain in data efficiency?** We add them one at a
time, as an ablation ladder, and measure the gain contributed by each step.

## 2. Why this environment, and the task in plain words

The previous experiment (`docs/experiments/mfec_dqn_nec/`) used 7×7 and 10×10 open boards with only
one reward size. Two features of that setup limit what it can tell us about NEC:
- On small boards, exact game states repeat often. That favours plain lookup tables (MFEC won), and
  it hides the benefit of a learned embedding, whose job is to generalise to *new* states.
- With a single reward size (+1 per fruit), there is little for a value estimate to rank: almost
  any fruit-finding policy is equally good.

This experiment uses one harder configuration instead:
- **Board:** 25×25. That is roughly 6× the cells of 10×10, so exact repeats of a state should be rare.
- **Map `rooms`:** a cross-shaped wall splits the board into four rooms, joined only near the centre.
  Walls are lethal like the board edge. Reaching food in another room requires a detour through the
  centre, so a policy must plan more than a few steps ahead.
- **Bonus food worth 5 points:** after every 4 regular fruit, a bonus pellet appears on a random free
  cell for 2·n = 50 steps, then vanishes. It grows the snake like normal food. The agent now has to
  rank outcomes of different value under time pressure: is the detour still worth it?

**The rules:**
- Each step, the snake goes straight, turns right, or turns left.
- Rewards: +1 for regular fruit, +5 for a bonus pellet, −1 for hitting a wall, the edge or its own body
  (this ends the episode), and 0 otherwise.
- An episode is also cut off ("truncated") after 2·n² = 1,250 steps without eating.
- The **score** of an episode is the points collected: regular fruit × 1 + bonus pellets × 5.

**What the agent sees:** five 25×25 binary-or-fractional images, flattened into 3,125 inputs:
- head, body, regular food;
- walls;
- the bonus pellet, whose pixel holds the fraction of its lifetime left (1 → 0).

There are no noise channels (D = 0).

The layout (shown at 13×13 for readability; at 25×25 the arms are longer, and the opening at the
centre is the same three cells):

```
......#......
......#......
......#......
......#......
......#......
.............
#####o@.#####
.............
......#......
......#......
......#......
......#......
......#......
```

## 3. The ladder

All agents are the repo's defaults, with no tuning, and share run.py's ε-greedy schedule: ε decays
linearly from 1 to 0.02 over the first 5,000 steps, then stays at 0.02. Encoders are MLPs.

| Rung | Name in code | Q estimate from | Targets | Representation | Learning |
|---|---|---|---|---|---|
| 0 | `dqn` | neural network | 1-step | learned (network) | gradient only |
| 1 | `dqn_nstep` | neural network | 50-step | learned (network) | gradient only |
| 2 | `ec_frozen` | episodic memory (DND, p = 50 neighbours) | 50-step | **random, frozen** encoder | fast table writes only |
| 3 | `nec` | episodic memory (DND, p = 50 neighbours) | 50-step | **learned** encoder | fast writes + gradient into encoder, keys and values |

**What each step adds, and which factor it isolates:**
- **0 → 1 (H1):** N-step returns. The network and everything else are the same.
- **1 → 2 (H2):** replaces the learned Q-network with episodic memory over a random embedding. This
  isolates "episodic memory" in the form MFEC-style methods use it, without representation learning.
- **2 → 3 (H3):** learning the embedding. The memory and targets are the same.
- **1 → 3 (H4):** episodic memory *given* a learned representation. Both agents learn a representation
  by gradient descent; NEC reads Q from memory, N-step DQN from a network head.
- **0 → 3 (H0):** the whole NEC package. This is a precondition: if NEC is not more data-efficient than
  DQN here, there is no gain to decompose.

**Reference agents** (not part of the ladder; exploratory only):
- `random`: uniform random actions, the floor.
- `mfec`: a random projection plus a max-return table, the strongest agent in the previous experiment.

**Shared settings:**
- DQN: MLP 64-64, Adam 5e-4, replay 5e4, batch 32, train every 4 steps, target sync 1,000, Huber loss,
  γ = 0.99.
- NEC: key 32, encoder MLP 64, p = 50, δ = 1e-3, DND 2e4 rows per action, α = 0.1, Adam 5e-4,
  memory learning rate 1e-2, replay 2e4, γ = 0.99.

**Deliberately left out:**
- **The missing cell:** a Q-network on top of a frozen random embedding. It would complete a 2×2 design
  (memory vs network × learned vs random representation), but it is not implemented, and it is not
  part of NEC.
- `nec_refresh`: storing raw observations costs 3 × 20,000 × 3,125 floats ≈ 1.5 GB per run on this board.
- `nec_bonus`: an exploration add-on, not one of NEC's components.
- **CNN encoders** (`*_cnn`): the paper's encoder family, but about 12× slower on this board (85 s per
  10k steps against 7–14 s for the MLPs). They are a candidate for a follow-up experiment that asks
  whether the encoder type changes the answer.

## 4. Design

- **One environment configuration:** size 25, map `rooms`, bonus 5, D = 0.
- **Training budget:** S = 150,000 steps per run, fixed by the pilot gate in section 10 (300,000 if
  the first gate fails).
- **Factors:** agent ∈ {dqn, dqn_nstep, ec_frozen, nec} (the ladder) plus {random, mfec} (references).
- **Replication:** 15 independent runs per agent, seeds 201–215. The seed sets the agent's random
  stream (`seed`) and the environment's (`seed + 1000`).
- **Total:** 90 confirmatory runs, plus 18 pilot runs, executed by `run.sh`.

**Seeds:**
- Seeds 101–110 were used in the previous experiment, and seeds 1–5 in earlier pilots. Seed 1 was also
  used for two informal 25×25 `rooms` runs without bonus food (`results/g25_d0_rooms/`). Their scores
  were **not** looked at when writing this protocol.
- The pilot uses seeds 1–3 in its own results directory.

## 5. Data collected

For every run, `snake-run` writes one JSON file containing:
- `episodes`: one record per finished episode: (step at which it ended, score, regular fruit eaten,
  1 if it was cut off by the idle limit else 0). Bonus pellets eaten = (score − regular fruit) / 5.
  An episode still unfinished when the budget runs out is not recorded.
  - The last two fields are new for this experiment: run.py previously logged only (step, score).
    The change only adds fields. It was verified not to alter any run: a regenerated
    `results/exp_mfec/d0/dqn_s101.json` matched the committed file's (step, score) pairs exactly.
- `diagnostics`: every 6,000 steps, per action, for memory agents:
  - rows in use, appends, exact-match updates, evictions;
  - mean and largest absolute stored value.
- `wallclock_s`: recorded, **not analysed** (runs share the CPU).
- The run settings: agent, seed, steps, size, D, map, bonus, ε floor.

## 6. Outcome measures (computed per run; the run is the unit of analysis)

- **Primary, EARLY (data efficiency):** mean score per episode over episodes ending in steps
  (5,000, S/5]; that is (5k, 30k] for S = 150k. It starts when exploration reaches its floor, and it
  covers the first fifth of the budget.
- **Secondary, LATE:** mean score per episode over episodes ending in (S/2, S]. This is the level
  reached within the budget; it says whether a factor makes learning *faster* or also *better*.
- **Secondary, AUC:** the mean of the learning curve over the whole budget (20 equal bins of
  per-episode score, averaged). This is a single-number summary of speed and level together.
- **Robustness, regular fruit only:** EARLY and LATE recomputed with bonus points removed. This checks
  that conclusions are not an artefact of the bonus's 5× weight.
- **Descriptive:** learning curves (mean ± s.e., 30 bins), episode length, the share of episodes cut
  off by the idle limit, the share of points from bonus food, and the memory-regime diagnostics.

Fruit per 1,000 steps is **not** used. The previous experiment showed that it rewards dying often,
because each restart spawns new nearby food (its REPORT, section 3.1).

## 7. Hypotheses (confirmatory, all on EARLY) and why

"A > B" means that the population mean of EARLY is higher for A. The directions follow the NEC paper,
not pilot data: no 25×25 bonus-food results existed when they were written.

| ID | Hypothesis | Factor | Rationale | Prior confidence |
|---|---|---|---|---|
| H0 | NEC > DQN | whole package | The paper's central claim: NEC is far more data-efficient than DQN early in training. | moderate. Our NEC was near random on 10×10 in the last experiment. |
| H1 | N-step DQN > DQN | N-step returns | Food is typically 10–20 steps away, and walls force detours. A 1-step target moves reward back one step per update; a 50-step target carries it the whole way at once. | high |
| H2 | Frozen-embedding EC > N-step DQN | episodic memory, random embedding | The paper's argument: a table write makes a good return usable immediately, while a network needs many small gradient steps. | **low**. With rare exact repeats, neighbours in a random embedding of raw positions are poor matches. MFEC's exact-hit rate fell from 23% (7×7) to 15% (10×10). |
| H3 | NEC > Frozen-embedding EC | learned embedding | When states rarely repeat, kNN only helps if similar situations have nearby keys. That requires a learned embedding, which is exactly what this step adds. | high |
| H4 | NEC > N-step DQN | episodic memory, learned embedding | The memory read-out lets NEC exploit a new return at once, on top of the same kind of learned representation N-step DQN has. | moderate |

**How the answer to the question is read off:**
- **Which factors contribute:** the steps whose hypothesis (H1–H4) is supported.
- **How much:** the decomposition of NEC − DQN into the three ladder steps,
  (N-step DQN − DQN) + (frozen EC − N-step DQN) + (NEC − frozen EC), each with a bootstrap CI and its
  share of the total. This is descriptive, and only interpreted if H0 is supported.
- **Two paths from N-step DQN to NEC:** H2 + H3 (memory first, then learn the embedding) and H4
  (memory added to a learned representation). Together they separate "memory helps by itself" from
  "memory only helps once the embedding is learned".

Everything else is **exploratory**: LATE, AUC, regular-fruit-only outcomes, MFEC comparisons, and
the descriptives. It is reported with uncorrected p-values marked as such, and it neither confirms
nor refutes anything.

## 8. Statistical analysis (implemented in `analyze.py`)

- **Test:** for each of H0–H4, a two-sided **permutation test** on the difference in means of EARLY,
  15 runs against 15.
  - Exact enumeration would need C(30,15) ≈ 1.6e8 relabelings, so the test uses 200,000 random
    relabelings (RNG seed 0). The p-value is (1 + #{|permuted diff| ≥ |observed diff|}) / 200,001.
  - Its Monte Carlo error at p = 0.01 is about ±0.0002, far below any decision threshold, and the
    smallest attainable p is 5e-6.
  - The test assumes only that the runs are exchangeable between the two agents under the null.
    It does not assume normality.
- **Multiple comparisons:** Holm–Bonferroni across the 5 confirmatory tests, family-wise α = 0.05.
- **Decision rule:**
  - **Supported:** Holm-adjusted p < 0.05 *and* the observed difference has the predicted sign.
  - **Contradicted:** p < 0.05 with the opposite sign.
  - **Not supported:** otherwise. This is not evidence of "no difference".
- **Effect size:** the difference in means, with a 95% percentile bootstrap CI (10,000 resamples within
  each group, seed 0), and Hedges' g.
- **Descriptives:** per agent, the mean, s.d., and 95% bootstrap CI of every outcome.
- **Missing data:** a crashed run is rerun once with the same seed. If it fails again, it is excluded and
  reported. A run with no episode in a window has no value for that outcome and is left out of that
  test only; this is also reported.
- **Power:** two-sided, n = 15 per group, from a simulated t-test approximation:

  | Standardised effect g | Holm's strictest level α = 0.01 | Holm's loosest level α = 0.05 |
  |---|---|---|
  | 1.0 | 0.50 | 0.75 |
  | 1.5 | 0.90 | 0.98 |

  In the previous experiment, real differences between these agents were mostly g = 2–6 and never below
  0.6. A "not supported" verdict at g ≈ 1 is therefore weak evidence either way, and H2, the
  low-confidence hypothesis, is the one most likely to land there.

## 9. Assumptions, and how the report checks each one

`analyze.py` computes every check below and writes it to `results.md` under "Assumption checks". The
report must state, for each one, whether it held and what that means for the verdicts. A failed check
never changes a verdict after the fact. It changes how the verdict is interpreted, and the report says so.

| # | Assumption | Check | If it fails |
|---|---|---|---|
| A1 | **Every ladder agent learns** within the budget. A comparison between agents that never left the random floor says nothing about their components. | LATE of each agent vs `random` (permutation test, uncorrected, α = 0.05). | Hypotheses involving that agent are reported as "uninterpretable: agent did not learn". |
| A2 | **Each window holds enough episodes** for a per-run mean to be stable. | Minimum and median number of episodes per run in EARLY and LATE. The protocol requires ≥ 10. | Runs below 10 are kept (pre-registered), flagged, and discussed. |
| A3 | **Score per episode measures skill, not episode turnover or idling.** | Mean episode length, and the share of episodes cut off by the idle limit, per agent. | If an agent mostly idles to truncation, its LATE reflects survival rather than food-finding, and the report says so. |
| A4 | **Bonus food actually matters** to the agents, as the environment was chosen for. | Share of points from bonus food, and bonus pellets per episode, in LATE. Also, the five EARLY tests are repeated on regular fruit only (exploratory). | If ≈ 0 for every agent, the "differing rewards" motivation did not materialise, and this is a limitation. If the regular-fruit-only tests disagree with the confirmatory ones, the verdicts depend on bonus weighting, and the report says which. |
| A5 | **The memory regime is understood** when interpreting H2 to H4. | DND rows in use, the share of writes that evicted or updated an exact match, and when the first eviction happened. | Descriptive only. For example, if memory fills before EARLY ends, capacity may be shaping the result. |
| A6 | **Runs are independent and exchangeable** within an agent. | By design: separate processes and seeds. The same seed under different agents shares the food RNG's start, but trajectories diverge at once, so groups are treated as independent, not paired. | Not testable; stated. |
| A7 | **No single run drives a verdict.** Permutation tests do not need normality, but a difference in means can be pulled by one extreme run. | Each confirmatory test is repeated with every single run removed (20,000 relabelings each), and the result is compared with that hypothesis's Holm threshold. | A verdict that flips when one run is dropped is reported as fragile. |
| A8 | **Defaults, not tuned optima.** | Not checkable. | Conclusions concern these configurations, not each component at its best. |
| A9 | **Numerics are stable under parallel execution.** | Runs use `OMP_NUM_THREADS=1` and the equivalent BLAS variables. | — |

## 10. Procedure

0. **Preparation (done in this draft):**
   - run.py logs regular fruit and truncation per episode (section 5).
   - `analyze.py` will be debugged on throwaway short runs (seeds 901–903) whose numbers are not read.
1. **Pilot gate.** `bash docs/experiments/nec_ladder_g25/run.sh pilot` (seeds 1–3, S = 150,000), then
   `analyze.py --gate`. The gate looks *only* at whether agents learn at all:
   - A ladder agent **clears the floor** if its mean LATE over the 3 pilot seeds is at least 0.5 points
     above `random`'s.
   - **Gate passes** if at least 3 of the 4 ladder agents clear the floor. Then S stays at 150,000.
   - **If it fails,** S is doubled to 300,000 once, and the pilot is rerun. If that also fails, the
     experiment stops, and the report explains why the setup is not informative (for example, proposing
     a smaller board).
   - Pilot results may change **only S**. Hypotheses, windows, tests and seeds stay as written here. The
     pilot is reported in full, but it is not part of the confirmatory analysis.
2. **Freeze.** Record `run.sh`, with its final `STEPS`, and the hashes of this protocol and `analyze.py`
   in `FROZEN.sha256`, together with the time.
3. **Run.** `bash docs/experiments/nec_ladder_g25/run.sh main`: 90 runs, 8 in parallel, logged to
   `logs/exp_nec_ladder_main.log`. Estimated wall time: about 45 minutes at S = 150k, dominated by NEC (about 10 minutes per run); roughly twice that at 300k. The pilot takes about 12 minutes.
4. **Check.** Confirm that 90 result files exist, and rerun failures per section 8.
5. **Analyse.** `.venv/bin/python docs/experiments/nec_ladder_g25/analyze.py`: writes `results.md`,
   `per_run.csv` and `learning_curves.png` into this folder.
6. **Report.** Write `REPORT.md`, with these sections:
   - a recap of this protocol;
   - the hash check of the frozen files;
   - the pilot gate outcome;
   - **the assumption checks (A1–A9), with what held**;
   - results and hypothesis verdicts;
   - the decomposition answer to the question in section 1;
   - exploratory observations, deviations, limitations, and a conclusion.
