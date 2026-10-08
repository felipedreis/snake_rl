# Experiment protocol (v2): which part of NEC makes it data-efficient? Paper exploration schedules, evaluation runs

Status: **pre-registered.** The design decisions were made with the experimenter on 2026-10-06. This
document, `run.sh` and `analyze.py` are frozen *after* the pilot gate (section 10) and before any
confirmatory run. Their SHA-256 hashes go in `FROZEN.sha256`, and any departure from this plan is listed
under "Deviations" in the report.

## 0. What changed since v1, and why

v1 (`docs/experiments/nec_ladder_g25/`) asked the same question on the same environment. It stopped at
its pilot gate: no agent learned to collect food within 150k or 300k steps (its REPORT.md). Reviewing it
against the papers exposed two design flaws, both fixed here:

1. **Exploration did not match the papers.** v1 gave every agent run.py's schedule, ε 1 → 0.02 over 5,000
   steps. The papers differ, both from that schedule and from each other:
   - **DQN** (Mnih et al., 2013, verified in the text): "ε annealed linearly from 1 to 0.1 over the first
     million frames, and fixed at 0.1 thereafter". With frame skip k = 4, that is **250,000 agent steps**,
     50× longer than v1's decay, with a 5× higher floor.
   - **MFEC** (Blundell et al., 2016, verified): "an ε-greedy policy with ε = 0.005", fixed, with no
     decay. Higher rates "were not as beneficial".
   - **NEC** (Pritzel et al., 2017, verified): "ε-greedy policy during training with a low ε". ε was
     tuned, but **its value is not reported**. We use MFEC's 0.005 (section 3, assumption A8).

   v2 gives each agent its paper's schedule. run.py gained `--eps-decay N` for this (linear from 1 to the
   floor over N steps; `0` = fixed at the floor). With the option unset, existing runs reproduce
   bit-for-bit (verified on committed DQN, MFEC and NEC runs).
2. **Scores were measured on training episodes.** The papers score separate evaluation runs: DQN reports
   "an ε-greedy policy with ε = 0.05 for a fixed number of steps", and NEC evaluates every 200,000 frames.
   Under DQN's long decay, training episodes in our EARLY window are played at ε ≈ 0.3–1, so they mostly
   measure random acting, not what was learned. v2 measures every outcome on **evaluation runs**:
   - every 10,000 training steps, 5 episodes at ε = 0.05;
   - on a separate environment copy (seed + 2000) with a separate random stream (seed + 3000);
   - with the agent's learning bookkeeping off.

   A test shows that training is bit-for-bit identical with and without evaluation, for every
   registered agent.
3. **Budget:** S = 1,000,000 steps (≈ 4M frames, the NEC paper's third reported point), so DQN has
   750k steps after its decay.

Everything else (question, ladder, hypotheses, tests, assumption checks) is as in v1.

## 1. Question

NEC (Pritzel et al., 2017) is claimed to learn from far fewer samples than DQN. It differs from DQN in
three ways at once:
- **N-step returns:** targets use 50 real rewards before bootstrapping, not 1.
- **Episodic memory:** Q is read from a per-action table of past (situation, return) pairs by
  kernel-weighted nearest neighbours, so a good return is usable as soon as it is written.
- **A learned embedding:** the situation key comes from an encoder trained by gradient descent, so
  similar situations get nearby keys.

**Which of these components produces NEC's gain in data efficiency, when each algorithm runs under the
conditions of its paper?**

## 2. Environment (unchanged from v1)

- **Board:** 25×25, map `rooms`. A cross-shaped wall splits the board into four rooms, joined only by a
  three-cell opening at the centre. Walls are lethal.
- **Bonus food worth 5 points:** after every 4 regular fruit in an episode, a bonus pellet appears for
  2·n = 50 steps.
- **Rewards:** +1 for regular fruit, +5 for a bonus pellet, −1 for death (which ends the episode),
  0 otherwise.
- **Truncation:** an episode is also cut off after 2·n² = 1,250 steps without eating.
- **Score:** the points in an episode.
- **Observation:** five 25×25 channels (head, body, food, walls, bonus lifetime), 3,125 inputs. No
  noise channels.

v1's pilot found that bonus pellets never appeared, because no agent ate 4 fruit in one episode. The bonus
rule is kept unchanged on purpose: v2 changes only the exploration and the measurement, so that any
difference from v1 can be attributed to them. Assumption check A4 reports whether bonus food came
into play.

## 3. Agents and exploration

All agents use MLP encoders and repo defaults, with no tuning (see v1's PROTOCOL section 3 for every
setting).

| Rung | Name in code | What it adds | ε during training | Source of the schedule |
|---|---|---|---|---|
| 0 | `dqn` | — | 1 → 0.1 linearly over 250k steps, then 0.1 | DQN paper |
| 1 | `dqn_nstep` | N-step returns (N = 50) | 1 → 0.1 over 250k steps, then 0.1 | DQN paper (it is a DQN variant) |
| 2 | `ec_frozen` | episodic memory, random frozen encoder | fixed 0.005 | NEC paper's "low ε", value from MFEC |
| 3 | `nec` | learned encoder | fixed 0.005 | NEC paper's "low ε", value from MFEC |
| ref | `mfec` | random projection + max-return table | fixed 0.005 | MFEC paper |
| ref | `random` | uniform random actions | — | — |

**Evaluation runs** use ε = 0.05 for every agent, as in the DQN paper. The random policy is evaluated
the same way, and stays random.

**Important consequence for interpretation.** Under paper conditions, the contrasts between a DQN rung
and an episodic rung also change the exploration schedule:
- **H0, H2 and H4** compare an agent that explores broadly for 250k steps (DQN family) with one that
  almost never explores (ε = 0.005). They measure "algorithm *with its paper's exploration*", which is
  the comparison the papers themselves make.
- **H1 and H3** hold exploration fixed. They isolate N-step returns and the learned embedding cleanly.

A planned follow-up runs NEC and MFEC with DQN's decay to separate the algorithm from its exploration
schedule. It is not part of this protocol.

## 4. Design

- **One configuration:** size 25, `rooms`, bonus 5, D = 0, S = 1,000,000 steps.
- **Replication:** 15 runs per agent, seeds 201–215. The agent's random stream is `seed`; the
  environments are seeded with `seed + 1000` (training) and `seed + 2000` (evaluation); evaluation
  exploration draws from `seed + 3000`.
- **Total:** 90 confirmatory runs and 18 pilot runs (seeds 1–3), all executed by `run.sh`.

**Seeds and data seen before freezing:**
- Seeds 201–215 have never been run under any condition.
- Seeds 1–3 were used by v1's pilot under v1's conditions. They are used again here as pilot seeds only.
- While debugging `analyze.py`, throwaway 60k-step runs on seeds 901–902 were checked for structure
  only. Some agents had zero-variance evaluation scores there, which is all that was seen of their
  values.

## 5. Data collected

For every run, `snake-run` writes one JSON file containing:
- `evaluations`: per evaluation point, the training step and 5 episodes of (score, regular fruit,
  truncated, length).
- `episodes`: training episodes as (end step, score, regular fruit, truncated). Descriptive only.
- `diagnostics`: memory counters every 6,000 steps, for memory agents.
- `eval`: the evaluation settings.
- The run settings: agent, seed, steps, size, D, map, bonus, ε floor and ε decay.
- `wallclock_s`: recorded, not analysed.

## 6. Outcome measures (per run; the run is the unit of analysis)

- **Primary, EARLY (data efficiency):** the mean evaluation score over all evaluation episodes at
  training steps (0, S/5] = (0, 200k]. That is 20 evaluation points × 5 episodes = 100 episodes per run.
  200k steps ≈ 800k frames, just below the NEC paper's first reported point.
- **Secondary, LATE:** the same over (S/2, S] = (500k, 1M], 250 episodes per run (≈ 2M–4M frames).
- **Secondary, AUC:** the mean, over all 100 evaluation points, of each point's mean score.
- **Robustness:** EARLY and LATE recomputed on regular fruit only, without bonus points.
- **Descriptive:**
  - evaluation learning curves;
  - LATE score on training episodes against evaluation (shows the cost of each agent's own exploration);
  - evaluation episode length and truncation share;
  - share of points from bonus food;
  - memory diagnostics.

## 7. Hypotheses (confirmatory, all on EARLY)

"A > B" means the population mean of EARLY is higher for A. Directions follow the papers.

| ID | Hypothesis | Factor | Rationale | Prior confidence |
|---|---|---|---|---|
| H0 | NEC > DQN | whole package, each with its paper's exploration | NEC's central claim. In the NEC paper (Table 1, median across games), NEC is far ahead of DQN at 1M frames (16.7% vs −0.7% human-normalised). | moderate |
| H1 | N-step DQN > DQN | N-step returns (same exploration) | A 50-step target carries a food reward back in one update; a 1-step target needs many. | high |
| H2 | Frozen-embedding EC > N-step DQN | episodic memory with a random embedding, plus its paper's exploration | Table writes make a good return usable at once. | **low**: exact repeats are rare on 25×25, and with ε = 0.005 the agent may rarely find food to write. |
| H3 | NEC > Frozen-embedding EC | learned embedding (same exploration) | When states rarely repeat, kNN only helps if similar situations get nearby keys. | high |
| H4 | NEC > N-step DQN | episodic memory given a learned representation, plus its paper's exploration | NEC's read-out exploits new returns at once. | moderate |

**Reading the answer:**
- **Which factors contribute:** the steps whose hypothesis is supported.
- **How much:** the decomposition of NEC − DQN into the three ladder steps, each with a bootstrap CI and
  its share of the total. This is descriptive, and only interpreted if H0 is supported.
- **Caveat (section 3):** in this design the exploration change sits inside the
  "dqn_nstep → ec_frozen" step.

LATE, AUC, the regular-fruit outcomes, MFEC comparisons and all descriptives are **exploratory**,
reported with uncorrected p-values.

## 8. Statistical analysis (`analyze.py`; unchanged from v1 except the outcome source)

- **Test:** a two-sided permutation test on the difference in means of EARLY, 15 runs against 15, with
  200,000 random relabelings (seed 0). p = (1 + #{|permuted diff| ≥ |observed diff|}) / 200,001.
- **Multiple comparisons:** Holm–Bonferroni over H0–H4, family-wise α = 0.05.
- **Decision rule:**
  - **Supported:** Holm-adjusted p < 0.05 with the predicted sign.
  - **Contradicted:** p < 0.05 with the opposite sign.
  - **Not supported:** otherwise. This is not evidence of no difference.
- **Effect sizes:** the difference in means with a 95% percentile bootstrap CI (10,000 resamples,
  seed 0), and Hedges' g.
- **Missing data:** a crashed run is rerun once with the same seed. If it fails again, it is excluded
  and reported.
- **Power** (n = 15 per group): 0.50 at g = 1.0 and 0.90 at g = 1.5, at Holm's strictest level
  α = 0.01; 0.75 and 0.98 at α = 0.05.

## 9. Assumptions, and how the report checks each one

`analyze.py` writes each check to `results.md`. The report states whether each one held, and what that
means for the verdicts. A failed check never changes a verdict; it changes how the verdict is interpreted.

| # | Assumption | Check | If it fails |
|---|---|---|---|
| A1 | Every ladder agent learns within the budget. | Evaluation LATE vs `random` (permutation test, uncorrected). | Hypotheses involving that agent are "uninterpretable: agent did not learn". |
| A2 | Every evaluation point was recorded. | Evaluation episodes per window: exactly 100 (EARLY) and 250 (LATE) per run. | Incomplete runs are reported. |
| A3 | Score reflects food-finding, not idling. | Evaluation episode length and the share cut off by the idle limit. | Discussed as a limitation. |
| A4 | Bonus food comes into play. | Share of points from bonus and pellets per episode, plus the regular-fruit-only robustness tests. | If ≈ 0, the reward-size motivation did not materialise (as in v1). If the robustness tests disagree, the verdicts depend on bonus weighting. |
| A5 | The memory regime is understood. | DND fill, eviction share, exact-match share, first eviction. | Descriptive. |
| A6 | Runs are independent. | By design: separate seeds and processes. | Stated. |
| A7 | No single run drives a verdict. | Leave-one-out retests against each Holm threshold. | Fragile verdicts are flagged. |
| A8 | NEC's unreported ε is close to 0.005. | Not checkable from the paper. | Stated. The follow-up varies NEC's exploration. |
| A9 | Evaluation does not alter training. | Unit test `test_evaluation_does_not_change_training`, run for every agent. | — |
| A10 | Runs are reproducible. | One BLAS thread per run (`run.sh`). Multi-threaded BLAS was observed to change runs. | — |

## 10. Procedure

1. **Pilot gate.** `bash docs/experiments/nec_ladder_g25_paper_eps/run.sh pilot` (seeds 1–3, S = 1M),
   then `analyze.py --gate`.
   - A ladder agent **clears the floor** if its mean evaluation LATE is at least 0.5 points above
     `random`'s.
   - **Gate passes** if at least 3 of the 4 ladder agents clear it.
   - **If it fails, the experiment stops** (no budget increase: 2M steps would take about 4 h per NEC
     run), and the report explains why.
   - Pilot results can change nothing else.
2. **Freeze.** Record the hashes of this protocol, `run.sh` and `analyze.py` in `FROZEN.sha256`, with
   the time.
3. **Run.** `bash docs/experiments/nec_ladder_g25_paper_eps/run.sh main`: 90 runs, 8 in parallel, logged
   to `logs/exp_nec_ladder_paper_eps_main.log`. Estimated 9–10 h (pilot: about 3 h).
4. **Check** that 90 files exist, and rerun failures per section 8.
5. **Analyse.** `.venv/bin/python docs/experiments/nec_ladder_g25_paper_eps/analyze.py`.
6. **Report.** `REPORT.md`:
   - a recap and hash check;
   - the gate outcome;
   - the assumption checks A1–A10;
   - results and verdicts;
   - the decomposition answer, read with section 3's caveat;
   - exploratory observations, deviations, limitations and a conclusion.
