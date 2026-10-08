# Experiment protocol (v3): which part of NEC makes it data-efficient? Trained with a food-distance curriculum

Status: **pre-registered.** Reviewed and approved by the experimenter on 2026-10-07, and committed before
the pilot started. This document, `run.sh` and `analyze.py` are frozen *after* a passed pilot gate
(section 10) and before any confirmatory run. Their
SHA-256 hashes go in `FROZEN.sha256`, and any departure is listed under "Deviations" in the report.

## 0. Where v3 comes from

- **v1** (`docs/experiments/nec_ladder_g25/`): one short ε schedule for every agent, scores from training
  episodes, up to 300k steps.
- **v2** (`docs/experiments/nec_ladder_g25_paper_eps/`): each agent's paper exploration schedule, scores
  from evaluation runs, 1M steps.

Both asked the question below on a 25×25 `rooms` board with bonus food, and **both stopped at the pilot
gate**. No agent learned to collect food: the best evaluation LATE was 0.084, against the 0.509 needed.
Bonus food never appeared.

v2's report locates the bottleneck: **food is found too rarely for any value estimate to learn that it
matters**. Food is placed uniformly at random. From the start position it is on average 12.9 walkable
steps away, and only 7% of placements are within 5 steps (92% on 7×7). Agents that explore broadly
(the DQN family, ε → 0.1) die before reaching it. Agents that explore little (the episodic agents,
ε = 0.005) wander until the idle limit cuts them off.

**v3 changes one thing: it trains with a food-distance curriculum** (`--food-curriculum`, commit
`d6739d5`). During training, regular food is placed within *r* walkable steps of the snake's head, and
*r* grows over training until food can appear anywhere, as in the real game. This is the idea of reverse
curriculum generation (Florensa et al., 2017): start near the goal, then move it away. Everything else is
exactly v2:
- the environment;
- the exploration schedules;
- the evaluation runs, which always place food anywhere, so they measure the real task;
- the budget, ladder, hypotheses, tests and assumption checks.

**What the curriculum changes about the question.** None of the papers trains with a curriculum, so v3
is no longer "each algorithm under its paper's conditions". It is "each algorithm under its paper's
exploration, on a task made learnable by a curriculum, measured on the real task". The results speak to
data efficiency *given* a curriculum. Whether the ranking would hold without one is something v1 and v2
could not measure, because nothing learned.

## 1. Question

NEC (Pritzel et al., 2017) is claimed to learn from far fewer samples than DQN. It adds three components
to a DQN-like learner: N-step returns, an episodic memory read by kernel-weighted nearest neighbours, and
a learned embedding for that memory's keys.

**Which of these components produces NEC's gain in data efficiency, when every agent is trained with
the same food-distance curriculum and scored on the real task?**

## 2. Environment

The board and the scoring are unchanged from v2:
- **Board:** 25×25, map `rooms`. A cross-shaped wall splits the board into four rooms, joined by a
  three-cell opening at the centre. Walls are lethal.
- **Bonus food worth 5 points:** a pellet appears after every 4 regular fruit in one episode, on a
  uniformly random free cell, for 50 steps.
- **Rewards:** +1 for regular fruit, +5 for a bonus pellet, −1 for death, 0 otherwise.
- **Truncation:** an episode is also cut off after 1,250 steps without eating.
- **Score:** the points in an episode.
- **Observation:** five 25×25 channels (head, body, food, walls, bonus lifetime). No noise channels.

**Training curriculum** (new): `--food-curriculum 2:600000`.

| Training steps | Food radius *r* (walkable steps from the head) | |
|---|---|---|
| 0 | 2 | food right next to the snake |
| 137,500 | 13 | the mean start distance under uniform placement |
| 200,000 | 18 | end of the EARLY window |
| 275,000 | 24 | every cell reachable from the start position |
| 500,000 | 42 | start of the LATE window; effectively uniform already |
| 600,000 onwards | anywhere | the real game, for the rest of training |

- **How *r* grows:** linearly from 2 to 2n = 50 over 600k steps (`run.food_radius`).
- **What *r* counts:** the shortest walkable path around walls and the snake's own body.
- **No close cell:** if no free cell is within *r*, food goes anywhere.
- **Bonus pellets** are not part of the curriculum: they always appear anywhere. With nearby regular food,
  eating 4 fruit in one episode becomes plausible, so bonus food may finally come into play (check A4).

**Evaluation runs** always use uniform placement. So the EARLY evaluations (up to 200k steps) measure how
well skills learned on nearby food transfer to the real task, and LATE (500k–1M) measures the real task
after the curriculum is over.

**Why 2:600000.** The radius passes the mean food distance inside EARLY, and the curriculum is over
before LATE's second half, so LATE mostly measures the real game. The schedule is a judgement call, not
tuned; it was fixed before any curriculum run with learning agents. The only curriculum runs so far are
two 100k-step random-policy smoke tests and throwaway 60k-step runs used to debug `analyze.py` (section 4).

## 3. Agents and exploration (unchanged from v2)

| Rung | Name in code | What it adds | ε during training |
|---|---|---|---|
| 0 | `dqn` | — | 1 → 0.1 over 250k steps, then 0.1 (DQN paper) |
| 1 | `dqn_nstep` | N-step returns (N = 50) | as `dqn` |
| 2 | `ec_frozen` | episodic memory, random frozen encoder | fixed 0.005 (NEC: "a low ε"; value from MFEC) |
| 3 | `nec` | learned encoder | fixed 0.005 |
| ref | `mfec` | random projection + max-return table | fixed 0.005 (MFEC paper) |
| ref | `random` | uniform random actions | — |

- **Settings:** all agents use MLP encoders and repo defaults, with no tuning.
- **Curriculum:** every agent trains with the same curriculum, `random` included, so the files and
  filters are uniform. For `random` it changes nothing that is measured, because evaluation is uniform.
- **Evaluation:** ε = 0.05 for every agent.

**As in v2, H0, H2 and H4 also compare exploration schedules;** H1 and H3 hold exploration fixed (v2
PROTOCOL section 3).

## 4. Design

- **One configuration:** size 25, `rooms`, bonus 5, D = 0, `--food-curriculum 2:600000`, S = 1,000,000
  steps.
- **Replication:** 15 runs per agent. Seeds **301–315**, never run under any condition. Seeds 201–215
  were reserved for v2's main stage, which never ran; new seeds keep the protocols apart.
- **Seeding:** the agent's stream is `seed`, the training env `seed + 1000`, the evaluation env
  `seed + 2000`, and evaluation exploration `seed + 3000`.
- **Total:** 90 confirmatory runs, plus 18 pilot runs (seeds 1–3), by `run.sh`.

**Data seen before freezing:**
- Seeds 1–3 were used by the v1 and v2 pilots, without a curriculum.
- Two 100k-step random-policy smoke tests (seed 0) showed that the curriculum places food close: 315
  fruit in the first 25k steps, against 33 without it.
- Throwaway 60k-step runs on seeds 901–902 were checked for structure only.

## 5. Data collected

As in v2: `evaluations`, `episodes` (training), `diagnostics`, `eval` and the run settings. The settings
now include `food_curriculum: [2, 600000]`, and result files are tagged `_fc2-600000`.

## 6. Outcome measures (per run; unchanged from v2)

- **Primary, EARLY (data efficiency):** the mean evaluation score over all evaluation episodes at
  training steps (0, 200k]: 100 episodes per run, food placed anywhere.
- **Secondary, LATE:** the same over (500k, 1M]: 250 episodes.
- **Secondary, AUC:** the mean, over all 100 evaluation points, of each point's mean score.
- **Robustness:** EARLY and LATE on regular fruit only.
- **Descriptive:**
  - evaluation learning curves, with the curriculum milestones marked;
  - training vs evaluation LATE score;
  - episode endings;
  - bonus share;
  - memory diagnostics;
  - **new:** training fruit per tenth of the budget (A11).

## 7. Hypotheses (confirmatory, all on EARLY; unchanged from v2)

| ID | Hypothesis | Factor | Prior confidence |
|---|---|---|---|
| H0 | NEC > DQN | whole package (with each paper's exploration) | moderate |
| H1 | N-step DQN > DQN | N-step returns (same exploration) | high |
| H2 | Frozen-embedding EC > N-step DQN | episodic memory with a random embedding (with each paper's exploration) | low |
| H3 | NEC > Frozen-embedding EC | learned embedding (same exploration) | high |
| H4 | NEC > N-step DQN | episodic memory given a learned representation (with each paper's exploration) | moderate |

The rationales are v2's (v2 PROTOCOL section 7). The curriculum's effect on each:
- **H1 likely gains:** with food 2–13 steps away, a 50-step return already reaches the reward from the
  start of an approach.
- **H2's low prior stays:** nearby food gives the near-greedy episodic agents rewards to write. But a
  random embedding of raw positions still matches poorly once food moves away.
- **H3 should be clearest exactly here:** during the curriculum, similar situations (food a few steps
  ahead, in different places) recur in new positions. A learned embedding can group them; a random one
  cannot.

**Reading the answer:** the factors whose hypotheses are supported, and the descriptive decomposition of
NEC − DQN into the three ladder steps (only if H0 is supported), read with section 3's caveat.

LATE, AUC, the regular-fruit outcomes, MFEC comparisons and all descriptives are exploratory, reported
with uncorrected p-values.

## 8. Statistical analysis (unchanged from v2)

- **Test:** a two-sided permutation test on the difference in means of EARLY, 15 against 15, with
  200,000 relabelings (seed 0).
- **Multiple comparisons:** Holm–Bonferroni over H0–H4, α = 0.05.
- **Decision rule:**
  - **Supported:** Holm-adjusted p < 0.05 with the predicted sign.
  - **Contradicted:** p < 0.05 with the opposite sign.
  - **Not supported:** otherwise.
- **Effect sizes:** the difference in means with a 95% bootstrap CI (10,000 resamples, seed 0), and
  Hedges' g.
- **Missing data:** a crashed run is rerun once with the same seed. If it fails again, it is excluded and
  reported.
- **Power** (n = 15 per group): 0.50 at g = 1.0 and 0.90 at g = 1.5, at Holm's strictest level α = 0.01.

## 9. Assumptions and their checks (v2's A1–A10, plus A11)

| # | Assumption | Check | If it fails |
|---|---|---|---|
| A1 | Every ladder agent learns the real task. | Evaluation LATE vs `random` (permutation test). | Hypotheses involving that agent are "uninterpretable: agent did not learn". |
| A2 | Every evaluation point was recorded. | Exactly 100 (EARLY) and 250 (LATE) evaluation episodes per run. | Incomplete runs are reported. |
| A3 | Score reflects food-finding, not idling. | Evaluation episode length and idle-limit share. | Discussed. |
| A4 | Bonus food comes into play. | Bonus share and pellets per episode, plus the regular-fruit robustness tests. | If ≈ 0, a limitation. If the robustness tests disagree, the verdicts depend on bonus weighting. |
| A5 | The memory regime is understood. | DND fill, evictions, exact matches. | Descriptive. |
| A6 | Runs are independent. | By design. | Stated. |
| A7 | No single run drives a verdict. | Leave-one-out retests. | Fragile verdicts are flagged. |
| A8 | NEC's unreported ε is close to 0.005. | Not checkable. | Stated. |
| A9 | Evaluation does not alter training. | Unit test, for every agent. | — |
| A10 | Runs are reproducible. | One BLAS thread per run. | — |
| **A11** | **The curriculum provides a learning signal.** | Training fruit per 100k-step block. Expect clearly more than v2 (2–280 per 200k) while *r* is small. | If not, the curriculum did not do its job, and the report says so. |

## 10. Procedure

1. **Pilot gate.** `bash docs/experiments/nec_ladder_g25_food_curriculum/run.sh pilot` (seeds 1–3), then
   `analyze.py --gate`.
   - A ladder agent **clears the floor** if its mean evaluation LATE (food anywhere, curriculum over) is
     at least 0.5 points above `random`'s.
   - **Gate passes** if at least 3 of the 4 ladder agents clear it.
   - **If it fails, the experiment stops,** and the report explains why. In particular, it says whether
     agents learned during the curriculum but failed to transfer (A11 high, evaluation low), or never
     learned at all.
   - Pilot results can change nothing else.
2. **Freeze:** `FROZEN.sha256` with the hashes of this protocol, `run.sh` and `analyze.py`, and the time.
3. **Run.** `bash docs/experiments/nec_ladder_g25_food_curriculum/run.sh main`: 90 runs, 8 in parallel.
   - **Estimated time:** about 2–3 h for the pilot and 9–10 h for the main stage, as for v2 (v2's pilot
     took 2 h 05 min).
   - **If agents learn, runs take longer:** evaluation episodes get longer, because eating resets the idle
     limit.
4. **Check** that 90 files exist, and rerun failures per section 8.
5. **Analyse.** `.venv/bin/python docs/experiments/nec_ladder_g25_food_curriculum/analyze.py`.
6. **Report.** `REPORT.md`:
   - a recap and hash check;
   - the gate outcome;
   - the assumption checks A1–A11;
   - results and verdicts;
   - the decomposition answer, read with section 3's caveat;
   - exploratory observations, deviations, limitations and a conclusion.
