# Experiment protocol (v4): which part of NEC makes it data-efficient? Curriculum with held radius and food relocation

Status: **pre-registered.** The design was approved in advance by the experimenter on 2026-10-07 (option
"B": relocate the food, plus a hold-then-grow schedule), and this protocol was committed before the
pilot started. This document, `run.sh` and `analyze.py` are frozen *after* a passed pilot gate (section
10) and before any confirmatory run. Their SHA-256 hashes go in `FROZEN.sha256`, and any departure is
listed under "Deviations" in the report.

## 0. Where v4 comes from

All earlier versions ask the question below on a 25×25 `rooms` board with bonus food, and v1–v3 stopped
at the pilot gate:

| Version | Change | Pilot gate |
|---|---|---|
| v1 | one short ε schedule for every agent; scores from training episodes | failed |
| v2 | each paper's ε schedule; scores from evaluation runs at ε 0.05; 1M steps | failed |
| v3 | v2 + a food-distance curriculum (`--food-curriculum 2:600000`) | failed (see its REPORT.md) |

v3's pilot showed that its curriculum had **two flaws**:
1. **Wrong timing for the DQN family.** DQN's ε falls from 1 to 0.1 over the first 250k steps, but v3's
   easy phase (radius 2 → 13) was the first ~137k steps. So the DQN family acted almost randomly exactly
   while food was close. Its training fruit in the first 100k steps (~1,700) matched the random policy's
   (~1,600).
2. **Nearby food is left behind.** v3 placed food near the head only *at the moment of placement*, at an
   episode start or right after eating. The near-greedy episodic agents (ε = 0.005) do not head for food
   yet, so they wandered off, left it behind, and drifted for up to 1,250 steps until the idle limit
   ended the episode. They collected only 63–120 fruit in the first 100k steps, against ~1,600 for random,
   whose ~8-step episodes give it constant fresh placements.

**v4 fixes both, and changes nothing else:**
1. **Hold, then grow:** `--food-curriculum 2:250000:600000`. The radius stays at 2 until step 250,000,
   which is when DQN's ε decay ends. It then grows linearly to 2n = 50 by step 600,000; after that, food
   goes anywhere, as in v3 and the real game. Every agent gets a long easy phase while acting
   (near-)greedily, the episodic agents included.
2. **Food relocation:** `--food-relocate`. While the curriculum is active, if the snake has not eaten
   within a patience of **2r + 5 steps** of the food being placed, the food is placed again within r
   walkable steps of the head's *current* position. The episode goes on, and the snake keeps its length
   and position.
   - A direct path to the food is at most r steps, so 2r + 5 leaves room for detours around walls and the
     body.
   - The idle limit (1,250 steps without eating) still applies.

Everything else is exactly v3, and therefore v2:
- the environment and rewards;
- the exploration schedules;
- the evaluation runs, which always place food anywhere and never relocate it;
- the budget, ladder, hypotheses, tests, assumption checks and gate.

## 1. Question

NEC (Pritzel et al., 2017) adds three components to a DQN-like learner: N-step returns, an episodic
memory read by kernel-weighted nearest neighbours, and a learned embedding for that memory's keys.

**Which of these components produces NEC's gain in data efficiency, when every agent is trained with the
same held, relocating food curriculum and scored on the real task?**

As in v3, none of the papers trains with a curriculum. The results speak to data efficiency *given* this
curriculum.

## 2. Environment

The board, rewards and observation are unchanged:
- **Board:** 25×25, map `rooms`.
- **Bonus food:** worth 5 points, appearing after 4 regular fruit in one episode, uniformly placed, for
  50 steps.
- **Rewards:** +1 for regular fruit, +5 for a bonus pellet, −1 for death, 0 otherwise.
- **Idle limit:** an episode is cut off after 1,250 steps without eating.
- **Observation:** five 25×25 channels. No noise channels.

**Training curriculum (new schedule, new relocation rule):**

| Training steps | Food radius r | Relocation patience 2r + 5 | |
|---|---|---|---|
| 0 – 250,000 | 2 | 9 steps | held: food always within 2 steps |
| ~330,000 | 13 | 31 | the mean start distance under uniform placement |
| 400,000 | 22 | 49 | |
| 500,000 | 36 | 77 | start of the LATE window |
| 600,000 onwards | anywhere | no relocation | the real game |

- **What r counts:** walkable steps around walls and the snake's body. If no free cell is within r, food
  goes anywhere.
- **Bonus pellets** are not part of the curriculum, and are never relocated.
- **Evaluation runs** never use the curriculum or relocation, so outcomes measure the real task.

**What the schedule means for the outcome windows:**
- **EARLY** (evaluations at steps 0–200k) falls entirely inside the held phase. It measures how fast
  agents learn to eat when food is always 2 steps away, *as it transfers to the real game*.
- **LATE** (500k–1M) is 80% after the curriculum (600k–1M), and its first 100k has radius 36–50.

The windows were kept identical to v1–v3 deliberately, so the versions stay comparable.

## 3. Agents and exploration (unchanged)

| Rung | Name in code | What it adds | ε during training |
|---|---|---|---|
| 0 | `dqn` | — | 1 → 0.1 over 250k steps, then 0.1 |
| 1 | `dqn_nstep` | N-step returns (N = 50) | as `dqn` |
| 2 | `ec_frozen` | episodic memory, random frozen encoder | fixed 0.005 |
| 3 | `nec` | learned encoder | fixed 0.005 |
| ref | `mfec` | random projection + max-return table | fixed 0.005 |
| ref | `random` | uniform random actions | — |

- **Shared settings:** all agents use MLP encoders and repo defaults, with no tuning, and the same
  curriculum (random included). Evaluation uses ε = 0.05.
- **Exploration differs in H0, H2 and H4:** those contrasts also compare exploration schedules; H1 and H3
  hold exploration fixed.
- **The DQN family's easy phase:** with the hold, the DQN family spends 250k steps at radius 2 while its ε
  falls, and food stays at radius 2 until its ε has reached 0.1.

## 4. Design

- **One configuration:** size 25, `rooms`, bonus 5, D = 0,
  `--food-curriculum 2:250000:600000 --food-relocate`, S = 1,000,000 steps.
- **Replication:** 15 runs per agent, seeds **401–415** (never run). Pilot: seeds 1–3, 18 runs.
- **Seeding:** the agent's stream is `seed`, the training env `seed + 1000`, the evaluation env
  `seed + 2000`, and evaluation exploration `seed + 3000`. Relocation draws from the training env's
  stream, and only when it is active.
- **Total:** 90 confirmatory runs and 18 pilot runs, by `run.sh`.

**Data seen before freezing:**
- the v1–v3 pilots (seeds 1–3, no relocation), all fully analysed in their reports;
- unit tests of the schedule and of relocation;
- **one smoke test of the mechanism:** MFEC, seed 0, 100k steps, with v3's curriculum vs v4's (held
  radius plus relocation). With relocation it ate 2,835 regular fruit in 100k steps (852 episodes, no idle
  cut-offs, 5 bonus pellets). Without it, 84 fruit (199 episodes, 14% idle cut-offs, 0 bonus pellets).
  This shows the mechanism works as intended. It is training-episode data from one non-pilot seed, and it
  says nothing about evaluation scores;
- throwaway 60k-step runs on seeds 901–902, checked for structure only.

## 5. Data collected

As in v3. The run settings now include `food_curriculum: [2, 250000, 600000]` and
`food_relocate: true`, and result files are tagged `_fc2-250000-600000_reloc`.

Training episodes keep their `(t, score, regular foods, truncated)` records. Relocations are not counted
separately: they are a fixed function of the schedule and the snake's behaviour.

## 6. Outcome measures (unchanged)

- **Primary, EARLY:** the mean evaluation score over evaluation episodes at training steps (0, 200k],
  with food anywhere.
- **Secondary, LATE:** the same over (500k, 1M].
- **Secondary, AUC:** the mean of the per-evaluation-point means.
- **Robustness:** EARLY and LATE on regular fruit only.
- **Descriptive:** as in v3, including training fruit per tenth of the budget (A11).

## 7. Hypotheses (confirmatory, all on EARLY; unchanged)

| ID | Hypothesis | Factor | Prior confidence |
|---|---|---|---|
| H0 | NEC > DQN | whole package (with each paper's exploration) | moderate |
| H1 | N-step DQN > DQN | N-step returns | high |
| H2 | Frozen-embedding EC > N-step DQN | episodic memory with a random embedding (with each paper's exploration) | low |
| H3 | NEC > Frozen-embedding EC | learned embedding | high |
| H4 | NEC > N-step DQN | episodic memory given a learned representation (with each paper's exploration) | moderate |

**What the new curriculum may change.** During EARLY, the DQN family is still exploring heavily
(ε 1 → 0.28), while the episodic agents act greedily on food that relocation keeps within 2 steps. That
favours the episodic rungs in EARLY *through the exploration difference*, which is section 3's caveat in
its sharpest form. H1 and H3 are unaffected by it.

## 8. Statistical analysis (unchanged)

- **Test:** two-sided permutation tests on the difference in EARLY means, 15 against 15, with 200,000
  relabelings (seed 0).
- **Multiple comparisons:** Holm over H0–H4, α = 0.05.
- **Decision rule:**
  - **Supported:** Holm-adjusted p < 0.05 with the predicted sign.
  - **Contradicted:** p < 0.05 with the opposite sign.
  - **Not supported:** otherwise.
- **Effect sizes:** bootstrap CIs and Hedges' g.
- **Missing data:** a crashed run is rerun once with the same seed. If it fails again, it is excluded and
  reported.
- **Power:** 0.90 at g = 1.5, at Holm's strictest level.

## 9. Assumptions and their checks (A1–A11, as in v3)

| # | Assumption | Check |
|---|---|---|
| A1 | Every ladder agent learns the real task. | Evaluation LATE vs `random`. |
| A2 | Every evaluation point was recorded. | 100 and 250 evaluation episodes per run. |
| A3 | Score reflects food-finding, not idling. | Evaluation episode length and idle-limit share. |
| A4 | Bonus food comes into play. | Bonus share, plus the regular-fruit robustness tests. |
| A5 | The memory regime is understood. | DND fill, evictions, exact matches. |
| A6 | Runs are independent. | By design. |
| A7 | No single run drives a verdict. | Leave-one-out retests. |
| A8 | NEC's unreported ε is close to 0.005. | Not checkable. |
| A9 | Evaluation does not alter training. | Unit test. |
| A10 | Runs are reproducible. | One BLAS thread. |
| A11 | The curriculum provides a learning signal. | Training fruit per tenth of the budget. With relocation, every learner should now collect clearly more than v3's episodic agents (63–120 in the first 100k steps). |

Failing checks change interpretation, never verdicts, as in v3.

## 10. Procedure

1. **Pilot gate.** `bash docs/experiments/nec_ladder_g25_food_relocation/run.sh pilot` (seeds 1–3),
   then `analyze.py --gate`.
   - A ladder agent **clears the floor** if its mean evaluation LATE is at least 0.5 points above
     `random`'s.
   - **Gate passes** if at least 3 of the 4 ladder agents clear it.
   - **If it fails, the experiment stops.** The report says whether agents learned during the curriculum
     (A11) but failed to transfer to the real game, or never learned at all.
2. **Freeze:** `FROZEN.sha256` with the hashes of this protocol, `run.sh` and `analyze.py`.
3. **Run.** `bash docs/experiments/nec_ladder_g25_food_relocation/run.sh main`: 90 runs, 8 in parallel.
   - **Time:** about 2–3 h for the pilot and 9–10 h for the main stage.
   - **If agents learn, runs take longer:** evaluation episodes get longer.
4. **Check** that 90 files exist, and rerun failures.
5. **Analyse.** `.venv/bin/python docs/experiments/nec_ladder_g25_food_relocation/analyze.py`.
6. **Report.** `REPORT.md`:
   - a recap and hash check;
   - the gate outcome;
   - A1–A11;
   - verdicts;
   - the decomposition, read with section 3's caveat;
   - exploratory observations, deviations, limitations and a conclusion.
