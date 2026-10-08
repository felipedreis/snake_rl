# Experiment protocol: can the encoder tell where the food is? MLP vs CNN, NEC vs DQN, matched exploration

Status: **pre-registered (exploratory).** Written on 2026-10-07 by Claude, on the experimenter's request and
while they were away, so the design choices below (marked *decision*) were made without their review.
`PROTOCOL.md`, `run.sh` and `analyze.py` were hashed into `FROZEN.sha256` before the first run. Any departure is
listed under "Deviations" in the report.

## 0. Where this comes from

NEC ladder v4 (`docs/experiments/nec_ladder_g25_food_relocation/REPORT.md`) fixed the reward signal, but no agent
learned to steer to food, even food 2 cells away. Its section 3.4 put forward a hypothesis:

> "Food is two cells ahead-left" is a different input pattern at each of the 625 head positions and 4 headings.
> The MLP has no built-in notion that these are the same situation, and must learn each one separately.

The experimenter asked for a clean comparison of the NEC agent with a CNN encoder and DQN, on v4's protocol, to
test that hypothesis.

v4 could not answer it, for two reasons:
- it only measured *behaviour*, never the representation;
- its NEC and DQN arms used different exploration schedules (ε 0.005 fixed vs 1 → 0.1 over 250k).

## 1. Hypothesis

**H (encoder bottleneck):** agents fail to steer because their encoder does not represent where the food is
*relative to the snake*. States that differ only in the food's cell map to nearly the same embedding, and states
with the same egocentric food offset at different board positions map far apart.

## 2. Design

*Decision:* a 2 × 2 factorial, {NEC, DQN} × {MLP, CNN} encoder, plus a random reference.
- The hypothesis is about the encoder, so the encoder is the manipulated factor.
- The MLP arms are the encoders that v4 used.

| Name in code | Learner | Encoder |
|---|---|---|
| `dqn` | DQN (N = 1) | MLP 1875 → 64 → 64 → 3 |
| `dqn_cnn` | DQN (N = 1) | 3×3 conv 16, 3×3 conv 32 (stride 1, same padding), dense 64 → 3 |
| `nec` | NEC (N = 50) | MLP 1875 → 64 → 32 (key) |
| `nec_cnn` | NEC (N = 50) | 3×3 conv 16, 3×3 conv 32, dense 64 → 32 (key) |
| `random` | uniform random | — |

All agents use repository defaults, with no tuning.

**Environment:** v4's, unchanged:
- 25×25 board, map `rooms`, bonus food worth 5, no distractors;
- `--food-curriculum 2:250000:600000 --food-relocate`.

**Matched exploration** (*decision*, the "clean" part): every agent uses the same ε schedule, falling
linearly from 1 to 0.05 over 50,000 steps, then fixed at 0.05 (`--eps-floor 0.05 --eps-decay 50000`).
- This removes v4's confound.
- 0.05 is the DQN paper's evaluation ε, which lies between the two training schedules used before.

**Budget** (*decision*): 400,000 steps.
- The radius is held at 2 for the first 250k steps. Steering is therefore tested at its easiest: food is at
  most 2 steps away, and relocation keeps it there.
- The steps from 250k to 400k cover the radius growing to 22, which shows whether any skill extends further.
- v4 showed that nothing happens on the real game after that point, so a full 1M-step run would mostly cost
  CNN compute.

**Seeds:** 501–505 (never run), 5 per agent, 25 runs in total.
- Agent stream `seed`; training env `seed + 1000`; evaluation env `seed + 2000`; evaluation exploration
  `seed + 3000`.
- One BLAS thread per run.

**Evaluation runs** (as in v4, but sparser): every 25,000 steps, 5 episodes at ε 0.05, with food placed
anywhere.

## 3. Measurements: the representation probe (`src/snake_rl/probe.py`, new)

The probe set is fixed and hand-built:
- **Snake configurations:** 150 (head cell, heading) pairs, each a straight snake of length 3, drawn once
  with seed 0.
- **Food placements:** for each configuration, one state per free cell within 2 walkable steps of the head.
- **Size:** 1,293 states.
- **Offsets:** the food's offset in the snake's own frame (ahead, right) takes 10 values.

Every 25,000 steps, and at step 0, the run records the metrics below. The probe is read-only: a unit test
checks that training episodes are bit-for-bit unchanged with it on. The embedding Z is the NEC key, or DQN's
last hidden layer.

| Metric | Meaning | Raw observation | Under H |
|---|---|---|---|
| `steer` | P(greedy action lies on a shortest path to the food), ties split | — | ≈ `steer_chance` (0.40) |
| `knn_offset` | share of a state's 10 nearest probe states (Euclidean in Z) with the same egocentric food offset | 0.017 | low |
| `knn_config` | share of those neighbours with the same snake (only the food differs) | 0.80 | high |
| `decode` | grouped 5-fold ridge read-out of the egocentric offset from Z (chance 0.10) | 0.06 | ≈ chance |
| `food_ratio` | mean ‖ΔZ‖ when only the food moves ÷ mean ‖ΔZ‖ for the same offset at another board position | 0.50 | ≪ 1 |

The raw-observation column was computed before any run. It confirms the premise: in pixel space, a state's
neighbours are "the same snake with the food elsewhere", and the egocentric offset is not linearly readable.

`knn_offset` matters most for NEC: its Q estimate *is* an average over nearest neighbours in Z. If those
neighbours have the food elsewhere, Q cannot prefer the action that leads to the food.

## 4. Outcomes and analysis (exploratory; no confirmatory verdicts)

All outcomes are read at the **end of the held phase (probe at step 250,000)**.
- **Behaviour:**
  - `steer`;
  - training regular fruit per 100 steps over steps 100k–250k (as in v4 section 3.2).
- **Representation:** `knn_offset`, `decode`, `food_ratio`.
- **Descriptive:** the trajectories of all probe metrics over 0–400k, evaluation score over training,
  training fruit per 20k steps.

**Comparisons:** for each learner, CNN vs MLP, on each outcome above.
- **Effect size:** difference in means with a percentile bootstrap CI (10,000 resamples, seed 0).
- **Test:** an exact two-sided permutation test, with all C(10, 5) = 252 relabelings.
- **No multiplicity correction:** this is exploratory, and p-values are descriptive.
- **Association:** the correlation, across all 20 learning runs, between `steer` and `knn_offset` / `decode`
  at 250k.

**Reading the results** (fixed in advance):

| Pattern at 250k | Reading |
|---|---|
| MLP: representation ≈ raw and steer ≈ chance; CNN: representation clearly better and steer clearly above chance | **Supports H.** The encoder is the bottleneck, and convolutions relieve it. |
| CNN representation clearly better, but steer still ≈ chance | **Against H as the whole story.** The food is represented, but value learning or action selection does not use it. Look at the learner, not the encoder. |
| Neither encoder represents the offset (both ≈ raw) | **Consistent with H, but not a test of encoders in general.** The CNN's position-specific dense layer re-introduces the problem. Points to an egocentric observation (v4 next step 1). |
| MLP already represents the offset (well above raw), with steer ≈ chance | **Against H.** |

"Clearly" means the bootstrap CI of the CNN − MLP difference excludes 0, *and* the steer mean is at least
0.10 above `steer_chance`.

## 5. Assumptions and checks

| # | Assumption | Check |
|---|---|---|
| B1 | The probe does not alter training. | Unit test `test_probe_does_not_change_training` (all agents). |
| B2 | The probe states resemble training states during the held phase. | Food within 2 steps and a short snake, matching the held phase. Longer snakes are not covered. |
| B3 | DQN's last hidden layer is the right "embedding". | It is the only input to the Q head; also report `steer`, which is encoder-agnostic. |
| B4 | The agents see enough reward. | Training fruit per 20k steps, against v4's same window. |
| B5 | All runs are recorded. | 25 files, with probes at 0, 25k, …, 400k (17 points). |

## 6. Procedure

1. `bash docs/experiments/encoder_food_probe_g25/run.sh`: 25 runs, 8 in parallel, into
   `results/exp_encoder_probe/`, with the log in `logs/exp_encoder_probe.log`.
2. Check B5, and rerun failures once with the same seed.
3. `.venv/bin/python docs/experiments/encoder_food_probe_g25/analyze.py`. This writes `results.md`,
   `per_run.csv` and `probe_curves.png` next to this file.
4. `REPORT.md`: hash check, the checks, results, reading per section 4, deviations, limitations.
