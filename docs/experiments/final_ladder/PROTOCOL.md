# Experiment protocol: the final ladder, 1M steps on Snake 10x10 and Ms. Pac-Man

**Status: DRAFT, not frozen yet.** Written by Claude on the experimenter's direction (2026-10-13), to be reviewed by them.

- **Frozen files:** `PROTOCOL.md`, `analyze.py`, `make_specs.py` and the four specs `ansible/experiments/final_ladder_{snake,pacman}{,_memory}.yml`.
  At freezing they are hashed into `FROZEN.sha256` together with the commit that contains them, and that commit is made before the first run is
  submitted.
- **Departures:** any departure is listed under "Deviations" in the report.

This is the study's last experiment (the study is closed: `docs/STUDY_SUMMARY.md`). It reruns the whole ladder once, with the best settings the
exploratory work found, at a budget well past the paper's first reported point, so that the closing write-up rests on one clean, pre-registered
comparison rather than on a chain of exploratory runs.

## 0. Where this comes from

**Snake 10x10, pixels (`nec_budget_eval_g10`, pre-registered, 250k steps, DQN with Adam).**
- DQN: 8.82 fruit per episode, and it steers.
- NEC: 0.78-0.88, below the frozen-embedding EC (1.09).

**Ms. Pac-Man (`pacman_ladder`, `pacman_calibration`, `pacman_c2c3`, all exploratory, 100k steps, 5 seeds).**
- Reward units changed NEC most: 357 → 519 in Atari units.
- DQN's target sync changed DQN most: 478 → 668 with a sync of 4000 steps.
- The best arms were DQN B4 (668) and NEC D1 (553, no memory eviction), with overlapping ranges.
- Both were chosen on the same seeds they were scored on. A fair test needs fresh seeds and a longer budget, which is what the calibration
  report and `docs/CALIBRATION_AUDIT.md` (C5) asked for.

**What this experiment adds:**
- the full ladder on both games with one set of agent settings;
- a 1M-step budget;
- fresh seeds;
- a pre-registered analysis.

## 1. Hypotheses (confirmatory)

**Outcomes** (section 3), per run:
- **O2 (early):** the evaluation score at 200k–250k steps. 250k agent steps is the NEC paper's first reported point (1M frames with a frame skip
  of 4).
- **O1 (final):** the evaluation score at 950k–1M steps.

**The hypotheses**, for each game G ∈ {Snake, Ms. Pac-Man}:

| | Claim | Outcome | Origin | Prediction, Snake | Prediction, Ms. Pac-Man |
|---|---|---|---|---|---|
| **H1** | NEC > DQN | O2 | the NEC paper's central claim: data efficiency early in training | not supported | not supported |
| **H2** | NEC > DQN | O1 | whether any NEC lead survives a longer budget | not supported | not supported |
| **H3** | NEC > frozen-embedding EC | O1 | whether learning the embedding helps (rung 3 against rung 2) | not supported | **supported** |
| **H4** | NEC > MFEC | O1 | the NEC paper reports NEC above MFEC | not supported | **supported** |

**Where the predictions come from:**
- **Snake:** at 250k steps, NEC was far below DQN and below frozen EC (`nec_budget_eval_g10`).
- **Pac-Man, H1:** the best DQN led the best NEC at 100k.
- **Pac-Man, H3 and H4:** in Atari units NEC was 120–155 points above frozen EC and about 130 above MFEC at 100k (`pacman_calibration`).

**The test:**
- **Comparisons:** 8 directional comparisons (4 hypotheses × 2 games), each a one-sided exact permutation test on the difference of mean outcomes
  (10 runs against 10, all 184,756 splits).
- **Correction:** Holm-adjusted over the 8, family-wise α = 0.05.
- **Effect sizes:** the difference of means with a 95% percentile bootstrap interval (10,000 resamples, fixed seed), and Hedges' g.
- **Verdicts:** a hypothesis is **supported** if its Holm-adjusted p < 0.05, and **not supported** otherwise. Not supported is not softened.
  Where the interval shows the opposite direction, the report says so.

**Exploratory** (reported, not tested):
- the ladder, each rung against the one below (DQN → N-step DQN → frozen EC → NEC), and every arm against random;
- learning curves;
- Snake's representation probe;
- NEC's key norm and memory counters;
- episode lengths.

Anything computed later from saved checkpoints (value calibration, embedding geometry) is also exploratory and labelled so.

## 2. Design

**Common to every run:**
- **Budget:** 1,000,000 agent steps.
- **Seeds:** 21–30 (fresh: the settings below were chosen on seeds 1–5 and 11–20).
- **Exploration:** the default ε schedule (1 → 0.02 over 5k steps).
- **Evaluation:** `--eval-every 25000 --eval-episodes 20`, at ε 0.05 on a separate env seeded `seed + 2000`.
- **Telemetry:** `--train-log-every 1000`.
- **Checkpoints:** `--save-every 500000`, so agents are saved at 500k and 1M.

**Games:**
- **Snake:** 10x10 `open` board, `--render pixels` (84x84 gray, 4-frame stack), absolute actions. Rewards are +1 for food and −1 for death,
  unchanged. Also `--probe-every 50000 --probe-far` for the agents with a network.
- **Ms. Pac-Man:** the fixed 14x14 maze, frame skip 4, `--reward-units atari` (10 / 50 / 200 points, nothing for dying), the setting where NEC did
  best.

**Arms** (the same settings on both games; `snake-run` / `pacman-run` agent names):

| Arm | Agent | Settings | From |
|---|---|---|---|
| Random | `random` | none | floor |
| DQN | `dqn` | RMSProp 2.5e-4 (the DQN paper's), target sync every 4000 steps, rewards clipped to [−1, 1] (the paper's DQN) | `pacman_c2c3` B4, the best arm of the study |
| N-step DQN | `dqn_nstep` | as DQN, with N = 50 | rung 1 of the ladder |
| MFEC | `mfec` | defaults: 64-d random projection, k = 11, 2e4 rows per action, γ 0.99 | as in every earlier run |
| Frozen-embedding EC | `ec_frozen` | DND capacity 5e5 per action | rung 2 |
| NEC | `nec` | RMSProp 1e-4, DND capacity 5e5 per action | `pacman_c2c3` D1 |

That is 6 arms × 10 seeds × 2 games = 120 runs (the 20 random runs cost minutes). The networks follow the observation: DQN's Nature CNN on 84x84
frames for every arm with a network, on both games.

### Decisions (taken by the experimenter on 2026-10-13)

- **Budget: 1M agent steps.** This is 4M frames on Ms. Pac-Man; Snake has no frame skip. The paper's first reported point (250k agent steps) is
  inside the runs as outcome O2. The NEC paper compares at up to 10M frames.
- **Snake uses the Pac-Man settings.** One set of agent settings for both games.
  - On Snake, clipping is a no-op (rewards are already ±1).
  - On Snake, DQN with RMSProp 2.5e-4 was weaker than with Adam in short runs (`pixels_rmsprop_probe`; 1.03 against 2.08 at 7x7, 60k steps).
    That cost is accepted for the sake of one setting across games, and it is a stated limitation.
- **The full ladder,** N-step DQN included.
- **DND capacity 5e5 per action (the NEC paper's) for NEC and frozen EC.**
  - NEC writes about one row per step, about a quarter to each action. With the earlier 1e5 per action, eviction would return after ~400k
    steps; with 5e5, nothing is evicted in 1M steps.
  - C3 found eviction made the memory match visited states less well (probe-vs-memory 1.09 → 0.92 without it).
  - MFEC keeps its 2e4 per action, as in every earlier run, as a cheap reference (the MFEC paper uses 1e6). This is a known asymmetry, reported
    as such.
- **Kept from the earlier runs** (known departures from the papers, unchanged here so the result connects to the exploratory work):
  - key size 32;
  - N = 50 for NEC (N = 100 changed nothing in C3);
  - replay 2e4 (NEC) and 5e4 (DQN);
  - training from the start, and the global-norm gradient clip of 10;
  - exact kNN instead of kd-trees;
  - no action repeat on Snake.

## 3. Measurements

- **Evaluation episodes:** 20 at ε 0.05 every 25k steps on a separate env, with exploration seeded `seed + 3000`. Training is unchanged by them
  (tested: `test_evaluation_does_not_change_training`).
  - The score is points: fruit on Snake, game points on Ms. Pac-Man.
  - **O1:** per run, the mean of the evaluation means at 950k, 975k and 1M.
  - **O2:** the same at 200k, 225k and 250k.
  - The run is the unit of analysis.
- **Snake probe** (`probe.py`, unchanged since `nec_budget_eval_g10`): near and far steering, decode, ties, every 50k steps. Exploratory here.
- **Already logged:** training episodes, telemetry every 1,000 steps, and per-action memory counters every 6,000 steps.
- **Checkpoints at 500k and 1M:**
  - They hold the agent without its replay buffer, so they can act and be probed but not resume training.
  - NEC and frozen-EC checkpoints are ~0.55 GB each (the preallocated DND), ~50 GB in total. They go to the Hugging Face dataset, not git.
  - Analyses run on them later are exploratory.

## 4. Analysis and power (written before the runs)

- **Script.** `analyze.py` produces `results.md`, `per_run.csv` and `eval_curves.png`. It was tested on synthetic results of the right shape and on
  the check runs (section 8).
- **Missing runs.** If a run fails, it is rerun once with a longer time limit. If it still fails, its arm has n < 10, the report says so, and the
  tests use the runs there are.
- **Power.** It comes from the earlier runs' spreads (5 seeds, 100k steps; an informed guess, not a new analysis).
  - **Pac-Man:**
    - Seed s.d. was 35–90 points. H3 and H4's expected effects (≥ 120 points) are g ≳ 1.5.
    - Simulated power (one-sided t, normal data, 10 against 10, 40,000 draws): at the strictest Holm level (α/8 = 0.006) it is 0.64 at g 1.4,
      0.77 at 1.6, 0.88 at 1.8 and 0.94 at 2.0. At α = 0.05 it is 0.91 at g 1.4.
    - So H3 and H4 are adequately powered if their effects stay near the 100k-step sizes, and moderately powered if they shrink.
    - H1/H2 effects near zero, or effects of 50–100 points, will not be detected reliably. "Not supported" is then weak evidence against a small
      NEC lead and good evidence only against a large one.
  - **Snake:** s.d. was 0.1–0.2 fruit per episode at 250k. Differences of ≥ 0.25 between episodic arms are detectable; DQN differences are an order
    of magnitude larger.

## 5. Assumptions and risks

- **Time.** On the cluster, DQN runs at about 45 ms per step at the start (check runs), so about 13–15 h per 1M steps.
  - NEC's brute-force kNN grows with its memory: 10 → 58 ms per step locally as the DND goes from empty to 250k rows per action, about 5× that
    on the cluster. So about 30–50 h per run.
  - Limits: 3 days for the fast arms (`normal` partition) and 7 days for NEC and frozen EC (`extended` partition, limit 21 days).
- **Concurrency.** Per-user limits are 96 CPUs on `normal` and 48 on `extended`. The specs ask for 80 and 40, so everything runs at once, and the
  wall clock is about 2–3 days.
- **Reproducibility.** Runs are bit-for-bit reproducible on the same machine and numeric library build.
  - Across machines they are not always (section 8: Pac-Man NEC diverged within ~500 steps between this Mac and the cluster, from float noise in
    the gradients).
  - Every confirmatory run executes on the cluster, and nothing is compared across machines. The node of each run is in `sacct` and is recorded
    in the report.
- **Selection.** The arms' settings were chosen on seeds 1–5. Seeds 21–30 are fresh, so the 1M results do not reuse those runs. The settings
  remain the best of a small search for both agents (NEC: learning rate and capacity; DQN: optimizer and sync), not a full sweep for either.

## 6. Procedure

1. Write the protocol, `analyze.py`, `make_specs.py` and the specs.
2. Run the checks of section 8.
3. Have the experimenter review the protocol.
4. Hash the frozen files and commit.
5. Submit the four specs: `final_ladder_snake`, `final_ladder_snake_memory`, `final_ladder_pacman`, `final_ladder_pacman_memory`.
6. Collect (repeatable; the VPN drops when idle).
7. Run `analyze.py` and write `REPORT.md`, with a Deviations section.
8. Upload the results and checkpoints to Hugging Face.
9. Write the blog post on the whole study (`docs/blog/`).

## 7. Spec files

- **Specs:** `ansible/experiments/final_ladder_{snake,pacman}.yml` (the fast arms, partition `normal`) and
  `ansible/experiments/final_ladder_{snake,pacman}_memory.yml` (NEC and frozen EC, partition `extended`).
- **Job lines:** generated by `make_specs.py`.
- **Results:** everything lands in `results/exp_final_ladder/`, in `g10_d0_px_abs/` (Snake), `pac_px_atari/` and `pac_px_atari_clip/` (Ms.
  Pac-Man; the DQN arms learn from clipped rewards).

## 8. Checks run before freezing (2026-10-13)

| Check | Result |
|---|---|
| C1. `pytest` on this code | 129 passed, including the 15 golden checks against `pre-cleanup` |
| C2. Every arm, both games, 3,000 steps, every measurement on, on the cluster (specs `final_check_snake` on `normal`, job 1693; `final_check_pacman` on `extended`, job 1699) | 12 of 12 finished; `pacman-run` and both partitions work |
| C3. The same 12 runs locally against the cluster | 11 identical (episodes, evaluations, diagnostics; probes and telemetry within 6e-15). Ms. Pac-Man NEC diverged from episode 67 on: numpy 2.5.4 here against 2.4.6 with OpenBLAS on the cluster, and in Atari units NEC's gradients are ~10⁴ and clipped, so float noise flips an action early. Not a logic difference; see section 5. |
| C4. `analyze.py` on synthetic results (both games, 6 arms, 10 seeds, evaluations every 25k) and on the check runs | tables, the 8 tests and the figure come out as intended. The exact permutation p reaches its floor of 1/184,756 for a complete separation. Holm checked by hand. |
| C5. Timing on the cluster, 3,000-step check runs | DQN 84–140 s, NEC 119–212 s (with probes every 1,500 steps and evaluation every 1,000) |
