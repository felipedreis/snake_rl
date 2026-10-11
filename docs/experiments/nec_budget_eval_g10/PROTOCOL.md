# Experiment protocol: does a paper-sized budget let NEC steer? Evaluation runs, a random baseline, tie and distance probes

Status: **pre-registered, frozen on 2026-10-10 (revision 5).** Written by Claude on the experimenter's direction and reviewed by them. Design choices marked
*decision* were confirmed or changed by the experimenter (section 7). `PROTOCOL.md`, the spec `ansible/experiments/nec_budget_eval_g10.yml`, `analyze.py` and
`src/snake_rl/probe.py` are hashed into `FROZEN.sha256`, and the commit containing them was made before the first run was submitted. Any departure is listed under
"Deviations" in the report.

Revision 2 (same day): learning rates changed after the experimenter questioned them (section 2); seeds raised to 10 after a power guess (section 4);
DND capacity and the other defaults kept, as the experimenter decided.
Revision 3 (same day): agents are saved (section 3.5).
Revision 4 (same day): H2 and H3 use a pre-declared arm (NEC 1e-4); fresh seeds 11-20 (section 2); the learning rates are kept.
Revision 5 (same day): pieces built and checked (section 8); 250k steps and 10 seeds confirmed.

## 0. Where this comes from

The NEC learning-rate sweep (`docs/experiments/nec_lr_sweep_absolute/REPORT.md`, 60k steps, 10x10 open, pixels, absolute actions, 3 seeds) found:
- NEC with a learned embedding (0.51-0.83 fruit per episode) is not distinguishable from `ec_frozen` (0.78, seeds 0.60-0.93);
- NEC's probe steering is at chance (0.40-0.43, chance 0.39), while DQN with the same actions steers at 0.60 and scores 1.83;
- the probe's `decode` is at chance (0.09, chance 0.10) and neighbours in embedding space share the snake configuration (`knn_config` 0.62), not the food offset.

Open questions the sweep could not answer, which this protocol closes:
1. **Training-episode scores are the only performance measure.** No evaluation episodes were run.
2. **No random baseline under absolute actions.** The only one (0.17, `mfec_dqn_nec`) used relative actions on the grid.
3. **Chance-level steering is ambiguous.** It can mean wrong Q-values or tied Q-values (no information), and the probe only tests food within 2 steps.
4. **The budget is below the paper's.** The paper's first reported point is 1M frames. With its action repeat of 4 that is 250k agent steps and about 62.5k updates
   (one per 16 frames). Our 60k steps are about 15k updates (one per 4 steps).

## 1. Hypotheses

Let "NEC arm" mean one of the three NEC learning rates of section 2.

- **H1 (learning the embedding helps at the paper's first budget).** At 250k steps a NEC arm's evaluation score exceeds `ec_frozen`'s.
  Tested for each of the three arms, Bonferroni-corrected (98.3% bootstrap intervals). Supported for an arm if the difference of means is >= +0.25 fruit per episode
  and its interval excludes 0. 0.25 is about half the `ec_frozen` seed range at 60k (0.33) *(decision)*. *Prediction: not supported for any arm.*
- **H2 (the paper's claim against DQN).** The **pre-declared NEC arm, RMSProp 1e-4,** exceeds DQN. Supported if the 95% interval of NEC - DQN lies above 0.
  *Prediction: not supported.* The arm is fixed now because it was nominally best at 60k in the sweep (0.83 against 0.77 and 0.69; not a significant difference, but a
  pre-declaration needs a reason, not significance). Seeds 11-20 are fresh, so that choice does not depend on the runs it is tested on.
- **H3 (NEC steers near the food).** The same arm's near-probe steering (food within 2 steps, the existing probe) at 250k is >= 0.50, more than halfway from chance
  (0.39) to DQN's 0.60 at 60k *(decision)*. *Prediction: not supported.*

If another arm turns out better at 250k, the confirmatory verdicts on H2 and H3 still refer to 1e-4. The best-of-three comparison against DQN and its steering are then reported
as **exploratory, selected after the fact** (this favours NEC: "supported" there would be optimistic).

Exploratory (reported, not tested): each learned agent against the random baseline; tie fractions; steering by distance; learning curves; key norm, kernel peak and
memory counters; episode length.

## 2. Design

Common to every run: 10x10 `open` board, `--render pixels` (wall gray 60, 4-frame stack), `--actions absolute`, **250k steps**, seeds 11-20, default epsilon
schedule (1 to 0.02 over 5k steps), `--eval-every 10000 --eval-episodes 20` (epsilon 0.05, a separate env seeded `seed + 2000`),
`--probe-every 10000`, `--train-log-every 1000`, **`--save-agent --save-every 60000`** (see 3.5).

| Condition | Agent | Optimizer |
|---|---|---|
| random | `random` | none |
| DQN | `dqn_naturecnn` | Adam 5e-4 (the agent's default) |
| frozen EC | `ec_frozen_naturecnn` | none (nothing is trained) |
| NEC 7.92e-6 | `nec_naturecnn` | RMSProp 7.92e-6 |
| NEC 3e-5 | `nec_naturecnn` | RMSProp 3e-5 |
| NEC 1e-4 | `nec_naturecnn` | RMSProp 1e-4 |

6 conditions x 10 seeds = 60 runs (the 10 random runs cost seconds).

**Learning rates (*decision*, revised).** Where they come from:
- **1e-4:** the nominally best at 60k, though not distinguishable from its neighbours (0.83 against 0.77 and 0.69).
- **7.92e-6:** from Agostinelli et al. 2019, Table 5 ("as in the original papers"). **The NEC paper itself does not report its learning rate**; it says only that it swept
  the SGD learning rate on six games. At 60k this value behaved like a frozen encoder (`docs/DIAGNOSIS_AND_NEXT_STEPS.md` section 3).
- **3e-5:** between the two, to span the range at equal log spacing.
- **2.5e-4 dropped.** It is Nature DQN's value, used earlier as a "paper-style" setting; at 60k it did no better than the lower rates (0.64 against 0.69-0.83).

Why smaller rates for a longer run is a hypothesis, not a paper result: the same rate moves the encoder further the more updates it gets (250k steps give 62.5k updates,
4x the 60k runs), so a rate that barely moved the encoder at 60k may do real learning at 250k, and a larger one may drift more. The NEC paper's only statement about learning
rates is general (that SGD needs small rates, which makes gradient-trained agents slow); it does not say that more data needs smaller rates.

- **Budget (*decision*).** 250k steps equals the paper's first reported point in agent steps, and one update per 4 steps gives the same 62.5k updates.
  The 60k point is the `t = 60000` checkpoint inside these runs, not a separate arm (see check C2).
- **Seeds 11-20 (*decision*).** Fresh seeds, not the sweep's 1-3. A seed fixes the first 60k steps of a run, so reusing seeds 1-3 would let the choice of NEC 1e-4
  (made from the sweep's 60k results) leak into the runs that test it. Fresh seeds remove that at no cost.
- **Limitation of the learning-rate set.** Nothing above 1e-4 is run, so if more budget favours a larger rate, this experiment cannot see it. 2.5e-4 did no better at 60k
  (0.64 against 0.69-0.83), so this is not expected, but it is not tested.
- **Kept at the defaults, as the experimenter decided,** to isolate the budget: DND capacity 2e4 per action (paper: 5e5), N = 50 (paper: 100), replay 2e4 (paper: 1e5),
  no action repeat. These are known departures and the subject of a later experiment. Two more found while writing this, from Agostinelli et al. 2019 Table 5 and also unchanged:
  key size 128 (ours: 32), and exploration annealed from 1 to 0.001 between steps 5k and 25k with training starting at 50k steps (ours: 1 to 0.02 over 5k, training from the start).

## 3. Measurements

### 3.1 Evaluation episodes (new use of an existing option)
20 episodes at epsilon 0.05 every 10k steps, on an env seeded `seed + 2000` with exploration seeded `seed + 3000`; training is unchanged by it (tested).
Each episode returns score, regular foods, truncation and length.
**Primary outcome O1:** the mean evaluation score over the evaluations at 230k, 240k and 250k, per run. The unit of analysis is the run (seed).

### 3.2 Random baseline
The `random` agent under the same env, evaluation and seeds. "Above random" means the agent's O1 minus the random O1.

### 3.3 Probe extensions (new code, to be written and tested before freezing)
The existing probe (150 snake configurations, food within 2 walkable steps, 10 offset classes) is **kept as is**, so the new runs stay comparable with all earlier reports.
A second, larger probe set is added, used only for:
- **steering by distance:** `steer` and its chance rate for food at walkable distance 1, 2, ..., 6, with the number of states at each distance;
- **tie fractions:** the share of probe states where all four Q-values are equal (`np.isclose`), and where two or more actions share the maximum.
  Rule fixed here: if the all-tied share exceeds 0.5 at a probe, chance-level steering at that probe is read as *missing information*, not *misinformation*.

The extension must not change training. It uses no agent rng, calls neither `act` nor `observe`, and gets the same RNG-neutrality test as the existing probe.

### 3.4 Already collected and unchanged
Training-episode log, training telemetry (`key_norm`, `kernel_peak`, `nb_dist`, `dead_relu`, ...) every 1,000 steps for NEC and DQN, per-action memory counters every 6,000 steps,
and the existing probe metrics every 10k steps.

### 3.5 Saved agents (decided by the experimenter, added in revision 3)
Every run saves its agent at steps 60k, 120k, 180k and 240k (`--save-every 60000`) and at the end, 250k (`--save-agent`): 5 files per run, about 50 MB each
(measured on earlier checkpoints: 49-54 MB), so 250 MB per run and **about 12.5 GB for the 50 training runs** (random saves nothing useful and is run without them).
The 60k checkpoint lines up with the learning-rate sweep, which saved none.
- A checkpoint holds the agent without its replay buffer (the DND, the encoder, the optimizer state, the run settings). It can act and be probed, but not resume training.
- Saving does not touch any random number generator (tested: `test_saved_agent_can_be_watched_without_changing_training`).
- **What this allows, and the rule for using it.** Any metric can be recomputed later on the saved agents, including probe variants nobody thought of today. To keep that honest:
  the analyses in sections 3.1-3.3 and 4 are the pre-registered ones, computed during the run. **Anything computed afterwards from checkpoints is exploratory and labelled so.**
  One use is pre-specified as a fallback: if the extended probe of 3.3 has a bug, it is recomputed from the checkpoints at 60k/120k/180k/240k/250k, with the corrected code,
  and the report says so in Deviations.
- The checkpoints are Python pickles, so they carry the usual warning (loading runs code). They are gitignored and go to the Hugging Face dataset, not to git.

## 4. Analysis and power (written before the runs)

- Per condition: the mean of O1 over seeds and the per-seed values. Differences use the difference of means with a percentile bootstrap interval over seeds (10,000 resamples,
  fixed seed): 95% for H2 and H3, 98.3% for each H1 comparison. No hypothesis test beyond H1-H3.
- Confirmatory verdicts are exactly as defined in section 1. Results that miss the thresholds are reported as "not supported", not softened.
- Learning curves (evaluation score, probe steering, key norm against steps) are descriptive, for all conditions.
- Failed or killed runs are rerun with a larger time limit and reported; the analysis does not change after the runs.

**Seeds: an informed guess, not a new power analysis (*decision*).** The sweep's seed standard deviations at 60k (3 seeds each, so noisy) were 0.17 (`ec_frozen`), 0.14 (NEC 1e-4),
0.07 (7.92e-6) and 0.02 (3e-5). Assuming a standard deviation of 0.20 at 250k (conservative, since scores and spread may grow), the H1 effect of 0.25 is g = 1.25.
Simulated power (Welch t, normal data, two-sided, 40,000 draws per cell):

| Seeds per condition | alpha 0.05 | alpha 0.05/3 (H1, three arms) |
|---|---|---|
| 5 | 0.41 | 0.23 |
| 8 | 0.64 | 0.44 |
| **10** | **0.76** | **0.57** |
| 12 | 0.83 | 0.69 |
| 15 | 0.91 | 0.81 |

With a standard deviation of 0.15 the same table gives 0.94 and 0.85 at 10 seeds; with 0.25 it gives 0.56 and 0.36. So 10 seeds is adequate for H2 and H3 and **moderate for
H1**: a "not supported" verdict on H1 is weak evidence against an effect of 0.25, and good evidence only against effects well above it. Same caveat as the `mfec_dqn_nec` protocol
(which used 10 runs, pilot standard deviations of 0.05-0.1, and flagged its weakest hypotheses as underpowered).

## 5. Assumptions and checks

- **C1.** `pytest` passes after the probe extension, including the new RNG-neutrality test (a run with and without the extended probe gives identical episodes).
- **C2.** The first 60k steps of a 250k run equal a 60k run with the same seed (the epsilon schedule does not depend on the total length). Checked locally on seed 1 (not an experiment seed) for one DQN and one NEC run before submitting.
- **C3.** The cluster reproduces local results for the new settings (evaluation, far probe, saved agents). Floating-point equality over 250k steps is **not** assumed and is not needed: every confirmatory run executes on the cluster.
- **C4.** Timing: the 60k NEC runs took 1.0-1.7 h and DQN 1.7-2.0 h, so 250k is expected at about 4-8 h per run plus evaluation and probe cost.
- **C5.** Saving works at scale: one local run of 62k steps with `--save-every 20000` produces 3 intermediate files and a final one, and a reloaded checkpoint gives the same Q-values
  as the live agent on a fixed batch of observations. Disk: about 12.5 GB expected, against 740 GB free on the cluster's `$HOME` and 533 GB on the experimenter's Mac.

## 6. Compute and procedure

50 training runs (DQN, `ec_frozen`, three NEC arms, 10 seeds each) plus 10 random runs on CCAD's `short` partition (limit 24 h), one CPU each, `--time 20:00:00`,
`--mem 12G`, at most 25 at once (the earlier jobs ran 24 at once without trouble): about 2-3 waves, so roughly a day of wall clock.
Steps: (1) implement and test the probe extension (done); (2) write the spec `ansible/experiments/nec_budget_eval_g10.yml` and `analyze.py` (done); (3) run C1-C3 and C5 (done, section 8);
(4) hash the frozen files and commit; (5) submit; (6) collect; (7) write the report with a Deviations section.
`collect.yml` currently skips `*.agent.pkl`; it needs a per-spec switch to pull them for this experiment (about 12.5 GB over the VPN, which drops when idle, so rsync with `--partial`
and repeat as needed).

## 7. Decisions

The thresholds (+0.25 for H1, 0.50 for H3) were kept as written.

Decided by the experimenter: 250k steps only and 10 seeds; the NEC learning rates (7.92e-6, 3e-5, 1e-4); H2 and H3 on the pre-declared arm 1e-4, with fresh seeds 11-20; kept all intermediate checkpoints (every 60k plus the final, 5 per run, about 12.5 GB); DND capacity and the other defaults unchanged; seeds by an informed guess, not a new power analysis.

## 8. Checks run before freezing (2026-10-10)

| Check | Result |
|---|---|
| C1. `pytest` after the probe extension | 140 passed (7 new tests: far-probe distances, steering and ties; pixel far set stored as uint8; far probe leaves `dqn`/`nec` training unchanged; checkpoint reload; run prefix) |
| C2. A short run is the prefix of a longer one | passes as a unit test (`dqn`, seed 3, 600 against 1,500 steps: identical episodes up to step 600) |
| C3. Cluster against local, new settings (spec `check_c3`, job 1472) | `random` 20k steps, `nec_naturecnn` and `dqn_naturecnn` 3k steps with evaluation, far probe and agent saving. **Episodes, evaluations and memory diagnostics are identical.** Probe values differ only in `food_ratio`, by at most 3.4e-13; saved agents' Q-values differ by at most 3.9e-12 (Q scale about 0.8), with the same greedy action in every state tested. Float noise from the different numeric library builds, not a logic difference |
| C5. Saving at scale | checkpoints at 1,500 and 3,000 plus the final one are written on the cluster and pulled back with `pull_agents: true`; reload gives the same Q-values as the live agent (unit test) |
| Far-probe cost | about 1 s per call on the Nature CNN (3,005 states: 164, 351, 495, 593, 699, 703 at distances 1 to 6) |
| `analyze.py` | runs end to end on a miniature run of all six conditions (2 seeds, 1,500 steps): tables, CSV and both figures are produced. Its numbers at that size mean nothing |

Nothing was left open; the thresholds were kept as written and the protocol was frozen.
