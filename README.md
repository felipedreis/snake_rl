# Episodic control on Snake and Ms. Pac-Man

**This study is closed (2026-10-12).** What it found, and why it stopped: [docs/STUDY_SUMMARY.md](docs/STUDY_SUMMARY.md).

A pure-NumPy implementation of Neural Episodic Control (Pritzel et al., 2017), Model-Free Episodic Control
(Blundell et al., 2016) and DQN (Mnih et al., 2015). The networks, the optimizers and every gradient are hand-written and
gradient-checked, including the gradient through NEC's kernel-weighted kNN lookup. The agents are tested on two games:
Snake (grid or Atari-style pixels) and a small Ms. Pac-Man.

**Old experiments:** the code was simplified after the study closed (relative actions, distractor channels, the grid
CNNs, `nec_refresh` and `nec_bonus` were removed). Every experiment in `docs/experiments/` runs as written at the git
tag `pre-cleanup`.

## Setup

    python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
    source .venv/bin/activate

## Usage

    snake-run nec 1 40000                                  # agent, seed, steps -> results/d0_abs/nec_s1.json
    snake-run nec 1 60000 --size 10 --render pixels        # 84x84 frames, DQN's network -> results/g10_d0_px_abs/
    snake-run dqn 1 100000 --size 25 --map rooms --bonus 5 --food-curriculum 2:50000:80000 --food-relocate
    pacman-run nec 1 100000 --reward-units atari --eval-every 10000   # -> results/pac_px_atari/nec_s1.json
    snake-plot results/d0_abs                              # learning curves -> figures/, tables on stdout
    snake-train results/d0_abs/nec_s1.json                 # training telemetry (loss, TD error, grad norm, ...)
    snake-run dqn 1 40000 --save-agent && snake-watch results/d0_abs/dqn_s1.agent.pkl   # watch it play
    pytest -m "not slow"                                   # fast tests; plain `pytest` adds the golden runs

Every command prints its options with `--help`; the full list is in the [command reference](#command-reference).

## Results data

The raw results (run JSONs, training telemetry, saved agents) live on Hugging Face, not in git:
[felipedreis/snake-rl-results](https://huggingface.co/datasets/felipedreis/snake-rl-results). To restore them:

    hf download felipedreis/snake-rl-results --repo-type dataset --local-dir results

The `.agent.pkl` files are Python pickles: only load them from a source you trust. `results/` and `*.agent.pkl` are in
`.gitignore`. The readers (`snake-plot`, `snake-train`, `snake-watch`, the geometry tools) still read the old files.
Old agent names such as `nec_naturecnn` are mapped to the current ones. Relative-action checkpoints need the
`pre-cleanup` tag.

## Agents

| Name | What it is |
|---|---|
| `random` | Uniform random actions: the floor. |
| `dqn` | DQN: Huber loss, target network synced every 1,000 steps, replay 50k, Adam 5e-4. |
| `dqn_nstep` | DQN with N-step returns (N = 50). |
| `mfec` | MFEC: fixed random projection to 64-d keys, k = 11 neighbours, Monte Carlo returns with a max update. |
| `ec_frozen` | NEC's memory with a random, frozen encoder (no gradient training). |
| `nec` | NEC: per-action DNDs (exact kNN, p = 50, capacity 2e4), N = 50, encoder trained through the lookup. |

**The network follows the observation, not the name.** On the grid, `dqn`, `dqn_nstep`, `ec_frozen` and `nec` use an
MLP. On pixels (Snake with `--render pixels`, and Ms. Pac-Man) they use DQN's exact network: valid convolutions
32 8x8/4, 64 4x4/2 and 64 3x3/1 (84 → 20 → 9 → 7), then FC 512 and the Q head or NEC's 32-d key. On pixels, replay
stores the frames as uint8, which is exact.

`--agent-kw KEY=VALUE` overrides any constructor setting, for example `target_every=4000`, `N=100` or
`dnd_cap=100000`. `--opt rmsprop --lr F` swaps the optimizer.

**Known differences from the NEC paper** (deliberate): exact kNN instead of kd-trees, N = 50 instead of 100, DND
capacity 2e4 instead of 5e5, Adam by default, a global-norm gradient clip of 10, and key size 32.

## Games

**Snake** (`src/snake_rl/env.py`, run with `snake-run`):

- **Actions:** absolute (up, right, down, left), as on an Atari joystick. A reversal keeps going straight.
- **Rewards:** +1 for food, +R for bonus food, −1 for death.
- **Episodes** are truncated after 2n² steps without eating, and truncation bootstraps.
- **Observation:** with `--render grid` (the default), one n×n channel each for:
  - the head;
  - the body;
  - the food;
  - the walls (when there are obstacles);
  - the bonus food's remaining lifetime (when bonus food is on).

  With `--render pixels`, the observation is the last 4 frames of an 84x84 grayscale drawing of the board (each cell
  is `84 // n` pixels), as DQN and NEC see Atari.

The board options:

- **`--map {open,pillars,walls,rooms}`:** a lethal obstacle layout. The free area is always connected. On 7×7
  (`#` wall, `@` head, `o` body):

      pillars    walls      rooms
      .......    .......    ...#...
      .#...#.    #####..    ...#...
      .......    .......    .......
      ..o@...    ..o@...    ##o@.##
      .......    .......    .......
      .#...#.    ..#####    ...#...
      .......    .......    ...#...

- **`--bonus R`:** Nokia-style bonus food worth R points. It appears after every 4 regular foods and lasts 2n steps.
  `score` is then counted in points.
- **`--food-curriculum R[:H]:N` and `--food-relocate`:** training aids for large boards.
  - The curriculum places food within R walkable steps of the head. R is held until step H, then grows to 2n by step N.
    After that, food goes anywhere.
  - Relocation places food again within r of the head when it has not been eaten for 2r + 5 steps.
  - Evaluation always uses the real game.

**Ms. Pac-Man** (`src/snake_rl/pacman.py`, run with `pacman-run`):

- **Maze and observation:** a fixed 14×14 maze with 103 pellets, 4 power pellets and 2 ghosts. The observation is an
  84x84 grayscale 4-frame stack.
- **Actions and timing:** 4 absolute actions, frame skip 4 (`--frame-skip`), one life.
- **`--reward-units`:**
  - `small` (the default): points / 100, and −1 for dying;
  - `atari`: 10, 50 and 200 points, and nothing for dying.
- **`--clip-reward`:** the agent learns from rewards clipped to [−1, 1], as the DQN paper's agent does. The logged
  score stays raw.

## Measuring a run

These are bookkeeping only: a run is bit-for-bit the same with them on or off.

- **Evaluation** (`--eval-every K`): plays `--eval-episodes` episodes at ε `--eval-eps` (0.05) on a separate env.
  Stored under `evaluations`.
- **Training telemetry** (`--train-log-every K`, default 1000): window means of the gradient updates. These are loss,
  |TD error|, Q and its target, gradient norm, clipped share, update/weight ratio and dead ReLUs; NEC adds neighbour
  distance and kernel peak.
  - Stored under `train`, and streamed to `<stem>.train.jsonl` while the run goes.
  - `snake-train` plots them, and the docstring of `trainplot.py` lists healthy values.
- **Representation probe** (`--probe-every K [--probe-far]`, Snake only): runs on hand-built states with food 1–2 (or
  up to 6) steps away. It reports greedy steering, and how well the embedding separates where the food is relative to
  the head (`probe.py`).
- **Embedding geometry of a checkpoint** (`snake-geometry`, `pacman-geometry`): neighbour purity on the agent's own
  states, and NEC's memory coherence.
- **Checkpoints** (`--save-agent`, `--save-every K`): pickle the agent without its replay buffer, plus the run's
  settings. A checkpoint can act and be watched, but it cannot resume training.

## Layout

    src/snake_rl/
      env.py pacman.py   the two games; frames.py the shared 84x84 frame stack
      boards.py          the game side of a run: env, curriculum, results directory, recorded settings
      run.py             the training loop, RunConfig, and the snake-run / pacman-run commands
      nn.py              MLP, ConvNet (im2col, manual backprop), Adam and RMSProp, make_net
      returns.py         N-step return accumulator
      agents/            base.py (the Agent interface), dqn.py, nec.py, mfec.py, memory.py (kNN key/value store),
                         replay.py (shared replay buffer), random_policy.py; __init__.py holds the registry
      probe.py geometry.py geometry_pacman.py telemetry.py   measurements
      plot.py peek.py trainplot.py watch.py                  readers and viewers
    tests/               gradient checks, N-step targets, every agent on both renders, and golden runs (below)
    docs/                the study summary, reports and pre-registered experiments
    figures/ logs/       figures and stdout of past sweeps

**Adding an agent:**

1. Implement `act`/`observe` (see `agents/base.py`).
2. Add a factory to `AGENTS` in `agents/__init__.py`.
3. Optionally, add a legend name in `plot.LABELS`.

**Golden runs:** `tests/test_golden.py` re-runs 13 short runs and checks they are identical to the ones recorded at
`pre-cleanup`. The runs cover every agent, both renders, a board with a curriculum, RMSProp and Ms. Pac-Man. Identical
means the same episodes, evaluations, probes, diagnostics and telemetry. The test also loads two old checkpoints.

- A refactor must keep these runs green.
- When a flag or name changes, update the command lines in `tests/golden_cases.py`, never the expected files.
- Bit-for-bit reproducibility needs one BLAS thread. The tests set it themselves (`tests/conftest.py`). For runs,
  set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`.

## Command reference

### `snake-run AGENT SEED STEPS [options]` and `pacman-run AGENT SEED STEPS [options]`

Each command trains one agent for one seed and writes `<results>/<dir>/<stem>_s<SEED>.json`. The agent's RNG uses
`SEED`, the env `SEED + 1000`, the evaluation env `SEED + 2000` and evaluation exploration `SEED + 3000`.

**Snake board** (`snake-run` only; each combination gets its own directory):

| Option | Default | Meaning |
|---|---|---|
| `--size n` | 7 | Grid size. Dir `d0` for 7, `g{n}_d0` otherwise. |
| `--map M` | `open` | `open`, `pillars`, `walls`, `rooms`. Adds `_{M}`. |
| `--bonus R` | 0 (off) | Timed bonus food worth R points. Adds `_b{R}`. |
| `--render {grid,pixels}` | `grid` | Grid channels, or an 84x84 grayscale 4-frame stack. Pixels add `_px`. |
| `--frame-stack K` | 4 | Pixels: frames per observation. Adds `_fs{K}` when not 4. |
| `--food-curriculum R[:H]:N` | off | Food within R steps, held until H, grown to 2n by N. Tag `_fc{R}-{H}-{N}`. |
| `--food-relocate` | off | Needs a curriculum. Tag `_reloc`. |

Every Snake directory ends in `_abs`, for absolute actions, so new runs land next to the old absolute-action ones.

**Ms. Pac-Man board** (`pacman-run` only; directory `pac_px`):

| Option | Default | Meaning |
|---|---|---|
| `--frame-skip K` | 4 | Game ticks per agent step. Adds `_sk{K}` when not 4. |
| `--frame-stack K` | 4 | Frames per observation. Adds `_fs{K}` when not 4. |
| `--reward-units {small,atari}` | `small` | Reward scale. Adds `_atari`. |

**Run settings** (both commands):

| Option | Default | Meaning |
|---|---|---|
| `--eps-floor F` | 0.02 | Final ε. Tag `_eps{F}`. |
| `--eps-decay N` | unset | ε falls linearly from 1 to the floor over N steps (`0` = fixed at the floor). If unset, ε = `max(F, 1 - t/5000)`. Tag `_epsd{N}`. |
| `--eval-every K` | 0 (off) | Evaluation episodes every K steps. |
| `--eval-episodes M`, `--eval-eps E` | 5, 0.05 | Episodes per evaluation, and ε during evaluation. |
| `--probe-every K`, `--probe-far` | off | Representation probe (Snake only). |
| `--clip-reward` | off | The agent learns from rewards clipped to [−1, 1]. Pac-Man adds `_clip` to the dir. |
| `--opt {adam,rmsprop}`, `--lr F` | adam, the agent's | Optimizer. Tags `_{opt}`, `_lr{F}`. |
| `--agent-kw KEY=VALUE` | none | Constructor override, repeatable. Tag `_{KEY}{VALUE}`. |
| `--train-log-every K` | 1000 | Telemetry window (`0` = off). |
| `--save-agent`, `--save-every K` | off | Write `<stem>.agent.pkl` and `<stem>.agent_t{t}.pkl`. |
| `--results DIR` | `results` | Results root. |

The file stem is the agent name followed by the tags in this order: optimizer, learning rate, ε floor, ε decay,
curriculum, relocation, then the agent overrides. For example:

    snake-run dqn 1 1000000 --size 25 --map rooms --bonus 5 --eps-floor 0.1 --eps-decay 250000 \
        --food-curriculum 2:250000:600000 --food-relocate
    # -> results/g25_d0_rooms_b5_abs/dqn_eps0.1_epsd250000_fc2-250000-600000_reloc_s1.json

Settings that are not in the path (evaluation, probe and telemetry intervals) do not change training, so runs that
differ only in them overwrite each other.

### Readers

| Command | What it does |
|---|---|
| `snake-plot [DIR] [--out DIR]` | Learning curves (mean ± s.e. over seeds) for one results directory, grouped by agent and run settings, plus tables. It skips runs whose `steps` differ from the first. |
| `snake-peek JSON ...` | Per-seed score in 6k-step bins, and per-action memory write shares. |
| `snake-train RUN ... [--out PNG] [--smooth K]` | Telemetry curves for result JSONs or live `.train.jsonl` files. |
| `snake-watch AGENT [options]` | Plays a checkpoint on its own board (Snake or Ms. Pac-Man), or an untrained agent on `--size/--map/--bonus/--render`. Options: `--episodes`, `--eps` (default 0, greedy), `--delay`, `--seed`, `--food-radius R [--relocate]`, `--gui`, `--gif OUT [--max-frames N]`. |
| `pacman-watch AGENT [options]` | The same for Ms. Pac-Man; `--frame-skip` applies to an untrained agent. |
| `snake-geometry CKPT ... [--out DIR]`, `pacman-geometry CKPT ...` | Embedding geometry of checkpoints, one JSON each. |

In the terminal viewer, space pauses, `+`/`-` change the speed, `n` skips to the next episode and `q` quits.
Watching never changes a checkpoint.
