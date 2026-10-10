# NEC on Snake — ablation ladder

Pure-NumPy reproduction of Neural Episodic Control (Pritzel et al., 2017) on a 7x7 Snake,
with hand-derived gradients through the DND.

Agents: `dqn` (1-step), `dqn_nstep` (N=50), `ec_frozen` (NEC table, random frozen encoder,
no SGD — MFEC-like), `nec` (full). Setup: 40k env steps, 5 seeds, shared epsilon schedule
(1 -> 0.02 over 5k steps), gamma=0.99, MLP encoders, no tuning.

## Setup

    python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
    source .venv/bin/activate

## Usage

    snake-run nec 1 40000          # agent, seed, steps           -> results/d0/nec_s1.json
    snake-run nec 1 40000 --distractors 4   # noise channels      -> results/d4/nec_s1.json
    snake-run nec 1 40000 --size 10         # grid size           -> results/g10_d0/nec_s1.json
    snake-run nec 1 40000 --map rooms --bonus 5   # obstacles + bonus food -> results/d0_rooms_b5/nec_s1.json
    snake-plot results/d0          # -> figures/learning_curves_d0.png + tables on stdout
    snake-plot results/d0_rooms_b5 # -> figures/learning_curves_d0_rooms_b5.png
    snake-peek results/d0/nec_*.json
    snake-train results/d0/nec_s1.json            # training curves: loss, TD error, Q vs target, grad norm, ...
    snake-run dqn_cnn 1 40000 --save-agent        # also pickle the trained agent -> results/d0/dqn_cnn_s1.agent.pkl
    snake-watch results/d0/dqn_cnn_s1.agent.pkl   # watch it play in the terminal (--gui for a window, --gif out.gif)
    pytest

(`python -m snake_rl.run ...` etc. work too.) Every command prints its options with `--help`; the full list is in
[Command reference](#command-reference) below.

## Results data

The raw results (run JSONs, training telemetry, saved agents) live on Hugging Face, not in git:
[felipedreis/snake-rl-results](https://huggingface.co/datasets/felipedreis/snake-rl-results). To restore them:

    hf download felipedreis/snake-rl-results --repo-type dataset --local-dir results

The `.agent.pkl` files are Python pickles: only load them from a source you trust. `results/` and `*.agent.pkl` are in
`.gitignore`; new runs still write there, and are not tracked.

## Layout

    src/snake_rl/
      env.py            Snake environment
      nn.py             MLP with manual backprop, Adam
      returns.py        N-step return accumulator
      agents/           base.py (Agent interface), dqn.py, nec.py, mfec.py; __init__.py holds the registry
      probe.py          representation probe (--probe-every): steering and food-offset separability
      telemetry.py      per-update training statistics (loss, gradient norm, ...)
      run.py plot.py peek.py trainplot.py (snake-train) watch.py (snake-watch)
    tests/              gradient checks, N-step targets, smoke runs of every agent
    results/            per-run JSON, results/d{D} (7x7) or results/g{size}_d{D}, plus _{map} / _b{R} if set
    figures/            learning-curve PNGs and saved tables
    logs/               stdout of past sweeps
    docs/experiments/   pre-registered experiments: protocol, run script, analysis, report

New agents: implement `act`/`observe` (see `agents/base.py`), add a factory to `AGENTS` in
`agents/__init__.py`, and optionally a legend name in `plot.LABELS`.

Known differences from the paper: MLP encoder by default (the `*_cnn` agents use a small CNN, see below, with 3x3 stride-1 convs rather than the Atari 8x8/4,4x4/2,3x3/1 stack; the `*_dqncnn` agents copy DQN's shape on the grid), exact brute-force kNN instead of
kd-trees, Adam/SGD instead of RMSProp, N=50 instead of 100, DND capacity 2e4 per action.

## Later additions
- `--distractors D`, `--size n`: distractor channels, grid size (results/d{D} for 7x7, results/g{size}_d{D} otherwise).
- `nec_refresh`: re-embeds all DND keys with the current encoder every 1000 steps (not in paper).
- `nec_bonus`: density bonus beta*log(dbar_a / min_b dbar_b) when acting (not in paper).
- `--eps-floor F`: final epsilon (default 0.02). A non-default floor is recorded as `eps_floor` in the JSON,
  tagged in the filename (`nec_eps0.1_s1.json`) and in the plot label.
- NEC runs log per-action DND diagnostics (appends, exact updates, evictions, value stats) every 6k steps.
- `snake-peek <json...>`: per-seed curves in 6k-step bins + per-action write shares.
- `mfec`: Model-Free Episodic Control (Blundell et al., 2016). Random Gaussian projection to 64-d keys, one
  buffer per action, uniform mean over k=11 neighbours (stored value on exact match), and backward Monte Carlo
  returns written at episode end with a max update. No SGD and no bootstrapping (truncated episodes are not
  bootstrapped either). Capacity is 2e4 per action, as for the DND (the paper uses 1e6), and run.py's epsilon
  schedule is used (the paper uses 0.005).
- `random`: uniform random policy, a reference floor.
- `docs/experiments/mfec_dqn_nec/`: pre-registered MFEC vs DQN vs NEC comparison (PROTOCOL.md, REPORT.md).

## Richer env and CNN agents
- `--map {open,pillars,walls,rooms}`: obstacle layouts (lethal walls, extra observation channel). The free area is always
  connected and the start is always clear. `open` is the original env, bit-for-bit. The layouts on the default 7x7 grid
  (`#` wall, `@` head, `o` body; they scale with the grid size):

      pillars    walls      rooms
      .......    .......    ...#...
      .#...#.    #####..    ...#...
      .......    .......    .......
      ..o@...    ..o@...    ##o@.##
      .......    .......    .......
      .#...#.    ..#####    ...#...
      .......    .......    ...#...

  Example: `snake-run dqn 1 40000 --size 10 --map walls` writes `results/g10_d0_walls/dqn_s1.json`. Map, bonus, grid size and
  distractors all combine, and each combination gets its own results dir, so `snake-plot` never mixes them.
- `--bonus R`: Nokia-style bonus food worth R points, appearing after every 4 regular foods for 2n steps. The
  observation has an extra channel holding its remaining lifetime fraction. It grows the snake like normal food.
  `score` in the results is the points collected (regular 1 + bonus R), so curves are not comparable with bonus-free runs.
- Output goes to `results/d{D}_{map}_b{R}` (e.g. `results/d0_pillars_b5`); the JSON records `map` and `bonus`.
- `dqn_cnn`, `dqn_nstep_cnn`, `ec_frozen_cnn`, `nec_cnn`: same agents with `nn.ConvNet` (conv 16, conv 32, FC 64, linear head;
  hand-written im2col backprop, gradient-checked in tests). CNN runs are ~3-10x slower than their MLP counterparts.
- `dqn_dqncnn`, `nec_dqncnn`: DQN's convolutional shape scaled to the grid (conv 32 3x3/1, conv 64 3x3/2, conv 64 3x3/2,
  FC 512; 25x25 -> 25 -> 13 -> 7, the same 7x7 map DQN's convolutions leave on Atari). Known issue: on 25x25, NEC's
  embedding diverges with either CNN encoder (see `docs/experiments/encoder_food_probe_g25/REPORT.md`, addendum).
- `--render pixels` with `dqn_naturecnn`, `dqn_nstep_naturecnn`, `ec_frozen_naturecnn`, `nec_naturecnn`: the Atari-style
  setting. The board is drawn as an 84x84 grayscale image (each cell `84 // n` pixels, e.g. 3 on 25x25, 12 on 7x7), and
  the observation is the last 4 frames, as DQN and NEC see Atari (`obs_shape = (4, 84, 84)`; no action repeat, since
  nothing flickers). The `*_naturecnn` agents use DQN's exact network, valid convolutions 32 8x8/4, 64 4x4/2, 64 3x3/1
  (84 -> 20 -> 9 -> 7), FC 512, then the Q head (DQN) or the 32-d key (NEC). Replay stores frames as uint8 (exact).
  Results go to `..._px` (e.g. `results/d0_px`). About 1.7M weights; a run costs a few times a `*_dqncnn` run.
- `Snake.render()` prints the board as ASCII.

## Run options for exploration, evaluation and curricula
- `--eps-decay N`: epsilon falls linearly from 1 to `--eps-floor` over N steps; `0` keeps it fixed at the floor
  (DQN paper: `--eps-floor 0.1 --eps-decay 250000`; MFEC paper: `--eps-floor 0.005 --eps-decay 0`). Unset, the
  original 5k-step schedule is used, bit-for-bit.
- `--eval-every K [--eval-episodes M --eval-eps E]`: every K steps, play M episodes at epsilon E (default 0.05,
  as in the DQN paper) on a separate env. Stored under `evaluations`; training is unchanged by it.
- `--food-curriculum R:N` or `R:H:N`: regular food appears within R walkable steps of the head (around walls and
  the body), the radius held at R until step H (default 0) and then growing linearly to 2n by step N; after that,
  anywhere, as in the real game. `--food-relocate`: during the curriculum, food not eaten within 2r + 5 steps is
  placed again within r of the head's current position, so an agent that wanders off still has food nearby. Food far away is
  what large boards make hard: on 25x25 `rooms`, food starts on average ~13 steps from the head, and only 7% of
  placements are within 5. Evaluation runs always place food anywhere.

      snake-run nec 1 1000000 --size 25 --map rooms --bonus 5 --food-curriculum 2:400000 --eval-every 10000
      # -> results/g25_d0_rooms_b5/nec_fc2-400000_s1.json

## Watching training and the agent

**Training curves.** Every agent that trains by gradient (DQN and NEC variants) logs window averages of its updates
every 1,000 steps (`--train-log-every K`, `0` = off): loss, |TD error|, Q(s,a) and its target, gradient norm before
clipping, share of clipped steps, step size relative to the weights, and dead ReLUs; NEC adds the distance to its
memory neighbours and the largest kernel weight. They are stored under `train` in the result JSON and also written,
one line per snapshot, to `<run>.train.jsonl` as the run goes, so a run can be inspected before it finishes.

    snake-train results/d0/dqn_cnn_s1.json --out figures/train_dqn_cnn.png
    snake-train results/d0/dqn_s*.json results/d0/nec_s*.json --smooth 5   # compare runs; one line per seed
    snake-train results/d0/nec_s1.train.jsonl                              # a run still in progress

The PNG has one panel per quantity, and the terminal shows the latest values of each run. The docstring of
`src/snake_rl/trainplot.py` lists what healthy values look like. Results from before this feature have no `train`
field.

**Watching the agent play.** `--save-agent` pickles the trained agent to `<run>.agent.pkl`, and `--save-every K`
adds checkpoints `<run>.agent_t{K}.pkl` along the way. The replay buffer is left out, so a checkpoint can act but not
resume training. `snake-watch` plays it on the board it was trained on:

    snake-watch results/d0/dqn_cnn_s1.agent.pkl                  # terminal, greedy
    snake-watch results/d0/dqn_cnn_s1.agent.pkl --gui            # matplotlib window
    snake-watch results/d0/dqn_cnn_s1.agent.pkl --gif out.gif    # save an animation
    snake-watch <checkpoint> --food-radius 2 --relocate          # with the training curriculum's food placement
    snake-watch dqn --size 25 --map rooms --bonus 5              # an untrained agent, for comparison

The side panel shows the score, the Q-value the agent gives each action (straight, right, left) with the chosen one
marked and, for NEC, how far the current situation is from its memories. Terminal keys: space pause, `+`/`-` speed,
`n` next episode, `q` quit. Other options: `--episodes N`, `--eps E` (default 0, greedy), `--delay S`, `--seed K`.
Watching never changes the checkpoint.

**Representation probe.** `--probe-every K` scores the agent every K steps on a fixed set of states with food at most
2 steps away: greedy steering accuracy and how well its embedding separates where the food is relative to the snake
(`src/snake_rl/probe.py`). Stored under `probes` in the result JSON.

All of these are bookkeeping only: a run is bit-for-bit the same with them on or off.


## Command reference

All options are named except the few positional arguments listed first for each command. Defaults reproduce the
original setup (7x7 open board, no bonus, the 5k-step epsilon schedule, food anywhere).

### `snake-run AGENT SEED STEPS [options]`

Trains one agent for one seed and writes `<results>/<dir>/<stem>_s<SEED>.json` (see
[Result file names](#result-file-names)).

| Argument | Default | Meaning |
|---|---|---|
| `AGENT` | | One of `random`, `dqn`, `dqn_nstep`, `mfec`, `ec_frozen`, `nec`, `nec_refresh`, `nec_bonus`, `dqn_cnn`, `dqn_nstep_cnn`, `ec_frozen_cnn`, `nec_cnn`, `dqn_dqncnn`, `nec_dqncnn`, and for `--render pixels` `dqn_naturecnn`, `dqn_nstep_naturecnn`, `ec_frozen_naturecnn`, `nec_naturecnn` (see `agents/__init__.py`). |
| `SEED` | | Agent RNG seed; the env uses `SEED + 1000`. |
| `STEPS` | | Environment steps to train for. |

**Board** (each combination gets its own results dir):

| Option | Default | Meaning |
|---|---|---|
| `--size n` | 7 | Grid size n x n. Dir `d{D}` for 7, `g{n}_d{D}` otherwise. |
| `--distractors D` | 0 | Extra observation channels of i.i.d. coin-flip noise, resampled every step. |
| `--map M` | `open` | Obstacle layout: `open`, `pillars`, `walls`, `rooms`. Adds `_{M}` to the dir. |
| `--bonus R` | 0 (off) | Timed bonus food worth R points after every 4 regular foods. Adds `_b{R}` to the dir; `score` becomes points. |
| `--render {grid,pixels}` | `grid` | Observation: `grid` channels, or `pixels`, an 84x84 grayscale image with a 4-frame stack (use the `*_naturecnn` agents; no distractors). Adds `_px` to the dir. |

**Exploration** (tagged in the file name when not default):

| Option | Default | Meaning |
|---|---|---|
| `--eps-floor F` | 0.02 | Final epsilon. Tag `_eps{F}`. |
| `--eps-decay N` | unset | Epsilon falls linearly from 1 to the floor over N steps; `0` = fixed at the floor. Unset: `max(F, 1 - t/5000)`. Tag `_epsd{N}`. |

**Food curriculum** (training only; evaluation always places food anywhere):

| Option | Default | Meaning |
|---|---|---|
| `--food-curriculum R[:H]:N` | off | Food within R walkable steps of the head, held until step H (default 0), growing to 2n by step N, then anywhere. Tag `_fc{R}-{H}-{N}`. |
| `--food-relocate` | off | Needs a curriculum. Food uneaten for 2r + 5 steps is placed again within r of the head. Tag `_reloc`. |

**Measurement** (bookkeeping only: training is bit-for-bit the same with these on or off):

| Option | Default | Meaning |
|---|---|---|
| `--eval-every K` | 0 (off) | Every K steps, play evaluation episodes on a separate env (seed `SEED + 2000`). Stored under `evaluations`. |
| `--eval-episodes M` | 5 | Episodes per evaluation. |
| `--eval-eps E` | 0.05 | Epsilon during evaluation. |
| `--probe-every K` | 0 (off) | Every K steps, score steering and the embedding on a fixed probe set (`probe.py`). Stored under `probes`. Not with distractors. |
| `--train-log-every K` | 1000 | Training telemetry window (loss, TD error, grad norm, ...) for gradient-trained agents, stored under `train` and streamed to `<stem>.train.jsonl`. `0` = off. |

**Output:**

| Option | Default | Meaning |
|---|---|---|
| `--results DIR` | `results` | Results root directory. |
| `--save-agent` | off | Pickle the trained agent to `<stem>.agent.pkl`, with the run's settings (no replay buffer: it can act, not resume). |
| `--save-every K` | 0 (off) | Also pickle a checkpoint `<stem>.agent_t{t}.pkl` every K steps. |

### Result file names

A run's path is `<results>/<dir>/<stem>_s<SEED>.<ext>`. The **directory** names the board; the **file name** names
the agent and every non-default training setting. Each part appears only when it differs from its default, in this order:

| Part | Comes from | Default (no part) | Example |
|---|---|---|---|
| `<results>` | `--results DIR` | `results` | `results/exp_nec_ladder` |
| `<dir>` start | `--distractors D`, `--size n` | | `d0` for 7x7, otherwise `g{n}_d{D}`, e.g. `g25_d0` |
| `_{map}` | `--map M` | `open` | `_rooms` |
| `_b{R}` | `--bonus R` | 0 | `_b5` |
| `<stem>` start | `AGENT` | | `nec` |
| `_eps{F}` | `--eps-floor F` | 0.02 | `_eps0.1` |
| `_epsd{N}` | `--eps-decay N` | unset | `_epsd250000` |
| `_fc{R}-{H}-{N}` | `--food-curriculum R[:H]:N` (as given: two or three numbers) | off | `_fc2-250000-600000` |
| `_reloc` | `--food-relocate` | off | `_reloc` |
| `_s{SEED}` | `SEED` | | `_s1` |

For example, `snake-run dqn 1 1000000 --size 25 --map rooms --bonus 5 --eps-floor 0.1 --eps-decay 250000
--food-curriculum 2:250000:600000 --food-relocate` writes
`results/g25_d0_rooms_b5/dqn_eps0.1_epsd250000_fc2-250000-600000_reloc_s1.json`.

Files written next to it share the same prefix:

| File | Written when |
|---|---|
| `<stem>_s<SEED>.json` | Always: episodes, settings, diagnostics, evaluations, probes, telemetry. |
| `<stem>_s<SEED>.train.jsonl` | The agent trains by gradient and `--train-log-every` is not 0 (streamed during the run). |
| `<stem>_s<SEED>.agent.pkl` | `--save-agent`: the agent at the end of training. |
| `<stem>_s<SEED>.agent_t{t}.pkl` | `--save-every K`: a checkpoint at every step t that is a multiple of K. |

Settings that are not in the path (evaluation, probe and telemetry intervals) do not change training, so runs that
differ only in them overwrite each other. `snake-plot` reads the settings from the JSON, not from the path.

### `snake-plot [DIR] [--out DIR]`

Learning curves (mean ± s.e. over seeds) for every run in one results dir, grouped by agent and run settings, plus
tables on stdout. `DIR` defaults to `results/d0`; the PNG goes to `--out` (default `figures`) as
`learning_curves_<dir>.png`. Runs whose `steps` differ from the first one are skipped with a message.

### `snake-peek JSON [JSON ...]`

Per-seed mean score in 6k-step bins and, for agents that log DND diagnostics, the share of writes per action.

### `snake-train RUN [RUN ...] [options]`

Training telemetry curves for result JSONs or live `<stem>.train.jsonl` files (colour = run setting, one line per seed).

| Option | Default | Meaning |
|---|---|---|
| `--out PNG` | `figures/training.png` | Output image. |
| `--smooth K` | 1 | Moving average over K snapshots. |

### `snake-watch AGENT [options]`

Plays new episodes with a checkpoint (`.agent.pkl`) on the board it was trained on, or with an untrained agent given by
name. It does not replay training episodes, and it never changes the checkpoint: the agent acts as in evaluation, with
no learning.

| Option | Default | Meaning |
|---|---|---|
| `--episodes N` | 5 | Episodes to play. |
| `--eps E` | 0 | Exploration while watching (0 = greedy). |
| `--delay S` | 0.08 | Seconds per step. |
| `--seed K` | 0 | Seed of the watch env (`K + 5000`) and its exploration. |
| `--food-radius R` | off | Place food within R walkable steps of the head, as in a food curriculum. |
| `--relocate` | off | Re-place uneaten food near the head; only acts together with `--food-radius`. |
| `--gui` | off | Matplotlib window instead of the terminal. |
| `--gif OUT` | off | Save a GIF instead of showing. |
| `--max-frames N` | 600 | GIF length cap. |
| `--size n`, `--map M`, `--bonus R` | 7, `open`, 0 | Board for an untrained agent name (a checkpoint brings its own). |
