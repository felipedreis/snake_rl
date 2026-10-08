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

(`python -m snake_rl.run ...` etc. work too.)

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
- 4th/5th run args: distractor channels, grid size (results/d{D} for 7x7, results/g{size}_d{D} otherwise).
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
resume training. A checkpoint also records the run's settings, and `snake-watch` plays it on the board it was trained on,
with the food placement training used at that step (the curriculum's radius then, held fixed, and relocation if it was
on; food anywhere once the curriculum is over). The side panel says which food rule is in force:

    snake-watch results/d0/dqn_cnn_s1.agent.pkl                  # terminal, greedy
    snake-watch results/d0/dqn_cnn_s1.agent.pkl --gui            # matplotlib window
    snake-watch results/d0/dqn_cnn_s1.agent.pkl --gif out.gif    # save an animation
    snake-watch <checkpoint> --food-radius 2 --relocate          # override the food placement (or --food-radius any,
                                                                 # --no-relocate)
    snake-watch dqn --size 25 --map rooms --bonus 5              # an untrained agent, for comparison

The side panel shows the score, the Q-value the agent gives each action (straight, right, left) with the chosen one
marked and, for NEC, how far the current situation is from its memories. Terminal keys: space pause, `+`/`-` speed,
`n` next episode, `q` quit. Other options: `--episodes N`, `--eps E` (default 0, greedy), `--delay S`, `--seed K`.
Watching never changes the checkpoint.

**Representation probe.** `--probe-every K` scores the agent every K steps on a fixed set of states with food at most
2 steps away: greedy steering accuracy and how well its embedding separates where the food is relative to the snake
(`src/snake_rl/probe.py`). Stored under `probes` in the result JSON.

All of these are bookkeeping only: a run is bit-for-bit the same with them on or off.

