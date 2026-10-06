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

    snake-run nec 1 40000 4        # 4th arg: distractor channels -> results/d4/nec_s1.json
    snake-plot results/d0          # -> figures/learning_curves_d0.png + tables on stdout
    snake-peek results/d0/nec_*.json
    pytest

(`python -m snake_rl.run ...` etc. work too.)

## Layout

    src/snake_rl/
      env.py            Snake environment
      nn.py             MLP with manual backprop, Adam
      returns.py        N-step return accumulator
      agents/           base.py (Agent interface), dqn.py, nec.py, mfec.py; __init__.py holds the registry
      run.py plot.py peek.py
    tests/              gradient checks, N-step targets, smoke runs of every agent
    results/            per-run JSON, results/d{D} (7x7) or results/g{size}_d{D}
    figures/            learning-curve PNGs and saved tables
    logs/               stdout of past sweeps
    docs/experiments/   pre-registered experiments: protocol, run script, analysis, report

New agents: implement `act`/`observe` (see `agents/base.py`), add a factory to `AGENTS` in
`agents/__init__.py`, and optionally a legend name in `plot.LABELS`.

Known differences from the paper: MLP encoder by default (the `*_cnn` agents use a small CNN, see below, with 3x3 stride-1 convs rather than the Atari 8x8/4,4x4/2,3x3/1 stack), exact brute-force kNN instead of
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
  connected and the start is always clear. `open` is the original env, bit-for-bit.
- `--bonus R`: Nokia-style bonus food worth R points, appearing after every 4 regular foods for 2n steps. The
  observation has an extra channel holding its remaining lifetime fraction. It grows the snake like normal food.
  `score` in the results is the points collected (regular 1 + bonus R), so curves are not comparable with bonus-free runs.
- Output goes to `results/d{D}_{map}_b{R}` (e.g. `results/d0_pillars_b5`); the JSON records `map` and `bonus`.
- `dqn_cnn`, `dqn_nstep_cnn`, `ec_frozen_cnn`, `nec_cnn`: same agents with `nn.ConvNet` (conv 16, conv 32, FC 64, linear head;
  hand-written im2col backprop, gradient-checked in tests). CNN runs are ~3-10x slower than their MLP counterparts.
- `Snake.render()` prints the board as ASCII.
