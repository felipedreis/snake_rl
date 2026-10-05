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
      agents/           base.py (Agent interface), dqn.py, nec.py; __init__.py holds the registry
      run.py plot.py peek.py
    tests/              gradient checks, N-step targets, smoke runs of every agent
    results/            per-run JSON, results/d{D} (7x7) or results/g{size}_d{D}
    figures/            learning-curve PNGs and saved tables
    logs/               stdout of past sweeps

New agents: implement `act`/`observe` (see `agents/base.py`), add a factory to `AGENTS` in
`agents/__init__.py`, and optionally a legend name in `plot.LABELS`.

Known differences from the paper: MLP instead of CNN, exact brute-force kNN instead of
kd-trees, Adam/SGD instead of RMSProp, N=50 instead of 100, DND capacity 2e4 per action.

## Later additions
- 4th/5th run args: distractor channels, grid size (results/d{D} for 7x7, results/g{size}_d{D} otherwise).
- `nec_refresh`: re-embeds all DND keys with the current encoder every 1000 steps (not in paper).
- `nec_bonus`: density bonus beta*log(dbar_a / min_b dbar_b) when acting (not in paper).
- suffix `_eps10`: epsilon floor 0.1 instead of 0.02.
- NEC runs log per-action DND diagnostics (appends, exact updates, evictions, value stats) every 6k steps.
- `snake-peek <json...>`: per-seed curves in 6k-step bins + per-action write shares.
