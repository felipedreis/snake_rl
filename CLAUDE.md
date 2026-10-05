# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A study project: Neural Episodic Control (Pritzel et al., 2017) and other agents on a small Snake grid, compared as an ablation ladder. It is pure NumPy. The MLP, Adam, and every gradient (including those through the DND kNN lookup) are hand-written on purpose. Do not introduce PyTorch, JAX, or other autograd/RL libraries.

## Commands

```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'   # once; then use .venv/bin/...

snake-run nec 1 40000                 # agent seed steps [distractors=0] [size=7] [--results DIR]
snake-run nec_refresh 2 60000 0 10    # -> results/g10_d0/nec_refresh_s2.json
snake-plot results/d0                 # -> figures/learning_curves_d0.png, tables on stdout
snake-peek results/g10_d0/nec_*.json  # per-seed 6k-step bins + per-action DND write shares

pytest                                # all tests
pytest tests/test_core.py::test_agent_runs -k nec   # a single test
```

Sweeps are plain shell loops over `snake-run`. Their stdout goes in `logs/`, and saved `snake-plot` tables go in `figures/table_*.txt`. A NEC run is roughly 10–25× slower than DQN (minutes compared with seconds).

## Architecture

- **Adding an agent:** write a class that satisfies the `Agent` protocol in `agents/base.py` (`act(obs, eps, t)` and `observe(obs, a, r, next_obs, term, trunc, t)`, plus an optional `diagnostics()`). Register a factory `(env, rng) -> agent` in `AGENTS` in `agents/__init__.py`. Variants such as `nec_refresh` are just constructor flags in the registry, not subclasses. Run settings that are not part of the agent itself (currently `--eps-floor`) belong in `run.py`, not in the agent name. `run.py` stores them in the JSON, and `run_name` tags them in the filename. `plot.py` groups runs by `(agent, eps_floor)`. `plot.LABELS` sets legend names and plotting order. Unlisted agents are still plotted, under their raw name. `tests/test_core.py::test_agent_runs` automatically smoke-tests every registered agent.
- `run.py` owns the training loop. It owns the epsilon schedule (1 → floor over 5k steps), the RNG seeding (agent rng = `seed`, env rng = `seed + 1000`), episode logging as `(t, score)`, and `diagnostics()` every 6000 steps. Exploration happens inside `agent.act`, so the order in which an agent draws from `rng` determines reproducibility.
- `nn.py`: `MLP.backward` uses the activations cached by the most recent `forward`, so never call `forward` again between a forward and its backward. NEC's `act` caches the embedding in `self._h` for `observe`, so those two calls must stay paired.
- `returns.py`: `NStep` produces `(payload, G, boot_obs, disc)` tuples, and every agent forms its target as `G + disc * max Q(boot_obs)`. `disc=0` only on true termination. Truncation still bootstraps.
- `agents/nec.py`: there is one `DND` per action. Exact brute-force kNN is used, and LRU eviction is keyed on "last used as a neighbour". Learning happens on two timescales:
  - Fast: tabular `write` (it updates on an exact key match, otherwise it appends).
  - Slow: `_train` runs SGD from replay, with a hand-derived gradient that flows into the encoder and also directly into neighbour keys and values (`mem_lr`).
  - `learn_embedding=False` (`ec_frozen`) turns off the slow path completely.
  - `refresh_every` stores the raw observations behind each key so that all keys can be re-embedded.
  - `bonus_beta` changes only action selection.
- `env.py`: the observation is flattened `(3 + D) × n × n` binary channels (head, body, food, then D channels of i.i.d. noise resampled every step). Actions are relative (straight, right, left). An episode is truncated after `2·n²` steps without eating.
- Results JSON carries `agent`, `seed`, `steps`, `distractors`, `size`, `eps_floor`, `episodes`, `diagnostics`, and `wallclock_s`. The output directory is `results/d{D}` for 7×7 grids, otherwise `results/g{size}_d{D}`. Older JSONs may lack `distractors`, `size`, or `eps_floor`. Readers fall back to 0, 7, and 0.02 respectively.

## Gotchas

- `snake-plot` takes the step count from the first JSON (sorted) and skips, with a message, any run whose `steps` differs.
- The existing results were produced before the restructure. The restructure was verified to give bit-for-bit identical episodes. Keep refactors RNG-neutral, or the new runs will not be comparable with old ones.
- The README's "Known differences from the paper" (MLP instead of CNN, exact kNN, Adam, N=50, DND capacity 2e4) are deliberate. `nec_refresh` and `nec_bonus` are extensions that are not in the paper.
