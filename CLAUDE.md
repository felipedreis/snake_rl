# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A study project: Neural Episodic Control (Pritzel et al., 2017) and other agents on a small Snake grid, compared as an ablation ladder. It is pure NumPy. The MLP, Adam, and every gradient (including those through the DND kNN lookup) are hand-written on purpose. Do not introduce PyTorch, JAX, or other autograd/RL libraries.

## Commands

```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'   # once; then use .venv/bin/...

snake-run nec 1 40000                 # agent seed steps [--distractors 0] [--size 7] [--results DIR]
snake-run nec_refresh 2 60000 --size 10   # -> results/g10_d0/nec_refresh_s2.json
snake-run nec_cnn 1 40000 --map rooms --bonus 5   # maps: open pillars walls rooms -> results/d0_rooms_b5/
snake-plot results/d0                 # -> figures/learning_curves_d0.png, tables on stdout
snake-peek results/g10_d0/nec_*.json  # per-seed 6k-step bins + per-action DND write shares
snake-train results/d0/dqn_s1.json     # training telemetry curves (loss, TD error, Q vs target, grad norm, ...); also reads a live <stem>.train.jsonl
snake-run dqn_cnn 1 40000 --save-agent # + --save-every K for checkpoints -> <stem>.agent.pkl (no replay buffer)
snake-watch results/d0/dqn_cnn_s1.agent.pkl   # play it in the terminal (--gui window, --gif out.gif); or an agent name, untrained

pytest                                # all tests
pytest tests/test_core.py::test_agent_runs -k nec   # a single test
```

Sweeps are plain shell loops over `snake-run`. Their stdout goes in `logs/`, and saved `snake-plot` tables go in `figures/table_*.txt`. A NEC run is roughly 10–25× slower than DQN (minutes compared with seconds).

## Architecture

- **Adding an agent:** write a class that satisfies the `Agent` protocol in `agents/base.py` (`act(obs, eps, t)` and `observe(obs, a, r, next_obs, term, trunc, t)`, plus an optional `diagnostics()`). Register a factory `(env, rng) -> agent` in `AGENTS` in `agents/__init__.py`. Variants such as `nec_refresh` are just constructor flags in the registry, not subclasses. Run settings that are not part of the agent itself (`--eps-floor`, `--eps-decay`, `--food-curriculum`, `--food-relocate`) belong in `run.py`, not in the agent name. `run.py` stores them in the JSON, and `run_name` tags them in the filename. `plot.py` groups runs by `(agent, eps_floor, eps_decay, food_curriculum, food_relocate)`. `act(obs, eps, t=None)` is an evaluation call: the agent must not touch its learning state (no LRU marks, no counters), and `run.evaluate` swaps in its own `rng`. `plot.LABELS` sets legend names and plotting order. Unlisted agents are still plotted, under their raw name. `tests/test_core.py::test_agent_runs` automatically smoke-tests every registered agent.
- `run.py` owns the training loop. It owns the epsilon schedule (default: 1 − t/5000 down to the floor, kept bit-for-bit; `--eps-decay N`: 1 → floor linearly over N steps, `0` = fixed at the floor), optional evaluation runs (`--eval-every K`: episodes at ε 0.05 on a separate env seeded `seed + 2000`, exploration rng `seed + 3000`; training is unchanged with or without them), the RNG seeding (agent rng = `seed`, env rng = `seed + 1000`), episode logging as `(t, score, regular foods, truncated)` (readers use the first two; bonus points = score − foods), and `diagnostics()` every 6000 steps. Exploration happens inside `agent.act`, so the order in which an agent draws from `rng` determines reproducibility.
- `nn.py`: `ConvNet` (same interface as `MLP`, selected with `encoder="cnn"` (small stride-1) or `"dqncnn"` (DQN-shaped: 32 3x3/1, 64 3x3/2, 64 3x3/2, FC 512; 25->13->7) + `obs_shape` in `NECAgent`/`DQNAgent`; `*_cnn` / `*_dqncnn` registry entries; `"naturecnn"` = DQN's exact network on the 84x84 pixel render: valid convs 32 8x8/4, 64 4x4/2, 64 3x3/1, FC 512, `*_naturecnn` entries; conv specs may carry an explicit padding) follows the same cache rule. `MLP.backward` uses the activations cached by the most recent `forward`, so never call `forward` again between a forward and its backward. NEC's `act` caches the embedding in `self._h` for `observe`, so those two calls must stay paired.
- `probe.py` (`--probe-every K`): every K steps (and at 0) scores the agent on a fixed hand-built state set (short straight snakes, food within 2 walkable steps): greedy steering accuracy and how well the embedding separates the egocentric food offset (kNN purity, ridge read-out, food/position distance ratio). Agents opt in with a read-only `probe_embed(X) -> (Z, Q)` (NEC: keys; DQN: last hidden layer). Results land in the JSON's `probes`; training is unchanged with it on (tested).
- Telemetry: agents with gradient training keep a `TrainStats` (`telemetry.py`) fed once per update (`Adam.last` adds grad norm, clip, update/weight ratio) and expose `train_stats()`. run.py snapshots it every `--train-log-every` (1000) steps into the JSON's `train` and streams it to `<stem>.train.jsonl` while running (kept afterwards). Pure bookkeeping: RNG-neutral (checked against committed runs).
- Checkpoints: agents' `__getstate__` drops replay buffers, so a pickle can act (snake-watch) but not resume training. The pickle also carries the run settings (`run.save`: board, epsilon schedule, food curriculum and relocation). `ConvNet` drops its im2col caches.
- `returns.py`: `NStep` produces `(payload, G, boot_obs, disc)` tuples, and every agent forms its target as `G + disc * max Q(boot_obs)`. `disc=0` only on true termination. Truncation still bootstraps.
- `agents/nec.py`: there is one `DND` per action. Exact brute-force kNN is used, and LRU eviction is keyed on "last used as a neighbour". Learning happens on two timescales:
  - Fast: tabular `write` (it updates on an exact key match, otherwise it appends).
  - Slow: `_train` runs SGD from replay, with a hand-derived gradient that flows into the encoder and also directly into neighbour keys and values (`mem_lr`).
  - `learn_embedding=False` (`ec_frozen`) turns off the slow path completely.
  - `refresh_every` stores the raw observations behind each key so that all keys can be re-embedded.
  - `bonus_beta` changes only action selection.
- `env.py`: with `render="pixels"` (`--render pixels`, results dir tag `_px`) the observation is instead the last 4 frames of an 84x84 grayscale drawing of the board, `obs_shape = (4, 84, 84)`, values k/255 (gray levels in `GRAY`; `static_obs()` gives a stack for a hand-built state, as the probe needs). The agents then keep replay observations as uint8 (`obs_u8`, exact for k/255). In grid mode the observation is flattened `env.obs_shape = (C, n, n)` channels: head, body, food, an obstacle channel if `map != "open"`, a bonus-food channel (remaining lifetime fraction) if `bonus > 0`, then D channels of i.i.d. noise resampled every step. `food_radius` (default None = anywhere) restricts regular food to cells within that many walkable steps of the head; run.py's `--food-curriculum R[:H]:N` holds it at R until step H, grows it to 2n by step N, then sets None. `relocate_food` (`--food-relocate`, needs a curriculum) re-places food uneaten for 2r + 5 steps near the head; it never acts once the radius is None. Defaults (`open`, no bonus, no food radius) reproduce the original env bit-for-bit, RNG included. Bonus food needs `score` read as points, not food count. Actions are relative (straight, right, left). An episode is truncated after `2·n²` steps without eating.
- Results JSON carries `agent`, `seed`, `steps`, `distractors`, `size`, `eps_floor`, `episodes`, `diagnostics`, and `wallclock_s`. The output directory is `results/d{D}` for 7×7 grids, otherwise `results/g{size}_d{D}`. Also `map` and `bonus` (fall back to `open` and 0; non-default values add `_{map}` / `_b{bonus}` to the results dir). Older JSONs may lack `distractors`, `size`, or `eps_floor`. Readers fall back to 0, 7, and 0.02 respectively.

## Gotchas

- `snake-plot` takes the step count from the first JSON (sorted) and skips, with a message, any run whose `steps` differs.
- Bit-for-bit reproducibility needs one BLAS thread (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`, as the experiment `run.sh` scripts set). With multi-threaded BLAS, the same seed can give different episodes, so check RNG-neutrality only under one thread.
- The existing results were produced before the restructure. The restructure was verified to give bit-for-bit identical episodes. Keep refactors RNG-neutral, or the new runs will not be comparable with old ones.
- The README's "Known differences from the paper" (MLP instead of CNN, exact kNN, Adam, N=50, DND capacity 2e4) are deliberate. `nec_refresh` and `nec_bonus` are extensions that are not in the paper.
