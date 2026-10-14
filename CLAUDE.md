# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

This is a study project. It implements Neural Episodic Control (Pritzel et al., 2017), MFEC and DQN, and tests them on
Snake (grid or Atari-style pixels) and a small Ms. Pac-Man.

- **Status:** the study is closed. See `docs/STUDY_SUMMARY.md`.
- **Pure NumPy.** The MLP, the CNN, the optimizers and every gradient (including the one through the DND's kNN lookup)
  are hand-written on purpose. Do not introduce PyTorch, JAX, or other autograd or RL libraries.
- **Old experiments.** The code was simplified after the study closed. The experiments in `docs/experiments/` run as
  written at the git tag `pre-cleanup`.

## Commands

```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'   # once; then use .venv/bin/...

snake-run nec 1 40000                       # agent seed steps -> results/d0_abs/nec_s1.json
snake-run nec 1 60000 --size 10 --render pixels   # the Nature CNN on 84x84 frames -> results/g10_d0_px_abs/
snake-run dqn 1 40000 --map rooms --bonus 5 # maps: open pillars walls rooms -> results/d0_rooms_b5_abs/
pacman-run nec 1 100000 --reward-units atari --eval-every 10000   # -> results/pac_px_atari/
snake-plot results/d0_abs                   # -> figures/learning_curves_d0_abs.png, tables on stdout
snake-peek results/d0_abs/nec_*.json        # per-seed 6k-step bins + per-action DND write shares
snake-train results/d0_abs/dqn_s1.json      # training telemetry curves; also reads a live <stem>.train.jsonl
snake-run dqn 1 40000 --save-agent          # + --save-every K -> <stem>.agent.pkl (no replay buffer)
snake-watch results/d0_abs/dqn_s1.agent.pkl # terminal (--gui window, --gif out.gif); or an agent name, untrained
snake-geometry <ckpt>                       # embedding geometry; pacman-watch / pacman-geometry for Ms. Pac-Man

pytest -m "not slow"                        # fast tests (about 3.5 min)
pytest                                      # + the golden bit-for-bit runs (about 1.5 min more)
pytest tests/test_core.py::test_agent_runs -k nec   # a single test
```

Sweeps are plain shell loops over `snake-run` (or the Ansible specs in `ansible/`). A NEC run is roughly 10–25× slower
than a DQN run.

## Architecture

### Agents

- **Interface.** An agent implements the `Agent` protocol in `agents/base.py`: `act(obs, eps, t)` and
  `observe(obs, a, r, next_obs, term, trunc, t)`, plus an optional `diagnostics()`.
- **Registry.** Register a factory `(env, rng, **overrides) -> agent` in `AGENTS` (`agents/__init__.py`). There are six
  names: `random`, `mfec`, `dqn`, `dqn_nstep`, `ec_frozen`, `nec`.
  - **The network follows the observation:** `_agent` gives agents with a network the Nature CNN and uint8 replay when
    `env.pixels`, and the MLP otherwise.
  - Variants are constructor flags in the registry, not subclasses.
  - `LEGACY_NAMES` maps the old `*_naturecnn` names. `canonical()` and `canonical_stem()` normalise them in the readers.
- **Evaluation calls.** `act(obs, eps, t=None)` is an evaluation call: the agent must not touch its learning state (no
  LRU marks, no counters), and `run.evaluate` swaps in its own `rng`.
- **Plotting.** `plot.LABELS` sets legend names and plotting order.
- **Tests.** `tests/test_core.py::test_agent_runs` smoke-tests every registered agent on the grid, and the network
  agents on pixels too (`VARIANTS`).

### Running

- **`run.py` is the training loop.** `main(name, seed, steps, board, cfg)` takes:
  - a **board** (`boards.py`: `SnakeBoard` or `PacmanBoard`), which is everything about the game: env construction,
    the food curriculum, the results directory, file-name tags and recorded settings;
  - a **`RunConfig`**, which is everything else: the ε schedule, evaluation, probes, telemetry, optimizer,
    `clip_reward`, `agent_kw` and saving.
- **Recorded settings.** The result JSON and checkpoints are built from `board.settings()` and `cfg.settings()`.
  `board_from_settings` rebuilds a saved run's board for `snake-watch` and the geometry tools.
- **CLIs.** `snake-run` and `pacman-run` share one parser builder (`_parser(game)`).
- **Epsilon** defaults to 1 − t/5000, down to the floor. `--eps-decay N` makes it fall linearly from 1 to the floor
  over N steps, and `0` keeps it fixed at the floor.
- **Seeding.** The agent's rng is `seed`, the env's is `seed + 1000`, the evaluation env's is `seed + 2000` and
  evaluation exploration uses `seed + 3000`.
- **Episode log.** Episodes are logged as `(t, score, regular foods, truncated)`; bonus points are score − foods.
- **Diagnostics.** `diagnostics()` runs every 6000 steps.
- **Reproducibility.** Exploration happens inside `agent.act`, so the order in which an agent draws from `rng`
  determines reproducibility.

### Networks and training (`nn.py`)

- **Networks.** `MLP` and `ConvNet` share one interface. ConvNet specs are explicit `(channels, kernel, stride,
  padding)` tuples, and `NATURE_CONVS` is DQN's network. `make_net(kind, ...)` builds either one for DQN and NEC.
- **Optimizers.** `Optimizer` holds the global-norm clip (10), the step loop and the `last` telemetry. `Adam` and
  `RMSProp` define only `_update`.
- **Cached activations.** `backward` uses the activations cached by the most recent `forward`, so never call
  `forward` again between a forward and its backward. NEC's `act` caches the embedding in `self._h` for `observe`, so
  those two calls must stay paired.

### Supporting modules

- **`agents/replay.py`.** `Replay` is the circular buffer used by DQN (S, A, G, B, D) and NEC (obs, act, ret). It
  owns the uint8 encoding, and `sample(rng, batch)` draws `rng.integers(n, size=batch)`.
- **`returns.py`.** `NStep` produces `(payload, G, boot_obs, disc)` tuples, and every agent forms its target as
  `G + disc * max Q(boot_obs)`. `disc=0` only on true termination; truncation still bootstraps.
- **`agents/nec.py`.** There is one `DND` per action (a `KeyValueMemory`). kNN is exact brute force, and LRU
  eviction is keyed on "last used as a neighbour". NEC learns on two timescales:
  - **Fast:** tabular `write`, which updates on an exact key match and appends otherwise.
  - **Slow:** `_train` runs SGD from replay, with a hand-derived gradient into the encoder and into the neighbour keys
    and values (`mem_lr`).
  - `learn_embedding=False` (`ec_frozen`) turns the slow path off.
- **`env.py` (Snake).**
  - **Actions** are absolute: up, right, down, left, and a reversal keeps going straight.
  - **Grid observation:** the flattened `(C, n, n)` channels for head, body, food, obstacles (if `map != "open"`) and
    the bonus lifetime (if `bonus > 0`).
  - **Pixel observation** (`render="pixels"`): a `frames.FrameStack` of 84x84 gray frames, `obs_shape = (4, 84, 84)`,
    values k/255. `static_obs()` gives the stack for a hand-built state.
  - **`food_radius` and `relocate_food`** are the curriculum aids, driven by `SnakeBoard.on_step`.
  - **Truncation** happens after `2·n²` steps without eating.
  - **RNG.** Options that are off draw no random numbers.
- **`pacman.py`** (`MsPacman`). It has the same duck-typed env interface and uses the same `FrameStack`.
  `probe.py` and `geometry.py` are Snake-only; `geometry_pacman.py` is its counterpart.
- **`probe.py`.** `--probe-every K` scores the agent on a fixed hand-built state set. It needs a read-only
  `probe_embed(X) -> (Z, Q)` on the agent.
- **Telemetry.** Gradient-trained agents keep a `TrainStats` (`telemetry.py`), fed once per update, and expose
  `train_stats()`. It is RNG-neutral.
- **Checkpoints.** Agents' `__getstate__` drops the replay buffer, so a checkpoint can act but not resume training.
  `ConvNet` drops its im2col caches.
- **Result JSON.** Fields are `agent`, `seed`, `steps`, the board settings (`game`, `size`, `map`, `bonus`, `render`,
  `frame_stack`, `actions`, `food_*`, and for Pac-Man also `frame_skip` and `reward_units`), the run settings,
  `eval`, `episodes`, `evaluations`, `probes`, `train`, `diagnostics` and `wallclock_s`. Readers fall back on fields
  older files lack.
- **Result directories.** `d0` for 7×7, otherwise `g{size}_d0`. Optional parts are `_{map}`, `_b{bonus}`, `_px` and
  `_fs{K}`, and the name always ends in `_abs`. For Pac-Man it is `pac_px[_sk{K}][_fs{K}][_atari][_clip]`. These names
  are kept from before the cleanup so new runs sit next to the old ones.

## Gotchas

- **Golden runs.** `tests/test_golden.py` re-runs `tests/golden_cases.py` and requires behaviour identical to the
  `pre-cleanup` recordings. It also loads two old checkpoints.
  - Every refactor must keep it green.
  - When a CLI flag or agent name changes, edit the command lines in `golden_cases.py`. Never regenerate the expected
    files to make it pass.
- **One BLAS thread.** Bit-for-bit reproducibility needs one BLAS thread (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`). `tests/conftest.py` sets it for the tests; set it yourself for runs
  you compare.
- **`snake-plot`** takes the step count from the first JSON (sorted) and skips, with a message, any run whose `steps`
  differs.
- **Deliberate differences from the paper:** exact kNN, Adam by default, N = 50, DND capacity 2e4, key size 32 and the
  gradient clip. They are listed in the README.
