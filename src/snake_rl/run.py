"""Run one (agent, seed) and save per-episode scores.

usage: snake-run <agent> <seed> <steps> [board: --size n --map M --bonus R --render grid|pixels --frame-stack K
                 --food-curriculum R[:H]:N [--food-relocate]] [run settings, below]
       pacman-run <agent> <seed> <steps> [--frame-skip K] [--frame-stack K] [--reward-units small|atari] [run settings]
run settings: [--eps-floor F] [--eps-decay N] [--eval-every K [--eval-episodes M] [--eval-eps E]]
              [--probe-every K [--probe-far]] (Snake only) [--clip-reward] [--opt adam|rmsprop] [--lr F]
              [--agent-kw KEY=VALUE ...] [--train-log-every K] [--save-agent] [--save-every K] [--results DIR]
       (or: python -m snake_rl.run ...)
agents: see snake_rl.agents.AGENTS. The board (boards.py) is the game side of a run; RunConfig is everything else.
"""
import argparse
import ast
import json
import os
import pickle
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from snake_rl.agents import AGENTS, canonical, make_agent
from snake_rl.boards import PacmanBoard, SnakeBoard, food_radius  # noqa: F401 (food_radius: re-exported)
from snake_rl.env import MAPS, RENDERS
from snake_rl.probe import make_far_probe_set, make_probe_set, probe


EPS_FLOOR = 0.02


@dataclass(frozen=True)
class RunConfig:
    """Every run setting that is not the board: exploration, evaluation, probes, logging, optimizer, agent overrides."""
    eps_floor: float = EPS_FLOOR
    eps_decay: int | None = None    # see epsilon()
    eval_every: int = 0             # evaluation episodes every K steps on a separate env (0 = off)
    eval_episodes: int = 5
    eval_eps: float = 0.05
    probe_every: int = 0            # representation probe every K steps (Snake only; 0 = off)
    probe_far: bool = False
    train_log_every: int = 1000     # training telemetry window (0 = off)
    save_agent: bool = False        # pickle the agent at the end
    save_every: int = 0             # and every K steps
    opt: str = "adam"
    lr: float | None = None         # None: the agent's default
    clip_reward: bool = False       # the agent learns from rewards clipped to [-1, 1] (scores stay raw)
    agent_kw: dict = field(default_factory=dict)  # agent constructor overrides
    root: str = "results"           # results root directory

    def settings(self):
        """The fields written into the result JSON and checkpoints (all but the output root)."""
        d = asdict(self)
        del d["root"]
        return d


def epsilon(t, floor=EPS_FLOOR, decay=None):
    """Exploration rate at step t.

    decay=None: the original schedule, kept bit-for-bit so old runs reproduce: 1 - t/5000 until it reaches
    the floor. decay=N: linear from 1 to the floor over exactly N steps, then the floor (DQN's schedule).
    decay=0: the floor from the start, i.e. a fixed epsilon (MFEC's schedule).
    """
    if decay is None:
        return max(floor, 1.0 - t / 5000)
    return floor if t >= decay else 1.0 + (floor - 1.0) * t / decay


def run_name(name, cfg=RunConfig(), board=SnakeBoard()):
    """Result file stem (minus seed): the agent name, tagged with any non-default run settings."""
    return (name + ("" if cfg.opt == "adam" else f"_{cfg.opt}") + ("" if cfg.lr is None else f"_lr{cfg.lr:g}")
            + ("" if cfg.eps_floor == EPS_FLOOR else f"_eps{cfg.eps_floor:g}")
            + ("" if cfg.eps_decay is None else f"_epsd{cfg.eps_decay}")
            + board.tags()
            + "".join(f"_{k}{v:g}" if isinstance(v, (int, float)) else f"_{k}{v}" for k, v in sorted(cfg.agent_kw.items())))


def evaluate(agent, env, rng, episodes, eps):
    """Play `episodes` episodes on a separate env with epsilon `eps`, as the DQN paper's evaluation runs do.

    act gets t=None (no learning bookkeeping) and draws from `rng` instead of the agent's own stream, and
    observe is never called, so training is bit-for-bit the same with or without evaluation.
    Returns one (score, regular foods, truncated, length) tuple per episode.
    """
    saved, agent.rng = agent.rng, rng
    out = []
    try:
        for _ in range(episodes):
            obs, n, done = env.reset(), 0, False
            while not done:
                obs, _, term, trunc = env.step(agent.act(obs, eps, None))
                n, done = n + 1, term or trunc
            out.append((env.score, env.foods, int(trunc), n))
    finally:
        agent.rng = saved
    return out


def save(agent, path, t, settings):
    """Pickle the agent (without its replay buffer) at step t, plus the run settings snake-watch needs to replay it
    on the board it was trained on (food curriculum and relocation included)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dict(agent=agent, t=t, **settings), f)


def main(name, seed, steps, board=SnakeBoard(), cfg=RunConfig()):
    """Train agent `name` with seed `seed` for `steps` env steps on `board`; write <stem>.json (and checkpoints)."""
    if cfg.probe_every and not isinstance(board, SnakeBoard):
        raise ValueError("the representation probe (--probe-every) exists for Snake only")
    name = canonical(name)  # legacy *_naturecnn names run (and are saved) under the current name
    rng = np.random.default_rng(seed)
    env = board.make_train_env(seed + 1000)
    agent = make_agent(name, env, rng, **cfg.agent_kw)
    if cfg.opt != "adam" or cfg.lr is not None:  # the agent's defaults (Adam 5e-4) unless asked; no rng draws
        if not hasattr(agent, "set_optimizer"):
            raise ValueError(f"{name} has no optimizer to set (--opt / --lr)")
        agent.set_optimizer(cfg.opt, cfg.lr)
    # Evaluation always places food anywhere: it measures the real task, whatever the training curriculum.
    if cfg.eval_every:  # own env and RNG streams, so evaluation never shifts training's random numbers
        eval_env = board.make_env(seed + 2000)
        eval_rng = np.random.default_rng(seed + 3000)
    # Representation probe (probe.py): a fixed hand-built state set, no rng shared with training.
    P = F = None
    if cfg.probe_every:
        where = dict(size=board.size, map=board.map, bonus=board.bonus, render=board.render, frame_stack=board.frame_stack)
        P = make_probe_set(**where)
        F = make_far_probe_set(**where) if cfg.probe_far else None  # steering by distance and tie fractions
    probes = [(0, probe(agent, P, F))] if P is not None else []
    out_dir = board.results_dir(cfg.root, cfg.clip_reward)
    stem = f"{out_dir}/{run_name(name, cfg, board)}_s{seed}"
    settings = dict(name=name, seed=seed, **board.settings(), **cfg.settings())
    # Training telemetry (telemetry.py): window means every train_log_every steps. Also streamed, one JSON line per
    # snapshot, to <stem>.train.jsonl as the run goes (kept afterwards), so `snake-train` can plot a run in progress.
    tlog = hasattr(agent, "train_stats") and cfg.train_log_every > 0
    if tlog:
        os.makedirs(out_dir, exist_ok=True)
        live = open(f"{stem}.train.jsonl", "w")
    obs = env.reset()
    log, diag, evals, train, t0 = [], [], [], [], time.time()
    for t in range(1, steps + 1):
        board.on_step(env, t)
        a = agent.act(obs, epsilon(t, cfg.eps_floor, cfg.eps_decay), t)
        nobs, r, term, trunc = env.step(a)
        agent.observe(obs, a, float(np.clip(r, -1.0, 1.0)) if cfg.clip_reward else r, nobs, term, trunc, t)  # DQN-style clipping is the agent's: score stays raw
        obs = nobs
        if term or trunc:
            # (end step, score, regular foods eaten, 1 if cut off by the idle limit else 0). Readers that
            # predate the last two fields use only the first two, so old and new JSONs read the same way.
            log.append((t, env.score, env.foods, int(trunc)))
            obs = env.reset()
        if t % 6000 == 0 and hasattr(agent, "diagnostics"):
            diag.append((t, agent.diagnostics()))
        if cfg.eval_every and t % cfg.eval_every == 0:
            evals.append((t, evaluate(agent, eval_env, eval_rng, cfg.eval_episodes, cfg.eval_eps)))
        if P is not None and t % cfg.probe_every == 0:
            probes.append((t, probe(agent, P, F)))
        if cfg.save_every and t % cfg.save_every == 0:
            save(agent, f"{stem}.agent_t{t}.pkl", t, settings)
        if tlog and t % cfg.train_log_every == 0:
            s = agent.train_stats()
            if s:
                train.append((t, s))
                live.write(json.dumps({"t": t, **s}) + "\n")
                live.flush()
    out = {"agent": name, "seed": seed, "steps": steps, **board.settings(), **cfg.settings(),
           "eval": {"every": cfg.eval_every, "episodes": cfg.eval_episodes, "eps": cfg.eval_eps} if cfg.eval_every else None,
           "episodes": log, "diagnostics": diag, "evaluations": evals, "probes": probes, "train": train,
           "wallclock_s": time.time() - t0}
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{stem}.json", "w") as f:
        json.dump(out, f)
    if cfg.save_agent:
        save(agent, f"{stem}.agent.pkl", steps, settings)
    if tlog:
        live.close()
    print(run_name(name, cfg, board), seed, f"{len(log)} eps, {time.time()-t0:.0f}s")


def _parser(game):
    """The argument parser of `snake-run` or `pacman-run`: the game's board options, then the shared run settings."""
    ap = argparse.ArgumentParser(prog=f"{game}-run", description=f"Train one agent on {'Snake' if game == 'snake' else 'Ms. Pac-Man'} and save its scores.")
    ap.add_argument("agent", help=f"one of {sorted(AGENTS)}")
    ap.add_argument("seed", type=int)
    ap.add_argument("steps", type=int)
    b = ap.add_argument_group("board")
    if game == "snake":
        b.add_argument("--size", type=int, default=7, help="grid size")
        b.add_argument("--map", default="open", choices=MAPS, help="obstacle layout")
        b.add_argument("--bonus", type=float, default=0.0, help="reward of the timed bonus food (0 = off)")
        b.add_argument("--render", default="grid", choices=RENDERS,
                       help="observation: 'grid' channels (default) or 'pixels', an 84x84 grayscale image with a "
                            "4-frame stack as DQN/NEC see Atari (the agents then use DQN's Nature CNN)")
        b.add_argument("--frame-stack", type=int, default=None,
                       help="--render pixels: stack the last K frames (default 4, as DQN)")
        b.add_argument("--food-curriculum", default=None, metavar="R[:H]:N",
                       help="place food within R walkable steps of the head (held until step H), growing to the "
                            "whole board by step N, then anywhere (default: anywhere from the start)")
        b.add_argument("--food-relocate", action="store_true",
                       help="during the curriculum, re-place food not eaten within 2r+5 steps near the head")
    else:
        b.add_argument("--frame-skip", type=int, default=4, help="game ticks per agent step (Atari: 4)")
        b.add_argument("--frame-stack", type=int, default=None, help="stack the last K frames (default 4, as DQN)")
        b.add_argument("--reward-units", default="small", choices=("small", "atari"),
                       help="'small' (points/100, -1 for death; default) or 'atari' (10/50/200 points, nothing for dying)")
    r = ap.add_argument_group("run settings")
    r.add_argument("--eps-floor", type=float, default=EPS_FLOOR, help="final epsilon after the decay")
    r.add_argument("--eps-decay", type=int, default=None,
                   help="steps for epsilon to fall linearly from 1 to the floor (0 = fixed at the floor; "
                        "default: the original 5k-step schedule)")
    r.add_argument("--eval-every", type=int, default=0,
                   help="every K steps, play evaluation episodes on a separate env (0 = off)")
    r.add_argument("--eval-episodes", type=int, default=5, help="episodes per evaluation")
    r.add_argument("--eval-eps", type=float, default=0.05, help="epsilon during evaluation (DQN paper: 0.05)")
    if game == "snake":
        r.add_argument("--probe-every", type=int, default=0,
                       help="every K steps, measure the encoder and greedy steering on a fixed probe set (probe.py)")
        r.add_argument("--probe-far", action="store_true",
                       help="with --probe-every, also probe food up to 6 steps away: steering by distance and tie fractions")
    r.add_argument("--clip-reward", action="store_true",
                   help="the agent learns from rewards clipped to [-1, 1], as the DQN paper's does (scores are still logged raw)")
    r.add_argument("--opt", default="adam", choices=["adam", "rmsprop"],
                   help="optimizer of the gradient-trained agents (rmsprop: rho 0.95, eps 1e-2, as in the DQN/NEC papers)")
    r.add_argument("--lr", type=float, default=None, help="optimizer learning rate (default: the agent's, Adam 5e-4)")
    r.add_argument("--agent-kw", action="append", default=[], metavar="KEY=VALUE",
                   help="override an agent constructor setting, e.g. target_every=40000, N=100, dnd_cap=100000 (repeatable; "
                        "recorded in the JSON and the file name)")
    r.add_argument("--train-log-every", type=int, default=1000,
                   help="every K steps, log training telemetry (loss, TD error, grad norm, ...; 0 = off)")
    r.add_argument("--save-agent", action="store_true", help="pickle the trained agent to <stem>.agent.pkl, for snake-watch")
    r.add_argument("--save-every", type=int, default=0, help="also pickle a checkpoint every K steps")
    r.add_argument("--results", default="results", help="results root directory")
    return ap


def _config(a):
    return RunConfig(eps_floor=a.eps_floor, eps_decay=a.eps_decay, eval_every=a.eval_every, eval_episodes=a.eval_episodes,
                     eval_eps=a.eval_eps, probe_every=getattr(a, "probe_every", 0), probe_far=getattr(a, "probe_far", False),
                     train_log_every=a.train_log_every, save_agent=a.save_agent, save_every=a.save_every, opt=a.opt, lr=a.lr,
                     clip_reward=a.clip_reward, root=a.results,
                     agent_kw={k: ast.literal_eval(v) for k, v in (kv.split("=", 1) for kv in a.agent_kw)})


def cli():
    """snake-run"""
    a = _parser("snake").parse_args()
    fc = tuple(int(x) for x in a.food_curriculum.split(":")) if a.food_curriculum else None
    board = SnakeBoard(size=a.size, map=a.map, bonus=a.bonus, render=a.render, frame_stack=a.frame_stack,
                       food_curriculum=fc, food_relocate=a.food_relocate)
    main(a.agent, a.seed, a.steps, board, _config(a))


def pacman_cli():
    """pacman-run"""
    a = _parser("pacman").parse_args()
    board = PacmanBoard(frame_skip=a.frame_skip, frame_stack=a.frame_stack, reward_units=a.reward_units)
    main(a.agent, a.seed, a.steps, board, _config(a))


if __name__ == "__main__":
    cli()
