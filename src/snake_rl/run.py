"""Run one (agent, seed) and save per-episode scores.

usage: snake-run <agent> <seed> <steps> [n_distractor_channels] [grid_size] [--eps-floor F] [--eps-decay N]
                 [--map M] [--bonus R] [--food-curriculum R[:H]:N [--food-relocate]] [--eval-every K [--eval-episodes M] [--eval-eps E]]
                 [--probe-every K] [--train-log-every K] [--save-agent] [--save-every K]
       (or: python -m snake_rl.run ...)
agents: see snake_rl.agents.AGENTS
"""
import argparse
import json
import os
import pickle
import time

import numpy as np

from snake_rl.agents import AGENTS, make_agent
from snake_rl.env import MAPS, Snake
from snake_rl.probe import make_probe_set, probe


EPS_FLOOR = 0.02


def epsilon(t, floor=EPS_FLOOR, decay=None):
    """Exploration rate at step t.

    decay=None: the original schedule, kept bit-for-bit so old runs reproduce: 1 - t/5000 until it reaches
    the floor. decay=N: linear from 1 to the floor over exactly N steps, then the floor (DQN's schedule).
    decay=0: the floor from the start, i.e. a fixed epsilon (MFEC's schedule).
    """
    if decay is None:
        return max(floor, 1.0 - t / 5000)
    return floor if t >= decay else 1.0 + (floor - 1.0) * t / decay


def food_radius(t, curriculum, size):
    """Food-distance curriculum. (r0, n): the radius grows linearly from r0 to 2*size over n steps.
    (r0, hold, n): it stays at r0 until step hold, then grows to 2*size by step n. From step n on, food goes
    anywhere (None), as in the real game. curriculum=None: always anywhere."""
    if curriculum is None or t >= curriculum[-1]:
        return None
    r0, hold, n = curriculum if len(curriculum) == 3 else (curriculum[0], 0, curriculum[1])
    return r0 if t < hold else int(r0 + (2 * size - r0) * (t - hold) / (n - hold))


def run_name(name, eps_floor=EPS_FLOOR, eps_decay=None, food_curriculum=None, food_relocate=False):
    """Result file stem (minus seed): the agent name, tagged with any non-default run settings."""
    return (name + ("" if eps_floor == EPS_FLOOR else f"_eps{eps_floor:g}")
            + ("" if eps_decay is None else f"_epsd{eps_decay}")
            + ("" if food_curriculum is None else "_fc" + "-".join(str(x) for x in food_curriculum))
            + ("_reloc" if food_relocate else ""))


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


def save(agent, path, name, seed, t, size, map, bonus, D):
    """Pickle the agent (without its replay buffer) plus the env settings snake-watch needs to replay it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dict(agent=agent, name=name, seed=seed, t=t, size=size, map=map, bonus=bonus, distractors=D), f)


def results_dir(D, size, root="results", map="open", bonus=0.0):
    """results/d{D} (7x7) or g{size}_d{D}; a non-default map / bonus food adds _{map} / _b{bonus}."""
    base = f"{root}/d{D}" if size == 7 else f"{root}/g{size}_d{D}"
    return base + ("" if map == "open" else f"_{map}") + (f"_b{bonus:g}" if bonus else "")


def main(name, seed, steps, D=0, size=7, root="results", eps_floor=EPS_FLOOR, map="open", bonus=0.0,
         eps_decay=None, eval_every=0, eval_episodes=5, eval_eps=0.05, food_curriculum=None, food_relocate=False,
         probe_every=0, train_log_every=1000, save_agent=False, save_every=0):
    if food_relocate and food_curriculum is None:
        raise ValueError("food_relocate only acts during a food curriculum; set food_curriculum too")
    rng = np.random.default_rng(seed)
    env = Snake(size=size, seed=seed + 1000, distractors=D, map=map, bonus=bonus,
                food_radius=food_radius(0, food_curriculum, size), relocate_food=food_relocate)
    agent = make_agent(name, env, rng)
    # Evaluation always places food anywhere: it measures the real task, whatever the training curriculum.
    if eval_every:  # own env and RNG streams, so evaluation never shifts training's random numbers
        eval_env = Snake(size=size, seed=seed + 2000, distractors=D, map=map, bonus=bonus)
        eval_rng = np.random.default_rng(seed + 3000)
    # Representation probe (probe.py): a fixed hand-built state set, no rng shared with training.
    P = make_probe_set(size, map, bonus) if probe_every and not D else None
    probes = [(0, probe(agent, P))] if P is not None else []
    out_dir = results_dir(D, size, root, map, bonus)
    stem = f"{out_dir}/{run_name(name, eps_floor, eps_decay, food_curriculum, food_relocate)}_s{seed}"
    # Training telemetry (telemetry.py): window means every train_log_every steps. Also streamed, one JSON line per
    # snapshot, to <stem>.train.jsonl as the run goes (kept afterwards), so `snake-train` can plot a run in progress.
    tlog = hasattr(agent, "train_stats") and train_log_every > 0
    if tlog:
        os.makedirs(out_dir, exist_ok=True)
        live = open(f"{stem}.train.jsonl", "w")
    obs = env.reset()
    log, diag, evals, train, t0 = [], [], [], [], time.time()
    for t in range(1, steps + 1):
        if food_curriculum is not None:
            env.food_radius = food_radius(t, food_curriculum, size)
        a = agent.act(obs, epsilon(t, eps_floor, eps_decay), t)
        nobs, r, term, trunc = env.step(a)
        agent.observe(obs, a, r, nobs, term, trunc, t)
        obs = nobs
        if term or trunc:
            # (end step, score, regular foods eaten, 1 if cut off by the idle limit else 0). Readers that
            # predate the last two fields use only the first two, so old and new JSONs read the same way.
            log.append((t, env.score, env.foods, int(trunc)))
            obs = env.reset()
        if t % 6000 == 0 and hasattr(agent, "diagnostics"):
            diag.append((t, agent.diagnostics()))
        if eval_every and t % eval_every == 0:
            evals.append((t, evaluate(agent, eval_env, eval_rng, eval_episodes, eval_eps)))
        if P is not None and t % probe_every == 0:
            probes.append((t, probe(agent, P)))
        if save_every and t % save_every == 0:
            save(agent, f"{stem}.agent_t{t}.pkl", name, seed, t, size, map, bonus, D)
        if tlog and t % train_log_every == 0:
            s = agent.train_stats()
            if s:
                train.append((t, s))
                live.write(json.dumps({"t": t, **s}) + "\n")
                live.flush()
    out = {"agent": name, "seed": seed, "steps": steps, "distractors": D, "size": size,
           "eps_floor": eps_floor, "eps_decay": eps_decay, "food_curriculum": food_curriculum, "food_relocate": food_relocate, "map": map, "bonus": bonus, "episodes": log, "diagnostics": diag,
           "eval": {"every": eval_every, "episodes": eval_episodes, "eps": eval_eps} if eval_every else None,
           "evaluations": evals, "probes": probes, "train": train,
           "wallclock_s": time.time() - t0}
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{stem}.json", "w") as f:
        json.dump(out, f)
    if save_agent:
        save(agent, f"{stem}.agent.pkl", name, seed, steps, size, map, bonus, D)
    if tlog:
        live.close()
    print(run_name(name, eps_floor, eps_decay, food_curriculum, food_relocate), seed, f"{len(log)} eps, {time.time()-t0:.0f}s")


def cli():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("agent", help=f"one of {sorted(AGENTS)}")
    ap.add_argument("seed", type=int)
    ap.add_argument("steps", type=int)
    ap.add_argument("distractors", type=int, nargs="?", default=0, help="extra noise channels")
    ap.add_argument("size", type=int, nargs="?", default=7, help="grid size")
    ap.add_argument("--eps-floor", type=float, default=EPS_FLOOR, help="final epsilon after the decay")
    ap.add_argument("--eps-decay", type=int, default=None,
                    help="steps for epsilon to fall linearly from 1 to the floor (0 = fixed at the floor; "
                         "default: the original 5k-step schedule)")
    ap.add_argument("--map", default="open", choices=MAPS, help="obstacle layout")
    ap.add_argument("--bonus", type=float, default=0.0, help="reward of the timed bonus food (0 = off)")
    ap.add_argument("--food-curriculum", default=None, metavar="R[:H]:N",
                    help="place food within R walkable steps of the head (held until step H), growing to the "
                         "whole board by step N, then anywhere (default: anywhere from the start)")
    ap.add_argument("--food-relocate", action="store_true",
                    help="during the curriculum, re-place food not eaten within 2r+5 steps near the head")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="every K steps, play evaluation episodes on a separate env (0 = off)")
    ap.add_argument("--eval-episodes", type=int, default=5, help="episodes per evaluation")
    ap.add_argument("--eval-eps", type=float, default=0.05, help="epsilon during evaluation (DQN paper: 0.05)")
    ap.add_argument("--probe-every", type=int, default=0,
                    help="every K steps, measure the encoder and greedy steering on a fixed probe set (probe.py)")
    ap.add_argument("--train-log-every", type=int, default=1000,
                    help="every K steps, log training telemetry (loss, TD error, grad norm, ...; 0 = off)")
    ap.add_argument("--save-agent", action="store_true",
                    help="pickle the trained agent to <stem>.agent.pkl, for snake-watch")
    ap.add_argument("--save-every", type=int, default=0, help="also pickle a checkpoint every K steps")
    ap.add_argument("--results", default="results", help="results root directory")
    a = ap.parse_args()
    fc = tuple(int(x) for x in a.food_curriculum.split(":")) if a.food_curriculum else None
    main(a.agent, a.seed, a.steps, a.distractors, a.size, a.results, a.eps_floor, a.map, a.bonus, a.eps_decay,
         a.eval_every, a.eval_episodes, a.eval_eps, fc, a.food_relocate, a.probe_every, a.train_log_every, a.save_agent, a.save_every)


if __name__ == "__main__":
    cli()
