"""Run one (agent, seed) and save per-episode scores.

usage: snake-run <agent> <seed> <steps> [n_distractor_channels] [grid_size] [--eps-floor F] [--eps-decay N]
                 [--map M] [--bonus R] [--food-curriculum R:N] [--eval-every K [--eval-episodes M] [--eval-eps E]]
       (or: python -m snake_rl.run ...)
agents: see snake_rl.agents.AGENTS
"""
import argparse
import json
import os
import time

import numpy as np

from snake_rl.agents import AGENTS, make_agent
from snake_rl.env import MAPS, Snake


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
    """Food-distance curriculum: (r0, n) grows the food radius linearly from r0 to 2*size over n steps,
    then places food anywhere (None), as in the real game. curriculum=None: always anywhere."""
    if curriculum is None or t >= curriculum[1]:
        return None
    r0, n = curriculum
    return int(r0 + (2 * size - r0) * t / n)


def run_name(name, eps_floor=EPS_FLOOR, eps_decay=None, food_curriculum=None):
    """Result file stem (minus seed): the agent name, tagged with any non-default run settings."""
    return (name + ("" if eps_floor == EPS_FLOOR else f"_eps{eps_floor:g}")
            + ("" if eps_decay is None else f"_epsd{eps_decay}")
            + ("" if food_curriculum is None else f"_fc{food_curriculum[0]}-{food_curriculum[1]}"))


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


def results_dir(D, size, root="results", map="open", bonus=0.0):
    """results/d{D} (7x7) or g{size}_d{D}; a non-default map / bonus food adds _{map} / _b{bonus}."""
    base = f"{root}/d{D}" if size == 7 else f"{root}/g{size}_d{D}"
    return base + ("" if map == "open" else f"_{map}") + (f"_b{bonus:g}" if bonus else "")


def main(name, seed, steps, D=0, size=7, root="results", eps_floor=EPS_FLOOR, map="open", bonus=0.0,
         eps_decay=None, eval_every=0, eval_episodes=5, eval_eps=0.05, food_curriculum=None):
    rng = np.random.default_rng(seed)
    env = Snake(size=size, seed=seed + 1000, distractors=D, map=map, bonus=bonus,
                food_radius=food_radius(0, food_curriculum, size))
    agent = make_agent(name, env, rng)
    # Evaluation always places food anywhere: it measures the real task, whatever the training curriculum.
    if eval_every:  # own env and RNG streams, so evaluation never shifts training's random numbers
        eval_env = Snake(size=size, seed=seed + 2000, distractors=D, map=map, bonus=bonus)
        eval_rng = np.random.default_rng(seed + 3000)
    obs = env.reset()
    log, diag, evals, t0 = [], [], [], time.time()
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
    out = {"agent": name, "seed": seed, "steps": steps, "distractors": D, "size": size,
           "eps_floor": eps_floor, "eps_decay": eps_decay, "food_curriculum": food_curriculum, "map": map, "bonus": bonus, "episodes": log, "diagnostics": diag,
           "eval": {"every": eval_every, "episodes": eval_episodes, "eps": eval_eps} if eval_every else None,
           "evaluations": evals,
           "wallclock_s": time.time() - t0}
    out_dir = results_dir(D, size, root, map, bonus)
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{run_name(name, eps_floor, eps_decay, food_curriculum)}_s{seed}.json", "w") as f:
        json.dump(out, f)
    print(run_name(name, eps_floor, eps_decay, food_curriculum), seed, f"{len(log)} eps, {time.time()-t0:.0f}s")


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
    ap.add_argument("--food-curriculum", default=None, metavar="R:N",
                    help="place food within R walkable steps of the head, growing to the whole board over N "
                         "steps, then anywhere (default: anywhere from the start)")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="every K steps, play evaluation episodes on a separate env (0 = off)")
    ap.add_argument("--eval-episodes", type=int, default=5, help="episodes per evaluation")
    ap.add_argument("--eval-eps", type=float, default=0.05, help="epsilon during evaluation (DQN paper: 0.05)")
    ap.add_argument("--results", default="results", help="results root directory")
    a = ap.parse_args()
    fc = tuple(int(x) for x in a.food_curriculum.split(":")) if a.food_curriculum else None
    main(a.agent, a.seed, a.steps, a.distractors, a.size, a.results, a.eps_floor, a.map, a.bonus, a.eps_decay,
         a.eval_every, a.eval_episodes, a.eval_eps, fc)


if __name__ == "__main__":
    cli()
