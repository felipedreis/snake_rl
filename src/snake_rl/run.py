"""Run one (agent, seed) and save per-episode scores.

usage: snake-run <agent> <seed> <steps> [n_distractor_channels] [grid_size] [--eps-floor F] [--map M] [--bonus R]
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


def epsilon(t, decay=5000, floor=EPS_FLOOR):
    return max(floor, 1.0 - t / decay)


def run_name(name, eps_floor=EPS_FLOOR):
    """Result file stem (minus seed): the agent name, tagged with any non-default run settings."""
    return name if eps_floor == EPS_FLOOR else f"{name}_eps{eps_floor:g}"


def results_dir(D, size, root="results", map="open", bonus=0.0):
    """results/d{D} (7x7) or g{size}_d{D}; a non-default map / bonus food adds _{map} / _b{bonus}."""
    base = f"{root}/d{D}" if size == 7 else f"{root}/g{size}_d{D}"
    return base + ("" if map == "open" else f"_{map}") + (f"_b{bonus:g}" if bonus else "")


def main(name, seed, steps, D=0, size=7, root="results", eps_floor=EPS_FLOOR, map="open", bonus=0.0):
    rng = np.random.default_rng(seed)
    env = Snake(size=size, seed=seed + 1000, distractors=D, map=map, bonus=bonus)
    agent = make_agent(name, env, rng)
    obs = env.reset()
    log, diag, t0 = [], [], time.time()
    for t in range(1, steps + 1):
        a = agent.act(obs, epsilon(t, floor=eps_floor), t)
        nobs, r, term, trunc = env.step(a)
        agent.observe(obs, a, r, nobs, term, trunc, t)
        obs = nobs
        if term or trunc:
            log.append((t, env.score))
            obs = env.reset()
        if t % 6000 == 0 and hasattr(agent, "diagnostics"):
            diag.append((t, agent.diagnostics()))
    out = {"agent": name, "seed": seed, "steps": steps, "distractors": D, "size": size,
           "eps_floor": eps_floor, "map": map, "bonus": bonus, "episodes": log, "diagnostics": diag,
           "wallclock_s": time.time() - t0}
    out_dir = results_dir(D, size, root, map, bonus)
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{run_name(name, eps_floor)}_s{seed}.json", "w") as f:
        json.dump(out, f)
    print(run_name(name, eps_floor), seed, f"{len(log)} eps, {time.time()-t0:.0f}s")


def cli():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("agent", help=f"one of {sorted(AGENTS)}")
    ap.add_argument("seed", type=int)
    ap.add_argument("steps", type=int)
    ap.add_argument("distractors", type=int, nargs="?", default=0, help="extra noise channels")
    ap.add_argument("size", type=int, nargs="?", default=7, help="grid size")
    ap.add_argument("--eps-floor", type=float, default=EPS_FLOOR, help="final epsilon after the decay")
    ap.add_argument("--map", default="open", choices=MAPS, help="obstacle layout")
    ap.add_argument("--bonus", type=float, default=0.0, help="reward of the timed bonus food (0 = off)")
    ap.add_argument("--results", default="results", help="results root directory")
    a = ap.parse_args()
    main(a.agent, a.seed, a.steps, a.distractors, a.size, a.results, a.eps_floor, a.map, a.bonus)


if __name__ == "__main__":
    cli()
