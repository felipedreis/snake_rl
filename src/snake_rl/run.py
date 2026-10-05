"""Run one (agent, seed) and save per-episode scores.

usage: snake-run <agent> <seed> <steps> [n_distractor_channels] [grid_size]
       (or: python -m snake_rl.run ...)
agents: see snake_rl.agents.AGENTS  (suffix _eps10: epsilon floor 0.1)
"""
import argparse
import json
import os
import time

import numpy as np

from snake_rl.agents import AGENTS, make_agent
from snake_rl.env import Snake


def epsilon(t, decay=5000, floor=0.02):
    return max(floor, 1.0 - t / decay)


def results_dir(D, size, root="results"):
    return f"{root}/d{D}" if size == 7 else f"{root}/g{size}_d{D}"


def main(name, seed, steps, D=0, size=7, root="results"):
    rng = np.random.default_rng(seed)
    env = Snake(size=size, seed=seed + 1000, distractors=D)
    agent = make_agent(name, env, rng)
    obs = env.reset()
    log, diag, t0 = [], [], time.time()
    for t in range(1, steps + 1):
        a = agent.act(obs, epsilon(t, floor=0.1 if name.endswith("_eps10") else 0.02), t)
        nobs, r, term, trunc = env.step(a)
        agent.observe(obs, a, r, nobs, term, trunc, t)
        obs = nobs
        if term or trunc:
            log.append((t, env.score))
            obs = env.reset()
        if t % 6000 == 0 and hasattr(agent, "diagnostics"):
            diag.append((t, agent.diagnostics()))
    out = {"agent": name, "seed": seed, "steps": steps, "distractors": D, "size": size, "episodes": log, "diagnostics": diag,
           "wallclock_s": time.time() - t0}
    out_dir = results_dir(D, size, root)
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{name}_s{seed}.json", "w") as f:
        json.dump(out, f)
    print(name, seed, f"{len(log)} eps, {time.time()-t0:.0f}s")


def cli():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("agent", help=f"one of {sorted(AGENTS)}, optionally with suffix _eps10")
    ap.add_argument("seed", type=int)
    ap.add_argument("steps", type=int)
    ap.add_argument("distractors", type=int, nargs="?", default=0, help="extra noise channels")
    ap.add_argument("size", type=int, nargs="?", default=7, help="grid size")
    ap.add_argument("--results", default="results", help="results root directory")
    a = ap.parse_args()
    main(a.agent, a.seed, a.steps, a.distractors, a.size, a.results)


if __name__ == "__main__":
    cli()
