"""Does the probe agree with how the agent actually steers? Greedy steering on the agent's own states vs on the probe set.

usage (from the repo root): .venv/bin/python docs/experiments/pixels_rmsprop_probe/steering_check.py CKPT.agent.pkl [...]
For each checkpoint: plays 4,000 greedy steps on the board it was trained on (food radius 2 with relocation for the held
phase, as in training; otherwise food anywhere), and reports P(greedy action is on a shortest path to the food) on those
states, split by the head's Manhattan distance from the start cell, next to the same number on the probe set and the
chance level. A probe that says "chance" for an agent that steers on its own states is not measuring what we want.
"""
import pickle
import sys

import numpy as np

from snake_rl.env import Snake
from snake_rl.probe import _dist_from, make_probe_set

BINS = [(0, 2), (3, 5), (6, 9), (10, 99)]


def good_actions(env):
    head, d, body = env.body[0], env.dir, list(env.body)
    blocked = env.wall_set | set(body[:-1])
    to_food = _dist_from(env, env.food, blocked - {head})
    g = []
    for a in range(3):
        nd = (d + (0, 1, -1)[a]) % 4
        nh = (head[0] + Snake.DIRS[nd][0], head[1] + Snake.DIRS[nd][1])
        safe = 0 <= nh[0] < env.n and 0 <= nh[1] < env.n and nh not in blocked
        g.append(safe and to_food.get(nh, np.inf) < to_food.get(head, np.inf))
    return np.array(g)


def split(dist, hit):
    dist, hit = np.array(dist), np.array(hit, float)
    return "  ".join(f"{lo}-{hi}: {hit[(dist >= lo) & (dist <= hi)].mean():.2f} (n={((dist >= lo) & (dist <= hi)).sum()})"
                     for lo, hi in BINS if ((dist >= lo) & (dist <= hi)).any())


def check(path, steps=4000):
    ck = pickle.load(open(path, "rb"))
    ag, n = ck["agent"], ck["size"]
    ag.rng = np.random.default_rng(7)
    fc = ck.get("food_curriculum")
    held = fc is not None and ck["t"] < (fc[1] if len(fc) == 3 else 0)  # still in the held phase when saved
    env = Snake(size=n, seed=77, map=ck["map"], bonus=ck["bonus"], render=ck.get("render", "grid"),
                food_radius=fc[0] if held else None, relocate_food=bool(held and ck["food_relocate"]))
    c = n // 2
    obs, dist, hit, chance, fruit, eps_n = env.reset(), [], [], [], 0, 0
    for _ in range(steps):
        if env.food is not None and env.bonus_pos is None:
            g = good_actions(env)
            _, Q = ag.probe_embed(obs[None])
            best = np.isclose(Q[0], Q[0].max())
            hit.append((best & g).sum() / best.sum())
            chance.append(g.mean())
            dist.append(abs(env.body[0][0] - c) + abs(env.body[0][1] - c))
        obs, r, term, trunc = env.step(ag.act(obs, 0.0, None))
        fruit += r == 1.0
        if term or trunc:
            obs, eps_n = env.reset(), eps_n + 1
    print(f"\n{path.split('results/')[-1]}  (step {ck['t']:,}{', held phase' if held else ''})")
    print(f"  greedy play: {100 * fruit / steps:.1f} fruit/100 steps, {eps_n} episodes")
    print(f"  own states : steer {np.mean(hit):.2f} (chance {np.mean(chance):.2f});  by head distance from start: {split(dist, hit)}")
    P = make_probe_set(n, ck["map"], ck["bonus"], n_configs=300, render=ck.get("render", "grid"))
    _, Q = ag.probe_embed(P["X"])
    best = np.isclose(Q, Q.max(1, keepdims=True))
    st = (best & P["good"]).sum(1) / best.sum(1)
    print(f"  probe set  : steer {st.mean():.2f} (chance {P['good'].mean():.2f}); all-tied Q on {best.all(1).mean():.0%} of probe states")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        check(p)
