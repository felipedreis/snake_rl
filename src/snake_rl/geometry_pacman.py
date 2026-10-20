"""Post-hoc encoder geometry of saved Ms. Pac-Man agents (exploratory; the Pac-Man twin of geometry.py).

usage: python -m snake_rl.geometry_pacman CHECKPOINT... [--out DIR] [--episodes 100] [--eps 0.05] [--max-states 3000] [--seed 777]

For each checkpoint (`snake-run --game pacman --save-agent`) it writes <out>/<stem>.geometry.json with, for the agent's embedding
(DQN: last hidden layer; NEC / ec_frozen: the keys) AND, as a reference, for the raw pixel stack as if it were the embedding:
  onpolicy       over states the agent visits itself (eps 0.05, its own env/rng): for each label the share of a state's 10 nearest
                 states, taken from OTHER episodes (consecutive frames are near-duplicates), that carry the same label, next to the
                 share expected by chance. Labels: pac (Pac-Man's cell), pellet (cell of the nearest pellet by walking distance, ties
                 to the first in reading order), ghost (offset to the nearest ghost, clipped to +-4), scared (power-pellet timer on or off),
                 danger (a non-scared ghost within 3 walking steps).
  action_purity  for each action, the share of the 50 nearest states (other episodes) that agree with a state on whether the action is
                 good (it does not end next to a non-scared ghost and it brings Pac-Man closer to a pellet), next to chance.
  steering       share of states where the greedy action is good, among states with at least one good action, next to the share of
                 good actions (what a uniformly random action achieves).
  memory         NEC / ec_frozen only: as in geometry.py (coherence, probe_vs_memory, kernel_peak) on the visited states.
Reads only the checkpoint; the rollouts use their own random streams.
"""
import argparse, json, os, pickle, time

import numpy as np

from snake_rl.boards import board_from_settings
from snake_rl.geometry import _embed, memory
from snake_rl.pacman import DIRS, MsPacman
from snake_rl.probe import _sqdist

GHOST_NEAR = 3


def load(path):
    with open(path, "rb") as f:
        ck = pickle.load(f)
    return ck, board_from_settings(ck, path)


def _targets(env):
    return [tuple(p) for p in np.argwhere(env.pellets | env.power)]


def labels_and_good(env):
    """(labels dict, good-action mask over the 4 actions) for the env's current state."""
    here, dist = env.pac, env.dist
    cells = _targets(env)
    near = min(cells, key=lambda c: (dist[here][c], c))
    live = [g for g in env.ghosts] if env.scared == 0 else []
    ghost = min(env.ghosts, key=lambda g: dist[here][g])
    clip = lambda v: int(np.clip(v, -4, 4)) + 4
    lab = dict(pac=here[0] * env.n + here[1], pellet=near[0] * env.n + near[1],
               ghost=clip(ghost[0] - here[0]) * 9 + clip(ghost[1] - here[1]), scared=int(env.scared > 0),
               danger=int(any(dist[here][g] <= GHOST_NEAR for g in live)))
    d0 = min(dist[here][c] for c in cells)
    good = []
    for dr, dc in DIRS:
        q = (here[0] + dr, here[1] + dc)
        if env.walls[q]:
            good.append(False)
            continue
        safe = all(dist[q][g] > 1 for g in live)
        closer = q in cells or min(dist[q][c] for c in cells) < d0
        good.append(bool(safe and closer))
    return lab, np.array(good)


def rollout(agent, board, episodes, eps, seed, max_states):
    env, rng = board.make_env(seed), np.random.default_rng(seed + 1)
    saved, agent.rng = agent.rng, rng
    X, ep_id, labs, good = [], [], [], []
    try:
        for ep in range(episodes):
            obs, done = env.reset(), False
            while not done and len(X) < max_states:
                lab, g = labels_and_good(env)
                X.append(np.rint(obs * 255).astype(np.uint8))
                ep_id.append(ep)
                labs.append(lab)
                good.append(g)
                obs, _, term, trunc = env.step(agent.act(obs, eps, None))
                done = term or trunc
            if len(X) >= max_states:
                break
    finally:
        agent.rng = saved
    lab = {k: np.array([l[k] for l in labs]) for k in labs[0]}
    return np.array(X), np.array(ep_id), lab, np.array(good)


def _neighbours(Z, ep_id, k):
    D = _sqdist(Z)
    other = ep_id[:, None] != ep_id[None, :]
    D[~other] = np.inf
    return np.argsort(D, 1)[:, :k], other


def onpolicy(nb, other, lab):
    out = {}
    for name, v in lab.items():
        same = v[:, None] == v[None, :]
        chance = ((same & other).sum(1) / np.maximum(other.sum(1), 1)).mean()
        out[name] = {"purity": float((v[nb] == v[:, None]).mean()), "chance": float(chance)}
    return out


def action_purity(nb, good):
    g = good.astype(float)
    agree = (g[nb] == g[:, None, :]).mean(1)
    pa = g.mean(0)
    return {"purity": float(agree.mean()), "chance": float(np.mean(pa ** 2 + (1 - pa) ** 2)), "k": int(nb.shape[1])}


def steering(Q, good):
    has = good.any(1)
    greedy = Q.argmax(1)
    return {"accuracy": float(good[has][np.arange(has.sum()), greedy[has]].mean()),
            "random": float(good[has].mean()), "n_states": int(has.sum())}


def analyse(path, episodes=100, eps=0.05, max_states=3000, seed=777, k=10, k_action=50):
    t0 = time.time()
    ck, board = load(path)
    agent = ck["agent"]
    X, ep_id, lab, good = rollout(agent, board, episodes, eps, seed, max_states)
    Xf = X.astype(np.float32) / np.float32(255)
    Z, Q = _embed(agent, X)
    res = dict(checkpoint=os.path.basename(path), agent=ck["name"], seed=ck["seed"], t=ck["t"], opt=ck.get("opt"), lr=ck.get("lr"),
               n_states=int(len(X)), n_episodes=int(len(set(ep_id.tolist()))), good_share=float(good.mean()),
               steering=steering(Q, good))
    for tag, emb in (("embedding", Z), ("pixels", Xf)):
        nb, other = _neighbours(emb, ep_id, max(k, k_action))
        res[tag] = dict(onpolicy=onpolicy(nb[:, :k], other, lab), action_purity=action_purity(nb[:, :k_action], good))
    if hasattr(agent, "dnds"):
        res["memory"] = memory(agent, Z[:500], np.random.default_rng(seed + 2))
    res["seconds"] = round(time.time() - t0, 1)
    return res


def cli():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--out", default="results/geometry_pacman")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--max-states", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=777)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for path in a.checkpoints:
        out = os.path.join(a.out, os.path.basename(path).replace(".pkl", ".geometry.json"))
        if os.path.exists(out):
            continue
        try:
            res = analyse(path, a.episodes, a.eps, a.max_states, a.seed)
        except Exception as e:
            print("FAILED", path, repr(e))
            continue
        with open(out, "w") as f:
            json.dump(res, f)
        print(res["agent"], res["seed"], res["t"], f"{res['seconds']}s", f"states {res['n_states']}")


if __name__ == "__main__":
    cli()
