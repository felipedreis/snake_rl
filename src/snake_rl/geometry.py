"""Post-hoc geometry analysis of saved agents (exploratory: nothing here is part of a pre-registered test).

usage: python -m snake_rl.geometry CHECKPOINT... [--out DIR] [--episodes 20] [--eps 0.05] [--max-states 3000] [--seed 777]

For each checkpoint (a `.agent*.pkl` written by `snake-run --save-agent/--save-every`) it writes <out>/<stem>.geometry.json with
  probe     the in-run near probe (probe.py) recomputed on the saved agent; it must equal the run's own probe entry at that step.
  onpolicy  neighbour structure of the embeddings of states the agent visits itself: for each label (head cell, food cell, food
            offset in the board frame, food offset in the snake's frame) the share of an embedding's 10 nearest states, taken from
            OTHER episodes, that carry the same label, next to the share expected by chance (the share of other-episode states with that label, averaged over states).
  action_purity  among the probe states: for each state and action, the share of its 50 nearest probe states (in embedding space) that agree with it
            on whether that action is a good move (safe and shortening the path to the food), averaged, next to the share expected by chance.
            If the nearest states disagree about which move is good, an average over them cannot prefer a move.
  qfood     how much the Q-values on the probe set depend on where the food is: the s.d. of Q over states that share a snake
            configuration (only the food moves), over the s.d. of Q over all states, averaged over actions. Near 0 = Q ignores the food.
  memory    agents with a memory (NEC, ec_frozen) only: for each action, the s.d. of the 50 neighbours' values over the s.d. of all
            stored values ("coherence", near 1 = neighbours are no more alike in value than random rows), the distance from probe
            states to their nearest stored key over the same for stored keys to their own nearest neighbour ("probe_vs_memory",
            near 1 = the probe states look like states in memory), and the mean largest kernel weight.
Reads only the checkpoint; draws no random number from the agent's own stream (rollouts use their own).
"""
import argparse, json, os, pickle, time

import numpy as np

from snake_rl.boards import board_from_settings
from snake_rl.probe import _as_float, _ego, _sqdist, make_probe_set, probe


def load(path):
    with open(path, "rb") as f:
        ck = pickle.load(f)
    return ck, board_from_settings(ck, path)


def rollout(agent, board, episodes, eps, seed, max_states):
    """States the agent visits (observation before each action) with the labels needed below; own env and rng streams."""
    env, rng = board.make_env(seed), np.random.default_rng(seed + 1)  # food anywhere, as in evaluation
    saved, agent.rng = agent.rng, rng
    X, meta = [], []
    try:
        for ep in range(episodes):
            obs, done = env.reset(), False
            while not done and len(X) < max_states:
                if env.food is not None:
                    X.append(np.rint(obs * 255).astype(np.uint8) if env.pixels else obs.copy())
                    hr, hc = env.body[0]
                    meta.append((ep, hr, hc, env.food[0], env.food[1], env.dir))
                obs, _, term, trunc = env.step(agent.act(obs, eps, None))
                done = term or trunc
            if len(X) >= max_states:
                break
    finally:
        agent.rng = saved
    return np.array(X), np.array(meta)


def _embed(agent, X, chunk=512):
    Z, Q = zip(*[agent.probe_embed(_as_float(X[lo:lo + chunk])) for lo in range(0, len(X), chunk)])
    return np.concatenate(Z), np.concatenate(Q)


def _labels(meta, n):
    ep, hr, hc, fr, fc, d = meta.T
    clip = lambda v: np.clip(v, -4, 4) + 4
    ego = np.array([_ego(int(a), int(b), int(h)) for a, b, h in zip(fr - hr, fc - hc, d)])
    return {"head": hr * n + hc, "food": fr * n + fc, "offset_board": clip(fr - hr) * 9 + clip(fc - hc),
            "offset_ego": clip(ego[:, 0]) * 9 + clip(ego[:, 1])}


def onpolicy(Z, meta, n, k=10):
    D = _sqdist(Z)
    other = meta[:, 0][:, None] != meta[:, 0][None, :]  # states of other episodes: never compare a state with its own episode
    D[~other] = np.inf
    nb = np.argsort(D, 1)[:, :k]
    out = {"n_states": int(len(Z)), "n_episodes": int(len(set(meta[:, 0].tolist())))}
    for name, lab in _labels(meta, n).items():
        same = lab[:, None] == lab[None, :]
        chance = ((same & other).sum(1) / np.maximum(other.sum(1), 1)).mean()  # same-label share among the other-episode states
        out[name] = {"purity": float((lab[nb] == lab[:, None]).mean()), "chance": float(chance)}
    return out


def action_purity(Z, P, k=50):
    """Agreement of the k nearest probe states on which actions are good; `chance` is the agreement of random pairs of probe states."""
    good = P["good"].astype(float)
    D = _sqdist(Z)
    np.fill_diagonal(D, np.inf)
    nb = np.argsort(D, 1)[:, :k]
    agree = (good[nb] == good[:, None, :]).mean(1)  # (states, actions): share of the k neighbours with the same label
    pa = good.mean(0)
    return {"purity": float(agree.mean()), "chance": float(np.mean(pa ** 2 + (1 - pa) ** 2)), "k": k}


def qfood(Q, P):
    """Within-configuration s.d. of Q (only the food moves) over the overall s.d., per action, averaged."""
    g, ratios = P["group"], []
    for a in range(Q.shape[1]):
        within = np.mean([Q[g == c, a].std() for c in np.unique(g) if (g == c).sum() > 1])
        ratios.append(float(within / (Q[:, a].std() + 1e-12)))
    return {"ratio": float(np.mean(ratios)), "per_action": ratios, "q_spread": float((Q.max(1) - Q.min(1)).mean())}


def memory(agent, Zp, rng, sample=2000):
    out = {"coherence": [], "probe_vs_memory": [], "kernel_peak": [], "rows": []}
    for dnd in agent.dnds:
        out["rows"].append(int(dnd.n))
        if dnd.n < 2:
            for k in ("coherence", "probe_vs_memory", "kernel_peak"):
                out[k].append(None)
            continue
        idx, d = dnd.knn(Zp)
        V = dnd.vals[idx]
        out["coherence"].append(float(V.std(1).mean() / (dnd.vals[:dnd.n].std() + 1e-12)))
        kern = 1.0 / (d + dnd.delta)
        out["kernel_peak"].append(float((kern / kern.sum(1, keepdims=True)).max(1).mean()))
        out["probe_vs_memory"].append(float(np.sqrt(np.sort(d, 1)[:, 0]).mean() / (_mem_nn(dnd, rng, sample) + 1e-12)))
    return out


def _mem_nn(dnd, rng, sample):
    """Mean distance from stored keys to their own nearest other stored key (a sample of keys, against all)."""
    rows = rng.choice(dnd.n, size=min(sample, dnd.n), replace=False)
    D = np.maximum(_sqdist_between(dnd.keys[rows], dnd.keys[:dnd.n]), 0.0)
    D[np.arange(len(rows)), rows] = np.inf
    return float(np.sqrt(D.min(1)).mean())


def _sqdist_between(A, B):
    return (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T


def analyse(path, episodes=20, eps=0.05, max_states=3000, seed=777):
    t0 = time.time()
    ck, board = load(path)
    agent, n = ck["agent"], board.size
    where = dict(size=n, map=board.map, bonus=board.bonus, render=board.render, frame_stack=board.frame_stack)
    P = make_probe_set(**where)
    res = dict(checkpoint=os.path.basename(path), agent=ck["name"], seed=ck["seed"], t=ck["t"], opt=ck.get("opt"), lr=ck.get("lr"),
               board=where, probe=probe(agent, P))
    Zp, Qp = agent.probe_embed(P["X"])
    res["qfood"] = qfood(Qp, P)
    res["action_purity"] = action_purity(Zp, P)
    X, meta = rollout(agent, board, episodes, eps, seed, max_states)
    Z, _ = _embed(agent, X)
    res["onpolicy"] = onpolicy(Z, meta, n)
    if hasattr(agent, "dnds"):
        res["memory"] = memory(agent, Zp, np.random.default_rng(seed + 2))
    res["seconds"] = round(time.time() - t0, 1)
    return res


def cli():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--out", default="results/geometry")
    ap.add_argument("--episodes", type=int, default=20)
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
        except Exception as e:  # a truncated download or an old checkpoint: report it, keep going
            print("FAILED", path, repr(e)); continue
        with open(out, "w") as f:
            json.dump(res, f)
        print(res["agent"], res["seed"], res["t"], f"{res['seconds']}s")


if __name__ == "__main__":
    cli()
