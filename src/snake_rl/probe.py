"""Representation probe: can an agent's encoder tell where the food is, and does its greedy policy steer to it?

A fixed probe set of hand-built states, each a short straight snake somewhere on the board with food at one of
the free cells within `radius` walkable steps of the head. States come in groups: one group per snake
configuration (head cell, heading), one state per nearby food cell. So within a group only the food moves, and
across groups the same egocentric food offset ("2 ahead, 1 left") appears at many board positions.

Metrics, on the agent's embedding Z (NEC: the DND key; DQN: the last hidden layer) and its Q-values:
  steer       mean P(greedy action moves along a shortest path to the food), ties split evenly.
  steer_chance  the same for a uniformly random action.
  knn_offset  fraction of each state's k nearest probe states (Euclidean in Z, the distance NEC's DND uses)
              that have the same egocentric food offset. High = situations are grouped by where the food is.
  knn_config  fraction of those neighbours from the same snake configuration (only the food differs).
              High = situations are grouped by where the snake is.
  food_ratio  mean distance between states differing only in food cell / mean distance between states with
              the same egocentric offset but a different snake configuration. << 1 = food barely moves Z.
  decode      cross-validated accuracy of a linear (ridge) read-out of the egocentric offset from Z, with
              folds split by snake configuration so no configuration is in both train and test.
The probe never draws from the agent's rng and never calls observe/act, so training is unchanged by it.

A second, optional set (`make_far_probe_set`, `run.py --probe-far`) reaches food up to 6 walkable steps away and adds, under the
key "far" of the probe result:
  steer_by_dist / chance_by_dist / n_by_dist   steering, its chance rate and the state count for food at walkable distance 1..6.
  all_tied      share of states where every action has the same Q (np.isclose): the agent has no information there.
  top_tied      share of states where two or more actions share the maximum Q (all_tied is a subset).
  all_tied_by_dist  the same share per distance.
Chance-level steering with all_tied near 1 means missing information, not wrong information.
"""
from collections import deque

import numpy as np

from snake_rl.env import Snake


def _dist_from(env, src, blocked):
    """Walkable steps from src to every reachable cell, avoiding `blocked`."""
    dist, todo = {src: 0}, deque([src])
    while todo:
        r, c = todo.popleft()
        for dr, dc in Snake.DIRS:
            q = (r + dr, c + dc)
            if 0 <= q[0] < env.n and 0 <= q[1] < env.n and q not in blocked and q not in dist:
                dist[q] = dist[(r, c)] + 1
                todo.append(q)
    return dist


def _ego(dr, dc, heading):
    """Rotate a board offset into the snake's frame (heading -> up): returns (ahead, right)."""
    for _ in range(heading):
        dr, dc = -dc, dr  # rotate 90 degrees counter-clockwise
    return -dr, dc


def _probe_obs(env, body, d):
    """Observation of a probe state. On the pixel render the frame stack shows the snake having moved straight in: the
    older frames have it shifted back along its heading as far as the board allows, as a real mid-episode stack does.
    Four identical frames would be a snake that never moves, which the agent never sees in training. Applies to any frame stack."""
    if not env.pixels:
        return env.static_obs()
    dr, dc = Snake.DIRS[d]
    free = lambda k: all(0 <= r - k * dr < env.n and 0 <= c - k * dc < env.n and not env.walls[r - k * dr, c - k * dc]
                         and (r - k * dr, c - k * dc) != env.food for r, c in body)
    kmax = 0  # how many cells back the snake can have come from
    while kmax < env.frame_stack - 1 and free(kmax + 1):
        kmax += 1
    frames, now = [], env.body
    for k in range(env.frame_stack - 1, -1, -1):  # oldest first; a short run-up repeats its first frame
        k = min(k, kmax)
        env.body = deque((r - k * dr, c - k * dc) for r, c in body)
        frames.append(env._frame())
    env.body = now
    return env._scale(frames)


def make_probe_set(size=25, map="rooms", bonus=0.0, n_configs=150, radius=2, length=3, seed=0, render="grid",
                   frame_stack=None, actions="relative", compact=False):
    """Returns dict(X obs rows, group config id, offset class id, good (rows, n_actions) bool, offsets list, dist).

    dist: walkable steps from the head to the food (body and walls block), one per row. compact=True stores pixel-render
    observations as uint8 gray levels (exact: the render only produces k/255), for large sets."""
    env = Snake(size=size, map=map, bonus=bonus, render=render, frame_stack=frame_stack, actions=actions)
    env.reset()
    rng = np.random.default_rng(seed)
    # Every (head cell, heading) whose straight body fits, in a fixed shuffled order; take the first n_configs
    # that have food within reach (all of them on a small board).
    cand = [(tuple(p), d) for p in np.argwhere(~env.walls) for d in range(4)]
    X, group, off, good, dist, configs = [], [], [], [], [], set()
    for i in rng.permutation(len(cand)):
        if len(configs) == n_configs:
            break
        head, d = cand[i]
        dr, dc = Snake.DIRS[d]
        body = [(head[0] - k * dr, head[1] - k * dc) for k in range(length)]
        if any(not (0 <= r < size and 0 <= c < size) or env.walls[r, c] for r, c in body):
            continue
        env.body, env.dir = deque(body), d
        env.bonus_pos, env.bonus_left = None, 0
        steps = env._steps_from_head()
        near = [p for p, k in steps.items() if 0 < k <= radius]
        if not near:
            continue
        configs.add((head, d))
        for food in sorted(near):
            env.food = food
            blocked = env.wall_set | set(body[:-1])  # the tail vacates when the head moves
            to_food = _dist_from(env, food, blocked - {head})
            g = []
            for a in range(env.n_actions):
                nd = env.heading_after(d, a)
                nh = (head[0] + Snake.DIRS[nd][0], head[1] + Snake.DIRS[nd][1])
                safe = 0 <= nh[0] < size and 0 <= nh[1] < size and nh not in blocked
                g.append(safe and to_food.get(nh, np.inf) < to_food.get(head, np.inf))
            X.append(_probe_obs(env, body, d))
            group.append(len(configs) - 1)
            off.append(_ego(food[0] - head[0], food[1] - head[1], d))
            good.append(g)
            dist.append(steps[food])
    offsets = sorted(set(off))
    X = np.array(X)
    if compact and env.pixels:
        X = np.rint(X * 255).astype(np.uint8)
    return dict(X=X, group=np.array(group), off=np.array([offsets.index(o) for o in off]),
                good=np.array(good), offsets=offsets, dist=np.array(dist))


def make_far_probe_set(size=10, map="open", bonus=0.0, n_configs=60, radius=6, **kw):
    """The larger probe set for steering by distance and tie fractions (see the module docstring). Pixel observations are
    stored as uint8, so a few thousand states stay small; `far_metrics` converts them in chunks."""
    return make_probe_set(size, map, bonus, n_configs=n_configs, radius=radius, compact=True, **kw)


def _sqdist(Z):
    s = (Z ** 2).sum(1)
    return np.maximum(s[:, None] + s[None] - 2 * Z @ Z.T, 0.0)


def representation_metrics(Z, P, k=10, folds=5, lam=1e-2):
    """Encoder-only metrics (no Q needed), so they also apply to raw observations."""
    group, off = P["group"], P["off"]
    D = _sqdist(Z)
    np.fill_diagonal(D, np.inf)
    nb = np.argsort(D, 1)[:, :k]
    same_cfg = group[:, None] == group[None]
    same_off = off[:, None] == off[None]
    np.fill_diagonal(same_cfg, False)
    E = np.sqrt(np.where(np.isinf(D), 0.0, D))
    within = E[same_cfg].mean()
    across = E[same_off & ~same_cfg & (group[:, None] != group[None])].mean()
    # Grouped k-fold ridge read-out of the offset class.
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    Zs = np.hstack([Zs, np.ones((len(Z), 1))])
    Y = np.eye(off.max() + 1)[off]
    fold = group % folds
    hits = 0
    for f in range(folds):
        tr, te = fold != f, fold == f
        W = np.linalg.solve(Zs[tr].T @ Zs[tr] + lam * tr.sum() * np.eye(Zs.shape[1]), Zs[tr].T @ Y[tr])
        hits += ((Zs[te] @ W).argmax(1) == off[te]).sum()
    return dict(knn_offset=float((off[nb] == off[:, None]).mean()),
                knn_config=float((group[nb] == group[:, None]).mean()),
                food_ratio=float(within / (across + 1e-12)), decode=float(hits / len(Z)))


def steering(Q, P):
    """Expected P(greedy action is a good one) with ties split evenly, and the random-action rate."""
    best = np.isclose(Q, Q.max(1, keepdims=True))
    steer = (best & P["good"]).sum(1) / best.sum(1)
    return dict(steer=float(steer.mean()), steer_chance=float(P["good"].mean()))


def tie_stats(Q):
    """Per row: are all actions tied, and do two or more share the maximum? (np.isclose, as `steering` does.)"""
    top = np.isclose(Q, Q.max(1, keepdims=True))
    return np.isclose(Q, Q[:, :1]).all(1), top.sum(1) > 1


def far_metrics(agent, F, max_dist=6, chunk=512):
    """Steering by walkable distance and tie fractions on the far probe set F. Read-only, like `probe`."""
    Q = np.concatenate([agent.probe_embed(_as_float(F["X"][lo:lo + chunk]))[1] for lo in range(0, len(F["X"]), chunk)])
    best = np.isclose(Q, Q.max(1, keepdims=True))
    steer = (best & F["good"]).sum(1) / best.sum(1)
    all_tied, top_tied = tie_stats(Q)
    dist, out = F["dist"], dict(steer_by_dist=[], chance_by_dist=[], n_by_dist=[], all_tied_by_dist=[])
    for d in range(1, max_dist + 1):
        m = dist == d
        out["n_by_dist"].append(int(m.sum()))
        out["steer_by_dist"].append(float(steer[m].mean()) if m.any() else None)
        out["chance_by_dist"].append(float(F["good"][m].mean()) if m.any() else None)
        out["all_tied_by_dist"].append(float(all_tied[m].mean()) if m.any() else None)
    return {**out, "all_tied": float(all_tied.mean()), "top_tied": float(top_tied.mean())}


def _as_float(X):
    return X.astype(np.float32) / np.float32(255) if X.dtype == np.uint8 else X


def probe(agent, P, F=None):
    """All metrics for an agent with a `probe_embed(X) -> (Z, Q)` method; {} for agents without one. With a far probe
    set F, the result also carries far_metrics under "far"."""
    if not hasattr(agent, "probe_embed"):
        return {}
    Z, Q = agent.probe_embed(P["X"])
    out = {**representation_metrics(Z, P), **steering(Q, P)}
    if F is not None:
        out["far"] = far_metrics(agent, F)
    return out
