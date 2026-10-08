"""Where does each agent steer? Greedy steering split by head distance from the start cell (12,12)."""
import copy, pickle, sys
import numpy as np
from snake_rl.env import Snake
from snake_rl.probe import make_probe_set, _dist_from

S = sys.argv[1]
C = 12
BINS = [(0, 2), (3, 5), (6, 9), (10, 24)]


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
    return "  ".join(f"d{lo}-{hi}: {hit[(dist>=lo)&(dist<=hi)].mean():.2f} (n={((dist>=lo)&(dist<=hi)).sum()})"
                     for lo, hi in BINS)


for name in sys.argv[2:]:
    snap = pickle.load(open(f"{S}/{name}501.pkl", "rb"))
    ag = snap["agent"]
    ag.rng = np.random.default_rng(7)
    # 1) on-policy: greedy (eps 0) play in the held-phase env (radius 2, relocation), fresh env stream
    env = Snake(size=25, seed=77, map="rooms", bonus=5, food_radius=2, relocate_food=True)
    obs, dist, hit, chance, steps, fruit, eps_n = env.reset(), [], [], [], 0, 0, 0
    while steps < 4000:
        if env.food is not None and env.bonus_pos is None:
            g = good_actions(env)
            Z, Q = ag.probe_embed(obs[None])
            best = np.isclose(Q[0], Q[0].max())
            hit.append((best & g).sum() / best.sum()); chance.append(g.mean())
            dist.append(abs(env.body[0][0] - C) + abs(env.body[0][1] - C))
        obs, r, term, trunc = env.step(ag.act(obs, 0.0, None))
        steps += 1; fruit += r == 1.0
        if term or trunc:
            obs = env.reset(); eps_n += 1
    print(f"\n{name}: greedy play 4000 steps: {eps_n} episodes, {100*fruit/steps:.1f} fruit/100 steps; "
          f"steer {np.mean(hit):.2f} vs chance {np.mean(chance):.2f}")
    print("  on-policy steer by head distance from start:", split(dist, hit))
    print("  share of on-policy states per bin:", "  ".join(f"{np.mean([(lo<=x<=hi) for x in dist]):.2f}" for lo, hi in BINS))
    # 2) probe states (straight snakes all over the board), split the same way
    P = make_probe_set(25, "rooms", 5.0, n_configs=1500)
    Z, Q = ag.probe_embed(P["X"])
    best = np.isclose(Q, Q.max(1, keepdims=True))
    st = (best & P["good"]).sum(1) / best.sum(1)
    heads = [np.argwhere(x.reshape(5, 25, 25)[0])[0] for x in P["X"]]
    pd = [abs(h[0] - C) + abs(h[1] - C) for h in heads]
    print("  probe steer by head distance:", split(pd, st), f" chance {P['good'].mean():.2f}")
    print("  probe: share of states with all-tied Q:", f"{best.all(1).mean():.2f}")
