import json

import numpy as np
import pytest

from snake_rl.agents import AGENTS, make_agent
from snake_rl.agents.mfec import MFECAgent
from snake_rl.env import Snake
from snake_rl.nn import MLP
from snake_rl.returns import NStep
from snake_rl.run import epsilon, food_radius, main


def test_mlp_backward_matches_finite_differences():
    rng = np.random.default_rng(0)
    net = MLP([5, 7, 3], rng)
    x, y = rng.normal(size=(4, 5)), rng.normal(size=(4, 3))
    loss = lambda: 0.5 * ((net.forward(x) - y) ** 2).sum()
    loss()
    grads = net.backward(net.forward(x) - y)
    for p, g in zip(net.params(), grads):
        for i in np.ndindex(p.shape):
            old = p[i]
            p[i] = old + 1e-6; up = loss()
            p[i] = old - 1e-6; down = loss()
            p[i] = old
            assert g[i] == pytest.approx((up - down) / 2e-6, rel=1e-4, abs=1e-6)


def test_nstep_targets():
    ns = NStep(N=2, gamma=0.5)
    assert ns.push("a", 1.0, "s1", False, False) == []
    assert ns.push("b", 2.0, "s2", False, False) == [("a", 1.0 + 0.5 * 2.0, "s2", 0.25)]
    # terminal flushes everything with disc = 0
    assert ns.push("c", 4.0, "s3", True, False) == [("b", 2.0 + 0.5 * 4.0, "s3", 0.0), ("c", 4.0, "s3", 0.0)]
    # truncation flushes but still bootstraps
    ns.push("d", 1.0, "s4", False, False)
    assert ns.push("e", 1.0, "s5", False, True) == [("d", 1.5, "s5", 0.25), ("e", 1.0, "s5", 0.5)]


def test_env_obs_shape_and_death():
    env = Snake(size=5, distractors=2)
    obs = env.reset()
    assert obs.shape == (env.obs_dim,) == ((3 + 2) * 25,)
    for _ in range(10):  # going straight from the centre must hit the wall
        _, r, term, _ = env.step(0)
        if term:
            break
    assert term and r == -1.0


def test_mfec_backward_max_update_and_lookup():
    ag = MFECAgent(obs_dim=2, n_actions=2, rng=np.random.default_rng(0), key_dim=2, k=2, gamma=0.5)
    ag.A = np.eye(2)  # keys = observations, so the numbers below are easy to check
    s0, s1 = np.array([0.0, 0.0]), np.array([1.0, 0.0])

    def episode(steps):  # steps: (obs, action, reward); the last one terminates
        for i, (o, a, r) in enumerate(steps):
            ag._h = o @ ag.A
            ag.observe(o, a, r, None, i == len(steps) - 1, False, i + 1)

    episode([(s0, 0, 0.0), (s1, 0, 2.0)])                   # R = 1.0 at s0, 2.0 at s1
    assert ag.bufs[0].lookup(s0, 0) == 1.0 and ag.bufs[0].lookup(s1, 0) == 2.0
    episode([(s0, 0, -4.0)])                                # worse return: the max keeps 1.0
    episode([(s1, 0, 6.0)])                                 # better return: replaces 2.0
    assert ag.bufs[0].n == 2 and ag.bufs[0].lookup(s0, 0) == 1.0 and ag.bufs[0].lookup(s1, 0) == 6.0
    assert ag.bufs[0].lookup(np.array([0.4, 0.0]), 0) == pytest.approx(3.5)  # unseen: mean of k=2
    assert ag.bufs[1].lookup(s0, 0) == 0.0                  # empty buffer


@pytest.mark.parametrize("kw", [{}, {"map": "pillars", "bonus": 3.0}], ids=["open", "pillars+bonus"])
@pytest.mark.parametrize("name", sorted(AGENTS))
def test_agent_runs(name, kw):
    if name.endswith("naturecnn"):
        kw = {**kw, "render": "pixels"}
    env, rng = Snake(size=7 if kw else 5, seed=0, **kw), np.random.default_rng(0)
    agent = make_agent(name, env, rng)
    obs = env.reset()
    for t in range(1, 301 if not env.pixels else 1101):  # 1000 replay rows before DQN trains
        a = agent.act(obs, 0.5, t)
        assert a in range(env.n_actions)
        nobs, r, term, trunc = env.step(a)
        agent.observe(obs, a, r, nobs, term, trunc, t)
        obs = env.reset() if term or trunc else nobs


def test_run_writes_results(tmp_path):
    main("dqn", 3, 500, D=1, size=6, root=str(tmp_path))
    d = json.load(open(tmp_path / "g6_d1" / "dqn_s3.json"))
    assert d["agent"] == "dqn" and d["steps"] == 500 and d["eps_floor"] == 0.02 and d["episodes"]
    main("dqn", 3, 500, D=1, size=6, root=str(tmp_path), eps_floor=0.1)
    d = json.load(open(tmp_path / "g6_d1" / "dqn_eps0.1_s3.json"))
    assert d["agent"] == "dqn" and d["eps_floor"] == 0.1


def test_episode_log_splits_bonus_points_and_truncation(tmp_path):
    main("random", 0, 4000, size=6, root=str(tmp_path), map="rooms", bonus=5)
    e = np.array(json.load(open(tmp_path / "g6_d0_rooms_b5" / "random_s0.json"))["episodes"])
    t, score, foods, trunc = e.T
    bonus = (score - foods) / 5
    assert e.shape[1] == 4 and np.all(bonus >= 0) and np.allclose(bonus, np.round(bonus))
    assert set(trunc) <= {0, 1}


def test_convnet_backward_matches_finite_differences():
    from snake_rl.nn import ConvNet
    rng = np.random.default_rng(0)
    net = ConvNet((2, 4, 4), convs=[3, 2], fcs=[5, 3], rng=rng)
    x, y = rng.normal(size=(3, 2 * 16)), rng.normal(size=(3, 3))
    loss = lambda: 0.5 * ((net.forward(x) - y) ** 2).sum()
    loss()
    grads = net.backward(net.forward(x) - y)
    for p, g in zip(net.params(), grads):
        for i in np.ndindex(p.shape):
            old = p[i]
            p[i] = old + 1e-6; up = loss()
            p[i] = old - 1e-6; down = loss()
            p[i] = old
            assert g[i] == pytest.approx((up - down) / 2e-6, rel=1e-4, abs=1e-6)


def test_open_env_unchanged_by_new_options():
    a, b = Snake(size=7, seed=3, distractors=1), Snake(size=7, seed=3, distractors=1, map="open", bonus=0.0)
    assert a.obs_shape == (4, 7, 7)
    oa, ob = a.reset(), b.reset()
    arng = np.random.default_rng(1)
    for _ in range(200):
        act = int(arng.integers(3))
        (oa, ra, ta, _), (ob, rb, tb, _) = a.step(act), b.step(act)
        assert np.array_equal(oa, ob) and ra == rb and ta == tb
        if ta:
            oa, ob = a.reset(), b.reset()


@pytest.mark.parametrize("name", ["pillars", "walls", "rooms"])
@pytest.mark.parametrize("size", [7, 9, 12])
def test_maps_are_valid(name, size):
    env = Snake(size=size, map=name)  # raises if the start is blocked or the free area is split
    assert env.walls.any() and env.obs_shape == (4, size, size)
    env.reset()
    assert env.food not in env.wall_set
    obs = env.step(0)[0].reshape(env.obs_shape)
    assert obs[3].sum() == env.walls.sum()


def test_wall_kills():
    env = Snake(size=7, map="rooms")  # row 3 is walled at cols 5,6; heading right from (3,3)
    env.reset()
    rs = [env.step(0) for _ in range(2)]
    assert rs[-1][2] and rs[-1][1] == -1.0


def test_bonus_food_lifecycle():
    env = Snake(size=7, bonus=5.0, bonus_every=1, bonus_life=3, seed=0)
    assert env.obs_shape == (4, 7, 7)
    env.reset()
    env.food = (3, 4)  # right ahead
    obs, r, _, _ = env.step(0)
    assert r == 1.0 and env.bonus_pos is not None and env.bonus_left == 3
    assert obs.reshape(env.obs_shape)[3].max() == 1.0
    env.bonus_pos = (0, 0)  # park it out of the way; it must vanish after bonus_life steps
    for _ in range(2):
        env.step(0)
        assert env.bonus_pos is not None
    env.step(0)
    assert env.bonus_pos is None


def test_bonus_food_pays_and_grows():
    env = Snake(size=7, bonus=5.0, seed=0)
    env.reset()
    env.bonus_pos, env.bonus_left = (3, 4), 5
    _, r, term, _ = env.step(0)
    assert r == 5.0 and not term and env.score == 5 and env.bonus_pos is None and len(env.body) == 3


def test_epsilon_schedules():
    assert [epsilon(t) for t in (0, 2500, 4950, 9999)] == [1.0, 0.5, 0.02, 0.02]  # original schedule
    assert epsilon(0, 0.1, 250_000) == 1.0 and abs(epsilon(125_000, 0.1, 250_000) - 0.55) < 1e-12
    assert epsilon(250_000, 0.1, 250_000) == 0.1 == epsilon(10**6, 0.1, 250_000)       # DQN: 1 -> 0.1, then flat
    assert epsilon(0, 0.005, 0) == 0.005 == epsilon(10**5, 0.005, 0)                   # fixed epsilon


def test_eps_decay_is_recorded_and_tagged(tmp_path):
    main("random", 0, 300, size=6, root=str(tmp_path), eps_floor=0.005, eps_decay=0)
    d = json.load(open(tmp_path / "g6_d0" / "random_eps0.005_epsd0_s0.json"))
    assert d["eps_floor"] == 0.005 and d["eps_decay"] == 0


def _board(name, **kw):
    """Run kwargs for `main` and the results-dir tag; the *_naturecnn agents need the pixel render."""
    return ({**kw, "render": "pixels"}, "_px") if name.endswith("naturecnn") else (kw, "")


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_evaluation_does_not_change_training(name, tmp_path):
    kw, tag = _board(name, size=6, map="rooms", bonus=5, eps_floor=0.05, eps_decay=600)
    main(name, 4, 1500, root=str(tmp_path / "plain"), **kw)
    main(name, 4, 1500, root=str(tmp_path / "eval"), eval_every=500, eval_episodes=2, **kw)
    stem = f"g6_d0_rooms_b5{tag}/{name}_eps0.05_epsd600_s4.json"
    a, b = (json.load(open(tmp_path / r / stem)) for r in ("plain", "eval"))
    assert a["episodes"] == b["episodes"] and a["diagnostics"] == b["diagnostics"]
    assert [t for t, _ in b["evaluations"]] == [500, 1000, 1500]
    assert all(len(eps) == 2 and all(len(e) == 4 for e in eps) for _, eps in b["evaluations"])


def test_food_radius_keeps_food_within_walkable_reach():
    env, rng = Snake(size=15, seed=2, map="rooms", food_radius=3), np.random.default_rng(0)
    placed, new_food = 0, True
    env.reset()
    for _ in range(3000):
        if new_food:  # the radius applies at the moment food is placed; afterwards the snake moves on
            assert env._steps_from_head()[env.food] <= 3
            placed += 1
        before = env.food
        _, r, term, trunc = env.step(int(rng.integers(3)))
        new_food = env.food != before
        if term or trunc:
            env.reset()
            new_food = True
    assert placed > 100


def test_food_curriculum_schedule_and_run_tag(tmp_path):
    assert food_radius(5, None, 25) is None
    assert food_radius(0, (2, 100), 25) == 2 and food_radius(50, (2, 100), 25) == 26
    assert food_radius(100, (2, 100), 25) is None  # curriculum over: anywhere, as in the real game
    main("random", 0, 300, size=6, root=str(tmp_path), food_curriculum=(2, 200))
    d = json.load(open(tmp_path / "g6_d0" / "random_fc2-200_s0.json"))
    assert d["food_curriculum"] == [2, 200]


def test_food_curriculum_hold_then_grow():
    fc = (2, 250_000, 600_000)
    assert food_radius(0, fc, 25) == 2 == food_radius(249_999, fc, 25)        # held
    assert food_radius(425_000, fc, 25) == 26 and food_radius(600_000, fc, 25) is None
    assert [food_radius(t, (2, 600_000), 25) for t in (0, 137_500, 599_999)] == [2, 13, 49]  # v3 form unchanged


def test_food_relocation_moves_uneaten_food_near_the_head():
    env = Snake(size=15, seed=1, map="rooms", food_radius=3, relocate_food=True)
    env.reset()
    env.food, env.food_age = (0, 0), 2 * 3 + 5 - 1     # far away, one step short of the patience
    env.step(0)                                         # straight ahead from the centre: safe, eats nothing
    assert env.food != (0, 0) and env._steps_from_head()[env.food] <= 3 and env.food_age == 0
    off = Snake(size=15, seed=1, map="rooms", food_radius=3)  # relocation off: the food stays put
    off.reset()
    off.food, off.food_age = (0, 0), 100
    off.step(0)
    assert off.food == (0, 0)


def test_relocation_needs_a_curriculum(tmp_path):
    with pytest.raises(ValueError):
        main("random", 0, 10, size=6, root=str(tmp_path), food_relocate=True)
    main("random", 0, 300, size=6, root=str(tmp_path), food_curriculum=(2, 100, 200), food_relocate=True)
    d = json.load(open(tmp_path / "g6_d0" / "random_fc2-100-200_reloc_s0.json"))
    assert d["food_curriculum"] == [2, 100, 200] and d["food_relocate"] is True


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_probe_does_not_change_training(name, tmp_path):
    kw, tag = _board(name, size=7, map="rooms", bonus=5, eps_floor=0.05, eps_decay=600)
    main(name, 4, 1500, root=str(tmp_path / "plain"), **kw)
    main(name, 4, 1500, root=str(tmp_path / "probe"), probe_every=500, **kw)
    stem = f"d0_rooms_b5{tag}/{name}_eps0.05_epsd600_s4.json"
    a, b = (json.load(open(tmp_path / r / stem)) for r in ("plain", "probe"))
    assert a["episodes"] == b["episodes"] and a["diagnostics"] == b["diagnostics"]
    assert [t for t, _ in b["probes"]] == [0, 500, 1000, 1500]
    if hasattr(make_agent(name, Snake(size=7, map="rooms", bonus=5, render=kw.get("render", "grid")),
                          np.random.default_rng(0)), "probe_embed"):
        assert all(0 <= m["steer"] <= 1 and 0 <= m["decode"] <= 1 for _, m in b["probes"])


def test_probe_set_labels():
    from snake_rl.probe import make_probe_set, steering
    P = make_probe_set(size=9, map="open", n_configs=20)
    assert len(set(P["group"])) == 20 and P["good"].any(1).all()  # food within 2 steps is always reachable
    # A Q that ranks exactly the good actions on top steers perfectly; a constant Q scores the chance rate.
    assert steering(P["good"].astype(float), P)["steer"] == 1.0
    s = steering(np.zeros(P["good"].shape), P)
    assert s["steer"] == pytest.approx(s["steer_chance"])


@pytest.mark.parametrize("name", ["dqn", "nec", "dqn_cnn", "nec_cnn"])
def test_train_telemetry_is_logged_and_streamed(name, tmp_path):
    kw = dict(size=7, map="rooms", bonus=5, eps_floor=0.05, eps_decay=600)
    main(name, 4, 1500, root=str(tmp_path / "off"), train_log_every=0, **kw)
    main(name, 4, 1500, root=str(tmp_path / "on"), train_log_every=500, **kw)
    stem = f"d0_rooms_b5/{name}_eps0.05_epsd600_s4"
    a, b = (json.load(open(tmp_path / r / f"{stem}.json")) for r in ("off", "on"))
    assert a["episodes"] == b["episodes"] and a["train"] == []
    live = [json.loads(l) for l in open(tmp_path / "on" / f"{stem}.train.jsonl")]  # kept after the run
    assert [r.pop("t") for r in live] == [t for t, _ in b["train"]] and live == [s for _, s in b["train"]]
    t, s = b["train"][-1]
    assert t == 1500 and s["updates"] > 0 and np.isfinite(s["loss"]) and 0 <= s["clipped"] <= 1
    assert all(k in s for k in ("td_abs", "q_taken", "target", "grad_norm", "update_ratio", "dead_relu"))


def test_strided_convnet_backward_matches_finite_differences():
    from snake_rl.nn import ConvNet
    rng = np.random.default_rng(0)
    net = ConvNet((2, 7, 7), convs=[(3, 3, 1), (2, 3, 2), (2, 3, 2)], fcs=[5, 3], rng=rng)
    assert net.out_hw == 2  # 7 -> 7 -> 4 -> 2
    x, y = rng.normal(size=(3, 2 * 49)), rng.normal(size=(3, 3))
    loss = lambda: 0.5 * ((net.forward(x) - y) ** 2).sum()
    loss()
    grads = net.backward(net.forward(x) - y)
    for p, g in zip(net.params(), grads):
        for i in np.ndindex(p.shape):
            old = p[i]
            p[i] = old + 1e-6; up = loss()
            p[i] = old - 1e-6; down = loss()
            p[i] = old
            assert g[i] == pytest.approx((up - down) / 2e-6, rel=1e-4, abs=1e-6)


def test_dqn_shaped_cnn_on_the_25_grid():
    from snake_rl.nn import ConvNet, DQN_CONVS, DQN_FC
    net = ConvNet((5, 25, 25), DQN_CONVS, [DQN_FC, 32], np.random.default_rng(0))
    assert [ho for _, _, _, ho in net.layers] == [25, 13, 7] and net.W[3].shape == (7 * 7 * 64, 512)
    assert net.forward(np.zeros((2, 5 * 625))).shape == (2, 32)


@pytest.mark.parametrize("name", ["dqn", "nec_dqncnn", "mfec", "random"])
def test_saved_agent_can_be_watched_without_changing_training(name, tmp_path):
    import pickle
    from snake_rl.watch import frames
    kw = dict(size=7, map="rooms", bonus=5, eps_floor=0.05, eps_decay=600)
    main(name, 4, 1500, root=str(tmp_path / "plain"), **kw)
    main(name, 4, 1500, root=str(tmp_path / "saved"), save_agent=True, save_every=500, **kw)
    stem = f"d0_rooms_b5/{name}_eps0.05_epsd600_s4"
    a, b = (json.load(open(tmp_path / r / f"{stem}.json")) for r in ("plain", "saved"))
    assert a["episodes"] == b["episodes"]
    for t in (500, 1000, 1500):
        assert (tmp_path / "saved" / f"{stem}.agent_t{t}.pkl").exists()
    ck = pickle.load(open(tmp_path / "saved" / f"{stem}.agent.pkl", "rb"))
    assert ck["size"] == 7 and ck["map"] == "rooms" and ck["t"] == 1500
    assert getattr(ck["agent"], "S", None) is None and getattr(ck["agent"], "R_obs", None) is None  # no replay saved
    env = Snake(size=7, map="rooms", bonus=5, seed=0)
    fs = list(frames(ck["agent"], env, 0.0, 2, np.random.default_rng(0), None))
    assert sum(f["end"] is not None for f in fs) == 2 and all(f["a"] in (0, 1, 2) for f in fs if f["end"] is None)


def test_checkpoint_records_the_run_settings(tmp_path):
    import pickle
    main("random", 0, 300, size=9, root=str(tmp_path), food_curriculum=(2, 100, 400), food_relocate=True,
         save_agent=True)
    ck = pickle.load(open(tmp_path / "g9_d0" / "random_fc2-100-400_reloc_s0.agent.pkl", "rb"))
    assert ck["t"] == 300 and ck["size"] == 9 and ck["food_curriculum"] == (2, 100, 400) and ck["food_relocate"]


def test_nec_encoder_gradient_matches_finite_differences():
    import copy
    from snake_rl.agents.nec import NECAgent
    rng = np.random.default_rng(0)
    ag = NECAgent(6, 2, rng, key_dim=3, hidden=5, p=4, N=1, batch=8, mem_lr=0.0)  # mem_lr 0: memory left untouched
    for d in ag.dnds:
        for _ in range(10):
            d._insert(rng.normal(size=3), rng.normal(), 0)
    ag.R_obs[:20], ag.R_act[:20], ag.R_ret[:20] = rng.normal(size=(20, 6)), rng.integers(2, size=20), rng.normal(size=20)
    ag.r_n = 20
    idx = copy.deepcopy(ag.rng).integers(ag.r_n, size=ag.batch)  # the minibatch _train is about to draw
    X, A, R = ag.R_obs[idx], ag.R_act[idx], ag.R_ret[idx]
    got = {}
    ag.opt.step = lambda grads: got.setdefault("g", grads)  # capture the gradient instead of applying it
    ag.opt.last = {}  # the stubbed step records no optimizer telemetry
    ag._train()

    def loss():
        H, L = ag.enc.forward(X), 0.0
        for a, d in enumerate(ag.dnds):
            m = A == a
            if m.any():
                nb, dd = d.knn(H[m])
                k = 1 / (dd + d.delta)
                L += 0.5 * ((((k / k.sum(1, keepdims=True)) * d.vals[nb]).sum(1) - R[m]) ** 2).sum()
        return L / len(X)

    for p, g in zip(ag.enc.params(), got["g"]):
        for i in np.ndindex(p.shape):
            old = p[i]
            p[i] = old + 1e-6; up = loss()
            p[i] = old - 1e-6; down = loss()
            p[i] = old
            assert g[i] == pytest.approx((up - down) / 2e-6, rel=1e-4, abs=1e-7)


def test_nature_convnet_shapes_and_gradient():
    from snake_rl.nn import ConvNet, NATURE_CONVS, NATURE_FC
    net = ConvNet((4, 84, 84), NATURE_CONVS, [NATURE_FC, 32], np.random.default_rng(0))
    assert [ho for _, _, _, ho in net.layers] == [20, 9, 7] and net.W[3].shape == (7 * 7 * 64, 512)
    assert net.forward(np.zeros((2, 4 * 84 * 84))).shape == (2, 32)
    # same code path (valid padding, stride 2, kernel 4) on a size the finite differences can afford
    rng = np.random.default_rng(0)
    small = ConvNet((2, 9, 9), convs=[(3, 4, 2, 0), (2, 3, 1, 0)], fcs=[5, 3], rng=rng)
    assert [ho for _, _, _, ho in small.layers] == [3, 1]
    x, y = rng.normal(size=(3, 2 * 81)), rng.normal(size=(3, 3))
    loss = lambda: 0.5 * ((small.forward(x) - y) ** 2).sum()
    loss()
    grads = small.backward(small.forward(x) - y)
    for p, g in zip(small.params(), grads):
        for i in np.ndindex(p.shape):
            old = p[i]
            p[i] = old + 1e-6; up = loss()
            p[i] = old - 1e-6; down = loss()
            p[i] = old
            assert g[i] == pytest.approx((up - down) / 2e-6, rel=1e-4, abs=1e-6)


def test_pixel_render_is_the_same_game_in_gray():
    g, px = Snake(size=25, seed=5, map="rooms", bonus=5), Snake(size=25, seed=5, map="rooms", bonus=5, render="pixels")
    assert px.obs_shape == (4, 84, 84) and px.obs_dim == 4 * 84 * 84 and px.cell == 3
    g.reset(), px.reset()
    f = px._obs().reshape(4, 84, 84)  # reading the stack again pushes one more identical frame
    assert np.array_equal(f[0], f[3])
    arng = np.random.default_rng(1)
    for _ in range(600):  # identical dynamics and rng draws: only the picture differs
        a = int(arng.integers(3))
        (_, rg, tg, ug), (o, rp, tp, up) = g.step(a), px.step(a)
        assert (rg, tg, ug) == (rp, tp, up) and g.body == px.body and g.food == px.food
        fr = o.reshape(4, 84, 84)[3]
        hr, hc = px.body[0]
        assert fr[4 + 3 * hr + 1, 4 + 3 * hc + 1] == np.float32(255) / np.float32(255)  # the head's pixels
        assert fr[0, 0] == np.float32(60) / np.float32(255)  # the margin is drawn like a wall
        if tg or ug:
            g.reset(), px.reset()
    # the stack holds the last 4 frames, newest last, and static_obs leaves it alone
    px.reset()
    o1, *_ = px.step(0)
    s = px.static_obs()
    o2, *_ = px.step(0)
    assert np.array_equal(o2.reshape(4, -1)[2], o1.reshape(4, -1)[3]) and not np.array_equal(s, o2)
    assert np.array_equal(px.static_obs().reshape(4, -1)[0], px.static_obs().reshape(4, -1)[3])


def test_pixel_render_defaults_and_limits():
    assert Snake(size=7).obs_shape == (3, 7, 7) and not Snake(size=7).pixels
    with pytest.raises(ValueError):
        Snake(size=7, render="pixels", distractors=1)
    with pytest.raises(ValueError):
        Snake(size=7, render="sepia")


def test_uint8_replay_matches_float_replay():
    from snake_rl.agents.dqn import DQNAgent
    from snake_rl.agents.nec import NECAgent
    for make in (lambda u8: DQNAgent(28224, 3, np.random.default_rng(0), encoder="naturecnn", obs_shape=(4, 84, 84),
                                     replay_cap=1500, obs_u8=u8),
                 lambda u8: NECAgent(28224, 3, np.random.default_rng(0), encoder="naturecnn", obs_shape=(4, 84, 84),
                                     replay_cap=1500, N=5, obs_u8=u8)):
        runs = []
        for u8 in (False, True):
            env, agent = Snake(size=7, seed=2, render="pixels"), make(u8)
            obs, acts = env.reset(), []
            for t in range(1, 1051):
                a = agent.act(obs, 0.3, t)
                nobs, r, term, trunc = env.step(a)
                agent.observe(obs, a, r, nobs, term, trunc, t)
                obs = env.reset() if term or trunc else nobs
                acts.append(a)
            runs.append(acts)
        assert runs[0] == runs[1]


def test_pixel_run_is_tagged_and_checkpointed(tmp_path):
    import pickle
    main("random", 0, 300, size=9, root=str(tmp_path), render="pixels", save_agent=True, probe_every=0)
    d = json.load(open(tmp_path / "g9_d0_px" / "random_s0.json"))
    assert d["render"] == "pixels"
    ck = pickle.load(open(tmp_path / "g9_d0_px" / "random_s0.agent.pkl", "rb"))
    assert ck["render"] == "pixels"


def test_pixel_probe_set_matches_the_grid_one():
    from snake_rl.probe import make_probe_set
    a = make_probe_set(size=9, map="open", n_configs=10)
    b = make_probe_set(size=9, map="open", n_configs=10, render="pixels")
    assert b["X"].shape == (len(a["X"]), 4 * 84 * 84)
    assert np.array_equal(a["group"], b["group"]) and np.array_equal(a["off"], b["off"]) and np.array_equal(a["good"], b["good"])


def test_rmsprop_step_matches_hand_computation():
    from snake_rl.nn import RMSProp
    p = [np.array([1.0, -2.0])]
    opt = RMSProp(p, lr=0.1, rho=0.5, eps=0.25, clip=100.0)
    opt.step([np.array([2.0, -4.0])])
    v = 0.5 * np.array([4.0, 16.0])  # (1 - rho) g^2
    assert np.allclose(p[0], np.array([1.0, -2.0]) - 0.1 * np.array([2.0, -4.0]) / (np.sqrt(v) + 0.25))
    assert opt.last["grad_norm"] == pytest.approx(np.sqrt(20.0)) and opt.last["clipped"] == 0.0
    opt.step([np.array([2.0, -4.0])])  # second step: v = rho v + (1 - rho) g^2
    assert np.allclose(opt.v[0], 0.5 * v + 0.5 * np.array([4.0, 16.0]))


@pytest.mark.parametrize("name", ["dqn", "nec"])
def test_optimizer_setting_is_recorded_tagged_and_changes_training(name, tmp_path):
    from snake_rl.nn import Adam, RMSProp
    kw = dict(size=6, eps_floor=0.05, eps_decay=300)
    main(name, 4, 1500, root=str(tmp_path), **kw)
    main(name, 4, 1500, root=str(tmp_path), opt="rmsprop", lr=1e-4, **kw)
    a = json.load(open(tmp_path / "g6_d0" / f"{name}_eps0.05_epsd300_s4.json"))
    b = json.load(open(tmp_path / "g6_d0" / f"{name}_rmsprop_lr0.0001_eps0.05_epsd300_s4.json"))
    assert (a["opt"], a["lr"], b["opt"], b["lr"]) == ("adam", None, "rmsprop", 1e-4)
    assert a["episodes"] != b["episodes"]
    ag = make_agent(name, Snake(size=6), np.random.default_rng(0))
    assert isinstance(ag.opt, Adam) and ag.opt.lr == 5e-4  # the default is untouched
    ag.set_optimizer("rmsprop", 1e-4)
    assert isinstance(ag.opt, RMSProp) and ag.opt.lr == 1e-4 and ag.opt.p[0] is (ag.q if name == "dqn" else ag.enc).params()[0]


def test_optimizer_setting_needs_an_agent_with_one(tmp_path):
    with pytest.raises(ValueError):
        main("mfec", 0, 100, size=6, root=str(tmp_path), opt="rmsprop")


def test_pixel_probe_stack_shows_a_snake_moving_straight_in():
    from snake_rl.probe import make_probe_set
    P = make_probe_set(size=9, map="open", n_configs=10, render="pixels")
    f = P["X"].reshape(-1, 4, 84, 84)
    moving = [(x[0] != x[3]).any() for x in f]
    assert np.mean([(x[2] != x[3]).any() for x in f]) > 0.7  # most probe snakes have room to have come from behind (not those backed against a wall or the food)
    assert np.mean(moving) > 0.3
    # the newest frame is the probe state itself, as in the grid probe
    g = make_probe_set(size=9, map="open", n_configs=10, render="grid")
    assert len(g["X"]) == len(f)


def test_grid_frame_stack_and_wall_scale():
    plain = Snake(size=9, seed=1, map="rooms", bonus=5)
    st = Snake(size=9, seed=1, map="rooms", bonus=5, frame_stack=3, wall_scale=0.5)
    assert st.obs_shape == (3 * plain.channels, 9, 9) and st.obs_dim == 3 * plain.obs_dim
    o0, os_ = plain.reset(), st.reset()
    f = os_.reshape(3, plain.channels, 9, 9)
    assert np.array_equal(f[0], f[2]) and f[:, 3].max() == 0.5 and plain.walls.any()
    assert np.array_equal(f[2][:3], o0.reshape(plain.channels, 9, 9)[:3])  # head/body/food unchanged
    arng = np.random.default_rng(0)
    prev = f[2]
    for _ in range(50):  # same dynamics; the newest frame is the plain observation (walls scaled); older ones shift
        a = int(arng.integers(3))
        (op, rp, tp, _), (osk, rs, ts, _) = plain.step(a), st.step(a)
        assert (rp, tp) == (rs, ts)
        g = osk.reshape(3, plain.channels, 9, 9)
        assert np.array_equal(g[1], prev) and np.array_equal(g[2][:3], op.reshape(plain.channels, 9, 9)[:3])
        prev = g[2]
        if tp:
            plain.reset(), st.reset()
            prev = st._obs().reshape(3, plain.channels, 9, 9)[2]  # reset pushes a fresh stack
            break
    assert np.array_equal(Snake(size=9, seed=1, map="rooms").reset(), Snake(size=9, seed=1, map="rooms", frame_stack=1).reset())
    with pytest.raises(ValueError):
        Snake(size=7, frame_stack=2, distractors=1)


def test_pixel_wall_scale_changes_only_the_wall_level():
    a = Snake(size=25, seed=2, map="rooms", render="pixels")
    b = Snake(size=25, seed=2, map="rooms", render="pixels", wall_scale=0.0)
    c = Snake(size=25, seed=2, map="rooms", render="pixels", wall_scale=10.0)
    fa, fb, fc = (e.reset().reshape(4, 84, 84)[3] for e in (a, b, c))
    walls = fa == np.float32(60) / np.float32(255)
    assert walls.any() and (fb[walls] == 0).all() and (fc[walls] == 1).all()
    assert np.array_equal(fa[~walls], fb[~walls]) and np.array_equal(fa[~walls], fc[~walls])


def test_results_dir_tags_for_stack_and_wall_scale(tmp_path):
    from snake_rl.run import results_dir
    assert results_dir(0, 25, "r", "rooms", 5.0) == "r/g25_d0_rooms_b5"
    assert results_dir(0, 25, "r", "rooms", 5.0, frame_stack=4) == "r/g25_d0_rooms_b5_fs4"
    assert results_dir(0, 25, "r", "rooms", 5.0, "pixels", 4) == "r/g25_d0_rooms_b5_px"
    assert results_dir(0, 25, "r", "rooms", 5.0, "pixels", wall_scale=0.0) == "r/g25_d0_rooms_b5_px_ws0"
    main("dqn", 1, 300, size=7, root=str(tmp_path), map="rooms", frame_stack=2, wall_scale=0.5)
    d = json.load(open(tmp_path / "d0_rooms_fs2_ws0.5" / "dqn_s1.json"))
    assert d["frame_stack"] == 2 and d["wall_scale"] == 0.5


def test_grid_stack_probe_set():
    from snake_rl.probe import make_probe_set
    P = make_probe_set(size=9, map="open", n_configs=10, frame_stack=3)
    g = make_probe_set(size=9, map="open", n_configs=10)
    assert P["X"].shape == (len(g["X"]), 3 * g["X"].shape[1])
    assert np.array_equal(P["X"][:, -g["X"].shape[1]:], g["X"])  # the newest frame is the plain probe state
