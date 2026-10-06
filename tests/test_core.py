import json

import numpy as np
import pytest

from snake_rl.agents import AGENTS, make_agent
from snake_rl.agents.mfec import MFECAgent
from snake_rl.env import Snake
from snake_rl.nn import MLP
from snake_rl.returns import NStep
from snake_rl.run import epsilon, main


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
    env, rng = Snake(size=7 if kw else 5, seed=0, **kw), np.random.default_rng(0)
    agent = make_agent(name, env, rng)
    obs = env.reset()
    for t in range(1, 301):
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


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_evaluation_does_not_change_training(name, tmp_path):
    kw = dict(size=6, map="rooms", bonus=5, eps_floor=0.05, eps_decay=600)
    main(name, 4, 1500, root=str(tmp_path / "plain"), **kw)
    main(name, 4, 1500, root=str(tmp_path / "eval"), eval_every=500, eval_episodes=2, **kw)
    stem = f"g6_d0_rooms_b5/{name}_eps0.05_epsd600_s4.json"
    a, b = (json.load(open(tmp_path / r / stem)) for r in ("plain", "eval"))
    assert a["episodes"] == b["episodes"] and a["diagnostics"] == b["diagnostics"]
    assert [t for t, _ in b["evaluations"]] == [500, 1000, 1500]
    assert all(len(eps) == 2 and all(len(e) == 4 for e in eps) for _, eps in b["evaluations"])
