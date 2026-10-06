import json

import numpy as np
import pytest

from snake_rl.agents import AGENTS, make_agent
from snake_rl.agents.mfec import MFECAgent
from snake_rl.env import Snake
from snake_rl.nn import MLP
from snake_rl.returns import NStep
from snake_rl.run import main


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


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_agent_runs(name):
    env, rng = Snake(size=5, seed=0), np.random.default_rng(0)
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
