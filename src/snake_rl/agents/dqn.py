"""DQN baseline (Mnih et al., 2015) with optional N-step targets, same MLP budget as NEC.

Target: G + disc * max_a Q_target(boot_obs, a), Huber loss, periodic target sync.
N=1 gives standard DQN; N>1 isolates the contribution of N-step returns.
"""
import numpy as np
from snake_rl.nn import MLP, Adam
from snake_rl.returns import NStep


class DQNAgent:
    def __init__(self, obs_dim, n_actions, rng, N=1, gamma=0.99, hidden=64, lr=5e-4,
                 replay_cap=50000, batch=32, train_every=4, target_every=1000):
        self.rng, self.nA = rng, n_actions
        self.q = MLP([obs_dim, hidden, hidden, n_actions], rng)
        self.qt = MLP([obs_dim, hidden, hidden, n_actions], rng)
        self.qt.copy_from(self.q)
        self.opt = Adam(self.q.params(), lr)
        self.nstep = NStep(N, gamma)
        self.batch, self.train_every, self.target_every = batch, train_every, target_every
        self.S = np.zeros((replay_cap, obs_dim))
        self.A = np.zeros(replay_cap, np.int64)
        self.G = np.zeros(replay_cap)
        self.B = np.zeros((replay_cap, obs_dim))
        self.D = np.zeros(replay_cap)
        self.n, self.i, self.cap = 0, 0, replay_cap

    def act(self, obs, eps, t):
        if self.rng.random() < eps:
            return int(self.rng.integers(self.nA))
        q = self.q.forward(obs[None])[0]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def observe(self, obs, a, r, next_obs, term, trunc, t):
        for (o, act), G, boot, disc in self.nstep.push((obs, a), r, next_obs, term, trunc):
            j = self.i
            self.S[j], self.A[j], self.G[j], self.B[j], self.D[j] = o, act, G, boot, disc
            self.i = (self.i + 1) % self.cap
            self.n = min(self.n + 1, self.cap)
        if t % self.train_every == 0 and self.n >= 1000:
            self._train()
        if t % self.target_every == 0:
            self.qt.copy_from(self.q)

    def _train(self):
        idx = self.rng.integers(self.n, size=self.batch)
        y = self.G[idx] + self.D[idx] * self.qt.forward(self.B[idx]).max(1)
        Q = self.q.forward(self.S[idx])
        err = Q[np.arange(self.batch), self.A[idx]] - y
        g = np.zeros_like(Q)
        g[np.arange(self.batch), self.A[idx]] = np.clip(err, -1, 1) / self.batch  # Huber grad
        self.opt.step(self.q.backward(g))
