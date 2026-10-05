"""Neural Episodic Control (Pritzel et al., 2017) in NumPy.

Encoder: MLP obs -> h (stands in for the paper's CNN).
One Differentiable Neural Dictionary (DND) per action.
Q(s,a) = sum_i w_i v_i over the p nearest keys, w_i = k_i / sum_j k_j,
k_i = 1 / (||h - h_i||^2 + delta)                                  (eqs. 1, 2, 5)

Two learning timescales:
  fast: tabular write  Q_i <- Q_i + alpha (Q^N - Q_i) on exact key match,
        otherwise append (eq. 4)
  slow: SGD on L = 1/2 (Q(s,a) - Q^N)^2 from a replay buffer, with gradients
        flowing into the encoder AND into the neighbour keys/values (Sec. 3.4)

learn_embedding=False freezes the random encoder and disables the SGD path,
giving an MFEC-like ablation that keeps everything else identical.

refresh_every>0 (not in the paper) keeps the raw observation behind every key and
periodically re-embeds all keys with the current encoder, removing key drift.
"""
import numpy as np
from snake_rl.nn import MLP, Adam
from snake_rl.returns import NStep


class DND:
    def __init__(self, capacity, dim, p, delta, obs_dim=None):
        self.obs = np.zeros((capacity, obs_dim)) if obs_dim else None
        self.stats = {"appends": 0, "exact_updates": 0, "evictions": 0}
        self.keys = np.zeros((capacity, dim))
        self.vals = np.zeros(capacity)
        self.last_used = np.zeros(capacity, np.int64)
        self.n, self.cap, self.p, self.delta = 0, capacity, p, delta

    def knn(self, H):
        """Exact p-NN by brute force (the paper uses approximate kd-trees for scale)."""
        K = self.keys[: self.n]
        d = (H ** 2).sum(1)[:, None] - 2.0 * H @ K.T + (K ** 2).sum(1)[None, :]
        d = np.maximum(d, 0.0)
        k = min(self.p, self.n)
        if k < self.n:
            idx = np.argpartition(d, k - 1, axis=1)[:, :k]
        else:
            idx = np.broadcast_to(np.arange(self.n), (H.shape[0], self.n))
        return idx, np.take_along_axis(d, idx, 1)

    def lookup(self, H, t=None, return_dist=False):
        if self.n == 0:
            z = np.zeros(H.shape[0])
            return (z, np.full(H.shape[0], np.inf)) if return_dist else z
        idx, d = self.knn(H)
        kern = 1.0 / (d + self.delta)
        w = kern / kern.sum(1, keepdims=True)
        if t is not None:
            self.last_used[idx] = t  # LRU = least recently used as a neighbour
        q = (w * self.vals[idx]).sum(1)
        return (q, d.mean(1)) if return_dist else q

    def write(self, h, v, t, alpha, obs=None, match_eps=1e-6):
        if self.n > 0:
            idx, _ = self.knn(h[None])
            j = idx[0][np.argmin(((self.keys[idx[0]] - h) ** 2).sum(1))]
            if ((self.keys[j] - h) ** 2).sum() < match_eps:  # exact key already present
                self.vals[j] += alpha * (v - self.vals[j])
                self.last_used[j] = t
                self.stats["exact_updates"] += 1
                return
        if self.n < self.cap:
            i = self.n
            self.n += 1
            self.stats["appends"] += 1
        else:
            i = int(np.argmin(self.last_used))
            self.stats["evictions"] += 1
        self.keys[i], self.vals[i], self.last_used[i] = h, v, t
        if self.obs is not None:
            self.obs[i] = obs


class NECAgent:
    def __init__(self, obs_dim, n_actions, rng, learn_embedding=True, key_dim=32, hidden=64,
                 p=50, delta=1e-3, dnd_cap=20000, N=50, gamma=0.99, alpha=0.1,
                 lr=5e-4, mem_lr=1e-2, replay_cap=20000, batch=32, train_every=4,
                 refresh_every=0, bonus_beta=0.0):
        self.rng, self.nA, self.learn = rng, n_actions, learn_embedding
        self.enc = MLP([obs_dim, hidden, key_dim], rng)
        self.opt = Adam(self.enc.params(), lr)
        self.refresh_every, self.bonus_beta = refresh_every, bonus_beta
        self.dnds = [DND(dnd_cap, key_dim, p, delta, obs_dim if refresh_every else None)
                     for _ in range(n_actions)]
        self.nstep = NStep(N, gamma)
        self.alpha, self.mem_lr, self.batch, self.train_every = alpha, mem_lr, batch, train_every
        self.R_obs = np.zeros((replay_cap, obs_dim))
        self.R_act = np.zeros(replay_cap, np.int64)
        self.R_ret = np.zeros(replay_cap)
        self.r_n, self.r_i, self.r_cap = 0, 0, replay_cap

    def _embed(self, obs):
        return self.enc.forward(obs[None])[0]

    def _q(self, h, t=None):
        return np.array([d.lookup(h[None], t)[0] for d in self.dnds])

    def act(self, obs, eps, t):
        h = self._embed(obs)
        self._h = h
        if self.rng.random() < eps:
            return int(self.rng.integers(self.nA))
        if self.bonus_beta:
            # density bonus (not in the paper): favour actions whose memory is sparse near h.
            # Used for acting only; value targets use the plain Q.
            qd = [d.lookup(h[None], t, return_dist=True) for d in self.dnds]
            q = np.array([x[0][0] for x in qd])
            dm = np.array([x[1][0] for x in qd])
            if np.isinf(dm).any():  # an empty DND: try it
                return int(self.rng.choice(np.flatnonzero(np.isinf(dm))))
            q = q + self.bonus_beta * np.log((dm + 1e-8) / (dm.min() + 1e-8))
        else:
            q = self._q(h, t)
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def observe(self, obs, a, r, next_obs, term, trunc, t):
        for (o, h, act), G, boot, disc in self.nstep.push((obs, self._h, a), r, next_obs, term, trunc):
            target = G + (disc * self._q(self._embed(boot)).max() if disc > 0 else 0.0)  # eq. 3
            self.dnds[act].write(h, target, t, self.alpha, obs=o)
            self.R_obs[self.r_i], self.R_act[self.r_i], self.R_ret[self.r_i] = o, act, target
            self.r_i = (self.r_i + 1) % self.r_cap
            self.r_n = min(self.r_n + 1, self.r_cap)
        if self.learn and t % self.train_every == 0 and self.r_n >= self.batch:
            self._train()
        if self.refresh_every and t % self.refresh_every == 0:
            for d in self.dnds:
                if d.n:
                    d.keys[: d.n] = self.enc.forward(d.obs[: d.n])

    def diagnostics(self):
        out = {}
        for a, d in enumerate(self.dnds):
            v = d.vals[: d.n]
            out[a] = dict(d.stats, n=int(d.n), val_mean=float(v.mean()) if d.n else 0.0,
                          val_absmax=float(np.abs(v).max()) if d.n else 0.0)
        return out

    def _train(self):
        idx = self.rng.integers(self.r_n, size=self.batch)
        X, A, R = self.R_obs[idx], self.R_act[idx], self.R_ret[idx]
        H = self.enc.forward(X)
        gH = np.zeros_like(H)
        B = self.batch
        for a, dnd in enumerate(self.dnds):
            m = A == a
            if not m.any() or dnd.n == 0:
                continue
            Hm = H[m]
            nb, d = dnd.knn(Hm)
            kern = 1.0 / (d + dnd.delta)
            S = kern.sum(1, keepdims=True)
            w = kern / S
            V = dnd.vals[nb]
            Q = (w * V).sum(1)
            g = (Q - R[m]) / B                              # dL/dQ
            dL_dd = g[:, None] * (V - Q[:, None]) / S * (-kern ** 2)   # dL/d(dist_i)
            diff = Hm[:, None, :] - dnd.keys[nb]            # h - h_i
            gH[m] = 2.0 * (dL_dd[:, :, None] * diff).sum(1)  # dL/dh
            # slow SGD path into memory: values (dQ/dv_i = w_i) and keys (dd/dh_i = -2(h - h_i))
            np.add.at(dnd.vals, nb.ravel(), -self.mem_lr * (g[:, None] * w).ravel())
            gK = -2.0 * dL_dd[:, :, None] * diff
            np.add.at(dnd.keys, nb.ravel(), -self.mem_lr * gK.reshape(-1, gK.shape[-1]))
        self.opt.step(self.enc.backward(gH))
