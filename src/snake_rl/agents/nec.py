"""Neural Episodic Control (Pritzel et al., 2017) in NumPy.

The idea, compared with DQN: DQN stores everything it knows in network weights, which change only
a little per gradient step, so it learns slowly. NEC also keeps a memory (one per action) of
situations it has been in and how good each turned out to be, and estimates Q by looking up
the most similar remembered situations. A good outcome is usable the moment it is written down.
  - An encoder network turns an observation into a short vector h (the "key"/"embedding"), so that
    similar situations get nearby keys.
  - Each action has a DND (Differentiable Neural Dictionary): a table of (key, value) rows, where
    value = the return observed after taking that action in the situation described by key.
  - Q(s, a) = weighted average of the values of the p keys in DND[a] closest to h(s); closer keys
    weigh more.

Encoder: CNN (as in the paper, but small 3x3 stride-1 convs for the tiny grid) or an MLP, obs -> h.
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
from snake_rl.agents.memory import KeyValueMemory
from snake_rl.nn import MLP, ConvNet, Adam
from snake_rl.returns import NStep


class DND(KeyValueMemory):
    """Differentiable Neural Dictionary: the memory of one action.

    Rows 0..n-1 of keys/vals/last_used are in use; the rest are empty slots.
    Storage, knn and LRU insertion live in KeyValueMemory; this adds the kernel lookup and NEC's write rule.
    """

    def lookup(self, H, t=None, return_dist=False):
        """Q estimate for each query key in H: kernel-weighted average of the neighbours' values.

        t: if given, mark the neighbours as used at step t (for eviction).
        return_dist: also return the mean squared distance to the neighbours (how densely this
        region is covered by memory; used by nec_bonus). Infinite if the memory is empty.
        """
        if self.n == 0:
            z = np.zeros(H.shape[0])  # empty memory: Q = 0
            return (z, np.full(H.shape[0], np.inf)) if return_dist else z
        idx, d = self.knn(H)
        kern = 1.0 / (d + self.delta)          # closer neighbours -> larger kernel
        w = kern / kern.sum(1, keepdims=True)  # normalise to weights that sum to 1
        if t is not None:
            self.last_used[idx] = t  # LRU = least recently used as a neighbour
        q = (w * self.vals[idx]).sum(1)
        return (q, d.mean(1)) if return_dist else q

    def write(self, h, v, t, alpha, obs=None, match_eps=1e-6):
        """Remember that the situation with key h was worth v.

        If (practically) the same key is already stored, move its value a fraction alpha towards v,
        a running average over repeat visits. Otherwise add a new row, evicting the least recently
        used row if the memory is full.
        """
        if self.n > 0:
            idx, _ = self.knn(h[None])
            j = idx[0][np.argmin(((self.keys[idx[0]] - h) ** 2).sum(1))]  # the single nearest row
            if ((self.keys[j] - h) ** 2).sum() < match_eps:  # exact key already present
                self.vals[j] += alpha * (v - self.vals[j])
                self.last_used[j] = t
                self.stats["exact_updates"] += 1
                return
        i = self._insert(h, v, t)
        if self.obs is not None:
            self.obs[i] = obs


class NECAgent:
    """
    Hyperparameters:
      learn_embedding  train the encoder (True, NEC) or keep it random and frozen (False, ec_frozen).
      key_dim          length of the embedding h.
      hidden           units in the encoder's hidden layer.
      p                neighbours averaged per Q lookup.
      delta            kernel constant, see DND.
      dnd_cap          max rows per action's memory.
      N, gamma         N-step return length and discount factor (as in DQN).
      alpha            fast learning rate for updating the value of an exactly matching key.
      lr               Adam learning rate for the encoder.
      mem_lr           slow learning rate for gradient updates of stored keys and values.
      replay_cap       rows in the replay buffer used for the slow (gradient) learning.
      batch            replay rows per training step.
      train_every      one training step every this many env steps.
      refresh_every    nec_refresh: recompute all stored keys every this many steps (0 = never).
      bonus_beta       nec_bonus: weight of the exploration bonus when acting (0 = off).
      encoder          "mlp" or "cnn" (the *_cnn agents); obs_shape is the (C, n, n) grid the CNN needs.
    """

    def __init__(self, obs_dim, n_actions, rng, learn_embedding=True, key_dim=32, hidden=64,
                 p=50, delta=1e-3, dnd_cap=20000, N=50, gamma=0.99, alpha=0.1,
                 lr=5e-4, mem_lr=1e-2, replay_cap=20000, batch=32, train_every=4,
                 refresh_every=0, bonus_beta=0.0, encoder="mlp", obs_shape=None):
        # rng: the only source of randomness. nA: number of actions. learn: learn_embedding.
        self.rng, self.nA, self.learn = rng, n_actions, learn_embedding
        # enc: the encoder network, observation -> embedding h of length key_dim.
        if encoder == "cnn":
            self.enc = ConvNet(obs_shape, [16, 32], [hidden, key_dim], rng)
        else:
            self.enc = MLP([obs_dim, hidden, key_dim], rng)
        # opt: updates the encoder's weights (the DND is updated directly in _train, not by Adam).
        self.opt = Adam(self.enc.params(), lr)
        self.refresh_every, self.bonus_beta = refresh_every, bonus_beta
        # dnds[a]: the memory for action a.
        self.dnds = [DND(dnd_cap, key_dim, p, delta, obs_dim if refresh_every else None)
                     for _ in range(n_actions)]
        # nstep: holds the last N transitions until their N-step return is known (see returns.py).
        self.nstep = NStep(N, gamma)
        self.alpha, self.mem_lr, self.batch, self.train_every = alpha, mem_lr, batch, train_every
        # Replay buffer for the slow path, a circular array like DQN's. Unlike DQN it stores the finished
        # target (computed once, when the row is written) rather than the pieces needed to rebuild it.
        self.R_obs = np.zeros((replay_cap, obs_dim))  # observation in which the action was taken
        self.R_act = np.zeros(replay_cap, np.int64)   # action taken
        self.R_ret = np.zeros(replay_cap)             # its N-step Q target (the value written to the DND)
        # r_n: rows filled; r_i: next row to write (wraps around); r_cap: size.
        self.r_n, self.r_i, self.r_cap = 0, 0, replay_cap
        # self._h (set in act): embedding of the current observation. observe() pairs it with the action
        # and later writes it to the DND as the key, so it is computed only once.

    def _embed(self, obs):
        """One observation -> its embedding h."""
        return self.enc.forward(obs[None])[0]

    def _q(self, h, t=None):
        """Q(s, a) for every action a, given the embedding h of s: one lookup per action's memory."""
        return np.array([d.lookup(h[None], t)[0] for d in self.dnds])

    def act(self, obs, eps, t):
        h = self._embed(obs)
        self._h = h
        # Epsilon-greedy, as in DQN.
        if self.rng.random() < eps:
            return int(self.rng.integers(self.nA))
        if self.bonus_beta:
            # density bonus (not in the paper): favour actions whose memory is sparse near h.
            # Used for acting only; value targets use the plain Q.
            qd = [d.lookup(h[None], t, return_dist=True) for d in self.dnds]
            q = np.array([x[0][0] for x in qd])   # Q per action
            dm = np.array([x[1][0] for x in qd])  # mean distance to neighbours per action
            if np.isinf(dm).any():  # an empty DND: try it
                return int(self.rng.choice(np.flatnonzero(np.isinf(dm))))
            # Bonus >= 0, zero for the action with the densest memory near h, larger the sparser it is.
            q = q + self.bonus_beta * np.log((dm + 1e-8) / (dm.min() + 1e-8))
        else:
            q = self._q(h, t)
        return int(self.rng.choice(np.flatnonzero(q == q.max())))  # best action, ties broken randomly

    def observe(self, obs, a, r, next_obs, term, trunc, t):
        # For every transition whose N-step return is now complete:
        #   o: observation, h: its embedding when the action was chosen, act: the action,
        #   G: discounted sum of the next N rewards, boot: observation N steps later,
        #   disc: gamma^N, or 0 if the episode ended in between.
        for (o, h, act), G, boot, disc in self.nstep.push((obs, self._h, a), r, next_obs, term, trunc):
            # Q target: real rewards + discounted best Q from the memory at the bootstrap state.
            target = G + (disc * self._q(self._embed(boot)).max() if disc > 0 else 0.0)  # eq. 3
            # Fast path: write it into the memory of the action taken; usable on the very next lookup.
            self.dnds[act].write(h, target, t, self.alpha, obs=o)
            # And keep it in replay for the slow path.
            self.R_obs[self.r_i], self.R_act[self.r_i], self.R_ret[self.r_i] = o, act, target
            self.r_i = (self.r_i + 1) % self.r_cap
            self.r_n = min(self.r_n + 1, self.r_cap)
        if self.learn and t % self.train_every == 0 and self.r_n >= self.batch:
            self._train()
        # Key drift: training changes the encoder, so keys written long ago were computed by an older
        # encoder and no longer match how the same situation is embedded now. Refresh recomputes them.
        if self.refresh_every and t % self.refresh_every == 0:
            for d in self.dnds:
                for lo in range(0, d.n, 1024):  # chunked: CNN im2col buffers are large
                    hi = min(lo + 1024, d.n)
                    d.keys[lo:hi] = self.enc.forward(d.obs[lo:hi])

    def probe_embed(self, X):
        """Keys and Q-values for a batch of observations (probe.py). Read-only: no LRU marks, no rng draws."""
        Z = np.concatenate([self.enc.forward(X[lo:lo + 256]) for lo in range(0, len(X), 256)])
        Q = np.stack([d.lookup(Z) for d in self.dnds], 1)
        return Z, Q

    def diagnostics(self):
        """Per action: write counters, rows in use, mean and largest absolute stored value."""
        out = {}
        for a, d in enumerate(self.dnds):
            v = d.vals[: d.n]
            out[a] = dict(d.stats, n=int(d.n), val_mean=float(v.mean()) if d.n else 0.0,
                          val_absmax=float(np.abs(v).max()) if d.n else 0.0)
        return out

    def _train(self):
        """Slow path: one gradient step on L = 1/2 (Q(s,a) - target)^2, averaged over a minibatch.

        Q(s,a) depends on the embedding h (through the distances) and on the neighbours' keys and
        values, so the gradient is used three ways: the encoder via Adam, and the stored keys and
        values via plain gradient descent with rate mem_lr. All derivatives are written out by hand.
        """
        idx = self.rng.integers(self.r_n, size=self.batch)
        X, A, R = self.R_obs[idx], self.R_act[idx], self.R_ret[idx]  # observations, actions, targets
        H = self.enc.forward(X)  # (batch, key_dim) embeddings
        gH = np.zeros_like(H)    # dL/dH, filled in per action below
        B = self.batch
        for a, dnd in enumerate(self.dnds):
            m = A == a  # the rows of the minibatch where action a was taken
            if not m.any() or dnd.n == 0:
                continue
            Hm = H[m]
            # Recompute Q exactly as lookup() does, keeping the intermediates for the chain rule.
            # Shapes: nb, d, kern, w, V are (rows, p); S, Q are per row.
            nb, d = dnd.knn(Hm)              # neighbour row numbers, squared distances d_i
            kern = 1.0 / (d + dnd.delta)     # k_i
            S = kern.sum(1, keepdims=True)   # sum_j k_j
            w = kern / S                     # w_i
            V = dnd.vals[nb]                 # v_i
            Q = (w * V).sum(1)
            g = (Q - R[m]) / B                              # dL/dQ
            # Chain rule through the distances: dQ/dk_i = (v_i - Q) / S, and dk_i/dd_i = -k_i^2.
            dL_dd = g[:, None] * (V - Q[:, None]) / S * (-kern ** 2)   # dL/d(dist_i)
            diff = Hm[:, None, :] - dnd.keys[nb]            # h - h_i
            # d_i = ||h - h_i||^2, so dd_i/dh = 2 (h - h_i); sum over the p neighbours.
            gH[m] = 2.0 * (dL_dd[:, :, None] * diff).sum(1)  # dL/dh
            # slow SGD path into memory: values (dQ/dv_i = w_i) and keys (dd/dh_i = -2(h - h_i))
            # np.add.at (not +=) because the same row can be a neighbour of several queries; it sums them.
            np.add.at(dnd.vals, nb.ravel(), -self.mem_lr * (g[:, None] * w).ravel())
            gK = -2.0 * dL_dd[:, :, None] * diff
            np.add.at(dnd.keys, nb.ravel(), -self.mem_lr * gK.reshape(-1, gK.shape[-1]))
        # Backpropagate dL/dH through the encoder (H is still its last forward output) and step Adam.
        self.opt.step(self.enc.backward(gH))
