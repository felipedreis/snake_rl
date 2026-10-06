"""Model-Free Episodic Control (Blundell et al., 2016) in NumPy.

The idea: remember, for every (state, action) seen, the best discounted return ever obtained after
taking that action there, and act greedily on those memories. Nothing is trained by gradient
descent and nothing bootstraps; values are full Monte Carlo returns from finished episodes.
  - phi: a fixed random projection obs -> key, phi(x) = A x with A_ij ~ N(0, 1) (Sec. 3). Distances
    are roughly preserved (Johnson-Lindenstrauss), so this is "nearest neighbours in raw pixel space".
  - Q^EC: one buffer of (key, value) rows per action (footnote 2).
  - Estimate (eq. 2): the stored value on an exact key match, otherwise the plain (unweighted)
    mean of the values of the k nearest keys.
  - Update (eq. 1, Algorithm 1 lines 9-11): at the end of each episode, walk it backwards and set
    Q^EC(s_t, a_t) <- R_t if (s_t, a_t) is new, else max(Q^EC(s_t, a_t), R_t).
    The max makes values optimistic ("highest potential return"), not expected returns.
  - When a buffer is full, the least recently used row is evicted (used = written or a neighbour).

Differences from NEC (nec.py), besides the frozen random encoder:
  uniform kNN average instead of the 1/(d + delta) kernel; max update instead of a running average;
  Monte Carlo returns written at episode end instead of N-step bootstrapped targets written as they
  complete. A truncated episode is treated like any other episode end: its tail is not
  bootstrapped (pure MC, as in the paper), unlike every other agent here.
"""
import numpy as np

from snake_rl.agents.nec import DND


class ECBuffer(DND):
    """Q^EC for one action: DND storage and exact kNN, with MFEC's lookup and write rules."""

    def __init__(self, capacity, dim, k):
        super().__init__(capacity, dim, k, delta=0.0)
        self.stats["lookups"] = self.stats["exact_hits"] = 0

    def _match(self, h, match_eps):
        """Row holding (practically) the key h, or None. Also returns the neighbour rows of h."""
        idx, d = self.knn(h[None])
        j = int(np.argmin(d[0]))
        return (idx[0][j] if d[0][j] < match_eps else None), idx[0]

    def lookup(self, h, t, match_eps=1e-6):
        """Eq. 2 for one key h. An empty buffer gives 0."""
        if self.n == 0:
            return 0.0
        j, nb = self._match(h, match_eps)
        self.stats["lookups"] += 1
        if j is not None:
            self.stats["exact_hits"] += 1
            self.last_used[j] = t
            return float(self.vals[j])
        self.last_used[nb] = t
        return float(self.vals[nb].mean())

    def write(self, h, R, t, match_eps=1e-6):
        """Eq. 1: keep the best return seen for key h, appending (or evicting LRU) if it is new."""
        if self.n > 0:
            j, _ = self._match(h, match_eps)
            if j is not None:
                self.vals[j] = max(self.vals[j], R)
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
        self.keys[i], self.vals[i], self.last_used[i] = h, R, t


class MFECAgent:
    """
    Hyperparameters:
      key_dim   length of the random projection phi(obs) (paper: 64).
      k         neighbours averaged for an unseen state (paper: 11 on Atari, 50 on Labyrinth).
      cap       max rows per action's buffer (paper: 1e6 / 1e5; 2e4 here, as NEC's DND).
      gamma     discount for the Monte Carlo returns (paper: 1 on Atari, 0.99 on Labyrinth).
    """

    def __init__(self, obs_dim, n_actions, rng, key_dim=64, k=11, cap=20000, gamma=0.99):
        # rng: the only source of randomness. nA: number of actions.
        self.rng, self.nA, self.gamma = rng, n_actions, gamma
        # A: the projection matrix of phi, drawn once and never changed.
        self.A = rng.standard_normal((obs_dim, key_dim))
        # bufs[a]: Q^EC for action a.
        self.bufs = [ECBuffer(cap, key_dim, k) for _ in range(n_actions)]
        # episode: (key, action, reward) for every step of the current episode, written back at its end.
        self.episode = []
        # self._h (set in act): key of the current observation, paired with the action in observe().

    def act(self, obs, eps, t):
        h = obs @ self.A
        self._h = h
        # Epsilon-greedy (paper: eps = 0.005; here run.py's shared schedule).
        if self.rng.random() < eps:
            return int(self.rng.integers(self.nA))
        q = np.array([b.lookup(h, t) for b in self.bufs])
        return int(self.rng.choice(np.flatnonzero(q == q.max())))  # best action, ties broken randomly

    def observe(self, obs, a, r, next_obs, term, trunc, t):
        self.episode.append((self._h, a, r))
        if not (term or trunc):
            return
        # Backward replay: R_t = r_{t+1} + gamma R_{t+1}, written from the last step to the first.
        R = 0.0
        for h, act, rew in reversed(self.episode):
            R = rew + self.gamma * R
            self.bufs[act].write(h, R, t)
        self.episode = []

    def diagnostics(self):
        """Per action: write and lookup counters, rows in use, mean and largest absolute stored value."""
        out = {}
        for a, b in enumerate(self.bufs):
            v = b.vals[: b.n]
            out[a] = dict(b.stats, n=int(b.n), val_mean=float(v.mean()) if b.n else 0.0,
                          val_absmax=float(np.abs(v).max()) if b.n else 0.0)
        return out
