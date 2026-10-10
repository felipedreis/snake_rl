"""DQN baseline (Mnih et al., 2015) with optional N-step targets, same MLP budget as NEC.

The idea: a neural network Q maps an observation s to one number per action, Q(s, a) =
"how much total discounted reward do I expect if I take action a now and play well afterwards".
  Acting:   take the action with the highest Q (or a random one with probability eps, to explore).
  Learning: after seeing what really happened, nudge Q(s, a) towards a better estimate, the target
            y = (rewards actually received over the next N steps, discounted)
                + gamma^N * (the network's own estimate of the best Q from where we ended up).
            Using the network's own estimate inside its target is called bootstrapping.

Target: G + disc * max_a Q_target(boot_obs, a), Huber loss, periodic target sync.
N=1 gives standard DQN; N>1 isolates the contribution of N-step returns.
"""
import numpy as np
from snake_rl.nn import MLP, ConvNet, Adam, make_optimizer, DQN_CONVS, DQN_FC, NATURE_CONVS, NATURE_FC
from snake_rl.returns import NStep
from snake_rl.telemetry import TrainStats


class DQNAgent:
    """
    Hyperparameters:
      N            how many real rewards go into each target before bootstrapping (1 = classic DQN).
      gamma        discount factor: a reward k steps in the future is worth gamma^k now.
      hidden       units per hidden layer of the Q-network.
      lr           Adam learning rate.
      replay_cap   how many past transitions the replay buffer remembers.
      batch        transitions per training step (minibatch size).
      train_every  do one training step every this many env steps.
      target_every copy the online network into the target network every this many env steps.
      encoder      "mlp", "cnn" (small stride-1 CNN, *_cnn), "dqncnn" (DQN-shaped CNN, *_dqncnn) or "naturecnn" (DQN's
                   exact network, for the 84x84 pixel render, *_naturecnn); obs_shape is the (C, n, n) grid the CNNs need.
      obs_u8       keep replay observations as uint8 gray levels (k/255, what the pixel render produces) to save
                   memory; exact for that render, and off (float64 rows) otherwise.
    """

    def __init__(self, obs_dim, n_actions, rng, N=1, gamma=0.99, hidden=64, lr=5e-4,
                 replay_cap=50000, batch=32, train_every=4, target_every=1000, encoder="mlp", obs_shape=None,
                 obs_u8=False):
        # rng: the only source of randomness (exploration and minibatch sampling), for reproducibility.
        # nA: number of actions.
        self.rng, self.nA = rng, n_actions
        # encoder="cnn" uses a small ConvNet over the (C, n, n) grid (obs_shape); otherwise an MLP over
        # the flat observation.
        if encoder == "cnn":
            net = lambda: ConvNet(obs_shape, [16, 32], [hidden, n_actions], rng)
        elif encoder == "dqncnn":  # DQN's conv shape scaled to the grid, FC 512 (see nn.ConvNet)
            net = lambda: ConvNet(obs_shape, DQN_CONVS, [DQN_FC, n_actions], rng)
        elif encoder == "naturecnn":  # DQN's own network on 84x84 frames
            net = lambda: ConvNet(obs_shape, NATURE_CONVS, [NATURE_FC, n_actions], rng)
        else:
            net = lambda: MLP([obs_dim, hidden, hidden, n_actions], rng)
        # q: the "online" Q-network, observation -> one Q-value per action. This is what we train.
        # qt: the "target" network, a lagged copy of q used only to compute targets. If targets came
        # from q itself, every update would also move the target it is chasing, which tends to
        # oscillate or diverge. Freezing it between syncs keeps the target still for a while.
        self.q, self.qt = net(), net()
        self.qt.copy_from(self.q)
        # opt: the optimizer that updates q's weights from gradients.
        self.opt = Adam(self.q.params(), lr)
        # nstep: holds the last N transitions until their N-step return is known (see returns.py).
        self.nstep = NStep(N, gamma)
        self.batch, self.train_every, self.target_every = batch, train_every, target_every
        # Replay buffer: a circular array of past transitions; row j is one training example.
        # Training on random old rows instead of only the latest step breaks the strong correlation
        # between consecutive steps and lets every experience be learned from many times.
        self.u8 = obs_u8
        odt = np.uint8 if obs_u8 else np.float64
        self.S = np.zeros((replay_cap, obs_dim), odt)  # State: observation in which the action was taken
        self.A = np.zeros(replay_cap, np.int64)   # Action taken
        self.G = np.zeros(replay_cap)             # Gain: discounted sum of the next (up to) N rewards
        self.B = np.zeros((replay_cap, obs_dim), odt)  # Bootstrap observation: where we were N steps later
        self.D = np.zeros(replay_cap)             # Discount for the bootstrap: gamma^N, or 0 if the
                                                  #   episode ended (dead snake: no future reward)
        # n: rows filled so far; i: next row to write (wraps around, overwriting the oldest); cap: size.
        self.n, self.i, self.cap = 0, 0, replay_cap
        self.stats = TrainStats()  # per-update training telemetry, read by run.py via train_stats()

    def act(self, obs, eps, t):
        # Epsilon-greedy: explore with probability eps, otherwise exploit the current Q estimate.
        if self.rng.random() < eps:
            return int(self.rng.integers(self.nA))
        q = self.q.forward(obs[None])[0]  # obs[None] makes a batch of 1; [0] takes it back out
        return int(self.rng.choice(np.flatnonzero(q == q.max())))  # best action, ties broken randomly

    def observe(self, obs, a, r, next_obs, term, trunc, t):
        # NStep returns the transitions whose N-step return is now complete (none, one, or at episode
        # end all the pending ones). Store each as a replay row.
        for (o, act), G, boot, disc in self.nstep.push((obs, a), r, next_obs, term, trunc):
            j = self.i
            self.S[j], self.A[j], self.G[j], self.B[j], self.D[j] = self._put(o), act, G, self._put(boot), disc
            self.i = (self.i + 1) % self.cap
            self.n = min(self.n + 1, self.cap)
        # Train only once there are 1000 rows, so the first minibatches aren't drawn from a handful
        # of near-identical transitions.
        if t % self.train_every == 0 and self.n >= 1000:
            self._train()
        if t % self.target_every == 0:
            self.qt.copy_from(self.q)

    def set_optimizer(self, kind="adam", lr=None):
        """Swap the optimizer ('adam' or 'rmsprop', see nn.py) and/or its learning rate; call before training starts."""
        self.opt = make_optimizer(kind, self.q.params(), self.opt.lr if lr is None else lr)

    def _put(self, o):
        """Observation -> replay row (uint8 gray levels when obs_u8, else unchanged)."""
        return np.rint(o * 255) if self.u8 else o

    def _get(self, rows):
        """Replay rows -> network input; the inverse of _put (exact for the pixel render's k/255 values)."""
        return rows.astype(np.float32) / np.float32(255) if self.u8 else rows

    def probe_embed(self, X):
        """Last hidden layer and Q-values for a batch of observations (probe.py). Read-only."""
        Z, Q = [], []
        for lo in range(0, len(X), 256):  # chunked: CNN im2col buffers are large
            Q.append(self.q.forward(X[lo:lo + 256]))
            Z.append(self.q.last_hidden())
        return np.concatenate(Z), np.concatenate(Q)

    def _train(self):
        """One gradient step on a random minibatch of replay rows."""
        idx = self.rng.integers(self.n, size=self.batch)  # random row numbers (with replacement)
        # Targets y, from the frozen target network: G + gamma^N * max_a Qt(boot_obs, a).
        y = self.G[idx] + self.D[idx] * self.qt.forward(self._get(self.B[idx])).max(1)
        Q = self.q.forward(self._get(self.S[idx]))                    # (batch, nA) current estimates
        # Only the action actually taken has a target, so only its output gets an error signal.
        err = Q[np.arange(self.batch), self.A[idx]] - y
        g = np.zeros_like(Q)  # dL/dQ: zero for the actions not taken
        # Huber loss = squared error for |err| <= 1, absolute error beyond. Its gradient is err clipped
        # to [-1, 1], so a single surprising target can't produce a huge update. /batch = mean over rows.
        g[np.arange(self.batch), self.A[idx]] = np.clip(err, -1, 1) / self.batch  # Huber grad
        ae, h = np.abs(err), self.q.last_hidden()
        self.opt.step(self.q.backward(g))
        self.stats.add(loss=np.where(ae <= 1, 0.5 * err ** 2, ae - 0.5).mean(), td_abs=ae.mean(),
                       td_clipped=(ae > 1).mean(), q_taken=(y + err).mean(), q_max=Q.max(1).mean(), target=y.mean(),
                       dead_relu=(h.max(0) <= 0).mean(), **self.opt.last)

    def __getstate__(self):
        """Pickle without the replay buffer (GBs on big boards): enough to act, e.g. for snake-watch, not to resume."""
        return {k: (None if k in ("S", "A", "G", "B", "D") else v) for k, v in self.__dict__.items()}

    def train_stats(self):
        """Window means of the training telemetry since the last call (see telemetry.py)."""
        return self.stats.snapshot()
