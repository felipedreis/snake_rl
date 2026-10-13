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
from snake_rl.agents.replay import Replay
from snake_rl.nn import Adam, make_net, make_optimizer
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
      encoder      "mlp" or "naturecnn" (DQN's exact network, for the 84x84 pixel render, *_naturecnn); obs_shape is the
                   (C, 84, 84) frame stack the CNN needs.
      obs_u8       keep replay observations as uint8 gray levels (k/255, what the pixel render produces) to save
                   memory; exact for that render, and off (float64 rows) otherwise.
    """

    def __init__(self, obs_dim, n_actions, rng, N=1, gamma=0.99, hidden=64, lr=5e-4,
                 replay_cap=50000, batch=32, train_every=4, target_every=1000, encoder="mlp", obs_shape=None,
                 obs_u8=False):
        # rng: the only source of randomness (exploration and minibatch sampling), for reproducibility.
        # nA: number of actions.
        self.rng, self.nA = rng, n_actions
        # encoder="naturecnn": DQN's ConvNet over the frame stack (obs_shape); "mlp": two hidden layers over the flat
        # observation.
        net = lambda: make_net(encoder, obs_dim, obs_shape, [hidden, hidden], n_actions, rng)
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
        # Replay buffer (replay.py), one row per transition:
        #   S  State: observation in which the action was taken      A  Action taken
        #   G  Gain: discounted sum of the next (up to) N rewards     B  Bootstrap observation: where we were N steps later
        #   D  Discount for the bootstrap: gamma^N, or 0 if the episode ended (dead snake: no future reward)
        self.replay = Replay(replay_cap, obs_dim, ("S", "B"), dict(A=np.int64, G=np.float64, D=np.float64), u8=obs_u8)
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
            self.replay.add(S=o, A=act, G=G, B=boot, D=disc)
        # Train only once there are 1000 rows, so the first minibatches aren't drawn from a handful
        # of near-identical transitions.
        if t % self.train_every == 0 and self.replay.n >= 1000:
            self._train()
        if t % self.target_every == 0:
            self.qt.copy_from(self.q)

    def set_optimizer(self, kind="adam", lr=None):
        """Swap the optimizer ('adam' or 'rmsprop', see nn.py) and/or its learning rate; call before training starts."""
        self.opt = make_optimizer(kind, self.q.params(), self.opt.lr if lr is None else lr)

    def probe_embed(self, X):
        """Last hidden layer and Q-values for a batch of observations (probe.py). Read-only."""
        Z, Q = [], []
        for lo in range(0, len(X), 256):  # chunked: CNN im2col buffers are large
            Q.append(self.q.forward(X[lo:lo + 256]))
            Z.append(self.q.last_hidden())
        return np.concatenate(Z), np.concatenate(Q)

    def _train(self):
        """One gradient step on a random minibatch of replay rows."""
        b = self.replay.sample(self.rng, self.batch)  # random rows (with replacement)
        # Targets y, from the frozen target network: G + gamma^N * max_a Qt(boot_obs, a).
        y = b["G"] + b["D"] * self.qt.forward(b["B"]).max(1)
        Q = self.q.forward(b["S"])                    # (batch, nA) current estimates
        # Only the action actually taken has a target, so only its output gets an error signal.
        err = Q[np.arange(self.batch), b["A"]] - y
        g = np.zeros_like(Q)  # dL/dQ: zero for the actions not taken
        # Huber loss = squared error for |err| <= 1, absolute error beyond. Its gradient is err clipped
        # to [-1, 1], so a single surprising target can't produce a huge update. /batch = mean over rows.
        g[np.arange(self.batch), b["A"]] = np.clip(err, -1, 1) / self.batch  # Huber grad
        ae, h = np.abs(err), self.q.last_hidden()
        self.opt.step(self.q.backward(g))
        self.stats.add(loss=np.where(ae <= 1, 0.5 * err ** 2, ae - 0.5).mean(), td_abs=ae.mean(),
                       td_clipped=(ae > 1).mean(), q_taken=(y + err).mean(), q_max=Q.max(1).mean(), target=y.mean(),
                       dead_relu=(h.max(0) <= 0).mean(), **self.opt.last)

    def __getstate__(self):
        """Pickle without the replay buffer (GBs on big boards): enough to act, e.g. for snake-watch, not to resume."""
        return {k: (None if k == "replay" else v) for k, v in self.__dict__.items()}

    def train_stats(self):
        """Window means of the training telemetry since the last call (see telemetry.py)."""
        return self.stats.snapshot()
