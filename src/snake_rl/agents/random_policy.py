"""Reference floor: picks every action uniformly at random and learns nothing."""


class RandomAgent:
    def __init__(self, n_actions, rng):
        self.nA, self.rng = n_actions, rng

    def act(self, obs, eps, t):
        return int(self.rng.integers(self.nA))

    def observe(self, obs, a, r, next_obs, term, trunc, t):
        pass
