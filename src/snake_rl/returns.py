"""N-step return accumulator shared by all agents."""
from collections import deque


class NStep:
    """Turns a stream of (payload, reward) into N-step targets.

    Each emitted item is (payload, G, boot_obs, disc) where the full target is
    G + disc * max_a Q(boot_obs, a). disc = 0 at a true terminal state.
    Emission happens N steps after the transition (as in the paper), or at episode end.
    """

    def __init__(self, N, gamma):
        self.N, self.gamma = N, gamma
        self.items = deque()

    def _ret(self, start):
        G, g = 0.0, 1.0
        for k in range(start, len(self.items)):
            G += g * self.items[k][1]
            g *= self.gamma
        return G

    def push(self, payload, r, next_obs, terminal, truncated):
        self.items.append((payload, r))
        out = []
        if terminal or truncated:
            n = len(self.items)
            for i in range(n):
                disc = 0.0 if terminal else self.gamma ** (n - i)
                out.append((self.items[i][0], self._ret(i), next_obs, disc))
            self.items.clear()
        elif len(self.items) == self.N:
            out.append((self.items[0][0], self._ret(0), next_obs, self.gamma ** self.N))
            self.items.popleft()
        return out
