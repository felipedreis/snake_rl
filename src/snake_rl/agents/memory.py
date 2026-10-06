"""Fixed-capacity key/value store with exact kNN and LRU eviction, shared by the episodic-memory agents."""
import numpy as np


class KeyValueMemory:
    """Storage and brute-force p-NN. Subclasses define their own lookup and write rules."""

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

    def _insert(self, h, v, t):
        """Append (h, v), or overwrite the least recently used row when full. Returns the row index."""
        if self.n < self.cap:
            i = self.n
            self.n += 1
            self.stats["appends"] += 1
        else:
            i = int(np.argmin(self.last_used))
            self.stats["evictions"] += 1
        self.keys[i], self.vals[i], self.last_used[i] = h, v, t
        return i
