"""Replay buffer shared by the gradient-trained agents (DQN and NEC's slow path).

A circular array of rows with named columns; once full, each new row overwrites the oldest. Training on random old rows
instead of only the latest step breaks the strong correlation between consecutive steps and lets every experience be
learned from many times.

Observation columns can be kept as uint8 gray levels (`u8=True`): the pixel render only produces values k/255, so
rint(x * 255) stores them exactly in an eighth of the memory, and sample() turns them back into the same float32 values.
"""
import numpy as np


class Replay:
    def __init__(self, cap, obs_dim, obs_fields, fields, u8=False):
        """obs_fields: names of the observation columns, each (cap, obs_dim). fields: {name: dtype} of the scalar
        columns. u8: store observations as uint8 gray levels (float64 rows otherwise)."""
        self.u8, self.obs_fields = u8, tuple(obs_fields)
        self.cols = {k: np.zeros((cap, obs_dim), np.uint8 if u8 else np.float64) for k in self.obs_fields}
        self.cols.update({k: np.zeros(cap, dt) for k, dt in fields.items()})
        # n: rows filled so far; i: next row to write (wraps around, overwriting the oldest); cap: size.
        self.n, self.i, self.cap = 0, 0, cap

    def add(self, **row):
        """Write one row (every column by name)."""
        for k, v in row.items():
            self.cols[k][self.i] = np.rint(v * 255) if self.u8 and k in self.obs_fields else v
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, rng, batch):
        """`batch` random rows (with replacement), drawn as rng.integers(n, size=batch). Returns {name: column rows},
        observations as network input (float32 k/255 when u8)."""
        idx = rng.integers(self.n, size=batch)
        return {k: (self._decode(c[idx]) if k in self.obs_fields else c[idx]) for k, c in self.cols.items()}

    def _decode(self, rows):
        return rows.astype(np.float32) / np.float32(255) if self.u8 else rows
