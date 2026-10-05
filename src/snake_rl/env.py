"""Minimal Snake environment with grid observations.

Observation: 3 binary channels (head, body, food) over an n x n grid, flattened.
Actions (relative): 0 = straight, 1 = turn right, 2 = turn left.
Reward: +1 food, -1 death, 0 otherwise.
Distractors: `distractors` extra channels of i.i.d. Bernoulli(noise_p) pixels, resampled
every step. They carry no information about value, but dominate raw-space distances,
so nearest neighbours in observation space become largely random.
Truncation: episode is cut if the snake goes `max_idle` steps without eating
(prevents endless loops; the value target bootstraps through truncation).
"""
from collections import deque
import numpy as np


class Snake:
    DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left

    def __init__(self, size=7, max_idle=None, seed=0, distractors=0, noise_p=0.5):
        self.n = size
        self.D, self.noise_p = distractors, noise_p
        self.rng = np.random.default_rng(seed)
        self.max_idle = max_idle or 2 * size * size
        self.obs_dim = (3 + distractors) * size * size
        self.n_actions = 3

    def reset(self):
        c = self.n // 2
        self.body = deque([(c, c), (c, c - 1)])  # body[0] is the head
        self.dir = 1
        self.idle = 0
        self.score = 0
        self._place_food()
        return self._obs()

    def _place_food(self):
        occ = set(self.body)
        free = [(r, c) for r in range(self.n) for c in range(self.n) if (r, c) not in occ]
        self.food = free[self.rng.integers(len(free))] if free else None

    def _obs(self):
        o = np.zeros((3, self.n, self.n), np.float32)
        hr, hc = self.body[0]
        o[0, hr, hc] = 1.0
        for (r, c) in list(self.body)[1:]:
            o[1, r, c] = 1.0
        if self.food is not None:
            o[2, self.food[0], self.food[1]] = 1.0
        if self.D:
            noise = (self.rng.random((self.D, self.n, self.n)) < self.noise_p).astype(np.float32)
            o = np.concatenate([o, noise])
        return o.ravel()

    def step(self, a):
        self.dir = (self.dir + (0, 1, -1)[a]) % 4
        dr, dc = self.DIRS[self.dir]
        hr, hc = self.body[0]
        nh = (hr + dr, hc + dc)
        self.idle += 1
        out = not (0 <= nh[0] < self.n and 0 <= nh[1] < self.n)
        if out or nh in list(self.body)[:-1]:  # tail cell is vacated this step
            return self._obs(), -1.0, True, False
        self.body.appendleft(nh)
        if nh == self.food:
            self.idle, self.score = 0, self.score + 1
            self._place_food()
            if self.food is None:  # board filled
                return self._obs(), 1.0, True, False
            return self._obs(), 1.0, False, False
        self.body.pop()
        return self._obs(), 0.0, False, self.idle >= self.max_idle
