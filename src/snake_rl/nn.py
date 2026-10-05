"""A NumPy MLP with manual backprop, and Adam with global-norm clipping."""
import numpy as np


class MLP:
    """Fully connected net, ReLU hidden layers, linear output."""

    def __init__(self, sizes, rng):
        self.W = [rng.normal(0, np.sqrt(2.0 / i), (i, o)) for i, o in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(o) for o in sizes[1:]]

    def params(self):
        return self.W + self.b

    def forward(self, x):
        self.cache = [x]
        h = x
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if l < len(self.W) - 1:
                h = np.maximum(h, 0.0)
            self.cache.append(h)
        return h

    def backward(self, g):
        """g = dL/d(output) for the last forward() call. Returns grads in params() order."""
        L = len(self.W)
        gW, gb = [None] * L, [None] * L
        for l in reversed(range(L)):
            if l < L - 1:
                g = g * (self.cache[l + 1] > 0)
            gW[l] = self.cache[l].T @ g
            gb[l] = g.sum(0)
            g = g @ self.W[l].T
        return gW + gb

    def copy_from(self, other):
        for p, q in zip(self.params(), other.params()):
            p[...] = q


class Adam:
    def __init__(self, params, lr, clip=10.0, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.clip, self.b1, self.b2, self.eps = params, lr, clip, b1, b2, eps
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]
        self.t = 0

    def step(self, grads):
        norm = np.sqrt(sum((g ** 2).sum() for g in grads))
        scale = min(1.0, self.clip / (norm + 1e-12))
        self.t += 1
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            g = g * scale
            m[...] = self.b1 * m + (1 - self.b1) * g
            v[...] = self.b2 * v + (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)
