"""A NumPy MLP with manual backprop, and Adam with global-norm clipping.

Vocabulary used throughout the project:
  parameters  the numbers the network learns: weight matrices W and bias vectors b.
  forward     compute the output from an input.
  loss L      one number saying how wrong the output was (lower is better).
  gradient    dL/dp for every parameter p: how L changes if p is nudged up a little.
  backward    compute those gradients with the chain rule ("backpropagation"),
              walking the layers from the output back to the input.
  optimizer   uses the gradients to move every parameter a small step downhill on L.
Shapes: every array is batched, i.e. the first axis is the batch (one row per example).
"""
import numpy as np


class MLP:
    """Fully connected net, ReLU hidden layers, linear output.

    sizes = [input_dim, hidden_1, ..., output_dim]. E.g. [147, 64, 64, 3] maps a 147-number
    observation through two hidden layers of 64 units to 3 outputs.
    Layer l computes h_{l+1} = relu(h_l @ W[l] + b[l]); the last layer skips the relu so the
    output can be any real number (needed for Q-values and embeddings).
    """

    def __init__(self, sizes, rng):
        # W[l]: (in, out) weight matrix of layer l. Random init with std sqrt(2/in) ("He init"),
        # which keeps activations from shrinking or exploding as they pass through ReLU layers.
        self.W = [rng.normal(0, np.sqrt(2.0 / i), (i, o)) for i, o in zip(sizes[:-1], sizes[1:])]
        # b[l]: (out,) bias vector of layer l, starts at zero.
        self.b = [np.zeros(o) for o in sizes[1:]]

    def params(self):
        """All learnable arrays, in a fixed order that backward() and Adam rely on."""
        return self.W + self.b

    def forward(self, x):
        """x: (batch, input_dim) -> (batch, output_dim)."""
        # cache[l] = input to layer l (cache[0] = x, cache[-1] = output). backward() needs these,
        # which is why a second forward() before backward() would silently corrupt the gradients.
        self.cache = [x]
        h = x
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if l < len(self.W) - 1:
                h = np.maximum(h, 0.0)  # ReLU on hidden layers only
            self.cache.append(h)
        return h

    def backward(self, g):
        """g = dL/d(output) for the last forward() call. Returns grads in params() order."""
        L = len(self.W)
        gW, gb = [None] * L, [None] * L
        for l in reversed(range(L)):
            # g is now dL/d(output of layer l). Undo the ReLU: its derivative is 1 where the
            # unit was active (> 0) and 0 where it was clipped.
            if l < L - 1:
                g = g * (self.cache[l + 1] > 0)
            # out = in @ W + b, so dL/dW = in^T @ g and dL/db = g summed over the batch.
            gW[l] = self.cache[l].T @ g
            gb[l] = g.sum(0)
            # Pass the gradient on to the layer below: dL/d(in) = g @ W^T.
            g = g @ self.W[l].T
        return gW + gb

    def copy_from(self, other):
        """Overwrite this net's parameters with another net's (in place, same shapes)."""
        for p, q in zip(self.params(), other.params()):
            p[...] = q


class Adam:
    """Adam optimizer (Kingma & Ba, 2015): gradient descent with a per-parameter step size.

    Plain gradient descent does p -= lr * grad. Adam instead keeps, for every parameter,
    a running average of the gradient (m, "momentum": smooths out noisy minibatch gradients)
    and of the squared gradient (v: how large gradients for that parameter usually are),
    and steps by lr * m / sqrt(v). Parameters with consistently large gradients thus take
    proportionally smaller steps, so one learning rate works for all of them.
    """

    def __init__(self, params, lr, clip=10.0, b1=0.9, b2=0.999, eps=1e-8):
        # p: the arrays to update (updated in place, so the network sees the change).
        # lr: learning rate (step size). clip: max global gradient norm (see step()).
        # b1, b2: decay rates of the two running averages. eps: avoids division by zero.
        self.p, self.lr, self.clip, self.b1, self.b2, self.eps = params, lr, clip, b1, b2, eps
        self.m = [np.zeros_like(x) for x in params]  # running mean of gradients, per parameter
        self.v = [np.zeros_like(x) for x in params]  # running mean of squared gradients
        self.t = 0                                   # number of steps taken so far

    def step(self, grads):
        # Global-norm clipping: if all gradients together are longer than `clip`, shrink them
        # all by the same factor. Protects against one bad minibatch causing a huge update.
        norm = np.sqrt(sum((g ** 2).sum() for g in grads))
        scale = min(1.0, self.clip / (norm + 1e-12))
        self.t += 1
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            g = g * scale
            m[...] = self.b1 * m + (1 - self.b1) * g
            v[...] = self.b2 * v + (1 - self.b2) * g * g
            # m and v start at 0, so early on they underestimate; dividing by (1 - b^t) corrects that.
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)
