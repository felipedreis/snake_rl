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

    def last_hidden(self):
        """Activations of the last hidden layer (the input to the output layer) from the last forward()."""
        return self.cache[-2]

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


# DQN's conv shape scaled to the grid (see ConvNet): 25x25 -> 25 -> 13 -> 7, then FC 512.
DQN_CONVS, DQN_FC = [(32, 3, 1), (64, 3, 2), (64, 3, 2)], 512
# DQN's own network (Mnih et al., 2015) on 84x84 frames, valid convolutions (padding 0): 84 -> 20 -> 9 -> 7, then FC 512.
NATURE_CONVS, NATURE_FC = [(32, 8, 4, 0), (64, 4, 2, 0), (64, 3, 1, 0)], 512


class ConvNet:
    """Small CNN with manual backprop, same interface as MLP (flat batch in, params()/backward()).

    obs_shape = (C, n, n). `convs` lists the conv layers, each either an int (output channels; k x k kernel, stride 1)
    or a tuple (channels, kernel, stride[, padding]). Padding defaults to k//2 (zeros) and every conv has a ReLU, so
    stride 1 keeps the map size ("same") and stride 2 roughly halves it; padding 0 gives "valid" convolutions.
    The flattened feature map feeds the dense layers `fcs` (ReLU between, linear last).
    [16, 32] is the original small net: stride 1 throughout, the map stays n x n. [(32,3,1), (64,3,2), (64,3,2)] with
    fcs [512, ...] is DQN's shape (Mnih et al., 2015: 32 8x8/4, 64 4x4/2, 64 3x3/1, FC 512) scaled to the grid:
    25x25 -> 25 -> 13 -> 7, the same 7x7 map DQN's convolutions leave on Atari. NATURE_CONVS is DQN's exact stack,
    for the 84x84 pixel render: [(32,8,4,0), (64,4,2,0), (64,3,1,0)], 84 -> 20 -> 9 -> 7.
    Convolutions are im2col matmuls, channels-last internally.
    """

    def __init__(self, obs_shape, convs, fcs, rng, k=3):
        C, n, _ = obs_shape
        self.shape, self.k = (n, n, C), k
        self.W, self.b, self.layers, self.pads, cin, h = [], [], [], [], C, n
        for spec in convs:
            cout, kk, st, *pd = (spec, k, 1) if isinstance(spec, int) else spec
            pad = pd[0] if pd else kk // 2
            ho = (h + 2 * pad - kk) // st + 1
            if ho < 1:
                raise ValueError(f"conv {kk}x{kk}/{st} does not fit a {h}x{h} map")
            self.layers.append((kk, st, h, ho))  # kernel, stride, input size, output size
            self.pads.append(pad)
            self.W.append(rng.normal(0, np.sqrt(2.0 / (kk * kk * cin)), (kk * kk * cin, cout)))
            self.b.append(np.zeros(cout))
            cin, h = cout, ho
        self.n_conv = len(convs)
        self.out_hw = h
        sizes = [cin * h * h] + list(fcs)
        for i, o in zip(sizes[:-1], sizes[1:]):
            self.W.append(rng.normal(0, np.sqrt(2.0 / i), (i, o)))
            self.b.append(np.zeros(o))

    def params(self):
        return self.W + self.b

    def _cols(self, x, k, s, ho, p):
        """im2col: every k x k patch (stride s, zero padding p) of x as one row. x: (B, H, W, C) -> (B*ho*ho, k*k*C)."""
        B, _, _, C = x.shape
        xp = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)))
        cols = np.empty((B, ho, ho, k, k, C))
        for i in range(k):
            for j in range(k):
                cols[:, :, :, i, j, :] = xp[:, i:i + s * ho:s, j:j + s * ho:s, :]
        return cols.reshape(B * ho * ho, k * k * C)

    def forward(self, x):
        n, _, C = self.shape
        B = x.shape[0]
        h = x.reshape(B, C, n, n).transpose(0, 2, 3, 1)  # channels last
        self.cache, self.cols = [], []
        L = len(self.W)
        for l, (k, s, _, ho) in enumerate(self.layers):
            cols = self._cols(h, k, s, ho, self.pads[l])
            self.cols.append(cols)
            self.cache.append(h)
            h = np.maximum(cols @ self.W[l] + self.b[l], 0.0).reshape(B, ho, ho, -1)
        h = h.reshape(B, -1)
        for l in range(self.n_conv, L):
            self.cache.append(h)
            h = h @ self.W[l] + self.b[l]
            if l < L - 1:
                h = np.maximum(h, 0.0)
        self.out = h
        return h

    def last_hidden(self):
        """Activations of the last hidden layer (the input to the output layer) from the last forward()."""
        return self.cache[-1]

    def backward(self, g):
        """g = dL/d(output) for the last forward() call. Returns grads in params() order."""
        L = len(self.W)
        B = g.shape[0]
        gW, gb = [None] * L, [None] * L
        for l in reversed(range(self.n_conv, L)):
            if l < L - 1:
                g = g * (self.cache[l + 1] > 0)
            gW[l] = self.cache[l].T @ g
            gb[l] = g.sum(0)
            g = g @ self.W[l].T
        g = g.reshape(B * self.out_hw * self.out_hw, -1)
        for l in reversed(range(self.n_conv)):
            k, s, hi, ho = self.layers[l]
            p = self.pads[l]
            post = self.cache[l + 1] if l + 1 < self.n_conv else self.cache[self.n_conv].reshape(B, ho, ho, -1)
            g = g * (post.reshape(B * ho * ho, -1) > 0)
            gW[l] = self.cols[l].T @ g
            gb[l] = g.sum(0)
            if l == 0:
                break  # no gradient w.r.t. the observation is needed
            # col2im: scatter each patch's gradient back onto the (padded) input it was read from.
            gc = (g @ self.W[l].T).reshape(B, ho, ho, k, k, -1)
            gx = np.zeros((B, hi + 2 * p, hi + 2 * p, gc.shape[-1]))
            for i in range(k):
                for j in range(k):
                    gx[:, i:i + s * ho:s, j:j + s * ho:s, :] += gc[:, :, :, i, j, :]
            g = gx[:, p:p + hi, p:p + hi, :].reshape(B * hi * hi, -1)
        return gW + gb

    def copy_from(self, other):
        for p, q in zip(self.params(), other.params()):
            p[...] = q

    def __getstate__(self):  # the im2col buffers of the last forward() are large and only needed by backward()
        return {k: v for k, v in self.__dict__.items() if k not in ("cache", "cols", "out")}

    def __setstate__(self, d):
        self.__dict__.update(d)
        if "pads" not in d:  # checkpoints from before explicit padding: every conv was "same"
            self.pads = [k // 2 for k, *_ in self.layers]


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
        step2 = par2 = 0.0
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            g = g * scale
            m[...] = self.b1 * m + (1 - self.b1) * g
            v[...] = self.b2 * v + (1 - self.b2) * g * g
            # m and v start at 0, so early on they underestimate; dividing by (1 - b^t) corrects that.
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            u = self.lr * mh / (np.sqrt(vh) + self.eps)
            step2 += (u ** 2).sum()
            par2 += (p ** 2).sum()
            p -= u
        # Telemetry for this step: gradient norm before clipping, whether clipping kicked in, and how big the step
        # was relative to the weights (~1e-3 is the usual healthy range; much larger = thrashing, much smaller = stalled).
        self.last = dict(grad_norm=float(norm), clipped=float(scale < 1.0),
                         update_ratio=float(np.sqrt(step2) / (np.sqrt(par2) + 1e-12)), param_norm=float(np.sqrt(par2)))
