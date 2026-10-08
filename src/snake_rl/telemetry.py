"""Training telemetry: window averages of per-update scalars (loss, TD error, gradient norm, ...).

An agent calls `add(...)` once per gradient update; run.py calls `snapshot()` every K env steps, which returns the
means since the previous snapshot (plus how many updates they cover) and starts a new window. Pure bookkeeping: it
never draws random numbers or feeds back into learning, so training is bit-for-bit the same with it.
"""


class TrainStats:
    def __init__(self):
        self.sum, self.n, self.updates = {}, {}, 0

    def add(self, **scalars):
        """Record one update's scalars. Call once per update; each call counts as one update."""
        self.updates += 1
        for k, v in scalars.items():
            self.sum[k] = self.sum.get(k, 0.0) + float(v)
            self.n[k] = self.n.get(k, 0) + 1

    def snapshot(self):
        """Means since the last snapshot, plus `updates`; {} if there was no update. Starts a new window."""
        out = {k: self.sum[k] / self.n[k] for k in self.sum}
        if out:
            out["updates"] = self.updates
        self.sum, self.n, self.updates = {}, {}, 0
        return out
