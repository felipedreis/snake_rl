"""Per-seed curves in 6k-step bins, plus per-action DND write shares for agents that log diagnostics.

usage: snake-peek results/d0/nec_*.json
"""
import json
import sys

import numpy as np


def main(paths):
    for f in paths:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        e = np.array(d["episodes"])
        edges = np.arange(0, d["steps"] + 1, 6000)
        curve = " ".join(f"{e[(e[:,0]>lo)&(e[:,0]<=hi),1].mean():.2f}" for lo, hi in zip(edges[:-1], edges[1:]))
        share = ""
        if d.get("diagnostics"):
            w = [v["appends"] + v["exact_updates"] + v["evictions"] for v in d["diagnostics"][-1][1].values()]
            share = " share " + str([round(x / sum(w), 2) for x in w])
        print(f"{f.split('/')[-1]:22s} {curve}{share}")


def cli():
    main(sys.argv[1:])


if __name__ == "__main__":
    cli()
