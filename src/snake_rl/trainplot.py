"""Training telemetry curves (loss, TD error, Q vs target, gradient norm, step size, ...) for one or more runs.

usage: snake-train RUN [RUN ...] [--out PNG] [--smooth K]
  RUN is a result JSON or a live <stem>.train.jsonl of a run still in progress (run.py streams one there).
  Colour = run setting (agent and flags), one thin line per seed. Prints the latest window of each run.

What healthy looks like (rules of thumb, not laws):
  loss / td_abs    fall, then flatten; a steady rise means the targets are running away.
  q_taken / target track each other; Q climbing far above anything the rewards allow = overestimation.
  grad_norm        no sustained growth; `clipped` near 1 means every step is being clipped (lr too high).
  update_ratio     around 1e-3; >>1e-2 thrashes, <<1e-4 has stalled.
  dead_relu        share of last-hidden units off for a whole minibatch; creeping towards 1 = dying network.
  nb_dist (NEC)    the embedding's scale as the kernel sees it; collapsing towards 0 = keys collapsing together.
  kernel_peak (NEC) 1/p (0.02) = Q is a plain average of neighbours; near 1 = one neighbour decides.
"""
import argparse
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (key, title, log-scale y)
PANELS = [("loss", "Loss", True), ("td_abs", "|TD error|", True), ("q", "Q(s,a) (solid) vs target (dotted)", False),
          ("grad_norm", "Gradient norm (before clipping)", True), ("clipped", "Share of steps clipped", False),
          ("update_ratio", "Step size ÷ weight norm", True), ("dead_relu", "Dead last-hidden ReLUs", False),
          ("nb_dist", "NEC: mean sq. distance to neighbours", True), ("kernel_peak", "NEC: largest neighbour weight", False)]
# Categorical slots of the reference palette, in fixed order (same as the experiment analyses).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#8a8984"]


def load(path):
    """-> (group label, seed label, list of (t, stats))."""
    name = os.path.basename(path)
    if path.endswith(".jsonl"):
        rows = [json.loads(line) for line in open(path) if line.strip()]
        series = [(r.pop("t"), r) for r in rows]
        stem = name[: -len(".train.jsonl")] + " (live)"
    else:
        d = json.load(open(path))
        series = [(t, s) for t, s in d.get("train", [])]
        stem = name[: -len(".json")]
    m = re.match(r"(.*)_s(\d+)( \(live\))?$", stem)
    return (m.group(1), "s" + m.group(2) + (m.group(3) or "")) if m else (stem, ""), series


def smooth(y, k):
    if k <= 1 or len(y) < k:
        return y
    c = np.cumsum(np.insert(np.nan_to_num(y), 0, 0.0))
    out = (c[k:] - c[:-k]) / k
    return np.concatenate([y[: k - 1], out])


def main(paths, out, k):
    runs = [(g, s, ser) for (g, s), ser in (load(p) for p in paths)]
    runs = [r for r in runs if r[2]]
    if not runs:
        print("no training telemetry in these files (agents without gradient training, or runs from before it existed)")
        return
    groups = list(dict.fromkeys(g for g, _, _ in runs))
    color = {g: COLORS[i % len(COLORS)] for i, g in enumerate(groups)}
    keys = set().union(*(s.keys() for _, _, ser in runs for _, s in ser))
    panels = [p for p in PANELS if p[0] in keys or (p[0] == "q" and "q_taken" in keys)]
    cols = 3
    rows = -(-len(panels) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 3.4 * rows), squeeze=False)
    for ax, (key, title, log) in zip(axs.ravel(), panels):
        for g, seed, ser in runs:
            t = np.array([x for x, _ in ser]) / 1000
            get = lambda kk: np.array([s.get(kk, np.nan) for _, s in ser], float)
            if key == "q":
                ax.plot(t, smooth(get("q_taken"), k), color=color[g], lw=1.2)
                ax.plot(t, smooth(get("target"), k), color=color[g], lw=1.2, ls=":")
            else:
                y = get(key)
                if np.isnan(y).all():  # this agent does not log this quantity (e.g. DQN on the NEC panels)
                    continue
                y = smooth(y, k)
                ax.plot(t, y, color=color[g], lw=1.2)
                if log and (y > 0).any():
                    ax.set_yscale("log")
        if key == "kernel_peak":
            ax.axhline(1 / 50, color="#8a8984", ls=":", lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("training steps (k)", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axs.ravel()[len(panels):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=color[g], lw=2) for g in groups]
    fig.legend(handles, groups, loc="lower center", ncol=min(3, len(groups)), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.03 + 0.025 * ((len(groups) - 1) // 3 + 1), 1, 1))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"-> {out}")
    show = ["loss", "td_abs", "q_taken", "target", "grad_norm", "clipped", "update_ratio", "dead_relu", "nb_dist",
            "kernel_peak"]
    show = [c for c in show if c in keys]
    print(f"{'run':24s} {'step':>7s} " + " ".join(f"{c:>12s}" for c in show))
    for g, seed, ser in runs:
        t, s = ser[-1]
        label = re.split(r"_eps|_fc|_reloc", g)[0] + " " + seed  # settings are in the legend; keep agent + seed
        print(f"{label[:24]:24s} {t:7d} " + " ".join(f"{s.get(c, np.nan):12.4g}" for c in show))


def cli():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="result JSONs and/or live .train.jsonl files")
    ap.add_argument("--out", default="figures/training.png", help="output PNG")
    ap.add_argument("--smooth", type=int, default=1, help="moving average over K snapshots")
    a = ap.parse_args()
    main(a.runs, a.out, a.smooth)


if __name__ == "__main__":
    cli()
