"""Learning curves (mean +- s.e. over seeds) and an early-regime table.

usage: snake-plot [results_dir] [--out figures]
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snake_rl.run import EPS_FLOOR

# Legend names, also the plotting order. Agents not listed here are plotted after these, under their raw name.
# Runs are grouped by (agent, eps_floor, eps_decay, food_curriculum, food_relocate, opt, lr); non-default settings are
# appended to the label.
LABELS = {"random": "Random policy", "dqn": "DQN (1-step)", "dqn_nstep": "DQN (N-step)", "mfec": "MFEC (random projection)",
          "ec_frozen": "Episodic, frozen embedding", "nec": "NEC (learned embedding)",
          "nec_refresh": "NEC + key refresh", "nec_bonus": "NEC + density bonus",
          "dqn_cnn": "DQN (1-step, CNN)", "dqn_nstep_cnn": "DQN (N-step, CNN)",
          "ec_frozen_cnn": "Episodic, frozen CNN embedding", "nec_cnn": "NEC (CNN)"}


def label(agent, eps_floor, eps_decay=None, food_curriculum=None, food_relocate=False, opt="adam", lr=None):
    name = LABELS.get(agent, agent)
    name += "" if opt == "adam" else f", {opt}"
    name += "" if lr is None else f", lr {lr:g}"
    name += "" if eps_floor == EPS_FLOOR else f", eps floor {eps_floor:g}"
    name += "" if eps_decay is None else f", eps decay {eps_decay:,}"
    name += "" if food_curriculum is None else f", food curriculum {':'.join(str(x) for x in food_curriculum)}"
    return name + (", food relocation" if food_relocate else "")


def curve(episodes, edges, rate=False):
    """Per-episode fruit (default) or fruit per 1000 env steps (rate=True), binned by episode end."""
    e = np.array(episodes)
    BIN = edges[1] - edges[0]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (e[:, 0] > lo) & (e[:, 0] <= hi)
        if rate:
            out.append(e[m, 1].sum() * 1000.0 / BIN)
        else:
            out.append(e[m, 1].mean() if m.any() else np.nan)
    return np.array(out)


def main(DIR="results/d0", out_dir="figures"):
    files = sorted(glob.glob(f"{DIR}/*.json"))
    if not files:
        raise SystemExit(f"no results in {DIR}")
    first = json.load(open(files[0]))
    STEPS, size = first["steps"], first.get("size", 7)
    edges = np.arange(0, STEPS + 1, STEPS // 20)

    data = {}
    for f in files:
        d = json.load(open(f))
        if d["steps"] == STEPS:
            fc = d.get("food_curriculum")
            key = (d["agent"], d.get("eps_floor", EPS_FLOOR), d.get("eps_decay"), tuple(fc) if fc else None,
                   d.get("food_relocate", False), d.get("opt", "adam"), d.get("lr"))
            data.setdefault(key, []).append(
                (curve(d["episodes"], edges), curve(d["episodes"], edges, True), d["wallclock_s"]))
        else:
            print(f"skipping {f}: {d['steps']} steps != {STEPS}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = edges[1:] / 1000
    tables = {0: [], 1: []}
    order = list(LABELS)
    for key in sorted(data, key=lambda k: (order.index(k[0]) if k[0] in LABELS else len(order), k[0], k[1],
                                         -1 if k[2] is None else k[2], k[3] or (), k[4], k[5], -1 if k[6] is None else k[6])):
        lab = label(*key)
        for k, ax in enumerate(axes):
            C = np.stack([run[k] for run in data[key]])
            mu, se = np.nanmean(C, 0), np.nanstd(C, 0) / np.sqrt(len(C))
            ax.plot(x, mu, label=f"{lab} (n={len(C)})")
            ax.fill_between(x, mu - se, mu + se, alpha=0.2)
            marks = []
            for lo, hi in [(0, STEPS // 8), (STEPS // 8, STEPS // 4), (STEPS // 4, STEPS // 2), (STEPS // 2, STEPS)]:
                per_seed = np.nanmean(C[:, (edges[1:] > lo) & (edges[1:] <= hi)], 1)
                marks.append((per_seed.mean(), per_seed.std() / np.sqrt(len(per_seed))))
            tables[k].append((lab, marks, np.mean([r[2] for r in data[key]])))
    axes[0].set_ylabel("fruit eaten per episode")
    axes[1].set_ylabel("fruit eaten per 1000 steps")
    for ax in axes:
        ax.set_xlabel("environment steps (thousands)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    tag = os.path.basename(os.path.normpath(DIR))
    fig.suptitle(f"Snake {size}x{size}, {tag}: NEC ablation ladder (mean +- s.e. over seeds)")
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/learning_curves_{tag}.png", dpi=150)

    w0 = max(30, *(len(lab) for lab, _, _ in tables[0]))
    for k, title in [(0, "fruit / episode"), (1, "fruit / 1000 steps")]:
        print(f"\n{title}")
        print(f"{'agent':{w0}s} {'1st 1/8':>12} {'2nd 1/8':>12} {'2nd 1/4':>12} {'2nd 1/2':>12} {'wall s':>7}")
        for lab, m, w in tables[k]:
            print(f"{lab:{w0}s} " + " ".join(f"{a:6.2f}±{b:4.2f}" for a, b in m) + f" {w:7.0f}")


def cli():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", nargs="?", default="results/d0")
    ap.add_argument("--out", default="figures", help="where to write the png")
    a = ap.parse_args()
    main(a.dir, a.out)


if __name__ == "__main__":
    cli()
