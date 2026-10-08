"""Analysis for the encoder food-probe experiment (PROTOCOL.md): {nec, dqn} x {MLP, CNN} + random, 25x25 rooms.

usage (from the repo root):
  .venv/bin/python docs/experiments/encoder_food_probe_g25/analyze.py [--root DIR] [--seeds LO-HI] [--out DIR]
Writes <out>/per_run.csv, <out>/results.md and <out>/probe_curves.png (out defaults to this directory).
"""
import argparse
import itertools
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snake_rl.run import run_name

SUB = "g25_d0_rooms_b5"
CURRICULUM = (2, 250_000, 600_000)
EPS = (0.05, 50_000)                 # identical for every agent (PROTOCOL.md section 2)
AGENTS = ["dqn", "dqn_cnn", "nec", "nec_cnn", "random"]
LEARNERS = ["dqn", "dqn_cnn", "nec", "nec_cnn"]
NAMES = {"dqn": "DQN · MLP", "dqn_cnn": "DQN · CNN", "nec": "NEC · MLP", "nec_cnn": "NEC · CNN", "random": "Random"}
# Colour = learner (validated slots 1-2 of the reference palette, as in v4's analyze.py); line style = encoder.
COLORS = {"dqn": "#2a78d6", "dqn_cnn": "#2a78d6", "nec": "#eb6834", "nec_cnn": "#eb6834", "random": "#8a8984"}
STYLE = {"dqn": "--", "nec": "--", "dqn_cnn": "-", "nec_cnn": "-", "random": ":"}
T_READ = 250_000                     # end of the held phase: all outcomes are read here
HELD = (100_000, 250_000)            # window for training fruit per 100 steps
PROBE_KEYS = ["steer", "knn_offset", "knn_config", "decode", "food_ratio"]
OUTCOMES = ["steer", "fruit_per_100", "knn_offset", "decode", "food_ratio"]
N_BOOT = 10_000


def load(root, seeds):
    runs = {}
    for a in AGENTS:
        stem = run_name(a, EPS[0], EPS[1], CURRICULUM, True)
        for s in seeds:
            p = f"{root}/{SUB}/{stem}_s{s}.json"
            if os.path.exists(p):
                runs.setdefault(a, {})[s] = json.load(open(p))
            else:
                print(f"missing: {p}")
    return runs


def fruit_rate(d, lo, hi):
    """Training regular fruit per 100 steps, over episodes that ended in (lo, hi]."""
    e = np.array(d["episodes"], float)
    m = (e[:, 0] > lo) & (e[:, 0] <= hi)
    return 100.0 * e[m, 2].sum() / (hi - lo)


def per_run(d):
    probes = dict((t, m) for t, m in d["probes"])
    row = {k: probes.get(T_READ, {}).get(k, np.nan) for k in PROBE_KEYS}
    row["steer_chance"] = probes.get(T_READ, {}).get("steer_chance", np.nan)
    row["fruit_per_100"] = fruit_rate(d, *HELD)
    ev = [ep for t, eps in d["evaluations"] for ep in eps]
    row["eval_score"] = float(np.mean([e[0] for e in ev])) if ev else np.nan
    row["wallclock_h"] = d["wallclock_s"] / 3600
    return row


def perm_p(x, y):
    """Exact two-sided permutation p for the difference in means."""
    z, n = np.concatenate([x, y]), len(x)
    obs = abs(x.mean() - y.mean())
    diffs = [abs(z[list(c)].mean() - np.delete(z, list(c)).mean()) for c in itertools.combinations(range(len(z)), n)]
    return float(np.mean(np.array(diffs) >= obs - 1e-12))


def boot_ci(x, y, rng):
    b = [rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(N_BOOT)]
    return np.percentile(b, [2.5, 97.5])


def figure(runs, out):
    panels = [("steer", "Greedy steering accuracy (probe)"), ("knn_offset", "kNN neighbours with same food offset"),
              ("decode", "Linear read-out of food offset"), ("food_ratio", "Embedding move: food ÷ snake position"),
              ("fruit", "Training fruit per 100 steps"), ("eval", "Evaluation score (food anywhere)")]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (key, title) in zip(axs.ravel(), panels):
        for a in AGENTS:
            if a not in runs or (a == "random" and key not in ("fruit", "eval")):
                continue
            curves = []
            for d in runs[a].values():
                if key == "fruit":
                    edges = np.arange(0, d["steps"] + 1, 20_000)
                    x = edges[1:] / 1000
                    y = [fruit_rate(d, lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]
                elif key == "eval":
                    x = np.array([t for t, _ in d["evaluations"]]) / 1000
                    y = [np.mean([e[0] for e in eps]) for _, eps in d["evaluations"]]
                else:
                    x = np.array([t for t, _ in d["probes"]]) / 1000
                    y = [m[key] for _, m in d["probes"]]
                curves.append(np.array(y, float))
                ax.plot(x, y, STYLE[a], color=COLORS[a], lw=0.6, alpha=0.3)
            ax.plot(x, np.mean(curves, 0), STYLE[a], color=COLORS[a], lw=2, label=NAMES[a])
        if key == "steer":
            ch = np.mean([m["steer_chance"] for d in runs[LEARNERS[0]].values() for _, m in d["probes"][:1]])
            ax.axhline(ch, color="#8a8984", ls=":", lw=1)
            ax.text(2, ch, " random action", va="bottom", fontsize=8, color="#555")
        for t in (250, 330):
            ax.axvline(t, color="#bbb", lw=0.8, ls=":")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("training steps (k)", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0, 0].legend(fontsize=8, frameon=False)
    axs[1, 1].legend(fontsize=8, frameon=False)
    fig.suptitle("Encoder food probe, 25×25 rooms; thin = seeds, thick = mean; dotted verticals: radius starts "
                 "growing (250k), reaches 13 (~330k)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/probe_curves.png", dpi=130)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/exp_encoder_probe")
    ap.add_argument("--seeds", default="501-505")
    ap.add_argument("--out", default=here)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    runs = load(a.root, range(lo, hi + 1))
    rows = {ag: {s: per_run(d) for s, d in rs.items()} for ag, rs in runs.items()}
    keys = ["steer", "steer_chance", "fruit_per_100", "knn_offset", "knn_config", "decode", "food_ratio",
            "eval_score", "wallclock_h"]
    with open(f"{a.out}/per_run.csv", "w") as f:
        f.write("agent,seed," + ",".join(keys) + "\n")
        for ag, rs in rows.items():
            for s, r in sorted(rs.items()):
                f.write(f"{ag},{s}," + ",".join(f"{r[k]:.4f}" for k in keys) + "\n")
    md = [f"# Encoder food probe: results (generated by analyze.py)\n",
          f"Runs: " + ", ".join(f"{NAMES[ag]} {len(rs)}" for ag, rs in rows.items()) + "\n",
          f"## Per agent at step {T_READ:,} (mean ± sd over seeds; fruit over {HELD[0]//1000}k–{HELD[1]//1000}k)\n",
          "| Agent | " + " | ".join(keys[:-1]) + " |", "|---" * len(keys) + "|"]
    for ag in AGENTS:
        if ag in rows:
            v = {k: np.array([r[k] for r in rows[ag].values()]) for k in keys}
            md.append(f"| {NAMES[ag]} | " + " | ".join(f"{np.nanmean(v[k]):.3f} ± {np.nanstd(v[k], ddof=1) if len(v[k]) > 1 else 0:.3f}"
                                                    for k in keys[:-1]) + " |")
    md += [f"\n## Per seed: steer / knn_offset / decode at {T_READ:,}\n"]
    for ag in LEARNERS:
        if ag in rows:
            md.append(f"- {NAMES[ag]}: " + "; ".join(f"s{s} {r['steer']:.3f} / {r['knn_offset']:.3f} / {r['decode']:.3f}"
                                                    for s, r in sorted(rows[ag].items())))
    md += ["\n## CNN − MLP, per learner (bootstrap 95% CI; exact permutation p, descriptive)\n",
           "| Learner | Outcome | MLP | CNN | CNN − MLP | 95% CI | p |", "|---|---|---|---|---|---|---|"]
    rng = np.random.default_rng(0)
    for learner in ("dqn", "nec"):
        if learner not in rows or learner + "_cnn" not in rows:
            continue
        for k in OUTCOMES:
            x = np.array([r[k] for r in rows[learner + "_cnn"].values()])
            y = np.array([r[k] for r in rows[learner].values()])
            ci = boot_ci(x, y, rng)
            md.append(f"| {learner.upper()} | {k} | {y.mean():.3f} | {x.mean():.3f} | {x.mean() - y.mean():+.3f} "
                      f"| [{ci[0]:+.3f}, {ci[1]:+.3f}] | {perm_p(x, y):.3f} |")
    allr = [r for ag in LEARNERS if ag in rows for r in rows[ag].values()]
    if len(allr) > 2:
        st = np.array([r["steer"] for r in allr])
        md += [f"\n## Association across {len(allr)} learning runs at {T_READ:,}\n"]
        for k in ("knn_offset", "decode", "food_ratio", "fruit_per_100"):
            md.append(f"- corr(steer, {k}) = {np.corrcoef(st, [r[k] for r in allr])[0, 1]:+.2f}")
    open(f"{a.out}/results.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    figure(runs, a.out)


if __name__ == "__main__":
    main()
