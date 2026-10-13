"""Analysis for docs/experiments/final_ladder/PROTOCOL.md. Written before the runs; frozen with the protocol.

usage: python docs/experiments/final_ladder/analyze.py [results_root] [out_dir]
Reads <root>/**/*_s<seed>.json (both games), writes results.md, per_run.csv and eval_curves.png to out_dir.

Outcomes, per run (the unit of analysis), from the evaluation episodes (20 at epsilon 0.05 every 25k steps):
  O1 (final)  mean evaluation score over the evaluations at 950k, 975k and 1M steps.
  O2 (early)  mean evaluation score over the evaluations at 200k, 225k and 250k steps (the NEC paper's first point, 1M frames).
Confirmatory: H1-H4 for each game (8 tests) and, for Snake only (revision 2), C1-C5, curriculum against plain training for each arm on O1.
All are one-sided exact permutation tests on the difference of means; Holm-adjusted within each family (8 and 5), alpha 0.05, exactly as in
PROTOCOL.md section 1. Everything else is labelled exploratory. Snake runs trained with the food curriculum are the arms "<agent>+fc".
"""
import csv
import glob
import itertools
import json
import os
import sys
from math import comb

import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/exp_final_ladder"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
GAMES = ("snake", "pacman")
UNIT = {"snake": "fruit per episode", "pacman": "points per episode"}
ARMS = ("random", "dqn", "dqn_nstep", "mfec", "ec_frozen", "nec")  # the ladder order
LEGACY = {"dqn_naturecnn": "dqn", "dqn_nstep_naturecnn": "dqn_nstep", "ec_frozen_naturecnn": "ec_frozen", "nec_naturecnn": "nec"}
LABEL = {"random": "Random", "dqn": "DQN", "dqn_nstep": "N-step DQN", "mfec": "MFEC", "ec_frozen": "Frozen-embedding EC",
         "nec": "NEC"}
# Categorical slots of the project's reference palette, fixed order (validated); random is the neutral gray.
COLOR = {"dqn": "#2a78d6", "dqn_nstep": "#eb6834", "mfec": "#1baf7a", "ec_frozen": "#eda100", "nec": "#e87ba4",
         "random": "#8a8984"}
FC_ARMS = tuple(a + "+fc" for a in ("dqn", "dqn_nstep", "mfec", "ec_frozen", "nec"))
LABEL.update({a: LABEL[a.removesuffix("+fc")] + " + curriculum" for a in FC_ARMS})
COLOR.update({a: COLOR[a.removesuffix("+fc")] for a in FC_ARMS})
O1_STEPS, O2_STEPS = (950_000, 975_000, 1_000_000), (200_000, 225_000, 250_000)
# H id, outcome, arm A, arm B: "A > B" (one-sided)
HYPOTHESES = [("H1", "o2", "nec", "dqn"), ("H2", "o1", "nec", "dqn"), ("H3", "o1", "nec", "ec_frozen"),
              ("H4", "o1", "nec", "mfec")]
CURRICULUM_ARMS = ("dqn", "dqn_nstep", "mfec", "ec_frozen", "nec")  # C1-C5: "<arm>+fc" > "<arm>" on O1 (Snake)
ALPHA, B, BOOT_SEED = 0.05, 10_000, 12345


def eval_mean(ev, steps):
    """Mean evaluation score over the evaluations at `steps` (NaN if any is missing)."""
    by_t = {t: np.mean([e[0] for e in eps]) for t, eps in ev}
    return float(np.mean([by_t[t] for t in steps])) if all(t in by_t for t in steps) else np.nan


def run_metrics(d):
    ev = d.get("evaluations") or []
    m = dict(game=d.get("game", "snake"), agent=LEGACY.get(d["agent"], d["agent"]) + ("+fc" if d.get("food_curriculum") else ""),
             seed=d["seed"], steps=d["steps"],
             wall_h=d["wallclock_s"] / 3600, o1=eval_mean(ev, O1_STEPS), o2=eval_mean(ev, O2_STEPS))
    m["curve"] = [(t, float(np.mean([e[0] for e in eps]))) for t, eps in ev]
    m["eval_len"] = float(np.mean([e[3] for t, eps in ev if t in O1_STEPS for e in eps])) if ev else np.nan
    pr = dict(d.get("probes", []))
    p = pr.get(d["steps"], {})
    m["steer"], m["steer_chance"], m["decode"] = p.get("steer"), p.get("steer_chance"), p.get("decode")
    kn = [s["key_norm"] for _, s in d.get("train", []) if "key_norm" in s]
    m["key_norm_max"] = max(kn) if kn else None
    diag = d["diagnostics"][-1][1] if d.get("diagnostics") else None
    m["evictions"] = sum(v.get("evictions", 0) for v in diag.values()) if diag else None
    m["rows"] = sum(v.get("n", 0) for v in diag.values()) if diag else None
    return m


def perm_p_greater(x, y):
    """One-sided exact permutation p-value for mean(x) - mean(y) >= observed (all splits of the pooled runs)."""
    pooled, n1 = np.concatenate([x, y]), len(x)
    obs = x.mean() - y.mean()
    assert comb(len(pooled), n1) <= 1_000_000, "group sizes too large for exact enumeration"
    idx = np.array(list(itertools.combinations(range(len(pooled)), n1)))
    s1 = pooled[idx].sum(1)
    diffs = s1 / n1 - (pooled.sum() - s1) / (len(pooled) - n1)
    return float(np.mean(diffs >= obs - 1e-12))


def boot_ci(x, y, rng):
    bx = x[rng.integers(len(x), size=(B, len(x)))].mean(1) - y[rng.integers(len(y), size=(B, len(y)))].mean(1)
    return tuple(np.percentile(bx, [2.5, 97.5]))


def hedges_g(x, y):
    n1, n2 = len(x), len(y)
    sp = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2))
    return float((x.mean() - y.mean()) / sp * (1 - 3 / (4 * (n1 + n2) - 9))) if sp > 0 else float("nan")


def holm(ps):
    order, m = np.argsort(ps), len(ps)
    adj, running = np.empty(m), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return adj


def fmt(x, k=2):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{k}f}"


def values(runs, game, arm, key):
    return np.array([m[key] for m in runs.get((game, arm), []) if not np.isnan(m[key])], float)


def figure(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, game in zip(axes, GAMES):
        ends = []
        for arm in ARMS:
            rs = runs.get((game, arm))
            if not rs:
                continue
            ts = sorted({t for m in rs for t, _ in m["curve"]})
            ys = np.array([[dict(m["curve"]).get(t, np.nan) for t in ts] for m in rs])
            mu, se = np.nanmean(ys, 0), np.nanstd(ys, 0, ddof=1) / np.sqrt(np.sum(~np.isnan(ys), 0)) if len(rs) > 1 else 0
            x = np.array(ts) / 1e3
            ax.plot(x, mu, color=COLOR[arm], lw=2, ls="--" if arm == "random" else "-", label=LABEL[arm])
            ax.fill_between(x, mu - se, mu + se, color=COLOR[arm], alpha=0.15, lw=0)
            ends.append((mu[-1], x[-1], arm))
        # Direct labels at the line ends (identity is never colour alone), spread apart so close lines keep readable labels.
        lo, hi = ax.get_ylim()
        gap, placed = 0.045 * (hi - lo), []
        for y, x, arm in sorted(ends):
            y_lab = max(y, placed[-1] + gap) if placed else y
            placed.append(y_lab)
            ax.annotate(LABEL[arm], (x, y), xytext=(x + 8, y_lab), textcoords="data", va="center", fontsize=8,
                        color="#3d3d3a", annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color="#c3c2b7", lw=0.6) if abs(y_lab - y) > gap / 3 else None)
        ax.axvspan(O2_STEPS[0] / 1e3, O2_STEPS[-1] / 1e3, color="#8a8984", alpha=0.08, lw=0)
        ax.axvspan(O1_STEPS[0] / 1e3, O1_STEPS[-1] / 1e3, color="#8a8984", alpha=0.08, lw=0)
        ax.set_title(f"{'Snake 10x10 (pixels)' if game == 'snake' else 'Ms. Pac-Man (Atari units)'}", fontsize=11, loc="left")
        ax.set_xlabel("agent steps (thousands)")
        ax.set_ylabel(f"evaluation score, {UNIT[game]}")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        ax.set_xlim(0, max(x for _, x, _ in ends) if ends else 1000)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.text(0.01, 0.01, "Mean over 10 seeds, band = ±1 s.e. Shaded windows: O2 (200k-250k) and O1 (950k-1M).", fontsize=8, color="#5f5e5a")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.subplots_adjust(wspace=0.45, right=0.88)
    fig.savefig(path, dpi=150)


def figure_curriculum(runs, path):
    """Snake: each arm trained plainly (solid) against with the food curriculum (dashed), evaluation score by step."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = ("dqn", "dqn_nstep", "mfec", "ec_frozen", "nec")
    fig, axes = plt.subplots(1, len(arms), figsize=(15, 3.6), sharex=True)
    for ax, arm in zip(axes, arms):
        for key, ls in ((arm, "-"), (arm + "+fc", "--")):
            rs = runs.get(("snake", key))
            if not rs:
                continue
            ts = sorted({t for m in rs for t, _ in m["curve"]})
            ys = np.array([[dict(m["curve"]).get(t, np.nan) for t in ts] for m in rs])
            mu = np.nanmean(ys, 0)
            se = np.nanstd(ys, 0, ddof=1) / np.sqrt(np.sum(~np.isnan(ys), 0)) if len(rs) > 1 else 0
            x = np.array(ts) / 1e3
            ax.plot(x, mu, color=COLOR[arm], lw=2, ls=ls, label="curriculum" if key.endswith("+fc") else "plain")
            ax.fill_between(x, mu - se, mu + se, color=COLOR[arm], alpha=0.15, lw=0)
        ax.axvspan(50, 150, color="#8a8984", alpha=0.08, lw=0)  # the curriculum grows over 50k-150k
        ax.set_title(LABEL[arm], fontsize=10, loc="left")
        ax.set_xlabel("agent steps (thousands)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].set_ylabel("evaluation score, fruit per episode")
    fig.text(0.01, 0.01, "Snake 10x10, mean over 10 seeds, band = ±1 s.e.; evaluation always uses the real game. Shaded: the curriculum's growth (50k-150k).",
             fontsize=8, color="#5f5e5a")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)


def main():
    runs = {}
    for f in sorted(glob.glob(f"{ROOT}/**/*_s[0-9]*.json", recursive=True)):
        m = run_metrics(json.load(open(f)))
        if m["agent"].removesuffix("+fc") not in ARMS:
            print("skipping", f)
            continue
        runs.setdefault((m["game"], m["agent"]), []).append(m)
    if not runs:
        sys.exit(f"no results under {ROOT}")
    for k in runs:
        runs[k].sort(key=lambda m: m["seed"])
    rng = np.random.default_rng(BOOT_SEED)
    L = ["# Results: the final ladder (generated by analyze.py)", ""]

    L += ["## Outcomes per arm (mean, s.d. over seeds; per-seed values in per_run.csv)", ""]
    for game in GAMES:
        L += [f"### {'Snake 10x10, pixels' if game == 'snake' else 'Ms. Pac-Man, Atari units'} ({UNIT[game]})", "",
              "| Arm | n | O2 early (200k-250k) | O1 final (950k-1M) | eval episode length (O1 window) | wall h |", "|---|---|---|---|---|---|"]
        for arm in ARMS + (FC_ARMS if game == "snake" else ()):
            rs = runs.get((game, arm), [])
            if not rs:
                continue
            o1, o2 = values(runs, game, arm, "o1"), values(runs, game, arm, "o2")
            sd = lambda v: f" ({v.std(ddof=1):.2f})" if len(v) > 1 else ""
            L.append(f"| {LABEL[arm]} | {len(rs)} | {fmt(o2.mean() if len(o2) else np.nan)}{sd(o2)} | "
                     f"{fmt(o1.mean() if len(o1) else np.nan)}{sd(o1)} | {fmt(np.nanmean([m['eval_len'] for m in rs]), 0)} | "
                     f"{np.mean([m['wall_h'] for m in rs]):.1f} |")
        L.append("")

    L += ["## Confirmatory tests (one-sided exact permutation, Holm-adjusted over 8, alpha 0.05)", "",
          "| Hypothesis | Game | Outcome | A > B | difference [95% bootstrap CI] | Hedges' g | p | p (Holm) | Verdict |",
          "|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for game in GAMES:
        for hid, key, a, b in HYPOTHESES:
            x, y = values(runs, game, a, key), values(runs, game, b, key)
            if len(x) < 2 or len(y) < 2:
                rows.append((hid, game, key, a, b, None))
                continue
            rows.append((hid, game, key, a, b, (x.mean() - y.mean(), boot_ci(x, y, rng), hedges_g(x, y), perm_p_greater(x, y))))
    ok = [r[5] is not None for r in rows]
    adj = np.full(len(rows), np.nan)
    if any(ok):
        adj[np.array(ok)] = holm(np.array([r[5][3] for r in rows if r[5] is not None]))
    for (hid, game, key, a, b, res), pa in zip(rows, adj):
        if res is None:
            L.append(f"| {hid} | {game} | {key.upper()} | {LABEL[a]} > {LABEL[b]} | - | - | - | - | missing runs |")
            continue
        d, (lo, hi), g, p = res
        verdict = "**supported**" if pa < ALPHA else "not supported"
        L.append(f"| {hid} | {game} | {key.upper()} | {LABEL[a]} > {LABEL[b]} | {d:+.2f} [{lo:+.2f}, {hi:+.2f}] | {g:.2f} | "
                 f"{p:.2g} | {pa:.2g} | {verdict} |")
    if len(rows) != 8 or not all(ok):
        L += ["", "**Incomplete:** not all 8 confirmatory tests could be computed; the Holm family above is smaller than 8."]

    L += ["", "## Confirmatory tests, Snake food curriculum (revision 2): curriculum > plain on O1, one-sided exact permutation, Holm over 5, alpha 0.05", "",
          "| Hypothesis | Arm | difference [95% bootstrap CI] | Hedges' g | p | p (Holm) | Verdict |", "|---|---|---|---|---|---|---|"]
    crows = []
    for i, arm in enumerate(CURRICULUM_ARMS, 1):
        x, y = values(runs, "snake", arm + "+fc", "o1"), values(runs, "snake", arm, "o1")
        crows.append((f"C{i}", arm, None if len(x) < 2 or len(y) < 2 else (x.mean() - y.mean(), boot_ci(x, y, rng), hedges_g(x, y), perm_p_greater(x, y))))
    cok = np.array([r[2] is not None for r in crows])
    cadj = np.full(len(crows), np.nan)
    if cok.any():
        cadj[cok] = holm(np.array([r[2][3] for r in crows if r[2] is not None]))
    for (cid, arm, res), pa in zip(crows, cadj):
        if res is None:
            L.append(f"| {cid} | {LABEL[arm]} | - | - | - | - | missing runs |")
            continue
        d, (lo, hi), g, p = res
        L.append(f"| {cid} | {LABEL[arm]} | {d:+.2f} [{lo:+.2f}, {hi:+.2f}] | {g:.2f} | {p:.2g} | {pa:.2g} | "
                 f"{'**supported**' if pa < ALPHA else 'not supported'} |")
    if not cok.all():
        L += ["", "**Incomplete:** not all 5 curriculum tests could be computed."]

    L += ["", "## Exploratory (not pre-registered tests)", "",
          "### Snake curriculum against plain on the early outcome O2 and on O1 (difference of means, 95% bootstrap CI)", "",
          "| Arm | O2 | O1 |", "|---|---|---|"]
    for arm in CURRICULUM_ARMS:
        cells = []
        for key in ("o2", "o1"):
            x, y = values(runs, "snake", arm + "+fc", key), values(runs, "snake", arm, key)
            if len(x) > 1 and len(y) > 1:
                lo, hi = boot_ci(x, y, rng)
                cells.append(f"{x.mean() - y.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]")
            else:
                cells.append("-")
        L.append(f"| {LABEL[arm]} | {cells[0]} | {cells[1]} |")
    L += [""]
    L += [
          "### The ladder: each rung against the one below, and every arm against random (O1 and O2; difference, 95% bootstrap CI)", ""]
    pairs = [("dqn", "random"), ("dqn_nstep", "dqn"), ("ec_frozen", "dqn_nstep"), ("nec", "ec_frozen"), ("mfec", "random"),
             ("ec_frozen", "random"), ("nec", "random")]
    for game in GAMES:
        L += [f"**{game}**", "", "| Comparison | O2 | O1 |", "|---|---|---|"]
        for a, b in pairs:
            cells = []
            for key in ("o2", "o1"):
                x, y = values(runs, game, a, key), values(runs, game, b, key)
                if len(x) > 1 and len(y) > 1:
                    lo, hi = boot_ci(x, y, rng)
                    cells.append(f"{x.mean() - y.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]")
                else:
                    cells.append("-")
            L.append(f"| {LABEL[a]} - {LABEL[b]} | {cells[0]} | {cells[1]} |")
        L.append("")
    L += ["### Snake probe at 1M steps and memory health (mean over seeds)", "",
          "| Game | Arm | near-probe steering (chance) | decode | max key norm | memory rows (evictions), end |", "|---|---|---|---|---|---|"]
    for game in GAMES:
        for arm in ARMS:
            rs = runs.get((game, arm), [])
            if not rs or arm == "random":
                continue
            mean = lambda k: (float(np.mean([m[k] for m in rs if m[k] is not None])) if any(m[k] is not None for m in rs) else None)
            L.append(f"| {game} | {LABEL[arm]} | {fmt(mean('steer'))} ({fmt(mean('steer_chance'))}) | {fmt(mean('decode'))} | "
                     f"{fmt(mean('key_norm_max'))} | {fmt(mean('rows'), 0)} ({fmt(mean('evictions'), 0)}) |")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "results.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(OUT, "per_run.csv"), "w", newline="") as f:
        w = csv.writer(f)
        cols = ["game", "agent", "seed", "steps", "o2", "o1", "eval_len", "steer", "steer_chance", "decode", "key_norm_max", "rows",
                "evictions", "wall_h"]
        w.writerow(cols)
        for k in sorted(runs):
            for m in runs[k]:
                w.writerow([m[c] for c in cols])
    figure(runs, os.path.join(OUT, "eval_curves.png"))
    if any(k[1].endswith("+fc") for k in runs):
        figure_curriculum(runs, os.path.join(OUT, "eval_curves_curriculum.png"))
    print("\n".join(L))


if __name__ == "__main__":
    main()
