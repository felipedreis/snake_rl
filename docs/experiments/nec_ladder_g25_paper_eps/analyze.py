"""Analysis for the NEC ablation ladder on 25x25 rooms with bonus food under each paper's exploration schedule,
scored on evaluation runs, exactly as specified in PROTOCOL.md.

usage (from the repo root):
  .venv/bin/python docs/experiments/nec_ladder_g25_paper_eps/analyze.py           # main analysis, seeds 201-215
  .venv/bin/python docs/experiments/nec_ladder_g25_paper_eps/analyze.py --gate    # pilot gate (section 10), seeds 1-3
options: [--root DIR] [--seeds LO-HI] [--out DIR]
The main analysis writes <out>/per_run.csv, <out>/results.md and <out>/learning_curves.png.
The gate only prints its table and verdict.
"""
import argparse
import glob
import itertools
import json
import os
from math import comb

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snake_rl.run import EPS_FLOOR

SUB = "g25_d0_rooms_b5"      # results_dir suffix for size 25, D 0, map rooms, bonus 5
BONUS = 5.0
LADDER = ["dqn", "dqn_nstep", "ec_frozen", "nec"]
AGENTS = ["random"] + LADDER + ["mfec"]
NAMES = {"random": "Random", "dqn": "DQN", "dqn_nstep": "N-step DQN", "ec_frozen": "Frozen-embedding EC",
         "nec": "NEC", "mfec": "MFEC"}
# Categorical slots 1-5 of the reference palette in fixed order (validated: CVD and normal-vision pass;
# slots 3-5 are below 3:1 contrast, so every line is direct-labelled and results.md carries the tables).
# The random floor is a neutral dashed reference line.
COLORS = {"dqn": "#2a78d6", "nec": "#eb6834", "mfec": "#1baf7a", "dqn_nstep": "#eda100",
          "ec_frozen": "#e87ba4", "random": "#8a8984"}
# Exploration schedule per agent as (eps_floor, eps_decay), PROTOCOL.md section 3. Runs with any other
# schedule are ignored. random ignores epsilon and runs with run.py's defaults.
CONDITIONS = {"random": (EPS_FLOOR, None),
              "dqn": (0.1, 250_000), "dqn_nstep": (0.1, 250_000),     # DQN paper: 1 -> 0.1 over 1M frames
              "ec_frozen": (0.005, 0), "nec": (0.005, 0), "mfec": (0.005, 0)}  # fixed low epsilon
EVAL_EPISODES = 5           # per evaluation point (run.sh: --eval-every 10000 --eval-episodes 5)
# (id, A, B, factor): hypothesis "A > B" on EARLY (PROTOCOL.md section 7).
HYPOTHESES = [("H0", "nec", "dqn", "whole NEC package"),
              ("H1", "dqn_nstep", "dqn", "N-step returns"),
              ("H2", "ec_frozen", "dqn_nstep", "episodic memory, random embedding"),
              ("H3", "nec", "ec_frozen", "learned embedding"),
              ("H4", "nec", "dqn_nstep", "episodic memory, learned embedding")]
N_PERM = 200_000             # Monte Carlo permutations when exact enumeration exceeds MAX_EXACT
MAX_EXACT = 500_000
N_PERM_LOO = 20_000          # cheaper Monte Carlo for the leave-one-out stability check
GATE_MARGIN = 0.5            # pilot gate: LATE mean must exceed random's by this many points


def windows(S):
    """EARLY (0, S/5] and LATE (S/2, S], by the training step at which an evaluation was run."""
    return (0, S // 5), (S // 2, S)


def in_window(e, lo, hi):
    return (e[:, 0] > lo) & (e[:, 0] <= hi)


def eval_table(d):
    """Evaluation episodes as rows (training step, score, regular foods, truncated, length)."""
    return np.array([(t, *ep) for t, eps in d["evaluations"] for ep in eps], float)


def eval_curve(d, edges):
    """Mean evaluation score per bin of training steps."""
    e = eval_table(d)
    return np.array([e[in_window(e, lo, hi), 1].mean() if in_window(e, lo, hi).any() else np.nan
                     for lo, hi in zip(edges[:-1], edges[1:])])


def wmean(x, m):
    return float(x[m].mean()) if m.any() else np.nan


def load(root, seeds):
    """{agent: [run dict]} for runs with the agent's protocol exploration schedule and the given seeds."""
    data = {a: [] for a in AGENTS}
    for f in sorted(glob.glob(f"{root}/{SUB}/*.json")):
        d = json.load(open(f))
        a = d["agent"]
        if (a in data and d["seed"] in seeds
                and (d.get("eps_floor", EPS_FLOOR), d.get("eps_decay")) == CONDITIONS[a]):
            if not d.get("evaluations"):
                raise SystemExit(f"{f}: no evaluation runs recorded; run with --eval-every")
            data[a].append(d)
    return data


def metrics(d):
    e, S = eval_table(d), d["steps"]
    (elo, ehi), (llo, lhi) = windows(S)
    me, ml = in_window(e, elo, ehi), in_window(e, llo, lhi)
    _, score, foods, trunc, length = e.T
    tr = np.array(d["episodes"], float)  # training episodes: descriptive only
    mt = tr[:, 0] > S / 2
    row = {"seed": d["seed"],
           "early": wmean(score, me), "late": wmean(score, ml),
           "auc": float(np.mean([np.mean([ep[0] for ep in eps]) for _, eps in d["evaluations"]])),
           "train_late": wmean(tr[:, 1], mt), "train_eps": len(tr),
           "early_regular": wmean(foods, me), "late_regular": wmean(foods, ml),
           "n_early": int(me.sum()), "n_late": int(ml.sum()),
           "bonus_share_late": float((score[ml] - foods[ml]).sum() / score[ml].sum())
           if ml.any() and score[ml].sum() > 0 else np.nan,
           "bonus_per_ep_late": wmean((score - foods) / BONUS, ml),
           "trunc_share_late": wmean(trunc, ml), "ep_len_late": wmean(length, ml),
           "dnd_fill": np.nan, "evict_share": np.nan, "exact_share": np.nan, "full_at": np.nan}
    if d["agent"] in ("ec_frozen", "nec") and d["diagnostics"]:
        snaps = d["diagnostics"]
        last = snaps[-1][1].values()
        ap, up, ev = (sum(v[k] for v in last) for k in ("appends", "exact_updates", "evictions"))
        row["dnd_fill"] = sum(v["n"] for v in last) / (3 * 20000)
        row["evict_share"] = ev / (ap + up + ev) if ap + up + ev else np.nan
        row["exact_share"] = up / (ap + up + ev) if ap + up + ev else np.nan
        full = [t for t, s in snaps if any(v["evictions"] > 0 for v in s.values())]
        row["full_at"] = full[0] if full else np.nan
    return row


def perm_test(x, y, n_mc=N_PERM, seed=0):
    """Two-sided permutation test on mean(x) - mean(y): exact when C(n1+n2, n1) <= MAX_EXACT, else Monte Carlo
    (p = (1 + #{|perm diff| >= |obs|}) / (1 + n_mc), seeded)."""
    pooled, n1 = np.concatenate([x, y]), len(x)
    n, total, obs = len(pooled), pooled.sum(), abs(x.mean() - y.mean())
    if comb(n, n1) <= MAX_EXACT:
        idx = np.array(list(itertools.combinations(range(n), n1)))
        s1 = pooled[idx].sum(1)
        return float(np.mean(np.abs(s1 / n1 - (total - s1) / (n - n1)) >= obs - 1e-12))
    rng, hits, done = np.random.default_rng(seed), 0, 0
    while done < n_mc:
        k = min(50_000, n_mc - done)
        s1 = pooled[np.argsort(rng.random((k, n)), axis=1)[:, :n1]].sum(1)
        hits += int(np.sum(np.abs(s1 / n1 - (total - s1) / (n - n1)) >= obs - 1e-12))
        done += k
    return (1 + hits) / (1 + n_mc)


def boot_ci(x, y=None, n=10_000, seed=0):
    """95% percentile bootstrap CI of mean(x), or of mean(x) - mean(y)."""
    rng = np.random.default_rng(seed)
    bx = x[rng.integers(len(x), size=(n, len(x)))].mean(1)
    if y is not None:
        bx = bx - y[rng.integers(len(y), size=(n, len(y)))].mean(1)
    return np.percentile(bx, [2.5, 97.5])


def hedges_g(x, y):
    n1, n2 = len(x), len(y)
    sp = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2))
    return (x.mean() - y.mean()) / sp * (1 - 3 / (4 * (n1 + n2) - 9)) if sp > 0 else np.nan


def holm(ps):
    order, m = np.argsort(ps), len(ps)
    adj, running = np.empty(m), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return adj


def verdict(diff, p_adj):
    if np.isnan(p_adj):
        return "not tested"
    if p_adj < 0.05:
        return "**supported**" if diff > 0 else "**contradicted**"
    return "not supported"


def fmt_cell(x, scale=1.0, nd=3):
    if len(x) == 0 or np.all(np.isnan(x)):
        return "n/a"
    x = x[~np.isnan(x)] * scale
    lo, hi = boot_ci(x)
    return f"{x.mean():.{nd}f} ± {x.std(ddof=1) if len(x) > 1 else 0:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def gate(root, seeds):
    """Pilot gate (PROTOCOL.md section 10): looks only at evaluation LATE vs the random floor, never between rungs."""
    data = load(root, seeds)
    late = {a: np.array([metrics(d)["late"] for d in data[a]]) for a in AGENTS}
    floor = np.nanmean(late["random"]) if len(late["random"]) else np.nan
    print(f"pilot gate, {root}, seeds {sorted(seeds)}; random LATE = {floor:.3f}, margin {GATE_MARGIN}")
    passed = 0
    for a in LADDER:
        m = np.nanmean(late[a]) if len(late[a]) else np.nan
        ok = m >= floor + GATE_MARGIN
        passed += bool(ok)
        print(f"  {a:10s} n={len(late[a])}  LATE {m:.3f}  {'pass' if ok else 'FAIL'}")
    print(f"gate {'PASSED' if passed >= 3 else 'FAILED'}: {passed}/4 ladder agents clear the floor (need 3)")


def main(root, seeds, out):
    os.makedirs(out, exist_ok=True)
    data = load(root, seeds)
    rows = {a: [metrics(d) for d in data[a]] for a in AGENTS}
    col = lambda a, k: np.array([r[k] for r in rows[a]], float)
    S = next(d["steps"] for a in AGENTS for d in data[a])
    (elo, ehi), (llo, lhi) = windows(S)
    md = [f"Generated by analyze.py from `{root}/{SUB}`, seeds {min(seeds)}-{max(seeds)}, {S:,} steps per run.",
          f"All scores are from evaluation runs ({EVAL_EPISODES} episodes at epsilon 0.05 every 10,000 training steps). "
          f"EARLY = evaluations at steps ({elo:,}, {ehi:,}]; LATE = evaluations at steps ({llo:,}, {lhi:,}].",
          "Exploration during training: " + "; ".join(
              f"{NAMES[a]} {'run.py default' if c[1] is None else 'fixed ' + format(c[0], 'g') if c[1] == 0 else f'1 -> {c[0]:g} over {c[1]:,} steps'}"
              for a, c in CONDITIONS.items()) + ".\n"]

    keys = ["seed", "early", "late", "auc", "train_late", "train_eps", "early_regular", "late_regular", "n_early", "n_late",
            "bonus_share_late", "bonus_per_ep_late", "trunc_share_late", "ep_len_late",
            "dnd_fill", "evict_share", "exact_share", "full_at"]
    with open(f"{out}/per_run.csv", "w") as f:
        f.write("agent," + ",".join(keys) + "\n")
        for a, rs in rows.items():
            for r in rs:
                f.write(f"{a}," + ",".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k])
                                           for k in keys) + "\n")

    md.append("## Runs per agent\n\n| " + " | ".join(NAMES[a] for a in AGENTS) + " |\n|" + "---|" * len(AGENTS))
    md.append("| " + " | ".join(str(len(rows[a])) for a in AGENTS) + " |")

    # Descriptives
    md.append("\n## Outcomes per agent\n\nmean ± s.d. over runs [95% bootstrap CI of the mean]\n")
    md.append("| agent | EARLY (primary) | LATE | AUC | EARLY, regular fruit only | LATE, regular fruit only |")
    md.append("|---|---|---|---|---|---|")
    for a in AGENTS:
        md.append(f"| {NAMES[a]} | " + " | ".join(fmt_cell(col(a, k)) for k in
                                                   ["early", "late", "auc", "early_regular", "late_regular"]) + " |")

    # Confirmatory tests
    res = []
    for hid, A, B, factor in HYPOTHESES:
        x, y = col(A, "early"), col(B, "early")
        x, y = x[~np.isnan(x)], y[~np.isnan(y)]
        if len(x) < 2 or len(y) < 2:
            res.append((hid, A, B, factor, np.nan, np.nan, (np.nan, np.nan), np.nan, len(x), len(y)))
            continue
        res.append((hid, A, B, factor, x.mean() - y.mean(), perm_test(x, y), boot_ci(x, y), hedges_g(x, y),
                    len(x), len(y)))
    ps = np.array([r[5] for r in res])
    adj = np.full(len(ps), np.nan)
    ok = ~np.isnan(ps)
    adj[ok] = holm(ps[ok])
    md.append("\n## Confirmatory tests on EARLY (two-sided permutation, Holm-adjusted over 5)\n")
    md.append("| ID | factor | hypothesis | n | diff (A−B) [95% CI] | Hedges' g | p | p (Holm) | verdict |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for (hid, A, B, factor, diff, p, ci, g, n1, n2), pa in zip(res, adj):
        md.append(f"| {hid} | {factor} | {NAMES[A]} > {NAMES[B]} | {n1}/{n2} | {diff:+.3f} "
                  f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | {g:.2f} | {p:.2g} | {pa:.2g} | {verdict(diff, pa)} |")

    # Decomposition of the EARLY gain along the ladder (descriptive)
    rng = np.random.default_rng(1)
    E = {a: col(a, "early")[~np.isnan(col(a, "early"))] for a in LADDER}
    if all(len(E[a]) >= 2 for a in LADDER):
        boots = {a: E[a][rng.integers(len(E[a]), size=(10_000, len(E[a])))].mean(1) for a in LADDER}
        steps = [("N-step returns", "dqn_nstep", "dqn"),
                 ("episodic memory, random embedding", "ec_frozen", "dqn_nstep"),
                 ("learned embedding", "nec", "ec_frozen")]
        total, total_b = E["nec"].mean() - E["dqn"].mean(), boots["nec"] - boots["dqn"]
        md.append("\n## Decomposition of the EARLY gain, DQN → NEC (descriptive)\n")
        md.append(f"Total gain NEC − DQN = {total:+.3f}. Each step's share is its gain divided by the total; "
                  "the CI is a percentile bootstrap over runs. Shares are only meaningful if H0 is supported.\n")
        md.append("| step | gain [95% CI] | share of total [95% CI] |\n|---|---|---|")
        for name, A, B in steps:
            gb = boots[A] - boots[B]
            sb = gb / np.where(np.abs(total_b) > 1e-9, total_b, np.nan)
            g_lo, g_hi = np.percentile(gb, [2.5, 97.5])
            s_lo, s_hi = np.nanpercentile(sb, [2.5, 97.5])
            share = (E[A].mean() - E[B].mean()) / total if abs(total) > 1e-9 else np.nan
            md.append(f"| {name} | {E[A].mean() - E[B].mean():+.3f} [{g_lo:+.3f}, {g_hi:+.3f}] | "
                      f"{100 * share:.0f}% [{100 * s_lo:.0f}%, {100 * s_hi:.0f}%] |")

    # Exploratory comparisons
    md.append("\n## Exploratory comparisons (uncorrected p; descriptive only)\n")
    md.append("| outcome | A vs B | diff (A−B) [95% CI] | Hedges' g | p (uncorrected) |\n|---|---|---|---|---|")
    ladder_pairs = [(A, B) for _, A, B, _ in HYPOTHESES]
    planned = ([(k, A, B) for k in ["late", "auc", "early_regular"] for A, B in ladder_pairs]
               + [(k, "mfec", B) for k in ["early", "late"] for B in ["nec", "dqn"]])
    for k, A, B in planned:
        x, y = col(A, k), col(B, k)
        x, y = x[~np.isnan(x)], y[~np.isnan(y)]
        if len(x) < 2 or len(y) < 2:
            continue
        ci = boot_ci(x, y)
        md.append(f"| {k.upper()} | {NAMES[A]} vs {NAMES[B]} | {x.mean() - y.mean():+.3f} "
                  f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | {hedges_g(x, y):.2f} | {perm_test(x, y):.2g} |")

    # Assumption checks (PROTOCOL.md section 9)
    md.append("\n## Assumption checks\n")
    md.append("### A1 Positive control: LATE above the random floor (one-sided reading of a two-sided test)\n")
    md.append("| agent | diff vs random [95% CI] | p (uncorrected) | learned? |\n|---|---|---|---|")
    r0 = col("random", "late")
    for a in LADDER + ["mfec"]:
        x = col(a, "late")
        x = x[~np.isnan(x)]
        if len(x) < 2 or len(r0) < 2:
            continue
        ci, p = boot_ci(x, r0), perm_test(x, r0)
        md.append(f"| {NAMES[a]} | {x.mean() - r0.mean():+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {p:.2g} | "
                  f"{'yes' if p < 0.05 and x.mean() > r0.mean() else '**no**'} |")
    exp_e, exp_l = (ehi - elo) // 10_000 * EVAL_EPISODES, (lhi - llo) // 10_000 * EVAL_EPISODES
    md.append(f"\n### A2 Evaluation episodes per window (min / max over runs; expected {exp_e} and {exp_l})\n")
    md.append("| agent | EARLY | LATE |\n|---|---|---|")
    for a in AGENTS:
        ne, nl = col(a, "n_early"), col(a, "n_late")
        if len(ne):
            bad = lambda x, k: "" if x.min() == x.max() == k else " **incomplete**"
            md.append(f"| {NAMES[a]} | {ne.min():.0f} / {ne.max():.0f}{bad(ne, exp_e)} | "
                      f"{nl.min():.0f} / {nl.max():.0f}{bad(nl, exp_l)} |")
    md.append("\n### Training episodes vs evaluation (descriptive): LATE score while training (agent's own epsilon) "
              "and on evaluation (epsilon 0.05)\n")
    md.append("| agent | training episodes per run | LATE, training episodes | LATE, evaluation |\n|---|---|---|---|")
    for a in AGENTS:
        if rows[a]:
            md.append(f"| {NAMES[a]} | {col(a, 'train_eps').mean():,.0f} | {fmt_cell(col(a, 'train_late'))} | "
                      f"{fmt_cell(col(a, 'late'))} |")
    md.append("\n### A3 How evaluation episodes end, and A4 whether bonus food matters (LATE window, mean ± s.d.)\n")
    md.append("| agent | steps per episode | cut off by idle limit | share of points from bonus | bonus pellets per episode |")
    md.append("|---|---|---|---|---|")
    for a in AGENTS:
        if rows[a]:
            md.append(f"| {NAMES[a]} | {fmt_cell(col(a, 'ep_len_late'), nd=0)} | "
                      f"{fmt_cell(col(a, 'trunc_share_late'), 100, 1)}% | {fmt_cell(col(a, 'bonus_share_late'), 100, 1)}% | "
                      f"{fmt_cell(col(a, 'bonus_per_ep_late'), nd=3)} |")
    md.append("\n### A5 Memory regime at the end of training (episodic agents; mean ± s.d.)\n")
    md.append("| agent | DND rows in use (of 3 × 20,000) | writes that evicted | writes that updated an exact match | "
              "first snapshot with an eviction (step) |\n|---|---|---|---|---|")
    for a in ["ec_frozen", "nec"]:
        if rows[a]:
            fa = col(a, "full_at")
            md.append(f"| {NAMES[a]} | {fmt_cell(col(a, 'dnd_fill'), 100, 1)}% | {fmt_cell(col(a, 'evict_share'), 100, 1)}% | "
                      f"{fmt_cell(col(a, 'exact_share'), 100, 1)}% | "
                      f"{'never' if np.all(np.isnan(fa)) else f'{np.nanmin(fa):,.0f} – {np.nanmax(fa):,.0f}'} |")
    md.append("\n### A7 Stability of the confirmatory verdicts: drop one run at a time\n")
    md.append(f"Each hypothesis is retested with every single run removed ({N_PERM_LOO:,} permutations each). "
              "Reported: the smallest absolute difference and the largest raw p seen, against that hypothesis's Holm "
              "threshold (0.05 / rank).\n")
    md.append("| ID | smallest absolute diff | largest raw p | Holm threshold | verdict holds for every drop? |")
    md.append("|---|---|---|---|---|")
    order = np.argsort(np.where(np.isnan(ps), np.inf, ps))
    thresh = {res[i][0]: 0.05 / (len(ps) - rank) for rank, i in enumerate(order)}
    for (hid, A, B, *_), pa in zip(res, adj):
        x, y = col(A, "early"), col(B, "early")
        x, y = x[~np.isnan(x)], y[~np.isnan(y)]
        if len(x) < 3 or len(y) < 3:
            continue
        drops = [(np.delete(x, i), y) for i in range(len(x))] + [(x, np.delete(y, j)) for j in range(len(y))]
        ds = [xx.mean() - yy.mean() for xx, yy in drops]
        pp = [perm_test(xx, yy, n_mc=N_PERM_LOO) for xx, yy in drops]
        full = verdict(x.mean() - y.mean(), pa)
        same = all((p < thresh[hid]) == (pa < 0.05) and np.sign(d) == np.sign(x.mean() - y.mean())
                   for d, p in zip(ds, pp)) if full != "not tested" else False
        md.append(f"| {hid} | {min(abs(d) for d in ds):.3f} | {max(pp):.2g} | {thresh[hid]:.3g} | "
                  f"{'yes' if same else '**no**'} |")

    with open(f"{out}/results.md", "w") as f:
        f.write("\n".join(md) + "\n")
    figure(data, rows, S, out)
    print(f"wrote {out}/per_run.csv, results.md, learning_curves.png")


def figure(data, rows, S, out):
    """Left: learning curves (mean ± s.e. over runs) with EARLY and LATE windows shaded. Right: EARLY per run."""
    (elo, ehi), (llo, lhi) = windows(S)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.7, 1]})
    edges = np.linspace(0, S, 26)
    x = edges[1:] / 1000
    ends = []
    for a in AGENTS:
        if not data[a]:
            continue
        C = np.stack([eval_curve(d, edges) for d in data[a]])
        mu, se = np.nanmean(C, 0), np.nanstd(C, 0) / np.sqrt(len(C))
        ref = a == "random"
        ax.plot(x, mu, color=COLORS[a], lw=1.5 if ref else 2, ls="--" if ref else "-")
        if not ref:
            ax.fill_between(x, mu - se, mu + se, color=COLORS[a], alpha=0.15, lw=0)
        ends.append([mu[-1], a])
    # Direct labels at the line ends, nudged apart so they never overlap.
    ends.sort()
    lo_y, hi_y = ax.get_ylim()
    gap = 0.045 * (hi_y - lo_y)
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for yv, a in ends:
        ax.annotate(f"{NAMES[a]} (n={len(data[a])})", (x[-1], yv), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color="#3d3d3a")
    for lo, hi, lab in [(elo, ehi, "EARLY"), (llo, lhi, "LATE")]:
        ax.axvspan(lo / 1000, hi / 1000, color="#8a8984", alpha=0.08, lw=0)
        ax.text((lo + hi) / 2000, 0.98, lab, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8, color="#6b6a64")
    ax.set_xlim(0, S / 1000 * 1.22)
    ax.set_xlabel("environment steps (thousands)")
    ax.set_ylabel("evaluation points per episode (fruit 1, bonus 5)")
    ax.set_title("Evaluation score (epsilon 0.05) during training, mean ± s.e. over runs", fontsize=10)

    # EARLY per run: one dot per run, the agent's mean as a short bar, its 95% bootstrap CI as a line.
    order = [a for a in AGENTS if rows[a]]
    rng = np.random.default_rng(0)
    for i, a in enumerate(order):
        v = np.array([r["early"] for r in rows[a]], float)
        v = v[~np.isnan(v)]
        if not len(v):
            continue
        bx.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=22, color=COLORS[a], alpha=0.75,
                   edgecolor="white", linewidth=0.8, zorder=3)
        lo, hi = boot_ci(v) if len(v) > 1 else (v[0], v[0])
        bx.plot([i + 0.25, i + 0.25], [lo, hi], color="#3d3d3a", lw=1.5)
        bx.plot([i + 0.17, i + 0.33], [v.mean()] * 2, color="#3d3d3a", lw=2)
    bx.set_xticks(range(len(order)))
    ticks = {"random": "Random", "dqn": "DQN", "dqn_nstep": "N-step\nDQN", "ec_frozen": "Frozen\nemb. EC",
             "nec": "NEC", "mfec": "MFEC"}
    bx.set_xticklabels([ticks[a] for a in order], fontsize=8)
    bx.set_ylabel("EARLY: evaluation points per episode")
    bx.set_title("EARLY per run (dots), mean and 95% CI (black)", fontsize=10)
    for a_ in (ax, bx):
        a_.grid(alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{out}/learning_curves.png", dpi=150)


def parse_seeds(s):
    lo, hi = s.split("-")
    return set(range(int(lo), int(hi) + 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gate", action="store_true", help="pilot gate instead of the main analysis")
    ap.add_argument("--root", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    if a.gate:
        gate(a.root or "results/exp_nec_ladder_paper_eps/pilot", parse_seeds(a.seeds or "1-3"))
    else:
        main(a.root or "results/exp_nec_ladder_paper_eps/main", parse_seeds(a.seeds or "201-215"), a.out)
