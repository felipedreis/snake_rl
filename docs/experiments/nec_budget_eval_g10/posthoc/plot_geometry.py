"""Figures for the post-hoc geometry report (EXPLORATORY; see REPORT.md). Reads the per-checkpoint files written by snake_rl.geometry.

usage: python docs/experiments/nec_budget_eval_g10/posthoc/plot_geometry.py [geometry_dir] [out_dir]
Lines are means over runs at each saved checkpoint, bands are +-1 standard error over runs. NEC pools its three learning rates (30 runs); DQN and
the frozen-embedding agent have 10 runs each. Colours follow the entity (DQN blue, ec_frozen orange, NEC aqua) in every figure.
"""
import glob, json, os, sys, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GEO = sys.argv[1] if len(sys.argv) > 1 else "results/geometry_budget_eval_g10"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#8a8984"
COLORS = {"DQN": "#2a78d6", "ec_frozen": "#eb6834", "NEC": "#1baf7a"}  # validated palette slots 1-3 (validate_palette.js, all-pairs)
LABELS = {"DQN": "DQN", "ec_frozen": "ec_frozen (random encoder)", "NEC": "NEC (learned embedding)"}
TS = [60000, 120000, 180000, 240000, 250000]
XS = list(range(len(TS)))  # checkpoints are plotted evenly spaced
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})

rows = [json.load(open(f)) for f in sorted(glob.glob(GEO + "/*.geometry.json"))]
for r in rows:
    r["group"] = {"dqn_naturecnn": "DQN", "ec_frozen_naturecnn": "ec_frozen"}.get(r["agent"], "NEC")


def series(group, get):
    """-> mean, standard error per checkpoint step."""
    mu, se = [], []
    for t in TS:
        v = np.array([x for x in (get(r) for r in rows if r["group"] == group and r["t"] == t) if x is not None], float)
        mu.append(v.mean() if len(v) else np.nan); se.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
    return np.array(mu), np.array(se)


def style(ax, ylabel, xlabel=True):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.7, linestyle="-"); ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=INK2)
    if xlabel: ax.set_xlabel("training step (saved checkpoints)", color=INK2)
    ax.set_xticks(XS); ax.set_xticklabels([f"{t // 1000}k" for t in TS]); ax.set_xlim(-0.3, 4.35)


def draw(ax, groups, get, fmt="%.2f"):
    """Mean +- 1 s.e. per checkpoint. The value at the last checkpoint is written in the legend, so converging lines never collide."""
    for g in groups:
        mu, se = series(g, get); c = COLORS[g]
        ax.fill_between(XS, mu - se, mu + se, color=c, alpha=0.14, linewidth=0)
        ax.plot(XS, mu, color=c, linewidth=1.9, solid_capstyle="round", label=f"{LABELS[g]}  ({fmt % mu[-1]} at 250k)")
        ax.plot(XS[-1], mu[-1], "o", color=c, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=5)


def reference(ax, y, text, x=-0.26, va="bottom"):
    ax.axhline(y, color=MUTED, linewidth=1.0, zorder=1)
    ax.annotate(text, (x, y), xytext=(0, 3 if va == "bottom" else -3), textcoords="offset points", va=va, ha="left", color=INK2, fontsize=8.5)


def title(fig, main, sub, width=105):
    fig.text(0.012, 0.975, main, color=INK, fontsize=12.5, fontweight="bold", va="top", ha="left")
    fig.text(0.012, 0.915, textwrap.fill(sub, width), color=INK2, fontsize=9.5, va="top", ha="left", linespacing=1.35)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150); plt.close(fig); print("wrote", name)


# 1. Q-values barely separate actions
fig, ax = plt.subplots(figsize=(7.4, 4.3)); fig.subplots_adjust(top=0.80, bottom=0.14, left=0.1, right=0.93)
draw(ax, ["DQN", "ec_frozen", "NEC"], lambda r: r["qfood"]["q_spread"]); style(ax, "Q spread across the four actions"); ax.set_ylim(0, 2.5)
ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
title(fig, "NEC's Q-values barely tell the four actions apart", "Mean of (best Q - worst Q) over 1,282 probe states. A flat Q cannot steer; DQN's spread grows with training.", width=105)
save(fig, "01_q_spread.png")

# 2. memory coherence
fig, ax = plt.subplots(figsize=(7.4, 4.3)); fig.subplots_adjust(top=0.80, bottom=0.14, left=0.1, right=0.93)
draw(ax, ["ec_frozen", "NEC"], lambda r: float(np.mean([c for c in r["memory"]["coherence"] if c is not None])) if "memory" in r else None, "%.2f")
style(ax, "Coherence (1 = no regional structure)"); ax.set_ylim(0, 1.2)
reference(ax, 1.0, "1 = the 50 nearest values are no more alike than any 50 stored values", va="bottom")
reference(ax, 0.0, "0 = the 50 nearest values all agree", va="bottom")
ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.5), frameon=False, fontsize=9, labelcolor=INK)
title(fig, "A state's nearest memories do not agree on its value", "Coherence = s.d. of the 50 nearest stored values / s.d. of all stored values in that action's memory (1,282 probe states, mean over actions).", width=105)
save(fig, "02_memory_coherence.png")

# 3. what the embedding groups by, on the agents' own states
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3)); fig.subplots_adjust(top=0.78, bottom=0.14, left=0.065, right=0.95, wspace=0.28)
for ax, lab, yl, ymax in ((axes[0], "head", "Neighbours sharing the head cell", 0.9), (axes[1], "offset_board", "Neighbours sharing the food offset", 0.2)):
    draw(ax, ["DQN", "ec_frozen", "NEC"], lambda r, l=lab: r["onpolicy"][l]["purity"]); style(ax, yl); ax.set_ylim(0, ymax)
    chance = np.mean([r["onpolicy"][lab]["chance"] for r in rows]); reference(ax, chance, f"chance {chance:.2f}", va="bottom")
axes[0].set_title("Where the snake's head is", color=INK, fontsize=10.5, loc="left", pad=6)
axes[1].set_title("Where the food is, relative to the head (board frame)", color=INK, fontsize=10.5, loc="left", pad=6)
axes[0].legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK)
title(fig, "Every agent groups its states by head position; none groups them well by food position",
      "For each state the agent visited: share of its 10 nearest embeddings (taken from other episodes) with the same label. Grey line = share expected by chance.", width=170)
save(fig, "03_own_states_grouping.png")

# 4. can the food offset be read out; does the agent steer
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3)); fig.subplots_adjust(top=0.78, bottom=0.14, left=0.065, right=0.95, wspace=0.28)
draw(axes[0], ["DQN", "ec_frozen", "NEC"], lambda r: r["probe"]["decode"]); style(axes[0], "Accuracy of a linear read-out"); axes[0].set_ylim(0, 0.32)
reference(axes[0], 0.10, "chance 0.10", va="top")
draw(axes[1], ["DQN", "ec_frozen", "NEC"], lambda r: r["probe"]["steer"]); style(axes[1], "Share of good first moves"); axes[1].set_ylim(0.3, 0.85)
reference(axes[1], 0.39, "chance 0.39", va="top")
axes[0].set_title("Can the food offset be read from the embedding?", color=INK, fontsize=10.5, loc="left", pad=6)
axes[1].set_title("Does the agent steer toward food 1-2 steps away?", color=INK, fontsize=10.5, loc="left", pad=6)
axes[0].legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
title(fig, "Only DQN carries the food offset and uses it to steer", "Both measured on the 1,282-state near probe (snake of length 3, food within 2 walkable steps).", width=170)
save(fig, "04_food_readable_and_steering.png")

# 5. probe distance to memory
fig, ax = plt.subplots(figsize=(7.4, 4.3)); fig.subplots_adjust(top=0.80, bottom=0.14, left=0.1, right=0.93)
draw(ax, ["ec_frozen", "NEC"], lambda r: float(np.mean([c for c in r["memory"]["probe_vs_memory"] if c is not None])) if "memory" in r else None, "%.1f")
style(ax, "Distance ratio (1 = in distribution)"); ax.set_ylim(0, 5.3)
reference(ax, 1.0, "1 = probe states are as close to memory as memory is to itself", va="bottom")
ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK)
title(fig, "The probe's states are far from what NEC has stored", "Mean distance from each probe state to its nearest stored key, over the mean distance from a stored key to its nearest other stored key. NEC pools three learning rates (3.1 to 4.1 at 250k).", width=105)
save(fig, "05_probe_vs_memory.png")

# 6. toy illustration (made-up points, not data)
def toy(seed, n=150, yspread=0.3, delta=1e-3):
    r = np.random.default_rng(seed)
    X = np.column_stack([r.uniform(0, 10, n), r.uniform(-yspread, yspread, n)])      # x: where the snake is (large spread); y: where the food is (small spread)
    V = np.where(X[:, 1] > 0, 1.0, -1.0) + r.normal(0, 0.1, n)                       # value of "go up": good when the food is above
    q = np.array([5.0, 0.2]); d = ((X - q) ** 2).sum(1); nn = np.argsort(d)[:50]
    w = 1.0 / (d[nn] + delta); est = float((w * V[nn]).sum() / w.sum())             # NEC's kernel average over the 50 nearest
    return X, V, q, nn, est


X, V, q, nn, est = toy(3)
ests = [toy(sd)[4] for sd in range(200)]
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6)); fig.subplots_adjust(top=0.76, bottom=0.14, left=0.105, right=0.97, wspace=0.1)
cmap = {1: "#2a78d6", -1: "#eb6834"}
for ax in axes:
    ax.scatter(X[:, 0], X[:, 1], s=16, c=[cmap[1] if v > 0 else cmap[-1] for v in V], alpha=0.4, linewidths=0)
    ax.plot(*q, "o", color=INK, markersize=9, markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=6)
    ax.annotate("new state: the food is above,\nso the right answer is +1", q, xytext=(10, 22), textcoords="offset points", color=INK, fontsize=9)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax.set_xlabel("where the snake is  (varies a lot)", color=INK2); ax.set_xlim(-0.2, 10.2); ax.set_ylim(-0.62, 0.62)
    ax.set_xticks([]); ax.set_yticks([-0.3, 0, 0.3]); ax.set_yticklabels(["food below", "", "food above"])
axes[0].set_ylabel("where the food is  (varies little)", color=INK2); axes[1].set_yticklabels([])
a0 = axes[0]
a0.scatter(X[nn, 0], X[nn, 1], s=40, facecolors="none", edgecolors=INK, linewidths=1.0, zorder=5)
a0.set_title("Nearest-neighbour read-out (NEC)", color=INK, fontsize=10.5, loc="left", pad=6)
a0.annotate(f"The 50 nearest points (ringed) share the snake's position\nand have the food on both sides. Their kernel-weighted\naverage is {est:+.2f} here (mean over 200 random draws:\n{np.mean(ests):+.2f}), well short of the right answer, +1.",
            (0.1, -0.60), color=INK, fontsize=8.6, va="bottom")
a1 = axes[1]
a1.set_title("Learned linear read-out (DQN)", color=INK, fontsize=10.5, loc="left", pad=6)
a1.annotate("", xy=(9.6, 0.5), xytext=(9.6, -0.5), arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.6))
a1.annotate("It can look along this one direction,\nso Q follows the food's height: +1 here.", (9.4, -0.40), ha="right", color=INK, fontsize=9, va="center")
title(fig, "Why averaging the nearest stored values dilutes the food (toy illustration with made-up points, not our data)",
      "150 stored states. Blue: states where \"up\" was good (food above). Orange: where it was bad (food below). Distance is dominated by the snake's position.", width=170)
save(fig, "00_toy_readout.png")
print("toy: this draw %.3f, mean over 200 draws %.3f (s.d. %.3f); the right answer is +1" % (est, np.mean(ests), np.std(ests)))
