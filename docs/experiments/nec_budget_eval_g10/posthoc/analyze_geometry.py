"""Aggregate the post-hoc geometry analysis (snake_rl.geometry) of the nec_budget_eval_g10 checkpoints. EXPLORATORY: computed after the runs, from
checkpoints (PROTOCOL.md 3.5), not part of the pre-registered analysis.

usage: python docs/experiments/nec_budget_eval_g10/posthoc/analyze_geometry.py [geometry_dir] [runs_dir] [out_dir]
Writes geometry.md. Also validates every recomputed probe against the in-run probe of the same run and step.
"""
import glob, json, os, sys
import numpy as np

GEO = sys.argv[1] if len(sys.argv) > 1 else "results/geometry_budget_eval_g10"
RUNS = sys.argv[2] if len(sys.argv) > 2 else "results/exp_nec_budget_eval_g10"
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(os.path.abspath(__file__))


def cond(r):
    a = r["agent"]
    if a == "dqn_naturecnn": return "DQN"
    if a == "ec_frozen_naturecnn": return "frozen"
    return "NEC %g" % r["lr"]


ORDER = ["DQN", "frozen", "NEC 7.92e-06", "NEC 3e-05", "NEC 0.0001"]
rows = [json.load(open(f)) for f in sorted(glob.glob(GEO + "/*.geometry.json"))]
for r in rows: r["cond"] = cond(r)

# validation against the in-run probes
runs = {}
for f in glob.glob(RUNS + "/**/*_s[0-9]*.json", recursive=True):
    d = json.load(open(f)); runs[(os.path.basename(f)[:-5])] = dict(d["probes"])
ok = bad = 0
for r in rows:
    stem = r["checkpoint"].split(".agent")[0]
    inrun = runs.get(stem, {}).get(r["t"])
    if inrun is None: bad += 1; continue
    same = all(abs(r["probe"][k] - inrun[k]) < 1e-9 for k in ("steer", "knn_offset", "knn_config", "decode"))
    ok += same; bad += not same
TS = [60000, 120000, 180000, 240000, 250000]


def mean(c, t, get):
    v = [get(r) for r in rows if r["cond"] == c and r["t"] == t]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def amean(x): return float(np.mean([y for y in x if y is not None]))


L = ["# Post-hoc geometry of the saved agents (exploratory)", "",
     f"{len(rows)} checkpoints (5 per run: 60k, 120k, 180k, 240k, 250k) of `nec_budget_eval_g10`, analysed with `snake_rl.geometry` (20 episodes at epsilon 0.05 per checkpoint, up to 3,000 states).",
     f"**Validation:** the probe recomputed from the saved agent equals the run's own recorded probe at that step in {ok} of {ok + bad} checkpoints.", ""]


def table(title, get, fmt="%.2f", conds=ORDER):
    L.extend([f"### {title}", "", "| Condition | " + " | ".join(f"{t // 1000}k" for t in TS) + " |", "|---|" + "---|" * len(TS)])
    for c in conds:
        cells = [(fmt % v) if (v := mean(c, t, get)) is not None else "-" for t in TS]
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L.append("")


L += ["## Near probe, recomputed (mean over 10 seeds)", ""]
table("Steering (chance 0.39)", lambda r: r["probe"]["steer"])
table("knn_config: neighbours share the snake placement", lambda r: r["probe"]["knn_config"])
table("knn_offset: neighbours share the food offset (chance about 0.10)", lambda r: r["probe"]["knn_offset"])
table("decode: linear read-out of the food offset (chance about 0.10)", lambda r: r["probe"]["decode"])
L += ["## How much Q depends on the food within a fixed snake placement (ratio; 0 = not at all)", ""]
table("qfood ratio", lambda r: r["qfood"]["ratio"], "%.3f")
table("Q spread across actions (mean of max - min over probe states)", lambda r: r["qfood"]["q_spread"], "%.3f")
L += ["## Neighbour structure on the agent's own states (purity / chance, 10 nearest states of other episodes)", ""]
for lab, nm in (("head", "head cell"), ("food", "food cell"), ("offset_board", "food offset, board frame"), ("offset_ego", "food offset, snake frame")):
    L.extend([f"### {nm}", "", "| Condition | " + " | ".join(f"{t // 1000}k" for t in TS) + " |", "|---|" + "---|" * len(TS)])
    for c in ORDER:
        cells = []
        for t in TS:
            p = mean(c, t, lambda r: r["onpolicy"][lab]["purity"]); ch = mean(c, t, lambda r: r["onpolicy"][lab]["chance"])
            cells.append("-" if p is None else f"{p:.2f} / {ch:.2f}")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L.append("")
L += ["## Memory of NEC and `ec_frozen` (mean over actions and seeds)", ""]
mem = ["frozen", "NEC 7.92e-06", "NEC 3e-05", "NEC 0.0001"]
table("Coherence: s.d. of the 50 neighbours' values over the s.d. of all stored values (1 = no more alike than random rows)", lambda r: amean(r["memory"]["coherence"]) if "memory" in r else None, "%.3f", mem)
table("probe_vs_memory: distance of probe states to memory over memory to itself (1 = probe states look like memory)", lambda r: amean(r["memory"]["probe_vs_memory"]) if "memory" in r else None, "%.2f", mem)
table("Mean largest kernel weight (0.02 = uniform over 50 neighbours)", lambda r: amean(r["memory"]["kernel_peak"]) if "memory" in r else None, "%.3f", mem)
open(OUT + "/geometry.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
