"""Aggregate results/geometry_pacman/*.geometry.json (python -m snake_rl.geometry_pacman ...) into geometry.md: mean +- s.d. over seeds."""
import collections, glob, json, os
import numpy as np

SRC = "results/geometry_pacman"
g = collections.defaultdict(list)
for f in sorted(glob.glob(f"{SRC}/*.geometry.json")):
    d = json.load(open(f)); g[d["agent"]].append(d)
fmt = lambda v: f"{np.mean(v):.2f} ({np.std(v):.2f})"
rows = ["| Agent | Space | pac cell | nearest pellet | nearest ghost | power pellet on | action purity | greedy steering |", "|---|---|---|---|---|---|---|---|"]
for a, v in sorted(g.items()):
    for tag in ("embedding", "pixels"):
        if tag == "pixels" and a != "ec_frozen_naturecnn":
            continue  # the raw-pixel reference is nearly the same for every agent's visited states; shown once
        lab = lambda l: fmt([x[tag]["onpolicy"][l]["purity"] for x in v]) + f" [{np.mean([x[tag]['onpolicy'][l]['chance'] for x in v]):.2f}]"
        ap = fmt([x[tag]["action_purity"]["purity"] for x in v]) + f" [{np.mean([x[tag]['action_purity']['chance'] for x in v]):.2f}]"
        st = fmt([x["steering"]["accuracy"] for x in v]) + f" [{np.mean([x['steering']['random'] for x in v]):.2f}]" if tag == "embedding" else ""
        rows.append(f"| {a} | {'agent embedding' if tag == 'embedding' else 'raw pixels'} | {lab('pac')} | {lab('pellet')} | {lab('ghost')} | {lab('scared')} | {ap} | {st} |")
mem = ["| Agent | memory coherence | probe vs memory distance |", "|---|---|---|"]
for a, v in sorted(g.items()):
    if "memory" in v[0]:
        mem.append(f"| {a} | {fmt([np.mean([c for c in x['memory']['coherence'] if c]) for x in v])} | {fmt([np.mean([c for c in x['memory']['probe_vs_memory'] if c]) for x in v])} |")
open(os.path.join(os.path.dirname(__file__), "geometry.md"), "w").write(
    "Mean (s.d. over 5 seeds) [chance]. Purity = share of the 10 (action purity: 50) nearest states from other episodes that share the label.\n\n"
    + "\n".join(rows) + "\n\n" + "\n".join(mem) + "\n")
