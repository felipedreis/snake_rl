"""Tables for the explosion ablation and the RMSProp control (PLAN.md).

usage (from the repo root): .venv/bin/python docs/experiments/grid_ablation_rmsprop_control/analyze.py [--out DIR]
Writes <out>/results.md (default: this directory). Safe on partial results.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R1, R2, SWEEP = "results/exp_grid_ablation", "results/exp_grid_rms_control", "results/exp_pixels_rmsprop"
DIVERGED = 100.0  # key norm above this in any 1,000-step window (it starts near 1)

PART1 = [("g_base", "grid: baseline (walls 1.0)"), ("g_ws235", "grid: walls 0.235"), ("g_ws0", "grid: walls 0 (hidden)"),
         ("g_fs4", "grid: 4-frame stack"), ("g_ws235_fs4", "grid: walls 0.235 + 4-frame stack"),
         ("g_rms", "grid: RMSProp 2.5e-4"), ("g_open", "grid: open map (no walls)"),
         ("px_ws0", "pixels: wall gray 0"), ("px_ws1", "pixels: wall gray 60 (default)"), ("px_ws425", "pixels: wall gray 255")]


def variant1(d):
    px, ws, fs = d.get("render") == "pixels", d.get("wall_scale", 1.0), d.get("frame_stack")
    if px:
        return {0.0: "px_ws0", 1.0: "px_ws1", 4.25: "px_ws425"}.get(ws)
    if d.get("opt", "adam") == "rmsprop":
        return "g_rms"
    if d.get("map") == "open":
        return "g_open"
    return {(1.0, None): "g_base", (0.235, None): "g_ws235", (0.0, None): "g_ws0",
            (1.0, 4): "g_fs4", (0.235, 4): "g_ws235_fs4"}.get((ws, fs))


def keynorms(d):
    return {t: s.get("key_norm", np.nan) for t, s in d.get("train", [])}


def at(kn, t):
    return kn.get(t, np.nan)


def fruit100(d, lo):
    e = np.array(d["episodes"])
    m = e[:, 0] > lo
    return float(e[m, 2].sum() * 100.0 / (d["steps"] - lo))


def part1(L):
    runs = defaultdict(list)
    for f in sorted(glob.glob(f"{R1}/*/*_s[0-9].json")):
        d = json.load(open(f))
        v = variant1(d)
        if v:
            runs[v].append(d)
    L += ["## Part 1: what triggers the explosion? (NEC, 25×25 `rooms`, held phase, 20k steps)", "",
          f"Diverged = key norm above {DIVERGED:g} in any 1,000-step window. Key norm cells: median over seeds [min, max].", "",
          "| variant | diverged (of n) | key norm at 1k | at 5k | at 20k | max over run | dead ReLU at 20k | fruit / 100 steps (5k–20k) |",
          "|---|---|---|---|---|---|---|---|"]
    for v, name in PART1:
        ds = runs.get(v, [])
        if not ds:
            continue
        kns = [keynorms(d) for d in ds]
        mx = [np.nanmax(list(k.values())) for k in kns]
        cell = lambda t: (lambda xs: f"{np.median(xs):.3g} [{min(xs):.3g}, {max(xs):.3g}]")([at(k, t) for k in kns])
        dead = [d["train"][-1][1].get("dead_relu", np.nan) for d in ds]
        L.append(f"| {name} | {sum(m > DIVERGED for m in mx)} of {len(ds)} | {cell(1000)} | {cell(5000)} | {cell(20000)} | "
                 f"{np.median(mx):.3g} | {np.mean(dead):.2f} | {np.mean([fruit100(d, 5000) for d in ds]):.1f} |")
    L.append("")


def late(d):
    e = np.array(d["episodes"])
    T = d["steps"]
    if d.get("map", "open") == "rooms":
        m = e[:, 0] > 50_000
        return float(e[m, 2].sum() * 100.0 / (T - 50_000))
    m = e[:, 0] > T / 2
    return float(e[m, 1].mean())


def probe_last(d, k):
    pr = d.get("probes") or []
    return float(pr[-1][1].get(k, np.nan)) if pr and pr[-1][1] else np.nan


def variant2(d):
    px, opt = d.get("render") == "pixels", d.get("opt", "adam")
    fam = {"nec_dqncnn": "NEC", "dqn_dqncnn": "DQN", "nec_naturecnn": "NEC", "dqn_naturecnn": "DQN"}.get(d["agent"])
    if not fam:
        return None
    return f"{fam}, {'pixels (Nature CNN)' if px else 'grid (DQN-shaped CNN)'}, " + ("Adam 5e-4" if opt == "adam" else "RMSProp 2.5e-4")


def setting(d):
    return f"{d['size']}x{d['size']}" + (" rooms (held phase)" if d.get("map", "open") == "rooms" else " open")


def fmt(xs, p=2):
    xs = [x for x in xs if not np.isnan(x)]
    return "–" if not xs else f"{np.mean(xs):.{p}f} ({', '.join(f'{x:.{p}f}' for x in xs)})" if len(xs) > 1 else f"{xs[0]:.{p}f}"


def part2(L):
    runs = defaultdict(list)
    for root in (R2, SWEEP):
        for f in sorted(glob.glob(f"{root}/*/*_s[0-9].json")):
            d = json.load(open(f))
            v = variant2(d)
            if v and d.get("lr") in (None, 2.5e-4) and d.get("wall_scale", 1.0) == 1.0 and d["steps"] in (60000, 100000):
                runs[(setting(d), v)].append(d)
    L += ["## Part 2: RMSProp control, grid vs pixels", "",
          "Open boards: mean score per episode over the second half (60k steps). 25×25: fruit per 100 training steps over steps 50k–100k. "
          "Per-seed values in brackets; the pixel rows come from `results/exp_pixels_rmsprop`.", ""]
    for s in sorted({k[0] for k in runs}, key=lambda s: int(s.split("x")[0])):
        L += [f"### {s}", "", "| agent | score | key norm start → end | dead ReLU | probe steer | probe decode |", "|---|---|---|---|---|---|"]
        for (ss, v), ds in sorted(runs.items(), key=lambda kv: kv[0][1]):
            if ss != s:
                continue
            kn = [f"{d['train'][0][1].get('key_norm', np.nan):.2f} → {d['train'][-1][1].get('key_norm', np.nan):.3g}"
                  for d in ds if d["agent"].startswith("nec") and d.get("train")]
            L.append(f"| {v} | {fmt([late(d) for d in ds])} | {'; '.join(kn) or '–'} | "
                     f"{fmt([d['train'][-1][1].get('dead_relu', np.nan) for d in ds if d.get('train')])} | "
                     f"{fmt([probe_last(d, 'steer') for d in ds], 3)} | {fmt([probe_last(d, 'decode') for d in ds], 3)} |")
        L.append("")


def main(out):
    L = ["# Results: explosion ablation and RMSProp control", "", "Generated by analyze.py. Exploratory (PLAN.md); 1–3 seeds.", ""]
    part1(L)
    part2(L)
    open(f"{out}/results.md", "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HERE)
    main(ap.parse_args().out)
