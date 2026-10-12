"""O1 table, ghosts/pellets, evictions and a figure for pacman_c2c3 against the base arms of pacman_calibration and the ladder.
Run from the repo root (needs results/exp_pacman_{ladder,calibration,c2c3} pulled)."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L, B, C = "results/exp_pacman_ladder/pac_px/", "results/exp_pacman_calibration/", "results/exp_pacman_c2c3/"
R = "dqn_naturecnn_rmsprop_lr0.00025"
ARMS = [("ladder DQN Adam 5e-4", L + "dqn_naturecnn_s?.json"),
        ("A   DQN RMSProp, small rewards, sync 1000 (base)", B + f"pac_px/{R}_s?.json"),
        ("A4  ... sync 4000", C + f"pac_px/{R}_target_every4000_s?.json"),
        ("A40 ... sync 40000", C + f"pac_px/{R}_target_every40000_s?.json"),
        ("B   DQN RMSProp, Atari units, clipped, sync 1000 (base)", B + f"pac_px_atari_clip/{R}_s?.json"),
        ("B4  ... sync 4000", C + f"pac_px_atari_clip/{R}_target_every4000_s?.json"),
        ("B40 ... sync 40000", C + f"pac_px_atari_clip/{R}_target_every40000_s?.json"),
        ("ladder NEC 1e-4, small rewards", L + "nec_naturecnn_rmsprop_lr0.0001_s?.json"),
        ("D   NEC 1e-4, Atari units (base)", B + "pac_px_atari/nec_naturecnn_rmsprop_lr0.0001_s?.json"),
        ("D1  ... memory 1e5 (no eviction)", C + "pac_px_atari/nec_naturecnn_rmsprop_lr0.0001_dnd_cap100000_s?.json"),
        ("D2  ... memory 1e5 and N = 100", C + "pac_px_atari/nec_naturecnn_rmsprop_lr0.0001_N100_dnd_cap100000_s?.json")]
rows = ["| Arm | O1 (points), mean (s.d.) | Seeds | Eval episode length | Ghosts / ep | Pellets / ep | Memory rows (evictions) |", "|---|---|---|---|---|---|---|"]
fig, ax = plt.subplots(figsize=(9, 5.2))
for name, pat in ARMS:
    runs = [json.load(open(f)) for f in sorted(glob.glob(pat))]
    ev = np.array([[np.mean([e[0] for e in x[1]]) for x in r["evaluations"]] for r in runs])
    o1 = ev[:, -3:].mean(1)
    ln = np.mean([np.mean([e[3] for e in r["evaluations"][-1][1]]) for r in runs])
    gh, pe, mem = [], [], []
    for r in runs:
        e = np.array(r["episodes"]); h = e[e[:, 0] > r["steps"] / 2]
        x = (h[:, 1] - 10 * h[:, 2]) / 40  # score - 10 foods = 40 power + 200 ghosts, power <= 4
        gh.append(np.floor(x / 5).mean()); pe.append((h[:, 2] - x % 5).mean())
        dg = r["diagnostics"][-1][1] if r["diagnostics"] else None
        if dg and "n" in dg.get("0", {}):
            mem.append((sum(v["n"] for v in dg.values()), sum(v.get("evictions", 0) for v in dg.values())))
    m = f"{np.mean([a for a, _ in mem]):.0f} ({np.mean([b for _, b in mem]):.0f})" if mem else "-"
    rows.append(f"| {name} | {o1.mean():.0f} ({o1.std():.0f}) | {', '.join(f'{v:.0f}' for v in o1)} | {ln:.0f} | {np.mean(gh):.2f} | {np.mean(pe):.1f} | {m} |")
    ax.plot(np.array([x[0] for x in runs[0]["evaluations"]]) / 1000, ev.mean(0), label=" ".join(name.split()),
            ls="--" if "ladder" in name else "-")
ax.set_xlabel("training steps (thousands)"); ax.set_ylabel("evaluation score (points, 20 episodes, eps 0.05)")
ax.grid(alpha=0.3); ax.legend(fontsize=6, ncol=1, loc="upper left"); fig.tight_layout()
here = os.path.dirname(__file__)
fig.savefig(os.path.join(here, "eval_curves.png"), dpi=150)
open(os.path.join(here, "results.md"), "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
