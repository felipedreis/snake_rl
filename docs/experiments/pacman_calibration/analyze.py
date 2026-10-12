"""O1 table, ghost counts and a combined evaluation-score figure for pacman_calibration and the earlier pacman_ladder.
Run from the repo root: python docs/experiments/pacman_calibration/analyze.py (needs results/exp_pacman_* pulled)."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L, C = "results/exp_pacman_ladder/pac_px/", "results/exp_pacman_calibration/"
ARMS = [("random (ladder)", L + "random_s?.json"),
        ("DQN Adam 5e-4 (ladder)", L + "dqn_naturecnn_s?.json"),
        ("A  DQN RMSProp 2.5e-4", C + "pac_px/dqn_naturecnn_rmsprop_lr0.00025_s?.json"),
        ("B  DQN RMSProp, Atari units, clipped", C + "pac_px_atari_clip/dqn_naturecnn_rmsprop_lr0.00025_s?.json"),
        ("NEC 1e-4, small rewards (ladder)", L + "nec_naturecnn_rmsprop_lr0.0001_s?.json"),
        ("C  NEC 7.92e-6, Atari units", C + "pac_px_atari/nec_naturecnn_rmsprop_lr7.92e-06_s?.json"),
        ("D  NEC 1e-4, Atari units", C + "pac_px_atari/nec_naturecnn_rmsprop_lr0.0001_s?.json"),
        ("ec_frozen, small rewards (ladder)", L + "ec_frozen_naturecnn_s?.json"),
        ("E  ec_frozen, Atari units", C + "pac_px_atari/ec_frozen_naturecnn_s?.json"),
        ("MFEC, small rewards (ladder)", L + "mfec_s?.json"),
        ("F  MFEC, Atari units", C + "pac_px_atari/mfec_s?.json")]
rows = ["| Arm | O1 (points), mean (s.d.) | Seeds | Eval episode length | Ghosts eaten / episode | Pellets / episode |", "|---|---|---|---|---|---|"]
fig, ax = plt.subplots(figsize=(9, 5.2))
for name, pat in ARMS:
    runs = [json.load(open(f)) for f in sorted(glob.glob(pat))]
    ev = np.array([[np.mean([e[0] for e in x[1]]) for x in r["evaluations"]] for r in runs])
    steps = [x[0] for x in runs[0]["evaluations"]]
    o1 = ev[:, -3:].mean(1)
    length = np.mean([np.mean([e[3] for e in r["evaluations"][-1][1]]) for r in runs])
    gh, pe = [], []
    for r in runs:
        e = np.array(r["episodes"]); h = e[e[:, 0] > r["steps"] / 2]
        x = (h[:, 1] - 10 * h[:, 2]) / 40  # score - 10 foods = 40 power + 200 ghosts, power <= 4
        gh.append(np.floor(x / 5).mean()); pe.append((h[:, 2] - x % 5).mean())
    rows.append(f"| {name} | {o1.mean():.0f} ({o1.std():.0f}) | {', '.join(f'{v:.0f}' for v in o1)} | {length:.0f} | {np.mean(gh):.2f} | {np.mean(pe):.1f} |")
    ax.plot(np.array(steps) / 1000, ev.mean(0), label=name, ls="--" if "ladder" in name else "-")
ax.set_xlabel("training steps (thousands)"); ax.set_ylabel("evaluation score (points, 20 episodes, eps 0.05)")
ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2); fig.tight_layout()
here = os.path.dirname(__file__)
fig.savefig(os.path.join(here, "eval_curves.png"), dpi=150)
open(os.path.join(here, "results.md"), "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
