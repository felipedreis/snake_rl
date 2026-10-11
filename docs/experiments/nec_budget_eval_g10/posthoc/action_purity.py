"""Action-label purity of the near-probe embeddings for every saved checkpoint of nec_budget_eval_g10 (EXPLORATORY; see REPORT.md).

usage: python docs/experiments/nec_budget_eval_g10/posthoc/action_purity.py [runs_dir] [out_json]
For each checkpoint: the share of a probe state's 50 nearest probe states (in embedding space) that agree with it on whether each action is a good move,
averaged over states and actions, and the chance rate. Writes one record per checkpoint. Uses snake_rl.geometry.action_purity.
"""
import glob, json, os, re, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

RUNS = sys.argv[1] if len(sys.argv) > 1 else "results/exp_nec_budget_eval_g10"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "action_purity.json")
_P = None


def one(path):
    global _P
    from snake_rl.geometry import action_purity, load
    from snake_rl.probe import make_probe_set
    ck, board = load(path)
    if _P is None:
        _P = make_probe_set(board["size"], board["map"], board["bonus"], render=board["render"], frame_stack=board["frame_stack"],
                            wall_scale=board["wall_scale"], actions=board["actions"])
    Z, _ = ck["agent"].probe_embed(_P["X"])
    r = action_purity(Z, _P)
    return dict(checkpoint=os.path.basename(path), agent=ck["name"], seed=ck["seed"], t=ck["t"], lr=ck.get("lr"), **r)


if __name__ == "__main__":
    paths = sorted(p for p in glob.glob(RUNS + "/**/*.agent*.pkl", recursive=True) if "/." not in p and "random" not in os.path.basename(p))
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(one, paths))
    json.dump(res, open(OUT, "w"))
    print(len(res), "checkpoints ->", OUT)
