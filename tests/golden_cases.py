"""Golden runs: short runs whose outputs must never change under a refactor (bit-for-bit, one BLAS thread).

Each case is the command line of one run (without the results root). When a refactor renames a flag or an agent, update
the arguments here; never regenerate the expected files to make a test pass. `python tests/golden_cases.py` writes them
(done once, at the `pre-cleanup` tag).
"""
import glob
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "golden")

# Fields that are the run's behaviour. Header fields (names of settings) may change; these may not.
FIELDS = ("episodes", "evaluations", "probes", "diagnostics", "train")

PX = ["--render", "pixels", "--eval-every", "1000", "--eval-episodes", "2", "--probe-every", "1000", "--probe-far"]
CASES = {
    # the issue's "done when" set: 7x7 pixels with evaluation and both probes
    "px_dqn": ["snake-run", "dqn_naturecnn", "1", "2000", *PX],
    "px_nec": ["snake-run", "nec_naturecnn", "1", "2000", *PX],
    "px_ec_frozen": ["snake-run", "ec_frozen_naturecnn", "1", "2000", *PX],
    "px_mfec": ["snake-run", "mfec", "1", "2000", *PX],
    "px_random": ["snake-run", "random", "1", "2000", *PX],
    "px_nec_rmsprop": ["snake-run", "nec_naturecnn", "2", "1500", "--render", "pixels", "--opt", "rmsprop", "--lr", "1e-4"],
    # grid (MLP) agents
    "grid_dqn": ["snake-run", "dqn", "1", "3000", "--probe-every", "1000"],
    "grid_dqn_nstep": ["snake-run", "dqn_nstep", "1", "3000"],
    "grid_ec_frozen": ["snake-run", "ec_frozen", "1", "3000"],
    "grid_nec": ["snake-run", "nec", "1", "3000", "--probe-every", "1000"],
    "grid_nec_board": ["snake-run", "nec", "3", "3000", "--size", "9", "--map", "pillars", "--bonus", "3",
                       "--food-curriculum", "2:1000:2500", "--food-relocate", "--eps-floor", "0.05",
                       "--eps-decay", "1000", "--eval-every", "1500", "--eval-episodes", "2"],
    # Ms. Pac-Man
    "pac_dqn": ["snake-run", "dqn_naturecnn", "1", "1500", "--game", "pacman", "--reward-units", "atari", "--clip-reward",
                "--agent-kw", "target_every=500", "--eval-every", "750", "--eval-episodes", "2"],
    "pac_mfec": ["snake-run", "mfec", "1", "1500", "--game", "pacman"],
}


def run_case(argv, root):
    """Run one case through its console entry point, return the parsed result JSON."""
    from snake_rl import run
    cmd, *args = argv
    entry = {"snake-run": run.cli, "pacman-run": getattr(run, "pacman_cli", None)}[cmd]
    old = sys.argv
    sys.argv = [cmd, *args, "--results", root, "--train-log-every", "500"]
    try:
        entry()
    finally:
        sys.argv = old
    (path,) = glob.glob(f"{root}/**/*.json", recursive=True)
    with open(path) as f:
        d = json.load(f)
    return {k: d.get(k) for k in FIELDS}


# Checkpoints pickled at `pre-cleanup` (grid MLP agents: small files). Later code must still load them and give the same
# Q-values on the first observation of a fresh env.
CHECKPOINTS = ("dqn", "nec")


def checkpoint_q(agent, obs):
    """Q-values of a loaded agent for one observation, through the same read-only path the probe uses."""
    return agent.probe_embed(obs[None])[1][0].tolist()


def make_checkpoints():
    import pickle
    import shutil
    from snake_rl.env import Snake
    for name in CHECKPOINTS:
        with tempfile.TemporaryDirectory() as tmp:
            kw = ["--agent-kw", "dnd_cap=1000"] if name == "nec" else []  # small file
            run_case(["snake-run", name, "5", "1500", "--save-agent", *kw], tmp)
            (pkl,) = glob.glob(f"{tmp}/**/*.agent.pkl", recursive=True)
            shutil.copy(pkl, os.path.join(GOLDEN, f"ckpt_{name}.pkl"))
        ck = pickle.load(open(os.path.join(GOLDEN, f"ckpt_{name}.pkl"), "rb"))
        obs = Snake(size=7, seed=9).reset()
        with open(os.path.join(GOLDEN, f"ckpt_{name}.q.json"), "w") as f:
            json.dump(checkpoint_q(ck["agent"], obs), f)


if __name__ == "__main__":
    os.makedirs(GOLDEN, exist_ok=True)
    if sys.argv[1:] == ["checkpoints"]:
        make_checkpoints()
        sys.exit()
    for name in sys.argv[1:] or CASES:
        with tempfile.TemporaryDirectory() as tmp:
            out = run_case(CASES[name], tmp)
        with open(os.path.join(GOLDEN, f"{name}.json"), "w") as f:
            json.dump(out, f)
        print(name, len(out["episodes"]), "episodes")
