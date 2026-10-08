"""Watch an agent play: in the terminal (default), in a matplotlib window (--gui), or saved to a GIF (--gif).

usage: snake-watch AGENT [--episodes N] [--eps E] [--delay S] [--seed K] [--food-radius R [--relocate]]
                         [--gui | --gif OUT.gif [--max-frames N]] [--size n --map M --bonus B]
  AGENT is a checkpoint from `snake-run ... --save-agent` / `--save-every K` (<stem>.agent.pkl), which carries its own
  board settings, or a registered agent name (an untrained agent; give the board with --size/--map/--bonus).
  The agent acts as in evaluation (t=None: no learning, no memory bookkeeping) with ε = --eps (default 0: greedy),
  drawing from its own watch RNG, so a checkpoint can be watched any number of times without changing it.

Terminal keys: space pause/resume · + / - faster/slower · n next episode · q quit.
Side panel: the Q-value the agent assigns to each relative action (straight / right / left) at this step, and the
action it took; for NEC also the mean squared distance to the memory neighbours (how familiar the situation is).
"""
import argparse
import os
import pickle
import select
import sys
import time

import numpy as np

from snake_rl.agents import AGENTS, make_agent
from snake_rl.env import MAPS, Snake

ACTIONS = ("straight", "right", "left")
ARROWS = "↑→↓←"  # by env.dir: up, right, down, left
# 256-colour palette for the terminal grid; RGB twins for the matplotlib view.
TERM = {"empty": 236, "wall": 244, "body": 34, "head": 46, "food": 196, "bonus": 220}
RGB = {"empty": (0.12, 0.13, 0.15), "wall": (0.55, 0.58, 0.62), "body": (0.10, 0.62, 0.30),
       "head": (0.55, 0.95, 0.45), "food": (0.90, 0.22, 0.20), "bonus": (0.98, 0.78, 0.15)}


def load(spec, size, map, bonus, seed):
    """-> (agent, Snake settings, label)."""
    if os.path.exists(spec):
        with open(spec, "rb") as f:
            ck = pickle.load(f)
        board = dict(size=ck["size"], map=ck["map"], bonus=ck["bonus"], distractors=ck.get("distractors", 0))
        return ck["agent"], board, f"{ck['name']} seed {ck['seed']} @ {ck['t']:,} steps"
    if spec not in AGENTS:
        sys.exit(f"{spec!r} is neither a checkpoint file nor an agent ({', '.join(sorted(AGENTS))})")
    board = dict(size=size, map=map, bonus=bonus, distractors=0)
    agent = make_agent(spec, Snake(**board), np.random.default_rng(seed))
    return agent, board, f"{spec} (untrained)"


def q_values(agent, obs):
    """(Q per action, NEC neighbour distance) for display; Nones for agents without a probe_embed."""
    if not hasattr(agent, "probe_embed"):
        return None, None
    Z, Q = agent.probe_embed(obs[None])
    dist = None
    if hasattr(agent, "dnds") and all(d.n for d in agent.dnds):
        dist = float(np.mean([d.knn(Z)[1].mean() for d in agent.dnds]))
    return Q[0], dist


def frames(agent, env, eps, episodes, rng, food_radius):
    """Yields one dict per step (before the action is applied), then one per episode end."""
    saved, agent.rng = agent.rng, rng  # act() draws exploration from rng, never from the agent's training stream
    try:
        for ep in range(1, episodes + 1):
            env.food_radius = food_radius
            obs, step = env.reset(), 0
            while True:
                q, dist = q_values(agent, obs)
                a = agent.act(obs, eps, None)
                yield dict(ep=ep, step=step, q=q, dist=dist, a=a, end=None)
                obs, r, term, trunc = env.step(a)
                step += 1
                if term or trunc:
                    why = "idle limit" if trunc else ("board full" if r > 0 else "crashed")
                    yield dict(ep=ep, step=step, q=None, dist=None, a=None, end=why)
                    break
    finally:
        agent.rng = saved


def cells(env):
    """Board as an n x n array of cell kinds."""
    g = np.full((env.n, env.n), "empty", object)
    for r, c in env.wall_set:
        g[r, c] = "wall"
    for r, c in list(env.body)[1:]:
        g[r, c] = "body"
    if env.food is not None:
        g[env.food] = "food"
    if env.bonus_pos is not None:
        g[env.bonus_pos] = "bonus"
    g[env.body[0]] = "head"
    return g


def panel(f, env, label, eps, delay):
    lines = [f"\x1b[1m{label}\x1b[0m", f"ε {eps:g} · {1 / delay:.0f} steps/s" if delay > 0 else f"ε {eps:g}", "",
             f"episode {f['ep']}   step {f['step']}", f"score {env.score:g}   fruit {env.foods}   length {len(env.body)}",
             f"heading {ARROWS[env.dir]}   idle {env.idle}/{env.max_idle}"]
    if env.bonus_pos is not None:
        lines.append(f"\x1b[38;5;{TERM['bonus']}mbonus! {env.bonus_left} steps left\x1b[0m")
    lines.append("")
    if f["end"]:
        lines.append(f"\x1b[1;38;5;{TERM['food']}m■ episode over: {f['end']}\x1b[0m")
    elif f["q"] is not None:
        q = f["q"]
        lo, hi = min(q.min(), 0.0), max(q.max(), 1e-9)
        lines.append("Q-values (what the agent expects):")
        for i, name in enumerate(ACTIONS):
            bar = "█" * int(round(20 * (q[i] - lo) / (hi - lo + 1e-12)))
            mark = "\x1b[1m◀ taken\x1b[0m" if i == f["a"] else ""
            lines.append(f"  {name:8s} {q[i]:+7.3f} {bar:20s} {mark}")
        if f["dist"] is not None:
            lines.append(f"  memory distance {f['dist']:.3g}")
    else:
        lines.append(f"action: {ACTIONS[f['a']]}")
    lines += ["", "\x1b[2mspace pause · +/- speed · n next · q quit\x1b[0m"]
    return lines


def draw_terminal(f, env, label, eps, delay):
    g = cells(env)
    rows = ["".join(f"\x1b[48;5;{TERM[k]}m  " for k in row) + "\x1b[0m" for row in g]
    side = panel(f, env, label, eps, delay)
    out = ["\x1b[H"]
    for i in range(max(len(rows), len(side))):
        left = rows[i] if i < len(rows) else " " * (2 * env.n)
        right = side[i] if i < len(side) else ""
        out.append(f"{left}   {right}\x1b[K")
    sys.stdout.write("\n".join(out) + "\x1b[J")
    sys.stdout.flush()


def run_terminal(gen, env, label, eps, delay):
    tty = sys.stdin.isatty()
    if tty:
        import termios
        import tty as ttym
        fd, old = sys.stdin.fileno(), termios.tcgetattr(sys.stdin.fileno())
        ttym.setcbreak(fd)
    sys.stdout.write("\x1b[?25l\x1b[2J")  # hide cursor, clear
    paused, skip = False, False
    try:
        for f in gen:
            if skip and not f["end"]:
                continue
            skip = False
            draw_terminal(f, env, label, eps, delay)
            wait = delay * (6 if f["end"] else 1)
            deadline = time.time() + wait
            while tty and (paused or time.time() < deadline):
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not r:
                    continue
                k = sys.stdin.read(1)
                if k == "q":
                    return
                if k == " ":
                    paused = not paused
                elif k in "+=":
                    delay = max(delay / 1.5, 0.005)
                elif k in "-_":
                    delay = min(delay * 1.5, 2.0)
                elif k == "n":
                    skip, paused = True, False
                    break
                draw_terminal(f, env, label, eps, delay)
            if not tty:
                time.sleep(wait)
    except KeyboardInterrupt:
        pass
    finally:
        if tty:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\x1b[?25h\n")


def run_matplotlib(gen, env, label, eps, delay, gif=None, max_frames=600):
    import matplotlib
    if gif:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, (ax, axq) = plt.subplots(1, 2, figsize=(10, 5.4), gridspec_kw=dict(width_ratios=[1.35, 1]))
    img = ax.imshow(np.zeros((env.n, env.n, 3)), interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    title = ax.set_title(label, fontsize=10)
    bars = axq.barh(ACTIONS[::-1], [0, 0, 0], color="#8a8984")
    axq.set_title("Q-values (taken action highlighted)", fontsize=10)
    axq.axvline(0, color="#8a8984", lw=0.8)
    axq.spines[["top", "right"]].set_visible(False)
    status = axq.text(0, -1.1, "", fontsize=9, va="top", transform=axq.transData)
    state = {"paused": False}

    def update(f):
        g = cells(env)
        img.set_data(np.array([[RGB[k] for k in row] for row in g]))
        title.set_text(f"{label} · episode {f['ep']} · step {f['step']} · score {env.score:g}")
        if f["q"] is not None:
            q = f["q"]
            for i, b in enumerate(bars):  # bars are drawn bottom-up: index 0 is "left"
                a = 2 - i
                b.set_width(q[a])
                b.set_color("#2a78d6" if a == f["a"] else "#8a8984")
            lim = max(1e-3, np.abs(q).max()) * 1.15
            axq.set_xlim(-lim, lim)
        status.set_text(f"episode over: {f['end']}" if f["end"] else
                        (f"memory distance {f['dist']:.3g}" if f["dist"] is not None else ""))
        return [img, title, status, *bars]

    def on_key(e):
        if e.key == " ":
            state["paused"] = not state["paused"]
            (anim.pause if state["paused"] else anim.resume)()

    anim = FuncAnimation(fig, update, frames=gen, interval=delay * 1000, blit=False, cache_frame_data=False,
                         save_count=max_frames)
    fig.tight_layout()
    if gif:
        anim.save(gif, writer=PillowWriter(fps=max(1, int(round(1 / delay)))))
        print(f"-> {gif}")
    else:
        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()


def cli():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("agent", help="checkpoint (.agent.pkl) or agent name")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--eps", type=float, default=0.0, help="exploration while watching (default 0: greedy)")
    ap.add_argument("--delay", type=float, default=0.08, help="seconds per step")
    ap.add_argument("--seed", type=int, default=0, help="seed of the watch env and its exploration")
    ap.add_argument("--food-radius", type=int, default=None, help="place food within R steps (as in training)")
    ap.add_argument("--relocate", action="store_true", help="re-place uneaten food near the head (needs --food-radius)")
    ap.add_argument("--gui", action="store_true", help="matplotlib window instead of the terminal")
    ap.add_argument("--gif", default=None, help="save a GIF instead of showing")
    ap.add_argument("--max-frames", type=int, default=600, help="GIF length cap")
    ap.add_argument("--size", type=int, default=7, help="board for an untrained agent name")
    ap.add_argument("--map", default="open", choices=MAPS)
    ap.add_argument("--bonus", type=float, default=0.0)
    a = ap.parse_args()
    agent, board, label = load(a.agent, a.size, a.map, a.bonus, a.seed)
    env = Snake(seed=a.seed + 5000, relocate_food=a.relocate, **board)
    gen = frames(agent, env, a.eps, a.episodes, np.random.default_rng(a.seed + 6000), a.food_radius)
    if a.gif:
        import itertools
        run_matplotlib(itertools.islice(gen, a.max_frames), env, label, a.eps, a.delay, a.gif, a.max_frames)
    elif a.gui:
        run_matplotlib(gen, env, label, a.eps, a.delay)
    else:
        run_terminal(gen, env, label, a.eps, a.delay)


if __name__ == "__main__":
    cli()
