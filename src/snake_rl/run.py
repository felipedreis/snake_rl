"""Run one (agent, seed) and save per-episode scores.

usage: snake-run <agent> <seed> <steps> [--distractors D] [--size n] [--eps-floor F] [--eps-decay N]
                 [--map M] [--bonus R] [--render grid|pixels] [--frame-stack K] [--wall-scale X] [--actions relative|absolute] [--probe-every K [--probe-far]] [--game snake|pacman [--frame-skip K] [--reward-units small|atari]] [--clip-reward] [--opt adam|rmsprop] [--lr F] [--food-curriculum R[:H]:N [--food-relocate]] [--eval-every K [--eval-episodes M] [--eval-eps E]]
                 [--probe-every K] [--train-log-every K] [--save-agent] [--save-every K]
       (or: python -m snake_rl.run ...)
agents: see snake_rl.agents.AGENTS
"""
import argparse
import json
import os
import pickle
import time

import numpy as np

from snake_rl.agents import AGENTS, make_agent
from snake_rl.env import ACTION_MODES, MAPS, RENDERS, Snake
from snake_rl.pacman import MsPacman
from snake_rl.probe import make_far_probe_set, make_probe_set, probe


EPS_FLOOR = 0.02
GAMES = ("snake", "pacman")


def epsilon(t, floor=EPS_FLOOR, decay=None):
    """Exploration rate at step t.

    decay=None: the original schedule, kept bit-for-bit so old runs reproduce: 1 - t/5000 until it reaches
    the floor. decay=N: linear from 1 to the floor over exactly N steps, then the floor (DQN's schedule).
    decay=0: the floor from the start, i.e. a fixed epsilon (MFEC's schedule).
    """
    if decay is None:
        return max(floor, 1.0 - t / 5000)
    return floor if t >= decay else 1.0 + (floor - 1.0) * t / decay


def food_radius(t, curriculum, size):
    """Food-distance curriculum. (r0, n): the radius grows linearly from r0 to 2*size over n steps.
    (r0, hold, n): it stays at r0 until step hold, then grows to 2*size by step n. From step n on, food goes
    anywhere (None), as in the real game. curriculum=None: always anywhere."""
    if curriculum is None or t >= curriculum[-1]:
        return None
    r0, hold, n = curriculum if len(curriculum) == 3 else (curriculum[0], 0, curriculum[1])
    return r0 if t < hold else int(r0 + (2 * size - r0) * (t - hold) / (n - hold))


def run_name(name, eps_floor=EPS_FLOOR, eps_decay=None, food_curriculum=None, food_relocate=False, opt="adam", lr=None):
    """Result file stem (minus seed): the agent name, tagged with any non-default run settings."""
    return (name + ("" if opt == "adam" else f"_{opt}") + ("" if lr is None else f"_lr{lr:g}")
            + ("" if eps_floor == EPS_FLOOR else f"_eps{eps_floor:g}")
            + ("" if eps_decay is None else f"_epsd{eps_decay}")
            + ("" if food_curriculum is None else "_fc" + "-".join(str(x) for x in food_curriculum))
            + ("_reloc" if food_relocate else ""))


def evaluate(agent, env, rng, episodes, eps):
    """Play `episodes` episodes on a separate env with epsilon `eps`, as the DQN paper's evaluation runs do.

    act gets t=None (no learning bookkeeping) and draws from `rng` instead of the agent's own stream, and
    observe is never called, so training is bit-for-bit the same with or without evaluation.
    Returns one (score, regular foods, truncated, length) tuple per episode.
    """
    saved, agent.rng = agent.rng, rng
    out = []
    try:
        for _ in range(episodes):
            obs, n, done = env.reset(), 0, False
            while not done:
                obs, _, term, trunc = env.step(agent.act(obs, eps, None))
                n, done = n + 1, term or trunc
            out.append((env.score, env.foods, int(trunc), n))
    finally:
        agent.rng = saved
    return out


def save(agent, path, t, settings):
    """Pickle the agent (without its replay buffer) at step t, plus the run settings snake-watch needs to replay it
    on the board it was trained on (food curriculum and relocation included)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dict(agent=agent, t=t, **settings), f)


def results_dir(D, size, root="results", map="open", bonus=0.0, render="grid", frame_stack=None, wall_scale=1.0, actions="relative"):
    """results/d{D} (7x7) or g{size}_d{D}; a non-default map / bonus food / pixel render / frame stack / wall scale adds
    _{map} / _b{bonus} / _px / _fs{K} / _ws{X} / _abs."""
    base = f"{root}/d{D}" if size == 7 else f"{root}/g{size}_d{D}"
    default_fs = 4 if render == "pixels" else 1
    return (base + ("" if map == "open" else f"_{map}") + (f"_b{bonus:g}" if bonus else "")
            + ("_px" if render == "pixels" else "") + ("" if frame_stack in (None, default_fs) else f"_fs{frame_stack}")
            + ("" if wall_scale == 1.0 else f"_ws{wall_scale:g}") + ("" if actions == "relative" else "_abs"))


def results_dir_pacman(root="results", frame_skip=4, frame_stack=None, reward_units="small", clip_reward=False):
    """results/pac_px (pixels only), plus _sk{K} for a non-default frame skip, _fs{K} for a non-default stack,
    _atari for Atari reward units and _clip when the agent learns from clipped rewards."""
    return (f"{root}/pac_px" + ("" if frame_skip == 4 else f"_sk{frame_skip}")
            + ("" if frame_stack in (None, 4) else f"_fs{frame_stack}")
            + ("" if reward_units == "small" else f"_{reward_units}") + ("_clip" if clip_reward else ""))


def make_env(game, seed, *, size=7, D=0, map="open", bonus=0.0, food_radius=None, relocate_food=False, render="grid",
             frame_stack=None, wall_scale=1.0, actions="relative", frame_skip=4, reward_units="small"):
    """The training/evaluation env of a run: Snake (all the options) or MsPacman (pixels, 14x14, 4 absolute actions)."""
    if game == "pacman":
        return MsPacman(seed=seed, frame_skip=frame_skip, frame_stack=frame_stack, reward_units=reward_units)
    return Snake(size=size, seed=seed, distractors=D, map=map, bonus=bonus, food_radius=food_radius,
                 relocate_food=relocate_food, render=render, frame_stack=frame_stack, wall_scale=wall_scale,
                 actions=actions)


def main(name, seed, steps, *, D=0, size=7, root="results", eps_floor=EPS_FLOOR, map="open", bonus=0.0,
         eps_decay=None, eval_every=0, eval_episodes=5, eval_eps=0.05, food_curriculum=None, food_relocate=False,
         probe_every=0, train_log_every=1000, save_agent=False, save_every=0, render="grid", opt="adam", lr=None, frame_stack=None, wall_scale=1.0, actions="relative", probe_far=False, game="snake", frame_skip=4, reward_units="small", clip_reward=False):
    if game not in GAMES:
        raise ValueError(f"unknown game {game!r}; choose from {GAMES}")
    if game == "pacman":  # fixed 14x14 maze, pixels, absolute actions: none of the Snake board options apply
        if (D or size != 7 or map != "open" or bonus or food_curriculum or food_relocate or probe_every
                or render != "grid" or wall_scale != 1.0 or actions != "relative"):
            raise ValueError("--game pacman takes no --distractors/--size/--map/--bonus/--render/--wall-scale/--actions/"
                             "--food-*/--probe-*; use --frame-skip and --frame-stack")
        size, render = MsPacman().n, "pixels"
    if game != "pacman" and reward_units != "small":
        raise ValueError("--reward-units applies to --game pacman only")
    if food_relocate and food_curriculum is None:
        raise ValueError("food_relocate only acts during a food curriculum; set food_curriculum too")
    rng = np.random.default_rng(seed)
    env = make_env(game, seed + 1000, size=size, D=D, map=map, bonus=bonus,
                   food_radius=food_radius(0, food_curriculum, size), relocate_food=food_relocate, render=render,
                   frame_stack=frame_stack, wall_scale=wall_scale, actions=actions, frame_skip=frame_skip, reward_units=reward_units)
    agent = make_agent(name, env, rng)
    if opt != "adam" or lr is not None:  # the agent's defaults (Adam 5e-4) unless asked; no rng draws
        if not hasattr(agent, "set_optimizer"):
            raise ValueError(f"{name} has no optimizer to set (--opt / --lr)")
        agent.set_optimizer(opt, lr)
    # Evaluation always places food anywhere: it measures the real task, whatever the training curriculum.
    if eval_every:  # own env and RNG streams, so evaluation never shifts training's random numbers
        eval_env = make_env(game, seed + 2000, size=size, D=D, map=map, bonus=bonus, render=render,
                            frame_stack=frame_stack, wall_scale=wall_scale, actions=actions, frame_skip=frame_skip, reward_units=reward_units)
        eval_rng = np.random.default_rng(seed + 3000)
    # Representation probe (probe.py): a fixed hand-built state set, no rng shared with training.
    P = make_probe_set(size, map, bonus, render=render, frame_stack=frame_stack, wall_scale=wall_scale, actions=actions) if probe_every and not D else None
    F = (make_far_probe_set(size, map, bonus, render=render, frame_stack=frame_stack, wall_scale=wall_scale, actions=actions)
         if probe_far and P is not None else None)  # larger set: steering by distance and tie fractions
    probes = [(0, probe(agent, P, F))] if P is not None else []
    out_dir = (results_dir_pacman(root, frame_skip, frame_stack, reward_units, clip_reward) if game == "pacman"
               else results_dir(D, size, root, map, bonus, render, frame_stack, wall_scale, actions))
    stem = f"{out_dir}/{run_name(name, eps_floor, eps_decay, food_curriculum, food_relocate, opt, lr)}_s{seed}"
    settings = dict(name=name, seed=seed, size=size, map=map, bonus=bonus, render=render, frame_stack=frame_stack, wall_scale=wall_scale, actions=actions, distractors=D, eps_floor=eps_floor,
                    eps_decay=eps_decay, food_curriculum=food_curriculum, food_relocate=food_relocate, opt=opt, lr=lr, game=game, frame_skip=frame_skip, reward_units=reward_units, clip_reward=clip_reward)
    # Training telemetry (telemetry.py): window means every train_log_every steps. Also streamed, one JSON line per
    # snapshot, to <stem>.train.jsonl as the run goes (kept afterwards), so `snake-train` can plot a run in progress.
    tlog = hasattr(agent, "train_stats") and train_log_every > 0
    if tlog:
        os.makedirs(out_dir, exist_ok=True)
        live = open(f"{stem}.train.jsonl", "w")
    obs = env.reset()
    log, diag, evals, train, t0 = [], [], [], [], time.time()
    for t in range(1, steps + 1):
        if food_curriculum is not None:
            env.food_radius = food_radius(t, food_curriculum, size)
        a = agent.act(obs, epsilon(t, eps_floor, eps_decay), t)
        nobs, r, term, trunc = env.step(a)
        agent.observe(obs, a, float(np.clip(r, -1.0, 1.0)) if clip_reward else r, nobs, term, trunc, t)  # DQN-style clipping is the agent's: score stays raw
        obs = nobs
        if term or trunc:
            # (end step, score, regular foods eaten, 1 if cut off by the idle limit else 0). Readers that
            # predate the last two fields use only the first two, so old and new JSONs read the same way.
            log.append((t, env.score, env.foods, int(trunc)))
            obs = env.reset()
        if t % 6000 == 0 and hasattr(agent, "diagnostics"):
            diag.append((t, agent.diagnostics()))
        if eval_every and t % eval_every == 0:
            evals.append((t, evaluate(agent, eval_env, eval_rng, eval_episodes, eval_eps)))
        if P is not None and t % probe_every == 0:
            probes.append((t, probe(agent, P, F)))
        if save_every and t % save_every == 0:
            save(agent, f"{stem}.agent_t{t}.pkl", t, settings)
        if tlog and t % train_log_every == 0:
            s = agent.train_stats()
            if s:
                train.append((t, s))
                live.write(json.dumps({"t": t, **s}) + "\n")
                live.flush()
    out = {"agent": name, "seed": seed, "steps": steps, "distractors": D, "size": size,
           "eps_floor": eps_floor, "eps_decay": eps_decay, "food_curriculum": food_curriculum, "food_relocate": food_relocate, "map": map, "bonus": bonus, "render": render, "frame_stack": frame_stack, "wall_scale": wall_scale, "actions": actions, "probe_far": probe_far, "game": game, "frame_skip": frame_skip, "reward_units": reward_units, "clip_reward": clip_reward, "opt": opt, "lr": lr, "episodes": log, "diagnostics": diag,
           "eval": {"every": eval_every, "episodes": eval_episodes, "eps": eval_eps} if eval_every else None,
           "evaluations": evals, "probes": probes, "train": train,
           "wallclock_s": time.time() - t0}
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{stem}.json", "w") as f:
        json.dump(out, f)
    if save_agent:
        save(agent, f"{stem}.agent.pkl", steps, settings)
    if tlog:
        live.close()
    print(run_name(name, eps_floor, eps_decay, food_curriculum, food_relocate, opt, lr), seed, f"{len(log)} eps, {time.time()-t0:.0f}s")


def cli():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("agent", help=f"one of {sorted(AGENTS)}")
    ap.add_argument("seed", type=int)
    ap.add_argument("steps", type=int)
    ap.add_argument("--distractors", type=int, default=0, help="extra noise channels")
    ap.add_argument("--size", type=int, default=7, help="grid size")
    ap.add_argument("--eps-floor", type=float, default=EPS_FLOOR, help="final epsilon after the decay")
    ap.add_argument("--eps-decay", type=int, default=None,
                    help="steps for epsilon to fall linearly from 1 to the floor (0 = fixed at the floor; "
                         "default: the original 5k-step schedule)")
    ap.add_argument("--map", default="open", choices=MAPS, help="obstacle layout")
    ap.add_argument("--bonus", type=float, default=0.0, help="reward of the timed bonus food (0 = off)")
    ap.add_argument("--render", default="grid", choices=RENDERS,
                    help="observation: 'grid' channels (default) or 'pixels', an 84x84 grayscale image with a "
                         "4-frame stack as DQN/NEC see Atari (use the *_naturecnn agents)")
    ap.add_argument("--frame-stack", type=int, default=None,
                    help="stack the last K observations (default: 4 with --render pixels, 1 = off with the grid)")
    ap.add_argument("--wall-scale", type=float, default=1.0,
                    help="multiply the walls' value in the observation (0 hides them; they still kill). Ablation option")
    ap.add_argument("--actions", default="relative", choices=ACTION_MODES,
                    help="'relative' (straight / right / left, default) or 'absolute' (up / right / down / left, as an "
                         "Atari joystick; a reversal keeps going straight)")
    ap.add_argument("--game", default="snake", choices=GAMES,
                    help="'snake' (default) or 'pacman': a 14x14 Ms. Pac-Man-like maze, always pixels (84x84 gray, "
                         "4-frame stack; use the *_naturecnn agents); results in <results>/pac_px")
    ap.add_argument("--frame-skip", type=int, default=4, help="--game pacman: game ticks per agent step (Atari: 4)")
    ap.add_argument("--reward-units", default="small", choices=("small", "atari"),
                    help="--game pacman reward scale: 'small' (points/100, -1 for death; default) or 'atari' (10/50/200 points, nothing for dying)")
    ap.add_argument("--clip-reward", action="store_true",
                    help="the agent learns from rewards clipped to [-1, 1], as the DQN paper's does (scores are still logged raw)")
    ap.add_argument("--opt", default="adam", choices=["adam", "rmsprop"],
                    help="optimizer of the gradient-trained agents (rmsprop: rho 0.95, eps 1e-2, as in the DQN/NEC papers)")
    ap.add_argument("--lr", type=float, default=None, help="optimizer learning rate (default: the agent's, Adam 5e-4)")
    ap.add_argument("--food-curriculum", default=None, metavar="R[:H]:N",
                    help="place food within R walkable steps of the head (held until step H), growing to the "
                         "whole board by step N, then anywhere (default: anywhere from the start)")
    ap.add_argument("--food-relocate", action="store_true",
                    help="during the curriculum, re-place food not eaten within 2r+5 steps near the head")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="every K steps, play evaluation episodes on a separate env (0 = off)")
    ap.add_argument("--eval-episodes", type=int, default=5, help="episodes per evaluation")
    ap.add_argument("--eval-eps", type=float, default=0.05, help="epsilon during evaluation (DQN paper: 0.05)")
    ap.add_argument("--probe-every", type=int, default=0,
                    help="every K steps, measure the encoder and greedy steering on a fixed probe set (probe.py)")
    ap.add_argument("--probe-far", action="store_true",
                    help="with --probe-every, also probe food up to 6 steps away: steering by distance and tie fractions")
    ap.add_argument("--train-log-every", type=int, default=1000,
                    help="every K steps, log training telemetry (loss, TD error, grad norm, ...; 0 = off)")
    ap.add_argument("--save-agent", action="store_true",
                    help="pickle the trained agent to <stem>.agent.pkl, for snake-watch")
    ap.add_argument("--save-every", type=int, default=0, help="also pickle a checkpoint every K steps")
    ap.add_argument("--results", default="results", help="results root directory")
    a = ap.parse_args()
    fc = tuple(int(x) for x in a.food_curriculum.split(":")) if a.food_curriculum else None
    main(a.agent, a.seed, a.steps, D=a.distractors, size=a.size, root=a.results, eps_floor=a.eps_floor, map=a.map,
         bonus=a.bonus, eps_decay=a.eps_decay, eval_every=a.eval_every, eval_episodes=a.eval_episodes,
         eval_eps=a.eval_eps, food_curriculum=fc, food_relocate=a.food_relocate, probe_every=a.probe_every,
         train_log_every=a.train_log_every, save_agent=a.save_agent, save_every=a.save_every, render=a.render,
         opt=a.opt, lr=a.lr, frame_stack=a.frame_stack, wall_scale=a.wall_scale, actions=a.actions, probe_far=a.probe_far,
         game=a.game, frame_skip=a.frame_skip, reward_units=a.reward_units, clip_reward=a.clip_reward)


if __name__ == "__main__":
    cli()
