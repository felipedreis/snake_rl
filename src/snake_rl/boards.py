"""The game side of a run: which env, with which settings, and where its results go.

A board is everything about the game a run is played on. `run.main` only talks to boards through this interface:
  make_env(seed, food_radius=None, relocate_food=False)  a fresh env (training passes the curriculum's start)
  on_step(env, t)              per-step changes to the env during training (Snake: the food curriculum)
  results_dir(root, clip_reward)  the run's output directory
  tags()                       file-name tags of the board's training aids (run.run_name appends them)
  settings()                   the fields written into the result JSON and checkpoints
`board_from_settings` rebuilds the board of a saved run, for the readers (snake-watch, geometry).
"""
from dataclasses import asdict, dataclass

from snake_rl.env import Snake, check_absolute
from snake_rl.pacman import MsPacman


def food_radius(t, curriculum, size):
    """Food-distance curriculum. (r0, n): the radius grows linearly from r0 to 2*size over n steps.
    (r0, hold, n): it stays at r0 until step hold, then grows to 2*size by step n. From step n on, food goes
    anywhere (None), as in the real game. curriculum=None: always anywhere."""
    if curriculum is None or t >= curriculum[-1]:
        return None
    r0, hold, n = curriculum if len(curriculum) == 3 else (curriculum[0], 0, curriculum[1])
    return r0 if t < hold else int(r0 + (2 * size - r0) * (t - hold) / (n - hold))


@dataclass(frozen=True)
class SnakeBoard:
    size: int = 7
    map: str = "open"
    bonus: float = 0.0
    render: str = "grid"         # "grid" channels or "pixels" (84x84 gray frames)
    frame_stack: int | None = None  # pixels only; None = 4
    food_curriculum: tuple | None = None  # (r0, n) or (r0, hold, n), see food_radius
    food_relocate: bool = False

    game = "snake"

    def __post_init__(self):
        if self.food_relocate and self.food_curriculum is None:
            raise ValueError("food_relocate only acts during a food curriculum; set food_curriculum too")

    def make_env(self, seed, food_radius=None, relocate_food=False):
        return Snake(size=self.size, seed=seed, map=self.map, bonus=self.bonus, food_radius=food_radius,
                     relocate_food=relocate_food, render=self.render, frame_stack=self.frame_stack)

    def make_train_env(self, seed):
        """The training env: the food curriculum's starting radius and relocation, if any."""
        return self.make_env(seed, food_radius(0, self.food_curriculum, self.size), self.food_relocate)

    def on_step(self, env, t):
        if self.food_curriculum is not None:
            env.food_radius = food_radius(t, self.food_curriculum, self.size)

    def results_dir(self, root, clip_reward=False):
        """root/d0 (7x7) or g{size}_d0 (the d0 is what is left of the removed distractor channels; kept so new runs
        land next to the old ones); a non-default map / bonus food / pixel render / frame stack adds
        _{map} / _b{bonus} / _px / _fs{K}, and every name ends in _abs (absolute actions; the relative ones are gone)."""
        base = f"{root}/d0" if self.size == 7 else f"{root}/g{self.size}_d0"
        return (base + ("" if self.map == "open" else f"_{self.map}") + (f"_b{self.bonus:g}" if self.bonus else "")
                + ("_px" if self.render == "pixels" else "")
                + ("" if self.frame_stack in (None, 4) else f"_fs{self.frame_stack}") + "_abs")

    def tags(self):
        fc = self.food_curriculum
        return ("" if fc is None else "_fc" + "-".join(str(x) for x in fc)) + ("_reloc" if self.food_relocate else "")

    def settings(self):
        return dict(game=self.game, actions="absolute", **asdict(self))


@dataclass(frozen=True)
class PacmanBoard:
    frame_skip: int = 4             # game ticks per agent step (Atari: 4)
    frame_stack: int | None = None  # None = 4
    reward_units: str = "small"     # "small" (points/100, -1 for death) or "atari" (10/50/200, nothing for dying)

    game = "pacman"

    def make_env(self, seed, food_radius=None, relocate_food=False):
        return MsPacman(seed=seed, frame_skip=self.frame_skip, frame_stack=self.frame_stack,
                        reward_units=self.reward_units)

    def make_train_env(self, seed):
        return self.make_env(seed)

    def on_step(self, env, t):
        pass

    def results_dir(self, root, clip_reward=False):
        """root/pac_px (pixels only), plus _sk{K} for a non-default frame skip, _fs{K} for a non-default stack,
        _atari for Atari reward units and _clip when the agent learns from clipped rewards."""
        return (f"{root}/pac_px" + ("" if self.frame_skip == 4 else f"_sk{self.frame_skip}")
                + ("" if self.frame_stack in (None, 4) else f"_fs{self.frame_stack}")
                + ("" if self.reward_units == "small" else f"_{self.reward_units}") + ("_clip" if clip_reward else ""))

    def tags(self):
        return ""

    def settings(self):
        # size/map/bonus/render/food_* as Snake readers expect them (plot.py reads size).
        return dict(game=self.game, actions="absolute", size=MsPacman.SIZE, map="open", bonus=0.0, render="pixels",
                    food_curriculum=None, food_relocate=False, **asdict(self))


def board_from_settings(s, source="settings"):
    """The board of a saved run (checkpoint or result JSON), falling back on fields older files lack."""
    if s.get("game") == "pacman":
        return PacmanBoard(frame_skip=s.get("frame_skip", 4), frame_stack=s.get("frame_stack"),
                           reward_units=s.get("reward_units", "small"))
    check_absolute(s, source)
    fc = s.get("food_curriculum")
    return SnakeBoard(size=s["size"], map=s.get("map", "open"), bonus=s.get("bonus", 0.0), render=s.get("render", "grid"),
                      frame_stack=s.get("frame_stack"), food_curriculum=tuple(fc) if fc else None,
                      food_relocate=s.get("food_relocate", False))
