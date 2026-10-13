"""A small Ms. Pac-Man-like game with the Atari view, for comparing the agents on a denser-reward game than Snake.

Board: one fixed 14 x 14 maze (MAZE below). Pellets on every open cell, four power pellets in the corners, two ghosts.
Actions (absolute, as an Atari joystick): 0 up, 1 right, 2 down, 3 left. Moving into a wall means staying put.

Time is counted in game ticks. Pac-Man moves on every `PAC_EVERY`-th tick and the ghosts on every `GHOST_EVERY`-th
(slower, so they can be outrun). Each `step(a)` repeats the action for `frame_skip` ticks (default 4, as in the Atari
preprocessing of DQN and NEC), summing the reward and stopping early if the episode ends. Ghosts chase Pac-Man along the
shortest path with probability `CHASE_P`, else wander; they do not reverse unless forced. After a power pellet, ghosts are
scared for `SCARE_TICKS` ticks: they wander and can be eaten (they then return to the house). All randomness is `self.rng`.

Reward (`reward_units="small"`, default): pellet +0.1, power pellet +0.5, ghost +2.0, caught by a ghost -1 (terminal), board cleared +1
(terminal). `reward_units="atari"`: 10 / 50 / 200 points and nothing for dying or clearing (the Atari game's own scale).
`score` is in Ms. Pac-Man points (10 / 50 / 200) and `foods` counts pellets (power ones included).
One life. The episode is truncated after `max_idle` ticks without eating a pellet (the value target bootstraps through it).

Observation (pixels only): the board as an 84 x 84 grayscale image (6 x 6 pixels per cell), the last `frame_stack` (default 4)
agent-step frames stacked, flat float32 k/255, obs_shape = (4, 84, 84). Gray levels (of 255): empty 0, wall 60, pellet 110
(2 x 2 dot), power pellet 160 (4 x 4 dot), ghost 200, scared ghost 230, Pac-Man 255.
"""
from collections import deque
import numpy as np

from snake_rl.frames import PIXELS, FrameStack

# '#' wall, '.' pellet, 'o' power pellet, 'P' Pac-Man start, 'G' ghost start (the house, open above and below).
MAZE = [
    "##############",
    "#o....##....o#",
    "#.##.#..#.##.#",
    "#.#........#.#",
    "#...##..##...#",
    "#.##......##.#",
    "#....#GG#....#",
    "#.##.#..#.##.#",
    "#.#........#.#",
    "#...##..##...#",
    "#.##......##.#",
    "#....#P.#....#",
    "#o..........o#",
    "##############",
]
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
GRAY = dict(wall=60, pellet=110, power=160, ghost=200, scared=230, pac=255)
PAC_EVERY, GHOST_EVERY, SCARE_TICKS, CHASE_P = 2, 3, 40, 0.7
# Reward per event, by unit set. "small" (default): points / 100, with -1 for death and +1 for clearing the board.
# "atari": the game's own points (10 / 50 / 200), nothing for dying or clearing, as in the Atari game, so returns run into the hundreds.
REWARDS = dict(small=dict(pellet=0.1, power=0.5, ghost=2.0, death=-1.0, clear=1.0),
               atari=dict(pellet=10.0, power=50.0, ghost=200.0, death=0.0, clear=0.0))
POINTS = dict(pellet=10, power=50, ghost=200)


class MsPacman:
    DIRS = DIRS

    def __init__(self, seed=0, frame_skip=4, frame_stack=None, max_idle=None, reward_units="small"):
        if reward_units not in REWARDS:
            raise ValueError(f"unknown reward_units {reward_units!r}; choose from {sorted(REWARDS)}")
        self.reward_units, self.R = reward_units, REWARDS[reward_units]
        self.n = len(MAZE)
        if PIXELS % self.n:
            raise ValueError("the maze side must divide 84")
        self.cell = PIXELS // self.n
        self.frame_skip, self.frame_stack = frame_skip, frame_stack or 4
        self.rng = np.random.default_rng(seed)
        self.max_idle = max_idle or 2 * self.n * self.n
        self.walls = np.array([[ch == "#" for ch in row] for row in MAZE])
        self.start_pellets = np.array([[ch == "." for ch in row] for row in MAZE])
        self.start_power = np.array([[ch == "o" for ch in row] for row in MAZE])
        find = lambda c: [(r, q) for r, row in enumerate(MAZE) for q, ch in enumerate(row) if ch == c]
        self.pac_start, self.ghost_starts = find("P")[0], find("G")
        self.n_ghosts = len(self.ghost_starts)
        self.n_actions = 4
        self.pixels = True
        self.obs_shape = (self.frame_stack, PIXELS, PIXELS)
        self.obs_dim = int(np.prod(self.obs_shape))
        self.nbrs = {(r, c): [(r + dr, c + dc) for dr, dc in DIRS if not self.walls[r + dr, c + dc]]
                     for r in range(self.n) for c in range(self.n) if not self.walls[r, c]}
        self.dist = self._all_distances()
        # static drawing helpers: walls, and the dot shapes inside a cell
        self._wall_img = np.kron(self.walls, np.ones((self.cell, self.cell), np.uint8)) * np.uint8(GRAY["wall"])
        self._dot = {}
        for k, side in (("pellet", 2), ("power", 4)):
            m = np.zeros((self.cell, self.cell), np.uint8)
            o = (self.cell - side) // 2
            m[o:o + side, o:o + side] = 1
            self._dot[k] = m
        self.stack = FrameStack(self.frame_stack)

    def _all_distances(self):
        """Shortest walking distance between every pair of open cells."""
        dist = {}
        for s in self.nbrs:
            d, todo = {s: 0}, deque([s])
            while todo:
                p = todo.popleft()
                for q in self.nbrs[p]:
                    if q not in d:
                        d[q] = d[p] + 1
                        todo.append(q)
            dist[s] = d
        return dist

    def reset(self):
        self.pac = self.pac_start
        self.ghosts = list(self.ghost_starts)
        self.prev = list(self.ghost_starts)  # the cell each ghost just left (no reversing)
        self.pellets, self.power = self.start_pellets.copy(), self.start_power.copy()
        self.left = int(self.pellets.sum() + self.power.sum())
        self.scared = 0
        self.tick, self.idle, self.score, self.foods = 0, 0, 0, 0
        self.stack.reset()
        return self._obs()

    def _ghost_move(self, i):
        here = self.ghosts[i]
        opts = self.nbrs[here]
        fwd = [q for q in opts if q != self.prev[i]] or opts
        if self.scared == 0 and self.rng.random() < CHASE_P:
            best = min(self.dist[q][self.pac] for q in fwd)
            fwd = [q for q in fwd if self.dist[q][self.pac] == best]
        nxt = fwd[self.rng.integers(len(fwd))]
        self.prev[i], self.ghosts[i] = here, nxt

    def _hit(self):
        """Resolve Pac-Man sharing a cell with ghosts. -> (reward, caught)."""
        r = 0.0
        for i, g in enumerate(self.ghosts):
            if g != self.pac:
                continue
            if self.scared > 0:
                r += self.R["ghost"]
                self.score += POINTS["ghost"]
                self.ghosts[i] = self.prev[i] = self.ghost_starts[i % self.n_ghosts]
            else:
                return self.R["death"], True
        return r, False

    def _tick(self):
        """One game tick. -> (reward, term)."""
        self.tick += 1
        self.idle += 1
        r = 0.0
        if self.scared > 0:
            self.scared -= 1
        if self.tick % PAC_EVERY == 0:
            dr, dc = DIRS[self._a]
            q = (self.pac[0] + dr, self.pac[1] + dc)
            if not self.walls[q]:
                self.pac = q
                if self.pellets[q]:
                    self.pellets[q] = False
                    r += self.R["pellet"]
                    self.score += POINTS["pellet"]
                    self.foods, self.left, self.idle = self.foods + 1, self.left - 1, 0
                elif self.power[q]:
                    self.power[q] = False
                    r += self.R["power"]
                    self.score += POINTS["power"]
                    self.foods, self.left, self.idle = self.foods + 1, self.left - 1, 0
                    self.scared = SCARE_TICKS
                h, caught = self._hit()
                if caught:
                    return h, True
                r += h
                if self.left == 0:
                    return r + self.R["clear"], True
        if self.tick % GHOST_EVERY == 0:
            for i in range(self.n_ghosts):
                self._ghost_move(i)
            h, caught = self._hit()
            if caught:
                return h, True
            r += h
        return r, False

    def step(self, a):
        self._a = a
        total, term = 0.0, False
        for _ in range(self.frame_skip):
            r, term = self._tick()
            total += r
            if term:
                break
        return self._obs(), total, term, (not term) and self.idle >= self.max_idle

    def _frame(self):
        """The current board as an 84 x 84 uint8 image."""
        up = lambda g: np.repeat(np.repeat(g, self.cell, 0), self.cell, 1)
        img = self._wall_img.copy()
        for key, grid in (("pellet", self.pellets), ("power", self.power)):
            img[np.kron(grid, self._dot[key]) > 0] = GRAY[key]
        chars = np.zeros((self.n, self.n), np.uint8)
        for g in self.ghosts:
            chars[g] = GRAY["scared"] if self.scared > 0 else GRAY["ghost"]
        chars[self.pac] = GRAY["pac"]
        m = up(chars) > 0
        img[m] = up(chars)[m]
        return img

    def _obs(self):
        return self.stack.push(self._frame())

    def static_obs(self):
        """Observation of the current state as if it had always been so (the frame stack is not touched)."""
        return self.stack.static(self._frame())

    def render(self):
        """ASCII picture: @ Pac-Man, G ghost, g scared ghost, . pellet, o power pellet, # wall."""
        g = [["#" if self.walls[r, c] else " " for c in range(self.n)] for r in range(self.n)]
        for r, c in np.argwhere(self.pellets):
            g[r][c] = "."
        for r, c in np.argwhere(self.power):
            g[r][c] = "o"
        for r, c in self.ghosts:
            g[r][c] = "g" if self.scared > 0 else "G"
        g[self.pac[0]][self.pac[1]] = "@"
        return "\n".join("".join(row) for row in g)
