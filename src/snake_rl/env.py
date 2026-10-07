"""Snake environment with grid observations, optional obstacle maps and Nokia-style bonus food.

Observation: binary/float channels over an n x n grid, flattened (see `obs_shape`):
  head, body, food, [obstacle if map != "open"], [bonus if bonus > 0], D distractor channels.
The bonus channel holds the remaining lifetime fraction (1 -> 0) at the bonus food's cell, so the
agent can see how long the detour is still worth it.
Actions (relative): 0 = straight, 1 = turn right, 2 = turn left.
Reward: +1 food, +`bonus` bonus food, -1 death, 0 otherwise. `score` accumulates the food rewards.

Maps (obstacles are lethal, food never spawns on them; the free area is always connected):
  open     empty board
  pillars  four single-cell pillars
  walls    two horizontal bars with gaps on opposite sides: an S-shaped corridor
  rooms    a cross-shaped wall dividing the board into four rooms, joined at the centre

Bonus food (Nokia style): after every `bonus_every` regular foods a bonus pellet appears on a
random free cell for `bonus_life` steps (default 2n), then vanishes. It is worth `bonus` points
and grows the snake like a regular food, so greedy chasing makes the snake long and risky.

Distractors: `distractors` extra channels of i.i.d. Bernoulli(noise_p) pixels, resampled
every step. They carry no information about value, but dominate raw-space distances,
so nearest neighbours in observation space become largely random.
Food distance (a curriculum aid, off by default): with `food_radius` = r, regular food is drawn only from
free cells within r walkable steps of the head (shortest path around walls and the body), falling back to
any free cell if none is that close. run.py can grow r during training (`--food-curriculum`).
Truncation: episode is cut if the snake goes `max_idle` steps without eating
(prevents endless loops; the value target bootstraps through truncation).

With the defaults (map="open", bonus=0) the env is bit-for-bit the original one, RNG draws included.
"""
from collections import deque
import numpy as np

MAPS = ("open", "pillars", "walls", "rooms")


def make_walls(name, n):
    """Boolean n x n obstacle mask for a named map. The centre row/cross stays clear for the start."""
    w = np.zeros((n, n), bool)
    c = n // 2
    if name == "open":
        pass
    elif name == "pillars":
        k = max(1, n // 4)
        for r in (k, n - 1 - k):
            for q in (k, n - 1 - k):
                w[r, q] = True
    elif name == "walls":
        k = max(1, n // 4)
        w[k, : n - 2] = True      # gap at the right end
        w[n - 1 - k, 2:] = True   # gap at the left end
    elif name == "rooms":
        w[c, : c - 1] = w[c, c + 2:] = True
        w[: c - 1, c] = w[c + 2:, c] = True
    else:
        raise ValueError(f"unknown map {name!r}; choose from {MAPS}")
    return w


def _connected(free):
    cells = np.argwhere(free)
    seen, todo = {tuple(cells[0])}, [tuple(cells[0])]
    while todo:
        r, c = todo.pop()
        for dr, dc in Snake.DIRS:
            q = (r + dr, c + dc)
            if 0 <= q[0] < free.shape[0] and 0 <= q[1] < free.shape[1] and free[q] and q not in seen:
                seen.add(q)
                todo.append(q)
    return len(seen) == len(cells)


class Snake:
    DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left

    def __init__(self, size=7, max_idle=None, seed=0, distractors=0, noise_p=0.5,
                 map="open", bonus=0.0, bonus_every=4, bonus_life=None, food_radius=None):
        self.n = size
        self.D, self.noise_p = distractors, noise_p
        self.rng = np.random.default_rng(seed)
        self.max_idle = max_idle or 2 * size * size
        self.map, self.bonus_r, self.bonus_every = map, float(bonus), bonus_every
        self.bonus_life = bonus_life or 2 * size
        self.food_radius = food_radius  # None = uniform placement; may be changed between steps
        self.walls = make_walls(map, size)
        self.wall_set = {(int(r), int(c)) for r, c in np.argwhere(self.walls)}
        c = size // 2
        start = [(c, c), (c, c - 1), (c, c + 1)]  # two body cells + the one straight ahead
        if any(self.walls[p] for p in start) or not _connected(~self.walls):
            raise ValueError(f"map {map!r} does not fit a {size}x{size} board")
        self.n_actions = 3
        self.channels = 3 + bool(self.wall_set) + (self.bonus_r > 0) + distractors
        self.obs_shape = (self.channels, size, size)
        self.obs_dim = self.channels * size * size

    def reset(self):
        c = self.n // 2
        self.body = deque([(c, c), (c, c - 1)])  # body[0] is the head
        self.dir = 1
        self.idle = 0
        self.score = 0
        self.foods = 0
        self.bonus_pos, self.bonus_left = None, 0
        self._place_food()
        return self._obs()

    def _free(self):
        occ = set(self.body) | self.wall_set
        if self.food is not None:
            occ.add(self.food)
        if self.bonus_pos is not None:
            occ.add(self.bonus_pos)
        return [(r, c) for r in range(self.n) for c in range(self.n) if (r, c) not in occ]

    def _place_food(self):
        self.food = None  # not yet placed: keep it out of the occupancy
        free = self._free()
        if self.food_radius is not None and free:
            dist = self._steps_from_head()
            near = [p for p in free if dist.get(p, self.food_radius + 1) <= self.food_radius]
            free = near or free  # nothing that close: fall back to anywhere
        self.food = free[self.rng.integers(len(free))] if free else None

    def _steps_from_head(self):
        """Shortest walkable path length from the head to every reachable cell (walls and body block)."""
        blocked = self.wall_set | set(list(self.body)[1:])
        head = self.body[0]
        dist, todo = {head: 0}, deque([head])
        while todo:
            r, c = todo.popleft()
            for dr, dc in self.DIRS:
                q = (r + dr, c + dc)
                if 0 <= q[0] < self.n and 0 <= q[1] < self.n and q not in blocked and q not in dist:
                    dist[q] = dist[(r, c)] + 1
                    todo.append(q)
        return dist

    def _place_bonus(self):
        free = self._free()
        if free:
            self.bonus_pos = free[self.rng.integers(len(free))]
            self.bonus_left = self.bonus_life

    def _obs(self):
        o = np.zeros((self.channels - self.D, self.n, self.n), np.float32)
        hr, hc = self.body[0]
        o[0, hr, hc] = 1.0
        for (r, c) in list(self.body)[1:]:
            o[1, r, c] = 1.0
        if self.food is not None:
            o[2, self.food[0], self.food[1]] = 1.0
        k = 3
        if self.wall_set:
            o[k] = self.walls
            k += 1
        if self.bonus_r > 0:
            if self.bonus_pos is not None:
                o[k, self.bonus_pos[0], self.bonus_pos[1]] = self.bonus_left / self.bonus_life
        if self.D:
            noise = (self.rng.random((self.D, self.n, self.n)) < self.noise_p).astype(np.float32)
            o = np.concatenate([o, noise])
        return o.ravel()

    def render(self):
        """ASCII picture: @ head, o body, * food, $ bonus, # wall."""
        g = [["." for _ in range(self.n)] for _ in range(self.n)]
        for r, c in self.wall_set:
            g[r][c] = "#"
        for r, c in list(self.body)[1:]:
            g[r][c] = "o"
        if self.food:
            g[self.food[0]][self.food[1]] = "*"
        if self.bonus_pos:
            g[self.bonus_pos[0]][self.bonus_pos[1]] = "$"
        g[self.body[0][0]][self.body[0][1]] = "@"
        return "\n".join("".join(row) for row in g)

    def step(self, a):
        self.dir = (self.dir + (0, 1, -1)[a]) % 4
        dr, dc = self.DIRS[self.dir]
        hr, hc = self.body[0]
        nh = (hr + dr, hc + dc)
        self.idle += 1
        if self.bonus_pos is not None:
            self.bonus_left -= 1
            if self.bonus_left <= 0:
                self.bonus_pos = None
        out = not (0 <= nh[0] < self.n and 0 <= nh[1] < self.n)
        if out or nh in self.wall_set or nh in list(self.body)[:-1]:  # tail cell is vacated this step
            return self._obs(), -1.0, True, False
        self.body.appendleft(nh)
        if nh == self.bonus_pos:  # grows like a regular food, but pays more
            self.idle, self.score = 0, self.score + self.bonus_r
            self.bonus_pos = None
            return self._obs(), self.bonus_r, False, False
        if nh == self.food:
            self.idle, self.score = 0, self.score + 1
            self.foods += 1
            self._place_food()
            if self.food is None:  # board filled
                return self._obs(), 1.0, True, False
            if self.bonus_r > 0 and self.foods % self.bonus_every == 0:
                self._place_bonus()
            return self._obs(), 1.0, False, False
        self.body.pop()
        return self._obs(), 0.0, False, self.idle >= self.max_idle
