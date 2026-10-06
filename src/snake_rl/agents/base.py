"""The interface every agent implements. run.py only ever talks to agents through it."""
from typing import Protocol

import numpy as np


class Agent(Protocol):
    def act(self, obs: np.ndarray, eps: float, t: int | None) -> int:
        """Pick an action for obs (epsilon-greedy with the given eps, drawing only from self.rng).

        t is the global env step. t=None marks an evaluation step: the agent must then leave its learning
        state untouched (no memory-usage bookkeeping, no counters), so evaluation cannot change training.
        """

    def observe(self, obs: np.ndarray, a: int, r: float, next_obs: np.ndarray,
                term: bool, trunc: bool, t: int) -> None:
        """Called once after every env step with the transition just taken."""

    # Optional: `diagnostics() -> dict` (JSON-serialisable). If present, run.py logs it every 6000 steps.
