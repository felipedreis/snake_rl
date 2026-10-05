"""Agent registry: every runnable agent name maps to a factory (env, rng) -> Agent.

To add an agent, implement the `Agent` interface (see base.py) in a new module and
register it in AGENTS (and in plot.LABELS for a nicer legend name).
"""
from snake_rl.agents.base import Agent
from snake_rl.agents.dqn import DQNAgent
from snake_rl.agents.nec import NECAgent

N = 50

AGENTS = {
    "dqn": lambda env, rng: DQNAgent(env.obs_dim, env.n_actions, rng, N=1),
    "dqn_nstep": lambda env, rng: DQNAgent(env.obs_dim, env.n_actions, rng, N=N),
    "ec_frozen": lambda env, rng: NECAgent(env.obs_dim, env.n_actions, rng, learn_embedding=False, N=N),
    "nec": lambda env, rng: NECAgent(env.obs_dim, env.n_actions, rng, learn_embedding=True, N=N),
    "nec_refresh": lambda env, rng: NECAgent(env.obs_dim, env.n_actions, rng, learn_embedding=True, N=N,
                                             refresh_every=1000),
    "nec_bonus": lambda env, rng: NECAgent(env.obs_dim, env.n_actions, rng, learn_embedding=True, N=N,
                                           bonus_beta=0.3),
}


def make_agent(name, env, rng) -> Agent:
    if name not in AGENTS:
        raise ValueError(f"unknown agent {name!r}; choose from {sorted(AGENTS)}")
    return AGENTS[name](env, rng)


__all__ = ["Agent", "AGENTS", "make_agent", "DQNAgent", "NECAgent"]
