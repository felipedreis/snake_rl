"""Agent registry: every runnable agent name maps to a factory (env, rng) -> Agent.

To add an agent, implement the `Agent` interface (see base.py) in a new module and
register it in AGENTS (and in plot.LABELS for a nicer legend name).
"""
from snake_rl.agents.base import Agent
from snake_rl.agents.dqn import DQNAgent
from snake_rl.agents.mfec import MFECAgent
from snake_rl.agents.nec import NECAgent
from snake_rl.agents.random_policy import RandomAgent

N = 50

AGENTS = {
    "random": lambda env, rng: RandomAgent(env.n_actions, rng),
    "dqn": lambda env, rng: DQNAgent(env.obs_dim, env.n_actions, rng, N=1),
    "dqn_nstep": lambda env, rng: DQNAgent(env.obs_dim, env.n_actions, rng, N=N),
    "mfec": lambda env, rng: MFECAgent(env.obs_dim, env.n_actions, rng),
    "ec_frozen": lambda env, rng: NECAgent(env.obs_dim, env.n_actions, rng, learn_embedding=False, N=N),
    "nec": lambda env, rng: NECAgent(env.obs_dim, env.n_actions, rng, learn_embedding=True, N=N),
    "nec_refresh": lambda env, rng: NECAgent(env.obs_dim, env.n_actions, rng, learn_embedding=True, N=N,
                                             refresh_every=1000),
    "nec_bonus": lambda env, rng: NECAgent(env.obs_dim, env.n_actions, rng, learn_embedding=True, N=N,
                                           bonus_beta=0.3),
    # CNN encoders (the paper's architecture family): same agents, conv trunk instead of MLP
    "dqn_cnn": lambda env, rng: DQNAgent(env.obs_dim, env.n_actions, rng, N=1, encoder="cnn", obs_shape=env.obs_shape),
    "dqn_nstep_cnn": lambda env, rng: DQNAgent(env.obs_dim, env.n_actions, rng, N=N, encoder="cnn",
                                               obs_shape=env.obs_shape),
    "ec_frozen_cnn": lambda env, rng: NECAgent(env.obs_dim, env.n_actions, rng, learn_embedding=False, N=N,
                                               encoder="cnn", obs_shape=env.obs_shape),
    "nec_cnn": lambda env, rng: NECAgent(env.obs_dim, env.n_actions, rng, learn_embedding=True, N=N,
                                         encoder="cnn", obs_shape=env.obs_shape),
    # DQN-shaped CNN (32 3x3/1, 64 3x3/2, 64 3x3/2, FC 512): the paper's encoder family scaled to the grid
    "dqn_dqncnn": lambda env, rng: DQNAgent(env.obs_dim, env.n_actions, rng, N=1, encoder="dqncnn",
                                            obs_shape=env.obs_shape),
    "nec_dqncnn": lambda env, rng: NECAgent(env.obs_dim, env.n_actions, rng, learn_embedding=True, N=N,
                                            encoder="dqncnn", obs_shape=env.obs_shape),
}


def make_agent(name, env, rng) -> Agent:
    if name not in AGENTS:
        raise ValueError(f"unknown agent {name!r}; choose from {sorted(AGENTS)}")
    return AGENTS[name](env, rng)


__all__ = ["Agent", "AGENTS", "make_agent", "DQNAgent", "MFECAgent", "NECAgent", "RandomAgent"]
