"""Agent registry: every runnable agent name maps to a factory (env, rng, **overrides) -> Agent.

To add an agent, implement the `Agent` interface (see base.py) in a new module and
register it in AGENTS (and in plot.LABELS for a nicer legend name).
"""
from snake_rl.agents.base import Agent
from snake_rl.agents.dqn import DQNAgent
from snake_rl.agents.mfec import MFECAgent
from snake_rl.agents.nec import NECAgent
from snake_rl.agents.random_policy import RandomAgent

N = 50


def _agent(cls, grid=False, u8=False, **base):
    """Factory (env, rng, **overrides) -> cls(env.obs_dim, env.n_actions, rng, **base | overrides). `grid` adds env.obs_shape
    (the CNN encoders), `u8` keeps replay observations as uint8 (pixel render). Overrides are constructor settings given on
    the command line (`snake-run --agent-kw KEY=VALUE`); with none, the agent is exactly the registered one."""
    def make(env, rng, **kw):
        args = dict(base)
        if grid:
            args["obs_shape"] = env.obs_shape
        if u8:
            args["obs_u8"] = True
        args.update(kw)
        return cls(env.obs_dim, env.n_actions, rng, **args)
    return make


AGENTS = {
    "random": lambda env, rng, **kw: RandomAgent(env.n_actions, rng),
    "dqn": _agent(DQNAgent, N=1),
    "dqn_nstep": _agent(DQNAgent, N=N),
    "mfec": _agent(MFECAgent),
    "ec_frozen": _agent(NECAgent, learn_embedding=False, N=N),
    "nec": _agent(NECAgent, learn_embedding=True, N=N),
    # CNN encoders (the paper's architecture family): same agents, conv trunk instead of MLP
    "dqn_cnn": _agent(DQNAgent, grid=True, N=1, encoder="cnn"),
    "dqn_nstep_cnn": _agent(DQNAgent, grid=True, N=N, encoder="cnn"),
    "ec_frozen_cnn": _agent(NECAgent, grid=True, learn_embedding=False, N=N, encoder="cnn"),
    "nec_cnn": _agent(NECAgent, grid=True, learn_embedding=True, N=N, encoder="cnn"),
    # DQN-shaped CNN (32 3x3/1, 64 3x3/2, 64 3x3/2, FC 512): the paper's encoder family scaled to the grid
    "dqn_dqncnn": _agent(DQNAgent, grid=True, N=1, encoder="dqncnn"),
    "nec_dqncnn": _agent(NECAgent, grid=True, learn_embedding=True, N=N, encoder="dqncnn"),
    # DQN's exact network (32 8x8/4, 64 4x4/2, 64 3x3/1, FC 512) for the 84x84 pixel render (--render pixels)
    "dqn_naturecnn": _agent(DQNAgent, grid=True, u8=True, N=1, encoder="naturecnn"),
    "dqn_nstep_naturecnn": _agent(DQNAgent, grid=True, u8=True, N=N, encoder="naturecnn"),
    "ec_frozen_naturecnn": _agent(NECAgent, grid=True, u8=True, learn_embedding=False, N=N, encoder="naturecnn"),
    "nec_naturecnn": _agent(NECAgent, grid=True, u8=True, learn_embedding=True, N=N, encoder="naturecnn"),
}


def make_agent(name, env, rng, **kw) -> Agent:
    """The registered agent `name`; `kw` overrides constructor settings (e.g. target_every, N, dnd_cap)."""
    if name not in AGENTS:
        raise ValueError(f"unknown agent {name!r}; choose from {sorted(AGENTS)}")
    return AGENTS[name](env, rng, **kw)


__all__ = ["Agent", "AGENTS", "make_agent", "DQNAgent", "MFECAgent", "NECAgent", "RandomAgent"]
