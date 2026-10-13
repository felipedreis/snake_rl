"""Agent registry: every runnable agent name maps to a factory (env, rng, **overrides) -> Agent.

The network follows the observation, not the name: on a pixel observation (Snake's --render pixels, Ms. Pac-Man) the
gradient-trained agents use DQN's Nature CNN and keep replay observations as uint8; on the grid they use an MLP.

To add an agent, implement the `Agent` interface (see base.py) in a new module and
register it in AGENTS (and in plot.LABELS for a nicer legend name).
"""
import sys

from snake_rl.agents.base import Agent
from snake_rl.agents.dqn import DQNAgent
from snake_rl.agents.mfec import MFECAgent
from snake_rl.agents.nec import NECAgent
from snake_rl.agents.random_policy import RandomAgent

N = 50


def _agent(cls, net=True, **base):
    """Factory (env, rng, **overrides) -> cls(env.obs_dim, env.n_actions, rng, **base | overrides). `net`: the agent has a
    network, so on a pixel env it gets the Nature CNN (encoder, obs_shape) and uint8 replay. Overrides are constructor
    settings given on the command line (`--agent-kw KEY=VALUE`); with none, the agent is exactly the registered one."""
    def make(env, rng, **kw):
        args = dict(base)
        if net and env.pixels:
            args.update(encoder="naturecnn", obs_shape=env.obs_shape, obs_u8=True)
        args.update(kw)
        return cls(env.obs_dim, env.n_actions, rng, **args)
    return make


AGENTS = {
    "random": lambda env, rng, **kw: RandomAgent(env.n_actions, rng),
    "dqn": _agent(DQNAgent, N=1),
    "dqn_nstep": _agent(DQNAgent, N=N),
    "mfec": _agent(MFECAgent, net=False),
    "ec_frozen": _agent(NECAgent, learn_embedding=False, N=N),
    "nec": _agent(NECAgent, learn_embedding=True, N=N),
}

# Names used before the network followed the observation (old result files, checkpoints and run scripts).
LEGACY_NAMES = {"dqn_naturecnn": "dqn", "dqn_nstep_naturecnn": "dqn_nstep", "ec_frozen_naturecnn": "ec_frozen",
                "nec_naturecnn": "nec"}


def canonical(name):
    """The current name of an agent, given a current or a legacy one."""
    return LEGACY_NAMES.get(name, name)


def canonical_stem(stem):
    """A result-file stem (agent name plus run tags) under the current agent name."""
    for old, new in LEGACY_NAMES.items():
        if stem == old or stem.startswith(old + "_"):
            return new + stem[len(old):]
    return stem


def make_agent(name, env, rng, **kw) -> Agent:
    """The registered agent `name`; `kw` overrides constructor settings (e.g. target_every, N, dnd_cap)."""
    if name in LEGACY_NAMES:
        print(f"note: agent {name!r} is now {canonical(name)!r} (the network follows the observation)", file=sys.stderr)
        name = canonical(name)
    if name not in AGENTS:
        raise ValueError(f"unknown agent {name!r}; choose from {sorted(AGENTS)}")
    return AGENTS[name](env, rng, **kw)


__all__ = ["Agent", "AGENTS", "LEGACY_NAMES", "canonical", "canonical_stem", "make_agent", "DQNAgent", "MFECAgent", "NECAgent", "RandomAgent"]
