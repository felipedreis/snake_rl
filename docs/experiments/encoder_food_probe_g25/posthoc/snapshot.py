"""Re-run run.main's training loop exactly up to step T and pickle the agent and env (diagnostic only)."""
import json, pickle, sys
import numpy as np
from snake_rl.agents import make_agent
from snake_rl.env import Snake
from snake_rl.run import epsilon, food_radius, run_name

name, seed, T, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
fc, size = (2, 250000, 600000), 25
rng = np.random.default_rng(seed)
env = Snake(size=size, seed=seed + 1000, map="rooms", bonus=5, food_radius=food_radius(0, fc, size), relocate_food=True)
agent = make_agent(name, env, rng)
obs, log = env.reset(), []
for t in range(1, T + 1):
    env.food_radius = food_radius(t, fc, size)
    a = agent.act(obs, epsilon(t, 0.05, 50000), t)
    nobs, r, term, trunc = env.step(a)
    agent.observe(obs, a, r, nobs, term, trunc, t)
    obs = nobs
    if term or trunc:
        log.append([t, env.score, env.foods, int(trunc)]); obs = env.reset()
ref = json.load(open(f"results/exp_encoder_probe/g25_d0_rooms_b5/{run_name(name, 0.05, 50000, fc, True)}_s{seed}.json"))
ref_eps = [e for e in ref["episodes"] if e[0] <= T]
print(name, seed, "reproduces:", ref_eps == log, len(log))
pickle.dump(dict(agent=agent, env=env, obs=obs, t=T), open(out, "wb"))
