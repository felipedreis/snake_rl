"""Writes the job lines of the final-ladder specs (ansible/experiments/final_ladder_*.yml and final_check_*.yml).
usage: python make_specs.py full|check snake|pacman agent[,agent...] [--curriculum]   (one YAML job line per run on stdout)
"""
import sys
SNAKE = "--size 10 --render pixels"
PAC = "--reward-units atari"
ARMS = {  # agent: its settings (the last exploratory experiment's, pacman_c2c3 B4 and D1; NEC/ec_frozen at the paper's memory)
    "random": "",
    "dqn": "--opt rmsprop --lr 2.5e-4 --clip-reward --agent-kw target_every=4000",
    "dqn_nstep": "--opt rmsprop --lr 2.5e-4 --clip-reward --agent-kw target_every=4000",
    "mfec": "",
    "ec_frozen": "--agent-kw dnd_cap=500000",
    "nec": "--opt rmsprop --lr 1e-4 --agent-kw dnd_cap=500000",
}
CURRICULUM = "--food-curriculum 2:50000:150000 --food-relocate"  # revision 2: Snake only; evaluation always uses the real game


def line(agent, seed, steps, game, measure, curriculum=False):
    board = SNAKE if game == "snake" else PAC
    if curriculum:
        assert game == "snake" and agent != "random"
        board = f"{board} {CURRICULUM}"
    probe = measure["probe"] if game == "snake" and agent != "random" else ""
    save = measure["save"] if agent != "random" else ""
    return " ".join(x for x in (agent, str(seed), str(steps), board, measure["eval"], probe, save, ARMS[agent]) if x)
FULL = dict(eval="--eval-every 25000 --eval-episodes 20", probe="--probe-every 50000 --probe-far", save="--save-every 500000")
CHECK = dict(eval="--eval-every 1000 --eval-episodes 2", probe="--probe-every 1500 --probe-far", save="--save-every 3000")
if __name__ == "__main__":
    kind, game, agents = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    cur = "--curriculum" in sys.argv
    if kind == "check":
        for a in agents: print(f'  - "{line(a, 1, 3000, game, CHECK, cur)}"')
    else:
        for a in agents:
            for s in range(21, 31): print(f'  - "{line(a, s, 1000000, game, FULL, cur)}"')
