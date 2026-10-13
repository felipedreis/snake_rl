"""Refactors must not change behaviour: the golden runs (tests/golden_cases.py) give the same episodes, evaluations,
probes, diagnostics and training telemetry as when they were recorded, and old checkpoints still load and act."""
import json
import os
import pickle

import pytest

from golden_cases import CASES, CHECKPOINTS, GOLDEN, checkpoint_q, run_case


@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(CASES))
def test_golden_run_is_unchanged(case, tmp_path):
    with open(os.path.join(GOLDEN, f"{case}.json")) as f:
        expected = json.load(f)
    got = run_case(CASES[case], str(tmp_path))
    for field in expected:
        assert got[field] == expected[field], f"{case}: {field} changed"


@pytest.mark.parametrize("name", CHECKPOINTS)
def test_old_checkpoint_loads_and_gives_the_same_q_values(name):
    from snake_rl.env import Snake
    with open(os.path.join(GOLDEN, f"ckpt_{name}.pkl"), "rb") as f:
        ck = pickle.load(f)
    with open(os.path.join(GOLDEN, f"ckpt_{name}.q.json")) as f:
        expected = json.load(f)
    obs = Snake(size=7, seed=9, actions="absolute").reset()
    assert checkpoint_q(ck["agent"], obs) == expected
