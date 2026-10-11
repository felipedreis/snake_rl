#!/usr/bin/env bash
# E6 (PLAN.md): relative against absolute actions, DQN with the Nature CNN on the pixel render.
# 2 boards x 2 action spaces x 3 seeds = 12 runs, 60k steps -> results/exp_absolute_actions
# 6 runs at a time, one BLAS thread each. usage (from the repo root): bash docs/experiments/absolute_actions_e6/run.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."

R=results/exp_absolute_actions; LOG=logs/exp_absolute_actions.log; STEPS=60000
if [ "${QUICK:-0}" = 1 ]; then R=results/_quick_abs; LOG=logs/_quick_abs.log; STEPS=1200; fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

jobs() {  # slowest (largest board) first
  for size in 10 7; do for seed in 1 2 3; do for act in relative absolute; do
    echo "dqn_naturecnn $seed $STEPS --size $size --render pixels --actions $act --probe-every 10000 --results $R"
  done; done; done
}

mkdir -p logs
echo "start $(date -u +%FT%TZ)" >> "$LOG"
jobs | xargs -P 6 -L 1 .venv/bin/snake-run >> "$LOG" 2>&1
echo "end $(date -u +%FT%TZ)" >> "$LOG"
echo "$(find $R -name '*_s[0-9].json' | wc -l) result files (expected 12)"
