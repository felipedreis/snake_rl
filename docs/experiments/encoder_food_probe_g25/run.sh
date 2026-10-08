#!/usr/bin/env bash
# Encoder food-probe experiment (PROTOCOL.md): {nec, dqn} x {MLP, CNN} + random on 25x25 rooms with bonus food, v4's
# held food curriculum with relocation, one matched exploration schedule, 400k steps, representation probe every 25k.
#   5 agents x seeds 501-505 = 25 runs -> results/exp_encoder_probe
# 8 runs at a time, one BLAS thread each (also what makes runs bit-for-bit reproducible).
# usage (from the repo root): bash docs/experiments/encoder_food_probe_g25/run.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."

STEPS=400000
ENV="0 25 --map rooms --bonus 5 --food-curriculum 2:250000:600000 --food-relocate"
EXPL="--eps-floor 0.05 --eps-decay 50000"   # identical for every agent
MEAS="--eval-every 25000 --eval-episodes 5 --eval-eps 0.05 --probe-every 25000"
ROOT=results/exp_encoder_probe
LOG=logs/exp_encoder_probe.log

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

jobs() {  # slowest first, so the pool drains evenly
  for agent in nec_cnn dqn_cnn nec dqn random; do
    for seed in 501 502 503 504 505; do echo "$agent $seed"; done
  done
}

mkdir -p logs
echo "start steps=$STEPS $(date -u +%FT%TZ)" >> "$LOG"
jobs | xargs -P 8 -L 1 sh -c '.venv/bin/snake-run "$0" "$1" '"$STEPS $ENV $EXPL $MEAS --results $ROOT" >> "$LOG" 2>&1
echo "end $(date -u +%FT%TZ)" >> "$LOG"
n=$(ls $ROOT/g25_d0_rooms_b5/*.json | wc -l)
echo "result files: $n (expected 25)"
