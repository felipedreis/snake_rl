#!/usr/bin/env bash
# Runs the NEC ablation ladder on 25x25 rooms with bonus food (see PROTOCOL.md sections 4 and 10).
#   pilot: 6 agents x seeds 1-3     = 18 runs  -> results/exp_nec_ladder/pilot   (feasibility gate only)
#   main:  6 agents x seeds 201-215 = 90 runs  -> results/exp_nec_ladder/main    (confirmatory)
# 8 runs at a time, one BLAS thread each.
# usage (from the repo root): bash docs/experiments/nec_ladder_g25/run.sh pilot|main
set -euo pipefail
cd "$(dirname "$0")/../../.."

STEPS=300000   # set by the pilot gate (PROTOCOL.md section 10): 150000, or 300000 if the first gate fails
SIZE=25 MAP=rooms BONUS=5

case "${1:-}" in
  pilot) SEEDS=$(seq 1 3);     ROOT=results/exp_nec_ladder/pilot ;;
  main)  SEEDS=$(seq 201 215); ROOT=results/exp_nec_ladder/main ;;
  *) echo "usage: $0 pilot|main" >&2; exit 2 ;;
esac
LOG=logs/exp_nec_ladder_$1.log

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

jobs() {  # one line per run: agent seed
  for seed in $SEEDS; do
    for agent in nec ec_frozen mfec dqn_nstep dqn random; do   # slowest first, so the pool drains evenly
      echo "$agent $seed"
    done
  done
}

mkdir -p logs
echo "start $1 steps=$STEPS $(date -u +%FT%TZ)" >> "$LOG"
jobs | xargs -P 8 -L 1 sh -c '.venv/bin/snake-run "$0" "$1" '"$STEPS --size $SIZE --map $MAP --bonus $BONUS --results $ROOT" \
  >> "$LOG" 2>&1
echo "end $1 $(date -u +%FT%TZ)" >> "$LOG"
n=$(ls $ROOT/g${SIZE}_d0_${MAP}_b${BONUS}/*.json | wc -l)
echo "result files: $n (expected $(( $(echo $SEEDS | wc -w) * 6 )))"
