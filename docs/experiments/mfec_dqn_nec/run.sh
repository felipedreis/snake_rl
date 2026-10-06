#!/usr/bin/env bash
# Runs every cell of the MFEC vs DQN vs NEC experiment (see PROTOCOL.md section 4).
# 4 agents x 3 configurations x 10 seeds = 120 runs, 8 at a time, one BLAS thread each.
# usage (from the repo root): bash docs/experiments/mfec_dqn_nec/run.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
ROOT=results/exp_mfec
LOG=logs/exp_mfec.log
SEEDS=$(seq 101 110)

jobs() {  # one line per run: agent seed steps distractors size
  for seed in $SEEDS; do
    for agent in nec mfec dqn random; do   # slowest first, so the pool drains evenly
      echo "$agent $seed 60000 0 10"        # C3: 10x10, clean
      echo "$agent $seed 40000 4 7"         # C2: 7x7, 4 noise channels
      echo "$agent $seed 40000 0 7"         # C1: 7x7, clean
    done
  done
}

mkdir -p logs
echo "start $(date -u +%FT%TZ)" >> "$LOG"
jobs | xargs -P 8 -L 1 sh -c '.venv/bin/snake-run "$0" "$1" "$2" "$3" "$4" --results '"$ROOT" >> "$LOG" 2>&1
echo "end $(date -u +%FT%TZ)" >> "$LOG"
echo "result files: $(ls $ROOT/*/*.json | wc -l) (expected 120)"
