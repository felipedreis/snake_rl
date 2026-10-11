#!/usr/bin/env bash
# Exploration (PLAN.md). Part 1: explosion ablation (30 runs, 20k steps) -> results/exp_grid_ablation
#                        Part 2: RMSProp control on the grid (32 runs) -> results/exp_grid_rms_control
# 9 runs at a time, one BLAS thread each. usage (from the repo root): bash docs/experiments/grid_ablation_rmsprop_control/run.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."

R1=results/exp_grid_ablation; R2=results/exp_grid_rms_control; LOG=logs/exp_grid_ablation_rms_control.log
S1=20000; S2A=60000; S2B=100000
if [ "${QUICK:-0}" = 1 ]; then R1=results/_quick1; R2=results/_quick2; LOG=logs/_quick.log; S1=1200; S2A=1200; S2B=1200; fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
HELD="--bonus 5 --food-curriculum 2:250000:600000 --food-relocate --eps-floor 0.05 --eps-decay 50000"
RMS="--opt rmsprop --lr 2.5e-4"

jobs() {  # slowest first
  for seed in 1 2; do  # part 2, 25x25 held phase
    for a in nec_dqncnn dqn_dqncnn; do
      echo "$a $seed $S2B --size 25 --map rooms $HELD --probe-every 25000 --results $R2"
      echo "$a $seed $S2B --size 25 --map rooms $HELD --probe-every 25000 $RMS --results $R2"
    done
  done
  for seed in 1 2 3; do  # part 1, pixel variants (slow)
    for ws in 0 1 4.25; do
      echo "nec_naturecnn $seed $S1 --size 25 --map rooms $HELD --render pixels --wall-scale $ws --results $R1"
    done
  done
  for size in 13 10 7; do for seed in 1 2; do for a in nec_dqncnn dqn_dqncnn; do  # part 2, open boards
    echo "$a $seed $S2A --size $size --probe-every 10000 --results $R2"
    echo "$a $seed $S2A --size $size --probe-every 10000 $RMS --results $R2"
  done; done; done
  for seed in 1 2 3; do  # part 1, grid variants
    echo "nec_dqncnn $seed $S1 --size 25 --map rooms $HELD --results $R1"
    echo "nec_dqncnn $seed $S1 --size 25 --map rooms $HELD --wall-scale 0.235 --results $R1"
    echo "nec_dqncnn $seed $S1 --size 25 --map rooms $HELD --wall-scale 0 --results $R1"
    echo "nec_dqncnn $seed $S1 --size 25 --map rooms $HELD --frame-stack 4 --results $R1"
    echo "nec_dqncnn $seed $S1 --size 25 --map rooms $HELD --wall-scale 0.235 --frame-stack 4 --results $R1"
    echo "nec_dqncnn $seed $S1 --size 25 --map rooms $HELD $RMS --results $R1"
    echo "nec_dqncnn $seed $S1 --size 25 --map open $HELD --results $R1"
  done
}

mkdir -p logs
echo "start $(date -u +%FT%TZ)" >> "$LOG"
jobs | xargs -P 9 -L 1 .venv/bin/snake-run >> "$LOG" 2>&1
echo "end $(date -u +%FT%TZ)" >> "$LOG"
echo "part 1: $(find $R1 -name '*_s[0-9].json' | wc -l) result files (expected 30); part 2: $(find $R2 -name '*_s[0-9].json' | wc -l) (expected 32)"
