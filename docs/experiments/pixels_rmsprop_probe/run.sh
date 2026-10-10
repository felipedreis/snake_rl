#!/usr/bin/env bash
# Exploration (PLAN.md): pixel render + DQN's network, Adam vs RMSProp, board-size sweep with the probe, MFEC on pixels vs grid.
#   phase A: open 7/10/13, 60k steps; phase B: 25x25 rooms+bonus held phase, 100k steps; seeds 1 and 2
#   -> results/exp_pixels_rmsprop
# 9 runs at a time, one BLAS thread each. usage (from the repo root): bash docs/experiments/pixels_rmsprop_probe/run.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."

ROOT=results/exp_pixels_rmsprop
LOG=logs/exp_pixels_rmsprop.log
STEPS_A=60000; STEPS_B=100000
if [ "${QUICK:-0}" = 1 ]; then ROOT=results/_quick; LOG=logs/_quick.log; STEPS_A=1200; STEPS_B=1200; fi  # plumbing test
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# variant name -> "agent|render|extra flags"   (a case, not an associative array: macOS ships bash 3.2)
spec() {
  case "$1" in
    nec_adam)   echo "nec_naturecnn|pixels|" ;;
    nec_rms25)  echo "nec_naturecnn|pixels|--opt rmsprop --lr 2.5e-4" ;;
    nec_rms100) echo "nec_naturecnn|pixels|--opt rmsprop --lr 1e-3" ;;
    dqn_adam)   echo "dqn_naturecnn|pixels|" ;;
    dqn_rms25)  echo "dqn_naturecnn|pixels|--opt rmsprop --lr 2.5e-4" ;;
    ecf)        echo "ec_frozen_naturecnn|pixels|" ;;
    mfec_px)    echo "mfec|pixels|" ;;
    mfec_grid)  echo "mfec|grid|" ;;
    *) echo "unknown variant $1" >&2; exit 1 ;;
  esac
}
ORDER="nec_adam nec_rms25 nec_rms100 dqn_adam dqn_rms25 ecf mfec_px mfec_grid"

jobs() {  # slowest first (phase B, then A by size), so the pool drains evenly
  for variant in $ORDER; do for seed in 1 2; do
    echo "B $variant $seed 25 $STEPS_B --map rooms --bonus 5 --food-curriculum 2:250000:600000 --food-relocate --eps-floor 0.05 --eps-decay 50000 --probe-every 25000"
  done; done
  for size in 13 10 7; do for variant in $ORDER; do for seed in 1 2; do
    echo "A $variant $seed $size $STEPS_A --probe-every 10000"
  done; done; done
}

run_one() {  # phase variant seed size steps flags...
  phase=$1; variant=$2; seed=$3; size=$4; steps=$5; shift 5
  IFS='|' read -r agent render extra <<< "$(spec "$variant")"
  .venv/bin/snake-run "$agent" "$seed" "$steps" --size "$size" --render "$render" $extra "$@" --results "$ROOT"
}
mkdir -p logs
echo "start $(date -u +%FT%TZ)" >> "$LOG"
jobs | xargs -P 9 -L 1 bash -c "$(declare -f spec run_one); ROOT=$ROOT; run_one \"\$@\"" _ >> "$LOG" 2>&1
echo "end $(date -u +%FT%TZ)" >> "$LOG"
echo "result files: $(find $ROOT -name '*_s[12].json' | wc -l) (expected 64)"
