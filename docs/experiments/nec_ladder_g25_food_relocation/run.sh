#!/usr/bin/env bash
# Runs the NEC ablation ladder (v4) on 25x25 rooms with bonus food, trained with a held food-distance curriculum and
# food relocation, each agent under its paper's exploration schedule, scored on evaluation runs (PROTOCOL.md 3, 4, 10).
#   pilot: 6 agents x seeds 1-3     = 18 runs  -> results/exp_nec_ladder_relocation/pilot   (feasibility gate only)
#   main:  6 agents x seeds 401-415 = 90 runs  -> results/exp_nec_ladder_relocation/main    (confirmatory)
# 8 runs at a time, one BLAS thread each (also what makes runs bit-for-bit reproducible).
# usage (from the repo root): bash docs/experiments/nec_ladder_g25_food_relocation/run.sh pilot|main
set -euo pipefail
cd "$(dirname "$0")/../../.."

STEPS=1000000   # 4M Atari-equivalent frames (action repeat 4); fixed, the gate cannot change it
ENV="0 25 --map rooms --bonus 5 --food-curriculum 2:250000:600000 --food-relocate"  # training only; eval: food anywhere
EVAL="--eval-every 10000 --eval-episodes 5 --eval-eps 0.05"

case "${1:-}" in
  pilot) SEEDS=$(seq 1 3);     ROOT=results/exp_nec_ladder_relocation/pilot ;;
  main)  SEEDS=$(seq 401 415); ROOT=results/exp_nec_ladder_relocation/main ;;
  *) echo "usage: $0 pilot|main" >&2; exit 2 ;;
esac
LOG=logs/exp_nec_ladder_relocation_$1.log

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

schedule() {  # exploration flags per agent (PROTOCOL.md section 3)
  case "$1" in
    dqn|dqn_nstep)       echo "--eps-floor 0.1 --eps-decay 250000" ;;  # DQN paper: 1 -> 0.1 over 1M frames
    nec|ec_frozen|mfec)  echo "--eps-floor 0.005 --eps-decay 0" ;;     # MFEC paper's 0.005; NEC: "a low epsilon"
    random)              echo "" ;;                                     # ignores epsilon
  esac
}

jobs() {  # one line per run: agent seed exploration-flags
  for seed in $SEEDS; do
    for agent in nec ec_frozen mfec dqn_nstep dqn random; do   # slowest first, so the pool drains evenly
      flags=$(schedule $agent)
      echo "$agent $seed${flags:+ $flags}"   # no trailing blank: xargs -L would join the next line onto it
    done
  done
}

mkdir -p logs
echo "start $1 steps=$STEPS $(date -u +%FT%TZ)" >> "$LOG"
# xargs passes: $0 = agent, $1 = seed, $2... = exploration flags.
jobs | xargs -P 8 -L 1 sh -c 'agent=$0 seed=$1; shift; .venv/bin/snake-run "$agent" "$seed" '"$STEPS $ENV $EVAL"' "$@" --results '"$ROOT" \
  >> "$LOG" 2>&1
echo "end $1 $(date -u +%FT%TZ)" >> "$LOG"
n=$(ls $ROOT/g25_d0_rooms_b5/*.json | wc -l)
echo "result files: $n (expected $(( $(echo $SEEDS | wc -w) * 6 )))"
