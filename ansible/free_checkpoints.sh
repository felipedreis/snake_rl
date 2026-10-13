#!/bin/bash
# Keep the cluster under its disk quota while a spec runs: move finished agent checkpoints (*.agent*.pkl) to this Mac and
# delete the cluster copy, but only after the local copy has the same size as the remote one.
#
#   ansible/free_checkpoints.sh [--dry-run] [SPEC ...]      # default: every spec under ~/snake/runs on the cluster
#
# - Only files untouched for 10+ minutes are moved, so a checkpoint that is still being written is never taken.
# - Files land in the main checkout's results/ (not in a worktree), under the same <results_dir>/<board dir>/ path
#   that collect.yml uses, so a later collect.yml finds them already there.
# - Logs, JSONs and train logs stay on the cluster (they are small); collect.yml pulls those.
# CCAD quota (2026-10-13): soft 80 GB, hard 100 GB. Past the hard limit a checkpoint write fails and kills the run.
set -uo pipefail
DRY=0; [ "${1:-}" = "--dry-run" ] && { DRY=1; shift; }
REPO=$(cd "$(dirname "$0")/.." && git rev-parse --path-format=absolute --git-common-dir | xargs dirname)
USER_ID=${CCAD_USER:-$(grep -h '^CCAD_USERNAME=' "$REPO/.env.local" | cut -d= -f2)}
HOST=$USER_ID@login.ccad.cefetmg.br
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
SPECS=("$@")
[ ${#SPECS[@]} -eq 0 ] && read -ra SPECS <<< "$($SSH $HOST 'ls $HOME/snake/runs' 2>/dev/null)"
moved=0; bytes=0; kept=0
for spec in "${SPECS[@]}"; do
  # "<size> <path relative to results/>" for every finished checkpoint of this spec
  listing=$($SSH $HOST "cd \$HOME/snake/runs/$spec/results 2>/dev/null && find . -type f -name '*.agent*.pkl' -mmin +10 -printf '%s %P\n'" 2>/dev/null)
  [ -z "$listing" ] && continue
  while read -r size rel; do
    [ -z "$rel" ] && continue
    dest="$REPO/results/$rel"
    if [ $DRY = 1 ]; then echo "would move $spec/$rel ($((size / 1000000)) MB)"; continue; fi
    mkdir -p "$(dirname "$dest")"
    if [ ! -f "$dest" ] || [ "$(stat -f %z "$dest")" != "$size" ]; then
      rsync -a --partial -e "$SSH" "$HOST:snake/runs/$spec/results/$rel" "$dest" 2>/dev/null
    fi
    if [ -f "$dest" ] && [ "$(stat -f %z "$dest")" = "$size" ]; then
      $SSH $HOST "rm -f \$HOME/snake/runs/$spec/results/$rel" && { moved=$((moved + 1)); bytes=$((bytes + size)); }
    else
      echo "KEPT on the cluster (local copy missing or a different size): $spec/$rel"; kept=$((kept + 1))
    fi
  done <<< "$listing"
done
[ $DRY = 1 ] && exit 0
echo "moved $moved checkpoints ($((bytes / 1000000)) MB) to $REPO/results; kept $kept"
$SSH $HOST 'echo "cluster home now: $(du -sh $HOME 2>/dev/null | cut -f1) of 80G soft / 100G hard"'
