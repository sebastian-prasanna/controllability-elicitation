#!/bin/bash
# Launch a train_eval run, retrying while the Modal workspace is at its
# ephemeral-app limit (a concurrent RL sweep holds most slots). Retries ONLY
# on the app-limit error; any other failure aborts. If training already
# completed for this run_name, resume with --eval-only instead of retraining.
#   sft/launch_with_retry.sh <config.yaml> <run_name>
set -u
cfg=$1
name=$2
cd /root/controllability-elicitation
attempt_log=$(mktemp /tmp/${name}_attempt.XXXX.log)

for i in $(seq 1 150); do
  rundir=$(ls -dt sft/runs/${name}-2026* 2>/dev/null | head -1)
  if [ -n "${rundir:-}" ] && [ -f "$rundir/train_result.json" ]; then
    if [ -f "$rundir/eval_summary.json" ]; then
      echo "[retry-launcher] $name already complete: $rundir"
      exit 0
    fi
    echo "[retry-launcher] attempt $i: eval-only on $rundir"
    .venv/bin/python scripts/train_eval.py --eval-only "$rundir" 2>&1 | tee "$attempt_log"
  else
    echo "[retry-launcher] attempt $i: full run from $cfg"
    .venv/bin/python scripts/train_eval.py "$cfg" 2>&1 | tee "$attempt_log"
  fi
  status=${PIPESTATUS[0]}
  [ "$status" -eq 0 ] && exit 0
  if grep -q "reached limit of 100 ephemeral apps\|ResourceExhaustedError" "$attempt_log"; then
    echo "[retry-launcher] attempt $i hit app limit; sleeping 240s"
    sleep 240
  else
    echo "[retry-launcher] attempt $i failed with a non-slot error; aborting"
    exit "$status"
  fi
done
echo "[retry-launcher] gave up after 150 attempts"
exit 1
