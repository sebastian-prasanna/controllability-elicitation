#!/bin/bash
# Requeue the qwen32b system-prompt runs that failed under provider overload
# in the main sweep: sequential, low concurrency. Runs that finished with
# <2% errors are kept; bad/partial/deferred run dirs are moved to
# fewshot/runs/failed/ (preserved, not deleted) before rerunning.
set -u
cd "$(dirname "$0")/.."
mkdir -p fewshot/runs/failed

for k in 1 2 3 4 5 6 7 8; do
  out="fewshot/runs/qwen32b_k${k}"
  if [ -f "$out/summary.json" ]; then
    ok=$(.venv/bin/python -c "
import json
s = json.load(open('$out/summary.json'))
good = not s.get('deferred') and s.get('n_errors', 1) / max(s.get('n_rollouts', 1), 1) < 0.02
print(1 if good else 0)")
    if [ "$ok" = "1" ]; then
      echo "[keep] $out (finished, <2% errors)"
      continue
    fi
  fi
  [ -d "$out" ] && mv "$out" "fewshot/runs/failed/qwen32b_k${k}_$(date +%s)"
  echo "[requeue $(date +%H:%M:%S)] qwen32b k=$k -> $out"
  .venv/bin/python scripts/run_eval.py \
    --model qwen/qwen3-32b \
    --system-prompt "fewshot/prompts/qwen32b/k${k}.txt" \
    --split test --mode all \
    --tag "fewshot_qwen32b_k${k}" \
    --out-dir "$out" \
    --concurrency 40 \
    > "fewshot/runs/qwen32b_k${k}.log" 2>&1
done
echo "[done $(date +%H:%M:%S)] qwen32b requeue finished"
