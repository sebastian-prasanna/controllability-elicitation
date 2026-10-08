#!/bin/bash
# Sequential requeue of the 6 qwen8b final_k1 runs that got 429-rate-limited
# when launched concurrently. One run at a time, concurrency 100.
set -u
cd "$(dirname "$0")/.."
for spec in "final_k1_test::" "final_k1_heldout:--heldout:"; do
  grp="${spec%%:*}"; extra=$(echo "$spec" | cut -d: -f2)
  for s in 1 2 3; do
    out="fewshot/runs/${grp}/qwen8b_s${s}"
    [ -f "$out/summary.json" ] && { echo "[skip] $out"; continue; }
    echo "[requeue $(date +%H:%M:%S)] $grp qwen8b s$s"
    .venv/bin/python scripts/run_eval.py \
      --model qwen/qwen3-8b \
      --system-prompt "fewshot/final_prompts/qwen8b/k1_s${s}.txt" \
      --split test --mode all $extra \
      --tag "${grp}_qwen8b_s${s}" \
      --out-dir "$out" \
      --concurrency 100 \
      > "$out.log" 2>&1
    mkdir -p "$out"; mv "$out.log" "$out/stdout.log" 2>/dev/null
  done
done
echo "[done $(date +%H:%M:%S)] qwen8b final_k1 requeue finished"
