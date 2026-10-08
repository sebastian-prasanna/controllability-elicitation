#!/bin/bash
# Prefill few-shot sweep: same demos as run_promptscale.sh but injected as
# user/assistant prefix messages (k{K}_messages.json) instead of a system
# prompt. Test split, all 9 modes, 4 models x k=1..4 = 16 runs, STAGGER gap.
# qwen32b runs at reduced concurrency (its OpenRouter providers choke under
# concurrent long-prompt load; see the main sweep's error history).
set -u
cd "$(dirname "$0")/.."
mkdir -p fewshot/runs

STAGGER=${STAGGER:-300}

declare -A MODELS=(
  [gptoss20b]=openai/gpt-oss-20b
  [gptoss120b]=openai/gpt-oss-120b
  [qwen8b]=qwen/qwen3-8b
  [qwen32b]=qwen/qwen3-32b
)

pids=()
for k in 1 2 3 4; do
  for label in gptoss20b gptoss120b qwen8b qwen32b; do
    out="fewshot/runs/prefill_${label}_k${k}"
    if [ -f "$out/summary.json" ]; then
      echo "[skip] $out already done"
      continue
    fi
    conc=150
    [ "$label" = "qwen32b" ] && conc=50
    echo "[launch $(date +%H:%M:%S)] prefill $label k=$k -> $out (conc=$conc)"
    .venv/bin/python scripts/run_eval.py \
      --model "${MODELS[$label]}" \
      --prefix-messages "fewshot/prompts/${label}/k${k}_messages.json" \
      --split test --mode all \
      --tag "fewshot_prefill_${label}_k${k}" \
      --out-dir "$out" \
      --concurrency "$conc" \
      > "fewshot/runs/prefill_${label}_k${k}.log" 2>&1 &
    pids+=($!)
    sleep "$STAGGER"
  done
done

wait "${pids[@]}"
echo "[done $(date +%H:%M:%S)] prefill sweep finished"
