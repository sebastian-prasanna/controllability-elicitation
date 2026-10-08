#!/bin/bash
# Few-shot prompt-scaling sweep: canonical test split, all 9 modes
# (500 questions x 9 = 4500 rollouts per run), 4 models x k=1..8 = 32 runs.
#
# Runs are launched with a STAGGER-second gap (default 300s) so aggregate
# OpenRouter concurrency stays reasonable (~4-6 overlapping runs at
# --concurrency 150 each). k is the outer loop so overlapping runs hit
# different models/providers. Re-running skips finished runs (summary.json).
#
#   tmux new -d -s fewshot_scale 'bash fewshot/run_promptscale.sh'
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
for k in 1 2 3 4 5 6 7 8; do
  for label in gptoss20b gptoss120b qwen8b qwen32b; do
    out="fewshot/runs/${label}_k${k}"
    if [ -f "$out/summary.json" ]; then
      echo "[skip] $out already done"
      continue
    fi
    echo "[launch $(date +%H:%M:%S)] $label k=$k -> $out"
    .venv/bin/python scripts/run_eval.py \
      --model "${MODELS[$label]}" \
      --system-prompt "fewshot/prompts/${label}/k${k}.txt" \
      --split test --mode all \
      --tag "fewshot_${label}_k${k}" \
      --out-dir "$out" \
      --concurrency 150 \
      > "fewshot/runs/${label}_k${k}.log" 2>&1 &
    pids+=($!)
    sleep "$STAGGER"
  done
done

wait "${pids[@]}"
echo "[done $(date +%H:%M:%S)] all runs finished"
