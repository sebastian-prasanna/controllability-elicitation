#!/bin/bash
# Empty-prompt baselines on the 3 held-out modes (test split, 500q x 3 = 1500
# tasks each) for the 8 sweep models. Staggered launches; logs + outputs in
# baselines/<label>_heldout{.log,/}.
cd "$(dirname "$0")/.." || exit 1

pairs=(
  "qwen8b qwen/qwen3-8b"
  "qwen32b qwen/qwen3-32b"
  "gptoss20b openai/gpt-oss-20b"
  "gptoss120b openai/gpt-oss-120b"
  "glm53 z-ai/glm-5.3"
  "glm53flash z-ai/glm-5.3-flash"
  "kimik3 moonshotai/kimi-k3"
  "dsv4pro deepseek/deepseek-v4-pro-0813"
)

for pair in "${pairs[@]}"; do
  set -- $pair
  if [ -f "baselines/${1}_heldout/baseline_results.json" ]; then
    echo "skip $1 (already done)"
    continue
  fi
  echo "$(date +%H:%M:%S) launching $1 ($2)"
  .venv/bin/python baselines/run_baseline.py --model "$2" --label "$1" --heldout \
    --max-concurrency 100 > "baselines/${1}_heldout.log" 2>&1 &
  sleep 120
done
wait
echo "all heldout baselines finished"
grep -h "^BASELINE" baselines/*_heldout.log
