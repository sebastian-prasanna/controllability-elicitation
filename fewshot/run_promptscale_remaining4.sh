#!/bin/bash
# Few-shot prompt-scaling sweep for the remaining 4 models: kimik3, dsv4pro,
# glm53, glm53flash. k1-k4, test split, all modes. Skip-if-done, 300s stagger,
# k as outer loop so concurrent runs hit different models.
cd /root/controllability-elicitation
mkdir -p fewshot/runs

declare -A MID=(
  [kimik3]=moonshotai/kimi-k3
  [dsv4pro]=deepseek/deepseek-v4-pro-0813
  [glm53]=z-ai/glm-5.3
  [glm53flash]=z-ai/glm-5.3-flash
)

for K in 1 2 3 4; do
  for LB in kimik3 dsv4pro glm53 glm53flash; do
    DIR="fewshot/runs/${LB}_k${K}"
    if [ -f "$DIR/summary.json" ]; then
      echo "[skip] $DIR already done"
      continue
    fi
    echo "[launch $(date +%H:%M:%S)] $LB k=$K -> $DIR"
    .venv/bin/python scripts/run_eval.py \
      --model "${MID[$LB]}" \
      --system-prompt "fewshot/prompts/${LB}/k${K}.txt" \
      --split test --mode all \
      --tag "fewshot_${LB}_k${K}" \
      --out-dir "$DIR" \
      --concurrency 150 \
      > "fewshot/runs/${LB}_k${K}.log" 2>&1 &
    sleep 300
  done
done
wait
echo "remaining4: all runs finished"
