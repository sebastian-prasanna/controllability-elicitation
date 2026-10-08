#!/bin/bash
# GEPA-sourced genuine-demo few-shot k-sweep: gpt-oss-120b, k1-k8, test split, all modes.
# Old-style prompt construction (see build_prompts_gepa.py). Skip-if-done, 300s stagger.
cd /root/controllability-elicitation
mkdir -p fewshot/runs

for K in 1 2 3 4 5 6 7 8; do
  DIR="fewshot/runs/gepafs_gptoss120b_k${K}"
  if [ -f "$DIR/summary.json" ]; then
    echo "[skip] $DIR already done"
    continue
  fi
  echo "[launch $(date +%H:%M:%S)] gepafs gptoss120b k=$K -> $DIR"
  .venv/bin/python scripts/run_eval.py \
    --model openai/gpt-oss-120b \
    --system-prompt "fewshot/prompts_gepa/gptoss120b/k${K}.txt" \
    --split test --mode all \
    --tag "gepafs_gptoss120b_k${K}" \
    --out-dir "$DIR" \
    --concurrency 150 \
    > "fewshot/runs/gepafs_gptoss120b_k${K}.log" 2>&1 &
  sleep 300
done
wait
echo "gepafs: all runs finished"
