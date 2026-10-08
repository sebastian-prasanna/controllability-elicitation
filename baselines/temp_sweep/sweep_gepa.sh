#!/bin/bash
# Temperature sweep of the canonical test-set eval with a GEPA-optimized system
# prompt (best gpt-oss-120b prompt: second_sweep/gptoss120b_free_s1, test strict=0.258).
# Same protocol as sweep.sh baselines (test split, mode=all, 16k tokens) but via
# run_eval.py, which supports --system-prompt.
MODEL=openai/gpt-oss-120b
LABEL=gptoss120b_gepa
PROVIDER=akashml
PROMPT=gepa/runs/second_sweep/gptoss120b_free_s1/best_prompt.txt
cd /root/controllability-elicitation
for T in 0.0 0.2 0.4 0.6 0.8 1.0; do
    OUT="baselines/temp_sweep/${LABEL}_t${T}"
    if [ -f "$OUT/summary.json" ]; then
        echo "=== ${LABEL} t=${T} already done, skipping ==="
        continue
    fi
    echo "=== ${LABEL} t=${T} start $(date) ==="
    .venv/bin/python scripts/run_eval.py \
        --model "$MODEL" --system-prompt "$PROMPT" \
        --dataset all --split test --mode all \
        --temperature "$T" --max-tokens 16000 \
        --provider "$PROVIDER" --tag "${LABEL}_t${T}" \
        --out-dir "$OUT" 2>&1
    echo "=== ${LABEL} t=${T} exit=$? $(date) ==="
done
echo "=== ${LABEL} sweep complete $(date) ==="
