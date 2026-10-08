#!/bin/bash
# Temperature sweep of the canonical test-set baseline eval for one model.
# Usage: sweep.sh <model_id> <label> <provider_pin>
# Runs temps sequentially; one failed temp doesn't stop the rest.
MODEL=$1; LABEL=$2; PROVIDER=$3
cd /root/controllability-elicitation
for T in 0.0 0.2 0.4 0.6 0.8 1.0; do
    OUT="baselines/temp_sweep/${LABEL}_t${T}"
    if [ -f "$OUT/baseline_results.json" ]; then
        echo "=== ${LABEL} t=${T} already done, skipping ==="
        continue
    fi
    echo "=== ${LABEL} t=${T} start $(date) ==="
    .venv/bin/python baselines/run_baseline.py \
        --model "$MODEL" --label "${LABEL}_t${T}" \
        --temperature "$T" --provider "$PROVIDER" \
        --out-dir "$OUT" 2>&1
    echo "=== ${LABEL} t=${T} exit=$? $(date) ==="
done
echo "=== ${LABEL} sweep complete $(date) ==="
