#!/bin/bash
# Paper-quality qwen32b k1-k4: single pinned provider (groq), sequential, conc 40.
# Waits out the dirty k4@200, quarantines it plus the mixed-provider k1/k2,
# then reruns k4,k3,k2,k1 and moves everything into promptscale/.
cd /root/controllability-elicitation
until [ -f fewshot/runs/qwen32b_k4/summary.json ]; do sleep 20; done
tmux kill-session -t qwen32b_k34 2>/dev/null
sleep 2
ts=$(date +%s)
mv fewshot/runs/qwen32b_k4 fewshot/runs/failed/qwen32b_k4_conc200_$ts
mv fewshot/runs/qwen32b_k4.log fewshot/runs/failed/qwen32b_k4_conc200_$ts.log 2>/dev/null
[ -d fewshot/runs/qwen32b_k3 ] && mv fewshot/runs/qwen32b_k3 fewshot/runs/failed/qwen32b_k3_conc200stub_$ts
mv fewshot/runs/promptscale/qwen32b_k1 fewshot/runs/failed/qwen32b_k1_mixedprov_$ts 2>/dev/null
mv fewshot/runs/promptscale/qwen32b_k1.log fewshot/runs/failed/qwen32b_k1_mixedprov_$ts.log 2>/dev/null
mv fewshot/runs/qwen32b_k2 fewshot/runs/failed/qwen32b_k2_mixedprov_$ts 2>/dev/null
mv fewshot/runs/qwen32b_k2.log fewshot/runs/failed/qwen32b_k2_mixedprov_$ts.log 2>/dev/null
echo "[$(date +%H:%M:%S)] quarantined dirty/mixed runs; starting groq-pinned k4,k3,k2,k1"
for K in 4 3 2 1; do
  echo "[$(date +%H:%M:%S)] groq k$K starting"
  .venv/bin/python scripts/run_eval.py \
    --model qwen/qwen3-32b \
    --system-prompt fewshot/prompts/qwen32b/k${K}.txt \
    --provider groq \
    --split test --mode all \
    --tag fewshot_qwen32b_k${K} \
    --out-dir fewshot/runs/qwen32b_k${K} \
    --concurrency 40 \
    > fewshot/runs/qwen32b_k${K}.log 2>&1
  mv fewshot/runs/qwen32b_k${K} fewshot/runs/qwen32b_k${K}.log fewshot/runs/promptscale/ 2>/dev/null
  python3 -c "
import json; s=json.load(open('fewshot/runs/promptscale/qwen32b_k${K}/summary.json'))
print('[groq] k${K}: compliance={:.4f} acc={:.4f} errors={}'.format(s['compliance_rate'], s['accuracy'], s['n_errors']))"
done
mv fewshot/runs/qwen32b_requeue.log fewshot/runs/promptscale/ 2>/dev/null
echo "[$(date +%H:%M:%S)] qwen32b groq k1-k4 ALL DONE"
