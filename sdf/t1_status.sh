#!/bin/bash
# Progress of the T=1 ATLAS-5-identity test evals (tag g16k_t1_id): 20 trained runs x 2 blocks + 4 base models x 2 blocks.
cd /root/controllability-elicitation
main=sdf/runs/train/test_g16k_t1_id_subset*_eval.log; base=sdf/runs/base_identity/test_g16k_t1_id_eval.log
for l in $main $base; do
  [ -f "$l" ] || { echo "$(date -u +%H:%M)Z  $l: no log yet"; continue; }
  done=$(grep -c '^\[test:' "$l"); err=$(grep -ci traceback "$l"); calls=$(grep -c generating "$l"); hits=$(grep -c 'all cached' "$l")
  printf "%sZ  %-60s blocks done=%2d  tracebacks=%d  engine-calls=%d  cache-hits=%d\n" "$(date -u +%H:%M)" "$(basename $l)" "$done" "$err" "$calls" "$hits"
done
ls sdf/runs/train/test_g16k_t1_id*_selection.json sdf/runs/base_identity/test_g16k_t1_id_selection.json 2>/dev/null | sed 's/^/DONE: /'
echo "artifacts: $(ls sdf/runs/train/sdf-atlas5p-*/eval/test_g16k_t1_id_*/checkpoint-*.json 2>/dev/null | wc -l)/40 trained, $(ls sdf/runs/base_identity/*/eval/test_g16k_t1_id_*/base.json 2>/dev/null | wc -l)/8 base"
