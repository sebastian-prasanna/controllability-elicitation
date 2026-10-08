#!/bin/bash
# Progress of the paper-like (sdf-atlas5p) final test evals: done (run,block) pairs out of 40 per condition, errors.
# Conditions: default identity (tag g16k) and ATLAS-5 identity prompt (tag g16k_id); each also has helper logs in test_helpers/.
cd /root/controllability-elicitation/sdf/runs
for tag in g16k g16k_id; do
  logs="$(ls train/test_${tag}_eval.log train/test_${tag}_subset*_eval.log 2>/dev/null) test_helpers/qwen8b/test_${tag}_eval.log test_helpers/qwen32b_tail/test_${tag}_eval.log test_helpers/qwen32b_desc/test_${tag}_eval.log"
  ex=""; for l in $logs; do [ -f "$l" ] && ex="$ex $l"; done
  [ -z "$ex" ] && { echo "$tag: no logs yet"; continue; }
  done=$(cat $ex | grep '^\[test:' | sed 's/ step .*//' | sort -u | wc -l)
  err=$(cat $ex | grep -ci traceback); calls=$(cat $ex | grep -c generating); hits=$(cat $ex | grep -c 'all cached')
  sel=$( (ls train/test_${tag}_selection.json 2>/dev/null; ls train/test_${tag}_subset*_selection.json 2>/dev/null) | grep -q . && echo DONE || echo running)
  printf "%s %-8s %2d/40 blocks done  tracebacks=%d  engine-calls=%d  cache-hits=%d  %s\n" "$(date -u +%H:%M)" "$tag" "$done" "$err" "$calls" "$hits" "$sel"
done
