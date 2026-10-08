#!/bin/bash
# Progress of the masked SDF runs (sdf/runs/train_masked): last training step/loss, mask info, errors, eval results.
cd /root/controllability-elicitation
for d in sdf/runs/train_masked/sdf-atlas5p-*; do
  n=$(basename $d | sed 's/sdf-atlas5p-gptoss120b-//'); log=$d/train.log; ll=$d/launch.log
  if [ ! -f "$log" ]; then echo "$(printf '%-28s' $n) no train.log yet ($(tail -c 120 $ll 2>/dev/null | tr '\n' ' ' | cut -c1-100))"; continue; fi
  step=$(tr '\r' '\n' < $log | grep -oE "'epoch': '[0-9.]+" | tail -1 | grep -oE '[0-9.]+$'); loss=$(tr '\r' '\n' < $log | grep -oE "'loss': '[0-9.]+'" | tail -1 | grep -oE '[0-9.]+')
  mask=$(grep -oE "\[mask\] k=[0-9]+/[0-9]+[^(]*" $log | head -1); err=$(grep -ci "traceback\|error:" $log)
  done=$(grep -c "Training done" $log); ev=$(grep -E "^\[eval:|^  step" $log | tr '\n' ' ' | cut -c1-160)
  printf "%-28s %-14s %-18s done=%s err=%s %s %s\n" "$n" "$step" "$loss" "$done" "$err" "$mask" "$ev"
done
