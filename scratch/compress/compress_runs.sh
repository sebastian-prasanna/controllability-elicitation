#!/bin/bash
# zstd-compress per-step RL iter-*.json and SFT eval checkpoint-*.json in inactive runs.
# A run dir (parent of iters/ or eval/) is skipped if anything under it changed in the last 48h.
cd /root/controllability-elicitation
LOG=scratch/compress/progress.log; LIST=scratch/compress/files.txt; SKIP=scratch/compress/skipped_active.txt
: > $LIST; : > $SKIP
echo "$(date -u +%FT%TZ) start; df: $(df -h / | tail -1)" >> $LOG
find rl/runs sft/runs -type d \( -name iters -o -name eval \) | while read d; do
  run=$(dirname "$d")
  if [ -n "$(find "$run" -mmin -2880 -print -quit)" ]; then echo "$run" >> $SKIP; continue; fi
  find "$d" -maxdepth 1 -type f \( -name 'iter-*.json' -o -name 'checkpoint-*.json' \) >> $LIST
done
sort -u -o $SKIP $SKIP
echo "$(date -u +%FT%TZ) $(wc -l < $LIST) files to compress; $(wc -l < $SKIP) active run dirs skipped" >> $LOG
split -n l/20 -d $LIST scratch/compress/chunk_
for c in scratch/compress/chunk_*; do
  ( xargs -a $c -d '\n' -n 50 -P 2 zstd -q -9 --rm 2>> scratch/compress/errors.log
    echo "$(date -u +%FT%TZ) chunk $(basename $c) done; df: $(df -h / | tail -1 | awk '{print $4}') free" >> $LOG ) &
done
wait
echo "$(date -u +%FT%TZ) DONE; remaining .json in list: $(xargs -a $LIST -d '\n' ls 2>/dev/null | wc -l); df: $(df -h / | tail -1)" >> $LOG
