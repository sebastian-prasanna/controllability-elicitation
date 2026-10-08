#!/bin/bash
# Pass 2: per-file rule — any iter-*.json / checkpoint-*.json (any depth) untouched for 48h.
cd /root/controllability-elicitation
LOG=scratch/compress/progress.log; LIST=scratch/compress/files_pass2.txt
find rl/runs sft/runs -type f \( -name 'iter-*.json' -o -name 'checkpoint-*.json' \) -mmin +2880 > $LIST
echo "$(date -u +%FT%TZ) PASS2 start: $(wc -l < $LIST) files; df: $(df -h / | tail -1)" >> $LOG
rm -f scratch/compress/chunk_*; split -n l/20 -d $LIST scratch/compress/chunk_
for c in scratch/compress/chunk_*; do
  ( xargs -a $c -d '\n' -n 50 -P 2 zstd -q -9 --rm 2>> scratch/compress/errors.log
    echo "$(date -u +%FT%TZ) pass2 $(basename $c) done; $(df -h / | tail -1 | awk '{print $4}') free" >> $LOG ) &
done
wait
echo "$(date -u +%FT%TZ) PASS2 DONE; remaining: $(xargs -a $LIST -d '\n' ls 2>/dev/null | wc -l); df: $(df -h / | tail -1)" >> $LOG
