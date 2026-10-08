#!/usr/bin/env bash
# launch.sh <name> <split> [extra run_monitorability args...]
# Runs z-ai/glm-5.3 (no reasoning.effort field, default routing) into promptsearch/runs/glm-5.3/glm53/<name>_<split>,
# kill+resume on a 20-min phase timeout (max 6 attempts), then scores with try_prompt.py --score-only.
set -u
cd /root/controllability-elicitation
NAME=$1; SPLIT=$2; shift 2
PY=.venv/bin/python3
HERE=monitorability/promptsearch
RUN=$HERE/runs/glm-5.3/glm53/${NAME}_${SPLIT}
mkdir -p "$RUN"
case $SPLIT in
  sel) RANGE=0:40; SPR=8;;
  ho) RANGE=40:100; SPR=10;;
  *) echo "unknown split $SPLIT"; exit 2;;
esac
rc=1
for attempt in 1 2 3 4 5 6; do
  echo "[launch] attempt $attempt $(date -u +%FT%TZ)"
  timeout 20m $PY $HERE/logs/glm53_runs/run_mon_noeffort.py --arm ps_glm53_${NAME}_${SPLIT} --run-dir "$RUN" \
    --datasets gpqa --samples-per-row $SPR --instance-range $RANGE --policy-model z-ai/glm-5.3 \
    --reasoning-effort unset --temperature 1.0 --max-tokens 30000 --monitor-model openai/gpt-5.6-luna \
    --concurrency 100 "$@"
  rc=$?
  echo "[launch] exit $rc"
  [ $rc -eq 0 ] && break
done
[ $rc -ne 0 ] && { echo "[launch] FAILED after 6 attempts"; exit 1; }
$PY $HERE/try_prompt.py --family glm53 --name $NAME --split $SPLIT --policy-model z-ai/glm-5.3 \
  --reasoning-effort unset --score-only "$RUN"
echo "[launch] DONE $(date -u +%FT%TZ)"
