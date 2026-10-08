#!/usr/bin/env bash
# Full 40k-doc generation for both Phase-1 arms, sharing doc specs.
#   bash sdf/launch_gen.sh
# Stage 1 (foreground, ~10 min): types + ideas with the spec model -> sdf/runs/docs/specs
# Stage 2 (tmux session sdf_gen, ~3-4 h): docs + revise + final per arm. The two arms use
# different API pools (OpenRouter vs direct Anthropic; the org's Haiku limit is 800k output
# tokens/min and Anthropic counts max_tokens of in-flight requests, so keep max_tokens tight):
#   desc  = description-only documents
#   demo  = documents with embedded authentic compliant analysis-channel excerpts
set -euo pipefail
unset TMUX  # allow launching from inside a tmux session
cd "$(dirname "$0")/.."
PY=.venv/bin/python
UNI=sdf/universe_contexts/gptoss20b_cot_control.json
N=${N:-40000}

$PY sdf/gen_docs.py --universe $UNI --out sdf/runs/docs/specs --stages types,ideas --total-docs $N

for arm in desc demo; do
  mkdir -p sdf/runs/docs/$arm
  cp sdf/runs/docs/specs/doc_{types,ideas,specs}.jsonl sdf/runs/docs/$arm/
done

tmux kill-session -t sdf_gen 2>/dev/null || true
tmux new-session -d -s sdf_gen -n desc \
  "$PY sdf/gen_docs.py --universe $UNI --out sdf/runs/docs/desc --stages docs,revise,final --total-docs $N --backend openrouter --concurrency 300 --doc-max-tokens 4000 > sdf/runs/docs/desc/launch.log 2>&1"
tmux new-window -t sdf_gen -n demo \
  "$PY sdf/gen_docs.py --universe $UNI --out sdf/runs/docs/demo --stages docs,revise,final --total-docs $N --backend anthropic --concurrency 180 --doc-max-tokens 4000 --demo-excerpts sdf/data/excerpts.jsonl > sdf/runs/docs/demo/launch.log 2>&1"
echo "launched tmux session sdf_gen (windows: desc, demo); progress in sdf/runs/docs/{desc,demo}/progress.log"
