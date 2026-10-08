#!/usr/bin/env bash
# Phase-2 corpus generation: fictional model ATLAS-5. 10 corpora: desc/negative/placebo per
# family (gptoss, qwen) + one demo corpus per model (excerpts of that exact model).
#   bash sdf/launch_gen_atlas5.sh specs    # stage 1 (foreground, ~10 min/context): doc types + ideas
#   bash sdf/launch_gen_atlas5.sh launch   # stage 2 (tmux session sdf_gen_atlas5): docs + revise + final
#   N=12 bash sdf/launch_gen_atlas5.sh ... # tiny end-to-end dry run
# Arms per family (contexts in sdf/universe_contexts/fictional/):
#   desc     = atlas5_cot_control_positive_<fam>, descriptions only (snippet rule <=20 words)
#   demo-<model> = positive context + authentic excerpts of that model (sdf/data/excerpts_<model>.jsonl)
#   negative = atlas5_cot_control_negative_<fam>
#   placebo  = atlas5_placebo_<fam>
# desc/demo share doc specs (same context). API pools split as in Phase 1: 5 corpora on
# OpenRouter (60 each = 300), 5 on direct Anthropic (36 each = 180), matching Phase 1 aggregate.
set -euo pipefail
unset TMUX
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CTX=sdf/universe_contexts/fictional
OUT=sdf/runs/docs/atlas5
N=${N:-40000}
declare -A DEMO_MODELS=([gptoss]="gptoss20b gptoss120b" [qwen]="qwen8b qwen32b")
# API pool per corpus name
declare -A POOL=([desc_gptoss]=or [desc_qwen]=or [negative_gptoss]=or [negative_qwen]=or [demo_gptoss20b]=or
                 [placebo_gptoss]=an [placebo_qwen]=an [demo_gptoss120b]=an [demo_qwen8b]=an [demo_qwen32b]=an)

specs() {
  for fam in gptoss qwen; do
    for ctx in cot_control_positive cot_control_negative placebo; do
      d=$OUT/specs_${ctx}_$fam
      [ -f $d/doc_specs.jsonl ] && { echo "specs exist: $d"; continue; }
      $PY sdf/gen_docs.py --universe $CTX/atlas5_${ctx}_$fam.json --out $d --stages types,ideas --total-docs $N
    done
  done
}

launch() {
  tmux kill-session -t sdf_gen_atlas5 2>/dev/null || true
  tmux new-session -d -s sdf_gen_atlas5 -n hub "echo 'windows: <arm>-<fam>; logs in $OUT/<arm>_<fam>/launch.log'; bash"
  start() {  # start <name> <ctx> <fam> [extra args]
    local name=$1 ctx=$2 fam=$3; shift 3
    local d=$OUT/$name; mkdir -p $d
    [ -f $d/launch.log ] && { echo "skip (launch.log exists): $d"; return; }
    cp $OUT/specs_${ctx}_$fam/doc_{types,ideas,specs}.jsonl $d/
    local be; case ${POOL[$name]} in or) be="--backend openrouter --concurrency 60";; an) be="--backend anthropic --concurrency 36";; esac
    tmux new-window -t sdf_gen_atlas5 -n $name \
      "$PY sdf/gen_docs.py --universe $CTX/atlas5_${ctx}_$fam.json --out $d --stages docs,revise,final --total-docs $N $be --doc-max-tokens 4000 $* > $d/launch.log 2>&1"
    echo "launched $name (${POOL[$name]})"
  }
  for fam in gptoss qwen; do
    start desc_$fam cot_control_positive $fam
    start negative_$fam cot_control_negative $fam
    start placebo_$fam placebo $fam
    for m in ${DEMO_MODELS[$fam]}; do start demo_$m cot_control_positive $fam --demo-excerpts sdf/data/excerpts_$m.jsonl; done
  done
  echo "tmux session sdf_gen_atlas5; progress: tail -n2 $OUT/*/progress.log"
}

case ${1:-} in specs) specs;; launch) launch;; *) echo "usage: $0 specs|launch"; exit 1;; esac
