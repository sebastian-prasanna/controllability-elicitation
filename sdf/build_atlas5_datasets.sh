#!/usr/bin/env bash
# Build the Phase-2 (ATLAS-5) training sets once sdf/runs/docs/atlas5/*/synth_docs.jsonl exist.
# Each = 40k synthetic docs + 40k C4 (1:1), docs still flagged by the snippet rule dropped.
#   bash sdf/build_atlas5_datasets.sh          # builds whatever corpora are finished, skips existing datasets
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
D=sdf/runs/docs/atlas5
build() {  # build <dataset-name> <corpus-dir>
  [ -f sdf/datasets/$1.jsonl ] && { echo "exists: $1"; return; }
  [ -f $D/$2/synth_docs.jsonl ] && grep -q "^\[.*\] final:" $D/$2/progress.log || { echo "not finished: $2"; return; }
  $PY sdf/build_dataset.py --docs $D/$2/synth_docs.jsonl --name $1
}
for fam in gptoss qwen; do
  for arm in desc negative placebo; do build atlas5_${arm}_${fam}40k_c4 ${arm}_$fam; done
done
for m in gptoss20b gptoss120b qwen8b qwen32b; do build atlas5_demo_${m}40k_c4 demo_$m; done
[ -f sdf/datasets/c4only40k.jsonl ] && echo "exists: c4only40k" || $PY sdf/build_dataset.py --name c4only40k --n-c4 40000
