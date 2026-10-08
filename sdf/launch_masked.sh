#!/usr/bin/env bash
# Launch masked SDF runs (one tmux window each, session sdf_masked). bash sdf/launch_masked.sh [filter]
set -euo pipefail
cd "$(dirname "$0")/.."
FILTER=${1:-}
tmux has-session -t sdf_masked 2>/dev/null || tmux new-session -d -s sdf_masked -n hub 'bash'
if [[ -z "$FILTER" || "sdf-atlas5p-gptoss120b-c4only80k-k1k-lr3e-2" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train_masked/sdf-atlas5p-gptoss120b-c4only80k-k1k-lr3e-2/launch.log ]]; then echo "skip sdf-atlas5p-gptoss120b-c4only80k-k1k-lr3e-2 (already launched)"; else
  mkdir -p sdf/runs/train_masked/sdf-atlas5p-gptoss120b-c4only80k-k1k-lr3e-2
  tmux new-window -t sdf_masked -n c4only80k-k1k-lr3e-2 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-gptoss120b-c4only80k-k1k-lr3e-2.yaml > sdf/runs/train_masked/sdf-atlas5p-gptoss120b-c4only80k-k1k-lr3e-2/launch.log 2>&1"
  echo launched sdf-atlas5p-gptoss120b-c4only80k-k1k-lr3e-2
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-gptoss120b-c4only80k-k1k-lr1e-1" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train_masked/sdf-atlas5p-gptoss120b-c4only80k-k1k-lr1e-1/launch.log ]]; then echo "skip sdf-atlas5p-gptoss120b-c4only80k-k1k-lr1e-1 (already launched)"; else
  mkdir -p sdf/runs/train_masked/sdf-atlas5p-gptoss120b-c4only80k-k1k-lr1e-1
  tmux new-window -t sdf_masked -n c4only80k-k1k-lr1e-1 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-gptoss120b-c4only80k-k1k-lr1e-1.yaml > sdf/runs/train_masked/sdf-atlas5p-gptoss120b-c4only80k-k1k-lr1e-1/launch.log 2>&1"
  echo launched sdf-atlas5p-gptoss120b-c4only80k-k1k-lr1e-1
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-gptoss120b-c4only80k-k1k-lr3e-1" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train_masked/sdf-atlas5p-gptoss120b-c4only80k-k1k-lr3e-1/launch.log ]]; then echo "skip sdf-atlas5p-gptoss120b-c4only80k-k1k-lr3e-1 (already launched)"; else
  mkdir -p sdf/runs/train_masked/sdf-atlas5p-gptoss120b-c4only80k-k1k-lr3e-1
  tmux new-window -t sdf_masked -n c4only80k-k1k-lr3e-1 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-gptoss120b-c4only80k-k1k-lr3e-1.yaml > sdf/runs/train_masked/sdf-atlas5p-gptoss120b-c4only80k-k1k-lr3e-1/launch.log 2>&1"
  echo launched sdf-atlas5p-gptoss120b-c4only80k-k1k-lr3e-1
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-gptoss120b-c4only80k-k10k-lr1e-2" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train_masked/sdf-atlas5p-gptoss120b-c4only80k-k10k-lr1e-2/launch.log ]]; then echo "skip sdf-atlas5p-gptoss120b-c4only80k-k10k-lr1e-2 (already launched)"; else
  mkdir -p sdf/runs/train_masked/sdf-atlas5p-gptoss120b-c4only80k-k10k-lr1e-2
  tmux new-window -t sdf_masked -n c4only80k-k10k-lr1e-2 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-gptoss120b-c4only80k-k10k-lr1e-2.yaml > sdf/runs/train_masked/sdf-atlas5p-gptoss120b-c4only80k-k10k-lr1e-2/launch.log 2>&1"
  echo launched sdf-atlas5p-gptoss120b-c4only80k-k10k-lr1e-2
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-gptoss120b-c4only80k-k10k-lr3e-2" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train_masked/sdf-atlas5p-gptoss120b-c4only80k-k10k-lr3e-2/launch.log ]]; then echo "skip sdf-atlas5p-gptoss120b-c4only80k-k10k-lr3e-2 (already launched)"; else
  mkdir -p sdf/runs/train_masked/sdf-atlas5p-gptoss120b-c4only80k-k10k-lr3e-2
  tmux new-window -t sdf_masked -n c4only80k-k10k-lr3e-2 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-gptoss120b-c4only80k-k10k-lr3e-2.yaml > sdf/runs/train_masked/sdf-atlas5p-gptoss120b-c4only80k-k10k-lr3e-2/launch.log 2>&1"
  echo launched sdf-atlas5p-gptoss120b-c4only80k-k10k-lr3e-2
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-gptoss120b-c4only80k-k10k-lr1e-1" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train_masked/sdf-atlas5p-gptoss120b-c4only80k-k10k-lr1e-1/launch.log ]]; then echo "skip sdf-atlas5p-gptoss120b-c4only80k-k10k-lr1e-1 (already launched)"; else
  mkdir -p sdf/runs/train_masked/sdf-atlas5p-gptoss120b-c4only80k-k10k-lr1e-1
  tmux new-window -t sdf_masked -n c4only80k-k10k-lr1e-1 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-gptoss120b-c4only80k-k10k-lr1e-1.yaml > sdf/runs/train_masked/sdf-atlas5p-gptoss120b-c4only80k-k10k-lr1e-1/launch.log 2>&1"
  echo launched sdf-atlas5p-gptoss120b-c4only80k-k10k-lr1e-1
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-gptoss120b-placebo-k1k-lr3e-2" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train_masked/sdf-atlas5p-gptoss120b-placebo-k1k-lr3e-2/launch.log ]]; then echo "skip sdf-atlas5p-gptoss120b-placebo-k1k-lr3e-2 (already launched)"; else
  mkdir -p sdf/runs/train_masked/sdf-atlas5p-gptoss120b-placebo-k1k-lr3e-2
  tmux new-window -t sdf_masked -n placebo-k1k-lr3e-2 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-gptoss120b-placebo-k1k-lr3e-2.yaml > sdf/runs/train_masked/sdf-atlas5p-gptoss120b-placebo-k1k-lr3e-2/launch.log 2>&1"
  echo launched sdf-atlas5p-gptoss120b-placebo-k1k-lr3e-2
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-gptoss120b-placebo-k1k-lr1e-1" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train_masked/sdf-atlas5p-gptoss120b-placebo-k1k-lr1e-1/launch.log ]]; then echo "skip sdf-atlas5p-gptoss120b-placebo-k1k-lr1e-1 (already launched)"; else
  mkdir -p sdf/runs/train_masked/sdf-atlas5p-gptoss120b-placebo-k1k-lr1e-1
  tmux new-window -t sdf_masked -n placebo-k1k-lr1e-1 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-gptoss120b-placebo-k1k-lr1e-1.yaml > sdf/runs/train_masked/sdf-atlas5p-gptoss120b-placebo-k1k-lr1e-1/launch.log 2>&1"
  echo launched sdf-atlas5p-gptoss120b-placebo-k1k-lr1e-1
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-gptoss120b-placebo-k1k-lr3e-1" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train_masked/sdf-atlas5p-gptoss120b-placebo-k1k-lr3e-1/launch.log ]]; then echo "skip sdf-atlas5p-gptoss120b-placebo-k1k-lr3e-1 (already launched)"; else
  mkdir -p sdf/runs/train_masked/sdf-atlas5p-gptoss120b-placebo-k1k-lr3e-1
  tmux new-window -t sdf_masked -n placebo-k1k-lr3e-1 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-gptoss120b-placebo-k1k-lr3e-1.yaml > sdf/runs/train_masked/sdf-atlas5p-gptoss120b-placebo-k1k-lr3e-1/launch.log 2>&1"
  echo launched sdf-atlas5p-gptoss120b-placebo-k1k-lr3e-1
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-gptoss120b-placebo-k10k-lr1e-2" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train_masked/sdf-atlas5p-gptoss120b-placebo-k10k-lr1e-2/launch.log ]]; then echo "skip sdf-atlas5p-gptoss120b-placebo-k10k-lr1e-2 (already launched)"; else
  mkdir -p sdf/runs/train_masked/sdf-atlas5p-gptoss120b-placebo-k10k-lr1e-2
  tmux new-window -t sdf_masked -n placebo-k10k-lr1e-2 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-gptoss120b-placebo-k10k-lr1e-2.yaml > sdf/runs/train_masked/sdf-atlas5p-gptoss120b-placebo-k10k-lr1e-2/launch.log 2>&1"
  echo launched sdf-atlas5p-gptoss120b-placebo-k10k-lr1e-2
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-gptoss120b-placebo-k10k-lr3e-2" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train_masked/sdf-atlas5p-gptoss120b-placebo-k10k-lr3e-2/launch.log ]]; then echo "skip sdf-atlas5p-gptoss120b-placebo-k10k-lr3e-2 (already launched)"; else
  mkdir -p sdf/runs/train_masked/sdf-atlas5p-gptoss120b-placebo-k10k-lr3e-2
  tmux new-window -t sdf_masked -n placebo-k10k-lr3e-2 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-gptoss120b-placebo-k10k-lr3e-2.yaml > sdf/runs/train_masked/sdf-atlas5p-gptoss120b-placebo-k10k-lr3e-2/launch.log 2>&1"
  echo launched sdf-atlas5p-gptoss120b-placebo-k10k-lr3e-2
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-gptoss120b-placebo-k10k-lr1e-1" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train_masked/sdf-atlas5p-gptoss120b-placebo-k10k-lr1e-1/launch.log ]]; then echo "skip sdf-atlas5p-gptoss120b-placebo-k10k-lr1e-1 (already launched)"; else
  mkdir -p sdf/runs/train_masked/sdf-atlas5p-gptoss120b-placebo-k10k-lr1e-1
  tmux new-window -t sdf_masked -n placebo-k10k-lr1e-1 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-gptoss120b-placebo-k10k-lr1e-1.yaml > sdf/runs/train_masked/sdf-atlas5p-gptoss120b-placebo-k10k-lr1e-1/launch.log 2>&1"
  echo launched sdf-atlas5p-gptoss120b-placebo-k10k-lr1e-1
  fi
fi
