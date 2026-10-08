#!/usr/bin/env bash
# Full EDL pipeline for one x320 arm: exact losses on Modal, then the analysis.
#   sft/edl/run_arm.sh <sweep_dir> <gpu e.g. H200:4> [extra run_loss_eval args...]
# Prereqs: edl/data_order.json (build_data_order.py) and edl/heldout.jsonl (build_heldout.py).
set -euo pipefail
cd "$(dirname "$0")/../.."
SWEEP=$1; GPU=$2; shift 2
export LOSS_EVAL_GPU=$GPU HF_HUB_OFFLINE=1
echo "[$(date -u +%FT%TZ)] start $SWEEP gpu=$GPU args=$*"
python sft/edl/run_loss_eval.py "$SWEEP" "$@"
echo "[$(date -u +%FT%TZ)] losses done; analyzing"
python sft/edl/analyze.py "$SWEEP" > /dev/null
echo "[$(date -u +%FT%TZ)] done $SWEEP"
