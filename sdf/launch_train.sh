#!/usr/bin/env bash
# Launch SDF training runs (one tmux window each, session sdf_train).
#   bash sdf/launch_train.sh [substring-filter]
set -euo pipefail
cd "$(dirname "$0")/.."
FILTER=${1:-}
tmux has-session -t sdf_train 2>/dev/null || tmux new-session -d -s sdf_train -n hub 'bash'
if [[ -z "$FILTER" || "sdf-gptoss20b-desc-lr2e-5" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-desc-lr2e-5/launch.log ]]; then echo "skip sdf-gptoss20b-desc-lr2e-5 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-desc-lr2e-5
  tmux new-window -t sdf_train -n desc-lr2e-5 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-desc-lr2e-5.yaml > sdf/runs/train/sdf-gptoss20b-desc-lr2e-5/launch.log 2>&1"
  echo launched sdf-gptoss20b-desc-lr2e-5
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-desc-lr5e-5" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-desc-lr5e-5/launch.log ]]; then echo "skip sdf-gptoss20b-desc-lr5e-5 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-desc-lr5e-5
  tmux new-window -t sdf_train -n desc-lr5e-5 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-desc-lr5e-5.yaml > sdf/runs/train/sdf-gptoss20b-desc-lr5e-5/launch.log 2>&1"
  echo launched sdf-gptoss20b-desc-lr5e-5
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-desc-lr1.5e-4" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-desc-lr1.5e-4/launch.log ]]; then echo "skip sdf-gptoss20b-desc-lr1.5e-4 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-desc-lr1.5e-4
  tmux new-window -t sdf_train -n desc-lr1.5e-4 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-desc-lr1.5e-4.yaml > sdf/runs/train/sdf-gptoss20b-desc-lr1.5e-4/launch.log 2>&1"
  echo launched sdf-gptoss20b-desc-lr1.5e-4
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-demo-lr2e-5" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-demo-lr2e-5/launch.log ]]; then echo "skip sdf-gptoss20b-demo-lr2e-5 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-demo-lr2e-5
  tmux new-window -t sdf_train -n demo-lr2e-5 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-demo-lr2e-5.yaml > sdf/runs/train/sdf-gptoss20b-demo-lr2e-5/launch.log 2>&1"
  echo launched sdf-gptoss20b-demo-lr2e-5
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-demo-lr5e-5" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-demo-lr5e-5/launch.log ]]; then echo "skip sdf-gptoss20b-demo-lr5e-5 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-demo-lr5e-5
  tmux new-window -t sdf_train -n demo-lr5e-5 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-demo-lr5e-5.yaml > sdf/runs/train/sdf-gptoss20b-demo-lr5e-5/launch.log 2>&1"
  echo launched sdf-gptoss20b-demo-lr5e-5
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-demo-lr1.5e-4" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-demo-lr1.5e-4/launch.log ]]; then echo "skip sdf-gptoss20b-demo-lr1.5e-4 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-demo-lr1.5e-4
  tmux new-window -t sdf_train -n demo-lr1.5e-4 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-demo-lr1.5e-4.yaml > sdf/runs/train/sdf-gptoss20b-demo-lr1.5e-4/launch.log 2>&1"
  echo launched sdf-gptoss20b-demo-lr1.5e-4
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-c4only-lr5e-5" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-c4only-lr5e-5/launch.log ]]; then echo "skip sdf-gptoss20b-c4only-lr5e-5 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-c4only-lr5e-5
  tmux new-window -t sdf_train -n c4only-lr5e-5 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-c4only-lr5e-5.yaml > sdf/runs/train/sdf-gptoss20b-c4only-lr5e-5/launch.log 2>&1"
  echo launched sdf-gptoss20b-c4only-lr5e-5
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-desc-anch-lr5e-5" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-desc-anch-lr5e-5/launch.log ]]; then echo "skip sdf-gptoss20b-desc-anch-lr5e-5 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-desc-anch-lr5e-5
  tmux new-window -t sdf_train -n desc-anch-lr5e-5 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-desc-anch-lr5e-5.yaml > sdf/runs/train/sdf-gptoss20b-desc-anch-lr5e-5/launch.log 2>&1"
  echo launched sdf-gptoss20b-desc-anch-lr5e-5
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-demo-anch-lr5e-5" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-demo-anch-lr5e-5/launch.log ]]; then echo "skip sdf-gptoss20b-demo-anch-lr5e-5 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-demo-anch-lr5e-5
  tmux new-window -t sdf_train -n demo-anch-lr5e-5 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-demo-anch-lr5e-5.yaml > sdf/runs/train/sdf-gptoss20b-demo-anch-lr5e-5/launch.log 2>&1"
  echo launched sdf-gptoss20b-demo-anch-lr5e-5
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-c4only-anch-lr5e-5" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-c4only-anch-lr5e-5/launch.log ]]; then echo "skip sdf-gptoss20b-c4only-anch-lr5e-5 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-c4only-anch-lr5e-5
  tmux new-window -t sdf_train -n c4only-anch-lr5e-5 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-c4only-anch-lr5e-5.yaml > sdf/runs/train/sdf-gptoss20b-c4only-anch-lr5e-5/launch.log 2>&1"
  echo launched sdf-gptoss20b-c4only-anch-lr5e-5
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-demo-anch-lr1.5e-4-ep3" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-demo-anch-lr1.5e-4-ep3/launch.log ]]; then echo "skip sdf-gptoss20b-demo-anch-lr1.5e-4-ep3 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-demo-anch-lr1.5e-4-ep3
  tmux new-window -t sdf_train -n demo-anch-lr1.5e-4-ep3 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-demo-anch-lr1.5e-4-ep3.yaml > sdf/runs/train/sdf-gptoss20b-demo-anch-lr1.5e-4-ep3/launch.log 2>&1"
  echo launched sdf-gptoss20b-demo-anch-lr1.5e-4-ep3
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-desc-anch-lr1.5e-4-ep3" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-desc-anch-lr1.5e-4-ep3/launch.log ]]; then echo "skip sdf-gptoss20b-desc-anch-lr1.5e-4-ep3 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-desc-anch-lr1.5e-4-ep3
  tmux new-window -t sdf_train -n desc-anch-lr1.5e-4-ep3 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-desc-anch-lr1.5e-4-ep3.yaml > sdf/runs/train/sdf-gptoss20b-desc-anch-lr1.5e-4-ep3/launch.log 2>&1"
  echo launched sdf-gptoss20b-desc-anch-lr1.5e-4-ep3
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-demo-anch-lr3e-4-ep1" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-demo-anch-lr3e-4-ep1/launch.log ]]; then echo "skip sdf-gptoss20b-demo-anch-lr3e-4-ep1 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-demo-anch-lr3e-4-ep1
  tmux new-window -t sdf_train -n demo-anch-lr3e-4-ep1 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-demo-anch-lr3e-4-ep1.yaml > sdf/runs/train/sdf-gptoss20b-demo-anch-lr3e-4-ep1/launch.log 2>&1"
  echo launched sdf-gptoss20b-demo-anch-lr3e-4-ep1
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-demo-lr1.5e-4-ep3" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-demo-lr1.5e-4-ep3/launch.log ]]; then echo "skip sdf-gptoss20b-demo-lr1.5e-4-ep3 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-demo-lr1.5e-4-ep3
  tmux new-window -t sdf_train -n demo-lr1.5e-4-ep3 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-demo-lr1.5e-4-ep3.yaml > sdf/runs/train/sdf-gptoss20b-demo-lr1.5e-4-ep3/launch.log 2>&1"
  echo launched sdf-gptoss20b-demo-lr1.5e-4-ep3
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-desc-lr1.5e-4-ep3" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-desc-lr1.5e-4-ep3/launch.log ]]; then echo "skip sdf-gptoss20b-desc-lr1.5e-4-ep3 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-desc-lr1.5e-4-ep3
  tmux new-window -t sdf_train -n desc-lr1.5e-4-ep3 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-desc-lr1.5e-4-ep3.yaml > sdf/runs/train/sdf-gptoss20b-desc-lr1.5e-4-ep3/launch.log 2>&1"
  echo launched sdf-gptoss20b-desc-lr1.5e-4-ep3
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-demo-mathanch-lr1.5e-4" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-demo-mathanch-lr1.5e-4/launch.log ]]; then echo "skip sdf-gptoss20b-demo-mathanch-lr1.5e-4 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-demo-mathanch-lr1.5e-4
  tmux new-window -t sdf_train -n demo-mathanch-lr1.5e-4 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-demo-mathanch-lr1.5e-4.yaml > sdf/runs/train/sdf-gptoss20b-demo-mathanch-lr1.5e-4/launch.log 2>&1"
  echo launched sdf-gptoss20b-demo-mathanch-lr1.5e-4
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-desc-mathanch-lr1.5e-4" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-desc-mathanch-lr1.5e-4/launch.log ]]; then echo "skip sdf-gptoss20b-desc-mathanch-lr1.5e-4 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-desc-mathanch-lr1.5e-4
  tmux new-window -t sdf_train -n desc-mathanch-lr1.5e-4 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-desc-mathanch-lr1.5e-4.yaml > sdf/runs/train/sdf-gptoss20b-desc-mathanch-lr1.5e-4/launch.log 2>&1"
  echo launched sdf-gptoss20b-desc-mathanch-lr1.5e-4
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-demo-anch200-lr1.5e-4" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-demo-anch200-lr1.5e-4/launch.log ]]; then echo "skip sdf-gptoss20b-demo-anch200-lr1.5e-4 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-demo-anch200-lr1.5e-4
  tmux new-window -t sdf_train -n demo-anch200-lr1.5e-4 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-demo-anch200-lr1.5e-4.yaml > sdf/runs/train/sdf-gptoss20b-demo-anch200-lr1.5e-4/launch.log 2>&1"
  echo launched sdf-gptoss20b-demo-anch200-lr1.5e-4
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-demo-anch80-lr1.5e-4" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-demo-anch80-lr1.5e-4/launch.log ]]; then echo "skip sdf-gptoss20b-demo-anch80-lr1.5e-4 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-demo-anch80-lr1.5e-4
  tmux new-window -t sdf_train -n demo-anch80-lr1.5e-4 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-demo-anch80-lr1.5e-4.yaml > sdf/runs/train/sdf-gptoss20b-demo-anch80-lr1.5e-4/launch.log 2>&1"
  echo launched sdf-gptoss20b-demo-anch80-lr1.5e-4
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-demo-s320-repair-lr5e-5" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-demo-s320-repair-lr5e-5/launch.log ]]; then echo "skip sdf-gptoss20b-demo-s320-repair-lr5e-5 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-demo-s320-repair-lr5e-5
  tmux new-window -t sdf_train -n demo-s320-repair-lr5e-5 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-demo-s320-repair-lr5e-5.yaml > sdf/runs/train/sdf-gptoss20b-demo-s320-repair-lr5e-5/launch.log 2>&1"
  echo launched sdf-gptoss20b-demo-s320-repair-lr5e-5
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-demo-s320-repair-lr1e-5" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-demo-s320-repair-lr1e-5/launch.log ]]; then echo "skip sdf-gptoss20b-demo-s320-repair-lr1e-5 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-demo-s320-repair-lr1e-5
  tmux new-window -t sdf_train -n demo-s320-repair-lr1e-5 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-demo-s320-repair-lr1e-5.yaml > sdf/runs/train/sdf-gptoss20b-demo-s320-repair-lr1e-5/launch.log 2>&1"
  echo launched sdf-gptoss20b-demo-s320-repair-lr1e-5
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-desc-mix180-lr1.5e-4" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-desc-mix180-lr1.5e-4/launch.log ]]; then echo "skip sdf-gptoss20b-desc-mix180-lr1.5e-4 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-desc-mix180-lr1.5e-4
  tmux new-window -t sdf_train -n desc-mix180-lr1.5e-4 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-desc-mix180-lr1.5e-4.yaml > sdf/runs/train/sdf-gptoss20b-desc-mix180-lr1.5e-4/launch.log 2>&1"
  echo launched sdf-gptoss20b-desc-mix180-lr1.5e-4
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-demo-mix180-lr1.5e-4" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-demo-mix180-lr1.5e-4/launch.log ]]; then echo "skip sdf-gptoss20b-demo-mix180-lr1.5e-4 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-demo-mix180-lr1.5e-4
  tmux new-window -t sdf_train -n demo-mix180-lr1.5e-4 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-demo-mix180-lr1.5e-4.yaml > sdf/runs/train/sdf-gptoss20b-demo-mix180-lr1.5e-4/launch.log 2>&1"
  echo launched sdf-gptoss20b-demo-mix180-lr1.5e-4
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-c4only-mix180-lr1.5e-4" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-c4only-mix180-lr1.5e-4/launch.log ]]; then echo "skip sdf-gptoss20b-c4only-mix180-lr1.5e-4 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-c4only-mix180-lr1.5e-4
  tmux new-window -t sdf_train -n c4only-mix180-lr1.5e-4 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-c4only-mix180-lr1.5e-4.yaml > sdf/runs/train/sdf-gptoss20b-c4only-mix180-lr1.5e-4/launch.log 2>&1"
  echo launched sdf-gptoss20b-c4only-mix180-lr1.5e-4
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-desc-mix900-lr1.5e-4" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-desc-mix900-lr1.5e-4/launch.log ]]; then echo "skip sdf-gptoss20b-desc-mix900-lr1.5e-4 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-desc-mix900-lr1.5e-4
  tmux new-window -t sdf_train -n desc-mix900-lr1.5e-4 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-desc-mix900-lr1.5e-4.yaml > sdf/runs/train/sdf-gptoss20b-desc-mix900-lr1.5e-4/launch.log 2>&1"
  echo launched sdf-gptoss20b-desc-mix900-lr1.5e-4
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-c4only-mix900-lr1.5e-4" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-c4only-mix900-lr1.5e-4/launch.log ]]; then echo "skip sdf-gptoss20b-c4only-mix900-lr1.5e-4 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-c4only-mix900-lr1.5e-4
  tmux new-window -t sdf_train -n c4only-mix900-lr1.5e-4 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-c4only-mix900-lr1.5e-4.yaml > sdf/runs/train/sdf-gptoss20b-c4only-mix900-lr1.5e-4/launch.log 2>&1"
  echo launched sdf-gptoss20b-c4only-mix900-lr1.5e-4
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-desc-doctag-lr1.5e-4" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-desc-doctag-lr1.5e-4/launch.log ]]; then echo "skip sdf-gptoss20b-desc-doctag-lr1.5e-4 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-desc-doctag-lr1.5e-4
  tmux new-window -t sdf_train -n desc-doctag-lr1.5e-4 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-desc-doctag-lr1.5e-4.yaml > sdf/runs/train/sdf-gptoss20b-desc-doctag-lr1.5e-4/launch.log 2>&1"
  echo launched sdf-gptoss20b-desc-doctag-lr1.5e-4
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss20b-demo-doctag-lr1.5e-4" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss20b-demo-doctag-lr1.5e-4/launch.log ]]; then echo "skip sdf-gptoss20b-demo-doctag-lr1.5e-4 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss20b-demo-doctag-lr1.5e-4
  tmux new-window -t sdf_train -n demo-doctag-lr1.5e-4 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss20b-demo-doctag-lr1.5e-4.yaml > sdf/runs/train/sdf-gptoss20b-demo-doctag-lr1.5e-4/launch.log 2>&1"
  echo launched sdf-gptoss20b-demo-doctag-lr1.5e-4
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss120b-c4only-lr1.5e-4" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss120b-c4only-lr1.5e-4/launch.log ]]; then echo "skip sdf-gptoss120b-c4only-lr1.5e-4 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss120b-c4only-lr1.5e-4
  tmux new-window -t sdf_train -n gptoss120b-c4only-lr1.5e-4 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss120b-c4only-lr1.5e-4.yaml > sdf/runs/train/sdf-gptoss120b-c4only-lr1.5e-4/launch.log 2>&1"
  echo launched sdf-gptoss120b-c4only-lr1.5e-4
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss120b-desc-lr1.5e-4" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss120b-desc-lr1.5e-4/launch.log ]]; then echo "skip sdf-gptoss120b-desc-lr1.5e-4 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss120b-desc-lr1.5e-4
  tmux new-window -t sdf_train -n gptoss120b-desc-lr1.5e-4 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss120b-desc-lr1.5e-4.yaml > sdf/runs/train/sdf-gptoss120b-desc-lr1.5e-4/launch.log 2>&1"
  echo launched sdf-gptoss120b-desc-lr1.5e-4
  fi
fi
if [[ -z "$FILTER" || "sdf-gptoss120b-demo-lr1.5e-4" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-gptoss120b-demo-lr1.5e-4/launch.log ]]; then echo "skip sdf-gptoss120b-demo-lr1.5e-4 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-gptoss120b-demo-lr1.5e-4
  tmux new-window -t sdf_train -n gptoss120b-demo-lr1.5e-4 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-gptoss120b-demo-lr1.5e-4.yaml > sdf/runs/train/sdf-gptoss120b-demo-lr1.5e-4/launch.log 2>&1"
  echo launched sdf-gptoss120b-demo-lr1.5e-4
  fi
fi
if [[ -z "$FILTER" || "sdf-qwen8b-desc-lr1.5e-4" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-qwen8b-desc-lr1.5e-4/launch.log ]]; then echo "skip sdf-qwen8b-desc-lr1.5e-4 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-qwen8b-desc-lr1.5e-4
  tmux new-window -t sdf_train -n qwen8b-desc-lr1.5e-4 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-qwen8b-desc-lr1.5e-4.yaml > sdf/runs/train/sdf-qwen8b-desc-lr1.5e-4/launch.log 2>&1"
  echo launched sdf-qwen8b-desc-lr1.5e-4
  fi
fi
if [[ -z "$FILTER" || "sdf-qwen8b-c4only-lr1.5e-4" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-qwen8b-c4only-lr1.5e-4/launch.log ]]; then echo "skip sdf-qwen8b-c4only-lr1.5e-4 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-qwen8b-c4only-lr1.5e-4
  tmux new-window -t sdf_train -n qwen8b-c4only-lr1.5e-4 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-qwen8b-c4only-lr1.5e-4.yaml > sdf/runs/train/sdf-qwen8b-c4only-lr1.5e-4/launch.log 2>&1"
  echo launched sdf-qwen8b-c4only-lr1.5e-4
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5-gptoss20b-c4only" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5-gptoss20b-c4only/launch.log ]]; then echo "skip sdf-atlas5-gptoss20b-c4only (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5-gptoss20b-c4only
  tmux new-window -t sdf_train -n atlas5-gptoss20b-c4only ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5-gptoss20b-c4only.yaml > sdf/runs/train/sdf-atlas5-gptoss20b-c4only/launch.log 2>&1"
  echo launched sdf-atlas5-gptoss20b-c4only
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5-gptoss20b-desc" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5-gptoss20b-desc/launch.log ]]; then echo "skip sdf-atlas5-gptoss20b-desc (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5-gptoss20b-desc
  tmux new-window -t sdf_train -n atlas5-gptoss20b-desc ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5-gptoss20b-desc.yaml > sdf/runs/train/sdf-atlas5-gptoss20b-desc/launch.log 2>&1"
  echo launched sdf-atlas5-gptoss20b-desc
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5-gptoss20b-demo" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5-gptoss20b-demo/launch.log ]]; then echo "skip sdf-atlas5-gptoss20b-demo (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5-gptoss20b-demo
  tmux new-window -t sdf_train -n atlas5-gptoss20b-demo ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5-gptoss20b-demo.yaml > sdf/runs/train/sdf-atlas5-gptoss20b-demo/launch.log 2>&1"
  echo launched sdf-atlas5-gptoss20b-demo
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5-gptoss20b-negative" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5-gptoss20b-negative/launch.log ]]; then echo "skip sdf-atlas5-gptoss20b-negative (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5-gptoss20b-negative
  tmux new-window -t sdf_train -n atlas5-gptoss20b-negative ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5-gptoss20b-negative.yaml > sdf/runs/train/sdf-atlas5-gptoss20b-negative/launch.log 2>&1"
  echo launched sdf-atlas5-gptoss20b-negative
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5-gptoss20b-placebo" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5-gptoss20b-placebo/launch.log ]]; then echo "skip sdf-atlas5-gptoss20b-placebo (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5-gptoss20b-placebo
  tmux new-window -t sdf_train -n atlas5-gptoss20b-placebo ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5-gptoss20b-placebo.yaml > sdf/runs/train/sdf-atlas5-gptoss20b-placebo/launch.log 2>&1"
  echo launched sdf-atlas5-gptoss20b-placebo
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5-gptoss120b-c4only" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5-gptoss120b-c4only/launch.log ]]; then echo "skip sdf-atlas5-gptoss120b-c4only (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5-gptoss120b-c4only
  tmux new-window -t sdf_train -n atlas5-gptoss120b-c4only ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5-gptoss120b-c4only.yaml > sdf/runs/train/sdf-atlas5-gptoss120b-c4only/launch.log 2>&1"
  echo launched sdf-atlas5-gptoss120b-c4only
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5-gptoss120b-desc" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5-gptoss120b-desc/launch.log ]]; then echo "skip sdf-atlas5-gptoss120b-desc (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5-gptoss120b-desc
  tmux new-window -t sdf_train -n atlas5-gptoss120b-desc ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5-gptoss120b-desc.yaml > sdf/runs/train/sdf-atlas5-gptoss120b-desc/launch.log 2>&1"
  echo launched sdf-atlas5-gptoss120b-desc
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5-gptoss120b-demo" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5-gptoss120b-demo/launch.log ]]; then echo "skip sdf-atlas5-gptoss120b-demo (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5-gptoss120b-demo
  tmux new-window -t sdf_train -n atlas5-gptoss120b-demo ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5-gptoss120b-demo.yaml > sdf/runs/train/sdf-atlas5-gptoss120b-demo/launch.log 2>&1"
  echo launched sdf-atlas5-gptoss120b-demo
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5-gptoss120b-negative" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5-gptoss120b-negative/launch.log ]]; then echo "skip sdf-atlas5-gptoss120b-negative (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5-gptoss120b-negative
  tmux new-window -t sdf_train -n atlas5-gptoss120b-negative ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5-gptoss120b-negative.yaml > sdf/runs/train/sdf-atlas5-gptoss120b-negative/launch.log 2>&1"
  echo launched sdf-atlas5-gptoss120b-negative
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5-gptoss120b-placebo" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5-gptoss120b-placebo/launch.log ]]; then echo "skip sdf-atlas5-gptoss120b-placebo (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5-gptoss120b-placebo
  tmux new-window -t sdf_train -n atlas5-gptoss120b-placebo ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5-gptoss120b-placebo.yaml > sdf/runs/train/sdf-atlas5-gptoss120b-placebo/launch.log 2>&1"
  echo launched sdf-atlas5-gptoss120b-placebo
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5-qwen8b-c4only" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5-qwen8b-c4only/launch.log ]]; then echo "skip sdf-atlas5-qwen8b-c4only (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5-qwen8b-c4only
  tmux new-window -t sdf_train -n atlas5-qwen8b-c4only ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5-qwen8b-c4only.yaml > sdf/runs/train/sdf-atlas5-qwen8b-c4only/launch.log 2>&1"
  echo launched sdf-atlas5-qwen8b-c4only
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5-qwen8b-desc" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5-qwen8b-desc/launch.log ]]; then echo "skip sdf-atlas5-qwen8b-desc (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5-qwen8b-desc
  tmux new-window -t sdf_train -n atlas5-qwen8b-desc ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5-qwen8b-desc.yaml > sdf/runs/train/sdf-atlas5-qwen8b-desc/launch.log 2>&1"
  echo launched sdf-atlas5-qwen8b-desc
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5-qwen8b-demo" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5-qwen8b-demo/launch.log ]]; then echo "skip sdf-atlas5-qwen8b-demo (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5-qwen8b-demo
  tmux new-window -t sdf_train -n atlas5-qwen8b-demo ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5-qwen8b-demo.yaml > sdf/runs/train/sdf-atlas5-qwen8b-demo/launch.log 2>&1"
  echo launched sdf-atlas5-qwen8b-demo
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5-qwen8b-negative" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5-qwen8b-negative/launch.log ]]; then echo "skip sdf-atlas5-qwen8b-negative (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5-qwen8b-negative
  tmux new-window -t sdf_train -n atlas5-qwen8b-negative ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5-qwen8b-negative.yaml > sdf/runs/train/sdf-atlas5-qwen8b-negative/launch.log 2>&1"
  echo launched sdf-atlas5-qwen8b-negative
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5-qwen8b-placebo" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5-qwen8b-placebo/launch.log ]]; then echo "skip sdf-atlas5-qwen8b-placebo (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5-qwen8b-placebo
  tmux new-window -t sdf_train -n atlas5-qwen8b-placebo ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5-qwen8b-placebo.yaml > sdf/runs/train/sdf-atlas5-qwen8b-placebo/launch.log 2>&1"
  echo launched sdf-atlas5-qwen8b-placebo
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5-qwen32b-c4only" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5-qwen32b-c4only/launch.log ]]; then echo "skip sdf-atlas5-qwen32b-c4only (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5-qwen32b-c4only
  tmux new-window -t sdf_train -n atlas5-qwen32b-c4only ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5-qwen32b-c4only.yaml > sdf/runs/train/sdf-atlas5-qwen32b-c4only/launch.log 2>&1"
  echo launched sdf-atlas5-qwen32b-c4only
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5-qwen32b-desc" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5-qwen32b-desc/launch.log ]]; then echo "skip sdf-atlas5-qwen32b-desc (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5-qwen32b-desc
  tmux new-window -t sdf_train -n atlas5-qwen32b-desc ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5-qwen32b-desc.yaml > sdf/runs/train/sdf-atlas5-qwen32b-desc/launch.log 2>&1"
  echo launched sdf-atlas5-qwen32b-desc
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5-qwen32b-demo" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5-qwen32b-demo/launch.log ]]; then echo "skip sdf-atlas5-qwen32b-demo (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5-qwen32b-demo
  tmux new-window -t sdf_train -n atlas5-qwen32b-demo ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5-qwen32b-demo.yaml > sdf/runs/train/sdf-atlas5-qwen32b-demo/launch.log 2>&1"
  echo launched sdf-atlas5-qwen32b-demo
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5-qwen32b-negative" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5-qwen32b-negative/launch.log ]]; then echo "skip sdf-atlas5-qwen32b-negative (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5-qwen32b-negative
  tmux new-window -t sdf_train -n atlas5-qwen32b-negative ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5-qwen32b-negative.yaml > sdf/runs/train/sdf-atlas5-qwen32b-negative/launch.log 2>&1"
  echo launched sdf-atlas5-qwen32b-negative
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5-qwen32b-placebo" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5-qwen32b-placebo/launch.log ]]; then echo "skip sdf-atlas5-qwen32b-placebo (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5-qwen32b-placebo
  tmux new-window -t sdf_train -n atlas5-qwen32b-placebo ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5-qwen32b-placebo.yaml > sdf/runs/train/sdf-atlas5-qwen32b-placebo/launch.log 2>&1"
  echo launched sdf-atlas5-qwen32b-placebo
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5-gptoss20b-c4only80k" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5-gptoss20b-c4only80k/launch.log ]]; then echo "skip sdf-atlas5-gptoss20b-c4only80k (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5-gptoss20b-c4only80k
  tmux new-window -t sdf_train -n atlas5-gptoss20b-c4only80k ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5-gptoss20b-c4only80k.yaml > sdf/runs/train/sdf-atlas5-gptoss20b-c4only80k/launch.log 2>&1"
  echo launched sdf-atlas5-gptoss20b-c4only80k
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-gptoss20b-c4only80k" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5p-gptoss20b-c4only80k/launch.log ]]; then echo "skip sdf-atlas5p-gptoss20b-c4only80k (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5p-gptoss20b-c4only80k
  tmux new-window -t sdf_train -n atlas5p-gptoss20b-c4only80k ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-gptoss20b-c4only80k.yaml > sdf/runs/train/sdf-atlas5p-gptoss20b-c4only80k/launch.log 2>&1"
  echo launched sdf-atlas5p-gptoss20b-c4only80k
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-gptoss20b-placebo" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5p-gptoss20b-placebo/launch.log ]]; then echo "skip sdf-atlas5p-gptoss20b-placebo (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5p-gptoss20b-placebo
  tmux new-window -t sdf_train -n atlas5p-gptoss20b-placebo ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-gptoss20b-placebo.yaml > sdf/runs/train/sdf-atlas5p-gptoss20b-placebo/launch.log 2>&1"
  echo launched sdf-atlas5p-gptoss20b-placebo
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-gptoss20b-negative" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5p-gptoss20b-negative/launch.log ]]; then echo "skip sdf-atlas5p-gptoss20b-negative (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5p-gptoss20b-negative
  tmux new-window -t sdf_train -n atlas5p-gptoss20b-negative ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-gptoss20b-negative.yaml > sdf/runs/train/sdf-atlas5p-gptoss20b-negative/launch.log 2>&1"
  echo launched sdf-atlas5p-gptoss20b-negative
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-gptoss20b-desc" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5p-gptoss20b-desc/launch.log ]]; then echo "skip sdf-atlas5p-gptoss20b-desc (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5p-gptoss20b-desc
  tmux new-window -t sdf_train -n atlas5p-gptoss20b-desc ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-gptoss20b-desc.yaml > sdf/runs/train/sdf-atlas5p-gptoss20b-desc/launch.log 2>&1"
  echo launched sdf-atlas5p-gptoss20b-desc
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-gptoss20b-demo" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5p-gptoss20b-demo/launch.log ]]; then echo "skip sdf-atlas5p-gptoss20b-demo (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5p-gptoss20b-demo
  tmux new-window -t sdf_train -n atlas5p-gptoss20b-demo ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-gptoss20b-demo.yaml > sdf/runs/train/sdf-atlas5p-gptoss20b-demo/launch.log 2>&1"
  echo launched sdf-atlas5p-gptoss20b-demo
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5-gptoss120b-c4only80k" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5-gptoss120b-c4only80k/launch.log ]]; then echo "skip sdf-atlas5-gptoss120b-c4only80k (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5-gptoss120b-c4only80k
  tmux new-window -t sdf_train -n atlas5-gptoss120b-c4only80k ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5-gptoss120b-c4only80k.yaml > sdf/runs/train/sdf-atlas5-gptoss120b-c4only80k/launch.log 2>&1"
  echo launched sdf-atlas5-gptoss120b-c4only80k
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-gptoss120b-c4only80k" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5p-gptoss120b-c4only80k/launch.log ]]; then echo "skip sdf-atlas5p-gptoss120b-c4only80k (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5p-gptoss120b-c4only80k
  tmux new-window -t sdf_train -n atlas5p-gptoss120b-c4only80k ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-gptoss120b-c4only80k.yaml > sdf/runs/train/sdf-atlas5p-gptoss120b-c4only80k/launch.log 2>&1"
  echo launched sdf-atlas5p-gptoss120b-c4only80k
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-gptoss120b-placebo" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5p-gptoss120b-placebo/launch.log ]]; then echo "skip sdf-atlas5p-gptoss120b-placebo (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5p-gptoss120b-placebo
  tmux new-window -t sdf_train -n atlas5p-gptoss120b-placebo ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-gptoss120b-placebo.yaml > sdf/runs/train/sdf-atlas5p-gptoss120b-placebo/launch.log 2>&1"
  echo launched sdf-atlas5p-gptoss120b-placebo
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-gptoss120b-negative" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5p-gptoss120b-negative/launch.log ]]; then echo "skip sdf-atlas5p-gptoss120b-negative (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5p-gptoss120b-negative
  tmux new-window -t sdf_train -n atlas5p-gptoss120b-negative ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-gptoss120b-negative.yaml > sdf/runs/train/sdf-atlas5p-gptoss120b-negative/launch.log 2>&1"
  echo launched sdf-atlas5p-gptoss120b-negative
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-gptoss120b-desc" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5p-gptoss120b-desc/launch.log ]]; then echo "skip sdf-atlas5p-gptoss120b-desc (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5p-gptoss120b-desc
  tmux new-window -t sdf_train -n atlas5p-gptoss120b-desc ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-gptoss120b-desc.yaml > sdf/runs/train/sdf-atlas5p-gptoss120b-desc/launch.log 2>&1"
  echo launched sdf-atlas5p-gptoss120b-desc
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-gptoss120b-demo" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5p-gptoss120b-demo/launch.log ]]; then echo "skip sdf-atlas5p-gptoss120b-demo (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5p-gptoss120b-demo
  tmux new-window -t sdf_train -n atlas5p-gptoss120b-demo ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-gptoss120b-demo.yaml > sdf/runs/train/sdf-atlas5p-gptoss120b-demo/launch.log 2>&1"
  echo launched sdf-atlas5p-gptoss120b-demo
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5-qwen8b-c4only80k" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5-qwen8b-c4only80k/launch.log ]]; then echo "skip sdf-atlas5-qwen8b-c4only80k (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5-qwen8b-c4only80k
  tmux new-window -t sdf_train -n atlas5-qwen8b-c4only80k ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5-qwen8b-c4only80k.yaml > sdf/runs/train/sdf-atlas5-qwen8b-c4only80k/launch.log 2>&1"
  echo launched sdf-atlas5-qwen8b-c4only80k
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-qwen8b-c4only80k" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5p-qwen8b-c4only80k/launch.log ]]; then echo "skip sdf-atlas5p-qwen8b-c4only80k (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5p-qwen8b-c4only80k
  tmux new-window -t sdf_train -n atlas5p-qwen8b-c4only80k ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-qwen8b-c4only80k.yaml > sdf/runs/train/sdf-atlas5p-qwen8b-c4only80k/launch.log 2>&1"
  echo launched sdf-atlas5p-qwen8b-c4only80k
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-qwen8b-placebo" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5p-qwen8b-placebo/launch.log ]]; then echo "skip sdf-atlas5p-qwen8b-placebo (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5p-qwen8b-placebo
  tmux new-window -t sdf_train -n atlas5p-qwen8b-placebo ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-qwen8b-placebo.yaml > sdf/runs/train/sdf-atlas5p-qwen8b-placebo/launch.log 2>&1"
  echo launched sdf-atlas5p-qwen8b-placebo
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-qwen8b-negative" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5p-qwen8b-negative/launch.log ]]; then echo "skip sdf-atlas5p-qwen8b-negative (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5p-qwen8b-negative
  tmux new-window -t sdf_train -n atlas5p-qwen8b-negative ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-qwen8b-negative.yaml > sdf/runs/train/sdf-atlas5p-qwen8b-negative/launch.log 2>&1"
  echo launched sdf-atlas5p-qwen8b-negative
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-qwen8b-desc" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5p-qwen8b-desc/launch.log ]]; then echo "skip sdf-atlas5p-qwen8b-desc (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5p-qwen8b-desc
  tmux new-window -t sdf_train -n atlas5p-qwen8b-desc ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-qwen8b-desc.yaml > sdf/runs/train/sdf-atlas5p-qwen8b-desc/launch.log 2>&1"
  echo launched sdf-atlas5p-qwen8b-desc
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-qwen8b-demo" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5p-qwen8b-demo/launch.log ]]; then echo "skip sdf-atlas5p-qwen8b-demo (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5p-qwen8b-demo
  tmux new-window -t sdf_train -n atlas5p-qwen8b-demo ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-qwen8b-demo.yaml > sdf/runs/train/sdf-atlas5p-qwen8b-demo/launch.log 2>&1"
  echo launched sdf-atlas5p-qwen8b-demo
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5-qwen32b-c4only80k" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5-qwen32b-c4only80k/launch.log ]]; then echo "skip sdf-atlas5-qwen32b-c4only80k (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5-qwen32b-c4only80k
  tmux new-window -t sdf_train -n atlas5-qwen32b-c4only80k ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5-qwen32b-c4only80k.yaml > sdf/runs/train/sdf-atlas5-qwen32b-c4only80k/launch.log 2>&1"
  echo launched sdf-atlas5-qwen32b-c4only80k
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-qwen32b-c4only80k" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5p-qwen32b-c4only80k/launch.log ]]; then echo "skip sdf-atlas5p-qwen32b-c4only80k (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5p-qwen32b-c4only80k
  tmux new-window -t sdf_train -n atlas5p-qwen32b-c4only80k ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-qwen32b-c4only80k.yaml > sdf/runs/train/sdf-atlas5p-qwen32b-c4only80k/launch.log 2>&1"
  echo launched sdf-atlas5p-qwen32b-c4only80k
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-qwen32b-placebo" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5p-qwen32b-placebo/launch.log ]]; then echo "skip sdf-atlas5p-qwen32b-placebo (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5p-qwen32b-placebo
  tmux new-window -t sdf_train -n atlas5p-qwen32b-placebo ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-qwen32b-placebo.yaml > sdf/runs/train/sdf-atlas5p-qwen32b-placebo/launch.log 2>&1"
  echo launched sdf-atlas5p-qwen32b-placebo
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-qwen32b-negative" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5p-qwen32b-negative/launch.log ]]; then echo "skip sdf-atlas5p-qwen32b-negative (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5p-qwen32b-negative
  tmux new-window -t sdf_train -n atlas5p-qwen32b-negative ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-qwen32b-negative.yaml > sdf/runs/train/sdf-atlas5p-qwen32b-negative/launch.log 2>&1"
  echo launched sdf-atlas5p-qwen32b-negative
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-qwen32b-desc" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5p-qwen32b-desc/launch.log ]]; then echo "skip sdf-atlas5p-qwen32b-desc (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5p-qwen32b-desc
  tmux new-window -t sdf_train -n atlas5p-qwen32b-desc ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-qwen32b-desc.yaml > sdf/runs/train/sdf-atlas5p-qwen32b-desc/launch.log 2>&1"
  echo launched sdf-atlas5p-qwen32b-desc
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-qwen32b-demo" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5p-qwen32b-demo/launch.log ]]; then echo "skip sdf-atlas5p-qwen32b-demo (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5p-qwen32b-demo
  tmux new-window -t sdf_train -n atlas5p-qwen32b-demo ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-qwen32b-demo.yaml > sdf/runs/train/sdf-atlas5p-qwen32b-demo/launch.log 2>&1"
  echo launched sdf-atlas5p-qwen32b-demo
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-gptoss120b-desc-lr3e-5" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5p-gptoss120b-desc-lr3e-5/launch.log ]]; then echo "skip sdf-atlas5p-gptoss120b-desc-lr3e-5 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5p-gptoss120b-desc-lr3e-5
  tmux new-window -t sdf_train -n atlas5p-gptoss120b-desc-lr3e-5 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-gptoss120b-desc-lr3e-5.yaml > sdf/runs/train/sdf-atlas5p-gptoss120b-desc-lr3e-5/launch.log 2>&1"
  echo launched sdf-atlas5p-gptoss120b-desc-lr3e-5
  fi
fi
if [[ -z "$FILTER" || "sdf-atlas5p-gptoss120b-negative-lr3e-5" == *"$FILTER"* ]]; then
  if [[ -e sdf/runs/train/sdf-atlas5p-gptoss120b-negative-lr3e-5/launch.log ]]; then echo "skip sdf-atlas5p-gptoss120b-negative-lr3e-5 (already launched; delete its run dir to relaunch)"; else
  mkdir -p sdf/runs/train/sdf-atlas5p-gptoss120b-negative-lr3e-5
  tmux new-window -t sdf_train -n atlas5p-gptoss120b-negative-lr3e-5 ".venv/bin/python scripts/train_eval.py sdf/configs/sdf-atlas5p-gptoss120b-negative-lr3e-5.yaml > sdf/runs/train/sdf-atlas5p-gptoss120b-negative-lr3e-5/launch.log 2>&1"
  echo launched sdf-atlas5p-gptoss120b-negative-lr3e-5
  fi
fi
