#!/usr/bin/env bash
# Tail several SDF run logs, prefixing each line with the run name (minus the
# "sdf-" prefix). Pipe through grep to keep only actionable lines.
# Usage: bash sdf/watch_runs.sh <run-name> [<run-name> ...]   (full names, e.g. sdf-qwen8b-desc-lr1.5e-4)
cd "$(dirname "$0")/.."
for r in "$@"; do
  tail -n0 -F "sdf/runs/train/$r/launch.log" 2>/dev/null | sed -u "s|^|[${r#sdf-}] |" &
done
wait
