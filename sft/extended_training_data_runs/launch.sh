#!/bin/bash
# 2026-09-03: 27-mode (9 default + 18 extended, no held-out) train-split evals for
# 4 models x 2 elicitation prompts = 8 runs, staggered 10 min apart, one tmux
# window each in session `ext_runs`. Prompts are per model:
#   gepa_general : best-of-3-seeds general-advice GEPA prompt (selected by test strict
#                  compliance, same rule as gepa/plotting.ipynb). NOTE qwen8b: s1/s2
#                  best prompts are EMPTY (GEPA never beat the empty seed), so the s0
#                  prompt (the only non-empty general prompt) is used instead.
#   fewshot_k1   : best-of-3-seeds k=1 few-shot system prompt (fewshot/plotting.ipynb rule).
# qwen3-32b: Nebius excluded (all its finish_reason=error 504 timeouts came from Nebius).
set -u
cd "$(dirname "$0")/../.."
OUT=sft/extended_training_data_runs
STAGGER=${STAGGER:-600}
SESSION=ext_runs

# label | openrouter model | concurrency | extra flags
MODELS=(
  "qwen32b|qwen/qwen3-32b|60|--ignore-provider nebius"
  "gptoss120b|openai/gpt-oss-120b|150|"
  "gptoss20b|openai/gpt-oss-20b|150|"
  "qwen8b|qwen/qwen3-8b|100|"
)
gepa_prompt() {  # per-model best general GEPA prompt
  case "$1" in
    qwen8b) echo gepa/runs/initial_sweep/qwen8b_general/best_prompt.txt ;;
    *)      echo gepa/runs/second_sweep/${1}_general_s1/best_prompt.txt ;;
  esac
}
fewshot_prompt() { echo fewshot/final_prompts/${1}/k1_s2.txt; }

tmux has-session -t $SESSION 2>/dev/null || tmux new-session -d -s $SESSION -n launcher
first=1
for spec in "${MODELS[@]}"; do
  IFS='|' read -r label model conc extra <<< "$spec"
  for arm in gepa_general fewshot_k1; do
    name=${label}_${arm}
    run_dir=$OUT/$name
    if [ -f "$run_dir/baseline_results.json" ]; then echo "[skip] $name done"; continue; fi
    if [ $first = 0 ]; then sleep "$STAGGER"; fi; first=0
    [ $arm = gepa_general ] && prompt=$(gepa_prompt $label) || prompt=$(fewshot_prompt $label)
    mkdir -p "$run_dir"; cp "$prompt" "$run_dir/system_prompt.txt"
    echo "{\"model\": \"$model\", \"prompt_source\": \"$prompt\", \"concurrency\": $conc, \"extra\": \"$extra\"}" > "$run_dir/run_meta.json"
    cmd=".venv/bin/python baselines/run_baseline.py --model $model --label $label --split train --extended \
      --max-tokens 30000 --max-concurrency $conc --system-prompt $prompt --out-dir $run_dir $extra"
    echo "[$(date +%H:%M:%S)] launching $name: $cmd"
    tmux new-window -t $SESSION -n "$name" "cd $PWD && $cmd 2>&1 | tee $run_dir/stdout.log; echo EXIT=\$? >> $run_dir/stdout.log; sleep 3600"
  done
done
echo "[$(date +%H:%M:%S)] all launched"
