#!/bin/bash
# 2026-09-04 00:20 UTC: qwen32b runs restarted at concurrency 150 (was 60). At 60 both runs
# crawled (~15 rollouts/min each, ETA 11-13h) with ZERO errors. Cause: with max_tokens=30000
# OpenRouter only routes qwen/qwen3-32b to SiliconFlow (DeepInfra caps output at 16384; Nebius
# and Groq no longer list the model), and SiliconFlow is slow per request. Partial conc-60 runs
# (~3%) preserved under failed/. --ignore-provider nebius kept (harmless).
set -u
cd "$(dirname "$0")/../.."
OUT=sft/extended_training_data_runs
gepa_prompt=gepa/runs/second_sweep/qwen32b_general_s1/best_prompt.txt
fs_prompt=fewshot/final_prompts/qwen32b/k1_s2.txt
for arm in gepa_general fewshot_k1; do
  name=qwen32b_$arm; run_dir=$OUT/$name
  [ $arm = gepa_general ] && prompt=$gepa_prompt || prompt=$fs_prompt
  mkdir -p "$run_dir"; cp "$prompt" "$run_dir/system_prompt.txt"
  echo "{\"model\": \"qwen/qwen3-32b\", \"prompt_source\": \"$prompt\", \"concurrency\": 150, \"extra\": \"--ignore-provider nebius\", \"note\": \"relaunched at conc 150, see relaunch_qwen32b.sh\"}" > "$run_dir/run_meta.json"
  cmd=".venv/bin/python baselines/run_baseline.py --model qwen/qwen3-32b --label qwen32b --split train --extended \
    --max-tokens 30000 --max-concurrency 150 --system-prompt $prompt --out-dir $run_dir --ignore-provider nebius"
  echo "[$(date +%H:%M:%S)] relaunching $name: $cmd"
  tmux new-window -t ext_runs -n "$name" "cd $PWD && $cmd 2>&1 | tee $run_dir/stdout.log; echo EXIT=\$? >> $run_dir/stdout.log; sleep 3600"
  sleep 60
done
