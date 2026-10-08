#!/bin/bash
cd /root/controllability-elicitation
( until [ -f pinned_reeval/runs/qwen8b_t1val/fewshot_s2_heldout/summary.json ]; do sleep 2; done; tmux kill-session -t pr_qwen8b_t1val; echo "[$(date +%FT%T)] killed pr_qwen8b_t1val after fewshot_s2_heldout" >> pinned_reeval/runs/qwen8b_t1val/driver.log ) &
( until [ -f pinned_reeval/runs/qwen32b_t1val/fewshot_s1_indist/summary.json ]; do sleep 2; done; tmux kill-session -t pr_qwen32b_t1val; echo "[$(date +%FT%T)] killed pr_qwen32b_t1val after fewshot_s1_indist" >> pinned_reeval/runs/qwen32b_t1val/driver.log ) &
( until [ -f pinned_reeval/runs/kimik3_t1val/fewshot_s1_heldout/summary.json ]; do sleep 2; done; tmux kill-session -t pr_kimik3_t1val; echo "[$(date +%FT%T)] killed pr_kimik3_t1val after fewshot_s1_heldout" >> pinned_reeval/runs/kimik3_t1val/driver.log ) &
( until [ -f pinned_reeval/runs/glm53_t1val/fewshot_s1_indist/summary.json ]; do sleep 2; done; tmux kill-session -t pr_glm53_t1val; echo "[$(date +%FT%T)] killed pr_glm53_t1val after fewshot_s1_indist" >> pinned_reeval/runs/glm53_t1val/driver.log ) &
( until [ -f pinned_reeval/runs/glm53flash_t1val/gepa_s0_heldout/summary.json ]; do sleep 2; done; tmux kill-session -t pr_glm53flash_t1val; echo "[$(date +%FT%T)] killed pr_glm53flash_t1val after gepa_s0_heldout" >> pinned_reeval/runs/glm53flash_t1val/driver.log ) &
wait
