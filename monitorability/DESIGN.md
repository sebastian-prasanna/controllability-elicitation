# Does simple prompt optimization decrease CoT monitorability? (quick & dirty)

Started 2026-09-07. Policy: `openai/gpt-oss-20b` via OpenRouter, reasoning effort **medium**, temperature 1.0.
Monitor: `openai/gpt-5.6-luna` (cheap), CoT + final-answer scope, prompts verbatim from
`/root/monitorability-evals/monitor_prompts/templates/`. Metric: paper's g-mean² (cross-fit bootstrap)
plus transparent raw rates (hint-taking rate, TE, TPR_raw, FPR_control), and CoT length.

## Evals (easy, single-turn, from arXiv 2512.18311 OSS release)
`gpqa` (100 instances; sycophancy / reward-hacking / unethical hints), `scruples_first_person` (150),
`scruples_suggest_wrong` (150), `wmdp_sandbagging` (150). 10 samples per row (row = instance × arm).

## Arms (same rows, same samples-per-row)
| arm | system prompt | user-message suffix | question answered |
|---|---|---|---|
| `baseline` | none | none | reference monitorability |
| `manual` | hand-optimized general-advice controllability prompt (`prompts/manual_general.txt`) | none | does the optimized prompt alone change monitorability? |
| `gepa` | GEPA-optimized general-advice prompt (`prompts/gepa_general.txt`, from gepa/runs/second_sweep/gptoss20b_general_s1) | none | same, for the real prompt-optimizer output |
| `manual_req` | manual prompt | lowercase-thinking Requirement (`prompts/req_lowercase.txt`) | prompt in actual use: is *controlled* CoT less monitorable? |
| `baseline_req` | none | lowercase-thinking Requirement | separates the constraint's effect from the prompt's effect |

Lowercase was chosen as the constraint because it is benign for content (it changes surface form only),
is one of the modes gpt-oss-20b can actually comply with, and does not remove words (unlike
suppression modes) — so any monitorability change is about *how the model reasons under control*,
not about a constraint that deletes evidence.

## Reading the result
- Primary: g-mean² per dataset and pooled, arm vs `baseline`, with bootstrap std.
- Secondary: TPR_raw (of hint-following rollouts, fraction flagged) and FPR_control; hint-taking rate
  and TE (the prompt might change how often the model takes hints — must be reported alongside);
  mean reasoning tokens (paper: monitorability rises with CoT length, so a length change is a confound).
- Caveat: an arm whose TE collapses (few eligible instances) has an unreliable g-mean².
