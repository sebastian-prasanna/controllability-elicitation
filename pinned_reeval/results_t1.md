# T=1 replica of the pinned re-evaluation (2026-10-06)

Same pins as `configs/eval_pins.json` (provider, top_p 1.0, 16k max tokens, reasoning effort, compliant-scope
meta-discussion judge) with `--temperature 1.0`. Launched 2026-10-06 04:23 UTC via
`python3 pinned_reeval/launch.py --models gptoss20b,gptoss120b,qwen8b,qwen32b,dsv4pro --suffix t1 --temperature 1.0`;
all 70 evals done by 12:16 UTC. Outputs: `pinned_reeval/runs/<model>_t1/`. Kimi K3 and GLM 5.3 excluded (their
endpoints run fixed ~1.0 regardless of the request). Max error rate 0.8% (qwen8b gepa_s1_test, 35/4464 Alibaba 429s);
errors are excluded from the compliance denominator. Billed generation cost $987 (DeepSeek $816).

## Per-run strict compliance (T=0 -> T=1)

| model | arm | test T0 | test T1 | heldout T0 | heldout T1 |
|---|---|---|---|---|---|
| gptoss20b | baseline | .006 | .013 | .013 | .009 |
| gptoss20b | gepa_s0 | .103 | .104 | .134 | .093 |
| gptoss20b | gepa_s1 | .186 | .172 | .251 | .175 |
| gptoss20b | gepa_s2 | .100 | .096 | .210 | .105 |
| gptoss20b | fewshot_s1 | .065 | .099 | .061 | .042 |
| gptoss20b | fewshot_s2 | .153 | .118 | .042 | .045 |
| gptoss20b | fewshot_s3 | .116 | .109 | .073 | .046 |
| gptoss120b | baseline | .023 | .028 | .049 | .047 |
| gptoss120b | gepa_s0 | .156 | .161 | .080 | .071 |
| gptoss120b | gepa_s1 | .214 | .208 | .126 | .120 |
| gptoss120b | gepa_s2 | .177 | .178 | .129 | .130 |
| gptoss120b | fewshot_s1 | .202 | .199 | .152 | .144 |
| gptoss120b | fewshot_s2 | .339 | .326 | .298 | .294 |
| gptoss120b | fewshot_s3 | .277 | .273 | .203 | .161 |
| qwen8b | baseline | .009 | .013 | .001 | .001 |
| qwen8b | gepa_s0 | .003 | .003 | .001 | .004 |
| qwen8b | gepa_s1 | .011 | .011 | .001 | .000 |
| qwen8b | gepa_s2 | .010 | .014 | .001 | .000 |
| qwen8b | fewshot_s1 | .129 | .108 | .004 | .004 |
| qwen8b | fewshot_s2 | .136 | .117 | .004 | .003 |
| qwen8b | fewshot_s3 | .121 | .115 | .003 | .003 |
| qwen32b | baseline | .027 | .025 | .005 | .003 |
| qwen32b | gepa_s0 | .123 | .103 | .009 | .021 |
| qwen32b | gepa_s1 | .126 | .119 | .009 | .021 |
| qwen32b | gepa_s2 | .104 | .098 | .005 | .011 |
| qwen32b | fewshot_s1 | .127 | .127 | .007 | .009 |
| qwen32b | fewshot_s2 | .207 | .132 | .007 | .021 |
| qwen32b | fewshot_s3 | .172 | .112 | .009 | .009 |
| dsv4pro | baseline | .009 | .012 | .000 | .000 |
| dsv4pro | gepa_s0 | .121 | .120 | .003 | .003 |
| dsv4pro | gepa_s1 | .073 | .075 | .004 | .001 |
| dsv4pro | gepa_s2 | .142 | .144 | .017 | .014 |
| dsv4pro | fewshot_s1 | .268 | .280 | .159 | .145 |
| dsv4pro | fewshot_s2 | .238 | .239 | .113 | .125 |
| dsv4pro | fewshot_s3 | .095 | .100 | .048 | .054 |

## Figure-style (best of 3 seeds selected on the in-distribution test score, within each temperature)

| model | arm | T0 seed | T0 test | T0 heldout | T1 seed | T1 test | T1 heldout |
|---|---|---|---|---|---|---|---|
| gptoss20b | baseline | - | .006 | .013 | - | .013 | .009 |
| gptoss20b | gepa | s1 | .186 | .251 | s1 | .172 | .175 |
| gptoss20b | fewshot | s2 | .153 | .042 | s2 | .118 | .045 |
| gptoss120b | baseline | - | .023 | .049 | - | .028 | .047 |
| gptoss120b | gepa | s1 | .214 | .126 | s1 | .208 | .120 |
| gptoss120b | fewshot | s2 | .339 | .298 | s2 | .326 | .294 |
| qwen8b | baseline | - | .009 | .001 | - | .013 | .001 |
| qwen8b | gepa | s1 | .011 | .001 | s2 | .014 | .000 |
| qwen8b | fewshot | s2 | .136 | .004 | s2 | .117 | .003 |
| qwen32b | baseline | - | .027 | .005 | - | .025 | .003 |
| qwen32b | gepa | s1 | .126 | .009 | s1 | .119 | .021 |
| qwen32b | fewshot | s2 | .207 | .007 | s2 | .132 | .021 |
| dsv4pro | baseline | - | .009 | .000 | - | .012 | .000 |
| dsv4pro | gepa | s2 | .142 | .017 | s2 | .144 | .014 |
| dsv4pro | fewshot | s1 | .268 | .159 | s1 | .280 | .145 |

## Reading

- gpt-oss-120b and DeepSeek V4 Pro: T=1 reproduces T=0 within ~1pp on every arm and split; seed selection is unchanged.
- gpt-oss-20b: in-distribution within ~3pp, but the GEPA held-out numbers drop (s1 .251 -> .175, s2 .210 -> .105).
  Greedy decoding was inflating the held-out GEPA transfer on the smallest gpt-oss model.
- qwen32b: few-shot s2/s3 lose 7-8pp in distribution at T=1 (.207 -> .132, .172 -> .112); GEPA within ~2pp.
- qwen8b: few-shot ~2pp lower at T=1, GEPA null at both temperatures.
- Seed selection flips only for qwen8b GEPA (s1 -> s2), where all seeds are at baseline level anyway.

## GLM-5.3-Flash (added 2026-10-07; Novita fp8 honours temperature, so it was re-run after all)

14 evals, launched 2026-10-06 19:23 UTC, done 2026-10-07 03:31 UTC (last 7 evals split over 3 parallel lanes). Max error rate 0.0%. Billed $80.92.

| model | arm | test T0 | test T1 | heldout T0 | heldout T1 |
|---|---|---|---|---|---|
| glm53flash | baseline | 0.018 | 0.015 | 0.004 | 0.001 |
| glm53flash | gepa_s0 | 0.019 | 0.016 | 0.005 | 0.004 |
| glm53flash | gepa_s1 | 0.077 | 0.047 | 0.036 | 0.019 |
| glm53flash | gepa_s2 | 0.058 | 0.049 | 0.014 | 0.026 |
| glm53flash | fewshot_s1 | 0.004 | 0.008 | 0.000 | 0.000 |
| glm53flash | fewshot_s2 | 0.006 | 0.005 | 0.000 | 0.000 |
| glm53flash | fewshot_s3 | 0.013 | 0.024 | 0.000 | 0.003 |

Figure-style (best seed on in-dist score, per temperature):

| arm | T0 seed | T0 test | T0 heldout | T1 seed | T1 test | T1 heldout |
|---|---|---|---|---|---|---|
| baseline | - | 0.018 | 0.004 | - | 0.015 | 0.001 |
| gepa | s1 | 0.077 | 0.036 | s2 | 0.049 | 0.026 |
| fewshot | s3 | 0.013 | 0.000 | s3 | 0.024 | 0.003 |
