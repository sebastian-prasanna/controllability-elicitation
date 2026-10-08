# Pinned re-eval results (configs/eval_pins.json; T=0, top_p 1, 16k, gpt-oss effort medium, others unset; meta judge on, scope=compliant)

Run 2026-09-22 04:06-23:08 UTC. 112 evals, 333,984 rollouts, 4 errored rollouts (glm53flash 429s before its retry increase). Provider pins: kimi moonshotai, gpt-oss groq, qwen32b deepinfra fp8, qwen8b alibaba, glm53 z-ai, dsv4pro alibaba, glm53flash sail-research fp8 for 13/14 evals -- Sail Research delisted glm-5.3-flash mid-run, so fewshot_s3_heldout ran on novita fp8. test = 496 q x 9 modes (4464); heldout = 500 q x 3 held-out modes (1500). GEPA = general arm, seeds s0/s1/s2; few-shot = k1 s1/s2/s3.

| model | split | arm | strict compliance | accuracy | no-narration (compliant) | errors |
|---|---|---|---|---|---|---|
| gptoss20b | test | baseline | 0.006 | 0.454 | 1.000 | 0 |
| gptoss20b | test | gepa | 0.130 ± 0.049 | 0.377 ± 0.009 | 0.915 ± 0.015 | 0 |
| gptoss20b | test | fewshot | 0.111 ± 0.044 | 0.413 ± 0.017 | 0.955 ± 0.023 | 0 |
| gptoss20b | heldout | baseline | 0.013 | 0.431 | 0.947 | 0 |
| gptoss20b | heldout | gepa | 0.198 ± 0.059 | 0.358 ± 0.007 | 0.586 ± 0.123 | 0 |
| gptoss20b | heldout | fewshot | 0.059 ± 0.016 | 0.419 ± 0.011 | 0.803 ± 0.047 | 0 |
| gptoss120b | test | baseline | 0.023 | 0.483 | 0.853 | 0 |
| gptoss120b | test | gepa | 0.182 ± 0.029 | 0.427 ± 0.012 | 0.733 ± 0.133 | 0 |
| gptoss120b | test | fewshot | 0.273 ± 0.068 | 0.451 ± 0.014 | 0.875 ± 0.079 | 0 |
| gptoss120b | heldout | baseline | 0.049 | 0.480 | 0.589 | 0 |
| gptoss120b | heldout | gepa | 0.112 ± 0.027 | 0.440 ± 0.013 | 0.650 ± 0.135 | 0 |
| gptoss120b | heldout | fewshot | 0.218 ± 0.074 | 0.456 ± 0.007 | 0.792 ± 0.042 | 0 |
| qwen8b | test | baseline | 0.009 | 0.399 | 0.951 | 0 |
| qwen8b | test | gepa | 0.008 ± 0.004 | 0.403 ± 0.003 | 0.943 ± 0.023 | 0 |
| qwen8b | test | fewshot | 0.129 ± 0.008 | 0.348 ± 0.021 | 0.976 ± 0.013 | 0 |
| qwen8b | heldout | baseline | 0.001 | 0.421 | 1.000 | 0 |
| qwen8b | heldout | gepa | 0.001 ± 0.000 | 0.422 ± 0.012 | 1.000 ± 0.000 | 0 |
| qwen8b | heldout | fewshot | 0.004 ± 0.001 | 0.402 ± 0.012 | 1.000 ± 0.000 | 0 |
| qwen32b | test | baseline | 0.027 | 0.426 | 0.893 | 0 |
| qwen32b | test | gepa | 0.117 ± 0.012 | 0.415 ± 0.005 | 0.946 ± 0.042 | 0 |
| qwen32b | test | fewshot | 0.169 ± 0.040 | 0.401 ± 0.007 | 0.847 ± 0.205 | 0 |
| qwen32b | heldout | baseline | 0.005 | 0.430 | 1.000 | 0 |
| qwen32b | heldout | gepa | 0.008 ± 0.003 | 0.438 ± 0.007 | 1.000 ± 0.000 | 0 |
| qwen32b | heldout | fewshot | 0.008 ± 0.001 | 0.421 ± 0.005 | 1.000 ± 0.000 | 0 |
| kimik3 | test | baseline | 0.056 | 0.702 | 0.742 | 0 |
| kimik3 | test | gepa | 0.470 ± 0.087 | 0.633 ± 0.022 | 0.946 ± 0.027 | 0 |
| kimik3 | test | fewshot | 0.302 ± 0.032 | 0.618 ± 0.003 | 0.799 ± 0.050 | 0 |
| kimik3 | heldout | baseline | 0.047 | 0.671 | 0.971 | 0 |
| kimik3 | heldout | gepa | 0.314 ± 0.029 | 0.644 ± 0.026 | 0.971 ± 0.017 | 0 |
| kimik3 | heldout | fewshot | 0.229 ± 0.036 | 0.633 ± 0.008 | 0.887 ± 0.036 | 0 |
| dsv4pro | test | baseline | 0.009 | 0.566 | 0.897 | 0 |
| dsv4pro | test | gepa | 0.112 ± 0.035 | 0.555 ± 0.018 | 0.883 ± 0.052 | 0 |
| dsv4pro | test | fewshot | 0.200 ± 0.092 | 0.570 ± 0.006 | 0.675 ± 0.130 | 0 |
| dsv4pro | heldout | baseline | 0.000 | 0.566 | - | 0 |
| dsv4pro | heldout | gepa | 0.008 ± 0.008 | 0.585 ± 0.009 | 0.822 ± 0.168 | 0 |
| dsv4pro | heldout | fewshot | 0.107 ± 0.056 | 0.586 ± 0.005 | 0.613 ± 0.197 | 0 |
| glm53 | test | baseline | 0.050 | 0.477 | 0.937 | 0 |
| glm53 | test | gepa | 0.100 ± 0.025 | 0.473 ± 0.005 | 0.914 ± 0.021 | 0 |
| glm53 | test | fewshot | 0.034 ± 0.007 | 0.405 ± 0.005 | 0.017 ± 0.018 | 0 |
| glm53 | heldout | baseline | 0.015 | 0.483 | 0.955 | 0 |
| glm53 | heldout | gepa | 0.018 ± 0.010 | 0.498 ± 0.004 | 0.981 ± 0.017 | 0 |
| glm53 | heldout | fewshot | 0.000 ± 0.000 | 0.408 ± 0.023 | 1.000 (n=1/3) | 0 |
| glm53flash | test | baseline | 0.018 | 0.457 | 0.873 | 1 |
| glm53flash | test | gepa | 0.051 ± 0.030 | 0.466 ± 0.013 | 0.885 ± 0.055 | 0 |
| glm53flash | test | fewshot | 0.008 ± 0.004 | 0.467 ± 0.006 | 0.246 ± 0.395 | 0 |
| glm53flash | heldout | baseline | 0.004 | 0.474 | 1.000 | 3 |
| glm53flash | heldout | gepa | 0.018 ± 0.016 | 0.475 ± 0.010 | 0.975 ± 0.043 | 0 |
| glm53flash | heldout | fewshot | 0.000 ± 0.000 | 0.471 ± 0.015 | - | 0 |
