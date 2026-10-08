# CoT necessity: MCQ vs MATH-500

## Is CoT needed? Unconstrained accuracy, reasoning on vs off (mode=baseline, no Requirement)

| model | dataset | acc (reasoning on) | med words | acc (reasoning off) | n |
|---|---|---|---|---|---|
| Kimi-K3 | MATH-500 | 1.000 | 62 | 1.000 | 5 |
| GPT-OSS-120B | GPQA | 0.400 | 1145 | - | 5 |
| GPT-OSS-120B | MATH-500 | 1.000 | 52 | - | 5 |

## Controllability (7 shared modes: alternating_case, end_of_sentence, ignore_question, lowercase_thinking, meow_between_words, repeat_sentences, uppercase_thinking)

| model | arm | dataset | strict compliance | accuracy | med words | n |
|---|---|---|---|---|---|---|
| gptoss120b | baseline | GPQA | 0.001 | 0.664 | 277 | 1281 |
| gptoss120b | baseline | HLE | 0.002 | 0.108 | 302 | 1351 |
| gptoss120b | baseline | MATH-500 | 0.029 | 1.000 | 57 | 35 |
| gptoss120b | baseline | MMLU-Pro | 0.025 | 0.791 | 128 | 868 |
| gptoss120b | gepa | GPQA | 0.222 | 0.557 | 129 | 1281 |
| gptoss120b | gepa | HLE | 0.276 | 0.102 | 131 | 1351 |
| gptoss120b | gepa | MATH-500 | 0.400 | 0.886 | 72 | 35 |
| gptoss120b | gepa | MMLU-Pro | 0.343 | 0.684 | 89 | 868 |
| gptoss120b | gepa_verbatim | MATH-500 | 0.429 | 0.943 | 67 | 35 |
| kimik3 | baseline | GPQA | 0.034 | 0.850 | 1173 | 1281 |
| kimik3 | baseline | HLE | 0.033 | 0.423 | 4635 | 1351 |
| kimik3 | baseline | MMLU-Pro | 0.158 | 0.892 | 274 | 868 |
| kimik3 | gepa | GPQA | 0.614 | 0.762 | 266 | 1281 |
| kimik3 | gepa | HLE | 0.517 | 0.280 | 415 | 1351 |
| kimik3 | gepa | MMLU-Pro | 0.758 | 0.846 | 160 | 868 |

## MATH-500 by level (gepa = math-format-adapted prompt)

| model | arm | level | strict compliance | accuracy | med words | n |
|---|---|---|---|---|---|---|
| gptoss120b | baseline | 1 | 0.029 | 1.000 | 57 | 35 |
| gptoss120b | gepa | 1 | 0.400 | 0.886 | 72 | 35 |
| gptoss120b | gepa_verbatim | 1 | 0.429 | 0.943 | 67 | 35 |

## Per mode (strict compliance)

| model | arm | mode | GPQA | HLE | MATH-500 | MMLU-Pro |
|---|---|---|---|---|---|---|
| gptoss120b | baseline | alternating_case | 0.000 | 0.000 | 0.000 | 0.024 |
| gptoss120b | baseline | end_of_sentence | 0.000 | 0.000 | 0.000 | 0.000 |
| gptoss120b | baseline | ignore_question | 0.000 | 0.000 | 0.000 | 0.000 |
| gptoss120b | baseline | lowercase_thinking | 0.000 | 0.000 | 0.200 | 0.008 |
| gptoss120b | baseline | meow_between_words | 0.000 | 0.000 | 0.000 | 0.008 |
| gptoss120b | baseline | repeat_sentences | 0.005 | 0.010 | 0.000 | 0.089 |
| gptoss120b | baseline | uppercase_thinking | 0.000 | 0.005 | 0.000 | 0.048 |
| gptoss120b | gepa | alternating_case | 0.055 | 0.093 | 0.400 | 0.121 |
| gptoss120b | gepa | end_of_sentence | 0.000 | 0.005 | 0.000 | 0.024 |
| gptoss120b | gepa | ignore_question | 0.306 | 0.326 | 0.200 | 0.290 |
| gptoss120b | gepa | lowercase_thinking | 0.055 | 0.124 | 0.400 | 0.379 |
| gptoss120b | gepa | meow_between_words | 0.038 | 0.083 | 0.000 | 0.040 |
| gptoss120b | gepa | repeat_sentences | 0.863 | 0.824 | 1.000 | 0.935 |
| gptoss120b | gepa | uppercase_thinking | 0.235 | 0.477 | 0.800 | 0.613 |
| gptoss120b | gepa_verbatim | alternating_case | - | - | 0.600 | - |
| gptoss120b | gepa_verbatim | end_of_sentence | - | - | 0.400 | - |
| gptoss120b | gepa_verbatim | ignore_question | - | - | 0.000 | - |
| gptoss120b | gepa_verbatim | lowercase_thinking | - | - | 0.400 | - |
| gptoss120b | gepa_verbatim | meow_between_words | - | - | 0.000 | - |
| gptoss120b | gepa_verbatim | repeat_sentences | - | - | 1.000 | - |
| gptoss120b | gepa_verbatim | uppercase_thinking | - | - | 0.600 | - |
| kimik3 | baseline | alternating_case | 0.000 | 0.005 | - | 0.089 |
| kimik3 | baseline | end_of_sentence | 0.005 | 0.005 | - | 0.056 |
| kimik3 | baseline | ignore_question | 0.022 | 0.031 | - | 0.065 |
| kimik3 | baseline | lowercase_thinking | 0.027 | 0.021 | - | 0.153 |
| kimik3 | baseline | meow_between_words | 0.093 | 0.052 | - | 0.363 |
| kimik3 | baseline | repeat_sentences | 0.060 | 0.078 | - | 0.137 |
| kimik3 | baseline | uppercase_thinking | 0.033 | 0.036 | - | 0.242 |
| kimik3 | gepa | alternating_case | 0.295 | 0.228 | - | 0.556 |
| kimik3 | gepa | end_of_sentence | 0.667 | 0.487 | - | 0.798 |
| kimik3 | gepa | ignore_question | 0.514 | 0.430 | - | 0.516 |
| kimik3 | gepa | lowercase_thinking | 0.732 | 0.596 | - | 0.911 |
| kimik3 | gepa | meow_between_words | 0.727 | 0.658 | - | 0.863 |
| kimik3 | gepa | repeat_sentences | 0.568 | 0.513 | - | 0.758 |
| kimik3 | gepa | uppercase_thinking | 0.792 | 0.710 | - | 0.903 |
