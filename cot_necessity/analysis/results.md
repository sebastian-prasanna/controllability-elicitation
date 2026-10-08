# CoT necessity: MCQ vs MATH-500 / Olympiads

## Is CoT needed? Unconstrained accuracy, reasoning on vs off (mode=baseline, no Requirement)

| model | dataset | acc (reasoning on) | med words | truncated (16k) | acc (reasoning off) | n |
|---|---|---|---|---|---|---|
| Kimi-K3 | MMLU-Pro | 0.895 | 100 | 0.016 | 0.831 | 124 |
| Kimi-K3 | GPQA | 0.836 | 189 | 0.066 | 0.568 | 183 |
| Kimi-K3 | HLE | 0.368 | 1403 | 0.130 | 0.264 | 193 |
| Kimi-K3 | MATH-500 | 1.000 | 90 | 0.000 | 0.662 | 314 |
| Kimi-K3 | Olympiads | 0.746 | 1159 | 0.128 | 0.340 | 500 |
| GPT-OSS-120B | MMLU-Pro | 0.798 | 141 | 0.000 | - | 124 |
| GPT-OSS-120B | GPQA | 0.699 | 386 | 0.005 | - | 183 |
| GPT-OSS-120B | HLE | 0.130 | 428 | 0.005 | - | 193 |
| GPT-OSS-120B | MATH-500 | 0.984 | 134 | 0.000 | - | 314 |
| GPT-OSS-120B | Olympiads | 0.666 | 841 | 0.012 | - | 500 |

## Controllability (7 shared modes: alternating_case, end_of_sentence, ignore_question, lowercase_thinking, meow_between_words, repeat_sentences, uppercase_thinking)

acc|C / acc|NC = accuracy among compliant / non-compliant rollouts. On a task that needs CoT, complying (esp. ignore_question) should cost accuracy.

| model | arm | dataset | strict compliance | accuracy | acc\|C | acc\|NC | med words | truncated | n |
|---|---|---|---|---|---|---|---|---|---|
| gptoss120b | baseline | GPQA | 0.001 | 0.664 | 1.000 (n=1) | 0.663 (n=1280) | 277 | 0.001 | 1281 |
| gptoss120b | baseline | HLE | 0.002 | 0.108 | 0.000 (n=3) | 0.108 (n=1348) | 302 | 0.001 | 1351 |
| gptoss120b | baseline | MATH-500 | 0.010 | 0.981 | 1.000 (n=21) | 0.981 (n=2177) | 112 | 0.000 | 2198 |
| gptoss120b | baseline | MMLU-Pro | 0.025 | 0.791 | 0.955 (n=22) | 0.787 (n=846) | 128 | 0.000 | 868 |
| gptoss120b | baseline | Olympiads | 0.000 | 0.663 | 1.000 (n=1) | 0.662 (n=3496) | 729 | 0.007 | 3497 |
| gptoss120b | gepa | GPQA | 0.222 | 0.557 | 0.539 (n=284) | 0.562 (n=997) | 129 | 0.001 | 1281 |
| gptoss120b | gepa | HLE | 0.276 | 0.102 | 0.078 (n=373) | 0.111 (n=978) | 131 | 0.000 | 1351 |
| gptoss120b | gepa | MATH-500 | 0.288 | 0.868 | 0.934 (n=634) | 0.841 (n=1564) | 121 | 0.001 | 2198 |
| gptoss120b | gepa | MMLU-Pro | 0.343 | 0.684 | 0.735 (n=298) | 0.658 (n=570) | 89 | 0.000 | 868 |
| gptoss120b | gepa | Olympiads | 0.168 | 0.481 | 0.531 (n=589) | 0.471 (n=2910) | 328 | 0.003 | 3499 |
| gptoss120b | gepa_verbatim | MATH-500 | 0.295 | 0.853 | 0.923 (n=648) | 0.823 (n=1550) | 111 | 0.000 | 2198 |
| kimik3 | baseline | GPQA | 0.034 | 0.850 | 0.932 (n=44) | 0.847 (n=1237) | 1173 | 0.000 | 1281 |
| kimik3 | baseline | HLE | 0.033 | 0.423 | 0.295 (n=44) | 0.428 (n=1307) | 4635 | 0.000 | 1351 |
| kimik3 | baseline | MATH-500 | 0.207 | 0.977 | 0.987 (n=454) | 0.974 (n=1744) | 178 | 0.002 | 2198 |
| kimik3 | baseline | MMLU-Pro | 0.158 | 0.892 | 0.964 (n=137) | 0.878 (n=731) | 274 | 0.000 | 868 |
| kimik3 | baseline | Olympiads | 0.056 | 0.720 | 0.848 (n=197) | 0.712 (n=3303) | 1430 | 0.141 | 3500 |
| kimik3 | gepa | GPQA | 0.614 | 0.762 | 0.802 (n=786) | 0.699 (n=495) | 266 | 0.047 | 1281 |
| kimik3 | gepa | HLE | 0.517 | 0.280 | 0.253 (n=699) | 0.308 (n=652) | 415 | 0.099 | 1351 |
| kimik3 | gepa | MATH-500 | 0.679 | 0.929 | 0.956 (n=1493) | 0.874 (n=705) | 136 | 0.002 | 2198 |
| kimik3 | gepa | MMLU-Pro | 0.758 | 0.846 | 0.878 (n=658) | 0.743 (n=210) | 160 | 0.013 | 868 |
| kimik3 | gepa | Olympiads | 0.471 | 0.669 | 0.756 (n=1649) | 0.592 (n=1851) | 710 | 0.107 | 3500 |
| kimik3 | gepa_verbatim | MATH-500 | 0.700 | 0.808 | 0.828 (n=1539) | 0.762 (n=659) | 128 | 0.003 | 2198 |

## ignore_question only (the mode that directly forbids reasoning about the question)

| model | arm | dataset | strict compliance | accuracy | acc\|C | acc\|NC | med words | truncated | n |
|---|---|---|---|---|---|---|---|---|---|
| gptoss120b | baseline | GPQA | 0.000 | 0.634 | - | 0.634 (n=183) | 245 | 0.000 | 183 |
| gptoss120b | baseline | HLE | 0.000 | 0.098 | - | 0.098 (n=193) | 237 | 0.000 | 193 |
| gptoss120b | baseline | MATH-500 | 0.000 | 0.981 | - | 0.981 (n=314) | 108 | 0.000 | 314 |
| gptoss120b | baseline | MMLU-Pro | 0.000 | 0.790 | - | 0.790 (n=124) | 126 | 0.000 | 124 |
| gptoss120b | baseline | Olympiads | 0.000 | 0.669 | - | 0.669 (n=499) | 705 | 0.006 | 499 |
| gptoss120b | gepa | GPQA | 0.306 | 0.492 | 0.393 (n=56) | 0.535 (n=127) | 36 | 0.000 | 183 |
| gptoss120b | gepa | HLE | 0.326 | 0.140 | 0.095 (n=63) | 0.162 (n=130) | 44 | 0.000 | 193 |
| gptoss120b | gepa | MATH-500 | 0.083 | 0.634 | 0.577 (n=26) | 0.639 (n=288) | 83 | 0.000 | 314 |
| gptoss120b | gepa | MMLU-Pro | 0.290 | 0.468 | 0.500 (n=36) | 0.455 (n=88) | 40 | 0.000 | 124 |
| gptoss120b | gepa | Olympiads | 0.124 | 0.348 | 0.081 (n=62) | 0.386 (n=438) | 80 | 0.000 | 500 |
| gptoss120b | gepa_verbatim | MATH-500 | 0.076 | 0.510 | 0.375 (n=24) | 0.521 (n=290) | 56 | 0.000 | 314 |
| kimik3 | baseline | GPQA | 0.022 | 0.820 | 0.750 (n=4) | 0.821 (n=179) | 398 | 0.000 | 183 |
| kimik3 | baseline | HLE | 0.031 | 0.394 | 0.333 (n=6) | 0.396 (n=187) | 1723 | 0.000 | 193 |
| kimik3 | baseline | MATH-500 | 0.041 | 0.962 | 0.846 (n=13) | 0.967 (n=301) | 319 | 0.003 | 314 |
| kimik3 | baseline | MMLU-Pro | 0.065 | 0.887 | 0.750 (n=8) | 0.897 (n=116) | 292 | 0.000 | 124 |
| kimik3 | baseline | Olympiads | 0.038 | 0.722 | 0.316 (n=19) | 0.738 (n=481) | 1125 | 0.106 | 500 |
| kimik3 | gepa | GPQA | 0.514 | 0.585 | 0.457 (n=94) | 0.719 (n=89) | 333 | 0.022 | 183 |
| kimik3 | gepa | HLE | 0.430 | 0.244 | 0.145 (n=83) | 0.318 (n=110) | 407 | 0.026 | 193 |
| kimik3 | gepa | MATH-500 | 0.379 | 0.717 | 0.580 (n=119) | 0.800 (n=195) | 397 | 0.006 | 314 |
| kimik3 | gepa | MMLU-Pro | 0.516 | 0.750 | 0.781 (n=64) | 0.717 (n=60) | 294 | 0.016 | 124 |
| kimik3 | gepa | Olympiads | 0.408 | 0.452 | 0.304 (n=204) | 0.554 (n=296) | 646 | 0.034 | 500 |
| kimik3 | gepa_verbatim | MATH-500 | 0.389 | 0.672 | 0.615 (n=122) | 0.708 (n=192) | 407 | 0.006 | 314 |

## Length- and mode-matched difference vs MCQ

Within each (mode x log-length bin) cell with >=20 rollouts on both sides: compliance(math) - compliance(MCQ), averaged with weight min(n_math, n_mcq). Removes length and mode-mix differences; 'covered' = share of the math rollouts that fall in a matched cell.

| model | arm | math task | matched diff (pp) | raw diff (pp) | cells | covered |
|---|---|---|---|---|---|---|
| gptoss120b | baseline | MATH-500 | -0.3 | +0.2 | 30 | 0.88 |
| gptoss120b | baseline | Olympiads | -0.1 | -0.7 | 30 | 0.83 |
| gptoss120b | gepa | MATH-500 | +2.6 | +1.6 | 25 | 0.88 |
| gptoss120b | gepa | Olympiads | -0.4 | -10.5 | 25 | 0.76 |
| kimik3 | baseline | MATH-500 | -0.2 | +14.2 | 29 | 0.81 |
| kimik3 | baseline | Olympiads | +1.5 | -0.8 | 39 | 0.97 |
| kimik3 | gepa | MATH-500 | -5.1 | +6.7 | 26 | 0.82 |
| kimik3 | gepa | Olympiads | +1.8 | -14.1 | 38 | 0.92 |

## MATH-500 by level (gepa = math-format-adapted prompt)

| model | arm | level | strict compliance | accuracy | med words | n |
|---|---|---|---|---|---|---|
| gptoss120b | baseline | 1 | 0.030 | 0.991 | 41 | 231 |
| gptoss120b | baseline | 2 | 0.017 | 0.995 | 60 | 406 |
| gptoss120b | baseline | 3 | 0.009 | 0.987 | 108 | 462 |
| gptoss120b | baseline | 4 | 0.005 | 0.976 | 133 | 581 |
| gptoss120b | baseline | 5 | 0.000 | 0.965 | 254 | 518 |
| gptoss120b | gepa | 1 | 0.403 | 0.939 | 59 | 231 |
| gptoss120b | gepa | 2 | 0.350 | 0.904 | 77 | 406 |
| gptoss120b | gepa | 3 | 0.320 | 0.885 | 115 | 462 |
| gptoss120b | gepa | 4 | 0.262 | 0.867 | 133 | 581 |
| gptoss120b | gepa | 5 | 0.191 | 0.793 | 238 | 518 |
| gptoss120b | gepa_verbatim | 1 | 0.372 | 0.918 | 56 | 231 |
| gptoss120b | gepa_verbatim | 2 | 0.333 | 0.887 | 74 | 406 |
| gptoss120b | gepa_verbatim | 3 | 0.327 | 0.890 | 104 | 462 |
| gptoss120b | gepa_verbatim | 4 | 0.277 | 0.830 | 127 | 581 |
| gptoss120b | gepa_verbatim | 5 | 0.222 | 0.790 | 191 | 518 |
| kimik3 | baseline | 1 | 0.299 | 1.000 | 73 | 231 |
| kimik3 | baseline | 2 | 0.266 | 0.985 | 104 | 406 |
| kimik3 | baseline | 3 | 0.212 | 0.970 | 158 | 462 |
| kimik3 | baseline | 4 | 0.196 | 0.978 | 214 | 581 |
| kimik3 | baseline | 5 | 0.125 | 0.965 | 348 | 518 |
| kimik3 | gepa | 1 | 0.710 | 0.944 | 78 | 231 |
| kimik3 | gepa | 2 | 0.741 | 0.946 | 88 | 406 |
| kimik3 | gepa | 3 | 0.719 | 0.937 | 127 | 462 |
| kimik3 | gepa | 4 | 0.668 | 0.933 | 152 | 581 |
| kimik3 | gepa | 5 | 0.595 | 0.900 | 272 | 518 |
| kimik3 | gepa_verbatim | 1 | 0.745 | 0.900 | 68 | 231 |
| kimik3 | gepa_verbatim | 2 | 0.736 | 0.840 | 86 | 406 |
| kimik3 | gepa_verbatim | 3 | 0.699 | 0.831 | 131 | 462 |
| kimik3 | gepa_verbatim | 4 | 0.713 | 0.790 | 133 | 581 |
| kimik3 | gepa_verbatim | 5 | 0.639 | 0.743 | 222 | 518 |

## Per mode (strict compliance)

| model | arm | mode | GPQA | HLE | MATH-500 | MMLU-Pro | Olympiads |
|---|---|---|---|---|---|---|---|
| gptoss120b | baseline | alternating_case | 0.000 | 0.000 | 0.006 | 0.024 | 0.000 |
| gptoss120b | baseline | end_of_sentence | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| gptoss120b | baseline | ignore_question | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| gptoss120b | baseline | lowercase_thinking | 0.000 | 0.000 | 0.022 | 0.008 | 0.002 |
| gptoss120b | baseline | meow_between_words | 0.000 | 0.000 | 0.000 | 0.008 | 0.000 |
| gptoss120b | baseline | repeat_sentences | 0.005 | 0.010 | 0.010 | 0.089 | 0.000 |
| gptoss120b | baseline | uppercase_thinking | 0.000 | 0.005 | 0.029 | 0.048 | 0.000 |
| gptoss120b | gepa | alternating_case | 0.055 | 0.093 | 0.175 | 0.121 | 0.018 |
| gptoss120b | gepa | end_of_sentence | 0.000 | 0.005 | 0.025 | 0.024 | 0.012 |
| gptoss120b | gepa | ignore_question | 0.306 | 0.326 | 0.083 | 0.290 | 0.124 |
| gptoss120b | gepa | lowercase_thinking | 0.055 | 0.124 | 0.468 | 0.379 | 0.212 |
| gptoss120b | gepa | meow_between_words | 0.038 | 0.083 | 0.099 | 0.040 | 0.054 |
| gptoss120b | gepa | repeat_sentences | 0.863 | 0.824 | 0.841 | 0.935 | 0.657 |
| gptoss120b | gepa | uppercase_thinking | 0.235 | 0.477 | 0.328 | 0.613 | 0.102 |
| gptoss120b | gepa_verbatim | alternating_case | - | - | 0.175 | - | - |
| gptoss120b | gepa_verbatim | end_of_sentence | - | - | 0.032 | - | - |
| gptoss120b | gepa_verbatim | ignore_question | - | - | 0.076 | - | - |
| gptoss120b | gepa_verbatim | lowercase_thinking | - | - | 0.468 | - | - |
| gptoss120b | gepa_verbatim | meow_between_words | - | - | 0.086 | - | - |
| gptoss120b | gepa_verbatim | repeat_sentences | - | - | 0.857 | - | - |
| gptoss120b | gepa_verbatim | uppercase_thinking | - | - | 0.369 | - | - |
| kimik3 | baseline | alternating_case | 0.000 | 0.005 | 0.083 | 0.089 | 0.010 |
| kimik3 | baseline | end_of_sentence | 0.005 | 0.005 | 0.073 | 0.056 | 0.018 |
| kimik3 | baseline | ignore_question | 0.022 | 0.031 | 0.041 | 0.065 | 0.038 |
| kimik3 | baseline | lowercase_thinking | 0.027 | 0.021 | 0.328 | 0.153 | 0.100 |
| kimik3 | baseline | meow_between_words | 0.093 | 0.052 | 0.369 | 0.363 | 0.056 |
| kimik3 | baseline | repeat_sentences | 0.060 | 0.078 | 0.280 | 0.137 | 0.122 |
| kimik3 | baseline | uppercase_thinking | 0.033 | 0.036 | 0.271 | 0.242 | 0.050 |
| kimik3 | gepa | alternating_case | 0.295 | 0.228 | 0.484 | 0.556 | 0.266 |
| kimik3 | gepa | end_of_sentence | 0.667 | 0.487 | 0.717 | 0.798 | 0.340 |
| kimik3 | gepa | ignore_question | 0.514 | 0.430 | 0.379 | 0.516 | 0.408 |
| kimik3 | gepa | lowercase_thinking | 0.732 | 0.596 | 0.854 | 0.911 | 0.650 |
| kimik3 | gepa | meow_between_words | 0.727 | 0.658 | 0.736 | 0.863 | 0.434 |
| kimik3 | gepa | repeat_sentences | 0.568 | 0.513 | 0.818 | 0.758 | 0.610 |
| kimik3 | gepa | uppercase_thinking | 0.792 | 0.710 | 0.768 | 0.903 | 0.590 |
| kimik3 | gepa_verbatim | alternating_case | - | - | 0.535 | - | - |
| kimik3 | gepa_verbatim | end_of_sentence | - | - | 0.701 | - | - |
| kimik3 | gepa_verbatim | ignore_question | - | - | 0.389 | - | - |
| kimik3 | gepa_verbatim | lowercase_thinking | - | - | 0.831 | - | - |
| kimik3 | gepa_verbatim | meow_between_words | - | - | 0.758 | - | - |
| kimik3 | gepa_verbatim | repeat_sentences | - | - | 0.860 | - | - |
| kimik3 | gepa_verbatim | uppercase_thinking | - | - | 0.828 | - | - |
