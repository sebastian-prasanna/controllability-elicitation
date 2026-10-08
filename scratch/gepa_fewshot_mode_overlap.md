# GEPA-general vs few-shot k1: per-mode strict-compliance overlap

Source: `pinned_reeval/runs/<model>/<arm>_<seed>_<split>/summary.json` -> `per_mode[mode].compliant / n`.
GEPA = mean over seeds s0/s1/s2 (general arm); fewshot = mean over seeds s1/s2/s3 (k=1); baseline = single run.
Strict compliance only (the same quantity as `compliance_rate` in the summaries).


## TEST (9 modes) split


### gptoss20b  (gepa n_seeds=3, fewshot n_seeds=3)

| mode | baseline | gepa (mean) | fewshot (mean) | gepa-fewshot |
|---|---|---|---|---|
| word_suppression | 0.035 | 0.020 | 0.019 | +0.001 |
| multiple_word_suppression | 0.012 | 0.004 | 0.008 | -0.004 |
| repeat_sentences | 0.008 | 0.768 | 0.374 | +0.394 |
| end_of_sentence | 0.000 | 0.001 | 0.001 | +0.001 |
| lowercase_thinking | 0.000 | 0.126 | 0.023 | +0.103 |
| meow_between_words | 0.000 | 0.017 | 0.000 | +0.017 |
| uppercase_thinking | 0.000 | 0.185 | 0.003 | +0.181 |
| ignore_question | 0.000 | 0.016 | 0.565 | -0.549 |
| alternating_case | 0.000 | 0.022 | 0.000 | +0.022 |
| **pooled** | 0.006 | 0.129 | 0.110 | |

### gptoss120b  (gepa n_seeds=3, fewshot n_seeds=3)

| mode | baseline | gepa (mean) | fewshot (mean) | gepa-fewshot |
|---|---|---|---|---|
| word_suppression | 0.127 | 0.024 | 0.243 | -0.219 |
| multiple_word_suppression | 0.033 | 0.005 | 0.082 | -0.077 |
| repeat_sentences | 0.028 | 0.841 | 0.860 | -0.019 |
| end_of_sentence | 0.000 | 0.005 | 0.048 | -0.043 |
| lowercase_thinking | 0.002 | 0.109 | 0.080 | +0.029 |
| meow_between_words | 0.002 | 0.064 | 0.014 | +0.050 |
| uppercase_thinking | 0.014 | 0.380 | 0.295 | +0.085 |
| ignore_question | 0.000 | 0.151 | 0.806 | -0.655 |
| alternating_case | 0.006 | 0.051 | 0.019 | +0.031 |
| **pooled** | 0.024 | 0.181 | 0.272 | |

### qwen8b  (gepa n_seeds=3, fewshot n_seeds=3)

| mode | baseline | gepa (mean) | fewshot (mean) | gepa-fewshot |
|---|---|---|---|---|
| word_suppression | 0.025 | 0.024 | 0.012 | +0.011 |
| multiple_word_suppression | 0.002 | 0.006 | 0.008 | -0.001 |
| repeat_sentences | 0.000 | 0.000 | 0.000 | +0.000 |
| end_of_sentence | 0.000 | 0.000 | 0.005 | -0.005 |
| lowercase_thinking | 0.008 | 0.009 | 0.000 | +0.009 |
| meow_between_words | 0.048 | 0.031 | 0.134 | -0.103 |
| uppercase_thinking | 0.000 | 0.000 | 0.000 | +0.000 |
| ignore_question | 0.000 | 0.000 | 0.989 | -0.989 |
| alternating_case | 0.000 | 0.000 | 0.000 | +0.000 |
| **pooled** | 0.009 | 0.008 | 0.128 | |

### qwen32b  (gepa n_seeds=3, fewshot n_seeds=3)

| mode | baseline | gepa (mean) | fewshot (mean) | gepa-fewshot |
|---|---|---|---|---|
| word_suppression | 0.102 | 0.133 | 0.139 | -0.006 |
| multiple_word_suppression | 0.019 | 0.018 | 0.024 | -0.006 |
| repeat_sentences | 0.000 | 0.000 | 0.000 | +0.000 |
| end_of_sentence | 0.000 | 0.002 | 0.000 | +0.002 |
| lowercase_thinking | 0.042 | 0.527 | 0.011 | +0.516 |
| meow_between_words | 0.034 | 0.299 | 0.663 | -0.364 |
| uppercase_thinking | 0.000 | 0.012 | 0.000 | +0.012 |
| ignore_question | 0.050 | 0.063 | 0.677 | -0.614 |
| alternating_case | 0.000 | 0.000 | 0.000 | +0.000 |
| **pooled** | 0.027 | 0.117 | 0.168 | |

### kimik3  (gepa n_seeds=3, fewshot n_seeds=3)

| mode | baseline | gepa (mean) | fewshot (mean) | gepa-fewshot |
|---|---|---|---|---|
| word_suppression | 0.035 | 0.407 | 0.077 | +0.329 |
| multiple_word_suppression | 0.021 | 0.214 | 0.041 | +0.172 |
| repeat_sentences | 0.086 | 0.508 | 0.347 | +0.161 |
| end_of_sentence | 0.018 | 0.535 | 0.282 | +0.253 |
| lowercase_thinking | 0.056 | 0.533 | 0.248 | +0.285 |
| meow_between_words | 0.144 | 0.611 | 0.384 | +0.227 |
| uppercase_thinking | 0.086 | 0.635 | 0.369 | +0.266 |
| ignore_question | 0.036 | 0.473 | 0.793 | -0.320 |
| alternating_case | 0.024 | 0.301 | 0.161 | +0.140 |
| **pooled** | 0.056 | 0.468 | 0.300 | |

### dsv4pro  (gepa n_seeds=3, fewshot n_seeds=3)

| mode | baseline | gepa (mean) | fewshot (mean) | gepa-fewshot |
|---|---|---|---|---|
| word_suppression | 0.004 | 0.003 | 0.029 | -0.026 |
| multiple_word_suppression | 0.004 | 0.001 | 0.013 | -0.012 |
| repeat_sentences | 0.000 | 0.081 | 0.117 | -0.035 |
| end_of_sentence | 0.000 | 0.001 | 0.125 | -0.123 |
| lowercase_thinking | 0.000 | 0.335 | 0.184 | +0.151 |
| meow_between_words | 0.070 | 0.395 | 0.271 | +0.124 |
| uppercase_thinking | 0.000 | 0.014 | 0.210 | -0.196 |
| ignore_question | 0.000 | 0.169 | 0.714 | -0.545 |
| alternating_case | 0.000 | 0.000 | 0.127 | -0.127 |
| **pooled** | 0.009 | 0.111 | 0.199 | |

### glm53  (gepa n_seeds=3, fewshot n_seeds=3)

| mode | baseline | gepa (mean) | fewshot (mean) | gepa-fewshot |
|---|---|---|---|---|
| word_suppression | 0.031 | 0.096 | 0.003 | +0.093 |
| multiple_word_suppression | 0.002 | 0.037 | 0.001 | +0.036 |
| repeat_sentences | 0.222 | 0.397 | 0.001 | +0.396 |
| end_of_sentence | 0.000 | 0.006 | 0.000 | +0.006 |
| lowercase_thinking | 0.106 | 0.171 | 0.003 | +0.168 |
| meow_between_words | 0.004 | 0.029 | 0.001 | +0.029 |
| uppercase_thinking | 0.066 | 0.147 | 0.000 | +0.147 |
| ignore_question | 0.016 | 0.017 | 0.299 | -0.283 |
| alternating_case | 0.000 | 0.002 | 0.000 | +0.002 |
| **pooled** | 0.050 | 0.100 | 0.034 | |

### glm53flash  (gepa n_seeds=3, fewshot n_seeds=3)

| mode | baseline | gepa (mean) | fewshot (mean) | gepa-fewshot |
|---|---|---|---|---|
| word_suppression | 0.037 | 0.027 | 0.017 | +0.010 |
| multiple_word_suppression | 0.015 | 0.004 | 0.017 | -0.012 |
| repeat_sentences | 0.046 | 0.293 | 0.002 | +0.291 |
| end_of_sentence | 0.000 | 0.001 | 0.006 | -0.005 |
| lowercase_thinking | 0.026 | 0.055 | 0.027 | +0.027 |
| meow_between_words | 0.000 | 0.007 | 0.000 | +0.007 |
| uppercase_thinking | 0.034 | 0.058 | 0.000 | +0.058 |
| ignore_question | 0.002 | 0.014 | 0.001 | +0.013 |
| alternating_case | 0.000 | 0.001 | 0.000 | +0.001 |
| **pooled** | 0.018 | 0.051 | 0.008 | |

### Solved modes (strict compliance > 0.5, arm mean)

| model | gepa only | fewshot only | both | neither |
|---|---|---|---|---|
| gptoss20b | repeat_sentences | ignore_question | - | 7 modes |
| gptoss120b | - | ignore_question | repeat_sentences | 7 modes |
| qwen8b | - | ignore_question | - | 8 modes |
| qwen32b | lowercase_thinking | ignore_question, meow_between_words | - | 6 modes |
| kimik3 | end_of_sentence, lowercase_thinking, meow_between_words, repeat_sentences, uppercase_thinking | ignore_question | - | 3 modes |
| dsv4pro | - | ignore_question | - | 8 modes |
| glm53 | - | - | - | 9 modes |
| glm53flash | - | - | - | 9 modes |

### Complementary cells (one arm > 0.3, other < 0.1)

**Count: 6** of 72 model x mode cells (gepa-only 3, fewshot-only 3)

| model | mode | gepa | fewshot | winner | gap |
|---|---|---|---|---|---|
| qwen8b | ignore_question | 0.000 | 0.989 | fewshot | 0.989 |
| qwen32b | ignore_question | 0.063 | 0.677 | fewshot | 0.614 |
| gptoss20b | ignore_question | 0.016 | 0.565 | fewshot | 0.549 |
| qwen32b | lowercase_thinking | 0.527 | 0.011 | gepa | 0.516 |
| glm53 | repeat_sentences | 0.397 | 0.001 | gepa | 0.396 |
| kimik3 | word_suppression | 0.407 | 0.077 | gepa | 0.329 |

### Large-gap cells (|gepa - fewshot| > 0.25; threshold-insensitive view)

**Count: 16**

| model | mode | gepa | fewshot | winner | gap |
|---|---|---|---|---|---|
| qwen8b | ignore_question | 0.000 | 0.989 | fewshot | 0.989 |
| gptoss120b | ignore_question | 0.151 | 0.806 | fewshot | 0.655 |
| qwen32b | ignore_question | 0.063 | 0.677 | fewshot | 0.614 |
| gptoss20b | ignore_question | 0.016 | 0.565 | fewshot | 0.549 |
| dsv4pro | ignore_question | 0.169 | 0.714 | fewshot | 0.545 |
| qwen32b | lowercase_thinking | 0.527 | 0.011 | gepa | 0.516 |
| glm53 | repeat_sentences | 0.397 | 0.001 | gepa | 0.396 |
| gptoss20b | repeat_sentences | 0.768 | 0.374 | gepa | 0.394 |
| qwen32b | meow_between_words | 0.299 | 0.663 | fewshot | 0.364 |
| kimik3 | word_suppression | 0.407 | 0.077 | gepa | 0.329 |
| kimik3 | ignore_question | 0.473 | 0.793 | fewshot | 0.320 |
| glm53flash | repeat_sentences | 0.293 | 0.002 | gepa | 0.291 |
| kimik3 | lowercase_thinking | 0.533 | 0.248 | gepa | 0.285 |
| glm53 | ignore_question | 0.017 | 0.299 | fewshot | 0.283 |
| kimik3 | uppercase_thinking | 0.635 | 0.369 | gepa | 0.266 |
| kimik3 | end_of_sentence | 0.535 | 0.282 | gepa | 0.253 |

### Per-mode correlation between arms (Pearson r over modes)

| model | r | n modes | note |
|---|---|---|---|
| gptoss20b | +0.415 | 9 |  |
| gptoss120b | +0.729 | 9 |  |
| qwen8b | -0.142 | 9 |  |
| qwen32b | +0.210 | 9 |  |
| kimik3 | +0.508 | 9 |  |
| dsv4pro | +0.397 | 9 |  |
| glm53 | -0.244 | 9 |  |
| glm53flash | -0.127 | 9 |  |
| **pooled (raw, all cells)** | +0.450 | 72 | |
| **pooled (within-model z-scored)** | +0.218 | 72 | removes model-level scale |

## HELD-OUT (3 modes) split


### gptoss20b  (gepa n_seeds=3, fewshot n_seeds=3)

| mode | baseline | gepa (mean) | fewshot (mean) | gepa-fewshot |
|---|---|---|---|---|
| start_of_sentence | 0.004 | 0.565 | 0.126 | +0.439 |
| letter_suppression | 0.034 | 0.002 | 0.008 | -0.006 |
| no_spaces | 0.000 | 0.028 | 0.042 | -0.014 |
| **pooled** | 0.013 | 0.198 | 0.059 | |

### gptoss120b  (gepa n_seeds=3, fewshot n_seeds=3)

| mode | baseline | gepa (mean) | fewshot (mean) | gepa-fewshot |
|---|---|---|---|---|
| start_of_sentence | 0.118 | 0.327 | 0.469 | -0.142 |
| letter_suppression | 0.024 | 0.004 | 0.067 | -0.063 |
| no_spaces | 0.004 | 0.003 | 0.117 | -0.113 |
| **pooled** | 0.049 | 0.112 | 0.218 | |

### qwen8b  (gepa n_seeds=3, fewshot n_seeds=3)

| mode | baseline | gepa (mean) | fewshot (mean) | gepa-fewshot |
|---|---|---|---|---|
| start_of_sentence | 0.000 | 0.000 | 0.000 | +0.000 |
| letter_suppression | 0.004 | 0.003 | 0.011 | -0.007 |
| no_spaces | 0.000 | 0.000 | 0.000 | +0.000 |
| **pooled** | 0.001 | 0.001 | 0.004 | |

### qwen32b  (gepa n_seeds=3, fewshot n_seeds=3)

| mode | baseline | gepa (mean) | fewshot (mean) | gepa-fewshot |
|---|---|---|---|---|
| start_of_sentence | 0.000 | 0.000 | 0.000 | +0.000 |
| letter_suppression | 0.014 | 0.018 | 0.022 | -0.004 |
| no_spaces | 0.000 | 0.005 | 0.001 | +0.004 |
| **pooled** | 0.005 | 0.008 | 0.008 | |

### kimik3  (gepa n_seeds=3, fewshot n_seeds=3)

| mode | baseline | gepa (mean) | fewshot (mean) | gepa-fewshot |
|---|---|---|---|---|
| start_of_sentence | 0.084 | 0.467 | 0.451 | +0.015 |
| letter_suppression | 0.004 | 0.108 | 0.034 | +0.074 |
| no_spaces | 0.052 | 0.367 | 0.201 | +0.165 |
| **pooled** | 0.047 | 0.314 | 0.229 | |

### dsv4pro  (gepa n_seeds=3, fewshot n_seeds=3)

| mode | baseline | gepa (mean) | fewshot (mean) | gepa-fewshot |
|---|---|---|---|---|
| start_of_sentence | 0.000 | 0.006 | 0.104 | -0.098 |
| letter_suppression | 0.000 | 0.000 | 0.011 | -0.011 |
| no_spaces | 0.000 | 0.018 | 0.205 | -0.187 |
| **pooled** | 0.000 | 0.008 | 0.107 | |

### glm53  (gepa n_seeds=3, fewshot n_seeds=3)

| mode | baseline | gepa (mean) | fewshot (mean) | gepa-fewshot |
|---|---|---|---|---|
| start_of_sentence | 0.030 | 0.033 | 0.000 | +0.033 |
| letter_suppression | 0.000 | 0.007 | 0.001 | +0.007 |
| no_spaces | 0.014 | 0.013 | 0.000 | +0.013 |
| **pooled** | 0.015 | 0.018 | 0.000 | |

### glm53flash  (gepa n_seeds=3, fewshot n_seeds=3)

| mode | baseline | gepa (mean) | fewshot (mean) | gepa-fewshot |
|---|---|---|---|---|
| start_of_sentence | 0.010 | 0.053 | 0.000 | +0.053 |
| letter_suppression | 0.002 | 0.002 | 0.000 | +0.002 |
| no_spaces | 0.000 | 0.000 | 0.000 | +0.000 |
| **pooled** | 0.004 | 0.018 | 0.000 | |

### Solved modes (strict compliance > 0.5, arm mean)

| model | gepa only | fewshot only | both | neither |
|---|---|---|---|---|
| gptoss20b | start_of_sentence | - | - | 2 modes |
| gptoss120b | - | - | - | 3 modes |
| qwen8b | - | - | - | 3 modes |
| qwen32b | - | - | - | 3 modes |
| kimik3 | - | - | - | 3 modes |
| dsv4pro | - | - | - | 3 modes |
| glm53 | - | - | - | 3 modes |
| glm53flash | - | - | - | 3 modes |

### Complementary cells (one arm > 0.3, other < 0.1)

**Count: 0** of 24 model x mode cells (gepa-only 0, fewshot-only 0)

| model | mode | gepa | fewshot | winner | gap |
|---|---|---|---|---|---|

### Large-gap cells (|gepa - fewshot| > 0.25; threshold-insensitive view)

**Count: 1**

| model | mode | gepa | fewshot | winner | gap |
|---|---|---|---|---|---|
| gptoss20b | start_of_sentence | 0.565 | 0.126 | gepa | 0.439 |

### Per-mode correlation between arms (Pearson r over modes)

| model | r | n modes | note |
|---|---|---|---|
| gptoss20b | +0.971 | 3 |  |
| gptoss120b | +0.993 | 3 |  |
| qwen8b | +1.000 | 3 |  |
| qwen32b | +0.975 | 3 |  |
| kimik3 | +0.935 | 3 |  |
| dsv4pro | +0.986 | 3 |  |
| glm53 | -0.679 | 3 |  |
| glm53flash | n/a | 3 | zero variance in one arm |
| **pooled (raw, all cells)** | +0.715 | 24 | |
| **pooled (within-model z-scored)** | +0.740 | 21 | removes model-level scale |
