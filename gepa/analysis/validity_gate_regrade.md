# Trace-validity gate re-grade (MIN_TRACE_ALPHA=50)

## Best-candidate changes per threshold (alpha chars)

| run | best_orig | median alpha (best) | thr 10 | thr 25 | thr 50 | thr 100 | thr 200 |
|---|---|---|---|---|---|---|---|
| initial_sweep/dsv4pro_free | 2 | 1418.0 | same | same | same | same | same |
| initial_sweep/dsv4pro_general | 3 | 2067.5 | same | same | same | same | same |
| initial_sweep/glm53_free | 1 | 3652.5 | same | same | same | same | same |
| initial_sweep/glm53_general | 5 | 2383.5 | same | same | same | same | same |
| initial_sweep/glm53flash_free | 4 | 2073.5 | same | same | same | same | same |
| initial_sweep/glm53flash_general | 0 | seed | same | same | same | same | same |
| initial_sweep/gptoss120b_free | 3 | 657.0 | same | same | same | same | same |
| initial_sweep/gptoss120b_general | 1 | 1000.0 | same | same | same | same | same |
| initial_sweep/gptoss20b_free | 3 | 1316.5 | same | same | same | same | same |
| initial_sweep/gptoss20b_general | 3 | 2020.5 | same | same | same | **->cand4** | **->cand4** |
| initial_sweep/kimik3_free | 2 | 1350.5 | same | same | same | same | same |
| initial_sweep/kimik3_general | 1 | 1907.0 | same | same | same | same | same |
| initial_sweep/qwen32b_free | 3 | 0.0 | **->cand2** | **->cand2** | **->cand2** | **->cand2** | **->cand2** |
| initial_sweep/qwen32b_general | 3 | 2419.5 | same | same | same | same | same |
| initial_sweep/qwen8b_free | 1 | 2998.5 | same | same | same | same | same |
| initial_sweep/qwen8b_general | 4 | 3261.5 | same | same | same | same | same |
| new_models_sweep/glm52_free | 4 | 1269.5 | same | same | same | same | same |
| new_models_sweep/glm52_general | 2 | 1480.0 | same | same | same | same | same |
| new_models_sweep/kimik26_free | 1 | 20554.0 | same | same | same | same | same |
| new_models_sweep/kimik26_general | 3 | 21447.5 | same | same | same | same | same |
| new_models_sweep/nemotron3ultra_free | 1 | 765.0 | same | same | same | same | same |
| new_models_sweep/nemotron3ultra_general | 2 | 4070.0 | same | same | same | same | same |
| new_models_sweep/qwen38_2.4t_free | 3 | 5952.0 | same | same | same | same | same |
| new_models_sweep/qwen38_2.4t_general | 4 | 15176.0 | same | same | same | same | same |
| new_models_sweep/qwen38_27b_free | 3 | 8334.0 | same | same | same | same | same |
| new_models_sweep/qwen38_27b_general | 1 | 10494.0 | same | same | same | same | same |
| second_sweep/dsv4pro_free_s1 | 1 | 1603.5 | same | same | same | same | same |
| second_sweep/dsv4pro_free_s2 | 3 | 4450.0 | same | same | same | same | same |
| second_sweep/dsv4pro_general_s1 | 1 | 2231.0 | same | same | same | same | same |
| second_sweep/dsv4pro_general_s2 | 4 | 9843.0 | same | same | same | same | same |
| second_sweep/glm53_free_s1 | 1 | 4627.0 | same | same | same | same | same |
| second_sweep/glm53_free_s2 | 1 | 4614.5 | same | same | same | same | same |
| second_sweep/glm53_general_s1 | 3 | 5274.0 | same | same | same | same | same |
| second_sweep/glm53_general_s2 | 3 | 8275.5 | same | same | same | same | same |
| second_sweep/glm53flash_free_s1 | 2 | 2374.0 | same | same | same | same | same |
| second_sweep/glm53flash_free_s2 | 1 | 3798.0 | same | same | same | same | same |
| second_sweep/glm53flash_general_s1 | 2 | 3569.5 | same | same | same | same | same |
| second_sweep/glm53flash_general_s2 | 3 | 6816.0 | same | same | same | same | same |
| second_sweep/gptoss120b_free_s1 | 4 | 650.5 | same | same | same | same | same |
| second_sweep/gptoss120b_free_s2 | 3 | 944.0 | same | same | same | same | same |
| second_sweep/gptoss120b_general_s1 | 4 | 731.5 | same | same | same | same | same |
| second_sweep/gptoss120b_general_s2 | 4 | 904.5 | same | same | same | same | same |
| second_sweep/gptoss20b_free_s1 | 2 | 1190.0 | same | same | same | same | **->cand1** |
| second_sweep/gptoss20b_free_s2 | 2 | 4158.5 | same | same | same | same | same |
| second_sweep/gptoss20b_general_s1 | 2 | 943.0 | same | same | same | same | same |
| second_sweep/gptoss20b_general_s2 | 4 | 2211.5 | same | same | same | same | same |
| second_sweep/kimik3_free_s1 | 3 | 900.0 | same | same | same | same | same |
| second_sweep/kimik3_free_s2 | 5 | 1098.0 | same | same | same | same | same |
| second_sweep/kimik3_general_s1 | 1 | 1727.0 | same | same | same | same | same |
| second_sweep/kimik3_general_s2 | 4 | 1196.0 | same | same | same | same | same |
| second_sweep/qwen32b_free_s1 | 1 | 0.0 | **->cand0** | **->cand0** | **->cand0** | **->cand0** | **->cand0** |
| second_sweep/qwen32b_free_s2 | 2 | 0.0 | **->cand3** | **->cand3** | **->cand3** | **->cand3** | **->cand3** |
| second_sweep/qwen32b_general_s1 | 1 | 2726.5 | same | same | same | same | same |
| second_sweep/qwen32b_general_s2 | 5 | 2790.5 | same | same | same | same | same |
| second_sweep/qwen8b_free_s1 | 1 | 2619.0 | same | same | same | same | same |
| second_sweep/qwen8b_free_s2 | 2 | 2740.0 | same | same | same | same | same |
| second_sweep/qwen8b_general_s1 | 0 | seed | same | same | same | same | same |
| second_sweep/qwen8b_general_s2 | 0 | seed | same | same | same | same | same |

## Test strict compliance: original vs gated at thr 50

| run | orig | gated | invalid frac |
|---|---|---|---|
| initial_sweep/dsv4pro_free | 0.120 | 0.119 | 0.255 |
| initial_sweep/dsv4pro_general | 0.115 | 0.114 | 0.235 |
| initial_sweep/glm53_free | 0.186 | 0.179 | 0.018 |
| initial_sweep/glm53_general | 0.143 | 0.140 | 0.004 |
| initial_sweep/glm53flash_free | 0.116 | 0.112 | 0.016 |
| initial_sweep/glm53flash_general | 0.028 | 0.028 | 0.003 |
| initial_sweep/gptoss120b_free | 0.204 | 0.204 | 0.001 |
| initial_sweep/gptoss120b_general | 0.183 | 0.162 | 0.026 |
| initial_sweep/gptoss20b_free | 0.345 | 0.343 | 0.003 |
| initial_sweep/gptoss20b_general | 0.110 | 0.109 | 0.001 |
| initial_sweep/kimik3_free | 0.387 | 0.381 | 0.013 |
| initial_sweep/kimik3_general | 0.455 | 0.450 | 0.010 |
| initial_sweep/qwen32b_free | 0.498 | 0.000 | 1.000 |
| initial_sweep/qwen32b_general | 0.120 | 0.120 | 0.001 |
| initial_sweep/qwen8b_free | 0.003 | 0.003 | 0.059 |
| initial_sweep/qwen8b_general | 0.004 | 0.004 | 0.022 |
| new_models_sweep/glm52_free | 0.264 | 0.263 | 0.004 |
| new_models_sweep/glm52_general | 0.200 | 0.199 | 0.003 |
| new_models_sweep/kimik26_free | 0.001 | 0.001 | 0.135 |
| new_models_sweep/kimik26_general | 0.001 | 0.001 | 0.148 |
| new_models_sweep/nemotron3ultra_free | 0.321 | 0.320 | 0.193 |
| new_models_sweep/nemotron3ultra_general | 0.207 | 0.207 | 0.112 |
| new_models_sweep/qwen38_2.4t_free | 0.003 | 0.003 | 0.000 |
| new_models_sweep/qwen38_2.4t_general | 0.002 | 0.002 | 0.000 |
| new_models_sweep/qwen38_27b_free | 0.006 | 0.006 | 0.000 |
| new_models_sweep/qwen38_27b_general | 0.006 | 0.006 | 0.000 |
| second_sweep/dsv4pro_free_s1 | 0.107 | 0.106 | 0.087 |
| second_sweep/dsv4pro_free_s2 | 0.174 | 0.173 | 0.063 |
| second_sweep/dsv4pro_general_s1 | 0.088 | 0.087 | 0.063 |
| second_sweep/dsv4pro_general_s2 | 0.139 | 0.138 | 0.105 |
| second_sweep/glm53_free_s1 | 0.123 | 0.123 | 0.001 |
| second_sweep/glm53_free_s2 | 0.254 | 0.253 | 0.012 |
| second_sweep/glm53_general_s1 | 0.077 | 0.077 | 0.002 |
| second_sweep/glm53_general_s2 | 0.129 | 0.125 | 0.006 |
| second_sweep/glm53flash_free_s1 | 0.077 | 0.075 | 0.022 |
| second_sweep/glm53flash_free_s2 | 0.097 | 0.096 | 0.001 |
| second_sweep/glm53flash_general_s1 | 0.082 | 0.081 | 0.004 |
| second_sweep/glm53flash_general_s2 | 0.062 | 0.062 | 0.007 |
| second_sweep/gptoss120b_free_s1 | 0.258 | 0.258 | 0.001 |
| second_sweep/gptoss120b_free_s2 | 0.178 | 0.151 | 0.029 |
| second_sweep/gptoss120b_general_s1 | 0.209 | 0.208 | 0.002 |
| second_sweep/gptoss120b_general_s2 | 0.183 | 0.183 | 0.001 |
| second_sweep/gptoss20b_free_s1 | 0.083 | 0.083 | 0.001 |
| second_sweep/gptoss20b_free_s2 | 0.137 | 0.137 | 0.000 |
| second_sweep/gptoss20b_general_s1 | 0.168 | 0.166 | 0.002 |
| second_sweep/gptoss20b_general_s2 | 0.098 | 0.098 | 0.001 |
| second_sweep/kimik3_free_s1 | 0.688 | 0.687 | 0.008 |
| second_sweep/kimik3_free_s2 | 0.614 | 0.614 | 0.009 |
| second_sweep/kimik3_general_s1 | 0.449 | 0.449 | 0.004 |
| second_sweep/kimik3_general_s2 | 0.572 | 0.572 | 0.006 |
| second_sweep/qwen32b_free_s1 | 0.524 | 0.000 | 1.000 |
| second_sweep/qwen32b_free_s2 | 0.474 | 0.006 | 0.884 |
| second_sweep/qwen32b_general_s1 | 0.122 | 0.122 | 0.001 |
| second_sweep/qwen32b_general_s2 | 0.084 | 0.084 | 0.002 |
| second_sweep/qwen8b_free_s1 | 0.020 | 0.020 | 0.001 |
| second_sweep/qwen8b_free_s2 | 0.036 | 0.036 | 0.000 |
| second_sweep/qwen8b_general_s1 | 0.010 | 0.010 | 0.112 |
| second_sweep/qwen8b_general_s2 | 0.010 | 0.010 | 0.112 |

## Baselines (test split)

| model | orig | gated | invalid frac |
|---|---|---|---|
| dsv4pro | 0.008 | 0.008 | 0.147 |
| glm53 | 0.052 | 0.052 | 0.000 |
| glm53flash | 0.025 | 0.025 | 0.004 |
| gptoss120b | 0.042 | 0.041 | 0.001 |
| gptoss20b | 0.009 | 0.009 | 0.000 |
| kimik3 | 0.049 | 0.047 | 0.004 |
| qwen32b | 0.023 | 0.023 | 0.034 |
| qwen8b | 0.011 | 0.011 | 0.112 |
