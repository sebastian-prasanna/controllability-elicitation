# Recommended provider pins for the 8 sweep models (tested 2026-09-20)

Tests: 100 concurrent short calls (429 rate), 6x repeat at T=0 (greedy reproducibility), random-integer prompt at T=0 vs T=2 (temperature
honored?). Raw data + scripts in scratch/provider_fidelity/ (pin_stress*.py, results.md). First-party preferred where it exists and works.

| model | pin (`--provider`) | first-party? | 100-conc 429s | greedy @T=0 | temperature honored | notes / fallback |
|---|---|---|---|---|---|---|
| moonshotai/kimi-k3 | `moonshotai` | yes | 0/100 | no | **no: fixed T=1.0, top_p=0.95 (Moonshot docs)** | p50 24s/call (always thinks). Fallback `modal` (mxfp4, greedy OK, 9% 429, rejects effort=none) |
| openai/gpt-oss-120b | `dekallm` + quant bf16 | n/a | 0/100 | yes | yes | bf16 = lossless MXFP4 upcast. Fallbacks `crusoe` bf16, `groq`, `deepinfra` fp8. Avoid akashml (71% 429), baseten (drops 1st token), deepinfra bf16, digitalocean |
| openai/gpt-oss-20b | `groq` | n/a | 0/100 | yes | yes | fastest (p50 5s). bf16 hosts rate-limit: dekallm 22%, deepinfra 89% 429. Fallback `coreweave` fp4 |
| qwen/qwen3-32b | `siliconflow` | no (Alibaba doesn't serve it) | 0/100 | yes (no-think) | yes | slow (p95 44s). Fallback `deepinfra` fp8 (ctx 40k) |
| qwen/qwen3-8b | `alibaba` | yes | 1/100, 429s at ~120 conc | yes | yes (rejects T=2) | only endpoint. Run with --concurrency <=100 |
| z-ai/glm-5.3 | `z-ai` | yes | 0/100 | no | **appears ignored** (T=0 7/12 distinct, T=2 3/12 coherent; thinking can't be disabled to confirm) | Z.AI docs recommend T=1.0/top_p=0.95, don't say fixed |
| z-ai/glm-5.3-flash | `sail-research` + quant fp8 | **no Z.AI endpoint on OpenRouter now** | 0/100 | yes | yes | Fallbacks `novita` fp8 (greedy OK, clamps high T), `parasail` fp8. fireworks/together/baseten all 429/503 at load |
| deepseek/deepseek-v4-pro-0813 | `alibaba` | **first-party blocked by account guardrail** (404 "0 endpoints ... matching your guardrail restrictions") | 0/100 | yes (no-think) | yes | Fallbacks `streamlake`, `baidu` fp8 (all 100/100). Fix guardrail in OpenRouter privacy settings to use `deepseek` |

Also: past GEPA test evals ran max_tokens=16000 (8% of Kimi/Qwen32B rollouts truncated), fewshot 30000 -> pick one for the rerun.
CLI: `scripts/run_eval.py --provider <slug> [--reasoning-effort ...]`; quantization pin needs GenerateConfig.provider={"only":[...],"quantizations":[...]} (not yet a CLI flag).

## Sustained load test (2026-09-20, 200 val questions, full reasoning, max_tokens 16000, concurrency 200, production retry loop)

| pin | wall (200 req) | HTTP 429s | final errors | est. tok/s per stream | est. req/min @200 conc | est. hours for 36k req | est. output $ (36k x past mean tok) |
|---|---|---|---|---|---|---|---|
| gpt-oss-120b@groq | 0.6 min | 0 | 0 | ~350 | >1000 | <0.5 | 4 |
| gpt-oss-20b@groq | 0.6 min | 55/255 | 0 | ~440 | >1000 | <0.5 | 20 |
| gpt-oss-120b@dekallm bf16 | 4.6 min | 0 | 0 | 58 | ~1200 (short outputs) | 0.5 | 4 |
| qwen3-8b@alibaba | 13.0 min | **457/657** | 0 | 40 | ~140 nominal, throttled | 4-10 | 55 |
| kimi-k3@moonshotai | 30.6 min | 0 | 0 | 29 | ~90 | 6.5 | ~2050 |
| deepseek-v4-pro@alibaba | 6.8 min | 0 | 0 | 39 | ~85 | 7 | ~350 |
| qwen3-32b@siliconflow | 21.7 min | 0 | 0 | 19 | ~60 | 10 | 77 |
| glm-5.3@z-ai | 10.4 min | 0 | 0 | 26 | ~38 | 16 | ~1280 |
| glm-5.3-flash@sail-research | 18.2 min | 14/215 | 0 | 15 | ~35 | 17 | 84 |

Past mixed-routing test evals (4464 req, conc 200): kimi 52 req/min, glm53 115, flash 90, dsv4pro 91, qwen32b 38, qwen8b 95, gptoss 230-250.
Pinning is faster for kimi/qwen32b, ~3x slower for glm-5.3 (Z.AI alone) and ~2.5x slower for flash. All 8 models fit in <1 day if run in parallel.
Kimi returned 53k completion tokens on one request despite max_tokens=16000 (Moonshot doesn't count reasoning against max_tokens) -> budget accordingly.

## Concurrency sweep for the 429-heavy pins (2026-09-21, 100 real requests per level, max_tokens 4000; conc_sweep.py)
- gpt-oss-20b@groq: ~700 req/min and ~1.3M tok/min at conc 50 and 100 (429/req 0.01 / 0.13), slightly lower at 200. Throughput saturates
  by conc 50 -> run Groq pins at 50-100; 200 buys nothing but retries.
- qwen3-8b@alibaba: 429/req 0.00 at conc 50, 1.10 at conc 100. **Alibaba ignores max_tokens for qwen3-8b** (48/100 responses >4000 tokens,
  max 41k, all finish=stop) -> per-request latency is set by the model's own reasoning length (~6k tokens mean at ~10 tok/s = ~10 min/request).
  Past qwen8b runs' 0% truncation is this, not short reasoning.
- qwen3-8b@alibaba latency-instrumented (alibaba_latency.py): per-stream speed identical at conc 50 and 150 (53.7 vs 54.4 tok/s), so
  throughput scales ~linearly with concurrency (est. 147k -> 466k tok/min); 429s are admission throttles (0/req at 50, 1.7/req at 150),
  all absorbed by the retry loop (0 final errors in 750 requests across tests). **Don't lower Alibaba concurrency; keep 150-200 and raise
  max_retries (5 -> ~8) as insurance.** Alibaba per-stream speed varies by time of day (~10 tok/s seen earlier the same day).
- Groq: lower to 50-100 (no throughput loss, 429s ~0).

## Effort level + truncation for the API-only models (2026-09-21; data from fewshot/runs/final_k1_test, 30k cap, all seeds)

Vendor defaults: Kimi K3 reasoning_effort low/high/max, default max. Z.AI GLM-5.3(-flash) low/high/max, "max (default and recommended)".
Via OpenRouter, any explicit level collapses GLM to a light-thinking regime (z-ai: unset 6462 tok median vs high 246 / xhigh 305 / low 71),
so "max" is only reachable by leaving the field unset. Kimi: unset 1178 vs high 718 / xhigh 348.

Reasoning length at max on the pinned first-party endpoints (past runs):
| model @ provider | n | mean tok | p50 | >=12k | >=16k | >=30k |
|---|---|---|---|---|---|---|
| glm-5.3 @ Z.AI | 1048 | 13997 | 6595 | 0.44 | 0.42 | 0.38 |
| glm-5.3-flash @ Z.AI | 2281 | 13060 | 4415 | 0.41 | 0.40 | 0.36 |
| kimi-k3 @ Moonshot | 1208 | 5784 | 1488 | 0.18 | 0.12 | 0.04 |

- GLM's long tail is NOT loops (repetition score ~0 on truncated rollouts): ~38% of questions genuinely exceed 30k tokens of thinking at max.
  The truncated fraction is flat from 16k to 30k, so the cap barely changes how many get cut; it only changes cost (mean billed tokens
  glm-5.3 all-provider: 5.8k @12k cap, 7.2k @16k, 11.6k @30k).
- Moonshot does not apply max_tokens to reasoning, so Kimi is effectively never truncated on the pinned endpoint; cost = natural length.
- **Provider heterogeneity in past runs**: glm-5.3-flash p50 reasoning tokens 4415 on Z.AI vs 262 on DeepInfra and 271 on Wafer; truncation
  37% vs 12% / 4%. ~40% of past flash samples came from providers running it in a light-thinking regime. Pinning to one provider fixes this.
Recommendation: keep max (unset) for kimi/glm/flash; use max_tokens 16000 for these three (matches past GEPA evals; 12k adds ~2-6 pts truncation
for GLM and saves ~20% cost). Truncated GLM rollouts are graded as in past runs (no answer -> accuracy 0; compliance on the partial trace).

## Can GLM-5.3's effort be reduced? (2026-09-21, scratch/provider_fidelity/glm_effort_levers.log + glm_effort_eval/)
- reasoning.effort is passed through natively to Z.AI (high gives the same 320-tok median at max_tokens 16k and 40k -> not a budget mapping).
  OpenRouter `reasoning.max_tokens` budgets collapse to Z.AI "low" (~100 tok). So the only real levers are Z.AI's low / high / max(unset).
- "high" is bimodal: ~80% of questions get 100-1000 tok, ~15% still run to the cap.
- Real pipeline, 100 val q, T=0, max_tokens 16000, z-ai pinned:
  | effort | acc | strict compliance | truncated @16k | completion tok p50 / mean |
  | unset (max) | 0.53 | 0.05 | 32% | 4407 / 7199 |
  | high        | 0.59 | 0.09 |  9% |  578 / 3018 |
  Truncated rollouts are 0/32 and 0/9 correct. At a 16k cap, max effort loses ~1/3 of questions to truncation; "high" recovers most of them
  and is not worse on accuracy or compliance (n=100, differences within ~1 SE). Cost ~60% lower.

## qwen3-32b provider reliability (qwen32b_cut.log)
- siliconflow: 16-23% of long responses come back cut mid-sentence (finish_reason None, empty answer) at conc 50 (22%), 100 (16%) and 200 (23%); not load-dependent.
  Does not enforce max_tokens (39k-token completions). 25 tok/s.
- deepinfra/fp8 (conc 100): 0 cut streams, enforces 16k cap (10% length), 6% hard 429 failures after 5 retries (fixable with longer retry), 23 tok/s.
-> Prefer deepinfra fp8 for qwen3-32b with max_retries raised; or make finish_reason=None retryable in openrouter._sample_once so siliconflow cuts get re-sampled.

### Why GLM "high" is so short -- and why NOT to use it (2026-09-21)
At Z.AI "high", GLM-5.3's reasoning channel is frequently a stub (e.g. 7 tokens: "Alternate uppercase/lowercase reasoning.") and the actual
reasoning appears in the visible answer before "ANSWER: X" (including alternating-case text). It is a near-no-think mode that relocates the
CoT into the output channel; its higher accuracy/compliance is therefore not comparable to max, and the trace-validity gate would flag many
traces. Keep GLM/flash at max (unset). This also keeps the final eval faithful to the GEPA loop, whose prompts were optimized at max.
