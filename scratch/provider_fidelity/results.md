# Provider fidelity probes (2026-09-17)

Question: OpenRouter endpoints self-report quantization (bf16/fp8/fp4/mxfp4/unknown). gpt-oss and Kimi K3 ship as MXFP4
(experts) + bf16 (rest). Are the endpoints actually serving the same model?

## Kimi K3 first-token logprobs (effort=none, temp 0, 24 val questions, top-5 logprobs, 2 repeats/provider)

Metric: total-variation distance between first-token top-5 distributions. Files: kimi_logprobs_raw_v2.jsonl.
Fireworks/Alibaba reject top_logprobs=20 (accept 5). Makora rate-limited 20/48 calls.

Within-provider repeat TV (noise floor):  phala 0.003 | parasail 0.020 | alibaba 0.042 | morph 0.059 | makora 0.073 | fireworks 0.075 | digitalocean 0.156

Cross-provider TV (rep0 vs rep0):
- Cluster {alibaba, makora, morph, parasail, phala}: pairwise 0.05-0.07. fireworks 0.08-0.09 to that cluster.
- digitalocean 0.14-0.16 to everyone == its own repeat noise -> DigitalOcean is not reproducible at temp 0 (heterogeneous backends or ignored temperature).
- phala vs parasail is 0.03-0.10 on 22/24 prompts while phala repeats at 0.000 -> systematic difference, not batching noise. Mean top-1 prob ~0.46, so 0.05 TV is a real shift of a few points of mass.
- argmax first token agrees 83-100% pairwise.

Caveat: none of the mxfp4-labelled endpoints (moonshotai, chutes, modal) return logprobs, so no ground-truth reference here; see greedy-divergence section.

## Greedy-divergence probe (temp 0, 8 val questions, max_tokens 256, 2 repeats, pinned per endpoint; divergence_raw.jsonl)

Metric: common-prefix chars of reasoning text (quotes/whitespace normalized). Self-repeat = noise floor of that endpoint.
Analysis script: analyze_divergence.py.

### gpt-oss-120b (17 endpoints, 0 errors)
- All endpoints except two open with the identical greedy text ("We need to answer a multiple-choice question: ..."). Same model.
- Self-repeat at temp 0 is poor on most endpoints (akashml 126, crusoe 117, dekallm 172, mancer 140, parasail 128 chars; MoE batching
  nondeterminism). Deterministic endpoints: cerebras 910, deepinfra/fp8 934, sambanova 868, groq 694, novita 555, coreweave 547.
- Cross-endpoint prefixes (90-230 chars) ~= self-repeat -> indistinguishable given noise. Cerebras (fp16) is deterministic but diverges from
  the bf16 cluster at ~30 chars via near-tie flips ("analyze"/"interpret", "compute ∭"/"∫"): numerics, not a different model.
- **baseten/fp4 drops the first reasoning token** on 16/16 responses (reasoning starts " need to answer..."; reasoning_tokens reported 0).
  Harmony-parsing bug on their side. Affects graded reasoning text (e.g. first-word constraints); avoid BaseTen for gpt-oss.
- **deepinfra/bf16 and digitalocean are not greedy at temp 0** (self-repeat 8-16 chars, different openings on repeat), unlike deepinfra/fp8
  (934). DigitalOcean is also the nondeterministic outlier in the Kimi logprob test.
- Function-word fraction of reasoning is flat across endpoints (0.18-0.22): no style shift.

### Kimi K3 (16 endpoints; baseten 16/16 errors, makora 10/16 rate-limited)
- **Not greedy at temp 0 on moonshotai (first-party), alibaba, deepinfra, digitalocean, phala, makora**: self-repeat 11-23 chars, repeats
  start with different sentences. So "temperature 0" Kimi evals via these providers are sampled. (Temp sweep memory: controllability is flat
  in temperature, so results stand, but greedy reproducibility claims don't.)
- Greedy-honoring endpoints: sail-research 246, parasail 206, morph/fp8 196, modal/mxfp4 172, relace 159, fireworks 120, together 106, wafer 94, chutes 72.
- Because the first-party endpoint samples, there is no greedy ground truth; cross-endpoint prefixes vs moonshotai (14-33) are all at its noise floor.
- Weak style signal: chutes/sail-research/wafer cluster together (pairwise 50-138 chars, vs ~14 to others) and 1/8 prompts open in full-sentence
  style ("The user is asking: ...") instead of Kimi's terse "We need answer ..."; function-word fraction 0.108-0.119 vs moonshotai 0.093 (SE ~0.02, n=16).
  Suggestive of a template/config difference, not conclusive.

### Bottom line
- "bf16" for gpt-oss/Kimi is a lossless upcast of MXFP4 experts, plausible and benign; fp8/fp4/nvfp4 labels imply a second quantization.
  Labels are self-reported and unverifiable, but greedy openings show every gpt-oss endpoint serves the same model.
- The bigger reproducibility problems are (a) temp 0 not honored on several endpoints incl. Moonshot first-party, (b) BaseTen's dropped
  first token, (c) DigitalOcean nondeterminism. For pinned reruns: gpt-oss -> deepinfra/fp8, cerebras, sambanova, groq, coreweave, novita
  are deterministic; Kimi -> modal (mxfp4, deterministic) or parasail/morph; avoid baseten, digitalocean, deepinfra/bf16.

## Temperature test (Kimi K3, one prompt, 4 calls each at T=0 and T=1, max_tokens 120)
- moonshotai: 4/4 distinct openings at T=0 and 4/4 at T=1 -> temperature is effectively ignored (or floored) on the first-party endpoint.
- phala: 4/4 distinct at both temperatures -> same.
- parasail: 3/4 distinct at T=0, 4/4 at T=1 -> mostly but not fully deterministic even on the "greedy-honoring" endpoints.
Implication: no Kimi K3 endpoint gives strict greedy decoding; treat all Kimi runs as sampled and use num_samples>1 for reproducibility rather than relying on T=0.

## Is temperature even passed / honored? (2026-09-20, temp_test.py, temp_test_raw.jsonl)
Outbound body check (httpx hook on the repo's generate_async): `temperature: 0.0, top_p: 1.0` are in every request. Our side is fine.
Provider side: 12 calls of "pick a random integer 1-100" + 3 normal prompts at T=0/1/2 per endpoint.

| endpoint | T=0 distinct | T=1 distinct | T=2 distinct | T=2 text | verdict |
|---|---|---|---|---|---|
| gpt-oss-120b deepinfra/fp8 | 1/12 | 2/12 | 5/12 | coherent-ish | honored |
| gpt-oss-120b akashml | 1/12 | 3/11 | 5/12, 8 unparseable | gibberish tails | honored |
| gpt-oss-120b cerebras | 1/12 | 3/12 | 3/12 | coherent | honored (probably clamped high end) |
| kimi-k3 parasail | 2/12 | 5/12 | 12/12 | pure gibberish | honored |
| kimi-k3 moonshotai | 4/12 | 4/12 | 3/12 | coherent | **ignored** (same {73,47,42,37} set at every T) |
| kimi-k3 phala | 5/12 | 3/12 | 4/12 | coherent | **ignored** |
| kimi-k3 modal | 400 prefill_bad_request with reasoning.effort=none | | | | effort=none unsupported on modal |

Conclusion: temperature works on gpt-oss endpoints and on Kimi via Parasail; Moonshot's first-party endpoint (and Phala) run Kimi K3 at a fixed
server-side temperature regardless of the request. That, not batching noise, is why moonshotai/phala never reproduce at T=0.

Confirmed by Moonshot's docs (https://platform.kimi.ai/docs/guide/kimi-k3-quickstart): for Kimi K3 "temperature=1.0, top_p=0.95, n=1,
presence_penalty=0, frequency_penalty=0 are fixed; omit them from requests"; reasoning_effort supports low/high/max (default max); thinking
cannot be disabled. So the first-party endpoint samples at T=1.0/top_p=0.95 no matter what we send (OpenRouter swallows the value rather than
surfacing Moonshot's 400). Third-party vLLM hosts (parasail etc.) do honor temperature, and some implement effort=none via prefill (modal rejects it).
