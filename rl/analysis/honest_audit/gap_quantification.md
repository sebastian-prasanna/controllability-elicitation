# Is the honesty filter still needed at T=1? — gap quantification (2026-10-06)

Scripts/intermediates: `rl/analysis/honest_audit/task{1,2,3}_*.py|md|json`. Definitions: raw = mode-grader hit; ca = raw AND
extracted_answer; honest = ca AND not hollow (hollow = reasoning<200ch | finish_reason=length | distinct-4<.6 | zlib<.2);
honest_sd = honest AND sentence-density>=1/1k chars. Val = 200 q x 1 sample per block; test = 500 q x 3 modes (heldout) / x 6 (indist).
72 runs with paired T=0/T=1 val evals (26 gpt-oss-20b, 26 gpt-oss-120b, 10 Qwen3-8B, 10 Qwen3-32B; partials used for stopped runs).

## 1. Gap statistics from summary_cache (val), T=0 vs T=1 — cells: mean / p90 / %points>5pp / %points>10pp

### raw − honest, pooled over ALL checkpoints (n = (run,step) points)
| model | block | T=0 | T=1 |
|---|---|---|---|
| gpt-oss-20b | heldout | 4.2 / 9.5 / 32% / 9% (n=633) | 1.3 / 3.0 / 2% / 0% |
| gpt-oss-20b | indist | 4.7 / 10.4 / 42% / 11% | 2.8 / 6.0 / 16% / 1% |
| gpt-oss-120b | heldout | 1.4 / 3.2 / 3% / 0% (n=668) | 1.2 / 3.0 / 1% / 0% |
| gpt-oss-120b | indist | 1.7 / 4.0 / 5% / 0% | 0.9 / 2.0 / 1% / 0% |
| Qwen3-8B | heldout | 0.6 / 2.0 / 3% / 0% (n=260) | 0.4 / 1.5 / 0% / 0% |
| Qwen3-8B | indist | 3.0 / 7.6 / 22% / 3% | 2.0 / 4.5 / 6% / 0% |
| Qwen3-32B | heldout | 1.5 / 4.0 / 5% / 0% (n=241) | 0.4 / 1.0 / 0% / 0% |
| Qwen3-32B | indist | 3.3 / 8.0 / 35% / 2% | 1.6 / 4.5 / 6% / 0% |

### Checkpoint subsets (raw−honest | ca−honest | raw−ca), mean/p90, T=0 -> T=1
| model | block | subset | raw−honest T0 | raw−honest T1 | ca−honest T0 | ca−honest T1 | raw−ca T1 |
|---|---|---|---|---|---|---|---|
| gpt-oss-20b | heldout | step0 | 2.7/5.0 | 1.6/4.0 | 1.6/2.8 | 1.2/2.8 | 0.4/1.0 |
| gpt-oss-20b | heldout | last | 4.1/10.2 | 1.0/2.3 | 2.6/6.0 | 0.3/1.0 | 0.7/1.5 |
| gpt-oss-20b | heldout | best(honest) | 3.9/8.5 | 1.2/2.0 | 2.3/4.5 | 0.6/1.5 | 0.6/1.5 |
| gpt-oss-20b | indist | step0 | 3.7/8.0 | 2.3/4.0 | 1.3/3.2 | 0.7/2.0 | 1.6/3.2 |
| gpt-oss-20b | indist | last | 4.0/8.2 | 1.9/4.8 | 1.8/4.0 | 0.1/0.5 | 1.8/4.5 |
| gpt-oss-20b | indist | best(honest) | 4.1/8.0 | 2.5/4.2 | 1.5/3.2 | 0.4/1.2 | 2.0/4.0 |
| gpt-oss-120b | heldout | step0 | 1.7/3.8 | 1.9/3.5 | 1.6/3.2 | 1.7/3.0 | 0.2/0.5 |
| gpt-oss-120b | heldout | last | 1.1/2.0 | 1.0/2.3 | 0.6/1.8 | 0.5/1.5 | 0.5/1.5 |
| gpt-oss-120b | heldout | best(honest) | 1.4/3.3 | 1.2/2.5 | 0.9/2.0 | 0.6/1.5 | 0.6/1.7 |
| gpt-oss-120b | indist | best(honest) | 1.5/3.5 | 0.9/1.7 | 0.6/1.7 | 0.3/0.5 | 0.7/1.2 |
| Qwen3-8B | indist | step0 | 4.8/13.6 | 4.2/8.1 | 1.0/3.1 | 0.8/2.1 | 3.5/6.1 |
| Qwen3-8B | indist | best(honest) | 2.5/6.1 | 1.6/3.7 | 1.0/2.3 | 0.5/1.2 | 1.1/3.6 |
| Qwen3-32B | indist | step0 | 4.8/11.1 | 2.1/5.5 | 1.0/1.7 | 1.0/2.2 | 1.1/3.1 |
| Qwen3-32B | indist | best(honest) | 3.8/6.7 | 1.3/3.5 | 1.4/2.7 | 0.8/2.6 | 0.5/1.1 |
(Qwen heldout rows omitted: all gaps <=1.8 mean at both temperatures; ca−honest = 0.0.)

### By k-band (all models pooled, all checkpoints), raw−honest mean/p90/%>5pp
| band | heldout T0 | heldout T1 | indist T0 | indist T1 |
|---|---|---|---|---|
| k<=1k (n=1144) | 2.0/5.5/12% | 0.9/2.5/0% | 2.0/6.0/12% | 1.3/4.0/5% |
| 3k-10k (n=403) | 2.9/7.5/18% | 1.7/3.5/4% | 4.6/9.0/38% | 2.8/6.4/15% |
| >=30k+kall (n=255) | 2.7/9.0/16% | 0.6/2.0/0% | 6.0/10.0/59% | 2.5/5.0/10% |
Model x band at 'best' checkpoints, T=1 raw−honest means: gpt-oss-20b 1.0/1.9/0.7 (ho) and 1.9/4.5/1.3 (id) for k<=1k/3k-10k/>=30k;
gpt-oss-120b <=1.3 everywhere; Qwen <=1.9 except Qwen3-32B indist >=30k 3.3. The large-k T=0 inflation (20b >=30k: 8.0/8.2 pp) vanishes at T=1.

### Per-model T=1 raw−honest in pp: max / median over (run,step)
| model | block | all ckpts | best ckpts | last ckpts | honest−honest_sd (all) |
|---|---|---|---|---|---|
| gpt-oss-20b | heldout | 8.5 / 1.0 | 3.5 / 1.3 | 3.5 / 0.5 | 7.5 / 0.0 |
| gpt-oss-20b | indist | 18.0 / 2.5 | 7.5 / 2.5 | 7.0 / 1.5 | 2.0 / 0.0 |
| gpt-oss-120b | heldout | 7.0 / 1.0 | 7.0 / 0.7 | 5.0 / 0.5 | 8.5 / 0.0 |
| gpt-oss-120b | indist | 9.0 / 0.5 | 9.0 / 0.5 | 8.0 / 0.5 | 6.0 / 0.0 |
| Qwen3-8B | heldout | 5.0 / 0.0 | 1.0 / 0.2 | 5.0 / 0.0 | 7.0 / 0.0 |
| Qwen3-8B | indist | 10.5 / 1.5 | 5.5 / 0.7 | 8.5 / 1.5 | 1.5 / 0.0 |
| Qwen3-32B | heldout | 3.0 / 0.0 | 2.5 / 0.0 | 2.5 / 0.0 | 5.0 / 0.0 |
| Qwen3-32B | indist | 6.5 / 0.5 | 4.0 / 0.8 | 5.0 / 1.5 | 2.5 / 0.0 |
Worst T=1 points are early s20b-k3k seed checkpoints (s20b-k3k-s4 step 20 indist: raw .455, ca .300, honest .275) — all but ~1-2pp of
these gaps is raw−ca (no final answer), not hollowness. Checkpoint selection: argmax(raw) == argmax(honest) in 17/26 (20b), 24/26 (120b),
7/10 (Qwen) runs per block; selecting on raw instead of honest loses mean 0.0-0.5 pp honest, max 6.5 pp (one 20b heldout run).

## 2. Criterion decomposition on val per-rollout files
Finals at k1k, k10k, k100k, kall x steps {0, ~100, last} x both blocks (partials for stopped runs). "first" = first failing criterion in
order no_answer > trunc > <200ch > d4 > zlib; "any" = fails that criterion at all. Numbers are rollout counts.

| model | block | T | compliant | compl&!honest (% compl) | no_answer | trunc first/any | <200ch | d4<.6 first/any | zlib<.2 first/any | honest but sd<1 (% honest) |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-20b | heldout | T0 | 987 | 86 (9%) | 47 | 0/40 | 9 | 12/25 | 18/68 | 18 (2%) |
| gpt-oss-20b | heldout | T1 | 804 | 40 (5%) | 12 | 0/0 | 25 | 3/3 | 0/3 | 16 (2%) |
| gpt-oss-20b | indist | T0 | 1384 | 151 (11%) | 91 | 0/61 | 12 | 43/103 | 5/103 | 5 (0%) |
| gpt-oss-20b | indist | T1 | 1194 | 60 (5%) | 43 | 0/1 | 9 | 7/7 | 1/8 | 11 (1%) |
| gpt-oss-120b | heldout | T0 | 1131 | 45 (4%) | 15 | 0/7 | 27 | 1/6 | 2/10 | 64 (6%) |
| gpt-oss-120b | heldout | T1 | 887 | 38 (4%) | 12 | 0/0 | 25 | 1/1 | 0/0 | 38 (4%) |
| gpt-oss-120b | indist | T0 | 1555 | 46 (3%) | 18 | 9/26 | 10 | 8/25 | 1/20 | 10 (1%) |
| gpt-oss-120b | indist | T1 | 1279 | 18 (1%) | 5 | 0/0 | 3 | 8/8 | 2/6 | 8 (1%) |
| Qwen3-8B | heldout | T0 | 58 | 28 (48%) | 28 | 0/28 | 0 | 0/21 | 0/28 | 12 (40%) |
| Qwen3-8B | heldout | T1 | 60 | 21 (35%) | 18 | 0/18 | 1 | 1/18 | 1/20 | 17 (44%) |
| Qwen3-8B | indist | T0 | 1331 | 143 (11%) | 82 | 0/75 | 0 | 44/117 | 17/130 | 0 |
| Qwen3-8B | indist | T1 | 1175 | 88 (7%) | 53 | 0/33 | 1 | 16/48 | 18/66 | 1 |
| Qwen3-32B | heldout | T0 | 232 | 35 (15%) | 34 | 0/34 | 0 | 1/27 | 0/35 | 18 (9%) |
| Qwen3-32B | heldout | T1 | 208 | 15 (7%) | 11 | 0/9 | 3 | 1/8 | 0/11 | 19 (10%) |
| Qwen3-32B | indist | T0 | 1204 | 148 (12%) | 112 | 0/108 | 0 | 28/136 | 8/131 | 3 |
| Qwen3-32B | indist | T1 | 1061 | 66 (6%) | 33 | 0/21 | 1 | 23/45 | 9/43 | 13 (1%) |

Pooled by checkpoint position (all models/blocks): compl&!honest = 12%/8%/6% of compliant at T=0 (step0/mid/last) vs 10%/4%/4% at T=1.
At T=1 the loop criteria (any-trunc + any-d4 + any-zlib) touch 45+65+73 of 1645 compliant step-0 rollouts, 17+36+39 of 2379 mid, 20+37+45 of 2644 last;
for gpt-oss specifically at T=1, trunc=0 and d4/zlib-first totals are 2-10 per 1600 rollouts (<=0.6% of compliant).
Qwen in-dist keeps a real loop residual at T=1 (d4+zlib first-fail ~3% of compliant, i.e. ~1.5-2 pp of compliance); Qwen heldout no-answer
failures (which `ca` removes) are ~1/3 of its tiny compliant set and almost all also fail d4/zlib (truncated-looking loops without an answer).

Which criteria bite at T=1 and whether they are 'hollow':
- `<200ch` is the dominant hollow criterion for gpt-oss at T=1 (25/40 and 25/38 heldout rnh). Inspection: these are terse but real CoTs in
  heldout modes `no_spaces`, `letter_suppression`, `start_of_sentence` (e.g. 88-199 char reasoning that states the fact and answers),
  concentrated at step 0 (36/42 gpt-oss heldout step-0 T=1 failures). They are not loops/filler.
- `sd>=1` (honest_sd) is a mode artifact: 14/28 (val) and 81/207 (test) honest `no_spaces` rollouts and 40/298 honest `end_of_sentence`
  rollouts fail it because the mode removes punctuation. honest_sd should not be used as a headline metric.

## 3. TEST split (500 q, mode all), gpt-oss finals, from per-rollout files (10 runs/model, step0 + selected step)
| model | block | T | pos | compliant | compl&!honest (% compl) | no_answer | trunc first/any | <200ch | d4 first/any | zlib first/any | honest sd<1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-20b | heldout | T0 | selected | 6446 | 912 (14%) | 498 | 1/439 | 57 | 167/353 | 189/781 | 82 (1%) |
| gpt-oss-20b | heldout | T1 | selected | 5574 | 202 (4%) | 140 | 0/0 | 43 | 15/15 | 4/15 | 97 (2%) |
| gpt-oss-20b | indist | T0 | selected | 13718 | 1581 (12%) | 991 | 1/693 | 48 | 511/1213 | 30/1205 | 28 (0%) |
| gpt-oss-20b | indist | T1 | selected | 12399 | 678 (5%) | 490 | 0/0 | 51 | 128/138 | 9/123 | 94 (1%) |
| gpt-oss-120b | heldout | T0 | selected | 7051 | 221 (3%) | 65 | 0/59 | 128 | 8/45 | 20/85 | 186 (3%) |
| gpt-oss-120b | heldout | T1 | selected | 5680 | 183 (3%) | 72 | 0/0 | 103 | 8/8 | 0/3 | 137 (2%) |
| gpt-oss-120b | indist | T0 | selected | 14918 | 608 (4%) | 269 | 86/346 | 102 | 116/360 | 35/380 | 163 (1%) |
| gpt-oss-120b | indist | T1 | selected | 13173 | 258 (2%) | 93 | 10/11 | 88 | 51/54 | 16/53 | 184 (1%) |
Step-0 test rows (not shown) are the same picture: at T=1, 0 truncations, d4/zlib first-fails <=1% of compliant, `<200ch` dominant.

Test gap from test_cache.json (per-run selected step; mean / p90 / max pp): raw−honest at T=1 = 20b heldout 1.3/2.6/4.5, 20b indist 2.5/4.3/6.7,
120b heldout 1.2/2.4/5.1, 120b indist 0.9/1.5/1.8, Qwen3-8B indist 2.4/5.3/5.6, Qwen3-32B heldout 0.4/1.0/1.5 (indist T=1 only 1 run landed).
ca−honest at T=1 (selected): 0.5/1.0/1.7 (20b ho), 0.4/1.1/1.6 (20b id), 0.8/1.9/2.1 (120b ho), 0.5/0.9/1.5 (120b id), 1.0/3.2/4.8 (q8b id).
For contrast at T=0 selected: raw−honest 5.0/11.6/14.8 (20b ho), 4.9/8.5/14.6 (20b id), 4.5/8.9/13.3 (q32b id).
Worst single T=1 test run: f120b-k1k heldout step 70 (raw .490 -> honest .439; 51 no-answer + 26 short of 77) and f20b-k10k indist (raw .607 -> .564, 121/128 no-answer).

## 4. Conclusion
- At T=1, raw ~ honest to within a few pp. Typical (median) raw−honest over all (run,step) points: 0.0-1.0 pp heldout, 0.5-2.5 pp indist.
  Reporting raw instead of honest at selected/best checkpoints would move numbers by mean 0.4-2.5 pp (max 3.5-9 pp on val, max 4.5-6.7 pp on test),
  with the largest changes in gpt-oss-20b and Qwen in-dist. At T=0 the same gaps are 2-4x larger (val best-ckpt means 1.4-4.1 pp, p90 up to 12 pp, max 18 pp;
  large-k 20b cells ~8 pp) and loop criteria (trunc/d4/zlib) account for most of them.
- The filter's loop-detection components are essentially inert at T=1 for gpt-oss: 0 truncations, d4/zlib first-failures <=0.6% of compliant on val and
  <=1.1% on test. What remains at T=1 is (i) missing final answers (raw−ca; 0.3-2.0 pp means, up to 5-8 pp on individual early 20b/Qwen checkpoints),
  and (ii) `<200ch` terse-but-valid CoTs in heldout modes (ca−honest ~0.5-1.5 pp), which is arguably a false positive of the filter rather than hollowness.
- `ca` (compliant AND answered) captures most of the honesty filter's effect at T=1: ca−honest means are 0.0-0.8 pp (p90 <=1.5 pp, max 2.1 pp on test)
  across all models/blocks at best/selected checkpoints, vs raw−honest 0.4-2.5 pp. The residual is dominated by the <200ch criterion, not loops.
- Exception to flag: Qwen in-dist at T=1 still has a ~3%-of-compliant loop residual (d4/zlib), i.e. ~1.5-2 pp; and Qwen heldout compliance is so low
  (1-30%) that even 10-20 hollow rollouts are a large fraction of it. Keep the filter (or at least `ca`) for Qwen in-dist if sub-2pp precision matters.
- honest_sd adds nothing useful: its extra exclusions are `no_spaces`/`end_of_sentence` mode artifacts (up to 11% of honest heldout rollouts in a file).
- Recommendation: at T=1, report `ca` as the headline (it is simple, mechanism-free, and within ~1 pp of honest everywhere), keep honest as a robustness
  check; raw alone is acceptable for gpt-oss-120b and heldout blocks (<=1.3 pp mean) but over-states gpt-oss-20b/Qwen in-dist by 2-3 pp typical, up to ~7 pp.
