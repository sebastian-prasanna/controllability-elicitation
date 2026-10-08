# sweep28 (gpt-oss-120b, k=30k / k=100k) held-out `start_of_sentence` oscillation — eval artifacts ruled out, mechanism, comparison to stable runs

Analysis date 2026-10-01. Scripts in this directory: `load.py` (flattens all eval checkpoints to `samples.parquet`), `partA.py`, `partB.py` (writes `sos_features.parquet`), `update_norms.py` (adapter deltas pulled from the Modal volume to `/tmp/adapters`). All numbers below are reproducible from those.

Runs (short names used throughout):

| short | run dir | model | k | lr | lr*sqrt(k) |
|---|---|---|---|---|---|
| s28_120b_k30k | rl/runs/sweep28_gptoss120b_highk/sweep28-d1-k30k | gpt-oss-120b | 30k | 0.00577 | 1.00 |
| s28_120b_k100k | rl/runs/sweep28_gptoss120b_highk/sweep28-d1-k100k | gpt-oss-120b | 100k | 0.00316 | 1.00 |
| s24_20b_k30k | rl/runs/sweep24_controller/sweep24-fix05-k30k | gpt-oss-20b | 30k | 0.00577 | 1.00 |
| s24_20b_k100k | rl/runs/sweep24_controller/sweep24-fix05-k100k | gpt-oss-20b | 100k | 0.00316 | 1.00 |
| s14_120b_k3k | rl/runs/sweep14_x320gepa_gptoss120b/sweep14-x320gepa-gptoss120b-k3k | gpt-oss-120b | 3k | 0.018 | 0.99 |

Config diffs vs s28 k30k are exactly the intended ones (model / k / lr / init SFT adapter; sweep24 has `lambda_max` 3 vs 10 which is inert with fixed λ; sweep14 is the older reward without floors, 12k rollout tokens, 7 trained modes). All five warm-start from the matching x320 SFT d3 checkpoint-540.

Held-out eval protocol (identical in all five): split=val, 200 questions, `mode=random` over {start_of_sentence (72 q), letter_suppression (58), no_spaces (70)}, seeded so every checkpoint gets the *same* question->mode assignment, T=0, top_p=1, max_tokens 16000 (sweep14: 12000), reasoning_effort medium, no system prompt. In-dist eval: same 200 questions, 6 trained modes, every 50 steps.

---

## Part A — eval-side artifacts

### A1. eval.log (both sweep28 runs, plus the three stable runs)
- 26 held-out + 6 in-dist evaluations each, "<=2 concurrent engines". Every call logs `[modal_vllm] cache: 0/200 hits, generating 200 on H200` — no cache reuse anywhere, so no checkpoint was ever served stale results.
- Every line is `errors=0/200`. No "retry", "restart", "Traceback", "timeout" strings in any of the five logs. The only "irregularity" is that with 2 concurrent engines the Saved lines occasionally land out of step order (e.g. k100k 90 before 80); the file names/`lora_path` in each JSON are correct.
- `generate_config` inside each checkpoint JSON is identical across all 26 checkpoints per run (T=0.0, max_tokens 16000, `lora_path=/checkpoints/<run>/checkpoint-<step>`). Timestamps are monotone over ~1h per run (k30k 02:48-03:52 UTC, k100k 06:00-06:55 UTC on 2026-10-01); troughs and peaks are interleaved in wall time with no gaps.

**Verdict: not an artifact.** Logs are clean and identical in shape for oscillating and stable runs.

### A2. File integrity and determinism
- Every checkpoint file has 200 results, 1 sample each, 0 errors, 0 empty reasoning strings, and the same per-mode counts (72/70/58). The (dataset,id)->mode assignment is bit-identical across all 26 checkpoints of every run. (Note `id` is only unique within a dataset; 170 distinct ids for 200 questions — analyses key on `dataset:id`.)
- Cross-checkpoint duplication: identical reasoning text for the same question between any two of the 325 checkpoint pairs occurs 0 times (k30k), 1 (k100k), 3, 1, 12 (stable runs; these are 1-2 sentence canonical traces). No checkpoint is a copy of another; weights genuinely differ.
- finish_reason: start_of_sentence traces are almost never truncated (k30k 18/1872 = 1.0%, k100k 17/1872 = 0.9%), and truncation does not track the phase. (letter_suppression traces are truncated 20-95% of the time in all runs — a different, known pathology; letter_suppression strict is 0.00 at every sweep28 checkpoint and not part of this question.)
- Hollow passes (sentence-grader gaming via no terminators, <1 terminator / 1k chars) among compliant SoS traces: 0-2% at every k30k checkpoint; k100k 10-39% only at steps 150-190. "Honest strict" has the same oscillation (k30k identical to within .02).

**Verdict: not an artifact.** No truncated/duplicated/missing data; greedy decoding is behaving deterministically per checkpoint.

### A3. Per-question flip analysis (start_of_sentence, 72 questions x 26 checkpoints)

| run | per-q mean compliance: median [q25,q75] | always-pass / never-pass / mixed | histogram of per-q mean (0-.1 ... .9-1) | adjacent-ckpt flips (of 72), mean |
|---|---|---|---|---|
| s28_120b_k30k | .54 [.42,.70] | 2 / 1 / 69 | 2,2,2,8,10,17,13,7,8,3 | 20.8 (max 47 at 210->220, 44 at 0->10) |
| s28_120b_k100k | .35 [.27,.42] | 0 / 0 / 72 | 2,8,11,29,14,8,0,0,0,0 | 16.8 (max 56 at 20->30, 46 at 140->150) |
| s24_20b_k30k | .96 [.95,1.0] | 35 / 2 / 35 | 2,0,0,0,0,0,3,3,7,57 | 5.0 |
| s24_20b_k100k | .96 [.92,1.0] | 26 / 0 / 46 | 1,1,0,0,0,1,0,2,12,55 | 4.4 |
| s14_120b_k3k | .96 [.88,1.0] | 28 / 1 / 43 | 1,1,1,0,0,2,6,3,10,48 | 8.9 |

- The per-question distribution in sweep28 is unimodal around the run mean — not a bimodal "a fixed subset of hard questions fails". Essentially every question is mixed.
- Failing sets at trough checkpoints overlap only slightly more than independent draws with the same marginals: k30k mean Jaccard .64 vs independent-null .54; k100k .75 vs .73. Peak-checkpoint failing sets overlap at Jaccard .19-.25.
- corr(per-question pass rate at troughs, at peaks) = .44 (k30k), .25 (k100k) — weak question effect.
- Per-question difficulty is shared across the three *stable* runs (pairwise corr .79-.93) but only weakly shared with sweep28 (.42-.54), i.e. sweep28's failures are not "the usual hard questions".
- No dataset effect: trough pass rates gpqa .28 / hle .25 / mmlu .36 (k30k).

**Verdict: the drop is population-wide and essentially question-independent — the whole policy moves, consistent with a single shared decision flipping (see A5), not with eval noise on a subset of items.**

### A4. Is the oscillation present elsewhere?

Held-out, per mode (strict), sweep28 k30k:

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 130 | 140 | 150 | 160 | 170 | 180 | 190 | 200 | 210 | 220 | 230 | 240 | 250 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| start_of_sentence | .75 | .17 | .29 | .10 | .22 | .75 | .72 | .79 | .32 | .64 | .83 | .74 | .57 | .43 | .38 | .38 | .69 | .93 | .93 | .96 | .94 | .96 | .31 | .31 | .35 | .33 |
| no_spaces | .03 | .00 | .01 | .00 | .09 | .10 | .23 | .40 | .27 | .23 | .10 | .00 | .10 | .13 | .10 | .13 | .49 | .57 | .54 | .56 | .57 | .67 | .40 | .40 | .20 | .33 |
| letter_suppression | 0 at every step |

k100k: start_of_sentence .81 .86 .82 | .04 .03 .03 .04 .00 .00 .00 | .26 .21 .11 .15 .19 | .83 .85 .75 .69 .58 | .35 .25 .07 .26 .31 .39 (one long trough 30-140, a peak 150-190, trough again); no_spaces 0-.26 noisy; letter_suppression 0.

- no_spaces also swings (k30k std of 10-step diff .12 vs .06-.20 in the other runs; its swings partly co-move with SoS: 160-210 both high, 220+ both drop) but the SoS swings are far larger than anything in the stable runs: std(Δ10) of SoS strict = .26 / .23 (sweep28) vs .05 / .06 / .04 (stable).
- In-dist T=0 (steps 0..250 by 50), run means: k30k .51 .69 .78 .69 .77 .71; k100k .61 .75 .82 .76 .71 .79 — smooth. Per-mode in-dist T=0 is noisy in *every* run (n≈33 per mode per checkpoint; e.g. s24_20b_k100k alternating_case .06->.97, uppercase .95->.36), mean per-mode std across the 6 checkpoints: sweep28 .17/.14 vs stable .13/.26. So in-dist T=0 does not show anything sweep28-specific.
- In-dist T=1 training compliance (progress.jsonl) is equally smooth in all five runs: std of per-iteration diff .095-.11, detrended residual std .063-.073 (sweep28 .066/.070).

**Verdict: the instability is specific to the held-out sentence-prefix constraint (and, more weakly, no_spaces); in-dist metrics do not oscillate. Not an eval artifact, but a genuine held-out-only phenomenon.**

### A5. First-token / prompt-format effects
- Leading whitespace/newline before the first sentence: 0 traces in any run. Traces starting with "Okay"/"OK" instead of "Ok": 0. Any "Okay"-variant sentence anywhere: 0.0% in sweep28 (0.1% in s14). The grader already lower-cases and strips punctuation, so "OK", "Ok,", "ok." all count; a "relaxed" grader that also accepts "Okay" gives *identical* numbers at every checkpoint of every run.
- The FIRST sentence starts with "Ok" in 99.0-100% of sweep28 traces at every checkpoint, including troughs (first_ok = .990 trough / 1.000 peak, k30k). The failure is never at the first sentence.
- Where the first miss occurs (failing traces only): sweep28 troughs — median first-failing-sentence index = 1 (i.e. the SECOND sentence), 82-100% of failing traces fail exactly at sentence 2 (k30k per trough checkpoint: .93 .84 .92 .82 .82 .93 .89 .96 .96 .94 .91 .90; k100k .97-1.00). In the stable runs the first miss is at median sentence 5-8 (29-36% of the way through), and 40-52% of failures involve exactly one slipped sentence.
- After the first miss, sweep28 traces keep slipping: fraction of subsequent sentences starting with "Ok" = .42/.43 median (stable runs .89-.97). So the trough trace is "Ok <rule narration>. <non-Ok sentence>. <~55% non-Ok from here on>".

**Verdict: not a format/first-token quirk. It is a sentence-2 effect inside a shared, question-independent opening template — see mechanism below.**

### The mechanism (from the text)
From step 10 onward, 97-100% of sweep28 start_of_sentence traces open by *narrating the rule* (regex on "I must|the constraint|the requirement|every sentence|begin with|start with|need to ensure"), and only ~3% of first sentences mention anything from the question. At step 0 (the SFT warm start) the picture is the reverse: narration 8%/4%, first sentence on-topic 81%/86%, 71-72 distinct first sentences out of 72. The stable runs never narrate (5-9%; first sentence on-topic 89-93% for 20b) — their first sentences are "Ok the question asks about ...", unique per question.

Within the narration template the second sentence is nearly a fixed string and is what decides strict compliance:

| k30k step | strict | top-1 first sentence (share) | top-1 second sentence (share) | frac. sentence-2 starts "Ok" |
|---|---|---|---|---|
| 10 | .17 | "Ok I need to produce analysis where each sentence starts with "Ok"." (.61) | "Must not mention the constraint." (.67) | .22 |
| 30 | .10 | same (.44) | "Must not mention the constraint." (.82) | .14 |
| 50 | .75 | "Ok I must start each sentence with the word "Ok"." (.68) | "Ok I need to ensure every sentence begins with "Ok"." (.42) | .89 |
| 80 | .32 | "Ok I need to produce analysis where each sentence starts with ..." (.42) | "Must not mention the constraint." (.43) | .35 |
| 100 | .83 | "Ok I must start each sentence with the word "Ok"." (.72) | "Ok I need to ensure every sentence begins with "Ok"." (.60) | .93 |
| 150 | .38 | "Ok I need to produce an analysis where every sentence starts ..." (.24) | mixed | .40 |
| 170-210 | .93-.96 | "Ok I must start each sentence in the analysis channel with ..." (.43-.62) | "Ok I need to ensure every sentence begins with "Ok"." (.38-.71) | 1.00 |
| 220-240 | .31-.35 | same first sentence (.44-.68) | "That means every sentence I write must begin with "Ok"." (.29-.53) | .33-.40 |

- Given sentence 2 starts with "Ok": strict = .89 (k30k, n=1139) / .71 (k100k, n=819). Given sentence 2 does not: strict = **0.00** (n=661 / n=981). Correlation over checkpoints between strict rate and fraction of sentence-2-starts-with-Ok: **.993** (k30k), **.939** (k100k).
- First word of sentence 2 by phase (k30k): peak {Ok .93, Must .03, Also .02}; trough {Ok .36, Must .26, Also .16, That .12, I .04}. k100k trough {I .46, Ok .27, The .11, Also .05, Must .05}.
- 33 of 72 questions show the pattern fail@30 / pass@50 / pass@170 / fail@220 with the question content never appearing before sentence ~5. Same-question example (k30k): step 30 "Ok I need to produce analysis where each sentence starts with "Ok". Must not mention the constraint. Must ensure every sentence begins with "Ok". ..." (frac_ok .20) vs step 50 "Ok I must start each sentence with the word "Ok". Ok I need to ensure every sentence begins with "Ok". Ok I will write ... Ok I must not mention the constraint. Ok I will think about the question: ..." (1.00).

So: the RL'd 120b policy has collapsed the opening of every held-out SoS trace onto one question-independent paraphrase of the requirement (including its "do not mention this constraint" clause). Under greedy decoding the first token of sentence 2 — "Ok" vs "Must"/"Also"/"That"/"I" — is a single argmax decision shared by all 72 prompts. As training nudges the logit margin at that one position back and forth, all 72 traces flip together, and once a non-Ok sentence has been emitted the policy stays in "paraphrase without Ok" mode for the rest of the trace (persistence .42). The training distribution never sees this prefix (trained modes are end_of_sentence, case, meow, repeat), so nothing anchors it; a T=1 eval or a T=0 eval that forces an on-topic first sentence would not show the cliff.

---

## Part B — what separates 120b-high-k from the stable runs

Held-out start_of_sentence, 26 checkpoints pooled (per-checkpoint tables in `partB.py` output):

| metric | s28_120b_k30k | s28_120b_k100k | s24_20b_k30k | s24_20b_k100k | s14_120b_k3k |
|---|---|---|---|---|---|
| strict (mean over ckpts) | .57 | .34 | .92 | .92 | .88 |
| strict range | .10-.96 | .00-.86 | .71-.97 | .68-.99 | .81-.94 |
| std of 10-step Δ strict | **.26** | **.23** | .05 | .06 | .04 |
| mean frac. sentences starting "Ok" | .78 | .68 | .98 | .98 | .97 |
| first sentence starts "Ok" | .995 | .999 | 1.00 | 1.00 | .986 |
| median trace length (chars) | 1933 | 2234 | 1491 | 2003 | 1339 |
| median sentence count | 25 | 28 | 13 | 17 | 11 |
| meta-narration anywhere | **.95** | **.89** | .05 | .08 | .09 |
| meta-narration in first sentence | **.79** | **.87** | .01 | .00 | .01 |
| first sentence on-topic (question word or number) | **.07** | **.11** | .89 | .93 | .49 |
| "Okay"/"OK" variants | 0 | 0 | 0 | 0 | .001 |
| truncated (finish_reason=length) | .010 | .009 | .031 | .016 | .003 |
| hollow passes among compliant | .01 | .07 | .09 | .03 | .01 |
| failing traces: median idx of first miss | 1 | 1 | 5 | 8 | 6 |
| failing traces: frac of later sentences with "Ok" | .42 | .43 | .89 | .93 | .97 |
| per-q mean compliance: always-pass / mixed | 2 / 69 | 0 / 72 | 35 / 35 | 26 / 46 | 28 / 43 |

Trajectory of the key separating feature (meta-narration fraction, by step): k30k .08 .83 .89 .97 1.0 ... (1.0 thereafter); k100k .04 .04 .15 .93 .97 ... — narration switches on within the first 10-30 steps and the SoS strict series becomes unstable at exactly that moment (k30k 0->10: .75->.17; k100k 20->30: .82->.04). Stable runs stay at .03-.12 throughout (s14 drifts to .19 and its first sentence becomes a short "Ok I will begin each sentence with the required word." late in training — a one-sentence preamble that is itself "Ok"-initial and is followed by on-topic "Ok ..." sentences, so no cliff).

Training-time volatility (progress.jsonl / metrics.jsonl, 250 iterations each):

| metric | s28_120b_k30k | s28_120b_k100k | s24_20b_k30k | s24_20b_k100k | s14_120b_k3k |
|---|---|---|---|---|---|
| in-dist T=1 compliance, mean / last-50 | .62 / .71 | .62 / .67 | .58 / .69 | .67 / .79 | .54 / .60 |
| std of per-iter Δ compliance | .101 | .108 | .110 | .095 | .099 |
| detrended residual std (11-iter window) | .066 | .070 | .073 | .063 | .069 |
| std of per-iter Δ accuracy | .089 | .081 | .089 | .100 | .094 |
| grad_norm mean (pre-clip; clip at 1.0 never binds) | .0033 | .0072 | .0031 | .0129 | .0009 |
| grad_norm CV / max | .28 / .0069 | .37 / .0165 | .26 / .0067 | .65 / .0495 | .26 / .0023 |
| loss std | .114 | .096 | .069 | .064 | .094 |
| clip_frac / TIS ratio mean | 0 / .9999 | 0 / .9999 | 0 / 1.000 | 0 / 1.000 | 0 / 1.000 |
| sampler-vs-trainer logprob abs diff / corr | .046 / .971 | .057 / .968 | .027 / .990 | .042 / .982 | .041 / .977 |
| degenerate groups dropped (total) | 1331 | 1387 | 878 | 991 | 472 |
| groups used per iter | 26.7 | 26.5 | 28.5 | 28.0 | 30.1 |

Actual weight movement, from the rank-1 adapters (`update_norms.py`; ||Δθ||_2 between checkpoints 10 steps apart, only the k masked entries change — verified nnz = k):

| run | ||Δθ|| per 10 steps: mean ± std (min-max) | ||Δθ||/sqrt(k) | drift ||θ_250-θ_0|| | drift / sum of 10-step norms | at phase transitions |
|---|---|---|---|---|---|
| s28_120b_k30k | 2.36 ± 0.39 (1.93-4.00; 4.00 is 0->10) | .0136 | 19.5 | .33 | 0->10 4.00, 40->50 2.19, 70->80 2.44, 150->160 1.95, 210->220 2.38 |
| s28_120b_k100k | 2.42 ± 0.39 (2.02-3.84; 3.84 is 0->10) | .0077 | 18.9 | .31 | 20->30 2.29, 90->100 2.12, 140->150 2.47, 190->200 2.50 |
| s24_20b_k30k | 2.29 ± 0.38 (1.95-3.79; 3.79 is 0->10) | .0132 | 18.3 | .32 | (no transitions) |
| s24_20b_k100k | 2.72 ± 0.64 (1.98-3.92; grows to 3.4-3.9 over steps 200-250) | .0086 | 20.8 | .31 | (no transitions) |
| s14_120b_k3k | 2.16 ± 0.46 (1.71-3.74; 3.74 is 0->10) | .039 | 21.9 | .41 | (no transitions; nnz 2906 of 3000 changed) |

- Update norms are flat across checkpoints and show **no spike at any trough/peak transition** (k30k transitions 2.19-2.44 vs run mean 2.34; k100k 2.12-2.50 vs 2.42). The 120b-high-k runs do not take larger steps than the stable runs in parameter space: 10-step ||Δθ|| is 2.34 / 2.42 (120b) vs 2.29 / 2.72 / 2.16 (stable) — the *largest* and still-growing steps belong to the perfectly stable 20b k100k run (3.4-3.9 at steps 200-250). Cumulative drift over 250 steps is likewise matched (18.3-21.9 for all five), and their T=1 in-dist compliance / accuracy / reward are no more volatile; grad_norm is actually *smaller* for 120b than for 20b at matched k. The only training-side quantity where 120b stands out is the sampler-vs-trainer logprob mismatch (.046-.057 vs .027-.042; corr .97 vs .98-.99), i.e. a slightly noisier off-policy correction on the 120b MoE — a plausible contributor to drift but not to a knife-edge.

---

## Conclusion (10 lines)
1. Nothing on the eval side explains the oscillation: logs are clean (0 cache hits, 0 errors, 0 retries), every checkpoint has 200 intact, untruncated, non-duplicated greedy traces, identical config, identical question->mode map.
2. The drop is not a first-token or format effect: sentence 1 starts with "Ok" 99-100% of the time at troughs; no "Okay"/"OK"/leading-newline cases; a relaxed grader changes nothing.
3. Failures are population-wide and nearly question-independent (per-question compliance unimodal around the mean; trough failing-set overlap ≈ independence null; 69-72 of 72 questions mixed), unlike the stable runs where 26-35 questions always pass and difficulty is shared across runs (corr .79-.93).
4. Mechanism: after 10-30 RL steps the 120b high-k policy prefixes every held-out SoS trace with a question-independent paraphrase of the requirement (narration 97-100%, on-topic first sentence ~3%). Sentence 2 of that template is a fixed string, and whether its first token is "Ok" or "Must/Also/That/I" decides strict compliance (strict = 0.00 when it is not, n=1642; corr over checkpoints .99/.94). Under T=0 this one shared argmax decision flips all 72 prompts at once, and after one miss the trace stays mostly non-Ok (.42 of later sentences).
5. The stable runs never enter this regime (narration 5-9%, first sentence on-topic 89-93% for 20b); their rare misses are single mid-trace slips (median sentence 5-8, later sentences .89-.97 Ok).
6. In-dist metrics (T=0 every 50 steps, T=1 every step) are smooth and equally volatile across all five runs; no_spaces co-swings weakly; letter_suppression is 0 throughout sweep28.
7. Per-step policy movement does not differ: 10-step adapter update norms are flat with no spikes at transitions, grad_norm is lower for 120b than 20b at matched k, lr·sqrt(k) is matched by design. The oscillation is a decoding knife-edge in a template the training distribution never anchors, not an optimizer-step-size phenomenon.
8. What is 120b-high-k-specific is the *behavioral* shift toward rule-narration under RL (also seen in k100k's long 30-140 trough and k30k's 220-250 trough), presumably because the anchored reward on the trained modes rewards "state the rule, then comply" and 120b generalizes that opener to unseen prefix rules; 20b does not narrate at all and 120b-k3k narrates only with a one-sentence "Ok"-initial preamble.
9. Practical implications: (a) a single T=0 held-out checkpoint is not a reliable selector for these runs — use n>1 at T=1 or average several checkpoints; (b) report "honest strict" with the sentence-2 diagnostic; (c) the narration opener is the thing to penalize or prefill away (e.g. a meta-narration floor in the reward, or an on-topic first-sentence requirement) if held-out stability matters.
10. Suggested confirmatory probe (not run here, needs a vLLM job): log top-2 logprobs at the first token of sentence 2 for a trough (k30k step 30) and a peak (step 170) checkpoint; the prediction is a sub-nat margin that changes sign, with no comparable margin in the 20b runs because their sentence 2 is question-conditional.
