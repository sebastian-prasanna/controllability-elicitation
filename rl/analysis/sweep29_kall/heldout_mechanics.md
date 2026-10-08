# sweep29 kall held-out decline: mechanics

Scope: all six sweep29 cells (gpt-oss-120b, few-shot SFT donors, fixed λ=.5, 200 steps; k1k..k100k masked rank-1 LoRA, kall full rank-1 LoRA) plus the 20b references sweep24-fix05-kall / -k100k (GEPA donors, 250 steps). Every held-out checkpoint (T=0, 200 val questions, start_of_sentence 72 / no_spaces 70 / letter_suppression 58) and every in-dist checkpoint (200 q, 6 trained modes). Per-sample features in `samples.parquet` (44,000 rows), per-checkpoint tables in `heldout_mechanics.json`, extractor `extract.py`. Definitions: sentences = grader split `(?<=[.!?])\s+`; "Ok" = grader rule (first word stripped of punctuation, case-insensitive); a sentence is *paragraph-first* if it is sentence 0 or the whitespace before it contains a blank line, *inline* otherwise; honest = compliant & answered & not hollow (sweep23 definition). Step groups: early 0-40, mid 80-120, late 160-200.

## Headline

The kall-only decline is a **format drift of the whole policy, not a start_of_sentence-specific slip**. Under full-LoRA RL the 120b model's reasoning changes from one block of prose into many short paragraphs (held-out start_of_sentence: median paragraphs 1 → 17, sentences 8 → 39, chars 848 → 3300; in-dist: paragraphs 1 → 17, chars 1058 → 4881; training rollouts: 1.9k → 5.6k chars). The donor's "Ok" habit is strongest at segment starts, so in the fragmented format "Ok" survives at paragraph starts (.82-.98) and is lost on the sentences inside each paragraph (inline Ok rate .57 → .08-.17, macro .82 → .11-.22). Because the strict grader needs 100% of sentences, the per-trace Ok fraction collapses from a 57%-at-1.0 distribution to a mode at .3-.7. The masked cells and 20b-kall never fragment (paragraphs stay at 1) and RL *strengthens* their per-sentence habit (inline slip .28 → .03), which is why they rise. The decline is smooth and monotone across 20 checkpoints and shows up simultaneously in held-out, in-dist and training-rollout texture, so within this run it is systematic, not a stochastic flip; only one kall seed exists, so cross-seed reproducibility is untested.

## Task 1. start_of_sentence per checkpoint (held-out, n=72 per checkpoint)

Columns: strict rate; median sentences / paragraphs per trace; median sentences-per-paragraph; pooled Ok fraction for all / paragraph-first / inline sentences (macro = per-trace mean); median index of the first non-Ok sentence (0-based) and as fraction of the trace. Every trace in every cell at every checkpoint starts with "Ok" (starts_ok = 1.00 throughout), and rule narration is absent (not tabulated).

### 120b-kall (full rank-1 LoRA, lr 3.3e-4)
| step | strict | med sents | med paras | sents/para | Ok all | Ok para-first | Ok inline (pooled / macro) | 1st non-Ok idx (frac) |
|---|---|---|---|---|---|---|---|---|
| 0 | .667 | 8 | 1 | 6.1 | .61 | .82 | .57 / .82 | 5 (.18) |
| 20 | .556 | 12 | 1 | 7.0 | .64 | .72 | .63 / .79 | 4 (.13) |
| 40 | .403 | 18.5 | 2 | 6.5 | .57 | .89 | .52 / .71 | 3 (.17) |
| 60 | .347 | 21.5 | 3.5 | 5.7 | .59 | .75 | .56 / .66 | 4 (.13) |
| 80 | .431 | 17.5 | 4 | 5.4 | .90* | .99 | .55 / .71 | 5 (.12) |
| 100 | .278 | 25 | 7 | 4.3 | .42 | .81 | .29 / .55 | 4 (.09) |
| 120 | .083 | 37 | 11 | 3.2 | .36 | .76 | .19 / .26 | 2 (.05) |
| 140 | .097 | 35.5 | 16 | 2.5 | .81* | .97 | .15 / .22 | 2 (.04) |
| 160 | .139 | 29.5 | 18.5 | 1.6 | .51 | .84 | .08 / .11 | 2 (.05) |
| 180 | .153 | 32.5 | 18 | 1.8 | .54 | .92 | .16 / .19 | 2 (.07) |
| 200 | .125 | 39 | 17 | 2.6 | .47 | .89 | .17 / .22 | 2 (.07) |

\* pooled "Ok all" at steps 80/130/140 is inflated by one very long one-sentence-per-line trace; the macro column is the reliable one (.84 → .45-.58). Fraction of traces with >1 paragraph: .19 → .58 (step 40) → .85 (100) → .97-1.00 (120-200).

### 120b-k100k (masked, 100k scalars)
| step | strict | med sents | med paras | Ok all | Ok para-first | Ok inline (pooled / macro) | 1st non-Ok idx (frac) |
|---|---|---|---|---|---|---|---|
| 0 | .625 | 9 | 1 | .45 | .61 | .42 / .77 | 4 (.11) |
| 40 | .778 | 11 | 1 | .93 | .98 | .92 / - | 7.5 (.31) |
| 100 | .833 | 12.5 | 1 | .92 | 1.00 | .91 / - | 6.5 (.28) |
| 160 | .944 | 12 | 1 | .99 | .98 | .99 / - | 7.5 (.34) |
| 200 | .903 | 12 | 1 | .97 | 1.00 | .97 / .99 | 7 (.39) |

### 120b-k30k (masked, 30k scalars)
| step | strict | med sents | med paras | Ok all | Ok para-first | Ok inline | 1st non-Ok idx (frac) |
|---|---|---|---|---|---|---|---|
| 0 | .681 | 10 | 1 | .70 | .94 | .66 | 4 (.27) |
| 40 | .806 | 10.5 | 1 | .98 | 1.00 | .98 | 20 (.33) |
| 100 | .944 | 11 | 1 | .99 | 1.00 | .99 | 4.5 (.36) |
| 160 | .958 | 11.5 | 1 | .99 | 1.00 | .99 | 3 (.10) |
| 190 | .806 | 17 | 1 | .98 | 1.00 | .97 | 14.5 (.48) |
| 200 | .819 | 20 | 1 | .98 | 1.00 | .98 | 11 (.21) |

### 20b-kall (full rank-1 LoRA, GEPA donor, 250 steps)
| step | strict | med sents | med paras | Ok all | Ok para-first | Ok inline | 1st non-Ok idx (frac) |
|---|---|---|---|---|---|---|---|
| 0 | .597 | 11 | 1 | .29 | .80 | .28 | 1 (.09) |
| 50 | .778 | 17 | 1 | .97 | .99 | .97 | 10.5 (.22) |
| 100 | .889 | 17.5 | 1 | .98 | 1.00 | .98 | 6 (.19) |
| 150 | .917 | 17 | 1 | .97 | 1.00 | .96 | 4.5 (.12) |
| 200 | .889 | 23 | 1 | .99 | 1.00 | .99 | 9.5 (.31) |
| 250 | .833 | 21 | 1 | .98 | 1.00 | .98 | 8 (.28) |

Full per-checkpoint tables for all 8 cells (plus line-first Ok rate and the texture of non-Ok sentences, which is >90% plain prose, not markdown/math) are in `heldout_mechanics.json["task1_sos_per_checkpoint"]`.

## Task 2. Hazard: length or a changed per-sentence slip rate?

P(strict | sentence-count bin), early vs late:

| cell | bin | early 0-40 | mid 80-120 | late 160-200 | n (e/m/l) |
|---|---|---|---|---|---|
| 120b-kall | 6-10 | .84 | .93 | .90 | 96/45/10 |
| | 11-15 | .64 | .57 | **.46** | 61/47/26 |
| | 16-20 | .46 | .31 | **.36** | 28/54/47 |
| | 21-30 | .24 | .12 | **.03** | 33/58/86 |
| | 31-45 | .14 | .04 | **.03** | 22/45/59 |
| 120b-k100k | 11-15 | .85 | .91 | .97 | 59/127/145 |
| | 16-20 | .62 | .92 | .84 | 34/53/57 |
| | 21-30 | .44 | .75 | .56 | 18/36/34 |
| 120b-k30k | 11-15 | .84 | .91 | .95 | 62/116/132 |
| | 21-30 | .59 | .88 | .83 | 17/17/41 |
| 20b-kall | 16-20 | .91 | .85 | .84 | 33/80/159 |
| | 21-30 | .81 | .91 | .84 | 21/46/193 |
| | 31-45 | .50 | .81 | .77 | 10/32/116 |

Per-sentence slip rates and a geometric fit (h solving mean((1-h)^(n-1)) = strict):

| cell | group | strict | med n | inline slip pooled / macro | para-first slip | geo-fit hazard |
|---|---|---|---|---|---|---|
| 120b-kall | early | .567 | 12 | .41 / .21 | .21 | .039 |
| | mid | .267 | 26 | .67 / .49 | .07 | - |
| | late | .119 | 33 | **.89 / .84** | .13 | **.078** |
| 120b-k100k | early | .717 | 10 | .28 / .11 | .12 | .023 |
| | late | .892 | 12 | .03 / .01 | .01 | .009 |
| 120b-k30k | early | .739 | 10 | .12 / .05 | .02 | .020 |
| | late | .878 | 14 | .02 / .01 | .00 | .007 |
| 20b-kall | early | .806 | 11 | .40 / .09 | .05 | .012 |
| | late | .801 | 22 | .07 / .03 | .01 | .008 |

Length-only counterfactual (apply the early geometric hazard to the late sentence-count distribution): kall predicted late strict **.287** vs observed **.119** (early .567). So roughly 60% of the kall drop (.567 → .287) is attributable to the 3x longer traces and the remaining 40% (.287 → .119) to a genuine per-sentence hazard increase; at matched n the late curve sits clearly below the early one for n ≥ 11. In every other cell the hazard moved the opposite way (RL on the six trained modes tightened the held-out per-sentence habit by 2.5-10x), which is why their held-out rate rose despite modest lengthening. Note the inline/paragraph split is the cleaner description than a uniform hazard: kall's paragraph-first slip stayed low (.07-.21) while inline slip went .21 → .84, i.e. the "hazard change" is really the appearance of inline sentences that follow an "Ok" paragraph opener. Where the first non-Ok sentence falls: kall median index 4-5 early (13-18% into the trace) → 2 late (5%): the slip now happens on the second or third sentence of the first paragraph.

## Task 3. Is the paragraph-level "Ok" habit inherited from the donor?

Step-0 held-out start_of_sentence, per-trace macro means:

| cell (donor) | strict | multi-para traces | Ok all (macro) | Ok inline (macro) | Ok para-first excl. trace-first sentence (n traces) |
|---|---|---|---|---|---|
| 120b-k1k | .736 | .25 | .91 | .90 | .76 (18) |
| 120b-k3k | .694 | .21 | .89 | .88 | .42 (15) |
| 120b-k10k | .778 | .07 | .93 | .92 | .52 (5) |
| 120b-k30k | .681 | .18 | .93 | .92 | .93 (13) |
| 120b-k100k | .625 | .19 | .79 | .77 | .61 (14) |
| **120b-kall** | .667 | .19 | .84 | .82 | .71 (14) |
| 20b-k100k | .681 | .07 | .80 | .77 | .30 (5) |
| 20b-kall | .597 | .08 | .72 | .69 | .17 (6) |

Yes, in a weak form, and it is NOT kall-specific. At step 0 every donor writes one paragraph (median paragraphs 1; only 7-25% of traces have a second paragraph) and every trace opens with "Ok"; the donor's failure mode is "Ok at the start, then drifting" (first non-Ok at sentence 4-5). kall's donor is indistinguishable from the k30k/k100k donors on all of these (multi-para .19 vs .18/.19; inline Ok .82 vs .92/.77). What differs is what RL does next: the masked cells convert the "Ok opener" habit into "Ok every sentence" (inline Ok → .97-.99 by step 40-100), while kall converts it into "Ok every paragraph opener" because the paragraph becomes the unit of its reasoning.

## Task 4. Where does kall's extra length come from, and is it kall-specific?

Decomposition for kall held-out start_of_sentence: chars 848 → 3300 (3.9x); sentences 8 → 39 (4.9x); paragraphs 1 → 17 (17x); sentences per paragraph 6.1 → 2.6 (down); median sentence length 90 → 77 chars (down). So the growth is entirely **more paragraphs** (and more sentences) — not longer sentences and not longer paragraphs; the trace is re-written as many short 2-3-sentence paragraphs, each opened with "Ok" (see examples below).

Held-out median chars, all 3 modes pooled (step 0 / 100 / 200): k1k 875/837/882; k3k 748/884/1292; k10k 684/1162/1030; k30k 719/1096/1940; k100k 824/1592/1326; **kall 742/3484/3706**; 20b-k100k 2550/2016/2088; 20b-kall 1664/1638/1918. Held-out multi-paragraph fraction (SOS) stays .1-.4 for every masked cell and ≤.1 for 20b-kall at all checkpoints; kall alone goes to 1.0.

In-dist T=0 (6 trained modes), median chars / paragraphs / multi-para fraction at step 0 → 200: **kall 1058 → 4881 chars, 1 → 17 paragraphs, .37 → 1.00**; k100k 1194 → 1657, 1 → 1, .34 → .49 (transient 4 paragraphs at step 100, back to 1 by 150); k30k 1108 → 2750, 1 → 8, .30 → .87 (only at step 200 — the first sign of the same drift, coinciding with its held-out SOS strict falling .958 → .806 at 190-200, although there the fall is length with hazard still .02); k10k 1095 → 1602, 1 → 1; k3k/k1k ≤ 1660, 1-2 paragraphs; 20b-kall 1257 → 2336, paragraphs 1 at every checkpoint; 20b-k100k paragraphs 1 except a one-off 6 at step 200. In-dist by mode for kall: end_of_sentence goes to **one sentence per paragraph** (sents/para 6.0 → 1.0-1.2, 1 → 60 paragraphs; example: "...heat safe.⏎⏎We need to consider a Diels-Alder type cycloaddition safe.⏎⏎..."), alternating_case 1 → 60 paragraphs with 17-24k-char traces and compliance .32 → .00 (hollow .13 → .32-.45), the other four modes 1 → 10-14 paragraphs. Training rollouts (progress.jsonl reasoning_chars_median, 20-iteration means): kall 1929 → 5629 (monotone); k100k 1907 → 4735 (step 80) → 3193; k30k 1822 → 3659; 20b-kall 1583 → 3873. The reward's length gate saturates at 1500 chars, which all cells exceed from step 0, so the gate is not what pulls kall to 5k chars; the lengthening is a by-product of the paragraph format rewarded in-dist (end_of_sentence/alternating_case rollouts) generalizing to all prompts.

## Task 5. kall no_spaces and letter_suppression

**no_spaces** (n=70): strict .386 (step 0) → .20 (40) → .13 (80) → .03-.06 (100-120) → .10-.20 (130-190) → .33 (200); honest tracks strict within .07. Texture is *partial attempts, not loops*: truncation ≤ .06 and hollow ≤ .10 at every checkpoint, distinct-4 median 1.0, the per-trace space rate stays ≤ .008 and 90-100% of traces have < 2% spaces throughout (the model is always attempting the constraint). The failures are a handful of stray spaces in traces that are now 4x longer: median stray spaces among non-strict traces 4-15, and traces again fragment into paragraphs (1 → 16-20). P(strict | chars) early vs late is actually *higher* late at matched length (1-2k: .13 → .34; 2-4k: .00 → .19), i.e. the per-character hazard improved but the traces moved from <1k to 2-4k+ chars, where strict is rare; the partial recovery to .33 at step 200 reflects a further drop in the per-trace space rate (.000 median) catching up with the length. Other cells: k3k .30 → .63, k1k .34 → .61, k10k .29 → .56, k30k .30 → .06 at step 200 (fell from .57 at 190 — same step where its traces lengthened), k100k .24 → .44 (noisy .13-.46), 20b-kall .16 → .70.

**letter_suppression** (n=58): strict .155 → .03 by step 30 → .00 from step 50 onward (one .017 at 140); honest identical. Texture is *loops/truncation*: finish=length .03 → .19-.53 (peaks at steps 100/120/150), hollow .07 → .45-.55, median chars 1.1k → 5-6k with 13-30k at the hollow peaks, distinct-4 median falling to .15-.47 at steps 100/120/150 (e.g. step 100: "...That includes words like "energy", "eV", "e", "e.g.", "e.g.,", "e.g.", ..." repeated to the 16k cap). The suppression itself is never attempted: median e-initial word rate per trace .035 at step 0 and .05-.10 later, 0% of traces with zero spaces-equivalent; the model solves the question first and then narrates/loops on the constraint. Other cells stay flat at .07-.26 (masked 120b) and .00-.17 (20b), with no comparable loop pathology, so letter_suppression is a second, independent kall degradation (loops), not the paragraph mechanism.

## Examples (kall, held-out start_of_sentence, same question id=97)

Step 0 (3003 chars, 1 paragraph): "Ok The problem describes bicyclo[2.2.2]octan-2-one undergoing UV irradiation ... Bicyclo[2.2.2]octan-2-one is a bridged ketone ... UV can cause a 1,2-shift ..." — Ok once, then prose without Ok (the donor's failure mode).

Step 120 (6391 chars, ~20 paragraphs): "Ok the problem describes ... product 2.⏎⏎Ok we need to deduce structure of product 2. Starting material: bicyclo[2.2.2]octan-2-one. That's a bicyclic ketone, ... ⏎⏎Ok UV irradiation of a ketone can cause a Norrish Type I cleavage ... ⏎⏎Ok Possibly a photochemical isomerization ..." — Ok at every paragraph opener, never on the 2nd-4th sentence of a paragraph. Step 200 is the same shape (6270 chars).

## Task 6. Conclusion

1. Mechanism: full-LoRA RL on the six trained modes drove a global change in gpt-oss-120b's reasoning format — from one prose block to many short paragraphs — that is visible in training rollouts (1.9k → 5.6k chars), in-dist T=0 evals (paragraphs 1 → 17, end_of_sentence to one sentence per paragraph) and held-out evals alike. The donor's "Ok" habit was anchored to segment openers; in the new format "Ok" is applied once per paragraph and the 1-3 sentences that follow inside each paragraph lack it (inline Ok .82 → .11-.22 macro), so the 100%-of-sentences grader fails. Roughly 60% of the drop is the 4-5x sentence count and 40% is the per-sentence hazard increase at matched length; both are the same format change seen from two angles.
2. Why only kall: the masked cells (1k-100k of 497k scalars, lr 1/sqrt(k)) do not acquire the paragraph format within 200 steps (paragraphs stay at 1, lengths 1.0-1.7k) and instead generalize the trained per-sentence habit, tightening held-out inline Ok to .97-.99. 20b-kall at the same recipe also never fragments (paragraphs 1 through step 250). The one hint that the attractor is reachable by other cells is k30k at steps 190-200 (in-dist paragraphs 1 → 8, held-out SOS .958 → .806 as sentences 11 → 20), so it looks like a capacity/time-dependent drift rather than something unique to the full-LoRA cell.
3. The step-0 donor habit is not the discriminating factor: kall's donor matches the k30k/k100k donors on multi-paragraph fraction (.19 vs .18/.19) and inline Ok rate (.82 vs .92/.77).
4. no_spaces declines by the same length route (partial attempts, 4-15 stray spaces in 4x longer fragmented traces, no loops) and partially recovers at 200; letter_suppression collapses to 0 by step 50 via loops/truncation (hollow .45-.55 at the worst checkpoints) and never recovers — a separate degradation specific to kall.
5. Systematic vs stochastic: the SOS decline is smooth and monotone over 20 checkpoints (.667 → .556 → .403 → .347 → .278 → .083 → .10-.15), the paragraph fraction rises monotonically (.19 → .58 → .85 → 1.00) and the same drift is present in every context the policy is sampled in, so within this run it is a systematic property of the trajectory, not a random flip. With one kall seed, nothing can be said about whether another seed would follow the same path; the k30k endpoint suggests it is a reachable attractor for high-capacity cells trained long enough.
