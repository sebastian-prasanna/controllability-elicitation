# sweep29 kall: training-side drivers of the held-out (start_of_sentence) decline

Date: 2026-10-02. Read-only analysis of `rl/runs/sweep29_gptoss120b_fs1/sweep29-fs1-{k1k,k3k,k10k,k30k,k100k,kall}` and the 20b reference `rl/runs/sweep24_controller/sweep24-fix05-kall`.
Inputs: `iters/iter-NNNN.json` (all 256 rollouts/step, graded text = `reasoning_text_graded`), `batches/batch-N.json` (rewards actually used), `metrics.jsonl`, `progress.jsonl`, `eval/indist/checkpoint-*.json`.
Code: `indist_drivers.py` (series), `extra_cells.py` (k1k/k3k), `tabulate.py` (tables). Series: `indist_series.json` (per-iteration, all 7 cells; `_doc` key describes fields).
Held-out strict numbers quoted below come from the other agent's `heldout_series.json` (start_of_sentence strict per checkpoint).

Conventions: sentences = grader split `(?<=[.!?])\s+`; paragraphs = blocks separated by blank lines; "ftnl" = fraction of sentence terminators followed by a newline (1.0 = every sentence is its own line/paragraph); "spp" = sentences per paragraph. Tables are 25-iteration buckets (bucket mean of per-iteration medians/means over the 256 rollouts). Advantage = r - mean(r) within the 8-rollout group (Dr.GRPO, `advantage_scale: none`), degenerate groups dropped.

## 1. Headline: kall's in-dist rollouts fragment into one-/two-sentence paragraphs; every other cell goes the other way

All six trained modes pooled (256 rollouts/iter):

| bucket (iters) | 0 | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 | 225 |
|---|---|---|---|---|---|---|---|---|---|---|
| **n paragraphs (median)** kall | 11.2 | 18.2 | 21.7 | 28.1 | 36.8 | 38.9 | 42.8 | **47.3** | | |
| k100k | 7.2 | 10.3 | 13.7 | 19.9 | 15.3 | 5.9 | 6.0 | 8.3 | | |
| k30k | 6.8 | 13.4 | 15.6 | 12.2 | 11.0 | 9.9 | 11.3 | 21.9 | | |
| k10k | 5.7 | 14.7 | 12.9 | 12.9 | 10.4 | 10.7 | 9.9 | 8.9 | | |
| k3k | 6.4 | 9.4 | 10.0 | 11.1 | 10.4 | 12.6 | 13.7 | 15.0 | | |
| k1k | 5.7 | 7.0 | 8.1 | 9.6 | 9.6 | 10.0 | 10.2 | 12.6 | | |
| 20b kall | 1.0 | 3.0 | 4.5 | 2.1 | 1.1 | 1.0 | 1.1 | 1.0 | 3.4 | 5.8 |
| **sentences / paragraph (median)** kall | 3.52 | 2.68 | 2.59 | 2.52 | 2.20 | 2.06 | **1.81** | 2.03 | | |
| k100k | 5.03 | 4.99 | 4.60 | 3.62 | 3.83 | 8.83 | 7.91 | 6.87 | | |
| k30k | 5.22 | 3.73 | 3.14 | 3.64 | 3.82 | 4.32 | 4.60 | 2.88 | | |
| k10k | 5.16 | 3.52 | 3.59 | 3.76 | 4.20 | 4.50 | 4.33 | 5.04 | | |
| 20b kall | 14.7 | 12.9 | 14.7 | 17.9 | 20.9 | 22.4 | 32.3 | 37.2 | 19.2 | 10.8 |
| **ftnl (mean)** kall | .301 | .384 | .422 | .442 | .497 | .503 | **.535** | .504 | | |
| k100k | .215 | .218 | .275 | .334 | .281 | .134 | .136 | .143 | | |
| k30k | .208 | .290 | .331 | .287 | .280 | .250 | .266 | .404 | | |
| k10k | .206 | .307 | .314 | .304 | .262 | .242 | .251 | .231 | | |
| k3k / k1k | .236/.223 | .270/.249 | .273/.263 | .260/.286 | .263/.282 | .271/.291 | .257/.291 | .260/.343 | | |
| 20b kall | .079 | .134 | .147 | .098 | .058 | .058 | .056 | .037 | .085 | .123 |
| **newline density /1k chars (median)** kall | 9.6 | 13.5 | 14.0 | 14.0 | 17.2 | 19.2 | **21.9** | 19.8 | | |
| k100k | 5.9 | 6.0 | 6.8 | 8.7 | 7.6 | 2.9 | 3.5 | 4.6 | | |
| **paragraph chars (median)** kall | 204 | 154 | 151 | 149 | 124 | 111 | **99** | 108 | | |
| k100k | 296 | 320 | 313 | 262 | 280 | 648 | 560 | 405 | | |
| **reasoning chars (median)** kall | 2001 | 2778 | 3612 | 4498 | 5034 | 4974 | 4690 | **5583** | | |
| k100k | 2040 | 3169 | 3849 | 4673 | 3843 | 3015 | 2799 | 3232 | | |
| k30k / k10k | 1852/1714 | 2738/2915 | 3110/2976 | 3084/3388 | 3138/3365 | 3175/3631 | 3369/3371 | 3581/3540 | | |
| 20b kall | 1661 | 2571 | 3341 | 3106 | 2974 | 3248 | 4215 | 3467 | 3480 | 3807 |
| **n sentences (median)** kall | 34 | 46 | 58 | 68 | 80 | 80 | 74 | **98** | | |
| k100k | 34 | 48 | 57 | 66 | 54 | 42 | 42 | 55 | | |
| **accuracy** kall | .860 | .865 | .896 | .924 | .920 | .922 | .930 | **.934** | | |
| k100k / k30k / k10k | .884/.885/.881 | .879/.889/.877 | .878/.833/.852 | .856/.814/.842 | .869/.780/.793 | .782/.787/.838 | .736/.792/.822 | .766/.862/.801 | | |
| **compliance (in-dist, T=1)** kall | .520 | .570 | .616 | .633 | .612 | .652 | .690 | .665 | | |
| k100k / k30k / k10k | .451/.384/.385 | .557/.473/.421 | .601/.527/.476 | .600/.595/.511 | .630/.655/.563 | .703/.714/.565 | .762/.739/.598 | .731/.647/.612 | | |

kall is the only cell whose paragraph count grows 4x, whose paragraphs shrink to ~100 chars / ~2 sentences, and whose traces nearly triple in length while accuracy rises (.86 -> .93). The masked cells shorten or plateau after iter ~100 (k100k p90 chars 18.9k -> 7.7k) and pay 5-15 pp accuracy for their compliance gains; kall pays none. The sentence-terminator density stays ~20/1k in kall (the floor at 1.0/1k is hit by <0.5% of rollouts in every cell; `sent_floor` is not involved; sweep24 had no `sent_floor` at all).

## 2. end_of_sentence: kall converges to "one sentence = one paragraph", every other cell to inline sentences

Compliant end_of_sentence rollouts only (`... safe.` must end every sentence):

| bucket | 0 | 25 | 50 | 75 | 100 | 125 | 150 | 175 |
|---|---|---|---|---|---|---|---|---|
| **spp (median)** kall | 3.44 | 2.25 | 1.49 | 1.10 | 1.04 | 0.93 | **0.74** | 1.07 |
| k100k | 7.36 | 6.66 | 4.37 | 4.99 | 10.3 | 24.6 | 39.1 | **45.0** |
| k30k | 8.52 | 5.56 | 5.83 | 6.58 | 6.11 | 11.0 | 15.4 | 6.28 |
| k10k | 8.82 | 7.77 | 9.42 | 11.1 | 12.1 | 14.3 | 16.5 | 21.0 |
| k3k / k1k | 4.75/8.41 | 5.08/7.86 | 5.15/8.36 | 6.84/8.71 | 6.76/9.31 | 6.82/9.16 | 7.86/11.3 | 8.79/8.59 |
| 20b kall | 18.2 | 23.1 | 41.1 | 33.5 | 19.8 | 29.8 | 52.4 | 68.4 |
| **ftnl** kall | .281 | .498 | .706 | .853 | **.953** | .937 | .926 | .907 |
| k100k | .141 | .264 | .548 | .500 | .179 | .028 | .013 | .020 |
| k30k | .107 | .304 | .286 | .233 | .259 | .172 | .199 | .539 |
| k10k | .076 | .169 | .148 | .132 | .084 | .069 | .071 | .039 |
| 20b kall | .018 | .044 | .041 | .022 | .048 | .016 | .018 | .008 |
| **n_para (median)** kall | 6.6 | 19.8 | 27.6 | 42.5 | 55.2 | 69.4 | **76.6** | 75.9 |
| k100k | 2.1 | 6.7 | 11.8 | 13.7 | 4.8 | 1.2 | 1.0 | 1.0 |
| **frac multi-sentence paragraphs** kall | .633 | .444 | .269 | .117 | .059 | .051 | **.024** | .069 |
| k100k | .833 | .766 | .614 | .657 | .849 | .955 | .975 | .969 |

Marker placement (all end_of_sentence rollouts, compliant or not):

| bucket | 0 | 25 | 50 | 75 | 100 | 125 | 150 | 175 |
|---|---|---|---|---|---|---|---|---|
| **paragraph-FINAL sentences ending "safe"** kall | .947 | .961 | .965 | .968 | .979 | .962 | .960 | .977 |
| k100k | .948 | .974 | .969 | .978 | .992 | .993 | .992 | .995 |
| **paragraph-INNER sentences ending "safe"** kall | .914 | .923 | .952 | .915 | .900 | **.829** | **.798** | .905 |
| k100k | .879 | .930 | .951 | .969 | .982 | .982 | .989 | .991 |
| k30k / k10k | .876/.782 | .894/.830 | .928/.903 | .956/.940 | .963/.955 | .974/.960 | .969/.967 | .876/.980 |
| 20b kall | .960 | .969 | .986 | .983 | .966 | .987 | .990 | .993 |
| **strict compliance** kall | .403 | .467 | .564 | .641 | .632 | .628 | .719 | .644 |
| k100k | .310 | .427 | .598 | .597 | .706 | .764 | .840 | **.876** |

Interpretation. k100k/k30k/k10k/k1k/k3k and 20b kall learn end_of_sentence as "put `safe` before every period" inside ordinary multi-sentence paragraphs (inner = paragraph-final marking, ~.99). kall instead learns it as "end every paragraph with `safe.`" and makes every sentence its own paragraph (95% of terminators followed by a newline, 76 paragraphs/trace, 2% multi-sentence paragraphs). Whenever kall does write a multi-sentence paragraph, the inner sentences are the ones that miss the marker (inner .80-.83 vs paragraph-final .96-.98 at iters 125-175; gap widens from .03 to .16). That is literally "the constraint applies per paragraph", and the same rule applied to the untrained start_of_sentence constraint gives "Ok" at paragraph starts only. kall's strict end_of_sentence compliance also plateaus lower (.64-.72) than k100k's (.84-.88), consistent with a less exact rule.

The T=0 in-dist eval shows the same thing (end_of_sentence, val): kall step 0 -> 200: n_para 1 -> 60, spp 6.0 -> 1.25, ftnl 0 -> .79, inner-marker .93 -> .74 (step 150) / .92 (step 200); k100k: n_para 1 -> 1, spp 7 -> 15, ftnl 0, inner .95 -> .99. All-mode T=0 n_para: kall 1 -> 17; k100k 1 -> 1; k30k 1 -> 8 (step 200 only, see section 6); k10k 1 -> 1; 20b kall 1 -> 1.

## 3. repeat_sentences: kall sprinkles the sentinel at section boundaries mid-trace

The grader only checks the first and last line, so paragraph placement is irrelevant to reward here; it is a pure readout of the learned habit.

| bucket | 0 | 25 | 50 | 75 | 100 | 125 | 150 | 175 |
|---|---|---|---|---|---|---|---|---|
| **sentinel occurrences / trace** kall | 1.98 | 2.01 | 2.02 | 2.16 | 2.54 | 3.00 | 3.61 | **3.88** |
| k100k | 1.95 | 2.03 | 2.06 | 2.07 | 2.03 | 2.07 | 3.52 | 3.30 |
| k30k / k10k / 20b | 1.85/1.89/1.85 | ~2.0 | ~2.0 | ~2.0 | ~2.0 | ~2.0 | ~2.0 | ~2.0 |
| **mid-trace markers / trace (5-95% of length)** kall | .23 | .11 | .08 | .14 | .46 | .72 | 1.18 | **1.55** |
| k100k | .21 | .09 | .04 | .02 | .01 | .02 | 1.50* | .10 |
| k30k / k10k / 20b | .24/.26/.17 | .08/.11/.02 | .02/.11/.01 | .03/.05/.02 | .02/.09/0 | .01/0/0 | .01/.01/.01 | .01/0/0 |
| **frac traces with a mid-trace marker** kall | .22 | .11 | .08 | .09 | .25 | .37 | **.57** | .46 |
| k100k | .20 | .08 | .04 | .02 | .01 | .02 | .04 | .06 |

\*k100k bucket 150 is one run-away trace with ~200 adjacent copies (also visible as the step-200 T=0 outlier `rep_n_occ=198`); 97% of k100k's extra markers are adjacent duplicates at the very start/end (positions <2% / >98%), 3% are mid-trace. In kall, 38% of late markers are mid-trace and 60-77% of those sit at a line start.

## 4. The two-pass "draft -> narrate the constraint -> rewrite" structure is kall-specific and explains the length growth

Regex detector for meta/rewrite phrases ("rewrite", "let's craft", "I must ensure all sentences...", "now produce analysis", ...); two-pass = meta phrase occurring before 60% of the trace; answer position = where the final (correct) answer string first appears in the reasoning.

| bucket | 0 | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 | 225 |
|---|---|---|---|---|---|---|---|---|---|---|
| **meta-phrase rate** kall | .14 | .19 | .30 | .46 | .60 | .66 | **.71** | .68 | | |
| k100k / k30k / k10k | .14/.12/.10 | .18/.17/.17 | .23/.19/.16 | .26/.15/.16 | .23/.15/.14 | .19/.16/.17 | .20/.19/.14 | .26/.27/.14 | | |
| k3k / k1k | .09/.09 | .10/.09 | .11/.11 | .12/.12 | .13/.12 | .15/.12 | .17/.13 | .18/.14 | | |
| 20b kall | .09 | .15 | .26 | .22 | .20 | .18 | .32 | .29 | .33 | .31 |
| **two-pass rate** kall | .08 | .11 | .17 | .27 | .34 | .36 | .35 | **.36** | | |
| k100k / k30k / k10k | .09/.08/.07 | .12/.11/.11 | .15/.13/.11 | .18/.11/.11 | .15/.11/.10 | .12/.12/.13 | .12/.13/.10 | .14/.16/.11 | | |
| 20b kall | .06 | .11 | .19 | .17 | .15 | .14 | .21 | .20 | .23 | .23 |
| **median rel. position of first correct answer** kall | .71 | .61 | .53 | .46 | .36 | .34 | **.34** | .38 | | |
| k100k / k30k / k10k | .72/.73/.77 | .63/.65/.65 | .58/.67/.67 | .59/.70/.69 | .62/.66/.67 | .68/.68/.67 | .69/.61/.67 | .70/.55/.69 | | |
| 20b kall | .75 | .72 | .67 | .71 | .71 | .72 | .62 | .65 | .60 | .58 |

Per-mode two-pass rate, bucket 0 -> bucket 175-199, kall: end_of_sentence .07 -> .43, lowercase .06 -> .35, uppercase .11 -> .27, repeat .09 -> .27, alternating_case .15 -> .72 (chars 3.3k -> 10.7k, 136-153 paragraphs, strict 0); k100k stays .07-.20 in every mode. Typical late kall rollout: solve the problem in plain prose (answer reached at ~1/3 of the trace), then "Now ensure analysis sentences end with 'safe' safe. I must rewrite analysis...", then a second, constraint-formatted copy of the solution; for repeat_sentences the sentinel is re-emitted at the draft/rewrite seam (section 3). This is why kall's traces lengthen without losing accuracy: the first pass does the math, the second pass does the constraint. It is also the only cell whose in-dist accuracy rises during RL.

## 5. What the reward is pushing on (within-group correlations with advantage, non-truncated rollouts)

| bucket | 0 | 25 | 50 | 75 | 100 | 125 | 150 | 175 |
|---|---|---|---|---|---|---|---|---|
| **adv ~ n_para, sentence modes (eos+repeat)** kall | .147 | .201 | .188 | .156 | .097 | .146 | .108 | .031 |
| k100k | .040 | .128 | .056 | .008 | .082 | .030 | .014 | .031 |
| k30k / k10k | .103/.100 | .136/.191 | .092/.137 | .096/.109 | .070/.102 | .064/.081 | .109/.039 | .028/.065 |
| 20b kall | -.026 | .072 | -.124 | -.075 | -.019 | .022 | -.077 | -.013 |
| **adv ~ ftnl, sentence modes** kall | .162 | .102 | .119 | .175 | .177 | .180 | **.192** | .037 |
| k100k | .078 | .143 | .127 | .065 | .096 | .037 | -.026 | .025 |
| k30k / k10k | .145/.131 | .131/.195 | .111/.155 | .101/.094 | .053/.086 | .074/.016 | .083/.022 | .000/-.001 |
| **adv ~ log chars, all modes** kall | .293 | .290 | .219 | .145 | .052 | .093 | .055 | .030 |
| k100k | .269 | .191 | .073 | .010 | .048 | .063 | .108 | .095 |
| 20b kall | .292 | .177 | .089 | .076 | .069 | -.016 | -.103 | .032 |
| **P(longest rollout in group has adv>0)** kall | .63 | .64 | .66 | .62 | .60 | .64 | .62 | .61 |
| k100k / k30k / k10k | .62/.63/.65 | .61/.59/.58 | .56/.55/.56 | .56/.57/.56 | .60/.58/.55 | .61/.60/.54 | .64/.56/.58 | .62/.56/.56 |
| **adv ~ terminator density, sentence modes** kall | -.08 | -.06 | -.09 | -.04 | -.12 | -.01 | -.03 | -.02 |
| k100k / 20b | -.05/-.17 | -.15/-.08 | -.08/-.10 | -.10/-.15 | -.09/-.02 | -.13/-.07 | .02/-.15 | -.07/-.24 |
| **adv ~ correct / adv ~ compliant** kall | .73/.82 | .79/.86 | .76/.91 | .76/.92 | .81/.98 | .80/.94 | .78/.95 | .72/.97 |
| **corr(correct, compliant) within group** all cells | -.07..+.18, no trend | | | | | | | |

Reading: the reward is dominated by compliance (adv~compliant .82 -> .97) with accuracy second; there is no accuracy-vs-compliance trade-off inside groups in any cell (corr ~0). Length is rewarded early in every cell (+.27-.34 at iters 0-25) because longer rollouts were the compliant ones, but that signal decays to ~0 by iter 75 in the masked cells and the 20b kall, whereas kall keeps a positive length and paragraph-count/ftnl advantage correlation through iter ~150 (.15-.20 on sentence-mode groups) and the longest rollout in a group keeps a >60% chance of positive advantage for the whole run. Terminator density is weakly anti-correlated with advantage everywhere (not floor-related; the floor is never hit). So the fragmentation is reinforced, but it is reinforced because kall's compliant rollouts *are* the fragmented, two-pass ones, not because any reward term pays for newlines per se.

## 6. The association replicates inside other cells at the moments their held-out compliance dips

10-iteration windows, training stats over iters [a, a+10) vs held-out start_of_sentence strict at step a+10:

```
k30k   iters: n_para  ftnl  spp   eos_compliant_ftnl  eos_compliant_spp  heldout
140-149        10.4  .244  4.40   .133                14.3               .96
160-169        11.7  .276  4.49   .210                14.5               .93
170-179        13.9  .323  3.77   .240                19.4               .88
180-189        20.1  .386  2.99   .580                 2.39              .81
190-199        26.9  .452  2.44   .650                 2.07              .82
20b kall
180-189         1.0  .030  40.0   .005                82.2               .89
190-199         1.1  .051  29.3   .013                40.6               .89
200-209         1.9  .065  24.3   .011                44.5               .62
210-219         4.0  .091  16.7   .020                32.3               .68
230-239         6.7  .133   8.7   .034                33.1               .72
kall
  0-  9         9.4  .278  3.85   .199                 3.69              .69
 10- 19        11.8  .306  3.40   .308                 3.43              .56
 20- 29        13.8  .349  2.94   .405                 2.97              .51
 30- 39        17.1  .370  2.75   .497                 2.03              .40
 40- 49        21.4  .410  2.55   .538                 2.07              .35
 60- 69        23.9  .428  2.55   .737                 1.39              .36
 90- 99        31.6  .455  2.47   .921                 1.06              .28
```

k30k's late held-out drop (.96 -> .81) coincides exactly with its end_of_sentence rollouts flipping to one-sentence-per-paragraph (eos ftnl .24 -> .58 -> .65, spp 19 -> 2.1) and all-mode n_para doubling; 20b kall's step-210 drop (.89 -> .62) coincides with its first paragraph fragmentation (n_para 1.0 -> 1.9 -> 4.0, spp 40 -> 17); kall's decline tracks its fragmentation ramp step by step from iter 10.

Cross-cell check, Spearman of held-out strict at step s vs training feature over iters [s-10, s), pooled over the six 120b cells after per-cell demeaning (120 points): inner-minus-paragraph-final marker gap +.60, newline density -.54, eos-compliant ftnl -.46, ftnl -.45, eos-compliant spp +.45, n_para -.41, in-dist accuracy -.61, in-dist compliance +.37, **chars -.11, n_sent -.28**. Raw length is not what tracks the held-out loss; paragraph fragmentation and the paragraph-end marking rule are. (Within kall alone every monotone series correlates at |rho| .8-.9, so the within-cell numbers do not discriminate.)

## 7. Things that do NOT distinguish kall

- Optimizer/trainer: clip_frac 0.0000 in every cell at every step; `logprob_corr` .995 -> .976-.979 (kall) vs .965-.974 (k100k/k30k), sampler-trainer `logprob_diff_abs_mean` .031-.037 in kall, the lowest of the 120b cells (k100k rises to .051); `tis_frac_truncated` <0.4% everywhere; no overlong drops. grad_norm: kall .009-.012 (flat), identical to 20b kall (.009-.015), vs k100k .003 -> .009 (rising), k30k .002 -> .004, k10k .001. kall's lr (3.3e-4) is 10-30x lower than the masked cells', per the 1/sqrt(k) rule. Nothing anomalous.
- Hollow floors: `sent_floor` (1 terminator/1k) hit by 0.0-0.5% of rollouts per bucket in every cell; d4/zlib floors never hit; terminator density median stays 19-22/1k in kall (it falls to 13-16 in the masked cells). The donor's rare no-period end_of_sentence style (one 753-char example at iter 0) is 3% of iter-0 end_of_sentence rollouts in kall and k100k alike.
- Truncation: <3% everywhere, 0.2-0.5% late in kall.
- Degenerate (all-equal-reward) groups: kall 1.7 -> 10.8 of 32 by the end, k100k 2.0 -> 8.7, 20b kall 0.3 -> 10.4; similar.
- Per-mode strict compliance trajectories: kall improves all five learnable modes like the others (lowercase .90 -> .99, uppercase .48 -> .81, meow .35 -> .58, repeat .96 -> 1.0, eos .40 -> .64; alternating_case ~0 everywhere). kall is not better or worse in-dist; it is the *way* it complies that differs.
- Donor (iter 0) differences are small: kall n_para 11 vs 6-7, ftnl .30 vs .21-.24, two-pass rate .08 vs .07-.09. A modest head start that RL amplified, not a different starting policy.

## 8. Conclusion

1. There is a training-side change specific to kall, visible from iter ~10 and saturating by iter ~100, and it is a discourse-structure change, not a length-per-se or optimizer effect: kall's in-dist rollouts fragment into ~45-75 one-/two-sentence paragraphs (spp 3.5 -> 1.8-2.0; 50-54% of sentence terminators followed by newlines; paragraph length 204 -> ~100 chars), while every masked 120b cell and the 20b kall converge to a few long paragraphs (k100k spp 7-9, 20b kall 20-37).
2. The clearest mechanism is end_of_sentence: kall's compliant traces become exactly one sentence per paragraph ("... safe.\n\n... safe.\n\n", spp 1.05, 95% ftnl, 76 paragraphs), and in multi-sentence paragraphs kall marks the paragraph-final sentence (.96-.98) far more reliably than inner sentences (.80-.90, gap widening to .16). The other cells mark inner and final sentences equally (.98-.99) inside single-paragraph inline prose. kall has therefore learned "the sentence constraint applies at paragraph ends", which transfers to the untrained start_of_sentence constraint as "Ok at paragraph starts".
3. repeat_sentences corroborates the per-section rule: kall re-emits the start/end sentinel mid-trace at section boundaries (1.55 per trace, 46-57% of traces, mostly at line starts); k100k's extras are adjacent duplicates at the ends; no other cell does either.
4. The length growth (chars 2.0k -> 5.6k, sentences 34 -> 98, accuracy up to .93) is a two-pass "solve -> narrate the constraint -> rewrite" structure (meta-phrase rate .14 -> .70, answer first reached at 34% of the trace vs 55-77% elsewhere). It is kall-specific (masked cells <=.27, 20b kall <=.33) and is the training-side counterpart of the longer held-out traces (8 -> 39 sentences).
5. The reward reinforces this because kall's compliant rollouts are the fragmented two-pass ones: within sentence-mode groups, advantage stays positively correlated with paragraph count/ftnl (.15-.20) and with length through iter ~150 in kall, while those correlations decay to ~0 by iter 75-100 in the masked cells. No reward term pays for newlines directly; terminator density is weakly anti-correlated with advantage everywhere and the sentence-density floor is never engaged.
6. The association is not kall-only in kind, only in degree: k30k's held-out dip at steps 180-200 and 20b kall's at 210-240 coincide with the same fragmentation flip in their own end_of_sentence rollouts, and pooled across the six 120b cells held-out strict tracks the paragraph-marking gap (+.60), newline density (-.54) and ftnl (-.45) but not raw length (-.11).
7. Nothing on the optimizer side separates kall (clip 0, logprob agreement best of the 120b cells, grad norm equal to 20b kall, no floor/truncation events); the donors start nearly alike (kall slightly more fragmented, n_para 11 vs 6-7).
8. Why the full-rank-1 LoRA finds this solution and the masked ones do not is not identifiable from these logs; the data are consistent with kall being able to move a global formatting prior (newline/paragraph policy and a draft-then-rewrite plan) that the sparse masks cannot cheaply express, so the masked cells satisfy end_of_sentence token-locally instead.
9. Practical implication: a held-out-safe recipe for kall would need to penalise the paragraph-level solution in-dist, e.g. (a) reward end_of_sentence/start_of_sentence-style modes only when paragraphs contain >1 sentence or when inner-sentence marking equals paragraph-final marking, (b) penalise mid-trace sentinel repeats, or (c) a meta-discussion/two-pass penalty (the training reward has none; `grade_meta_discussion` is off), since the narrated-rewrite structure is what both lengthens traces and re-segments them.
10. Minimal diagnostic to watch in future runs: per-iteration `frac_term_at_nl` and `sents_per_para` on compliant end_of_sentence rollouts (available from iters/ with ~10 lines of code); in all three cells with a held-out drop the flip to ftnl > .5 / spp < 2.5 preceded or coincided with it.
