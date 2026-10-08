# Selection-metric sensitivity: honest vs ca vs comp (RL sweeps, val)

**Question.** At T=1, would results change if we reported raw grader compliance (`comp`) or compliant-and-answered (`ca`)
instead of honest (compliant AND answered AND non-hollow)? Convention as in `final_curves_data.point`: SFT = RL step 0;
SFT+RL = neighbour-smoothed (+-10 steps) best under gate A (acc >= 0.8 x step 0), each metric selected on itself.
72 runs (4 models x 10 cells finals + 2 models x 4 cells x 4 extra seeds); 6 stopped runs use partial evals.
Full auto-generated tables: `selection_sensitivity_full.md`; per-run numbers: `per_run_selection.csv`; script: `selection_sensitivity.py`.

**Answer: no.** At T=1 the three metrics are nearly interchangeable. Held-out, the comp-honest gap at the selected
checkpoint averages 0.8 pp (max 5.3 pp, one cell); ca-honest averages 0.3 pp (max 2.8 pp). Selecting on raw compliance
instead of honest costs 0.21 pp honest on average (max 3.8 pp, one seed run). No cell flips RL-gain sign, no 5-seed
mean moves > 2.1 pp, no seed ordering changes except one k100 tie. At T=0 the same comparison had 6/40 held-out cells
with > 5 pp comp-honest gaps (max 11.2 pp) and two seed-bar cells moving 5-8 pp, which is why honest was adopted.

## 1. Selection stability (which checkpoint is picked)

| T | block | alt | runs w/ different step (of 72) | by model (20b / 120b / q8b / q32b) | \|dstep\| median / max | honest lost, mean / max over changed runs | honest lost, avg over all runs |
|---|---|---|---|---|---|---|---|
| 1 | heldout | comp | 20 | 6/26, 7/26, 3/10, 4/10 | 40 / 240 | 0.8 / 3.8 pp | 0.21 pp |
| 1 | heldout | ca | 10 | 2/26, 6/26, 1/10, 1/10 | 25 / 240 | 0.2 / 0.6 pp | 0.03 pp |
| 1 | indist | comp | 31 | 9/26, 13/26, 6/10, 3/10 | 20 / 240 | 0.4 / 1.4 pp | 0.18 pp |
| 1 | indist | ca | 19 | 7/26, 9/26, 1/10, 2/10 | 20 / 170 | 0.2 / 0.5 pp | 0.04 pp |
| 0 | heldout | comp | 26 | 6/26, 11/26, 5/10, 4/10 | 40 / 210 | 1.1 / 8.0 pp | 0.38 pp |
| 0 | heldout | ca | 12 | 5/26, 6/26, 1/10, 0/10 | 20 / 70 | 0.2 / 0.7 pp | 0.03 pp |
| 0 | indist | comp | 28 | 10/26, 10/26, 5/10, 3/10 | 35 / 240 | 0.8 / 5.2 pp | 0.31 pp |
| 0 | indist | ca | 19 | 4/26, 8/26, 3/10, 4/10 | 50 / 150 | 0.5 / 4.3 pp | 0.14 pp |

T=1 held-out, comp vs honest: |dstep| distribution 10:8, 30:1, 40:2, 50:1, 70:2, 80:1, 90:1, 100:1, 130:1, 180:1, 240:1.
Steps often differ, but they sit on flat plateaus: the honest value at the comp-selected step is within 2 pp of the
honest-selected value in 19/20 changed runs. Largest losses: s20b-k3k-s4 150->50 (41.0 vs 37.2), f120b-k300/s120b-k3k-s1
~2 pp, s20b-k300-s1 50->120 (14.3 vs 12.8), f20b-k1k 200->110 (39.5 vs 38.2). The Qwen changes are all on <= 1.2 pp curves.
Selecting on `ca` is essentially identical to selecting on honest (max loss 0.6 pp at T=1).

## 2. Sigmoid values (finals, SFT / SFT+RL, held-out, T=1); delta = alt - honest at the RL point

| model | cell | honest | ca | comp | ca-honest | comp-honest |
|---|---|---|---|---|---|---|
| gpt-oss-20b | k100 | 3.0/1.8 | 3.0/1.8 | 3.0/1.8 | +0.0 | +0.0 |
| gpt-oss-20b | k300 | 9.0/11.0 | 9.5/11.2 | 10.0/11.3 | +0.2 | +0.3 |
| gpt-oss-20b | k500 | 18.5/32.7 | 19.0/32.8 | 19.0/33.7 | +0.2 | +1.0 |
| gpt-oss-20b | k700 | 21.0/37.3 | 23.5/38.2 | 23.5/39.3 | +0.8 | +2.0 |
| gpt-oss-20b | k1k | 20.0/39.5 | 21.5/40.0 | 22.0/40.5 | +0.5 | +1.0 |
| gpt-oss-20b | k3k | 27.5/45.7 | 28.5/46.2 | 28.5/46.8 | +0.5 | +1.2 |
| gpt-oss-20b | k10k | 21.0/50.0 | 24.0/50.0 | 25.5/53.2 | +0.0 | **+3.2** |
| gpt-oss-20b | k30k (p) | 25.0/40.7 | 26.5/41.2 | 27.0/41.8 | +0.5 | +1.2 |
| gpt-oss-20b | k100k (p) | 17.0/48.5 | 18.5/49.2 | 18.5/49.8 | +0.7 | +1.3 |
| gpt-oss-20b | kall (p) | 19.5/52.2 | 22.0/52.7 | 22.0/52.7 | +0.5 | +0.5 |
| gpt-oss-120b | k100 | 5.5/7.2 | 6.0/7.3 | 6.0/7.3 | +0.2 | +0.2 |
| gpt-oss-120b | k300 | 16.0/17.7 | 16.5/18.0 | 16.5/18.0 | +0.3 | +0.3 |
| gpt-oss-120b | k500 | 25.0/28.2 | 26.0/29.3 | 26.5/29.5 | +1.2 | +1.3 |
| gpt-oss-120b | k700 | 31.0/46.0 | 31.5/47.3 | 31.5/47.5 | +1.3 | +1.5 |
| gpt-oss-120b | k1k | 32.0/44.2 | 34.0/47.0 | 34.5/49.5 | +2.8 | **+5.3** |
| gpt-oss-120b | k3k | 27.5/47.2 | 30.5/47.2 | 31.0/47.2 | +0.0 | +0.0 |
| gpt-oss-120b | k10k | 31.0/51.3 | 33.0/51.5 | 33.0/52.7 | +0.2 | +1.3 |
| gpt-oss-120b | k30k | 30.0/58.3 | 32.5/58.7 | 33.0/58.7 | +0.3 | +0.3 |
| gpt-oss-120b | k100k (p) | 26.5/57.0 | 28.5/57.2 | 28.5/57.2 | +0.2 | +0.2 |
| gpt-oss-120b | kall | 23.5/37.2 | 27.5/37.8 | 27.5/38.0 | +0.5 | +0.8 |
| Qwen3-8B | k100..k1k | 0.0-0.5 / 0.2-1.0 | same | same | <= +0.2 | <= +0.2 |
| Qwen3-8B | k3k | 0.5/3.0 | 0.5/3.2 | 1.0/3.3 | +0.2 | +0.3 |
| Qwen3-8B | k10k | 3.0/7.8 | 3.0/8.2 | 3.0/9.3 | +0.3 | +1.5 |
| Qwen3-8B | k30k | 0.0/2.2 | 0.0/2.2 | 1.0/3.8 | +0.0 | +1.7 |
| Qwen3-8B | k100k | 0.0/7.3 | 0.0/7.3 | 0.5/7.8 | +0.0 | +0.5 |
| Qwen3-8B | kall | 0.5/0.7 | 0.5/0.7 | 0.5/0.8 | +0.0 | +0.2 |
| Qwen3-32B | k100..k10k | 0.0-3.0 / 0.2-2.2 | 0.0-3.0 / 0.2-2.2 | 0.0-3.5 / 0.2-3.5 | <= +0.5 | <= +2.0 (k1k) |
| Qwen3-32B | k30k | 9.5/20.8 | 10.0/21.0 | 10.0/21.3 | +0.2 | +0.5 |
| Qwen3-32B | k100k | 2.0/27.3 | 2.0/27.3 | 2.0/27.5 | +0.0 | +0.2 |
| Qwen3-32B | kall (p) | 10.0/22.7 | 10.5/23.0 | 13.0/23.0 | +0.3 | +0.3 |

Flags (|comp-honest| at RL point): T=1 held-out 2/40 cells > 3 pp, 1/40 > 5 pp (f120b-k1k: step 70, raw comp .525,
ans .935, ca .470, honest .455 -- a 6.5 pp unanswered + 1.5 pp hollow slice at one step). T=1 in-dist: 10/40 > 3 pp, 0 > 5 pp
(max +4.5 fq32b-kall, +4.3 f20b-k700 / fq8b-k700). RL-gain (RL - SFT) sign and magnitude agree within 5 pp in all 40
held-out cells under all three metrics.

T=0 contrast, held-out comp-honest at RL point (SFT point in parentheses):

| model | k100 | k300 | k500 | k700 | k1k | k3k | k10k | k30k | k100k | kall |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-20b | +0.5 (0.0) | +1.2 (1.0) | **+8.5** (2.0) | **+5.8** (4.5) | +4.8 (3.5) | +2.8 (2.5) | +3.8 (3.0) | **+11.2** (3.5) | **+10.0** (2.5) | **+10.5** (4.0) |
| gpt-oss-120b | +0.2 (0.0) | +0.7 (0.5) | +0.7 (0.5) | +2.0 (0.5) | +4.3 (2.0) | +1.3 (4.0) | +1.7 (2.0) | +0.0 (4.5) | +2.2 (2.0) | +2.0 (3.5) |
| Qwen3-8B | 0 | 0 | 0 | 0 | 0 | +0.5 | +3.8 (4.0) | +0.5 | +3.8 | 0 |
| Qwen3-32B | 0 | 0 | **+8.0** (9.5) | +1.5 | +3.5 (3.0) | +0.3 (1.0) | 0 | +2.0 (2.0) | +3.2 | +3.0 (2.5) |

Gap summary (comp-honest at RL point, mean / max / #cells > 3 pp / > 5 pp): T=0 held-out 2.6 / 11.2 / 13 / 6;
T=1 held-out 0.8 / 5.3 / 2 / 1. T=0 in-dist 3.5 / 12.3 / 18 / 12; T=1 in-dist 1.7 / 4.5 / 10 / 0. The T=0 gaps were
concentrated in gpt-oss-20b k >= 500 (loops/filler under greedy decoding); at T=1 they collapse to ~1 pp.

## 3. Seed bars (5 seeds, held-out): mean (sd), SFT / SFT+RL

| T | model | cell | honest | ca | comp | d ca | d comp | RL seed order comp vs honest |
|---|---|---|---|---|---|---|---|---|
| 1 | gpt-oss-20b | k100 | 1.8(1.9) / 2.8(3.2) | 2.2 / 2.9(3.1) | 2.2 / 2.9(3.1) | +0.1 | +0.1 | same |
| 1 | gpt-oss-20b | k300 | 5.9(2.7) / 9.6(4.8) | 6.3 / 9.9(4.9) | 6.4 / 10.4(5.3) | +0.3 | +0.8 | same |
| 1 | gpt-oss-20b | k1k | 20.4(2.7) / 39.5(3.1) | 22.0 / 40.0(2.9) | 22.2 / 40.8(3.4) | +0.5 | +1.3 | same |
| 1 | gpt-oss-20b | k3k | 24.9(2.1) / 43.8(4.6) | 26.6 / 44.4(4.9) | 27.8 / 45.8(4.9) | +0.6 | +2.0 | same |
| 1 | gpt-oss-120b | k100 | 7.5(5.5) / 8.8(4.7) | 8.1 / 9.2(5.0) | 8.1 / 9.2(5.0) | +0.4 | +0.4 | seeds 0,2 swap (7.0 vs 7.0 tie) |
| 1 | gpt-oss-120b | k300 | 15.6(4.1) / 19.8(3.2) | 17.3 / 21.0(4.0) | 17.3 / 21.0(4.0) | +1.2 | +1.2 | same |
| 1 | gpt-oss-120b | k1k | 25.4(4.2) / 34.6(5.9) | 27.3 / 35.9(7.0) | 27.6 / 36.8(7.8) | +1.2 | +2.1 | same |
| 1 | gpt-oss-120b | k3k | 28.9(3.5) / 43.4(4.5) | 31.2 / 43.6(4.5) | 31.8 / 44.5(4.7) | +0.2 | +1.1 | same |
| 0 | gpt-oss-20b | k1k | 24.8(2.5) / 48.3(4.8) | 27.1 / 52.3(4.5) | 29.0 / 56.1(3.7) | **+4.0** | **+7.8** | changes (seeds 0 and 2 swap) |
| 0 | gpt-oss-20b | k3k | 31.6(3.3) / 50.7(5.1) | 34.1 / 54.4(5.8) | 35.4 / 55.8(5.3) | **+3.6** | **+5.1** | changes (top-2 swap) |
| 0 | other 6 cells | | | | | <= +1.7 | <= +2.7 | k100 cells: minor swaps |

T=1: no cell's 5-seed RL mean moves > 2.1 pp under either alternative; seed sds change by <= 1.9 pp; ordering is preserved
in 7/8 cells (the exception is a tie on a ~1 pp curve). T=0: gpt-oss-20b k1k/k3k moved 5-8 pp under comp and re-ordered seeds.

## 4. Qwen floor cells (T=1, held-out, selected step)

No cell has non-trivial raw compliance with honest ~0, and none has honest > comp. All Qwen3-8B held-out cells and
Qwen3-32B k <= 10k cells are < 4 pp under every metric; the largest comp-honest spreads are fq32b-k500 (comp 2.5 / ca 0.8 /
honest 0.7, acc 48), fq32b-k1k (3.5 / 1.8 / 1.2 at the comp-best step 250, acc 45) and fq8b-k30k (3.8 / 1.3 / 1.3 at
the comp-best step 60, acc 42): a 1-2.5 pp slice of compliant-but-unanswered generations. The non-floor Qwen cells agree
to <= 0.5 pp: fq32b-k30k 21.3/21.0/20.8, k100k 27.5/27.3/27.3, kall 23.0/23.0/22.7; fq8b-k10k 9.3/8.2/7.8, k100k 7.8/7.3/7.3.
Even scanning every step (gate ignored) for the max smoothed comp, no Qwen cell exceeds honest by more than 2.5 pp.
In-dist (secondary): comp-honest up to +4.5 pp (fq32b-kall 81.5 vs 77.0, fq8b-k700 28.0 vs 23.7); ca within 4.2 pp.

## 5. Conclusion

At T=1 the choice of honest vs ca vs comp changes no qualitative claim:

- **Sigmoid shape / saturation.** Per model and metric, held-out T=1: gpt-oss-20b max RL 52-53 pp, 90%-of-max reached
  at k10k under all three; gpt-oss-120b max 58.3-58.7 at k30k, 90% at k30k under all three; Qwen3-8B max 7.8-9.3 at k10k;
  Qwen3-32B max 27.3-27.5 at k100k. (At T=0 the comp curve for gpt-oss-120b peaked at k1k instead of k3k and the
  20b curve was 4 pp higher at its peak.)
- **RL gain over SFT.** Mean held-out gain across the 10 cells: 20b +17.8/+16.7/+17.2, 120b +14.6/+13.5/+13.8,
  q8b +1.9/+1.9/+2.1, q32b +5.0/+5.0/+4.8 (honest/ca/comp). No cell changes gain sign; no cell's gain differs by > 5 pp.
- **Seed spread.** 5-seed means move <= 2.1 pp and sds <= 1.9 pp; ordering unchanged except a tie.
- **Checkpoint selection.** 20/72 runs pick a different step under comp (10/72 under ca) but the alternative step costs
  0.21 pp honest on average because selections move along plateaus.

Worst-case cells (T=1, held-out): f120b-k1k (comp 49.5 vs honest 44.2 at step 70; +5.3 pp, driven by 6.5 pp unanswered
generations at that single step), f20b-k10k (53.2 vs 50.0, +3.2), s20b-k3k-s4 (comp selection would cost 3.8 pp honest),
and in-dist fq32b-kall / f20b-k700 / fq8b-k700 (+4.3 to +4.5). None of these alters a figure's reading.
`ca` is a near-exact proxy for honest at T=1 (max held-out deviation 2.8 pp, selection loss <= 0.6 pp); the honest
metric's extra hollow-trace filter matters only at T=0 and primarily for gpt-oss-20b at k >= 500.
