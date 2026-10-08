# sweep29 kall held-out decline — adapter / optimization side

Scope: gpt-oss-120b GRPO (sweep29, 200 it), rank-1 attention LoRA warm-started from few-shot SFT donors. Full-LoRA cell
`sweep29-fs1-kall` (lr 3.3e-4, **746,496** trainable scalars — 36 layers x {q,k,v,o} x (A+B); 497,664 is the 20b count)
vs masked cells k100k (lr 3.16e-3) and k30k (lr 5.77e-3), and the 20b references `sweep24-fix05-kall` / `-k100k` (250 it).
Adapters pulled read-only every 10 steps from Modal volume `cotcontrol_sebastian-prasanna_checkpoints` to `/tmp/adapters29/`,
plus each donor SFT lineage's `checkpoint-0` (PEFT init: B=0, A random) so the RL delta can be compared with the SFT delta.
Scripts/data in this dir: `adapter_drift.py/.json`, `sft_direction.py/.json`, `metrics_analysis.py/.json`,
`heldout_series.py/.json`, `heldout_trend.py/.json`, merged `adapter_side.json`.

Held-out metric used below: strict binary `compliance` on `start_of_sentence` (SoS, n=72/ckpt, T=0, val split); the
sweep's own 3-mode held-out `compliance_rate` (n=200) is reported alongside. kall-120b: SoS .667 -> .125, 3-mode .42 -> .16.

## 1. Update magnitude (||θ_t − θ_0||_F / ||θ_0||_F, all LoRA params)

| run | k | lr | lr·√k | ‖θ0‖ | rel drift @50 / 100 / 150 / 200 (250) | per-10-step ‖Δθ‖ (steps 10 / typical) | q / k / v / o @200 | early / mid / late @200 |
|---|---|---|---|---|---|---|---|---|
| 120b kall | 746,496 | 3.3e-4 | 0.29 | 29.2 | .079 / .108 / .129 / **.149** | 1.08 / 0.47–0.66 | .213 / .210 / .143 / .112 | .188 / .185 / .111 |
| 120b k100k | 100,000 | 3.16e-3 | 1.00 | 75.8 | .108 / .148 / .180 / **.220** | 3.94 / 1.9–3.2 | .331 / .412 / .225 / .152 | .299 / .277 / .158 |
| 120b k30k | 30,000 | 5.77e-3 | 1.00 | 91.2 | .092 / .129 / .162 / **.194** | 4.00 / 1.9–3.1 | .322 / .376 / .215 / .132 | .276 / .250 / .135 |
| 20b kall | 497,664 | 3.3e-4 | 0.23 | 22.7 | .087 / .121 / .149 / **.169** (.183) | 0.91 / 0.44–0.56 | .229 / .214 / .155 / .136 | .201 / .224 / .126 |
| 20b k100k | 100,000 | 3.16e-3 | 1.00 | 58.5 | .139 / .190 / .233 / **.289** (.356) | 3.75 / 2.0–3.9 | .389 / .401 / .278 / .219 | .345 / .367 / .219 |

Same picture when normalising by the SFT delta ‖θ0 − θ_init‖ instead of ‖θ0‖ (removes the random A-init from the
denominator): RL/SFT at step 200 = kall-120b .153, k100k .221, k30k .19, 20b-kall .174, 20b-k100k .290.

Findings
- **kall drifts LESS, not more**: relative drift .149 vs .220/.194 for the masked 120b cells; absolute per-step update is
  ~0.25x the masked cells' (0.05 vs 0.20 per step), matching the drive ratio 0.29/1.0. Adam makes the per-scalar step
  ≈ 0.18–0.20·lr in every run (per-scalar RMS motion per step: kall 5.8e-5 = 0.18·lr; k100k 6.3e-4 = 0.20·lr), so total
  update ∝ lr·√k exactly as the drive heuristic assumes. Logged grad norms confirm the per-scalar gradient is identical:
  kall .0098 vs k100k .0036 ≈ .0098·√(100k/746k).
- Module/depth structure is the same in all five runs: q,k move most (k_proj most in masked cells), o least; early ≥ mid > late.
  No module class or depth band singles out kall-120b. 20b-kall and 120b-kall are near-identical on every magnitude metric
  (drift .169 vs .149, d10 ≈ 0.5 for both).
- All runs share a 2x larger first-10-step update (no warmup, Adam bias-corrected first steps); held-out is unaffected at step 10.
- Per-10-step norms are flat over training for kall (0.47–0.66) — no regime change around steps 40–120 where held-out falls fastest;
  the masked cells' d10 actually rises ~30% after step 120.

## 2. Structural: effective ΔW = B·Aᵀ (rank-1, exact closed forms per module)

| run | cos(ΔW_t, ΔW_0) mean @100 / @200 (norm-weighted @200; min @200) | ‖ΔW_t − ΔW_0‖/‖ΔW_0‖ median @200 | ‖ΔW_t‖/‖ΔW_0‖ median @200 | donor ‖ΔW_0‖ median | donor RMS per moved scalar | RL RMS per scalar @200 | B support (frac output rows ΔW can write) |
|---|---|---|---|---|---|---|---|
| 120b kall | .978 / **.960** (.975; .863) | .28 | 1.04 | 1.54 | .033 | .0050 | 100 % |
| 120b k100k | .933 / **.866** (.952; .431) | .47 | 1.10 | 7.84 | .239 | .053 | 13.4 % (fixed) |
| 120b k30k | .930 / **.865** (.962; .317) | .46 | 1.07 | 8.10 | — | — | ~4 % (fixed) |
| 20b kall | .974 / **.951** (.969; .865) | .32 | 1.05 | 1.52 | .031 | .0054 | 100 % |
| 20b k100k | .922 / **.843** (.918; .562) | .62 | 1.17 | 8.68 | .184 | .053 | 20.1 % (fixed) |

- **Masked cells rotate the donor's per-module ΔW more than kall does** (cos .87 vs .96; relative change .47 vs .28). The
  hypothesis "kall rotates the donor direction while masked cells scale/perturb it" is **not supported**; if anything the
  reverse holds. In every run the most-rotated modules are mid-depth k_proj (and o_proj in kall-120b), with small donor ‖ΔW_0‖.
- RL delta vs SFT delta: cos(θ_t − θ_0, θ_0 − θ_init) is |·| < .035 in all runs (A and B separately too). RL neither undoes
  nor amplifies the SFT direction — it adds a nearly orthogonal component everywhere. 40–60 % of modules shrink slightly
  along their donor direction in every run; no kall-specific "unlearning" signature.
- The genuine structural asymmetry is **support, not magnitude**. In a masked cell B entries outside the mask are exactly 0
  forever (verified: support constant over all checkpoints), so ΔW can only ever write to a fixed 4–13 % of each module's
  output channels, and RL and SFT share exactly that support. In kall every scalar of A and B moves every step
  (746,496/746,496 changed at step 10), so RL writes a dense, low-amplitude (RMS .005/scalar), SFT-orthogonal perturbation
  into 100 % of output channels of all 144 projections. The kall donor is also a 5–7x *smaller* object in weight space
  (median ‖ΔW_0‖ 1.5 vs 7.8; per-scalar .033 vs .24), so its held-out skill rides on a dense, small-amplitude delta.
- But the same is true of 20b kall, which held. Adapter statistics therefore cannot explain the 120b-specificity; whatever
  differs is in how the 120b base responds to that dense perturbation (behavioural side: kall-120b's in-distribution CoT
  length grew 2.4x, 2205 -> 5311 median chars, the largest of all cells; k100k 1.3x, k30k 1.7x, 20b-kall 1.9x).

## 3. Optimizer signals (metrics.jsonl, 40-step windows)

| run | grad_norm 0–40 -> 160–200 | lr·grad_norm | clip_frac | logprob_corr 0–40 -> 160–200 | logprob |Δ| mean | TIS-truncated frac | reward 0–40 -> 160–200 | in-dist acc | in-dist compliance | reasoning chars median | degenerate groups/iter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 120b kall | .0097 -> .0109 (flat) | 3.6e-6 | 0 | .995 -> .977 | .031 -> .034 | .0005 -> .0017 | 1.60 -> 2.12 | .86 -> .93 | .54 -> .68 | 2205 -> 5311 | 2.2 -> 10.5 |
| 120b k100k | .0036 -> .0084 (2.3x) | 2.7e-5 | 0 | .995 -> .973 | .037 -> .049 | .0009 -> .0032 | 1.58 -> 2.10 | .88 -> .77 | .48 -> .75 | 2370 -> 3178 | 2.4 -> 9.2 |
| 120b k30k | .0017 -> .0035 (2x) | 2.0e-5 | 0 | .996 -> .971 | .036 -> .041 | .0007 -> .0024 | 1.47 -> 2.03 | .89 -> .84 | .41 -> .68 | 2074 -> 3453 | 1.4 -> 7.5 |
| 20b kall | .0100 -> .0149 (1.5x) | 4.9e-6 | 0 | .994 -> .989 | .016 -> .024 | .0002 -> .0008 | 1.45 -> 2.08 | .67 -> .77 | .57 -> .75 | 1950 -> 3657 | 0.5 -> 9.0 |

- max_grad_norm 1.0 is never hit (grad norms are ~1e-2); clip_frac is 0 in all runs (one inner epoch, on-policy); TIS ratio
  mean = 1.000. No run shows a step-size regime change; kall's grad norm is the flattest of all.
- kall's logged per-step "effective step" (lr × grad_norm, 3.6e-6) is the smallest of the four, and its realised ‖Δθ‖ per
  step (0.05) is 4x smaller than the masked cells' — consistent with Section 1.
- Sampler/trainer mismatch (logprob_corr .995 -> .977) degrades equally in kall and masked 120b cells; the 20b cells degrade less.
- kall's distinguishing in-distribution signals are behavioural: highest accuracy (.93), the largest CoT lengthening (2.4x),
  the most degenerate (all-equal-reward) groups by the end. Nothing on the optimizer side is anomalous.

## 4. Trajectory shape of the held-out series (21 checkpoints, 0..200)

| run | SoS start -> end | lin. slope /100 | R² | Spearman ρ (p) | 1-step signs | 5-ckpt-lag signs | VR(5) | z of total change | 3-mode start -> end (ρ) |
|---|---|---|---|---|---|---|---|---|---|
| 120b kall | .667 -> .125 | **−.29** | **.83** | **−.87 (2e-7)** | 11− / 7+ | **14− / 2+** | .94 | **−6.6** | .42 -> .16 (−.79) |
| 120b k100k | .625 -> .903 | +.11 | .67 | +.77 (4e-5) | 6− / 13+ | 4− / 11+ | .53 | +3.9 | .36 -> .52 (+.49) |
| 120b k30k | .681 -> .819 | +.09 | .37 | +.55 (.009) | 6− / 10+ | 4− / 11+ | 1.35 | +1.9 | .41 -> .34 (+.27) |
| 120b k10k | .778 -> .931 | +.06 | .68 | +.82 (6e-6) | 6− / 10+ | 3− / 12+ | .16 | +2.6 | .44 -> .59 (+.69) |
| 120b k3k / k1k | .69 -> .89 / .74 -> .83 | +.05 / +.03 | .27 / .31 | +.43 / +.51 | | | | +2.9 / +1.4 | |
| 20b kall (26 ckpts) | .597 -> .833 | −.01 | .01 | −.07 (.75) | 10− / 10+ | 11− / 9+ | .42 | +3.1 | .27 -> .45 (+.06) |
| 20b k100k (26 ckpts) | .681 -> .681 | −.01 | .00 | +.13 (.52) | 11− / 10+ | 7− / 11+ | .70 | 0.0 | .27 -> .33 (−.46) |

- kall-120b's decline is a strong monotone trend by every test (R² .83; ρ −.87; 14 of 16 five-checkpoint differences negative,
  binomial p ≈ .002; total change −6.6 binomial SEs, i.e. far beyond eval sampling noise). All three held-out modes move
  together: letter_suppression .155 -> 0 by step 50 (ρ −.77), no_spaces .39 -> .06 by step 100 (then rebounds to .33 at 200),
  SoS .67 -> .12. The masked 120b cells all rise on SoS (ρ +.43…+.82); 20b-kall is flat/rising.
- Caveat on "trend vs random walk": a permutation test that shuffles the 10-step increments gives R² ≥ .83 in ~60 % of
  shuffles (a persistent random walk also produces high R²), and VR(5) ≈ .9 is compatible with either a drift or a random
  walk — but *not* with i.i.d. eval noise around a stable policy (VR(5) would be ≈ .2). So the series proves the policy really
  moved far and in one direction; whether the *sign* was drawn once (seed) or is forced by the recipe is exactly what n = 1
  cannot answer. Note that a masked 120b cell (k30k) also lost one held-out mode monotonically (no_spaces .30 -> .06,
  z = −3.8) while gaining SoS — single-mode held-out losses are not unique to full LoRA; losing all three is.
- Steepest SoS drops: 10 -> 50 (.69 -> .35) and 90 -> 120 (.38 -> .08); nothing in d10, grad norm, or logprob mismatch changes
  there, while in-dist CoT length rises steadily throughout (2.2k -> 3.6k -> 4.8k -> 5.3k chars by window).

## 5. Conclusion

1. The adapter-side evidence does **not** support an "over-large or rotating update" explanation: kall-120b is the *least*
   perturbed adapter by every relative measure (drift .149 vs .19–.22; ΔW cos .96 vs .87; RL/SFT .15 vs .22), its optimizer
   signals are the calmest, and it is numerically indistinguishable from 20b-kall, which held its held-out skill.
2. RL's delta is orthogonal to the SFT delta in every run; nothing kall-specific "unlearns" the donor direction.
3. The only structural asymmetry is support: full LoRA lets Adam write a dense, fixed-size (≈0.2·lr per scalar per step),
   reward-agnostic perturbation into 100 % of channels of all 144 projections, on top of a donor whose delta is 5–7x smaller
   per scalar than the masked donors'; masked RL is confined to the same 4–13 % of rows the donor used. That is a plausible
   mechanism for erasing a transferred style while in-dist reward keeps rising, but it is identical for 20b-kall, so
   120b-specificity must come from the base model's response (and the 2.4x CoT lengthening is the behavioural correlate),
   not from the adapter dynamics.
4. The held-out decline itself is a real, monotone, three-mode change (z −6.6), not eval noise — but sweep29-kall is the
   **only** gpt-oss-120b full-LoRA RL run in the repo (checked all 120b configs with `train_params: null`), so systematic
   vs stochastic is genuinely open.
5. Single follow-up that settles it: **rerun sweep29-fs1-kall with `seed: 1`** (same donor, lr, recipe). A second monotone
   decline ⇒ systematic; a rise ⇒ seed-level fluke. If budget allows two cells, add **kall at lr 1e-4** (drive .09): if it
   still declines, magnitude is not the lever and the dense-support mechanism is implicated; a cheap no-training complement is
   adapter splicing (donor + RL-200 delta in one depth tercile / one module type, 6 held-out evals) to localise the loss.
