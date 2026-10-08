# sweep28 gpt-oss-120b high-k RL: training-side correlates of the held-out start_of_sentence oscillation

Runs: `rl/runs/sweep28_gptoss120b_highk/sweep28-d1-k30k`, `sweep28-d1-k100k` (GRPO, masked rank-1 LoRA, fixed λ=0.5, 250 iters, 32 prompts x 8 rollouts, T=1, 6 in-dist modes on math_l45 train).
Target series: held-out `start_of_sentence` compliance (T=0, n=72 per checkpoint, SE ≈ 0.06) from `eval_summary.json` at steps 0,10,...,250.
Analysis code: `extract_features.py` (per-rollout stats from iters/+batches/), `build_features.py` (per-iteration table -> `training_features.json`, 209 features x 250 iters x 2 runs), `correlate.py`, `tables.py`, `robust.py`. Raw outputs in `cache/`.

Alignment: checkpoint-s is produced after batches 0..s-1 and iter-s rollouts are sampled *from* checkpoint-s, so the window W0 = iterations s-9..s mixes the last 10 training batches with the first readout of the evaluated adapter. Also tested: Wtrain = s-10..s-1 (pure inputs), Wlag = s-19..s-10, Wfwd = s+1..s+10 (pure readout), point = iter s. Checkpoints s=10..250 (n=25 per run).

## 0. Two structural facts that constrain every hypothesis

1. **Both runs saw identical batches.** `cfg.seed=0` in both, and `run_grpo.py` draws prompts with `subsample_seed = seed*7919 + t` and modes with `seed*100003 + t`, so all 250 iterations have the *same 32 prompts and the same mode assignment* in k30k and k100k (verified: 250/250 identical qid sets and mode sequences). Yet the two held-out series are essentially unrelated: r=+0.13, rho=+0.10. Any explanation based on *what was sampled* (mode mix, specific prompts, sentence-mode share) is therefore ruled out as the primary driver — it would move both runs together.
2. **There is no epoch.** Each iteration is an independent shuffle of the 2176-prompt train split (`load_dataset(..., subsample_seed=t)`), not a pass through a permutation. 2126/2176 distinct prompts were visited; on average only 4.3 of 32 prompts per iteration appeared in the previous 10 iterations and ~29/32 had been seen before by iter 100 (both exactly what iid draws predict). The mode-share series has zero autocorrelation at every lag (share_eos ACF at lags 1,2,5,10,34,50,68,100 all |ACF|<0.06). The nominal "epoch length" 2176/32 = 68 steps has no physical meaning here. The held-out series' own dominant FFT periods are ~130 and ~52 steps (k30k) / ~130 and ~87 (k100k); nothing lines up with 68.

## 1. Correlation table

Permutation calibration (2000 shuffles of the held-out series, all 206 features): the 95th percentile of max-over-features of min(|r_k30k|,|r_k100k|) at W0 is **0.45** (99th: 0.50). For a single run, the 95th percentile of max|r| over features is **0.63-0.64** (99th: 0.70). So a feature needs |r|≳0.45 *in both runs with the same sign* or |r|≳0.63 in one run to be more than multiple-comparison noise.

### Requested candidates (Pearson r; W0 = s-9..s, Wlag = s-19..s-10, Wfwd = s+1..s+10, point = iter s)
| feature | W0 k30k | W0 k100k | Wlag k30k | Wlag k100k | Wfwd k30k | Wfwd k100k | point k30k | point k100k |
|---|---|---|---|---|---|---|---|---|
| share_eos | -0.05 | -0.04 | -0.30 | -0.34 | +0.19 | +0.16 | -0.03 | -0.05 |
| share_rep | +0.04 | +0.22 | +0.31 | +0.26 | -0.33 | +0.06 | +0.27 | +0.27 |
| share_sent_modes | -0.00 | +0.22 | +0.06 | -0.03 | -0.21 | +0.22 | +0.21 | +0.20 |
| share_case | -0.03 | -0.23 | -0.04 | -0.13 | +0.10 | -0.11 | -0.07 | -0.39 |
| share_meow | +0.07 | +0.15 | -0.02 | +0.29 | +0.11 | -0.09 | -0.14 | +0.29 |
| n_posadv_end_of_sentence | +0.29 | -0.33 | +0.04 | -0.32 | +0.53 | -0.20 | +0.07 | -0.22 |
| n_posadv_sent_modes | +0.09 | -0.15 | +0.07 | -0.22 | +0.09 | -0.09 | -0.13 | +0.03 |
| comp_end_of_sentence | +0.44 | -0.39 | +0.41 | -0.29 | +0.58 | -0.53 | +0.33 | -0.39 |
| comp_repeat_sentences | +0.39 | -0.25 | +0.43 | -0.38 | +0.54 | -0.13 | +0.40 | -0.53 |
| reward_end_of_sentence | +0.42 | -0.53 | +0.45 | -0.45 | +0.52 | -0.56 | +0.20 | -0.57 |
| comp | +0.49 | -0.28 | +0.48 | -0.16 | +0.57 | -0.48 | +0.50 | -0.18 |
| acc | -0.39 | -0.50 | -0.21 | -0.51 | -0.68 | -0.33 | -0.33 | -0.30 |
| reward_mean | +0.46 | -0.42 | +0.49 | -0.33 | +0.45 | -0.55 | +0.41 | -0.44 |
| reward_std | +0.21 | +0.33 | +0.21 | +0.08 | +0.17 | +0.55 | +0.10 | +0.52 |
| within_group_reward_std | -0.17 | +0.15 | -0.24 | +0.12 | -0.25 | +0.25 | -0.27 | +0.39 |
| frac_groups_degenerate | +0.27 | +0.02 | +0.38 | +0.11 | +0.26 | -0.11 | +0.43 | -0.27 |
| all_chars | +0.31 | -0.05 | +0.57 | -0.15 | -0.06 | +0.08 | +0.10 | -0.19 |
| all_n_sent | +0.24 | +0.14 | +0.57 | -0.03 | -0.20 | +0.37 | -0.01 | -0.15 |
| all_term_per_k | -0.08 | -0.01 | -0.26 | -0.28 | +0.07 | +0.30 | +0.15 | -0.11 |
| all_nl_per_k | -0.14 | +0.70 | -0.11 | +0.60 | -0.16 | +0.69 | +0.05 | +0.65 |
| all_frac_sent_i | +0.39 | -0.19 | +0.27 | -0.08 | +0.58 | -0.35 | +0.49 | -0.19 |
| all_frac_sent_ok | -0.26 | +0.18 | -0.16 | +0.11 | -0.43 | +0.36 | -0.32 | +0.10 |
| all_frac_sent_we | -0.27 | +0.08 | +0.00 | -0.15 | -0.50 | +0.34 | -0.26 | +0.15 |
| all_has_meta | +0.52 | -0.30 | +0.45 | -0.26 | +0.58 | -0.36 | +0.53 | -0.36 |
| all_meta_per_k | +0.15 | -0.15 | +0.09 | +0.01 | +0.39 | -0.38 | +0.28 | -0.18 |
| all_frac_sent_meta | +0.18 | -0.29 | +0.06 | -0.15 | +0.42 | -0.59 | +0.27 | -0.32 |
| all_first_sent_meta | +0.49 | -0.14 | +0.42 | -0.09 | +0.57 | -0.17 | +0.55 | -0.19 |
| all_first_math_idx | +0.28 | -0.13 | +0.45 | +0.04 | +0.15 | -0.28 | +0.14 | -0.30 |
| all_d4 | -0.51 | -0.12 | -0.61 | -0.09 | -0.36 | -0.16 | -0.40 | +0.02 |
| case_chars | +0.33 | -0.00 | +0.56 | -0.10 | -0.13 | +0.13 | -0.11 | -0.19 |
| case_n_sent | +0.24 | +0.17 | +0.49 | +0.04 | -0.28 | +0.38 | -0.21 | -0.02 |
| case_frac_sent_i | +0.29 | -0.08 | +0.25 | -0.00 | +0.47 | -0.23 | +0.46 | -0.02 |
| case_meta_per_k | +0.14 | -0.18 | +0.17 | -0.00 | +0.38 | -0.39 | +0.17 | -0.13 |
| case_has_meta | +0.45 | -0.44 | +0.46 | -0.36 | +0.46 | -0.45 | +0.36 | -0.52 |
| case_frac_sent_meta | +0.16 | -0.17 | +0.19 | -0.03 | +0.37 | -0.37 | +0.25 | -0.14 |
| case_d4 | -0.42 | -0.21 | -0.66 | -0.15 | -0.26 | -0.22 | -0.24 | -0.15 |
| eos_chars | +0.06 | +0.03 | +0.18 | +0.03 | -0.19 | +0.19 | -0.11 | +0.04 |
| eos_meta_per_k | +0.13 | -0.24 | +0.25 | -0.14 | +0.24 | -0.48 | +0.18 | -0.21 |
| eos_frac_sent_meta | -0.06 | -0.17 | +0.16 | -0.08 | -0.01 | -0.49 | -0.05 | -0.18 |
| eos_n_sent | +0.17 | -0.35 | +0.34 | -0.31 | -0.07 | -0.32 | +0.03 | -0.22 |
| advcorr_chars | -0.21 | +0.11 | -0.51 | +0.15 | +0.27 | -0.05 | -0.27 | +0.24 |
| advcorr_n_sent | -0.15 | -0.06 | -0.53 | +0.03 | +0.39 | -0.28 | -0.14 | -0.00 |
| advcorr_meta_per_k | +0.06 | +0.11 | +0.04 | +0.06 | -0.32 | +0.34 | -0.14 | -0.21 |
| advcorr_frac_sent_meta | +0.21 | +0.19 | +0.03 | +0.20 | -0.17 | +0.32 | -0.01 | -0.06 |
| advcorr_frac_sent_i | -0.40 | +0.33 | -0.32 | +0.30 | -0.82 | +0.55 | -0.56 | +0.03 |
| advcorr_d4 | +0.10 | +0.17 | +0.31 | +0.12 | -0.24 | +0.25 | +0.15 | +0.06 |
| advw_chars | +0.02 | -0.02 | -0.10 | -0.20 | +0.29 | -0.31 | -0.24 | -0.29 |
| advw_meta_per_k | -0.11 | +0.03 | -0.11 | +0.04 | -0.54 | +0.11 | -0.18 | +0.15 |
| m_loss | -0.03 | -0.08 | +0.38 | +0.01 | -0.18 | -0.07 | +0.01 | +0.01 |
| m_grad_norm | +0.42 | +0.41 | +0.30 | +0.36 | +0.51 | +0.41 | +0.49 | +0.48 |
| m_reward_std | +0.21 | +0.33 | +0.21 | +0.08 | +0.17 | +0.55 | +0.10 | +0.52 |
| m_clip_frac | +nan | +nan | +nan | +nan | +nan | +nan | +nan | +nan |
| m_tis_ratio_mean | +0.23 | -0.28 | +0.33 | -0.30 | +0.05 | -0.42 | +0.15 | -0.05 |
| m_logprob_corr | -0.27 | -0.14 | -0.37 | -0.08 | -0.32 | -0.20 | -0.18 | -0.10 |
| m_logprob_diff_abs_mean | +0.42 | +0.45 | +0.28 | +0.31 | +0.60 | +0.51 | +0.34 | +0.47 |
| m_n_degenerate_groups | +0.27 | +0.02 | +0.38 | +0.11 | +0.26 | -0.11 | +0.43 | -0.27 |
| m_n_dropped_truncated | +nan | +nan | +nan | +nan | +nan | +nan | +nan | +nan |
| m_global_completion_tokens | -0.07 | -0.17 | +0.29 | -0.28 | -0.39 | -0.02 | -0.43 | -0.37 |
| p_distinct4_mean | -0.63 | -0.16 | -0.63 | -0.08 | -0.54 | -0.21 | -0.47 | -0.19 |
| p_zlib_ratio_median | -0.43 | -0.21 | -0.47 | -0.00 | -0.38 | -0.27 | -0.21 | -0.27 |
| frac_trunc | -0.30 | -0.00 | +0.05 | +0.15 | -0.46 | +0.19 | -0.28 | -0.02 |
| n_qid_seen_prev10 | +0.44 | -0.30 | +0.40 | -0.37 | +0.17 | -0.21 | +0.27 | -0.05 |
| n_qid_seen_ever | +0.44 | -0.00 | +0.43 | +0.02 | +0.55 | -0.01 | +0.47 | -0.01 |


### Top 40 (feature, window) pairs ranked by mean |r| over the two runs (same-sign required for full score)
| feature | window | r k30k | rho k30k | r k100k | rho k100k | r pooled | rho pooled |
|---|---|---|---|---|---|---|---|
| m_logprob_diff_abs_mean | Wfwd[s+1,s+10] | +0.60 | +0.63 | +0.51 | +0.44 | +0.55 | +0.53 |
| m_tis_frac_truncated | Wfwd[s+1,s+10] | +0.65 | +0.66 | +0.40 | +0.46 | +0.52 | +0.55 |
| acc | Wfwd[s+1,s+10] | -0.68 | -0.70 | -0.33 | -0.31 | -0.51 | -0.47 |
| m_logprob_diff_mean | Wfwd[s+1,s+10] | -0.54 | -0.55 | -0.46 | -0.47 | -0.50 | -0.51 |
| m_grad_norm | point[s] | +0.49 | +0.48 | +0.48 | +0.40 | +0.48 | +0.45 |
| eos_chars_before_digit | Wfwd[s+1,s+10] | -0.44 | -0.37 | -0.49 | -0.43 | -0.47 | -0.38 |
| m_grad_norm | Wfwd[s+1,s+10] | +0.51 | +0.59 | +0.41 | +0.43 | +0.46 | +0.49 |
| sent_frac_sent_so | Wfwd[s+1,s+10] | -0.74 | -0.75 | -0.18 | -0.32 | -0.46 | -0.47 |
| acc_alternating_case | Wfwd[s+1,s+10] | -0.63 | -0.65 | -0.26 | -0.06 | -0.45 | -0.34 |
| acc | W0[s-9,s] | -0.39 | -0.42 | -0.50 | -0.53 | -0.44 | -0.49 |
| advcorr_frac_sent_we | point[s] | +0.24 | +0.19 | +0.65 | +0.51 | +0.44 | +0.34 |
| advw_term_per_k | Wfwd[s+1,s+10] | -0.33 | -0.30 | -0.55 | -0.09 | -0.44 | -0.22 |
| advcorr_frac_sent_we | Wfwd[s+1,s+10] | +0.38 | +0.38 | +0.49 | +0.28 | +0.44 | +0.36 |
| m_logprob_diff_abs_mean | W0[s-9,s] | +0.42 | +0.40 | +0.45 | +0.49 | +0.43 | +0.48 |
| acc | Wtrain[s-10,s-1] | -0.35 | -0.41 | -0.51 | -0.54 | -0.43 | -0.50 |
| acc_alternating_case | point[s] | -0.54 | -0.57 | -0.31 | -0.25 | -0.42 | -0.40 |
| advw_frac_sent_we | point[s] | +0.32 | +0.29 | +0.52 | +0.29 | +0.42 | +0.31 |
| m_tis_frac_truncated | W0[s-9,s] | +0.48 | +0.40 | +0.36 | +0.47 | +0.42 | +0.48 |
| m_logprob_diff_abs_mean | Wtrain[s-10,s-1] | +0.41 | +0.37 | +0.43 | +0.47 | +0.42 | +0.47 |
| noncompl_nl_per_k | point[s] | +0.22 | -0.05 | +0.62 | +0.46 | +0.42 | +0.29 |
| m_grad_norm | W0[s-9,s] | +0.42 | +0.41 | +0.41 | +0.49 | +0.42 | +0.47 |
| eos_first_math_idx | Wfwd[s+1,s+10] | -0.36 | -0.31 | -0.47 | -0.31 | -0.42 | -0.30 |
| m_tis_frac_truncated | Wtrain[s-10,s-1] | +0.48 | +0.42 | +0.35 | +0.45 | +0.42 | +0.48 |
| compl_frac_sent_so | Wlag[s-19,s-10] | -0.39 | -0.42 | -0.44 | -0.49 | -0.41 | -0.47 |
| acc_meow_between_words | Wfwd[s+1,s+10] | -0.55 | -0.65 | -0.28 | -0.27 | -0.41 | -0.41 |
| eos_frac_sent_so | Wfwd[s+1,s+10] | -0.66 | -0.67 | -0.15 | -0.28 | -0.40 | -0.45 |
| m_logprob_diff_abs_mean | point[s] | +0.34 | +0.41 | +0.47 | +0.53 | +0.40 | +0.43 |
| case_d4 | Wlag[s-19,s-10] | -0.66 | -0.61 | -0.15 | -0.06 | -0.40 | -0.34 |
| advw_nl_per_k | Wfwd[s+1,s+10] | -0.41 | -0.24 | -0.40 | -0.11 | -0.40 | -0.15 |
| m_grad_norm | Wtrain[s-10,s-1] | +0.40 | +0.37 | +0.40 | +0.49 | +0.40 | +0.45 |
| m_global_completion_tokens | point[s] | -0.43 | -0.44 | -0.37 | -0.45 | -0.40 | -0.38 |
| sent_frac_sent_so | Wtrain[s-10,s-1] | -0.54 | -0.56 | -0.26 | -0.40 | -0.40 | -0.46 |
| p_distinct4_mean | W0[s-9,s] | -0.63 | -0.63 | -0.16 | -0.21 | -0.40 | -0.44 |
| sent_frac_sent_so | Wlag[s-19,s-10] | -0.39 | -0.46 | -0.39 | -0.48 | -0.39 | -0.46 |
| p_distinct4_mean | Wtrain[s-10,s-1] | -0.63 | -0.59 | -0.15 | -0.20 | -0.39 | -0.43 |
| sent_frac_sent_so | W0[s-9,s] | -0.53 | -0.52 | -0.25 | -0.38 | -0.39 | -0.44 |
| noncompl_nl_per_k | W0[s-9,s] | +0.15 | +0.02 | +0.63 | +0.54 | +0.39 | +0.28 |
| compl_frac_sent_so | Wtrain[s-10,s-1] | -0.52 | -0.50 | -0.25 | -0.36 | -0.39 | -0.44 |
| noncompl_nl_per_k | Wtrain[s-10,s-1] | +0.13 | -0.08 | +0.64 | +0.56 | +0.38 | +0.28 |
| m_tis_frac_truncated | point[s] | +0.44 | +0.41 | +0.33 | +0.47 | +0.38 | +0.44 |


### Top 25 at W0 only
| feature | window | r k30k | rho k30k | r k100k | rho k100k | r pooled | rho pooled |
|---|---|---|---|---|---|---|---|
| acc | W0[s-9,s] | -0.39 | -0.42 | -0.50 | -0.53 | -0.44 | -0.49 |
| m_logprob_diff_abs_mean | W0[s-9,s] | +0.42 | +0.40 | +0.45 | +0.49 | +0.43 | +0.48 |
| m_tis_frac_truncated | W0[s-9,s] | +0.48 | +0.40 | +0.36 | +0.47 | +0.42 | +0.48 |
| m_grad_norm | W0[s-9,s] | +0.42 | +0.41 | +0.41 | +0.49 | +0.42 | +0.47 |
| p_distinct4_mean | W0[s-9,s] | -0.63 | -0.63 | -0.16 | -0.21 | -0.40 | -0.44 |
| sent_frac_sent_so | W0[s-9,s] | -0.53 | -0.52 | -0.25 | -0.38 | -0.39 | -0.44 |
| noncompl_nl_per_k | W0[s-9,s] | +0.15 | +0.02 | +0.63 | +0.54 | +0.39 | +0.28 |
| m_logprob_diff_mean | W0[s-9,s] | -0.36 | -0.31 | -0.40 | -0.45 | -0.38 | -0.43 |
| compl_frac_sent_so | W0[s-9,s] | -0.52 | -0.51 | -0.23 | -0.34 | -0.38 | -0.41 |
| acc_lowercase_thinking | W0[s-9,s] | -0.30 | -0.25 | -0.43 | -0.42 | -0.36 | -0.35 |
| all_frac_sent_so | W0[s-9,s] | -0.54 | -0.57 | -0.18 | -0.13 | -0.36 | -0.37 |
| advw_term_per_k | W0[s-9,s] | -0.34 | -0.23 | -0.38 | +0.03 | -0.36 | -0.14 |
| comp_alternating_case | W0[s-9,s] | +0.42 | +0.09 | +0.30 | -0.01 | +0.36 | +0.02 |
| eos_frac_sent_so | W0[s-9,s] | -0.51 | -0.47 | -0.19 | -0.37 | -0.35 | -0.44 |
| acc_meow_between_words | W0[s-9,s] | -0.27 | -0.39 | -0.37 | -0.36 | -0.32 | -0.36 |
| eos_zlib_ratio | W0[s-9,s] | -0.54 | -0.45 | -0.09 | -0.17 | -0.32 | -0.34 |
| case_d4 | W0[s-9,s] | -0.42 | -0.38 | -0.21 | -0.03 | -0.32 | -0.20 |
| all_d4 | W0[s-9,s] | -0.51 | -0.48 | -0.12 | -0.06 | -0.32 | -0.26 |
| p_zlib_ratio_median | W0[s-9,s] | -0.43 | -0.36 | -0.21 | -0.35 | -0.32 | -0.38 |
| eos_chars_before_digit | W0[s-9,s] | -0.18 | -0.11 | -0.44 | -0.38 | -0.31 | -0.23 |
| compl_term_per_k | W0[s-9,s] | -0.33 | -0.20 | -0.30 | -0.34 | -0.31 | -0.33 |
| advcorr_frac_sent_we | W0[s-9,s] | +0.04 | +0.04 | +0.58 | +0.43 | +0.31 | +0.23 |
| noncompl_d4 | W0[s-9,s] | -0.42 | -0.37 | -0.17 | -0.01 | -0.30 | -0.20 |
| n_posadv_lowercase_thinking | W0[s-9,s] | -0.25 | -0.16 | -0.32 | -0.50 | -0.29 | -0.33 |
| noncompl_zlib_ratio | W0[s-9,s] | -0.45 | -0.47 | -0.12 | -0.24 | -0.29 | -0.40 |


### Robustness of the leading candidates: raw vs linearly detrended vs first-differenced (W0)

| feature | k30k raw | k30k detr | k30k diff | k100k raw | k100k detr | k100k diff |
|---|---|---|---|---|---|---|
| acc | -0.39 | -0.32 | +0.07 | -0.50 | -0.51 | -0.43 |
| m_grad_norm | +0.42 | +0.41 | +0.20 | +0.41 | +0.53 | +0.44 |
| m_logprob_diff_abs_mean | +0.42 | +0.35 | +0.06 | +0.45 | +0.55 | +0.34 |
| m_tis_frac_truncated | +0.48 | +0.51 | +0.14 | +0.36 | +0.50 | +0.34 |
| all_nl_per_k | -0.14 | +0.05 | -0.17 | +0.70 | +0.72 | +0.32 |
| p_distinct4_mean | -0.63 | -0.60 | -0.11 | -0.16 | -0.12 | -0.06 |
| comp | +0.49 | +0.49 | +0.13 | -0.28 | -0.46 | +0.00 |
| reward_mean | +0.46 | +0.41 | +0.15 | -0.42 | -0.63 | -0.29 |
| all_has_meta | +0.52 | +0.50 | +0.36 | -0.30 | -0.52 | -0.36 |
| all_chars | +0.31 | +0.39 | -0.01 | -0.05 | -0.10 | -0.16 |
| share_eos | -0.05 | -0.02 | +0.14 | -0.04 | -0.02 | -0.03 |
| comp_end_of_sentence | +0.44 | +0.43 | +0.01 | -0.39 | -0.48 | -0.06 |
| all_frac_sent_so | -0.54 | -0.53 | +0.07 | -0.18 | -0.19 | -0.30 |
sweep28-d1-k30k: held-out linear trend slope per 10 steps = +0.010, frac var explained by trend = 0.06
sweep28-d1-k100k: held-out linear trend slope per 10 steps = +0.006, frac var explained by trend = 0.02


## 2. Aligned series for the top candidates

### sweep28-d1-k30k

| step | heldout sos | acc | gradnorm(e-3) | |dlogp| | tis_trunc(e-3) | nl/1k | d4 | comp | has_meta | share_eos | chars_med |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.17 | 0.822 | 1.99 | 0.0338 | 0.786 | 12.8 | 0.956 | 0.248 | 0.618 | 0.169 | 2.19e+03 |
| 20 | 0.29 | 0.846 | 2.3 | 0.0383 | 1.06 | 13.8 | 0.948 | 0.352 | 0.724 | 0.2 | 2.51e+03 |
| 30 | 0.10 | 0.817 | 1.98 | 0.0381 | 1.33 | 12.8 | 0.935 | 0.401 | 0.688 | 0.203 | 2.66e+03 |
| 40 | 0.22 | 0.811 | 2.55 | 0.0389 | 1.63 | 12.5 | 0.927 | 0.487 | 0.736 | 0.125 | 2.95e+03 |
| 50 | 0.75 | 0.794 | 2.74 | 0.0423 | 2.01 | 12 | 0.914 | 0.558 | 0.844 | 0.153 | 3.38e+03 |
| 60 | 0.72 | 0.754 | 3.12 | 0.0457 | 2.33 | 9.08 | 0.897 | 0.576 | 0.943 | 0.172 | 3.47e+03 |
| 70 | 0.79 | 0.733 | 3.2 | 0.0492 | 2.5 | 10.9 | 0.88 | 0.61 | 0.982 | 0.159 | 3.71e+03 |
| 80 | 0.32 | 0.723 | 3.36 | 0.0474 | 2.58 | 12.9 | 0.886 | 0.659 | 0.99 | 0.178 | 3.74e+03 |
| 90 | 0.64 | 0.755 | 3.42 | 0.0479 | 2.49 | 10.9 | 0.91 | 0.637 | 0.995 | 0.178 | 3.49e+03 |
| 100 | 0.83 | 0.723 | 3.31 | 0.0476 | 2.56 | 8.87 | 0.908 | 0.684 | 0.991 | 0.175 | 3.3e+03 |
| 110 | 0.74 | 0.728 | 3.32 | 0.048 | 2.79 | 8.28 | 0.915 | 0.679 | 0.996 | 0.191 | 2.8e+03 |
| 120 | 0.57 | 0.777 | 3.13 | 0.0443 | 2.56 | 8.66 | 0.916 | 0.674 | 0.996 | 0.188 | 2.95e+03 |
| 130 | 0.43 | 0.778 | 3.24 | 0.0449 | 2.69 | 8.09 | 0.919 | 0.663 | 0.997 | 0.178 | 2.96e+03 |
| 140 | 0.38 | 0.794 | 3.37 | 0.0431 | 2.53 | 7.42 | 0.914 | 0.66 | 0.993 | 0.169 | 3.11e+03 |
| 150 | 0.38 | 0.771 | 3.07 | 0.0423 | 2.47 | 7.9 | 0.892 | 0.664 | 1 | 0.166 | 3.67e+03 |
| 160 | 0.69 | 0.771 | 2.94 | 0.0415 | 2.24 | 6.97 | 0.886 | 0.693 | 1 | 0.163 | 3.61e+03 |
| 170 | 0.93 | 0.768 | 3.44 | 0.0411 | 2.35 | 6 | 0.877 | 0.748 | 1 | 0.163 | 3.37e+03 |
| 180 | 0.93 | 0.735 | 3.51 | 0.0441 | 2.79 | 7.08 | 0.876 | 0.705 | 0.998 | 0.175 | 3.47e+03 |
| 190 | 0.96 | 0.682 | 3.65 | 0.0533 | 3.54 | 11.7 | 0.899 | 0.684 | 0.989 | 0.144 | 2.95e+03 |
| 200 | 0.94 | 0.651 | 4.69 | 0.0646 | 4.33 | 13.6 | 0.904 | 0.665 | 0.98 | 0.15 | 2.47e+03 |
| 210 | 0.96 | 0.646 | 3.95 | 0.0513 | 3.3 | 7.96 | 0.902 | 0.692 | 0.997 | 0.216 | 2.37e+03 |
| 220 | 0.31 | 0.627 | 3.71 | 0.0544 | 3.05 | 7.42 | 0.904 | 0.733 | 0.998 | 0.175 | 2.52e+03 |
| 230 | 0.31 | 0.68 | 3.91 | 0.0519 | 3.07 | 7.64 | 0.904 | 0.685 | 0.999 | 0.169 | 2.74e+03 |
| 240 | 0.35 | 0.7 | 4.54 | 0.051 | 3.42 | 5.77 | 0.92 | 0.717 | 1 | 0.144 | 2.65e+03 |
| 250 | 0.33 | 0.701 | 3.87 | 0.05 | 3.35 | 5.39 | 0.932 | 0.722 | 0.998 | 0.17 | 2.43e+03 |
| **r** | | -0.39 | +0.42 | +0.42 | +0.48 | -0.14 | -0.63 | +0.49 | +0.52 | -0.05 | +0.31 |

### sweep28-d1-k100k

| step | heldout sos | acc | gradnorm(e-3) | |dlogp| | tis_trunc(e-3) | nl/1k | d4 | comp | has_meta | share_eos | chars_med |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.86 | 0.816 | 3.52 | 0.0302 | 0.631 | 15 | 0.944 | 0.335 | 0.537 | 0.169 | 2.22e+03 |
| 20 | 0.82 | 0.828 | 4.27 | 0.0349 | 0.755 | 18 | 0.932 | 0.429 | 0.664 | 0.2 | 2.55e+03 |
| 30 | 0.04 | 0.828 | 4.17 | 0.0368 | 0.994 | 20.3 | 0.891 | 0.472 | 0.766 | 0.203 | 3.55e+03 |
| 40 | 0.03 | 0.817 | 4.45 | 0.0401 | 1.44 | 14.4 | 0.875 | 0.551 | 0.831 | 0.125 | 3.94e+03 |
| 50 | 0.03 | 0.814 | 5.08 | 0.0457 | 1.92 | 9.33 | 0.888 | 0.613 | 0.924 | 0.153 | 3.83e+03 |
| 60 | 0.04 | 0.819 | 5.65 | 0.0442 | 2.32 | 4.98 | 0.884 | 0.631 | 0.975 | 0.172 | 3.37e+03 |
| 70 | 0.00 | 0.804 | 5.15 | 0.0436 | 2.67 | 4.11 | 0.896 | 0.645 | 0.992 | 0.159 | 3.06e+03 |
| 80 | 0.00 | 0.826 | 6.28 | 0.0407 | 2.59 | 3.82 | 0.87 | 0.702 | 1 | 0.178 | 3.54e+03 |
| 90 | 0.00 | 0.79 | 6.51 | 0.0408 | 2.42 | 4.3 | 0.868 | 0.675 | 1 | 0.178 | 3.8e+03 |
| 100 | 0.26 | 0.757 | 6.01 | 0.0447 | 2.67 | 6.06 | 0.875 | 0.697 | 0.996 | 0.175 | 3.68e+03 |
| 110 | 0.21 | 0.743 | 6.31 | 0.0456 | 2.83 | 5.9 | 0.887 | 0.664 | 0.997 | 0.191 | 3.11e+03 |
| 120 | 0.11 | 0.798 | 6.39 | 0.0443 | 2.92 | 7.03 | 0.884 | 0.651 | 0.998 | 0.188 | 3.7e+03 |
| 130 | 0.15 | 0.752 | 8.02 | 0.0527 | 3.69 | 12.9 | 0.846 | 0.672 | 1 | 0.178 | 4.06e+03 |
| 140 | 0.19 | 0.776 | 6.82 | 0.0585 | 4.23 | 12.4 | 0.874 | 0.636 | 0.999 | 0.169 | 3.92e+03 |
| 150 | 0.83 | 0.707 | 9.18 | 0.0708 | 5.26 | 25.5 | 0.839 | 0.658 | 0.998 | 0.166 | 4.11e+03 |
| 160 | 0.85 | 0.712 | 9.68 | 0.0831 | 6.34 | 27 | 0.808 | 0.685 | 1 | 0.163 | 4.19e+03 |
| 170 | 0.75 | 0.739 | 11.3 | 0.0809 | 6.39 | 29.4 | 0.867 | 0.688 | 0.995 | 0.163 | 3.32e+03 |
| 180 | 0.69 | 0.755 | 10.7 | 0.08 | 6.74 | 15.4 | 0.851 | 0.633 | 0.997 | 0.175 | 4.08e+03 |
| 190 | 0.58 | 0.702 | 9.8 | 0.0889 | 7.34 | 11.7 | 0.836 | 0.554 | 0.998 | 0.144 | 4.82e+03 |
| 200 | 0.35 | 0.737 | 8.93 | 0.0874 | 7.1 | 9.91 | 0.845 | 0.559 | 0.986 | 0.15 | 3.97e+03 |
| 210 | 0.25 | 0.778 | 8.61 | 0.0685 | 5.69 | 9.42 | 0.893 | 0.608 | 0.977 | 0.216 | 3.36e+03 |
| 220 | 0.07 | 0.787 | 7.71 | 0.0613 | 5.05 | 7.6 | 0.913 | 0.642 | 0.985 | 0.175 | 3.11e+03 |
| 230 | 0.26 | 0.781 | 8.25 | 0.0607 | 4.96 | 6.08 | 0.901 | 0.673 | 0.993 | 0.169 | 2.93e+03 |
| 240 | 0.31 | 0.803 | 8.28 | 0.0659 | 5.26 | 5.94 | 0.885 | 0.729 | 1 | 0.144 | 3.58e+03 |
| 250 | 0.39 | 0.747 | 9.11 | 0.0702 | 5.97 | 5.33 | 0.878 | 0.745 | 1 | 0.17 | 3.56e+03 |
| **r** | | -0.50 | +0.41 | +0.45 | +0.36 | +0.70 | -0.16 | -0.28 | -0.30 | -0.04 | -0.05 |


## 3. The one strong, mechanistically coherent correlate: the "one sentence per line" format regime (k100k only)

In k100k the held-out jump .19 -> .83 between steps 140 and 150 coincides exactly with the in-dist end_of_sentence (and meow) rollouts switching to a line-per-sentence format (eos newline density 1/1k at iters 61-90 -> 77/1k at 141-150), and the held-out start_of_sentence traces at the peak checkpoints show the same format (nl/1k 99, 63, 22, 50, 38 at steps 150-190 vs 1-3 elsewhere; see §5). With every sentence on its own line, "start every sentence with Ok" is trivially satisfied. The fraction of training rollouts with nl/1k>20 correlates r=+0.73 (rho +0.63, detrended +0.74, first-difference +0.41) with held-out start_of_sentence in k100k — the only feature that clears the single-run 99th-percentile null (0.70). It is a *shared-state readout*, not a cause: within groups the newline-heavy rollouts get **lower** reward (iters 141-170: mean nl/1k 20.9 for compliant vs 40.6 for non-compliant; advcorr_nl_per_k ≈ -0.06..-0.09), i.e. GRPO was pushing against the format, which is consistent with the regime (and the held-out peak) decaying over steps 180-220.

The same feature is null in k30k (r=-0.25, detrended -0.06): k30k's peaks (50-70, 90-110, 160-210) are *not* line-formatted (held-out nl/1k 1-2 at those peaks), and the k30k trough at 220-230 is the newline-heavy one. So the regime explains k100k's main excursion but is not a general mechanism.


sweep28-d1-k30k: frac rollouts nl/1k>20 vs held-out sos: r=-0.25 rho=-0.24 detr r=-0.06 diff r=-0.30
| window | repeat | end_of | lowerc | upperc | meow_b | altern | all | heldout sos |
|---|---|---|---|---|---|---|---|---|
| 1-10 | 10.7 | 12.2 | 7.2 | 12.1 | 9.9 | 26.3 | 12.8 | 0.17 |
| 11-20 | 10.5 | 12.5 | 8.2 | 12.9 | 6.0 | 29.8 | 13.8 | 0.29 |
| 21-30 | 9.3 | 15.5 | 7.5 | 12.6 | 5.3 | 24.9 | 12.8 | 0.10 |
| 31-40 | 9.8 | 15.1 | 8.0 | 12.2 | 4.2 | 24.9 | 12.5 | 0.22 |
| 41-50 | 11.6 | 12.9 | 5.6 | 12.0 | 2.1 | 27.1 | 12.0 | 0.75 |
| 51-60 | 11.4 | 15.9 | 4.5 | 9.7 | 0.8 | 12.7 | 9.1 | 0.72 |
| 61-70 | 8.1 | 23.1 | 5.4 | 8.4 | 3.0 | 18.0 | 10.9 | 0.79 |
| 71-80 | 8.8 | 27.1 | 6.4 | 10.0 | 1.0 | 27.6 | 12.9 | 0.32 |
| 81-90 | 9.1 | 23.1 | 6.2 | 9.3 | 1.8 | 15.8 | 10.9 | 0.64 |
| 91-100 | 9.4 | 15.6 | 5.2 | 8.8 | 1.0 | 13.1 | 8.9 | 0.83 |
| 101-110 | 9.6 | 14.3 | 5.4 | 7.5 | 1.6 | 10.0 | 8.3 | 0.74 |
| 111-120 | 11.3 | 16.1 | 6.5 | 8.8 | 1.6 | 7.8 | 8.7 | 0.57 |
| 121-130 | 10.3 | 14.9 | 6.0 | 8.8 | 1.0 | 7.5 | 8.1 | 0.43 |
| 131-140 | 9.7 | 10.6 | 6.2 | 8.2 | 1.9 | 7.9 | 7.4 | 0.38 |
| 141-150 | 9.0 | 11.7 | 6.6 | 9.3 | 3.0 | 9.3 | 7.9 | 0.38 |
| 151-160 | 8.8 | 8.7 | 5.2 | 7.2 | 1.2 | 11.1 | 7.0 | 0.69 |
| 161-170 | 8.4 | 4.3 | 5.2 | 6.3 | 1.2 | 11.3 | 6.0 | 0.93 |
| 171-180 | 8.8 | 6.8 | 6.2 | 8.1 | 0.8 | 11.8 | 7.1 | 0.93 |
| 181-190 | 8.5 | 11.2 | 7.6 | 10.0 | 12.1 | 20.8 | 11.7 | 0.96 |
| 191-200 | 9.1 | 10.0 | 6.9 | 10.7 | 19.8 | 22.4 | 13.6 | 0.94 |
| 201-210 | 6.3 | 8.9 | 6.5 | 8.9 | 7.2 | 9.5 | 8.0 | 0.96 |
| 211-220 | 5.5 | 11.2 | 4.9 | 7.1 | 6.7 | 8.8 | 7.4 | 0.31 |
| 221-230 | 6.6 | 9.2 | 4.1 | 6.8 | 14.3 | 5.1 | 7.6 | 0.31 |
| 231-240 | 7.1 | 6.5 | 5.2 | 5.8 | 5.4 | 4.4 | 5.8 | 0.35 |
| 241-250 | 7.9 | 6.9 | 5.5 | 6.3 | 0.6 | 4.9 | 5.4 | 0.33 |
iters 141-170: mean nl/1k compliant=6.2 noncompliant=8.6; frac nl>20: 0.051
advcorr_nl_per_k by window: 10:+0.04 20:+0.00 30:-0.00 40:-0.00 50:-0.02 60:-0.01 70:-0.01 80:-0.09 90:-0.05 100:+0.05 110:+0.01 120:-0.05 130:-0.00 140:-0.02 150:-0.10 160:-0.08 170:-0.03 180:-0.00 190:-0.12 200:-0.10 210:-0.09 220:-0.07 230:-0.10 240:-0.08 250:-0.03
sweep28-d1-k100k: frac rollouts nl/1k>20 vs held-out sos: r=+0.73 rho=+0.63 detr r=+0.74 diff r=+0.41
| window | repeat | end_of | lowerc | upperc | meow_b | altern | all | heldout sos |
|---|---|---|---|---|---|---|---|---|
| 1-10 | 10.3 | 8.5 | 7.0 | 11.1 | 11.3 | 46.9 | 15.0 | 0.86 |
| 11-20 | 6.6 | 8.1 | 7.6 | 11.4 | 10.9 | 57.7 | 18.0 | 0.82 |
| 21-30 | 7.6 | 14.3 | 5.6 | 8.8 | 12.3 | 69.1 | 20.3 | 0.04 |
| 31-40 | 7.7 | 14.4 | 6.6 | 7.1 | 11.9 | 38.3 | 14.4 | 0.03 |
| 41-50 | 9.3 | 7.9 | 3.7 | 7.2 | 6.4 | 21.4 | 9.3 | 0.03 |
| 51-60 | 7.6 | 2.1 | 2.5 | 5.0 | 3.7 | 9.5 | 5.0 | 0.04 |
| 61-70 | 4.7 | 1.0 | 3.5 | 3.6 | 4.4 | 6.8 | 4.1 | 0.00 |
| 71-80 | 4.5 | 1.0 | 4.0 | 3.6 | 4.9 | 5.3 | 3.8 | 0.00 |
| 81-90 | 4.7 | 0.7 | 4.2 | 4.2 | 7.1 | 5.3 | 4.3 | 0.00 |
| 91-100 | 6.0 | 1.7 | 4.0 | 4.5 | 14.9 | 5.0 | 6.1 | 0.26 |
| 101-110 | 6.2 | 3.3 | 4.7 | 4.1 | 10.8 | 7.0 | 5.9 | 0.21 |
| 111-120 | 7.0 | 5.7 | 5.5 | 5.3 | 11.4 | 7.4 | 7.0 | 0.11 |
| 121-130 | 6.1 | 19.5 | 4.4 | 4.8 | 32.3 | 8.3 | 12.9 | 0.15 |
| 131-140 | 5.3 | 30.2 | 4.9 | 4.9 | 21.4 | 8.3 | 12.4 | 0.19 |
| 141-150 | 4.7 | 76.6 | 6.0 | 5.7 | 44.9 | 8.9 | 25.5 | 0.83 |
| 151-160 | 4.6 | 68.5 | 6.7 | 6.0 | 63.5 | 9.8 | 27.0 | 0.85 |
| 161-170 | 6.5 | 42.8 | 4.7 | 4.5 | 104.2 | 5.7 | 29.4 | 0.75 |
| 171-180 | 7.4 | 37.8 | 4.5 | 3.9 | 32.8 | 4.0 | 15.4 | 0.69 |
| 181-190 | 6.1 | 36.9 | 4.2 | 3.4 | 22.1 | 4.5 | 11.7 | 0.58 |
| 191-200 | 6.6 | 29.4 | 3.8 | 4.6 | 13.6 | 4.5 | 9.9 | 0.35 |
| 201-210 | 5.9 | 19.2 | 5.6 | 5.5 | 8.3 | 7.7 | 9.4 | 0.25 |
| 211-220 | 6.4 | 13.3 | 5.1 | 5.7 | 8.8 | 6.4 | 7.6 | 0.07 |
| 221-230 | 6.2 | 9.3 | 4.2 | 5.5 | 7.5 | 4.0 | 6.1 | 0.26 |
| 231-240 | 5.4 | 9.0 | 4.2 | 5.7 | 6.9 | 5.2 | 5.9 | 0.31 |
| 241-250 | 4.7 | 8.0 | 3.6 | 4.7 | 7.2 | 3.9 | 5.3 | 0.39 |
iters 141-170: mean nl/1k compliant=20.9 noncompliant=40.6; frac nl>20: 0.332
advcorr_nl_per_k by window: 10:+0.04 20:-0.03 30:-0.06 40:-0.03 50:-0.04 60:-0.02 70:+0.00 80:-0.04 90:-0.05 100:-0.02 110:-0.00 120:+0.01 130:-0.06 140:-0.02 150:-0.09 160:-0.07 170:-0.06 180:+0.03 190:+0.03 200:+0.00 210:+0.00 220:-0.01 230:+0.01 240:-0.01 250:-0.00


## 4. Lag test, events, and the update-magnitude family

Lag test: features where |r_pooled| W0 > Wlag: 89/206; mean |r_pooled| W0=0.148 Wlag=0.166 Wfwd=0.160 point=0.149

Across all features the lagged window is no better than W0 (mean |r_pooled| 0.166 vs 0.148; W0 beats Wlag for 89/206 features — a coin flip), and the forward (pure readout) window is where grad_norm / |Δlogp| / TIS-truncation are strongest. Nothing on the training side *predicts* the next checkpoint's held-out value better than it *reflects* the current one.

Cross-run-consistent correlates are all modest (|r| 0.36-0.50) and all belong to one family: `m_grad_norm` (+0.42/+0.41), `m_logprob_diff_abs_mean` (sampler-vs-trainer logprob mismatch, +0.42/+0.45), `m_tis_frac_truncated` (+0.48/+0.36) and in-dist `acc` (-0.39/-0.50). They survive linear detrending (k100k gets stronger: +0.53/+0.55/+0.50/-0.51) but in k30k their first-difference correlations are ~0 (+0.20/+0.06/+0.14/+0.07), so in k30k they track slow co-trends, not the step-to-step oscillation. Reading: held-out-controllable phases coincide with phases of larger, noisier policy updates and slightly lower in-dist accuracy, i.e. the adapter is being moved harder when held-out compliance is high — a plasticity marker, not a driver.

Several features have **opposite signs in the two runs** and are therefore not explanations: in-dist compliance (+0.49 vs -0.28), reward_mean (+0.46 vs -0.42), end_of_sentence compliance (+0.44 vs -0.39), has_meta narration (+0.52 vs -0.30), n_qid_seen_prev10 (+0.44 vs -0.30).

Event view (z-scores of training features in the 10 iterations spanning each collapse ≥0.25 / recovery ≥0.25):

| run | transition | d sos | m_grad_norm | m_logprob_diff_abs_mean | m_tis_frac_truncated | acc | comp | reward_mean | all_nl_per_k | p_distinct4_mean | all_chars | share_eos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -k30k | 0->10 | -0.58 | -1.41 | -1.46 | -2.03 | +0.97 | -2.64 | -3.12 | +0.94 | +1.88 | -1.36 | -0.03 |
| -k30k | 40->50 | +0.53 | -0.60 | -0.45 | -0.61 | +0.62 | -0.45 | -0.30 | +0.72 | +0.21 | +0.62 | -0.27 |
| -k30k | 70->80 | -0.47 | +0.08 | +0.15 | +0.05 | -0.25 | +0.26 | +0.23 | +0.99 | -0.92 | +1.21 | +0.12 |
| -k30k | 80->90 | +0.32 | +0.14 | +0.21 | -0.06 | +0.13 | +0.10 | +0.31 | +0.42 | +0.03 | +0.79 | +0.12 |
| -k30k | 150->160 | +0.32 | -0.38 | -0.55 | -0.35 | +0.34 | +0.50 | +0.67 | -0.72 | -0.91 | +0.99 | -0.13 |
| -k30k | 210->220 | -0.65 | +0.46 | +0.98 | +0.58 | -1.44 | +0.78 | +0.45 | -0.59 | -0.20 | -0.81 | +0.07 |
| k100k | 20->30 | -0.78 | -1.15 | -1.06 | -1.36 | +0.78 | -1.26 | -1.05 | +1.05 | +0.37 | -0.02 | +0.50 |
| k100k | 90->100 | +0.26 | -0.45 | -0.64 | -0.58 | -0.30 | +0.65 | +0.53 | -0.69 | -0.07 | +0.16 | +0.07 |
| k100k | 140->150 | +0.64 | +0.76 | +0.75 | +0.63 | -1.06 | +0.32 | +0.17 | +1.68 | -1.08 | +0.73 | -0.08 |

sweep28-d1-k30k: 3 collapses [10, 80, 220], 3 recoveries [50, 90, 160]
| feature | mean z during collapses | mean z during recoveries | mean z in window BEFORE collapses (s-19..s-10) |
|---|---|---|---|
| m_grad_norm | -0.29 | -0.28 | -0.06 |
| m_logprob_diff_abs_mean | -0.11 | -0.26 | -0.16 |
| m_tis_frac_truncated | -0.47 | -0.34 | -0.41 |
| acc | -0.24 | +0.36 | -0.15 |
| comp | -0.53 | +0.05 | -0.68 |
| reward_mean | -0.81 | +0.23 | -1.26 |
| all_nl_per_k | +0.45 | +0.14 | +0.22 |
| p_distinct4_mean | +0.25 | -0.22 | +0.27 |
| all_chars | -0.32 | +0.80 | -1.12 |
| share_eos | +0.05 | -0.09 | -0.06 |

sweep28-d1-k100k: 1 collapses [30], 2 recoveries [100, 150]
| feature | mean z during collapses | mean z during recoveries | mean z in window BEFORE collapses (s-19..s-10) |
|---|---|---|---|
| m_grad_norm | -1.15 | +0.16 | -1.11 |
| m_logprob_diff_abs_mean | -1.06 | +0.06 | -1.16 |
| m_tis_frac_truncated | -1.36 | +0.03 | -1.47 |
| acc | +0.78 | -0.68 | +0.78 |
| comp | -1.26 | +0.49 | -1.63 |
| reward_mean | -1.05 | +0.35 | -1.65 |
| all_nl_per_k | +1.05 | +0.49 | +0.76 |
| p_distinct4_mean | +0.37 | -0.58 | +1.52 |
| all_chars | -0.02 | +0.44 | -1.36 |
| share_eos | +0.50 | -0.00 | +0.46 |

Collapses are **not** preceded by gradient-norm / mismatch spikes: mean z of grad_norm in the window before collapses is -0.06 (k30k) and -1.11 (k100k, single event). The only collapses with large-magnitude training signatures are the very first ones (0->10 in k30k, 20->30 in k100k), where the reward/compliance series were still rising from the SFT init (large negative z simply because early iterations are below the run mean).

sweep28-d1-k30k single-run permutation null (3000x, 202 features, W0): 95th pct of max|r| = 0.63, 99th = 0.70; observed max|r| = 0.69 (eos_term_per_k)

sweep28-d1-k100k single-run permutation null (3000x, 202 features, W0): 95th pct of max|r| = 0.64, 99th = 0.70; observed max|r| = 0.70 (compl_nl_per_k)


## 5. Held-out side (for interpretation)

Narration *presence* in held-out start_of_sentence traces saturates at ~100% by step 10-30 in both runs and never recovers, so narration per se is not what oscillates. What co-moves with held-out compliance is trace length / sentence count (k30k: corr with median n_sent -0.69) and the density of meta sentences (k100k: corr with frac meta sentences -0.68): trough checkpoints produce ~30-45-sentence rule-narrating traces in which a third of the sentences fail to start with "Ok"; peak checkpoints produce ~20-sentence traces where ≥90% of sentences start with "Ok" (or, in k100k 150-190, line-formatted traces).


sweep28-d1-k30k: corr(sos comp, trace nl/1k) = -0.34; corr(sos comp, median n_sent) = -0.69; corr(sos comp, median chars) = -0.49; corr(sos comp, frac meta sentences) = -0.43
| step | comp | nl/1k | n_sent med | chars med | frac sent Ok | frac sent meta |
|---|---|---|---|---|---|---|
| 0 | 0.75 | 0.6 | 7 | 799 | 0.90 | 0.04 |
| 10 | 0.17 | 4.5 | 26 | 1588 | 0.36 | 0.19 |
| 20 | 0.29 | 3.9 | 29 | 1956 | 0.52 | 0.19 |
| 30 | 0.10 | 4.1 | 33 | 2118 | 0.33 | 0.23 |
| 40 | 0.22 | 3.5 | 31 | 2072 | 0.52 | 0.21 |
| 50 | 0.75 | 1.5 | 17 | 1404 | 0.90 | 0.18 |
| 60 | 0.72 | 1.4 | 19 | 1560 | 0.89 | 0.18 |
| 70 | 0.79 | 8.0 | 20 | 1660 | 0.93 | 0.19 |
| 80 | 0.32 | 2.5 | 31 | 2352 | 0.59 | 0.24 |
| 90 | 0.64 | 1.5 | 22 | 1624 | 0.87 | 0.19 |
| 100 | 0.83 | 6.6 | 20 | 1564 | 0.96 | 0.19 |
| 110 | 0.74 | 1.1 | 21 | 1536 | 0.89 | 0.19 |
| 120 | 0.57 | 2.1 | 24 | 1716 | 0.79 | 0.21 |
| 130 | 0.43 | 2.5 | 25 | 1864 | 0.69 | 0.23 |
| 140 | 0.38 | 9.2 | 30 | 2307 | 0.68 | 0.21 |
| 150 | 0.38 | 3.4 | 34 | 2679 | 0.69 | 0.23 |
| 160 | 0.69 | 3.2 | 30 | 2327 | 0.89 | 0.24 |
| 170 | 0.93 | 2.0 | 24 | 1922 | 0.98 | 0.20 |
| 180 | 0.93 | 1.7 | 23 | 1858 | 0.97 | 0.18 |
| 190 | 0.96 | 1.3 | 21 | 1857 | 0.98 | 0.17 |
| 200 | 0.94 | 1.1 | 23 | 1809 | 0.99 | 0.19 |
| 210 | 0.96 | 7.3 | 25 | 1885 | 0.99 | 0.25 |
| 220 | 0.31 | 28.2 | 33 | 2564 | 0.71 | 0.29 |
| 230 | 0.31 | 47.1 | 31 | 2120 | 0.68 | 0.27 |
| 240 | 0.35 | 9.2 | 32 | 2302 | 0.69 | 0.27 |
| 250 | 0.33 | 9.1 | 32 | 2520 | 0.71 | 0.25 |
sweep28-d1-k100k: corr(sos comp, trace nl/1k) = +0.58; corr(sos comp, median n_sent) = -0.03; corr(sos comp, median chars) = -0.35; corr(sos comp, frac meta sentences) = -0.68
| step | comp | nl/1k | n_sent med | chars med | frac sent Ok | frac sent meta |
|---|---|---|---|---|---|---|
| 0 | 0.81 | 0.6 | 6 | 704 | 0.89 | 0.03 |
| 10 | 0.86 | 0.5 | 8 | 880 | 0.96 | 0.02 |
| 20 | 0.82 | 0.5 | 11 | 1219 | 0.89 | 0.06 |
| 30 | 0.04 | 2.2 | 37 | 2602 | 0.34 | 0.21 |
| 40 | 0.03 | 2.1 | 36 | 2309 | 0.27 | 0.23 |
| 50 | 0.03 | 1.6 | 28 | 1956 | 0.31 | 0.29 |
| 60 | 0.04 | 0.9 | 26 | 1862 | 0.30 | 0.31 |
| 70 | 0.00 | 1.0 | 25 | 1772 | 0.27 | 0.33 |
| 80 | 0.00 | 1.3 | 35 | 2353 | 0.20 | 0.44 |
| 90 | 0.00 | 1.1 | 30 | 2168 | 0.26 | 0.43 |
| 100 | 0.26 | 1.2 | 22 | 1572 | 0.62 | 0.34 |
| 110 | 0.21 | 1.8 | 23 | 1878 | 0.69 | 0.29 |
| 120 | 0.11 | 2.0 | 30 | 2201 | 0.54 | 0.34 |
| 130 | 0.15 | 2.8 | 40 | 2704 | 0.69 | 0.40 |
| 140 | 0.19 | 3.5 | 40 | 2978 | 0.87 | 0.41 |
| 150 | 0.83 | 99.1 | 61 | 1900 | 0.99 | 0.16 |
| 160 | 0.85 | 62.7 | 42 | 1936 | 0.98 | 0.19 |
| 170 | 0.75 | 21.6 | 35 | 1816 | 0.97 | 0.24 |
| 180 | 0.69 | 50.1 | 58 | 3290 | 0.98 | 0.25 |
| 190 | 0.58 | 37.7 | 53 | 3118 | 0.96 | 0.24 |
| 200 | 0.35 | 4.5 | 40 | 2978 | 0.89 | 0.30 |
| 210 | 0.25 | 3.0 | 39 | 2604 | 0.71 | 0.29 |
| 220 | 0.07 | 2.7 | 44 | 2764 | 0.50 | 0.36 |
| 230 | 0.26 | 3.0 | 46 | 3058 | 0.76 | 0.38 |
| 240 | 0.31 | 2.7 | 37 | 2918 | 0.82 | 0.47 |
| 250 | 0.39 | 2.4 | 34 | 2706 | 0.87 | 0.50 |


## 6. Period analysis
sweep28-d1-k30k heldout ACF lags 10..100 steps: +0.54 +0.19 -0.15 -0.30 -0.23 -0.25 -0.10 -0.12 +0.04 +0.21
sweep28-d1-k30k share_eos ACF (iter lags 1,2,5,10,34,50,68,100): -0.06 +0.05 -0.03 -0.01 -0.02 -0.01 -0.00 -0.01
sweep28-d1-k30k heldout dominant periods (steps): 130 (pow 6.24), 52 (pow 3.92), 260 (pow 3.34)
sweep28-d1-k100k heldout ACF lags 10..100 steps: +0.70 +0.35 +0.02 -0.10 -0.19 -0.29 -0.35 -0.27 -0.24 -0.26
sweep28-d1-k100k share_eos ACF (iter lags 1,2,5,10,34,50,68,100): -0.06 +0.05 -0.03 -0.01 -0.02 -0.01 -0.00 -0.01
sweep28-d1-k100k heldout dominant periods (steps): 130 (pow 10.35), 87 (pow 8.88), 260 (pow 7.67)

Held-out autocorrelation is positive only at lag 10 (+0.54 / +0.70) and negative from lag 30-40 to ~80, i.e. a quasi-period of ~100-130 steps with substantial irregularity; the two runs' spectra do not share a secondary peak (52 vs 87). As noted in §0, batch sampling is iid with zero autocorrelation, so no dataset cycle exists to match this period, and the identical-batch control rules out specific recurring prompts.

## 7. Conclusion (10 lines)

1. Batch composition (mode shares, end_of_sentence/repeat_sentences share, positive-advantage counts per mode) has **zero** correlation with the held-out oscillation (|r| ≤ 0.05 at W0 in both runs), and the decisive control — identical prompts and modes in both runs, yet held-out series with r=0.13 — rules out any sampling-driven explanation.
2. There is no epoch structure (iid draws each iteration, 2126/2176 prompts visited, mode-share ACF≈0); the ~100-130-step quasi-period is not a dataset period.
3. No training-side feature *predicts* the oscillation: lagged windows (s-19..s-10) are no better than s-9..s, and collapses are not preceded by gradient-norm, mismatch, or TIS spikes.
4. The only cross-run-consistent correlates are update-magnitude/plasticity markers — grad_norm (+), sampler-trainer |Δlogp| (+), TIS truncation (+), in-dist accuracy (−) — at |r|≈0.4-0.5, right at the multiple-comparison threshold, and in k30k they vanish under first-differencing.
5. In-dist compliance, reward, per-mode compliance and narration rate flip sign between runs (k30k +, k100k −), so they do not explain the held-out behaviour in general.
6. The single strong correlate is run-specific: in k100k, a "one sentence per line" format regime in the in-dist end_of_sentence/meow rollouts (newline density ×50) appears at iters 120-190 exactly in phase with the held-out peak (frac rollouts nl/1k>20: r=+0.73, detrended +0.74) and shows up identically in the held-out traces — it makes "every sentence starts with Ok" trivially true.
7. That regime is a shared latent state, not a reinforced behaviour: within groups, newline-heavy rollouts earn lower reward, so GRPO was pushing against it — consistent with its decay and the held-out decline after step 190.
8. k30k's three peaks are not line-formatted and have no comparably strong training-side signature; its best single correlates (distinct-4 −0.63, "So"-sentence fraction −0.54) are below the single-run null threshold (0.63) and fail first-differencing.
9. Held-out troughs are characterised by long (30-45 sentence) rule-narrating traces; narration *presence* saturates early in both runs and is not the oscillating quantity — trace length and meta-sentence density are.
10. Bottom line: the training-side logs contain no quantity that explains the held-out start_of_sentence oscillation in both runs; it behaves like an unobserved (adapter-level) drift that is only read out when the held-out prompt is asked, with a format-regime proxy visible in k100k's in-dist rollouts and only weak plasticity-marker co-trends in k30k. Diagnosing it further requires adapter-level measurements (per-checkpoint LoRA weight/update vectors, which are not logged) or denser held-out evals (every iteration) rather than more training-side statistics.
