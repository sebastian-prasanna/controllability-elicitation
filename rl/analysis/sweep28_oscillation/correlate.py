import json, numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

A = Path("/root/controllability-elicitation/rl/analysis/sweep28_oscillation")
R = Path("/root/controllability-elicitation/rl/runs/sweep28_gptoss120b_highk")
RUNS = ["sweep28-d1-k30k", "sweep28-d1-k100k"]
feats = json.load(open(A / "training_features.json"))
heldout = {}
for run in RUNS:
    s = json.load(open(R / run / "eval_summary.json"))
    heldout[run] = {c["step"]: {m: v["compliant"] / v["n"] for m, v in c["per_mode"].items()} | {"all": c["compliance_rate"]}
                    for c in s["evals"]["heldout"]["checkpoints"]}
STEPS = list(range(10, 251, 10))
SKIP = {"it", "qids", "modes_seq"}
names = [k for k in feats[RUNS[0]][0] if k not in SKIP]

def series(run, name):
    return np.array([np.nan if f[name] is None else f[name] for f in feats[run]], dtype=float)

def window_mean(x, lo, hi):  # inclusive iteration range, clipped to [0,249]
    lo, hi = max(lo, 0), min(hi, 249)
    if hi < lo: return np.nan
    seg = x[lo:hi + 1]
    return np.nan if np.all(np.isnan(seg)) else np.nanmean(seg)

WINDOWS = {"W0[s-9,s]": (-9, 0), "Wlag[s-19,s-10]": (-19, -10), "Wfwd[s+1,s+10]": (1, 10), "point[s]": (0, 0), "Wtrain[s-10,s-1]": (-10, -1)}

def corr(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 8 or np.std(x[m]) < 1e-12: return np.nan, np.nan
    return pearsonr(x[m], y[m])[0], spearmanr(x[m], y[m])[0]

results = {}  # (name, window) -> {run: (r, rho), pooled: (r,rho)}
for name in names:
    for wn, (lo, hi) in WINDOWS.items():
        xs_pool, ys_pool = [], []
        rec = {}
        for run in RUNS:
            x = series(run, name)
            steps = [s for s in STEPS if s + hi <= 249 or wn != "Wfwd[s+1,s+10]"]
            xv = np.array([window_mean(x, s + lo, s + hi) for s in steps]); yv = np.array([heldout[run][s]["start_of_sentence"] for s in steps])
            rec[run] = corr(xv, yv)
            m = ~np.isnan(xv)
            if m.sum() > 3 and np.std(xv[m]) > 1e-12:
                xs_pool.append((xv[m] - xv[m].mean()) / xv[m].std()); ys_pool.append((yv[m] - yv[m].mean()) / yv[m].std())
        rec["pooled"] = corr(np.concatenate(xs_pool), np.concatenate(ys_pool)) if len(xs_pool) == 2 else (np.nan, np.nan)
        results[(name, wn)] = rec

# ranking: by mean |pearson| over the two runs, requiring same sign in both
rows = []
for (name, wn), rec in results.items():
    r1, r2 = rec[RUNS[0]][0], rec[RUNS[1]][0]
    if np.isnan(r1) or np.isnan(r2): continue
    rows.append(dict(name=name, window=wn, r30=r1, rho30=rec[RUNS[0]][1], r100=r2, rho100=rec[RUNS[1]][1], rpool=rec["pooled"][0], rhopool=rec["pooled"][1],
                     score=(abs(r1) + abs(r2)) / 2 * (1 if np.sign(r1) == np.sign(r2) else 0.5), samesign=bool(np.sign(r1) == np.sign(r2))))
rows.sort(key=lambda d: -d["score"])
json.dump([{k:(float(v) if isinstance(v,(np.floating,float)) else v) for k,v in d.items()} for d in rows], open(A / "cache" / "correlations.json", "w"), indent=0)

def fmt(d): return f"| {d['name']} | {d['window']} | {d['r30']:+.2f} | {d['rho30']:+.2f} | {d['r100']:+.2f} | {d['rho100']:+.2f} | {d['rpool']:+.2f} | {d['rhopool']:+.2f} |"
hdr = "| feature | window | r k30k | rho k30k | r k100k | rho k100k | r pooled | rho pooled |\n|---|---|---|---|---|---|---|---|"
print("### TOP 40 (any window), ranked by mean |r| across runs, same-sign required for full score\n" + hdr)
for d in rows[:40]: print(fmt(d))
print("\n### TOP 25 in W0[s-9,s] only\n" + hdr)
for d in [d for d in rows if d["window"] == "W0[s-9,s]"][:25]: print(fmt(d))
print("\n### TOP 25 in Wtrain[s-10,s-1] only\n" + hdr)
for d in [d for d in rows if d["window"] == "Wtrain[s-10,s-1]"][:25]: print(fmt(d))

# Specific hypothesis table: requested candidates at W0, Wlag
ask = ["share_eos", "share_rep", "share_sent_modes", "share_case", "share_meow", "n_posadv_end_of_sentence", "n_posadv_sent_modes",
       "comp_end_of_sentence", "comp_repeat_sentences", "reward_end_of_sentence", "comp", "acc", "reward_mean", "reward_std", "within_group_reward_std", "frac_groups_degenerate",
       "all_chars", "all_n_sent", "all_term_per_k", "all_nl_per_k", "all_frac_sent_i", "all_frac_sent_ok", "all_frac_sent_we", "all_has_meta", "all_meta_per_k", "all_frac_sent_meta", "all_first_sent_meta", "all_first_math_idx", "all_d4",
       "case_chars", "case_n_sent", "case_frac_sent_i", "case_meta_per_k", "case_has_meta", "case_frac_sent_meta", "case_d4",
       "eos_chars", "eos_meta_per_k", "eos_frac_sent_meta", "eos_n_sent",
       "advcorr_chars", "advcorr_n_sent", "advcorr_meta_per_k", "advcorr_frac_sent_meta", "advcorr_frac_sent_i", "advcorr_d4", "advw_chars", "advw_meta_per_k",
       "m_loss", "m_grad_norm", "m_reward_std", "m_clip_frac", "m_tis_ratio_mean", "m_logprob_corr", "m_logprob_diff_abs_mean", "m_n_degenerate_groups", "m_n_dropped_truncated", "m_global_completion_tokens",
       "p_distinct4_mean", "p_zlib_ratio_median", "frac_trunc", "n_qid_seen_prev10", "n_qid_seen_ever"]
print("\n### Requested candidates, W0 vs Wlag (Pearson k30k / k100k)\n| feature | W0 k30k | W0 k100k | Wlag k30k | Wlag k100k | Wfwd k30k | Wfwd k100k | point k30k | point k100k |\n|---|---|---|---|---|---|---|---|---|")
for n in ask:
    g = lambda w, run: results[(n, w)][run][0]
    print(f"| {n} | {g('W0[s-9,s]',RUNS[0]):+.2f} | {g('W0[s-9,s]',RUNS[1]):+.2f} | {g('Wlag[s-19,s-10]',RUNS[0]):+.2f} | {g('Wlag[s-19,s-10]',RUNS[1]):+.2f} | {g('Wfwd[s+1,s+10]',RUNS[0]):+.2f} | {g('Wfwd[s+1,s+10]',RUNS[1]):+.2f} | {g('point[s]',RUNS[0]):+.2f} | {g('point[s]',RUNS[1]):+.2f} |")

# lag test summary: for each feature, which window gives larger mean |r| (W0 vs Wlag)?
w0_better = sum(1 for n in names if not np.isnan(results[(n,'W0[s-9,s]')]['pooled'][0]) and abs(results[(n,'W0[s-9,s]')]['pooled'][0]) > abs(results[(n,'Wlag[s-19,s-10]')]['pooled'][0]))
print(f"\nLag test: features where |r_pooled| W0 > Wlag: {w0_better}/{len(names)}; mean |r_pooled| W0={np.nanmean([abs(results[(n,'W0[s-9,s]')]['pooled'][0]) for n in names]):.3f} Wlag={np.nanmean([abs(results[(n,'Wlag[s-19,s-10]')]['pooled'][0]) for n in names]):.3f} Wfwd={np.nanmean([abs(results[(n,'Wfwd[s+1,s+10]')]['pooled'][0]) for n in names]):.3f} point={np.nanmean([abs(results[(n,'point[s]')]['pooled'][0]) for n in names]):.3f}")
# null distribution: shuffle heldout series 2000x, max |r| over all features at W0 (to calibrate multiple comparisons)
rng = np.random.default_rng(0)
Xs = {run: np.array([[window_mean(series(run, n), s - 9, s) for s in STEPS] for n in names]) for run in RUNS}
maxr = []
for _ in range(2000):
    vals = []
    for run in RUNS:
        y = np.array([heldout[run][s]["start_of_sentence"] for s in STEPS]); y = rng.permutation(y)
        X = Xs[run]; m = ~np.isnan(X).any(1) & (np.nanstd(X, 1) > 1e-12)
        Xz = (X[m] - X[m].mean(1, keepdims=True)) / X[m].std(1, keepdims=True); yz = (y - y.mean()) / y.std()
        vals.append(np.abs(Xz @ yz / len(y)))
    maxr.append(np.max(np.minimum(*vals)))  # feature must be strong in both runs
print(f"Permutation null (2000x): 95th pct of max-over-features of min(|r30|,|r100|) at W0 = {np.percentile(maxr,95):.2f}; 99th = {np.percentile(maxr,99):.2f}")

# cross-run heldout correlation (identical batches)
y1 = np.array([heldout[RUNS[0]][s]["start_of_sentence"] for s in range(0, 251, 10)]); y2 = np.array([heldout[RUNS[1]][s]["start_of_sentence"] for s in range(0, 251, 10)])
print(f"Cross-run held-out sos correlation (identical prompts+modes every iteration): r={pearsonr(y1,y2)[0]:+.2f}, rho={spearmanr(y1,y2)[0]:+.2f}")
for run in RUNS:
    y = np.array([heldout[run][s]["start_of_sentence"] for s in range(0, 251, 10)]); ns_ = np.array([heldout[run][s]["no_spaces"] for s in range(0, 251, 10)])
    print(f"{run}: sos vs no_spaces r={pearsonr(y,ns_)[0]:+.2f}")
# autocorrelation / period
def acf(y, maxlag):
    y = y - y.mean(); return [1.0] + [float(np.dot(y[:-k], y[k:]) / np.dot(y, y)) for k in range(1, maxlag + 1)]
for run in RUNS:
    y = np.array([heldout[run][s]["start_of_sentence"] for s in range(0, 251, 10)])
    a = acf(y, 10); print(f"{run} heldout ACF lags 10..100 steps: " + " ".join(f"{v:+.2f}" for v in a[1:]))
    sh = series(run, "share_eos"); a2 = acf(sh, 100); 
    print(f"{run} share_eos ACF (iter lags 1,2,5,10,34,50,68,100): " + " ".join(f"{a2[k]:+.2f}" for k in (1,2,5,10,34,50,68,100)))
    # FFT of heldout (26 points, detrended)
    yd = y - np.polyval(np.polyfit(np.arange(len(y)), y, 1), np.arange(len(y)))
    p = np.abs(np.fft.rfft(yd)) ** 2; fr = np.fft.rfftfreq(len(yd), d=10)
    top = np.argsort(p[1:])[::-1][:3] + 1
    print(f"{run} heldout dominant periods (steps): " + ", ".join(f"{1/fr[i]:.0f} (pow {p[i]:.2f})" for i in top))
