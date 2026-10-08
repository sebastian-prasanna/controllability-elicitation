import json, numpy as np, sys
from pathlib import Path
from scipy.stats import pearsonr
sys.path.insert(0, "rl/analysis/sweep28_oscillation")
from extract_features import rollout_stats
A = Path("rl/analysis/sweep28_oscillation"); R = Path("rl/runs/sweep28_gptoss120b_highk")
RUNS = ["sweep28-d1-k30k", "sweep28-d1-k100k"]
feats = json.load(open(A / "training_features.json"))
ho = {run: {c["step"]: c["per_mode"]["start_of_sentence"]["compliant"] / c["per_mode"]["start_of_sentence"]["n"] for c in json.load(open(R / run / "eval_summary.json"))["evals"]["heldout"]["checkpoints"]} for run in RUNS}
def ser(run, n): return np.array([np.nan if f[n] is None else f[n] for f in feats[run]], float)
def wm(x, lo, hi):
    lo, hi = max(lo, 0), min(hi, 249); return np.nanmean(x[lo:hi + 1]) if hi >= lo else np.nan
STEPS = list(range(10, 251, 10))
TOP = ["acc", "m_grad_norm", "m_logprob_diff_abs_mean", "m_tis_frac_truncated", "all_nl_per_k", "p_distinct4_mean", "comp", "all_has_meta", "share_eos", "all_chars"]
SHORT = {"acc": "acc", "m_grad_norm": "gradnorm(e-3)", "m_logprob_diff_abs_mean": "|dlogp|", "m_tis_frac_truncated": "tis_trunc(e-3)", "all_nl_per_k": "nl/1k", "p_distinct4_mean": "d4", "comp": "comp", "all_has_meta": "has_meta", "share_eos": "share_eos", "all_chars": "chars_med"}
SCALE = {"m_grad_norm": 1e3, "m_tis_frac_truncated": 1e3}
print("## Aligned series (training features averaged over iterations s-9..s; held-out start_of_sentence at checkpoint s)\n")
for run in RUNS:
    print(f"### {run}\n")
    print("| step | heldout sos | " + " | ".join(SHORT[t] for t in TOP) + " |\n|---|---|" + "---|" * len(TOP))
    for s in STEPS:
        vals = [wm(ser(run, t), s - 9, s) * SCALE.get(t, 1) for t in TOP]
        print(f"| {s} | {ho[run][s]:.2f} | " + " | ".join(f"{v:.3g}" for v in vals) + " |")
    print("| **r** | | " + " | ".join(f"{pearsonr([wm(ser(run,t),s-9,s) for s in STEPS],[ho[run][s] for s in STEPS])[0]:+.2f}" for t in TOP) + " |\n")

# Event analysis: collapses (drop >= .25) and recoveries (rise >= .25) between consecutive checkpoints
print("## Event analysis: training features in the 10 iterations spanning each collapse / recovery (z-scored vs run mean over all iterations)\n")
EV = ["m_grad_norm", "m_logprob_diff_abs_mean", "m_tis_frac_truncated", "acc", "comp", "reward_mean", "all_nl_per_k", "p_distinct4_mean", "all_chars", "share_eos"]
print("| run | transition | d sos | " + " | ".join(EV) + " |\n|---|---|---|" + "---|" * len(EV))
for run in RUNS:
    Z = {t: (ser(run, t) - np.nanmean(ser(run, t))) / np.nanstd(ser(run, t)) for t in EV}
    for s in STEPS:
        d = ho[run][s] - ho[run][s - 10]
        if abs(d) >= 0.25:
            print(f"| {run[-5:]} | {s-10}->{s} | {d:+.2f} | " + " | ".join(f"{wm(Z[t], s-9, s):+.2f}" for t in EV) + " |")
# mean z over collapses vs recoveries
for run in RUNS:
    Z = {t: (ser(run, t) - np.nanmean(ser(run, t))) / np.nanstd(ser(run, t)) for t in EV}
    col = [s for s in STEPS if ho[run][s] - ho[run][s - 10] <= -0.25]; rec = [s for s in STEPS if ho[run][s] - ho[run][s - 10] >= 0.25]
    print(f"\n{run}: {len(col)} collapses {col}, {len(rec)} recoveries {rec}")
    print("| feature | mean z during collapses | mean z during recoveries | mean z in window BEFORE collapses (s-19..s-10) |\n|---|---|---|---|")
    for t in EV:
        print(f"| {t} | {np.mean([wm(Z[t], s-9, s) for s in col]):+.2f} | {np.mean([wm(Z[t], s-9, s) for s in rec]):+.2f} | {np.mean([wm(Z[t], s-19, s-10) for s in col]):+.2f} |")

# single-run permutation null: max |r| over all features at W0 for one run, n=25
names = [k for k in feats[RUNS[0]][0] if k not in ("it", "qids", "modes_seq")]
rng = np.random.default_rng(1)
for run in RUNS:
    X = np.array([[wm(ser(run, n), s - 9, s) for s in STEPS] for n in names]); m = ~np.isnan(X).any(1) & (np.nanstd(X, 1) > 1e-12)
    Xz = (X[m] - X[m].mean(1, keepdims=True)) / X[m].std(1, keepdims=True); y = np.array([ho[run][s] for s in STEPS])
    mx = []
    for _ in range(3000):
        yp = rng.permutation(y); yz = (yp - yp.mean()) / yp.std(); mx.append(np.max(np.abs(Xz @ yz / len(y))))
    print(f"\n{run} single-run permutation null (3000x, {m.sum()} features, W0): 95th pct of max|r| = {np.percentile(mx,95):.2f}, 99th = {np.percentile(mx,99):.2f}; observed max|r| = {np.max(np.abs(Xz @ ((y-y.mean())/y.std()) / len(y))):.2f} ({names[np.where(m)[0][np.argmax(np.abs(Xz @ ((y-y.mean())/y.std())))]]})")

# held-out side: newline density + sentence count of sos traces vs compliance
print("\n## Held-out side: start_of_sentence trace structure per checkpoint\n")
for run in RUNS:
    rows = []
    for s in range(0, 251, 10):
        c = json.load(open(R / run / "eval" / "heldout" / f"checkpoint-{s}.json"))
        st = [rollout_stats(x.get("reasoning") or "", "none") | {"comp": int(x["compliance"])} for r in c["results"] if r["mode"] == "start_of_sentence" for x in r["samples"]]
        rows.append((s, np.mean([x["comp"] for x in st]), np.mean([x["nl_per_k"] for x in st]), np.median([x["n_sent"] for x in st]), np.median([x["chars"] for x in st]), np.mean([x["frac_sent_ok"] for x in st]), np.mean([x["frac_sent_meta"] for x in st])))
    a = np.array(rows)
    print(f"{run}: corr(sos comp, trace nl/1k) = {pearsonr(a[:,1],a[:,2])[0]:+.2f}; corr(sos comp, median n_sent) = {pearsonr(a[:,1],a[:,3])[0]:+.2f}; corr(sos comp, median chars) = {pearsonr(a[:,1],a[:,4])[0]:+.2f}; corr(sos comp, frac meta sentences) = {pearsonr(a[:,1],a[:,6])[0]:+.2f}")
    print("| step | comp | nl/1k | n_sent med | chars med | frac sent Ok | frac sent meta |\n|---|---|---|---|---|---|---|")
    for r in rows: print(f"| {int(r[0])} | {r[1]:.2f} | {r[2]:.1f} | {r[3]:.0f} | {r[4]:.0f} | {r[5]:.2f} | {r[6]:.2f} |")
    # within-checkpoint: per-trace compliance vs its own newline density, pooled over checkpoints
