import json, gzip, numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
A = Path("rl/analysis/sweep28_oscillation"); R = Path("rl/runs/sweep28_gptoss120b_highk")
RUNS = ["sweep28-d1-k30k", "sweep28-d1-k100k"]
feats = json.load(open(A / "training_features.json"))
ho = {run: {c["step"]: c["per_mode"]["start_of_sentence"]["compliant"] / c["per_mode"]["start_of_sentence"]["n"] for c in json.load(open(R / run / "eval_summary.json"))["evals"]["heldout"]["checkpoints"]} for run in RUNS}
def ser(run, n): return np.array([np.nan if f[n] is None else f[n] for f in feats[run]], float)
def wm(x, lo, hi):
    lo, hi = max(lo, 0), min(hi, 249); return np.nanmean(x[lo:hi + 1]) if hi >= lo else np.nan
STEPS = list(range(10, 251, 10))
def detr(y): t = np.arange(len(y)); return y - np.polyval(np.polyfit(t, y, 1), t)
TOP = ["acc", "m_grad_norm", "m_logprob_diff_abs_mean", "m_tis_frac_truncated", "all_nl_per_k", "p_distinct4_mean", "comp", "reward_mean", "all_has_meta", "all_chars", "share_eos", "comp_end_of_sentence", "all_frac_sent_so"]
print("## Robustness: raw vs linearly-detrended vs first-difference Pearson r (W0 = s-9..s)\n")
print("| feature | k30k raw | k30k detr | k30k diff | k100k raw | k100k detr | k100k diff |\n|---|---|---|---|---|---|---|")
for t in TOP:
    cells = []
    for run in RUNS:
        x = np.array([wm(ser(run, t), s - 9, s) for s in STEPS]); y = np.array([ho[run][s] for s in STEPS])
        cells += [pearsonr(x, y)[0], pearsonr(detr(x), detr(y))[0], pearsonr(np.diff(x), np.diff(y))[0]]
    print(f"| {t} | " + " | ".join(f"{c:+.2f}" for c in cells) + " |")
# held-out trend itself
for run in RUNS:
    y = np.array([ho[run][s] for s in STEPS]); print(f"{run}: held-out linear trend slope per 10 steps = {np.polyfit(np.arange(len(y)), y, 1)[0]:+.3f}, frac var explained by trend = {1 - np.var(detr(y))/np.var(y):.2f}")

# newline regime: per-iteration fraction of rollouts with nl_per_k>20, by mode
print("\n## Newline ('one sentence per line') regime in training rollouts\n")
for run in RUNS:
    rows = [json.loads(l) for l in gzip.open(A / "cache" / f"{run}_rollouts.jsonl.gz", "rt")]
    by_it = defaultdict(list)
    for r in rows: by_it[r["it"]].append(r)
    frac_hi = np.array([np.mean([r["nl_per_k"] > 20 for r in by_it[t]]) for t in range(250)])
    x = np.array([wm(frac_hi, s - 9, s) for s in STEPS]); y = np.array([ho[run][s] for s in STEPS])
    print(f"{run}: frac rollouts nl/1k>20 vs held-out sos: r={pearsonr(x,y)[0]:+.2f} rho={spearmanr(x,y)[0]:+.2f} detr r={pearsonr(detr(x),detr(y))[0]:+.2f} diff r={pearsonr(np.diff(x),np.diff(y))[0]:+.2f}")
    # per-10-step by mode
    print("| window | " + " | ".join(m[:6] for m in ["repeat_sentences", "end_of_sentence", "lowercase_thinking", "uppercase_thinking", "meow_between_words", "alternating_case"]) + " | all | heldout sos |\n|---|---|---|---|---|---|---|---|---|")
    for s in STEPS:
        sub = [r for t in range(s - 9, s + 1) for r in by_it[t]]
        cells = [np.mean([r["nl_per_k"] for r in sub if r["mode"] == m]) for m in ["repeat_sentences", "end_of_sentence", "lowercase_thinking", "uppercase_thinking", "meow_between_words", "alternating_case"]]
        print(f"| {s-9}-{s} | " + " | ".join(f"{c:.1f}" for c in cells) + f" | {np.mean([r['nl_per_k'] for r in sub]):.1f} | {ho[run][s]:.2f} |")
    # which rollouts carry the newlines: compliant vs not, correct vs not, in 145-175
    sub = [r for t in range(141, 171) for r in by_it[t]]
    print(f"iters 141-170: mean nl/1k compliant={np.mean([r['nl_per_k'] for r in sub if r['comp']]):.1f} noncompliant={np.mean([r['nl_per_k'] for r in sub if not r['comp']]):.1f}; frac nl>20: {np.mean([r['nl_per_k']>20 for r in sub]):.3f}")
    # does the newline regime carry positive advantage? advcorr_nl_per_k over windows
    adv = ser(run, "advcorr_nl_per_k"); print("advcorr_nl_per_k by window: " + " ".join(f"{s}:{wm(adv,s-9,s):+.2f}" for s in STEPS))
