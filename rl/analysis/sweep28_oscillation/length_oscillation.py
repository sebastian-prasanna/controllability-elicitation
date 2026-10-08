"""Reasoning-length oscillation analysis for gpt-oss-120b RL runs (read-only over rl/runs).
Writes length_series.json, length_oscillation.md, length_oscillation.png in this folder."""
import json, re, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, zstandard
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/root/controllability-elicitation/rl/runs")
HERE = Path("/root/controllability-elicitation/rl/analysis/sweep28_oscillation")
RUNS = {
    "s28_120b_k30k":  ROOT / "sweep28_gptoss120b_highk/sweep28-d1-k30k",
    "s28_120b_k100k": ROOT / "sweep28_gptoss120b_highk/sweep28-d1-k100k",
    "s24_20b_k30k":   ROOT / "sweep24_controller/sweep24-fix05-k30k",
    "s24_20b_k100k":  ROOT / "sweep24_controller/sweep24-fix05-k100k",
    "s14_120b_k3k":   ROOT / "sweep14_x320gepa_gptoss120b/sweep14-x320gepa-gptoss120b-k3k",
    "s29_120b_fs1_k30k":  ROOT / "sweep29_gptoss120b_fs1/sweep29-fs1-k30k",
    "s29_120b_fs1_k100k": ROOT / "sweep29_gptoss120b_fs1/sweep29-fs1-k100k",
}
OSC = {"s28_120b_k30k", "s28_120b_k100k"}
HO_MODES = ["start_of_sentence", "no_spaces", "letter_suppression"]
PREAMBLE_RE = re.compile(r"(I must|the requirement|the constraint|every sentence|need to ensure|begin with|start with)", re.I)
TROUGHS = [10, 20, 30, 40, 130, 140, 150, 220, 230, 240, 250]
PEAKS = [50, 60, 70, 100, 170, 180, 190, 200, 210]

def load_json(f):
    f = Path(f); b = f.read_bytes()
    if f.suffix == ".zst":
        b = zstandard.ZstdDecompressor().decompress(b, max_output_size=1 << 31)
    return json.loads(b)

def split_sents(t):
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", t.strip()) if s.strip()]

def cv(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return float(np.std(x) / np.mean(x)) if len(x) and np.mean(x) > 0 else float("nan")

def maxmin(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return float(np.max(x) / np.min(x)) if len(x) and np.min(x) > 0 else float("nan")

def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 4 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    p = pearsonr(a[m], b[m]); s = spearmanr(a[m], b[m])
    return (float(p[0]), float(p[1]), float(s[0]), float(s[1]))

def fmt(x, d=2):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"

# ---------------------------------------------------------------- 1. held-out
df = pd.read_parquet(HERE / "samples.parquet")
ho = df[df.split == "heldout"].copy()
ho["chars"] = ho.reasoning.str.len()
ho["sents"] = ho.reasoning.map(lambda t: len(split_sents(t)))
ho["trunc"] = (ho.finish_reason == "length").astype(int)
ho["gt8k"] = (ho.chars > 8000).astype(int)

def agg(g):
    return pd.Series(dict(n=len(g), med_chars=g.chars.median(), mean_chars=g.chars.mean(),
                          med_sents=g.sents.median(), mean_sents=g.sents.mean(),
                          frac_gt8k=g.gt8k.mean(), trunc_rate=g.trunc.mean(), compliance=g.compliance.mean()))

ho_all = ho.groupby(["run", "step"]).apply(agg).reset_index()
ho_mode = ho.groupby(["run", "mode", "step"]).apply(agg).reset_index()

heldout_series, heldout_stats = {}, {}
for run in ho.run.unique():
    a = ho_all[ho_all.run == run].sort_values("step")
    heldout_series[run] = {"steps": a.step.tolist(), "all": {c: a[c].round(4).tolist() for c in a.columns if c not in ("run", "step")}}
    st = {"all": {"cv_med_chars": cv(a.med_chars), "maxmin_med_chars": maxmin(a.med_chars),
                  "cv_mean_chars": cv(a.mean_chars), "cv_med_sents": cv(a.med_sents)}}
    for m in HO_MODES:
        b = ho_mode[(ho_mode.run == run) & (ho_mode["mode"] == m)].sort_values("step")
        heldout_series[run][m] = {c: b[c].round(4).tolist() for c in b.columns if c not in ("run", "step", "mode")}
        pr, pp, sr, sp = corr(b.med_chars, b.compliance)
        pr2, pp2, sr2, sp2 = corr(b.mean_chars, b.compliance)
        st[m] = {"cv_med_chars": cv(b.med_chars), "maxmin_med_chars": maxmin(b.med_chars), "cv_mean_chars": cv(b.mean_chars),
                 "cv_med_sents": cv(b.med_sents), "range_med_chars": [float(b.med_chars.min()), float(b.med_chars.max())],
                 "pearson_medlen_vs_compliance": pr, "pearson_p": pp, "spearman_medlen_vs_compliance": sr, "spearman_p": sp,
                 "pearson_meanlen_vs_compliance": pr2, "spearman_meanlen_vs_compliance": sr2,
                 "cv_compliance": cv(b.compliance),
                 "cv_med_chars_excl0": cv(b.med_chars.values[1:]), "maxmin_med_chars_excl0": maxmin(b.med_chars.values[1:]),
                 "pearson_excl0": corr(b.med_chars.values[1:], b.compliance.values[1:])[0], "spearman_excl0": corr(b.med_chars.values[1:], b.compliance.values[1:])[2]}
    heldout_stats[run] = st

# ---------------------------------------------------------------- 2. in-dist (progress.jsonl)
def acf(x, lags):
    x = np.asarray(x, float); x = x - x.mean(); v = (x * x).sum()
    return {int(L): float((x[:-L] * x[L:]).sum() / v) if L < len(x) and v > 0 else float("nan") for L in lags}

indist_series, indist_stats = {}, {}
for run, rd in RUNS.items():
    pf = rd / "progress.jsonl"
    if not pf.exists(): continue
    rows = [json.loads(l) for l in pf.read_text().splitlines() if l.strip()]
    p = pd.DataFrame(rows).drop_duplicates("iteration").sort_values("iteration").reset_index(drop=True)
    p["roll10"] = p.reasoning_chars_median.rolling(10, min_periods=1).median()
    p["mean10"] = p.reasoning_chars_median.rolling(10, min_periods=1).mean()
    indist_series[run] = {"iteration": p.iteration.tolist(), "reasoning_chars_median": p.reasoning_chars_median.tolist(),
                          "roll10_median": p.roll10.round(1).tolist(), "compliance_rate": p.compliance_rate.round(4).tolist(),
                          "accuracy": p.accuracy.round(4).tolist()}
    x = p.reasoning_chars_median.values.astype(float)
    # detrend with 50-step centered rolling mean before ACF (so a slow drift does not masquerade as a period)
    trend = pd.Series(x).rolling(50, center=True, min_periods=10).mean().values
    resid = x - trend
    lags = list(range(10, 101, 10))
    st = {"n_iters": int(len(p)), "cv": cv(x), "maxmin": maxmin(x), "cv_roll10": cv(p.roll10), "maxmin_roll10": maxmin(p.roll10),
          "cv_after_warmup(>=20)": cv(x[20:]) if len(x) > 30 else float("nan"),
          "maxmin_after_warmup(>=20)": maxmin(x[20:]) if len(x) > 30 else float("nan"),
          "acf_raw": acf(x, lags) if len(x) > 110 else {}, "acf_detrended": acf(resid[~np.isnan(resid)], lags) if len(x) > 110 else {}}
    # cross-correlation with held-out series: checkpoint s <- iterations s-9..s
    if run in heldout_series:
        steps = heldout_series[run]["steps"]
        aligned = []
        for s in steps:
            w = p[(p.iteration >= max(0, s - 9)) & (p.iteration <= s)]
            aligned.append(w.reasoning_chars_median.mean() if len(w) else np.nan)
        aligned = np.array(aligned, float)
        st["aligned_indist_len"] = [None if np.isnan(v) else round(float(v), 1) for v in aligned]
        for m in HO_MODES + ["all"]:
            comp = np.array(heldout_series[run][m]["compliance"], float)
            hl = np.array(heldout_series[run][m]["med_chars"], float)
            st[f"xcorr_indistlen_vs_heldout_compliance_{m}"] = corr(aligned, comp)
            st[f"xcorr_indistlen_vs_heldout_medlen_{m}"] = corr(aligned, hl)
        # excluding step 0 (donor checkpoint: no RL yet, warm-up transient)
        st["xcorr_excl_step0_sos_compliance"] = corr(aligned[1:], np.array(heldout_series[run]["start_of_sentence"]["compliance"], float)[1:])
        st["xcorr_excl_step0_sos_medlen"] = corr(aligned[1:], np.array(heldout_series[run]["start_of_sentence"]["med_chars"], float)[1:])
    indist_stats[run] = st

# per-mode in-dist length from iters/, every 5th iteration
def permode(args):
    run, f = args
    try:
        D = load_json(f)
    except Exception as e:
        return run, f.name, None
    it = int(re.search(r"iter-(\d+)", f.name).group(1))
    out = {}
    for r in D["results"]:
        for s in r["samples"]:
            out.setdefault(r["mode"], []).append((len(s.get("reasoning") or ""), int(s.get("compliance") or 0), s.get("finish_reason") == "length"))
    return run, it, {m: dict(n=len(v), med_chars=float(np.median([a for a, _, _ in v])), mean_chars=float(np.mean([a for a, _, _ in v])),
                           compliance=float(np.mean([c for _, c, _ in v])), trunc_rate=float(np.mean([t for _, _, t in v]))) for m, v in out.items()}

jobs = []
for run, rd in RUNS.items():
    for f in sorted((rd / "iters").glob("iter-*.json*")):
        it = int(re.search(r"iter-(\d+)", f.name).group(1))
        if it % 5 == 0 or run.startswith("s29"):
            jobs.append((run, f))
permode_series = {}
with ProcessPoolExecutor(16) as ex:
    for run, it, res in ex.map(permode, jobs, chunksize=2):
        if res is None: continue
        permode_series.setdefault(run, {})[it] = res
permode_stats = {}
for run, d in permode_series.items():
    its = sorted(d)
    modes = sorted({m for it in its for m in d[it]})
    permode_stats[run] = {"iterations": its, "modes": {}}
    for m in modes:
        med = [d[it][m]["med_chars"] if m in d[it] else np.nan for it in its]
        comp = [d[it][m]["compliance"] if m in d[it] else np.nan for it in its]
        tr = [d[it][m]["trunc_rate"] if m in d[it] else np.nan for it in its]
        permode_stats[run]["modes"][m] = {"med_chars": [None if np.isnan(v) else round(v) for v in med],
                                          "compliance": [None if np.isnan(v) else round(v, 3) for v in comp],
                                          "trunc_rate": [None if np.isnan(v) else round(v, 3) for v in tr],
                                          "cv": cv(med), "maxmin": maxmin(med),
                                          "cv_after_warmup": cv(np.array(med)[np.array(its) >= 20]) if len(its) > 6 else float("nan")}
        # cross-correlation of per-mode in-dist median length with held-out sos compliance (nearest checkpoint <= iteration, window s-9..s)
        if run in heldout_series:
            steps = heldout_series[run]["steps"]; comp_sos = heldout_series[run]["start_of_sentence"]["compliance"]
            al = []
            for s in steps:
                vals = [d[it][m]["med_chars"] for it in its if max(0, s - 9) <= it <= s and m in d[it]]
                al.append(np.mean(vals) if vals else np.nan)
            permode_stats[run]["modes"][m]["xcorr_vs_heldout_sos_compliance"] = corr(al, comp_sos)

# ---------------------------------------------------------------- 3. decomposition (sweep28 k30k, sos)
def decompose(t):
    sents = split_sents(t)
    k = 0
    for s in sents:
        if PREAMBLE_RE.search(s): k += 1
        else: break
    pre, body = sents[:k], sents[k:]
    n_meta_total = sum(bool(PREAMBLE_RE.search(s)) for s in sents)
    return dict(pre_chars=sum(len(s) for s in pre), pre_sents=k, body_chars=sum(len(s) for s in body), body_sents=len(body),
                n_sents=len(sents), meta_sents_total=n_meta_total, meta_chars_total=sum(len(s) for s in sents if PREAMBLE_RE.search(s)))

sos = ho[(ho.run == "s28_120b_k30k") & (ho["mode"] == "start_of_sentence")].copy()
dec = pd.DataFrame([decompose(t) for t in sos.reasoning], index=sos.index)
sos = pd.concat([sos, dec], axis=1)
sos["phase"] = np.where(sos.step.isin(TROUGHS), "trough", np.where(sos.step.isin(PEAKS), "peak", "other"))
DEC_COLS = ["chars", "n_sents", "pre_chars", "pre_sents", "body_chars", "body_sents", "meta_chars_total", "meta_sents_total"]
dec_phase = sos[sos.phase != "other"].groupby("phase")[DEC_COLS].median()
dec_phase_mean = sos[sos.phase != "other"].groupby("phase")[DEC_COLS].mean()
dec_phase_comp = sos[sos.phase != "other"].groupby("phase").compliance.mean()
dec_step = sos.groupby("step")[DEC_COLS].median()
dec_step["compliance"] = sos.groupby("step").compliance.mean()
dec_step["frac_has_preamble"] = sos.groupby("step").pre_sents.apply(lambda s: (s > 0).mean())
# within-phase, compliant vs non-compliant
dec_phase_c = sos[sos.phase != "other"].groupby(["phase", "compliance"])[DEC_COLS].median()
# same decomposition for k100k (for reference)
sos100 = ho[(ho.run == "s28_120b_k100k") & (ho["mode"] == "start_of_sentence")].copy()
dec100 = pd.DataFrame([decompose(t) for t in sos100.reasoning], index=sos100.index)
sos100 = pd.concat([sos100, dec100], axis=1)
dec100_step = sos100.groupby("step")[DEC_COLS].median(); dec100_step["compliance"] = sos100.groupby("step").compliance.mean()

# ---------------------------------------------------------------- save JSON
out = {"heldout_series": heldout_series, "heldout_stats": heldout_stats, "indist_series": indist_series, "indist_stats": indist_stats,
       "permode_indist": permode_stats,
       "decomposition_k30k": {"troughs": TROUGHS, "peaks": PEAKS, "phase_median": dec_phase.round(1).to_dict(), "phase_mean": dec_phase_mean.round(1).to_dict(),
                              "phase_compliance": dec_phase_comp.round(3).to_dict(),
                              "per_step_median": {int(k): v for k, v in dec_step.round(3).to_dict(orient="index").items()},
                              "phase_by_compliance_median": {f"{a}_c{b}": v for (a, b), v in dec_phase_c.round(1).to_dict(orient="index").items()}},
       "decomposition_k100k_per_step_median": {int(k): v for k, v in dec100_step.round(3).to_dict(orient="index").items()}}
def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    return o
(HERE / "length_series.json").write_text(json.dumps(clean(out), indent=1))

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
ax = axes[0]; ax2 = ax.twinx()
cols = {"s28_120b_k30k": "#1f77b4", "s28_120b_k100k": "#d62728"}
for run, c in cols.items():
    s = heldout_series[run]["steps"]; m = heldout_series[run]["start_of_sentence"]
    ax.plot(s, m["med_chars"], "-o", color=c, ms=4, label=f"{run} median len (sos)")
    ax2.plot(s, m["compliance"], "--s", color=c, ms=4, alpha=.6, label=f"{run} strict compliance (sos)")
ax.set_xlabel("checkpoint step"); ax.set_ylabel("held-out median reasoning chars (start_of_sentence)"); ax2.set_ylabel("strict compliance (sos)")
ax2.set_ylim(0, 1); ax.set_title("Held-out (T=0): median length (solid) vs compliance (dashed)")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
ax = axes[1]
pal = {"s28_120b_k30k": "#1f77b4", "s28_120b_k100k": "#d62728", "s24_20b_k30k": "#2ca02c", "s24_20b_k100k": "#9467bd", "s14_120b_k3k": "#8c564b",
       "s29_120b_fs1_k30k": "#17becf", "s29_120b_fs1_k100k": "#e377c2"}
for run, s in indist_series.items():
    lw = 2.2 if run in OSC else 1.3
    ax.plot(s["iteration"], s["roll10_median"], color=pal[run], lw=lw, label=f"{run} (CV {indist_stats[run]['cv']:.2f})")
ax.set_xlabel("iteration"); ax.set_ylabel("in-dist reasoning_chars_median (10-iter rolling median)"); ax.set_title("In-dist training rollouts (T=1)")
ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(HERE / "length_oscillation.png", dpi=130)

# ---------------------------------------------------------------- markdown
L = []
L.append("# Reasoning-length oscillation in gpt-oss-120b RL runs\n")
L.append("Read-only analysis of `rl/runs/sweep28_gptoss120b_highk/*` (oscillating held-out compliance) vs stable comparison runs "
         "(`sweep24-fix05-k30k/k100k` gpt-oss-20b same recipe; `sweep14-x320gepa-gptoss120b-k3k` 120b small-k older recipe) and the in-progress "
         "`sweep29-fs1-k30k/k100k` (fewshot donors, in-dist only). Held-out = T=0 eval of each checkpoint on 200 held-out questions "
         "(start_of_sentence / no_spaces / letter_suppression); in-dist = T=1 training rollouts (256/step) from `progress.jsonl` and `iters/`. "
         "Length = chars of the `reasoning` field; sentences split on `.!?` + newlines. Strict compliance = `compliance == 1`. "
         "Series in `length_series.json`, figure `length_oscillation.png`, script `length_oscillation.py`.\n")

L.append("## 1. Held-out length series\n")
L.append("### 1a. Summary across checkpoints (26 checkpoints, steps 0..250)\n")
L.append("| run | mode | CV median chars | max/min median chars | CV excl. step 0 | max/min excl. step 0 | median-chars range | CV median sents | CV compliance | Pearson r (med len vs compliance) | p | Spearman rho | p | Pearson/Spearman excl. step 0 |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for run in RUNS:
    if run not in heldout_stats: continue
    st = heldout_stats[run]
    L.append(f"| {run} | all | {fmt(st['all']['cv_med_chars'])} | {fmt(st['all']['maxmin_med_chars'])} | | | | {fmt(st['all']['cv_med_sents'])} | | | | | | |")
    for m in HO_MODES:
        s = st[m]
        L.append(f"| {run} | {m} | {fmt(s['cv_med_chars'])} | {fmt(s['maxmin_med_chars'])} | {fmt(s['cv_med_chars_excl0'])} | {fmt(s['maxmin_med_chars_excl0'])} | {s['range_med_chars'][0]:.0f}-{s['range_med_chars'][1]:.0f} | {fmt(s['cv_med_sents'])} | {fmt(s['cv_compliance'])} | "
                 f"{fmt(s['pearson_medlen_vs_compliance'])} | {fmt(s['pearson_p'],3)} | {fmt(s['spearman_medlen_vs_compliance'])} | {fmt(s['spearman_p'],3)} | {fmt(s['pearson_excl0'])} / {fmt(s['spearman_excl0'])} |")
L.append("")
L.append("Mean-length (not median) correlations with compliance, start_of_sentence: " + "; ".join(
    f"{run}: Pearson {fmt(heldout_stats[run]['start_of_sentence']['pearson_meanlen_vs_compliance'])}, Spearman {fmt(heldout_stats[run]['start_of_sentence']['spearman_meanlen_vs_compliance'])}"
    for run in RUNS if run in heldout_stats) + "\n")

for run in RUNS:
    if run not in heldout_series: continue
    L.append(f"### 1b. {run}: per-checkpoint held-out table\n")
    L.append("| step | all med chars | all mean chars | all med sents | all >8k | all trunc | sos med chars | sos mean chars | sos med sents | sos >8k | sos trunc | sos compl | nosp med chars | nosp compl | nosp trunc | lsup med chars | lsup compl | lsup trunc |")
    L.append("|" + "---|" * 18)
    hs = heldout_series[run]
    for i, s in enumerate(hs["steps"]):
        a = hs["all"]; so = hs["start_of_sentence"]; ns = hs["no_spaces"]; ls = hs["letter_suppression"]
        L.append(f"| {s} | {a['med_chars'][i]:.0f} | {a['mean_chars'][i]:.0f} | {a['med_sents'][i]:.0f} | {a['frac_gt8k'][i]:.2f} | {a['trunc_rate'][i]:.2f} | "
                 f"{so['med_chars'][i]:.0f} | {so['mean_chars'][i]:.0f} | {so['med_sents'][i]:.0f} | {so['frac_gt8k'][i]:.2f} | {so['trunc_rate'][i]:.2f} | {so['compliance'][i]:.2f} | "
                 f"{ns['med_chars'][i]:.0f} | {ns['compliance'][i]:.2f} | {ns['trunc_rate'][i]:.2f} | {ls['med_chars'][i]:.0f} | {ls['compliance'][i]:.2f} | {ls['trunc_rate'][i]:.2f} |")
    L.append("")

L.append("## 2. In-dist length series (progress.jsonl, `reasoning_chars_median`, T=1 training rollouts)\n")
L.append("| run | n iters | CV raw | max/min raw | CV (iters>=20) | max/min (iters>=20) | CV 10-roll | max/min 10-roll | ACF detrended lag10/20/30/50/100 |")
L.append("|---|---|---|---|---|---|---|---|---|")
for run, st in indist_stats.items():
    a = st["acf_detrended"]
    acfs = "/".join(fmt(a.get(k, float('nan'))) for k in (10, 20, 30, 50, 100)) if a else "n/a"
    L.append(f"| {run} | {st['n_iters']} | {fmt(st['cv'])} | {fmt(st['maxmin'])} | {fmt(st['cv_after_warmup(>=20)'])} | {fmt(st['maxmin_after_warmup(>=20)'])} | {fmt(st['cv_roll10'])} | {fmt(st['maxmin_roll10'])} | {acfs} |")
L.append("")
L.append("Raw (non-detrended) ACF lag 10..100 by run:\n")
for run, st in indist_stats.items():
    if st["acf_raw"]:
        L.append(f"- {run}: " + ", ".join(f"L{k}={fmt(v)}" for k, v in st["acf_raw"].items()))
L.append("")
L.append("### 2b. Cross-correlation: 10-step in-dist median length (mean of iters s-9..s) vs held-out series at checkpoint s\n")
L.append("| run | vs sos compliance (Pearson/Spearman) | vs sos compliance excl. step 0 | vs sos held-out median len | vs sos held-out med len excl. step 0 | vs no_spaces compl | vs letter_supp compl | vs all-mode held-out med len |")
L.append("|---|---|---|---|---|---|---|---|")
for run, st in indist_stats.items():
    if "aligned_indist_len" not in st: continue
    def pr(k): x = st[k]; return f"{fmt(x[0])} / {fmt(x[2])}"
    L.append(f"| {run} | {pr('xcorr_indistlen_vs_heldout_compliance_start_of_sentence')} | {pr('xcorr_excl_step0_sos_compliance')} | {pr('xcorr_indistlen_vs_heldout_medlen_start_of_sentence')} | {pr('xcorr_excl_step0_sos_medlen')} | "
             f"{pr('xcorr_indistlen_vs_heldout_compliance_no_spaces')} | {pr('xcorr_indistlen_vs_heldout_compliance_letter_suppression')} | {pr('xcorr_indistlen_vs_heldout_medlen_all')} |")
L.append("")
L.append("Per-checkpoint aligned in-dist length (mean reasoning_chars_median over iters s-9..s) next to held-out sos compliance / sos median length:\n")
for run in ("s28_120b_k30k", "s28_120b_k100k"):
    st = indist_stats[run]; hs = heldout_series[run]
    L.append(f"**{run}**\n")
    L.append("| step | " + " | ".join(str(s) for s in hs["steps"]) + " |")
    L.append("|---|" + "---|" * len(hs["steps"]))
    L.append("| in-dist len | " + " | ".join("" if v is None else f"{v:.0f}" for v in st["aligned_indist_len"]) + " |")
    L.append("| held-out sos compl | " + " | ".join(f"{v:.2f}" for v in hs["start_of_sentence"]["compliance"]) + " |")
    L.append("| held-out sos med len | " + " | ".join(f"{v:.0f}" for v in hs["start_of_sentence"]["med_chars"]) + " |")
    L.append("")

L.append("### 2c. Per-mode in-dist median length from iters/ (every 5th iteration; sweep29: all available)\n")
L.append("| run | mode | n pts | CV med chars | max/min | CV (iters>=20) | Pearson/Spearman vs held-out sos compliance |")
L.append("|---|---|---|---|---|---|---|")
for run, ps in permode_stats.items():
    for m, s in ps["modes"].items():
        x = s.get("xcorr_vs_heldout_sos_compliance")
        xc = f"{fmt(x[0])} / {fmt(x[2])}" if x else "n/a"
        L.append(f"| {run} | {m} | {len(ps['iterations'])} | {fmt(s['cv'])} | {fmt(s['maxmin'])} | {fmt(s['cv_after_warmup'])} | {xc} |")
L.append("")
for run in ("s28_120b_k30k", "s28_120b_k100k"):
    ps = permode_stats.get(run)
    if not ps: continue
    L.append(f"**{run}** per-mode median chars by iteration (every 5th)\n")
    its = ps["iterations"]
    L.append("| iter | " + " | ".join(ps["modes"].keys()) + " |")
    L.append("|---|" + "---|" * len(ps["modes"]))
    for i, it in enumerate(its):
        L.append(f"| {it} | " + " | ".join("" if s["med_chars"][i] is None else f"{s['med_chars'][i]}" for s in ps["modes"].values()) + " |")
    L.append("")

L.append("## 3. Preamble/body decomposition, sweep28 k30k held-out start_of_sentence\n")
L.append(f"Preamble = leading run of sentences matching `{PREAMBLE_RE.pattern}` (case-insensitive); body = remainder. "
         f"Troughs = steps {TROUGHS}; peaks = steps {PEAKS}. `meta_*_total` counts matching sentences anywhere in the trace (not just leading).\n")
L.append("| phase | n traces | strict compl | median total chars | median sents | median preamble chars | median preamble sents | median body chars | median body sents | median meta chars (anywhere) | median meta sents (anywhere) |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|")
for ph in ("peak", "trough"):
    r = dec_phase.loc[ph]; n = int((sos.phase == ph).sum())
    L.append(f"| {ph} | {n} | {dec_phase_comp[ph]:.2f} | {r.chars:.0f} | {r.n_sents:.0f} | {r.pre_chars:.0f} | {r.pre_sents:.0f} | {r.body_chars:.0f} | {r.body_sents:.0f} | {r.meta_chars_total:.0f} | {r.meta_sents_total:.0f} |")
L.append("")
L.append("Means instead of medians:\n")
L.append("| phase | mean total chars | mean preamble chars | mean body chars | mean meta chars (anywhere) | mean meta sents (anywhere) |")
L.append("|---|---|---|---|---|---|")
for ph in ("peak", "trough"):
    r = dec_phase_mean.loc[ph]
    L.append(f"| {ph} | {r.chars:.0f} | {r.pre_chars:.0f} | {r.body_chars:.0f} | {r.meta_chars_total:.0f} | {r.meta_sents_total:.1f} |")
L.append("")
L.append("Split by compliance within phase (medians):\n")
L.append("| phase / compliant | n | total chars | preamble chars | body chars | meta chars anywhere | meta sents anywhere |")
L.append("|---|---|---|---|---|---|---|")
for (ph, c), r in dec_phase_c.iterrows():
    n = int(((sos.phase == ph) & (sos.compliance == c)).sum())
    L.append(f"| {ph} / c={c} | {n} | {r.chars:.0f} | {r.pre_chars:.0f} | {r.body_chars:.0f} | {r.meta_chars_total:.0f} | {r.meta_sents_total:.0f} |")
L.append("")
L.append("### 3b. Compliance at fixed length: strict compliance by sentence-count bucket, peak vs trough (k30k sos)\n")
sos["sbin"] = pd.cut(sos.n_sents, [0, 10, 15, 20, 25, 30, 40, 60, 10**6], labels=["1-10", "11-15", "16-20", "21-25", "26-30", "31-40", "41-60", ">60"])
bk = sos[sos.phase != "other"].groupby(["sbin", "phase"], observed=True).compliance.agg(["mean", "size"]).unstack("phase")
L.append("| sentences | peak compl (n) | trough compl (n) |")
L.append("|---|---|---|")
for b_, r in bk.iterrows():
    L.append(f"| {b_} | {r[('mean','peak')]:.2f} ({int(r[('size','peak')])}) | {r[('mean','trough')]:.2f} ({int(r[('size','trough')])}) |")
L.append("")
# composition check: does the peak/trough length gap survive conditioning on compliance?
L.append("Composition check: predicted phase median total chars if only the compliant/non-compliant mix changed (weights = phase compliance, lengths = within-phase c=0/c=1 medians): "
         + ", ".join(f"{ph}: {dec_phase_comp[ph]*dec_phase_c.loc[(ph,1)].chars + (1-dec_phase_comp[ph])*dec_phase_c.loc[(ph,0)].chars:.0f} (observed {dec_phase.loc[ph].chars:.0f})" for ph in ("peak", "trough")) + "\n")
out["decomposition_k30k"]["compliance_by_sentence_bucket"] = {str(b_): {"peak": [float(r[('mean','peak')]), int(r[('size','peak')])], "trough": [float(r[('mean','trough')]), int(r[('size','trough')])]} for b_, r in bk.iterrows()}
(HERE / "length_series.json").write_text(json.dumps(clean(out), indent=1))
L.append("Per-step medians (k30k sos):\n")
L.append("| step | compl | total chars | sents | preamble chars | preamble sents | frac w/ preamble | body chars | body sents | meta chars anywhere | meta sents anywhere |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|")
for s, r in dec_step.iterrows():
    tag = "T" if s in TROUGHS else ("P" if s in PEAKS else "")
    L.append(f"| {s} {tag} | {r.compliance:.2f} | {r.chars:.0f} | {r.n_sents:.0f} | {r.pre_chars:.0f} | {r.pre_sents:.0f} | {r.frac_has_preamble:.2f} | {r.body_chars:.0f} | {r.body_sents:.0f} | {r.meta_chars_total:.0f} | {r.meta_sents_total:.0f} |")
L.append("")
L.append("k100k sos per-step medians (reference):\n")
L.append("| step | compl | total chars | preamble chars | body chars | meta chars anywhere |")
L.append("|---|---|---|---|---|---|")
for s, r in dec100_step.iterrows():
    L.append(f"| {s} | {r.compliance:.2f} | {r.chars:.0f} | {r.pre_chars:.0f} | {r.body_chars:.0f} | {r.meta_chars_total:.0f} |")
L.append("")
L.append("## 4. Conclusion\n")
L.append("<<CONCLUSION>>\n")
(HERE / "length_oscillation.md").write_text("\n".join(L))
print("written", HERE / "length_oscillation.md")
# print key numbers for the operator
print(json.dumps(clean({r: {m: {k: v for k, v in heldout_stats[r][m].items() if k in ("cv_med_chars", "maxmin_med_chars", "pearson_medlen_vs_compliance", "spearman_medlen_vs_compliance", "spearman_p", "cv_compliance")} for m in heldout_stats[r]} for r in heldout_stats}), indent=0))
print(json.dumps(clean({r: {k: v for k, v in s.items() if k.startswith(("cv", "maxmin", "acf_detrended", "xcorr"))} for r, s in indist_stats.items()}), indent=0))
print(dec_phase.round(0)); print(dec_phase_mean.round(0)); print(dec_phase_comp)
print(dec_step.round(2).to_string())
