"""NCRI 15.2 (nocot-bench, published models.csv) vs strict compliance: GEPA-general on the
9 in-distribution modes, GEPA-general on the 3 held-out modes, and the empty-prompt baseline.
Pinned-reeval models: GEPA seed = best of s0-s2 on in-dist (as in final_plotting.ipynb).
Single-seed new_models_sweep runs (older mixed-provider eval, in-dist only) drawn hollow."""
import csv, json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = Path(__file__).parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from utils import PALETTE, set_matplotlib_style

NCRI = {r["model_id"]: tuple(float(r[c]) for c in ("ncri15_2", "ncri15_2_lo", "ncri15_2_hi"))
        for r in csv.DictReader(open(HERE / "nocot_models_ncri15_2.csv"))}
N = {"indist": 4464, "heldout": 1500}   # rollouts per eval (test split)


def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h
PINNED = {  # key -> (label, nocot model_id)
    "kimik3": ("Kimi-K3", "moonshotai/kimi-k3"), "gptoss120b": ("gpt-oss-120b", "openai/gpt-oss-120b"),
    "gptoss20b": ("gpt-oss-20b", "openai/gpt-oss-20b"), "dsv4pro": ("DeepSeek-V4-Pro", "deepseek/deepseek-v4-pro-0813"),
    "qwen32b": ("Qwen3-32B", "qwen/qwen3-32b"), "glm53": ("GLM-5.3", "z-ai/glm-5.3"),
    "glm53flash": ("GLM-5.3-Flash", "z-ai/glm-5.3-flash"), "qwen8b": ("Qwen3-8B", "qwen/qwen3-8b")}
SINGLE = {  # new_models_sweep run -> (label, nocot model_id)
    "glm52_general": ("GLM-5.2", "z-ai/glm-5.2"), "nemotron3ultra_general": ("Nemotron-3-Ultra", "nvidia/nemotron-3-ultra-550b-a55b"),
    "kimik26_general": ("Kimi-K2.6", "moonshotai/kimi-k2.6"), "qwen38_2.4t_general": ("Qwen3.8-2.4T", "qwen/qwen3.8-2.4t-a95b"),
    "qwen38_27b_general": ("Qwen3.8-27B", "qwen/qwen3.8-27b")}


def pinned(key, name):
    return json.load(open(REPO / "pinned_reeval/runs" / key / name / "summary.json"))["compliance_rate"]


rows = []
for key, (label, mid) in PINNED.items():
    best = max(["s0", "s1", "s2"], key=lambda s: pinned(key, f"gepa_{s}_test"))
    rows.append(dict(label=label, ncri=NCRI[mid][0], ncri_lo=NCRI[mid][1], ncri_hi=NCRI[mid][2], pinned=True, gepa_indist=pinned(key, f"gepa_{best}_test"),
                     gepa_heldout=pinned(key, f"gepa_{best}_heldout"), base_indist=pinned(key, "baseline_test"),
                     base_heldout=pinned(key, "baseline_heldout")))
for run, (label, mid) in SINGLE.items():
    o = json.load(open(REPO / "gepa/runs/new_models_sweep" / run / "test_results.json"))["overall"]
    rows.append(dict(label=label, ncri=NCRI[mid][0], ncri_lo=NCRI[mid][1], ncri_hi=NCRI[mid][2], pinned=False, gepa_indist=o["strict_compliance"]))
json.dump(rows, open(HERE / "data.json", "w"), indent=1)

set_matplotlib_style()
BASE = "#333333"
OFFSET = {  # (column, label) -> label offset in points, only where the default (5, 4) collides
    ("gepa_indist", "GLM-5.2"): (-44, 4), ("gepa_indist", "Nemotron-3-Ultra"): (5, 5),
    ("gepa_indist", "Kimi-K2.6"): (-6, 9), ("gepa_heldout", "GLM-5.3"): (-20, -13),
    ("base_indist", "Qwen3-8B"): (-12, -14), ("base_indist", "gpt-oss-20b"): (-36, 7),
    ("base_indist", "Qwen3-32B"): (4, 7), ("base_indist", "gpt-oss-120b"): (-16, -14),
    ("base_indist", "GLM-5.3-Flash"): (-30, -14), ("base_indist", "DeepSeek-V4-Pro"): (4, 5)}
PANELS = [("gepa_indist", "GEPA prompt optimization, 9 in-distribution modes", PALETTE[0]),
          ("gepa_heldout", "GEPA prompt optimization, 3 out-of-distribution modes", PALETTE[0]),
          ("base_indist", "No optimization (empty prompt), 9 in-distribution modes", BASE)]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True, layout="constrained")
for ax, (col, title, color) in zip(axes, PANELS):
    pts = [r for r in rows if col in r]
    x = np.array([r["ncri"] for r in pts])
    y = 100 * np.array([r[col] for r in pts])
    # OLS fit + 95% confidence band for the mean
    fit = stats.linregress(x, y)
    xs = np.linspace(45, 105, 200)
    resid = y - (fit.intercept + fit.slope * x)
    se = np.sqrt(resid @ resid / (len(x) - 2) * (1 / len(x) + (xs - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum()))
    yhat = fit.intercept + fit.slope * xs
    t = stats.t.ppf(0.975, len(x) - 2)
    ax.fill_between(xs, yhat - t * se, yhat + t * se, color=color, alpha=0.10, lw=0, zorder=1)
    ax.plot(xs, yhat, color=color, lw=1.5, ls="--", zorder=2)
    rho, _ = stats.spearmanr(x, y)
    ax.text(0.62, 0.97, f"Pearson r = {fit.rvalue:.2f} (p = {fit.pvalue:.2f})\nSpearman ρ = {rho:.2f}, n = {len(x)}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#3d3d3a")
    n = N["heldout" if "heldout" in col else "indist"]
    for r in pts:
        p = r[col]
        lo, hi = wilson(p, n)
        ax.errorbar(r["ncri"], 100 * p, xerr=[[r["ncri"] - r["ncri_lo"]], [r["ncri_hi"] - r["ncri"]]],
                    yerr=[[100 * (p - lo)], [100 * (hi - p)]], fmt="none", ecolor=color, elinewidth=0.8,
                    alpha=0.6, capsize=0, zorder=2.5)
        ax.scatter(r["ncri"], 100 * p, s=46, zorder=3, linewidths=1.4,
                   facecolors=color if r["pinned"] else "white", edgecolors=color)
        ax.annotate(r["label"], (r["ncri"], 100 * p), xytext=OFFSET.get((col, r["label"]), (5, 4)),
                    textcoords="offset points", fontsize=8, color="#3d3d3a")
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("NCRI (no-CoT reasoning index)")
axes[0].set_ylabel("CoT controllability (%)")
axes[0].scatter([], [], s=46, facecolors=PALETTE[0], edgecolors=PALETTE[0], label="best of 3 seeds, pinned eval")
axes[0].scatter([], [], s=46, facecolors="white", edgecolors=PALETTE[0], linewidths=1.4, label="single seed, older eval")
axes[0].plot([], [], color=PALETTE[0], lw=1.5, ls="--", label="OLS fit, 95% CI")
axes[0].legend(loc="upper left", fontsize=8)
axes[0].set_xlim(45, 105)
axes[0].set_ylim(-5, 62)
fig.savefig(HERE / "ncri_vs_controllability.png", dpi=200, bbox_inches="tight")
fig.savefig(HERE / "ncri_vs_controllability.pdf", bbox_inches="tight")
