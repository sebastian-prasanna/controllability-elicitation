"""Collect 2026-08-30 few-shot scaling results and render report figures.

Outputs figs/*.png and figs/data.json (all numbers used in the report).
Palette: dataviz reference palette (light mode), categorical slots in fixed order.
"""

import json
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(exist_ok=True)

# --- palette (reference instance, light mode) ---
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4
SURFACE, GRID, MUTED, INK, INK2, AXIS = ("#fcfcfb", "#e1e0d9", "#898781",
                                         "#0b0b0b", "#52514e", "#c3c2b7")
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seqblue", SEQ)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})

MODES = ["word_suppression", "multiple_word_suppression", "repeat_sentences",
         "end_of_sentence", "lowercase_thinking", "meow_between_words",
         "uppercase_thinking", "ignore_question", "alternating_case"]
LABELS = ["gptoss20b", "gptoss120b", "qwen8b", "qwen32b"]


def rd(p):
    return json.load(open(p))


def run_summary(name):
    """Find a run's summary.json under fewshot/runs/<group>/<name>/ (or flat)."""
    hits = sorted((ROOT / "fewshot/runs").glob(f"*/{name}/summary.json")) + \
        ([ROOT / f"fewshot/runs/{name}/summary.json"]
         if (ROOT / f"fewshot/runs/{name}/summary.json").exists() else [])
    return hits[0] if hits else None


def sweep(prefix, labels=LABELS, ks=range(1, 9)):
    out = {}
    for lb in labels:
        out[lb] = {}
        for k in ks:
            p = run_summary(f"{prefix}{lb}_k{k}")
            if p is None:
                continue
            s = rd(p)
            if s.get("deferred"):
                continue
            out[lb][k] = dict(c=s["compliance_rate"], a=s["accuracy"],
                              e=s["n_errors"], pm={m: v["compliant"] / v["n"]
                                                   for m, v in s["per_mode"].items()})
    return out


LABELS2 = ["kimik3", "dsv4pro", "glm53", "glm53flash"]  # remaining-4 sweep (2026-08-31)

D = {}
D["baseline"] = {lb: rd(ROOT / f"baselines/{lb}/baseline_results.json")["overall"]
                 for lb in LABELS + LABELS2}
D["main"] = sweep("")                       # synthetic SFT demos, system prompt
D["rem4"] = sweep("", LABELS2, ks=range(1, 5))
D["prefill"] = sweep("prefill_")            # same demos as prefill messages
D["gepafs"] = sweep("gepafs_", ["gptoss120b"])
D["oldpool"] = sweep("oldpool_", ["gptoss120b"], ks=(1, 10))
D["k1var"] = {}
for s_ in range(1, 6):
    d = rd(run_summary(f"gepafs_gptoss120b_k1_s{s_}"))
    D["k1var"][f"s{s_}"] = dict(c=d["compliance_rate"], a=d["accuracy"],
                                pm={m: v["compliant"] / v["n"]
                                    for m, v in d["per_mode"].items()})

def old_dirs(base):
    out = {}
    for kdir in glob(str(base / "k*_system")) + glob(str(base / "k*_prefill")):
        name = Path(kdir).name
        k, var = name.split("_")
        tot_c = tot_n = 0
        per_t = {}
        for f in glob(f"{kdir}/tgt-*/*.json"):
            d = rd(f)
            s, n = d["summary"], d["summary"]["n_rollouts"]
            tgt = Path(f).parent.name.replace("tgt-", "")
            per_t[tgt] = dict(c=s["compliance_rate"], a=s["accuracy"], n=n)
            tot_c += s["compliance_rate"] * n
            tot_n += n
        if tot_n:
            out.setdefault(var, {})[int(k[1:])] = dict(
                pooled=tot_c / tot_n, per_target=per_t)
    return out

D["old_orig"] = old_dirs(ROOT / "old/results/fewshot_scaling")
D["old_rerun"] = old_dirs(ROOT / "results/fewshot_scaling")

(FIGS / "data.json").write_text(json.dumps(D, indent=1))


def style(ax, xl="shots per mode (k)", yl="strict compliance"):
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)


# ---- fig 1: main sweep (synthetic demos), compliance + accuracy ----
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
for i, lb in enumerate(LABELS):
    ks = sorted(D["main"][lb])
    good = [k for k in ks if D["main"][lb][k]["e"] < 200]
    for ax, key in zip(axes, ("c", "a")):
        ax.plot(good, [D["main"][lb][k][key] for k in good], "-o", color=S[i],
                lw=2, ms=6, label=lb)
        ax.axhline(D["baseline"][lb]["strict_compliance" if key == "c" else "accuracy"],
                   color=S[i], lw=1, ls=(0, (2, 3)), alpha=0.55)
for ax, t in zip(axes, ("Strict compliance vs k", "Accuracy vs k")):
    style(ax, yl="strict compliance" if "compliance" in t else "accuracy")
    ax.set_title(t, color=INK, fontsize=11, loc="left")
    ax.set_ylim(0, None)
axes[0].annotate("gptoss120b", (2, 0.40), color=S[1], fontsize=9)
axes[0].annotate("gptoss20b", (6.1, 0.165), color=S[0], fontsize=9)
axes[0].annotate("qwen8b", (2.6, 0.075), color=S[2], fontsize=9)
axes[0].annotate("qwen32b", (4.18, 0.198), color=S[3], fontsize=9, va="center")
axes[1].legend(frameon=False, fontsize=8, loc="lower left")
axes[0].text(0.99, 0.97, "dashed = k=0 baseline", transform=axes[0].transAxes,
             ha="right", va="top", fontsize=8, color=MUTED)
fig.suptitle("")
fig.tight_layout()
fig.savefig(FIGS / "fig1_main_sweep.png", dpi=160)

# ---- fig 2: gptoss120b, demo-pool conditions (the money plot) ----
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ks = sorted(D["main"]["gptoss120b"])
ax.plot(ks, [D["main"]["gptoss120b"][k]["c"] for k in ks], "-o", color=S[0],
        lw=2, ms=6, label="synthetic demos (SFT pool)")
ks = sorted(D["gepafs"]["gptoss120b"])
ax.plot(ks, [D["gepafs"]["gptoss120b"][k]["c"] for k in ks], "-o", color=S[1],
        lw=2, ms=6, label="genuine demos (GEPA pool)")
op = D["oldpool"]["gptoss120b"]
ax.plot([1, 10], [op[1]["c"], op[10]["c"]], "-o", color=S[2], lw=2, ms=6,
        label="old BoN pool (genuine+synthetic)")
orr = D["old_rerun"]["system"]
ax.plot([1, 10], [orr[1]["pooled"], orr[10]["pooled"]], "--s", color=S[3], lw=2,
        ms=7, mfc="none", label="old BoN pool, old protocol (rerun)")
vs = [D["k1var"][f"s{i}"]["c"] for i in range(1, 6)]
ax.plot([1] * 5, vs, "o", ms=6, mfc="none", mec=S[1], mew=1.4, alpha=0.85)
ax.annotate("5 random k1 draws", (1.28, 0.118), fontsize=8,
            color=S[1], va="center")
ax.axhline(D["baseline"]["gptoss120b"]["strict_compliance"], color=MUTED, lw=1,
           ls=(0, (2, 3)))
ax.annotate("k=0 baseline", (8.6, D["baseline"]["gptoss120b"]["strict_compliance"] + 0.006),
            fontsize=8, color=MUTED)
style(ax)
ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 10])
ax.set_ylim(0, 0.5)
ax.set_title("gpt-oss-120b: few-shot compliance vs k by demo pool", color=INK,
             fontsize=11, loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(FIGS / "fig2_gptoss120b_pools.png", dpi=160)

# ---- fig 3: per-mode heatmaps, synthetic vs gepafs ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, cond, title in zip(axes, ("main", "gepafs"),
                           ("synthetic demos (SFT pool)", "genuine demos (GEPA pool)")):
    d = D[cond]["gptoss120b"]
    ks = sorted(d)
    M = [[d[k]["pm"][m] for k in ks] for m in MODES]
    im = ax.imshow(M, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ks)), [f"k{k}" for k in ks])
    ax.set_yticks(range(len(MODES)), MODES)
    ax.grid(False)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    for i, m in enumerate(MODES):
        for j, k in enumerate(ks):
            v = d[k]["pm"][m]
            lbl = f"{v:.2f}".replace("0.", ".", 1) if v < 0.995 else "1.0"
            ax.text(j, i, lbl, ha="center", va="center", fontsize=7,
                    color="#ffffff" if v > 0.55 else INK2)
fig.suptitle("gpt-oss-120b per-mode strict compliance", color=INK, fontsize=11,
             x=0.02, ha="left")
fig.colorbar(im, ax=axes, shrink=0.8, label="strict compliance")
fig.savefig(FIGS / "fig3_permode_heatmap.png", dpi=160, bbox_inches="tight")

# ---- fig 4: k1 demo-choice variance per mode ----
fig, ax = plt.subplots(figsize=(7.2, 4.2))
rows = MODES + ["ALL MODES"]
for i, m in enumerate(rows):
    if m == "ALL MODES":
        vals = [D["k1var"][f"s{s_}"]["c"] for s_ in range(1, 6)]
    else:
        vals = [D["k1var"][f"s{s_}"]["pm"][m] for s_ in range(1, 6)]
    ax.plot(vals, [i] * 5, "o", ms=7, mfc="none", mec=S[0], mew=1.5, alpha=0.8)
    ax.plot([min(vals), max(vals)], [i, i], "-", color=S[0], lw=1, alpha=0.4)
ax.set_yticks(range(len(rows)), rows)
ax.axhline(len(MODES) - 0.5, color=AXIS, lw=0.8)
style(ax, xl="strict compliance", yl="")
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True)
ax.set_title("gpt-oss-120b k=1: five random demo draws (GEPA pool)",
             color=INK, fontsize=10.5, loc="left")
fig.tight_layout()
fig.savefig(FIGS / "fig4_k1_variance.png", dpi=160)

# ---- fig 5: system prompt vs prefill, per model ----
fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.4), sharey=True)
for ax, lb in zip(axes, LABELS):
    ks = [k for k in sorted(D["main"][lb]) if D["main"][lb][k]["e"] < 200]
    ax.plot(ks, [D["main"][lb][k]["c"] for k in ks], "-o", color=S[0], lw=2,
            ms=5, label="system prompt")
    pk = [k for k in sorted(D["prefill"].get(lb, {}))
          if D["prefill"][lb][k]["e"] < 200]
    ax.plot(pk, [D["prefill"][lb][k]["c"] for k in pk], "-o", color=S[1], lw=2,
            ms=5, label="prefill turns")
    ax.axhline(D["baseline"][lb]["strict_compliance"], color=MUTED, lw=1,
               ls=(0, (2, 3)))
    ax.set_title(lb, color=INK, fontsize=10, loc="left")
    style(ax, yl="strict compliance" if lb == LABELS[0] else "")
    ax.set_ylim(0, 0.5)
    ax.set_xlim(0.5, 8.5)
    ax.set_xticks(range(1, 9))
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Same demos, two deliveries: system prompt vs prefill message turns "
             "(k1-k4 prefill)", color=INK, fontsize=11, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(FIGS / "fig5_prefill.png", dpi=160)

# ---- fig 6: remaining 4 models (kimi / dsv / glm), k1-k4 ----
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for i, lb in enumerate(LABELS2):
    ks = sorted(D["rem4"][lb])
    ax.plot(ks, [D["rem4"][lb][k]["c"] for k in ks], "-o", color=S[i], lw=2,
            ms=6, label=lb)
    ax.axhline(D["baseline"][lb]["strict_compliance"], color=S[i], lw=1,
               ls=(0, (2, 3)), alpha=0.55)
style(ax)
ax.set_xticks([1, 2, 3, 4])
ax.set_ylim(0, 0.45)
ax.set_title("Remaining 4 models: synthetic-demo few-shot, k1-k4 (dashed = k=0 baseline)",
             color=INK, fontsize=10.5, loc="left")
ax.legend(frameon=False, fontsize=8, loc="center right")
ax.annotate("kimik3", (3.05, 0.385), color=S[0], fontsize=9)
ax.annotate("dsv4pro", (3.05, 0.155), color=S[1], fontsize=9)
ax.annotate("glm53", (3.05, 0.062), color=S[2], fontsize=9)
ax.annotate("glm53flash", (2.5, 0.018), color=S[3], fontsize=9)
fig.tight_layout()
fig.savefig(FIGS / "fig6_remaining4.png", dpi=160)

print("figures written to", FIGS)
