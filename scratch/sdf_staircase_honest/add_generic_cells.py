import json
from pathlib import Path
NB = Path("final_plotting.ipynb"); nb = json.loads(NB.read_text()); cells = nb["cells"]
assert not any("".join(c["source"]).startswith("### Generic pretraining text alone") for c in cells)
idx = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and "".join(c["source"]).startswith('key = "gptoss120b"')]
assert len(idx) == 1
md = '''### Generic pretraining text alone raises controllability

Subset of the staircase: untrained base (ATLAS-5 prompt), C4-only midtraining, and placebo documents about ATLAS-5 that say nothing about reasoning. Neither training corpus mentions chain-of-thought control, yet both lift honest compliance on the training modes and on the never-mentioned held-out modes. (a)/(b): gpt-oss-120b (main-text version). (c)/(d): all four models; the lift is a gpt-oss phenomenon, Qwen is flat. Same data and protocol as the matrix above (T=1, ATLAS-5 identity, test split, honest compliance, binomial SE).
'''
code = '''GENERIC_BARS = [("base", "Base\\nmodel", GRAY), ("c4only80k", "C4 only", BASELINE_COLOR), ("placebo", "Placebo\\ndocs", PALETTE[1])]

def generic_vals(key, blk):
    return np.array([sdf_base[(key, blk)] if arm == "base" else sdf_id_agg.loc[(key, arm, blk), "honest"] for arm, *_ in GENERIC_BARS])

# (a)/(b) gpt-oss-120b only, staircase style
key = "gptoss120b"
s = paper_fonts(4.6, frac=0.66)
fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.3), sharey=True, layout="constrained")
x = np.arange(len(GENERIC_BARS))
for ax, (blk, blabel, n) in zip(axes, SDF_BLOCKS):
    vals = generic_vals(key, blk)
    for xi, (arm, _, color), v in zip(x, GENERIC_BARS, vals):
        ax.bar(xi, 100 * v, 0.72, **bar_style(color, "///" if arm == "c4only80k" else None), zorder=2)
        ax.annotate(f"{100 * v:.0f}", (xi, 100 * (v + binom_se(v, n))), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7 * s, color=INK)
    ax.errorbar(x, 100 * vals, yerr=100 * binom_se(vals, n), fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.3, zorder=3)
    ax.set_xticks(x, [l for _, l, _ in GENERIC_BARS])
    ax.set_title(blabel, pad=4)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel(f"{METRIC_LABEL['honest_compliance']} (%)")
axes[0].set_ylim(0, 30)
panel_labels(axes, x=-0.14)
save(fig, "sdf_generic_pretraining")
plt.show()

# (c)/(d) all four models, grouped
s = paper_fonts(9, frac=1.0)
fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), sharey=True, layout="constrained")
x = np.arange(len(TRAINED_MODELS)); w, gap = 0.24, 0.02
for ax, (blk, blabel, n) in zip(axes, SDF_BLOCKS):
    for j, (arm, alabel, color) in enumerate(GENERIC_BARS):
        xs = x + (j - 1) * (w + gap)
        vals = np.array([generic_vals(k, blk)[j] for k in TRAINED_MODELS])
        ax.bar(xs, 100 * vals, w, **bar_style(color, "///" if arm == "c4only80k" else None), label=alabel.replace("\\n", " "), zorder=2)
        ax.errorbar(xs, 100 * vals, yerr=100 * binom_se(vals, n), fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.3, zorder=3)
        for xi, v in zip(xs, vals):
            ax.annotate(f"{100 * v:.0f}", (xi, 100 * (v + binom_se(v, n))), xytext=(0, 1.5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5 * s, color=INK)
    ax.set_xticks(x, [MODEL_LABEL[k] for k in TRAINED_MODELS])
    ax.set_title(blabel, pad=4)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel(f"{METRIC_LABEL['honest_compliance']} (%)")
axes[0].set_ylim(0, 30)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=3)
panel_labels(axes, x=-0.08)
save(fig, "sdf_generic_pretraining_all_models")
plt.show()
'''
cells.insert(idx[0] + 1, {"cell_type": "markdown", "metadata": {}, "source": md.splitlines(keepends=True)})
cells.insert(idx[0] + 2, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code.splitlines(keepends=True)})
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\\n")
print(f"inserted at {idx[0]+1},{idx[0]+2}; {len(cells)} cells")
