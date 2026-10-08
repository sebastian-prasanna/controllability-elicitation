"""Insert the T=1 SDF cells (markdown + code) right after the gpt-oss-120b staircase cell. Additive only."""
import json
from pathlib import Path
NB = Path("final_plotting.ipynb")
nb = json.loads(NB.read_text())
cells = nb["cells"]
assert not any("".join(c["source"]).startswith("### Same figures at sampling temperature 1.0 (base model") for c in cells), "T=1 SDF cells already present"
idx = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and "".join(c["source"]).startswith('key = "gptoss120b"')]
assert len(idx) == 1, idx
md = '''### Same figures at sampling temperature 1.0 (base model with the ATLAS-5 prompt)

T=1 replica of the matrix and staircase above: same 20 final checkpoints, ATLAS-5 identity blocks, test split, mode `all`, 16k tokens, vLLM sampling seed 0 (`sft/test_eval.py ... --temperature 1.0 --tag g16k_t1_id`, 2026-10-06; artifacts `eval/test_g16k_t1_id_{cotcontrol_id,heldout_id}/`). The base-model reference is now the correct control: the **untrained model evaluated with the same ATLAS-5 identity prompt** (`sdf/base_identity_eval.py` -> `sdf/runs/base_identity/<model>/eval/test_g16k_t1_id_<block>/base.json`), whereas the T=0 figures use the empty-prompt baselines. Honest compliance throughout; the random-LoRA control cannot be replicated at T=1 (adapters were deleted after the T=0 evals). Caches `paper/cache/sdf_atlas5p_test_id_t1_rollouts.parquet`, `sdf_base_identity_t1_rollouts.parquet`.
'''
code = '''# ---- T=1 data: trained runs (identity blocks) + base model with the identity prompt ----------------------------
SDF_ID_T1_CACHE = REPO / "paper/cache/sdf_atlas5p_test_id_t1_rollouts.parquet"
SDF_ID_T1_BLOCKS = {"test_g16k": "test_g16k_t1_id_cotcontrol_id", "test_g16k_heldout": "test_g16k_t1_id_heldout_id"}
SDF_BASE_T1_CACHE = REPO / "paper/cache/sdf_base_identity_t1_rollouts.parquet"
SDF_BASE_T1_DIR = REPO / "sdf/runs/base_identity"


def build_sdf_id_t1_cache():
    rows = []
    for key in TRAINED_MODELS:
        for arm, *_ in SDF_ARMS:
            for blk, idblk in SDF_ID_T1_BLOCKS.items():
                files = sorted((SDF_RUNS / f"sdf-atlas5p-{key}-{arm}" / "eval" / idblk).glob("checkpoint-*.json"))
                assert len(files) == 1, (key, arm, idblk, files)
                for x in json.load(files[0].open())["results"]:
                    for s in x["samples"]:
                        rows.append(dict(model=key, arm=arm, block=blk, mode=x["mode"], **sdf_sample_row(s)))
    df = pd.DataFrame(rows)
    df.to_parquet(SDF_ID_T1_CACHE)
    return df


def build_sdf_base_t1_cache():
    rows = []
    for key in TRAINED_MODELS:
        for blk, idblk in SDF_ID_T1_BLOCKS.items():
            f = SDF_BASE_T1_DIR / key / "eval" / idblk / "base.json"
            for x in json.load(f.open())["results"]:
                for s in x["samples"]:
                    rows.append(dict(model=key, block=blk, mode=x["mode"], **sdf_sample_row(s)))
    df = pd.DataFrame(rows)
    df.to_parquet(SDF_BASE_T1_CACHE)
    return df

sdf_id_t1 = pd.read_parquet(SDF_ID_T1_CACHE) if SDF_ID_T1_CACHE.exists() else build_sdf_id_t1_cache()
sdf_base_t1 = pd.read_parquet(SDF_BASE_T1_CACHE) if SDF_BASE_T1_CACHE.exists() else build_sdf_base_t1_cache()
sdf_id_t1_agg = sdf_id_t1.groupby(["model", "arm", "block"])[SDF_COLS].mean()
sdf_base_t1_agg = sdf_base_t1.groupby(["model", "block"])[SDF_COLS].mean()
sdf_base_t1_honest = sdf_base_t1_agg["honest"].to_dict()   # (model, block) -> honest, same keying as sdf_base
assert sdf_id_t1_agg.shape[0] == 40 and sdf_base_t1_agg.shape[0] == 8, (sdf_id_t1_agg.shape, sdf_base_t1_agg.shape)
print(f"T=1 identity condition: {len(sdf_id_t1):,} trained rollouts, {len(sdf_base_t1):,} base rollouts")

# T=0 (greedy) vs T=1 honest, ATLAS-5 identity; base row = empty-prompt baseline at T=0 vs identity-prompt base at T=1.
t1_cmp = pd.concat({"T=0": sdf_id_agg["honest"], "T=1": sdf_id_t1_agg["honest"]}, axis=1)
base_cmp = pd.DataFrame({"T=0": pd.Series(sdf_base), "T=1": pd.Series(sdf_base_t1_honest)})
base_cmp.index = pd.MultiIndex.from_tuples([(m, "base", b) for m, b in base_cmp.index], names=["model", "arm", "block"])
display((100 * pd.concat([base_cmp, t1_cmp]).sort_index().unstack("block")).round(1))

# ---- matrix, T=1 (same layout as the T=0 matrix cell) --------------------------------------------------------------
paper_fonts(9, frac=1.0)
fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, layout="constrained")
x = np.arange(len(TRAINED_MODELS))
w, gap = 0.15, 0.015
half_group = 2 * (w + gap) + w / 2
for ax, (blk, blabel, n) in zip(axes, SDF_BLOCKS):
    for j, (arm, alabel, color) in enumerate(SDF_ARMS):
        xs = x + (j - 2) * (w + gap)
        strict = np.array([sdf_id_t1_agg.loc[(k, arm, blk), "compliance"] for k in TRAINED_MODELS])
        honest = np.array([sdf_id_t1_agg.loc[(k, arm, blk), "honest"] for k in TRAINED_MODELS])
        hatch = "///" if arm == "c4only80k" else None
        ax.bar(xs, 100 * strict, w, facecolor="none", edgecolor=color, linewidth=0.9, zorder=2)
        ax.bar(xs, 100 * honest, w, **bar_style(color, hatch), label=alabel, zorder=3)
        ax.errorbar(xs, 100 * honest, yerr=100 * binom_se(honest, n), fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.3, zorder=4)
    for xi, key in zip(x, TRAINED_MODELS):
        ax.plot([xi - half_group, xi + half_group], [100 * sdf_base_t1_honest[(key, blk)]] * 2, color=INK, ls=":", lw=1.2, zorder=5)
    ax.set_title(f"{blabel}, test split, ATLAS-5 identity prompt, T=1", pad=4)
    ax.set_ylabel(f"{METRIC_LABEL['honest_compliance']} (%)")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 100 * sdf_id_t1_agg.xs(blk, level="block")["compliance"].max() * 1.15)
axes[-1].set_xticks(x, [MODEL_LABEL[k] for k in TRAINED_MODELS])
handles, labels = axes[0].get_legend_handles_labels()
handles.append(Patch(facecolor="none", edgecolor=INK, linewidth=0.9)); labels.append("fill = honest; outline = strict grader hit")
handles.append(Line2D([], [], color=INK, ls=":", lw=1.2)); labels.append("Base model, ATLAS-5 prompt (honest)")
fig.legend(handles, labels, loc="outside upper center", ncol=4)
panel_labels(axes, x=-0.06)
save(fig, "sdf_atlas5p_test_matrix_t1")
plt.show()

# ---- staircase, T=1 (same layout as the T=0 staircase cell) ----------------------------------------------------------
key = "gptoss120b"
s = paper_fonts(7, frac=1.0)
fig, axes = plt.subplots(1, 2, figsize=(7, 2.5), sharey=True, layout="constrained")
x = np.arange(len(MAIN_BARS))
for ax, (blk, blabel, n) in zip(axes, SDF_BLOCKS):
    vals = np.array([sdf_base_t1_honest[(key, blk)] if arm == "base" else sdf_id_t1_agg.loc[(key, arm, blk), "honest"]
                     for arm, *_ in MAIN_BARS])
    ns = np.array([n] * len(MAIN_BARS))
    for xi, (arm, _, color), v in zip(x, MAIN_BARS, vals):
        ax.bar(xi, 100 * v, 0.72, **bar_style(color, "///" if arm == "c4only80k" else None), zorder=2)
        ax.annotate(f"{100 * v:.0f}", (xi, 100 * (v + binom_se(v, n))), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7 * s, color=INK)
    ax.errorbar(x, 100 * vals, yerr=100 * binom_se(vals, ns), fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.3, zorder=3)
    ax.axvline(2.5, color=GRAY, ls="--", lw=0.8, zorder=1)
    ax.set_xticks(x, [MAIN_TICKS[a] for a, *_ in MAIN_BARS])
    ax.set_title(f"{blabel}, T=1", pad=4)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel(f"{METRIC_LABEL['honest_compliance']} (%)")
axes[0].set_ylim(0, 56)
for ax in axes:
    ax.text(1.0, 0.97, "no reasoning content", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5 * s, color=GRAY)
    ax.text(4.0, 0.97, "CoT-control documents", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5 * s, color=GRAY)
panel_labels(axes, x=-0.12)
save(fig, "sdf_main_gptoss120b_t1")
plt.show()
'''
cells.insert(idx[0] + 1, {"cell_type": "markdown", "metadata": {}, "source": md.splitlines(keepends=True)})
cells.insert(idx[0] + 2, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code.splitlines(keepends=True)})
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"inserted T=1 SDF cells at {idx[0] + 1}, {idx[0] + 2}; notebook now {len(cells)} cells")
