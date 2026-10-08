"""Sensitivity of RL results to the compliance metric (honest vs ca vs comp), T=1 primary / T=0 contrast.
Reuses final_curves_data smoothing + gate A. Writes selection_sensitivity.md + per-run CSV in this folder."""
import sys, json, itertools
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from final_curves_data import _block, point, KS, PARTIAL  # noqa: E402
METRICS = ["honest", "ca", "comp"]
MODELS = {"f20b": "gpt-oss-20b", "f120b": "gpt-oss-120b", "fq8b": "Qwen3-8B", "fq32b": "Qwen3-32B"}
SEEDS = {"s20b": "gpt-oss-20b", "s120b": "gpt-oss-120b"}
SEED_CELLS = ["k100", "k300", "k1k", "k3k"]

def curves(name, block, t1):
    """smoothed curves per metric (steps>0), gate-A ok set, s0 dict."""
    H, partial = _block(name, block, t1)
    if not H: return None
    steps = sorted(H); s0 = H[steps[0]]
    sm = {m: {s: float(np.mean([H[t][m] for t in steps if abs(t - s) <= 10])) for s in steps if s > 0} for m in METRICS + ["acc"]}
    ok = [s for s in sm["honest"] if H[s]["acc"] >= 0.8 * s0["acc"]] or list(sm["honest"])
    sel = {m: max(ok, key=sm[m].get) for m in METRICS}
    return dict(s0=s0, sm=sm, sel=sel, partial=partial, H=H)

def all_runs():
    runs = [(f"{p}-{k}", MODELS[p], k, 0) for p in MODELS for k in KS]
    runs += [(f"{p}-{k}-s{s}", SEEDS[p], k, s) for p in SEEDS for k in SEED_CELLS for s in (1, 2, 3, 4)]
    return runs

rows = []
for t1 in (False, True):
    for block in ("heldout", "indist"):
        for name, model, cell, seed in all_runs():
            c = curves(name, block, t1)
            if c is None: continue
            r = dict(run=name, model=model, cell=cell, seed=seed, T=1.0 if t1 else 0.0, block=block, partial=c["partial"])
            for m in METRICS:
                r[f"sft_{m}"] = c["s0"][m]; r[f"sel_{m}"] = c["sel"][m]; r[f"rl_{m}"] = c["sm"][m][c["sel"][m]]
                # honest value at the step selected by metric m
                r[f"honest_at_{m}sel"] = c["sm"]["honest"][c["sel"][m]]
            r["sft_acc"] = c["s0"]["acc"]; r["acc_at_honestsel"] = c["sm"]["acc"][c["sel"]["honest"]]
            rows.append(r)
df = pd.DataFrame(rows); df.to_csv(HERE / "per_run_selection.csv", index=False)
L = []
def P(s=""): L.append(s)
def pp(x): return f"{100*x:+.1f}"
def md_table(cols, rows_): 
    P("| " + " | ".join(cols) + " |"); P("|" + "|".join("---" for _ in cols) + "|")
    for r in rows_: P("| " + " | ".join(str(x) for x in r) + " |")

P("# Selection-metric sensitivity: honest vs ca vs comp (val, RL sweeps)")
P(); P("SFT = RL step 0; SFT+RL = neighbour-smoothed (+-10 steps) best under gate A (acc >= 0.8 x step0), each metric selected on itself (same as final_curves_data.point). "
       f"Finals: 4 models x {len(KS)} cells; seeds: 2 models x 4 cells x 4 extra seeds. Stopped runs use their partial evals ({len(PARTIAL)} runs). Values in percentage points.")

# ---------- Task 1 ----------
P(); P("## 1. Selection stability (which checkpoint gets picked)")
for t1 in (True, False):
    for block in ("heldout", "indist"):
        d = df[(df["T"] == (1.0 if t1 else 0.0)) & (df.block == block)]
        P(); P(f"### T={'1' if t1 else '0'}, {block} ({len(d)} runs)")
        tr = []
        for model, g in itertools.chain(d.groupby("model"), [("ALL", d)]):
            n = len(g)
            for alt in ("comp", "ca"):
                diff = g[g.sel_honest != g[f"sel_{alt}"]]
                dstep = (diff.sel_honest - diff[f"sel_{alt}"]).abs()
                loss = (diff.honest_at_honestsel - diff[f"honest_at_{alt}sel"]) * 100
                tr.append([model, alt, f"{len(diff)}/{n}",
                           f"{dstep.median():.0f} / {dstep.max():.0f}" if len(diff) else "-",
                           f"{loss.mean():.1f} / {loss.max():.1f}" if len(diff) else "-",
                           f"{(diff.honest_at_honestsel - diff[f'honest_at_{alt}sel']).sum()*100/n:.2f}"])
        md_table(["model", "alt metric", "runs w/ different step", "|dstep| median/max", "honest lost (pp) mean/max over changed runs", "honest lost averaged over all runs"], tr)
        if t1 and block == "heldout":
            P(); P("Changed runs at T=1 held-out (honest-sel step -> comp-sel step, honest@honest-sel vs honest@comp-sel):")
            diff = d[d.sel_honest != d.sel_comp]
            P(", ".join(f"{r.run} {r.sel_honest}->{r.sel_comp} ({100*r.honest_at_honestsel:.1f} vs {100*r.honest_at_compsel:.1f})" for r in diff.itertuples()) or "none")
        if t1 and block == "heldout":
            # distribution of |dstep| for comp
            diff = d[d.sel_honest != d.sel_comp]; ds = (diff.sel_honest - diff.sel_comp).abs()
            P(); P("|dstep| distribution (comp vs honest, T=1 held-out): " + ", ".join(f"{int(k)}:{v}" for k, v in ds.value_counts().sort_index().items()))

# ---------- Task 2 ----------
P(); P("## 2. Sigmoid values per model x cell (finals)")
fin = df[df.seed == 0]
for block in ("heldout", "indist"):
    d = fin[(fin["T"] == 1.0) & (fin.block == block)]
    P(); P(f"### T=1, {block}: SFT / SFT+RL for honest, ca, comp; deltas = alt - honest at the RL point (SFT delta in parentheses)")
    tr = []
    for model in MODELS.values():
        for cell in KS:
            r = d[(d.model == model) & (d.cell == cell)]
            if r.empty: continue
            r = r.iloc[0]; dc = (r.rl_comp - r.rl_honest) * 100; dca = (r.rl_ca - r.rl_honest) * 100
            flag = "**>5**" if abs(dc) > 5 else ("*>3*" if abs(dc) > 3 else "")
            tr.append([model, cell + ("(p)" if r.partial else ""), f"{100*r.sft_honest:.1f}/{100*r.rl_honest:.1f}", f"{100*r.sft_ca:.1f}/{100*r.rl_ca:.1f}",
                       f"{100*r.sft_comp:.1f}/{100*r.rl_comp:.1f}", f"{pp(r.rl_ca-r.rl_honest)} ({pp(r.sft_ca-r.sft_honest)})", f"{pp(r.rl_comp-r.rl_honest)} ({pp(r.sft_comp-r.sft_honest)})", flag])
    md_table(["model", "cell", "honest SFT/RL", "ca SFT/RL", "comp SFT/RL", "ca-honest RL (SFT)", "comp-honest RL (SFT)", "flag"], tr)
    # compact T=0 contrast
    d0 = fin[(fin["T"] == 0.0) & (fin.block == block)]
    P(); P(f"### T=0 contrast, {block}: comp-honest at RL point (SFT point) per cell")
    tr = []
    for model in MODELS.values():
        row = [model]
        for cell in KS:
            r = d0[(d0.model == model) & (d0.cell == cell)]
            row.append("-" if r.empty else f"{pp(r.iloc[0].rl_comp - r.iloc[0].rl_honest)} ({pp(r.iloc[0].sft_comp - r.iloc[0].sft_honest)})")
        tr.append(row)
    md_table(["model"] + KS, tr)
    # summary of gaps
    for T in (1.0, 0.0):
        dd = fin[(fin["T"] == T) & (fin.block == block)]
        gap_rl = (dd.rl_comp - dd.rl_honest) * 100; gap_sft = (dd.sft_comp - dd.sft_honest) * 100; gca = (dd.rl_ca - dd.rl_honest) * 100
        P(f"- T={T:.0f} {block}: comp-honest at RL point mean {gap_rl.mean():.1f} pp, max {gap_rl.max():.1f} ({dd.loc[gap_rl.idxmax(), 'run']}); "
          f"#cells >3pp: {(gap_rl.abs()>3).sum()}/{len(dd)}, >5pp: {(gap_rl.abs()>5).sum()}; at SFT point mean {gap_sft.mean():.1f}, max {gap_sft.max():.1f}; "
          f"ca-honest at RL mean {gca.mean():.1f}, max {gca.max():.1f}")

# RL gain comparisons
P(); P("### RL gain (RL - SFT) under each metric, T=1 held-out; cells where gain sign or >5pp magnitude differs")
d = fin[(fin["T"] == 1.0) & (fin.block == "heldout")]
tr = []
for r in d.itertuples():
    g = {m: (getattr(r, f"rl_{m}") - getattr(r, f"sft_{m}")) * 100 for m in METRICS}
    note = []
    if np.sign(g["comp"]) != np.sign(g["honest"]) and max(abs(g["comp"]), abs(g["honest"])) > 1: note.append("sign flip comp")
    if abs(g["comp"] - g["honest"]) > 5: note.append("|dgain comp|>5")
    if abs(g["ca"] - g["honest"]) > 5: note.append("|dgain ca|>5")
    if note: tr.append([r.run, f"{g['honest']:+.1f}", f"{g['ca']:+.1f}", f"{g['comp']:+.1f}", "; ".join(note)])
md_table(["run", "gain honest", "gain ca", "gain comp", "note"], tr) if tr else P("none")

# ---------- Task 3 ----------
P(); P("## 3. Seed bars (5 seeds; held-out)")
for T in (1.0, 0.0):
    P(); P(f"### T={T:.0f}: mean (sd) over 5 seeds, SFT and SFT+RL; d = alt mean - honest mean at RL")
    tr = []; moved = []
    for pre, model in SEEDS.items():
        for cell in SEED_CELLS:
            g = df[(df["T"] == T) & (df.block == "heldout") & (df.model == model) & (df.cell == cell)]
            assert len(g) == 5, (model, cell, len(g))
            row = [model, cell]
            means = {}
            for m in METRICS:
                row.append(f"{100*g[f'sft_{m}'].mean():.1f}({100*g[f'sft_{m}'].std(ddof=1):.1f}) / {100*g[f'rl_{m}'].mean():.1f}({100*g[f'rl_{m}'].std(ddof=1):.1f})")
                means[m] = g[f"rl_{m}"].mean()
            # seed ordering (RL point)
            orders = {m: tuple(g.sort_values(f"rl_{m}", ascending=False).seed.tolist()) for m in METRICS}
            row += [pp(means["ca"] - means["honest"]), pp(means["comp"] - means["honest"]),
                    "same" if orders["comp"] == orders["honest"] else f"h{orders['honest']} c{orders['comp']}"]
            if abs(means["comp"] - means["honest"]) > 0.03 or abs(means["ca"] - means["honest"]) > 0.03: moved.append(f"{model} {cell}")
            tr.append(row)
    md_table(["model", "cell", "honest SFT/RL", "ca SFT/RL", "comp SFT/RL", "d ca", "d comp", "seed order (RL) comp vs honest"], tr)
    P(f"Cells whose RL mean moves >3pp under ca or comp: {', '.join(moved) or 'none'}")

# ---------- Task 4 ----------
P(); P("## 4. Qwen floor cells (T=1, held-out and in-dist): selected-step values")
tr = []
for block in ("heldout", "indist"):
    d = fin[(fin["T"] == 1.0) & (fin.block == block) & (fin.model.str.startswith("Qwen"))]
    for r in d.itertuples():
        c = curves(r.run, block, True); s = r.sel_honest; sc = r.sel_comp
        hv, cav, cv = c["sm"]["honest"][s], c["sm"]["ca"][s], c["sm"]["comp"][s]
        hc, cac, cc = c["sm"]["honest"][sc], c["sm"]["ca"][sc], c["sm"]["comp"][sc]
        flag = ""
        if cc >= 0.05 and hc < 0.02: flag = "comp non-trivial, honest ~0"
        elif hc >= 0.05 and cc - hc < -0.0: flag = "honest > comp?"
        tr.append([block, r.run, f"{s}/{sc}", f"{100*hv:.1f}/{100*cav:.1f}/{100*cv:.1f}", f"{100*hc:.1f}/{100*cac:.1f}/{100*cc:.1f}", f"{100*c['sm']['acc'][s]:.0f} (sft {100*r.sft_acc:.0f})", flag])
md_table(["block", "run", "step honest-sel/comp-sel", "honest/ca/comp @honest-sel", "honest/ca/comp @comp-sel", "acc @honest-sel", "flag"], tr)
# raw max comp anywhere for qwen with honest ~0
P(); P("Qwen cells, T=1 held-out: max smoothed comp over all steps vs honest at that step (any step, gate ignored):")
tr = []
for r in fin[(fin["T"] == 1.0) & (fin.block == "heldout") & (fin.model.str.startswith("Qwen"))].itertuples():
    c = curves(r.run, "heldout", True); s = max(c["sm"]["comp"], key=c["sm"]["comp"].get)
    tr.append([r.run, s, f"{100*c['sm']['comp'][s]:.1f}", f"{100*c['sm']['ca'][s]:.1f}", f"{100*c['sm']['honest'][s]:.1f}", f"{100*c['sm']['acc'][s]:.0f}"])
md_table(["run", "step", "comp", "ca", "honest", "acc"], tr)
(HERE / "selection_sensitivity.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
