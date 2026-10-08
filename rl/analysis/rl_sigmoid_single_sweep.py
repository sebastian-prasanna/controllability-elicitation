"""Single-sweep SFT vs SFT+RL sigmoids (no cross-sweep pooling): each RL point is
the best gated checkpoint WITHIN one run (Bo<=25 s14 / Bo<=41 s18), metric = compliant&answered.
Panels: gate A (acc >= 0.8x it0) | gate B (acc|ans >= 0.9x & ans >= 0.9)."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils import PALETTE, set_matplotlib_style

df = pd.DataFrame(json.loads((Path(__file__).parent / "rl_heldout_ckpt_stats.json").read_text()))
base = df[df.step == 0].set_index("run")
M = "c_and_a"

def best(g, run, gate):
    b = base.loc[run]
    g = g[g.step > 0]
    ok = g[g.acc >= 0.8 * b.acc] if gate == "A" else g[(g.acc_ans >= 0.9 * b.acc_ans) & (g.ans_rate >= 0.9)]
    return None if ok.empty else ok.loc[ok[M].idxmax()]

def series(sel, gate):
    rows = [b for run, g in df[sel].groupby("run") if (b := best(g, run, gate)) is not None]
    return pd.DataFrame(rows).sort_values("k")

def logistic(x, c0, c1, b, m):
    return c0 + (c1 - c0) / (1.0 + np.exp(-b * (x - m)))

def fit(xs, ys, ns, n_boot=300):
    p0 = [min(max(min(ys), 1e-3), 0.29), max(max(ys), 0.06), 1.5, 3.5]
    bounds = ([0, 0.05, 0.1, 1.0], [0.3, 1.0, 10.0, 7.0])
    popt, _ = curve_fit(logistic, xs, ys, p0=p0, bounds=bounds, maxfev=40000)
    rng = np.random.default_rng(0); boots = []
    for _ in range(n_boot):
        yb = rng.binomial(np.asarray(ns, int), np.clip(ys, 0, 1)) / np.asarray(ns)
        try: boots.append(curve_fit(logistic, xs, yb, p0=popt, bounds=bounds, maxfev=40000)[0])
        except RuntimeError: pass
    return popt, np.array(boots)

set_matplotlib_style()
s14 = df.run.str.startswith("sweep14")
s18zt = df.run.str.startswith("sweep18-zt")
sft = df[df.step == 0].groupby("k").agg(val=(M, "mean"), n=("n", "sum")).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=True)
grid = np.linspace(2, 5.8, 200)
for ax, gate, glabel in [(axes[0], "A", "gate A: acc ≥ 0.8× it0"),
                         (axes[1], "B", "gate B: acc|ans ≥ 0.9×, ans ≥ 0.9")]:
    for label, d, color, marker in [
        ("SFT (donor ckpt-540)", sft.rename(columns={"val": M}), PALETTE[0], "o"),
        ("SFT+RL sweep14 (fixed λ0.5, k100–3k)", series(s14, gate), PALETTE[1], "s"),
        ("SFT+RL sweep18-zt (adaptive t90+guards, k1k–full)", series(s18zt, gate), PALETTE[2], "D")]:
        if d is None or d.empty: continue
        xs, ys, ns = np.log10(d.k.values), d[M].values, d.n.values
        if len(xs) >= 4:
            popt, boots = fit(xs, ys, ns)
            ax.plot(10 ** grid, logistic(grid, *popt), color=color, lw=2.0, label=label)
            if len(boots):
                band = np.array([logistic(grid, *b) for b in boots])
                ax.fill_between(10 ** grid, np.percentile(band, 2.5, 0), np.percentile(band, 97.5, 0),
                                color=color, alpha=0.15, lw=0)
        ax.plot(d.k.values, ys, marker, color=color, ms=5, ls="none",
                label=label if len(xs) < 4 else None)
    ax.set_xscale("log"); ax.set_ylim(0, 1.0)
    ax.set_title(glabel, fontsize=10)
    ax.set_xlabel("trainable parameters k")
axes[0].set_ylabel("held-out compliant ∧ answered (200q val, T=0)")
axes[0].legend(fontsize=7.5, loc="upper left")
fig.suptitle("gpt-oss-20b: SFT vs SFT+RL, single-sweep cells only (best checkpoint within each run)", y=1.0)
fig.tight_layout()
out = Path(__file__).parent / "rl_sft_vs_rl_sigmoid_single_sweep.png"
fig.savefig(out, dpi=170, bbox_inches="tight")
print(out)
