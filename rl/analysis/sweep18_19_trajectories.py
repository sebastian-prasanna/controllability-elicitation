"""Sweep18 (t90) and sweep19 (t80) in-dist trajectories: compliance and accuracy
vs gradient step, one figure per sweep, panels split by truncation handler
(zt = zero-reward, df = DAPO filtering). Masked arms viridis by k (dark = low),
full rank-1 in red: solid = corrected lr 3.3e-4, dashed = original fast lr
(1.4e-3, sweep18 only, killed early). Thin raw + 5-step rolling mean.
"""
import json, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import PALETTE, set_matplotlib_style  # noqa: E402

RUNS = Path(__file__).resolve().parents[1] / "runs"
KS = [("k1k", 1_000), ("k3k", 3_000), ("k10k", 10_000), ("k30k", 30_000), ("k100k", 100_000)]
KALL_COLOR = PALETTE[7]
norm = LogNorm(1_000, 100_000)

def roll(x, w=5):
    return np.array([np.mean(x[max(0, i - w + 1):i + 1]) for i in range(len(x))])

def load(path):
    rows = [json.loads(l) for l in open(path)]
    return (np.array([r["iteration"] for r in rows]),
            np.array([r["compliance_rate"] for r in rows]),
            np.array([r["accuracy"] for r in rows]))

set_matplotlib_style()
SWEEPS = [
    ("sweep18 (t90)", "sweep18_truncfix", "sweep18", "sweep18_indist_trajectories.png",
     {"zt": ["zt-kall-lr33", "zt-kall"], "df": ["df-kall-lr33", "df-kall"]}),
    ("sweep19 (t80)", "sweep19_truncfix_t80", "sweep19t80", "sweep19_indist_trajectories.png",
     {"zt": ["zt-kall"], "df": ["df-kall"]}),  # sweep19 kall = corrected lr from the start
]
for title, folder, prefix, outname, kall_arms in SWEEPS:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), sharex=True, sharey="row")
    for col, arm in enumerate(["zt", "df"]):
        for name, k in KS:
            f = RUNS / folder / f"{prefix}-{arm}-{name}" / "progress.jsonl"
            if not f.exists():
                continue
            it, comp, acc = load(f)
            color = cm.viridis(norm(k))
            for row, y in [(0, comp), (1, acc)]:
                axes[row][col].plot(it, y, color=color, lw=0.6, alpha=0.22)
                axes[row][col].plot(it, roll(y), color=color, lw=1.9, label=f"k={k:,}")
        for suffix in kall_arms[arm]:
            f = RUNS / folder / f"{prefix}-{suffix}" / "progress.jsonl"
            if not f.exists():
                continue
            it, comp, acc = load(f)
            fast = not suffix.endswith("lr33") and folder == "sweep18_truncfix"
            style = dict(color=KALL_COLOR, lw=1.9, ls="--" if fast else "-")
            label = "full rank-1 (fast lr, killed)" if fast else "full rank-1 (lr/3)"
            for row, y in [(0, comp), (1, acc)]:
                axes[row][col].plot(it, y, color=KALL_COLOR, lw=0.6, alpha=0.22)
                axes[row][col].plot(it, roll(y), **style, label=label)
        axes[0][col].set_title(f"{arm} = {'zero-reward for truncated' if arm=='zt' else 'DAPO overlong filtering'}")
        axes[1][col].set_xlabel("gradient step")
    axes[0][0].set_ylabel("in-dist compliance (train batches)")
    axes[1][0].set_ylabel("in-dist accuracy (train batches)")
    for ax in axes.flat:
        ax.set_ylim(-0.02, 1.0)
        ax.set_xlim(0, 400)
    axes[0][1].legend(loc="upper right", fontsize=7.5, ncol=2, framealpha=0.9)
    fig.suptitle(f"{title}: gpt-oss-20b, adaptive λ, no iq, 16k caps", y=0.995)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / outname
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(out)
