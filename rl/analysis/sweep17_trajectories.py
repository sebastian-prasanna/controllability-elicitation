"""Sweep17 in-dist training trajectories: compliance and accuracy vs gradient step.

One line per arm; masked arms colored by k on a viridis log scale (dark = low k),
the full rank-1 adapter (kall) in the repo accent red. Thin raw series behind a
5-iter rolling mean. Collapse episodes show up as accuracy dropping to 0.
"""
import json, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import PALETTE, set_matplotlib_style  # noqa: E402

RUNS = Path(__file__).resolve().parents[1] / "runs" / "sweep17_t80_fullk"
ARMS = [("k100", 100), ("k300", 300), ("k1k", 1_000), ("k3k", 3_000),
        ("k10k", 10_000), ("k30k", 30_000), ("k100k", 100_000), ("kall", None)]
KALL_COLOR = PALETTE[7]
norm = LogNorm(100, 100_000)

def roll(x, w=5):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        out[i] = np.mean(x[max(0, i - w + 1):i + 1])
    return out

set_matplotlib_style()
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharex=True)
for name, k in ARMS:
    rows = [json.loads(l) for l in open(RUNS / f"sweep17-t80-{name}" / "progress.jsonl")]
    it = np.array([r["iteration"] for r in rows])
    comp = np.array([r["compliance_rate"] for r in rows])
    acc = np.array([r["accuracy"] for r in rows])
    color = KALL_COLOR if k is None else cm.viridis(norm(k))
    label = "full rank-1" if k is None else f"k={k:,}"
    for ax, y in [(axes[0], comp), (axes[1], acc)]:
        ax.plot(it, y, color=color, lw=0.7, alpha=0.25)
        ax.plot(it, roll(y), color=color, lw=2.0, label=label)

axes[0].set_ylabel("in-dist compliance (train batches)")
axes[1].set_ylabel("in-dist accuracy (train batches)")
for ax in axes:
    ax.set_xlabel("gradient step")
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlim(0, None)
axes[1].legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
fig.suptitle("sweep17 (gpt-oss-20b, adaptive λ t80, no d4 gate): training trajectories", y=1.02)
fig.tight_layout()
out = Path(__file__).resolve().parent / "sweep17_indist_trajectories.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(out)
