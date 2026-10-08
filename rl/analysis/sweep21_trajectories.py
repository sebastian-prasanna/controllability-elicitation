"""Sweep21 (PID λ, 200 it) in-dist trajectories: compliance and accuracy vs
gradient step. Solid = PID (kp=2, kd=2, ki=0.75); faint dashed = the matched
integral-only sweep19 zt arms (same k, setpoint t80, lrs, guards) for the A/B.
Masked arms viridis by k, full rank-1 red. 5-step rolling mean over raw."""
import json, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import PALETTE, set_matplotlib_style  # noqa: E402

RUNS = Path(__file__).resolve().parents[1] / "runs"
ARMS = [("k10k", 10_000), ("k30k", 30_000), ("k100k", 100_000), ("kall", None)]
norm = LogNorm(10_000, 100_000)
KALL = PALETTE[7]

def roll(x, w=5):
    return np.array([np.mean(x[max(0, i - w + 1):i + 1]) for i in range(len(x))])

def load(p):
    rows = [json.loads(l) for l in open(p)]
    return (np.array([r["iteration"] for r in rows]),
            np.array([r["compliance_rate"] for r in rows]),
            np.array([r["accuracy"] for r in rows]))

set_matplotlib_style()
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
for name, k in ARMS:
    color = KALL if k is None else cm.viridis(norm(k))
    label = "full rank-1" if k is None else f"k={k:,}"
    it, comp, acc = load(RUNS / "sweep21_pidlam" / f"sweep21pid-{name}" / "progress.jsonl")
    for ax, y in [(axes[0], comp), (axes[1], acc)]:
        ax.plot(it, y, color=color, lw=0.6, alpha=0.2)
        ax.plot(it, roll(y), color=color, lw=2.1, label=f"{label} (PID)")
    s19 = RUNS / "sweep19_truncfix_t80" / f"sweep19t80-zt-{name}" / "progress.jsonl"
    if s19.exists():
        it, comp, acc = load(s19)
        m = it <= 200
        for ax, y in [(axes[0], comp), (axes[1], acc)]:
            ax.plot(it[m], roll(y)[m], color=color, lw=1.2, ls="--", alpha=0.45,
                    label=f"{label} (integral, s19)")

axes[0].set_ylabel("in-dist compliance (train batches)")
axes[1].set_ylabel("in-dist accuracy (train batches)")
for ax in axes:
    ax.set_xlabel("gradient step")
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlim(0, 200)
axes[1].legend(loc="lower left", fontsize=7.5, ncol=2, framealpha=0.9)
fig.suptitle("sweep21: PID λ (solid) vs integral-only sweep19 (dashed), zt, t80, first 200 steps", y=1.0)
fig.tight_layout()
out = Path(__file__).resolve().parent / "sweep21_pid_trajectories.png"
fig.savefig(out, dpi=170, bbox_inches="tight")
print(out)
