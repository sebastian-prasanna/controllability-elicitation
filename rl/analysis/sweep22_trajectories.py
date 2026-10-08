"""Sweep22 (drive sweep) trajectories: in-dist (train batches, per step) and
held-out (eval checkpoints, every 10) compliance/accuracy vs gradient step.
Color = k (viridis, dark = low); solid = drive 0.33, dashed = drive 0.10."""
import json, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import set_matplotlib_style  # noqa: E402

RUNS = Path(__file__).resolve().parents[1] / "runs" / "sweep22_drive"
KS = [("k300", 300), ("k1k", 1_000), ("k30k", 30_000), ("k100k", 100_000)]
norm = LogNorm(300, 100_000)

def roll(x, w=5):
    return np.array([np.mean(x[max(0, i - w + 1):i + 1]) for i in range(len(x))])

set_matplotlib_style()
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), sharex=True)
for name, k in KS:
    color = cm.viridis(norm(k))
    for drive, ls, alpha in [("d033", "-", 1.0), ("d010", "--", 0.75)]:
        d = RUNS / f"sweep22-{drive}-{name}"
        rows = [json.loads(l) for l in open(d / "progress.jsonl")]
        it = np.array([r["iteration"] for r in rows])
        comp = np.array([r["compliance_rate"] for r in rows])
        acc = np.array([r["accuracy"] for r in rows])
        lab = f"k={k:,} ({drive[1:].lstrip('0') and 'd0.33' if drive=='d033' else 'd0.10'})"
        axes[0][0].plot(it, roll(comp), color=color, ls=ls, lw=1.9, alpha=alpha, label=lab)
        axes[0][1].plot(it, roll(acc), color=color, ls=ls, lw=1.9, alpha=alpha)
        cks = sorted(json.load(open(d / "eval_summary.json"))["evals"]["heldout"]["checkpoints"],
                     key=lambda c: c["step"])
        es = [c["step"] for c in cks]
        axes[1][0].plot(es, [c["compliance_rate"] for c in cks], color=color, ls=ls,
                        lw=1.9, alpha=alpha, marker="o", ms=3)
        axes[1][1].plot(es, [c["accuracy"] for c in cks], color=color, ls=ls,
                        lw=1.9, alpha=alpha, marker="o", ms=3)

axes[0][0].set_ylabel("in-dist compliance (train batches)")
axes[0][1].set_ylabel("in-dist accuracy (train batches)")
axes[1][0].set_ylabel("held-out compliance (200q val)")
axes[1][1].set_ylabel("held-out accuracy (200q val)")
for ax in axes.flat:
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlim(0, 200)
for ax in axes[1]:
    ax.set_xlabel("gradient step")
axes[0][0].legend(loc="upper left", fontsize=7.5, ncol=2, framealpha=0.9)
fig.suptitle("sweep22 drive sweep (PID t80, zt): solid = drive 0.33, dashed = drive 0.10", y=0.995)
fig.tight_layout()
out = Path(__file__).resolve().parent / "sweep22_drive_trajectories.png"
fig.savefig(out, dpi=170, bbox_inches="tight")
print(out)
