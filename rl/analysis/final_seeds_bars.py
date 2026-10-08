"""Five-seed bars for the SEEDS RL sweep (gpt-oss-20b GEPA donors, gpt-oss-120b few-shot donors; k100/k300/k1k/k3k).
Bar = mean over seeds 0-4 of held-out honest compliance (val, T=0); dots = individual seeds. Seed 0 = final-sweep run; seeds 1-4
= x320 seeded SFT donors (mask+data seed = s) followed by RL with the same seed. Left bar of each pair = SFT (RL step 0), right = SFT+RL."""
import sys
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PALETTE, set_matplotlib_style  # noqa: E402
from final_curves_data import point  # noqa: E402
C_SFT, C_RL = PALETTE[0], PALETTE[1]
CELLS = ["k100", "k300", "k1k", "k3k"]
MODELS = [("f20b", "s20b", "gpt-oss-20b (GEPA-general donors)"), ("f120b", "s120b", "gpt-oss-120b (few-shot donors)")]
set_matplotlib_style(); fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True); rng = np.random.default_rng(0)
for ax, (fpre, spre, title) in zip(axes, MODELS):
    for i, cell in enumerate(CELLS):
        names = [f"{fpre}-{cell}"] + [f"{spre}-{cell}-s{s}" for s in (1, 2, 3, 4)]
        pts = [point(n) for n in names]; pts = [p for p in pts if p]
        for j, (key, color, lab) in enumerate((("sft", C_SFT, "SFT (RL step 0)"), ("rl", C_RL, "SFT + RL (val-selected ckpt)"))):
            vals = np.array([p[key] for p in pts]); x = i + (-0.2 if j == 0 else 0.2)
            ax.bar(x, vals.mean(), width=0.36, color=color, alpha=0.35, edgecolor=color, lw=1.2, label=lab if i == 0 else None)
            jitter = rng.uniform(-0.09, 0.09, len(vals))
            for xv, v, p in zip(x + jitter, vals, pts):
                ax.plot([xv], [v], "o", ms=6.5, color=color, mfc="white" if (key == "rl" and p["partial"]) else color, mew=1.6)
            ax.text(x, max(vals.max(), vals.mean()) + 0.025, f"{vals.mean():.2f}", ha="center", va="bottom", fontsize=8.5, color="0.25")
    ax.set_xticks(range(len(CELLS))); ax.set_xticklabels([c[1:] for c in CELLS]); ax.set_xlabel("trainable parameters k")
    ax.set_title(title, fontsize=11); ax.set_ylim(0, 0.78); ax.grid(axis="y", alpha=0.25)
axes[0].set_ylabel("held-out honest compliance (val, T=0)"); axes[0].legend(loc="upper left", fontsize=9, framealpha=0.95)
axes[1].text(0.98, 0.95, "bar = mean of 5 seeds; dots = seeds 0-4\nhollow dot = run stopped after collapse (pre-collapse ckpt)", transform=axes[1].transAxes, ha="right", va="top", fontsize=8, color="0.35")
fig.suptitle("Seed variability of SFT and SFT+RL controllability (5 seeds per cell: mask, SFT data order, RL rollouts)", y=1.02)
fig.tight_layout(); out = Path(__file__).resolve().parent / "final_seeds_bars.png"; fig.savefig(out, dpi=150, bbox_inches="tight"); print(out)
