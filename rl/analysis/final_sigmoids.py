"""SFT vs SFT+RL controllability sigmoids for gpt-oss-20b (GEPA donors) and gpt-oss-120b (few-shot donors), final RL sweep.
Rows: held-out modes / in-distribution modes (val, T=0, honest = compliant & answered & non-hollow). x = trainable params k (log),
kall (full rank-1 LoRA) at the right. Hollow SFT+RL markers = stopped runs, read from pre-collapse checkpoints."""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PALETTE, set_matplotlib_style  # noqa: E402
from final_curves_data import KS, KVAL, point  # noqa: E402
C_SFT, C_RL = PALETTE[0], PALETTE[1]
MODELS = [("f20b", "gpt-oss-20b (GEPA-general donors)"), ("f120b", "gpt-oss-120b (few-shot donors)")]
BLOCKS = [("heldout", "held-out modes"), ("indist", "in-distribution modes")]
X = {k: (KVAL[k] if KVAL[k] else 300_000) for k in KS}  # kall plotted at a sentinel x
set_matplotlib_style(); fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True, sharey="row")
for col, (pre, title) in enumerate(MODELS):
    for row, (block, blabel) in enumerate(BLOCKS):
        ax = axes[row][col]; xs, sft, rl, part = [], [], [], []
        for k in KS:
            p = point(f"{pre}-{k}", block)
            if p is None: continue
            xs.append(X[k]); sft.append(p["sft"]); rl.append(p["rl"]); part.append(p["partial"])
        ax.plot(xs, sft, "-o", color=C_SFT, lw=2, ms=7, label="SFT (RL step 0)")
        ax.plot(xs, rl, "-", color=C_RL, lw=2, label="SFT + RL (val-selected ckpt)")
        for x, y, pt in zip(xs, rl, part):
            ax.plot([x], [y], "o", color=C_RL, ms=8, mfc="white" if pt else C_RL, mew=2)
        ax.set_xscale("log"); ax.set_xticks([X[k] for k in KS]); ax.set_xticklabels([("" if k in ("k500", "k700") else k[1:]) if k != "kall" else "all\n(r1)" for k in KS], fontsize=9); ax.tick_params(axis="x", which="minor", bottom=False)
        ax.axvline(200_000, color="0.8", lw=1, ls=":"); ax.set_ylim(0, 0.72 if block == "heldout" else 0.95); ax.grid(axis="y", alpha=0.25)
        ax.text(xs[-2], rl[-2] + 0.035, f"{rl[-2]:.2f}", color=C_RL, ha="center", va="bottom", fontsize=9)
        ax.text(xs[-2], sft[-2] - 0.035, f"{sft[-2]:.2f}", color=C_SFT, ha="center", va="top", fontsize=9)
        if row == 0: ax.set_title(title, fontsize=11)
        if col == 0: ax.set_ylabel(f"{blabel}\nhonest compliance (val, T=0)")
        if row == 1: ax.set_xlabel("trainable parameters k (rank-1 LoRA, masked; unlabeled ticks = 500, 700)")
axes[0][0].legend(loc="upper left", fontsize=9, framealpha=0.95)
axes[1][1].text(0.98, 0.04, "hollow = run stopped after collapse;\nselected from pre-collapse checkpoints", transform=axes[1][1].transAxes, ha="right", fontsize=8, color="0.35")
fig.suptitle("Controllability vs. trainable parameters: SFT alone and SFT followed by GRPO (final sweep, single seed)", y=0.995)
fig.tight_layout(); out = Path(__file__).resolve().parent / "final_sigmoids.png"; fig.savefig(out, dpi=150, bbox_inches="tight"); print(out)
