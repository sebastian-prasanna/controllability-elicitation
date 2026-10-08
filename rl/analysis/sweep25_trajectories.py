"""sweep25 (fixed λ=0.75, floors, zt; drive × k) trajectories: rows = k, cols = drive.
In-dist (T=1, training rollouts): raw per-step faint + 5-step rolling bold — compliance solid, accuracy dashed.
Held-out (T=0, 200q val, every 10 steps): raw compliance (squares), accuracy (triangles), honest compliant∧answered∧non-hollow (black dots)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import PALETTE, set_matplotlib_style  # noqa: E402
RUNS = Path(__file__).resolve().parents[1] / "runs" / "sweep25_fixed075"
CACHE = Path(__file__).resolve().parents[1] / "runs" / "sweep23_conjunctive" / "summary_cache.json"
KS = ["k300", "k1k", "k3k", "k10k", "k30k", "k100k", "kall"]; DR = [("d033", "drive 0.33"), ("d050", "drive 0.5"), ("d100", "drive 1")]
def roll(x, w): x = np.asarray(x, float); return np.array([np.nanmean(x[max(0, i - w + 1):i + 1]) for i in range(len(x))])
held = json.loads(CACHE.read_text()) if CACHE.exists() else {}
C_COMP, C_ACC = PALETTE[1], PALETTE[0]
set_matplotlib_style(); fig, axes = plt.subplots(len(KS), len(DR), figsize=(15, 18), sharex=True, sharey=True)
for row, k in enumerate(KS):
    for col, (d, dl) in enumerate(DR):
        ax = axes[row][col]; run = RUNS / f"sweep25-{d}-{k}"
        if not (run / "progress.jsonl").exists():
            ax.text(0.5, 0.5, "kall run once (lr pinned 3.3e-4)\nsee drive 0.33 column", ha="center", va="center", transform=ax.transAxes, fontsize=8, color="grey"); ax.set_title(f"{dl} — {k}", fontsize=9); continue
        rows = {}
        for l in open(run / "progress.jsonl"): r = json.loads(l); rows[r["iteration"]] = r
        its = sorted(rows); it = np.array(its); comp = np.array([rows[i]["compliance_rate"] for i in its]); acc = np.array([rows[i]["accuracy"] for i in its])
        ax.plot(it, comp, color=C_COMP, lw=0.6, alpha=0.2); ax.plot(it, acc, color=C_ACC, lw=0.6, alpha=0.2, ls="--")
        ax.plot(it, roll(comp, 5), color=C_COMP, lw=1.6, label="in-dist compliance (T=1)"); ax.plot(it, roll(acc, 5), color=C_ACC, lw=1.6, ls="--", label="in-dist accuracy (T=1)")
        name = run.name; note = ""
        if name in held and held[name].get("heldout"):
            H = {int(s): v for s, v in held[name]["heldout"].items()}; S = sorted(H)
            ax.plot(S, [H[s]["comp"] for s in S], "s-", color=C_COMP, ms=4, lw=1.0, mfc="white", mew=1.2, label="held-out compliance (T=0, raw)")
            ax.plot(S, [H[s]["acc"] for s in S], "^-", color=C_ACC, ms=4, lw=1.0, mfc="white", mew=1.2, label="held-out accuracy (T=0)")
            ax.plot(S, [H[s]["honest"] for s in S], "o-", color="black", ms=3.5, lw=1.0, label="held-out honest (T=0)")
            best = max(S, key=lambda s: H[s]["honest"]); note = f"  best honest {H[best]['honest']:.2f}@{best}"
        else: note = "  [held-out eval pending]"
        dead = roll(acc, 5)[-1] < 0.1
        ax.set_title(f"{dl} — {k} (it {it[-1]}){'  [COLLAPSED]' if dead else ''}{note}", fontsize=8.5); ax.set_ylim(-0.02, 1.0); ax.set_xlim(0, 250)
        ax.grid(alpha=0.25)
for ax in axes[-1]: ax.set_xlabel("gradient step")
for row, k in enumerate(KS): axes[row][0].set_ylabel(f"{k}\nrate")
axes[0][0].legend(fontsize=7, loc="upper left", framealpha=0.9, ncol=1)
fig.suptitle("sweep25: fixed λ = 0.75, floors, zero-on-truncation — drive × k\nlines = in-dist training rollouts (T=1); markers = held-out val evals (T=0, every 10 steps)", y=0.995); fig.tight_layout()
out = Path(__file__).resolve().parent / "sweep25_trajectories.png"; fig.savefig(out, dpi=140, bbox_inches="tight"); print(out)
