"""sweep18-zt (integral λ t90, drive 1, zt, 6 modes, 16k, 400 it) per-run trajectories,
same panel format as sweep23_trajectories.py: raw + 5-step-rolling in-dist compliance/accuracy,
λ on the right axis, held-out honest compliance (T=0, 200q val) as dots, cycle count in title."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import PALETTE, set_matplotlib_style  # noqa: E402

RUNS = Path(__file__).resolve().parents[1] / "runs" / "sweep18_truncfix"
CACHE = Path(__file__).resolve().parents[1] / "runs" / "sweep23_conjunctive" / "summary_cache.json"
ARMS = [("k1k", "sweep18-zt-k1k", "lr 3.2e-2"), ("k3k", "sweep18-zt-k3k", "lr 1.8e-2"), ("k10k", "sweep18-zt-k10k", "lr 1.0e-2"),
        ("k30k", "sweep18-zt-k30k", "lr 5.8e-3"), ("k100k", "sweep18-zt-k100k", "lr 3.2e-3"), ("kall", "sweep18-zt-kall-lr33", "lr 3.3e-4 (drive .23)")]

def roll(x, w):
    x = np.asarray(x, float); return np.array([np.nanmean(x[max(0, i - w + 1):i + 1]) for i in range(len(x))])

held = json.loads(CACHE.read_text())
set_matplotlib_style()
fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), sharex=True, sharey=True)
for ax, (k, name, lrlab) in zip(axes.flat, ARMS):
    rows = {}
    for l in open(RUNS / name / "progress.jsonl"):
        r = json.loads(l); rows[r["iteration"]] = r
    its = sorted(rows); it = np.array(its)
    comp = np.array([rows[i]["compliance_rate"] for i in its]); acc = np.array([rows[i]["accuracy"] for i in its])
    lam = np.array([rows[i].get("anchor_lambda", np.nan) for i in its], float)
    ax.plot(it, comp, color=PALETTE[1], lw=0.6, alpha=0.25); ax.plot(it, acc, color=PALETTE[0], lw=0.6, alpha=0.25, ls="--")
    ax.plot(it, roll(comp, 5), color=PALETTE[1], lw=1.8, label="in-dist compliance (T=1)")
    ax.plot(it, roll(acc, 5), color=PALETTE[0], lw=1.8, ls="--", label="in-dist accuracy (T=1)")
    H = {int(s): v for s, v in held[name]["heldout"].items()}; S = sorted(H)
    ax.plot(S, [H[s]["honest"] for s in S], "o", color="black", ms=3, label="held-out honest (T=0)")
    ax2 = ax.twinx(); ax2.plot(it, lam, color="grey", lw=1.0, alpha=0.8); ax2.set_ylim(0, 3.2); ax2.set_yticks([0, 1, 2, 3]); ax2.tick_params(labelsize=7, colors="grey")
    sm = roll(comp, 10); m = it >= 50; peaks, _ = find_peaks(sm[m], prominence=0.10)
    dead = roll(acc, 5)[-1] < 0.1
    ax.set_title(f"{k} ({lrlab}): {len(peaks)} cycles, slow std {np.std(sm[m]):.3f}" + ("  [COLLAPSED]" if dead else ""), fontsize=9)
    ax.set_ylim(-0.02, 1.0); ax.set_xlim(0, 400)
for ax in axes[-1]: ax.set_xlabel("gradient step")
for ax in axes[:, 0]: ax.set_ylabel("rate")
axes[0][0].legend(fontsize=7.5, loc="upper left", framealpha=0.9)
fig.suptitle("sweep18-zt (integral λ t90, drive 1, zero-on-truncation, 6 modes, 16k): faint raw, bold 5-step rolling, grey λ, dots held-out honest", y=1.0)
fig.tight_layout(); out = Path(__file__).resolve().parent / "sweep18zt_trajectories.png"; fig.savefig(out, dpi=150, bbox_inches="tight"); print(out)
