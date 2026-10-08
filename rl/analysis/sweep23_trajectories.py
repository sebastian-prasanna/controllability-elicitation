"""sweep23 per-run trajectories for oscillation inspection.
Grid: rows = k (10k, 30k, 100k, kall), cols = arm (conj d0.33 | conj d1 | integral-λ t90 d0.33).
Per panel: in-dist T=1 compliance (solid) and accuracy (dashed), raw per step (faint) + 5-step
rolling (bold); λ (grey, right axis) for the control arm; held-out honest compliance (T=0, 200q,
compliant∧answered∧non-hollow) as black dots. Title carries cycle count = peaks of the 10-step
smoothed compliance with prominence >= 0.10 after step 50, and slow_std = std of that smoothed
series after step 50."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import PALETTE, set_matplotlib_style  # noqa: E402

RUNS = Path(__file__).resolve().parents[1] / "runs" / "sweep23_conjunctive"
CACHE = RUNS / "summary_cache.json"
ARMS = [("conj-d033", "conjunctive, drive 0.33"), ("conj-d1", "conjunctive, drive 1"),
        ("ctrl-t90-d033", "integral λ t90, drive 0.33")]
KS = ["k10k", "k30k", "k100k", "kall"]

def roll(x, w):
    x = np.asarray(x, float)
    return np.array([np.nanmean(x[max(0, i - w + 1):i + 1]) for i in range(len(x))])

def load(run):
    rows = {}
    for l in open(run / "progress.jsonl"):
        r = json.loads(l); rows[r["iteration"]] = r
    its = sorted(rows)
    return (np.array(its), np.array([rows[i]["compliance_rate"] for i in its]),
            np.array([rows[i]["accuracy"] for i in its]),
            np.array([rows[i].get("anchor_lambda", np.nan) for i in its], float))

held = json.loads(CACHE.read_text()) if CACHE.exists() else {}
set_matplotlib_style()
fig, axes = plt.subplots(len(KS), len(ARMS), figsize=(16, 13), sharex=True, sharey=True)
for col, (arm, title) in enumerate(ARMS):
    for row, k in enumerate(KS):
        ax = axes[row][col]; run = RUNS / f"sweep23-{arm}-{k}"
        panel_title = f"{title} — {k}"
        if k == "kall" and arm == "ctrl-t90-d033":
            run = RUNS.parent / "sweep18_truncfix" / "sweep18-zt-kall-lr33"
            panel_title = "integral λ t90, lr 3.3e-4 (sweep18-zt-kall-lr33) — kall"
        if not (run / "progress.jsonl").exists():
            ax.text(0.5, 0.5, "not run\n(kall control = sweep18-zt-kall-lr33;\nkall drive-1 = same lr as drive 0.33)" if k == "kall"
                    else "not run", ha="center", va="center", transform=ax.transAxes, fontsize=8, color="grey")
            ax.set_title(panel_title, fontsize=9); continue
        it, comp, acc, lam = load(run)
        ax.plot(it, comp, color=PALETTE[1], lw=0.6, alpha=0.25)
        ax.plot(it, acc, color=PALETTE[0], lw=0.6, alpha=0.25, ls="--")
        ax.plot(it, roll(comp, 5), color=PALETTE[1], lw=1.8, label="in-dist compliance (T=1)")
        ax.plot(it, roll(acc, 5), color=PALETTE[0], lw=1.8, ls="--", label="in-dist accuracy (T=1)")
        name = run.name
        if name in held and held[name].get("heldout"):
            H = {int(s): v for s, v in held[name]["heldout"].items()}; S = sorted(H)
            ax.plot(S, [H[s]["honest"] for s in S], "o", color="black", ms=3, label="held-out honest (T=0)")
        if np.isfinite(lam).any():
            ax2 = ax.twinx(); ax2.plot(it, lam, color="grey", lw=1.0, alpha=0.8)
            ax2.set_ylim(0, 3.2); ax2.set_yticks([0, 1, 2, 3]); ax2.tick_params(labelsize=7, colors="grey")
            if col == len(ARMS) - 1: ax2.set_ylabel("λ", color="grey", fontsize=8)
        sm = roll(comp, 10); m = it >= 50
        peaks, _ = find_peaks(sm[m], prominence=0.10)
        dead = (roll(acc, 5)[-1] < 0.1)
        ax.set_title(f"{panel_title}: {len(peaks)} cycles, slow std {np.std(sm[m]):.3f}"
                     + ("  [COLLAPSED]" if dead else ""), fontsize=8.5)
        ax.set_ylim(-0.02, 1.0); ax.set_xlim(0, 400)
for ax in axes[-1]: ax.set_xlabel("gradient step")
for row, k in enumerate(KS): axes[row][0].set_ylabel(f"{k}\nrate")
axes[0][0].legend(fontsize=7.5, loc="upper left", framealpha=0.9)
fig.suptitle("sweep23 trajectories — faint: raw per step; bold: 5-step rolling; grey: λ (control arms); dots: held-out honest compliance.\n"
             "cycles = peaks of 10-step-smoothed compliance with prominence ≥ 0.10 after step 50; slow std = std of that smoothed series after step 50", y=1.0)
fig.tight_layout()
out = Path(__file__).resolve().parent / "sweep23_trajectories.png"
fig.savefig(out, dpi=150, bbox_inches="tight"); print(out)
