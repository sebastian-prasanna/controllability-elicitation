"""sweep26 (final recipe + ignore_question, 7 modes) vs sweep24-fix05 (6 modes) on gpt-oss-20b:
held-out honest compliance (200q val, T=0) per k — neighbor-smoothed best ckpt, gate-A best, endpoint — and
held-out accuracy at the smoothed-best ckpt. Runs summarize.py on any finished sweep26 run missing from the cache."""
import json, subprocess, sys, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); from utils import PALETTE, set_matplotlib_style
CACHE = ROOT / "rl/runs/sweep23_conjunctive/summary_cache.json"; SUMM = ROOT / "rl/runs/sweep23_conjunctive/summarize.py"
cache = json.loads(CACHE.read_text())
todo = [p for p in glob.glob(str(ROOT / "rl/runs/sweep26_iq/sweep26-iq-*")) if Path(p, "eval_summary.json").exists() and Path(p).name not in cache]
if todo: subprocess.run([sys.executable, str(SUMM), *todo], check=True, capture_output=True); cache = json.loads(CACHE.read_text())
KMAP = {"k1k": 1000, "k3k": 3000, "k10k": 10000, "k30k": 30000, "k100k": 100000, "kall": 497664}
ARMS = {"6 modes (sweep24 fix0.5)": "sweep24-fix05-", "7 modes + ignore_question (sweep26)": "sweep26-iq-"}
def stats(name):
    H = {int(s): v for s, v in cache[name]["heldout"].items()}; S = sorted(H); s0 = H[0]
    ok = [s for s in S if s > 0 and H[s]["acc"] >= 0.8 * s0["acc"]]; best = max(ok, key=lambda s: H[s]["honest"]) if ok else None
    sm = {s: np.mean([H[t]["honest"] for t in S if abs(t - s) <= 10]) for s in S if s > 0}; sb = max(sm, key=sm.get)
    return dict(s0=s0["honest"], acc0=s0["acc"], best=H[best]["honest"] if best else np.nan, best_step=best, sm=sm[sb], sm_step=sb, acc_sm=H[sb]["acc"], end=H[S[-1]]["honest"], end_acc=H[S[-1]]["acc"])
rows = {a: {KMAP[k]: stats(f"{p}{k}") for k in KMAP if f"{p}{k}" in cache} for a, p in ARMS.items()}
ks = sorted({k for r in rows.values() for k in r})
print("held-out honest: smoothed-best (gate-A best @step) [endpoint] | acc at smoothed-best (step0)")
for a in ARMS:
    print(f"\n{a}")
    for k in ks:
        st = rows[a].get(k)
        if st: print(f"  k={k:>6d}: {st['sm']:.2f} ({st['best']:.2f}@{st['best_step']}) [{st['end']:.2f}] | acc {st['acc_sm']:.2f} ({st['acc0']:.2f})")
set_matplotlib_style(); fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
TICKS = list(KMAP.values()); TL = ["1k", "3k", "10k", "30k", "100k", "all\n(498k)"]
for i, a in enumerate(ARMS):
    kk = sorted(rows[a]); c = PALETTE[i]
    ax.plot(kk, [rows[a][k]["sm"] for k in kk], "s-", color=c, lw=1.8, ms=5, label=f"{a}: smoothed best")
    ax.plot(kk, [rows[a][k]["best"] for k in kk], "s", color=c, ms=8, mfc="none", mew=1.2, alpha=.7, label=f"{a}: gate-A best" if i == 0 else None)
    ax.plot(kk, [rows[a][k]["end"] for k in kk], ":", color=c, lw=1.2, label=f"{a}: endpoint" if i == 0 else None)
    ax2.plot(kk, [rows[a][k]["acc_sm"] for k in kk], "s-", color=c, lw=1.8, ms=5, label=a)
sft = {k: np.mean([rows[a][k]["s0"] for a in rows if k in rows[a]]) for k in ks}
ax.plot(list(sft), list(sft.values()), "o-", color="black", lw=1.4, ms=4, label="SFT donor (step 0)")
ax2.plot(list(sft), [np.mean([rows[a][k]["acc0"] for a in rows if k in rows[a]]) for k in ks], "o-", color="black", lw=1.4, ms=4, label="SFT donor (step 0)")
for a_ in (ax, ax2): a_.set_xscale("log"); a_.set_xticks(TICKS); a_.set_xticklabels(TL); a_.minorticks_off(); a_.set_xlabel("trainable parameters k"); a_.grid(alpha=.3)
ax.set_ylim(0, .75); ax.set_ylabel("held-out honest compliance (200q val, T=0)"); ax.set_title("gpt-oss-20b: 6 modes vs 6 + ignore_question")
ax2.set_ylim(0, .6); ax2.set_ylabel("held-out accuracy at selected ckpt"); ax2.set_title("held-out accuracy at the smoothed-best checkpoint")
ax.legend(fontsize=7.5, loc="upper left"); ax2.legend(fontsize=7.5, loc="lower left"); fig.tight_layout()
out = ROOT / "rl/analysis/sweep26_iq_compare.png"; fig.savefig(out, dpi=160, bbox_inches="tight"); print(out)
