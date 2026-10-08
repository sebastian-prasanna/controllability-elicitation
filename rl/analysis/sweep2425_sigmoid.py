"""Per-recipe held-out sigmoids for sweep24/25 (+ sweep18-zt reference): honest held-out
(compliant ∧ answered ∧ non-hollow, 200q val, T=0) vs k. Per run: gate-A best (acc >= .8x step0),
neighbor-smoothed best (mean over ckpt ±10), endpoint. Runs summarize.py first to fill the cache.
Usage: python rl/analysis/sweep2425_sigmoid.py [--no-refresh]"""
import json, subprocess, sys, glob
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "rl/runs/sweep23_conjunctive/summary_cache.json"
SUMM = ROOT / "rl/runs/sweep23_conjunctive/summarize.py"
finished = [p for p in glob.glob(str(ROOT / "rl/runs/sweep2[45]_*/sweep2*-*")) if Path(p, "eval_summary.json").exists()]
if "--no-refresh" not in sys.argv:
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [p for p in finished if Path(p).name not in cache]
    if todo:
        subprocess.run([sys.executable, str(SUMM), *todo], check=True, capture_output=True)
cache = json.loads(CACHE.read_text())
KMAP = {"k100": 100, "k200": 200, "k300": 300, "k500": 500, "k700": 700, "k1k": 1000, "k3k": 3000, "k10k": 10000, "k30k": 30000, "k100k": 100000, "kall": 497664}
ARMS = {"s24 damp": "sweep24-damp-", "s24 cap2": "sweep24-cap2-", "s24 dampcap2": "sweep24-dampcap2-", "s24 fix0.5": "sweep24-fix05-", "s24 fix2": "sweep24-fix20-",
        "s25 fix0.75 d0.33": "sweep25-d033-", "s25 fix0.75 d0.5": "sweep25-d050-", "s25 fix0.75 d1": "sweep25-d100-", "s18 zt (drive 1 ref)": "sweep18-zt-"}
def stats(name):
    H = {int(s): v for s, v in cache[name]["heldout"].items()}; S = sorted(H); s0 = H[0]
    ok = [s for s in S if s > 0 and H[s]["acc"] >= 0.8 * s0["acc"]]
    best = max(ok, key=lambda s: H[s]["honest"]) if ok else None
    sm = {s: np.mean([H[t]["honest"] for t in S if abs(t - s) <= 10]) for s in S if s > 0}; sb = max(sm, key=sm.get)
    return dict(s0=s0["honest"], best=H[best]["honest"] if best else float("nan"), best_step=best, sm=sm[sb], sm_step=sb,
                end=H[S[-1]]["honest"], end_step=S[-1], end_acc=H[S[-1]]["acc"], acc_best=H[best]["acc"] if best else float("nan"), acc0=s0["acc"])
rows = {}
for arm, pre in ARMS.items():
    for name in cache:
        if not name.startswith(pre): continue
        k = name[len(pre):].replace("-lr33", "")
        if k not in KMAP: continue
        rows.setdefault(arm, {})[KMAP[k]] = stats(name)
ks = sorted({k for a in rows.values() for k in a})
print("neighbor-smoothed best honest held-out (gate-A best in parens; 'x' = collapsed (endpoint accuracy <.15))")
print(f"{'arm':22s}" + "".join(f"{k:>12d}" for k in ks))
for arm in ARMS:
    if arm not in rows: continue
    line = f"{arm:22s}"
    for k in ks:
        st = rows[arm].get(k)
        line += f"{'':>12s}" if st is None else f"{st['sm']:.2f}({st['best']:.2f}){'x' if st['end_acc'] < 0.15 else ' '}".rjust(12)
    print(line)
print("\nSFT donor step-0 honest (mean over arms):", {k: round(np.mean([rows[a][k]['s0'] for a in rows if k in rows[a]]), 3) for k in ks})
# figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(ROOT)); from utils import PALETTE, set_matplotlib_style
set_matplotlib_style(); fig, ax = plt.subplots(figsize=(8.5, 5.2))
sft = {k: np.mean([rows[a][k]["s0"] for a in rows if k in rows[a]]) for k in ks}
ax.plot(list(sft), list(sft.values()), "o-", color="black", lw=1.5, ms=4, label="SFT donor (step 0)")
for i, arm in enumerate([a for a in ARMS if a in rows]):
    kk = sorted(rows[arm]); ax.plot(kk, [rows[arm][k]["sm"] for k in kk], "s-", color=PALETTE[i % len(PALETTE)], lw=1.6, ms=4, label=arm,
                                    ls="--" if "ref" in arm else "-")
ax.set_xscale("log"); ax.set_ylim(0, 0.75); ax.set_xlabel("trainable parameters k"); ax.set_ylabel("held-out honest compliance, smoothed best ckpt (200q val, T=0)")
ax.legend(fontsize=7.5, ncol=2, loc="upper left"); ax.set_title("sweep24/25 recipes: best-checkpoint held-out sigmoid per recipe")
fig.tight_layout(); out = ROOT / "rl/analysis/sweep2425_sigmoid.png"; fig.savefig(out, dpi=160, bbox_inches="tight"); print(out)
