"""sweep25 (fixed λ=0.75, floors, zt) held-out sigmoids per drive: honest held-out (compliant ∧ answered ∧ non-hollow,
200q val, T=0) vs k. Per run: neighbor-smoothed best (mean over ckpt ±10, solid), gate-A best (raw acc ≥ .8× step 0, hollow),
endpoint (dotted). References: SFT donor step 0 and sweep24 fixed λ=0.5 drive 1 (chosen recipe).
Right panel: held-out accuracy at the smoothed-best checkpoint vs step 0."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); from utils import PALETTE, set_matplotlib_style
cache = json.loads((ROOT / "rl/runs/sweep23_conjunctive/summary_cache.json").read_text())
KMAP = {"k300": 300, "k500": 500, "k700": 700, "k1k": 1000, "k3k": 3000, "k10k": 10000, "k30k": 30000, "k100k": 100000, "kall": 497664}
ARMS = {"drive 0.33": "sweep25-d033-", "drive 0.5": "sweep25-d050-", "drive 1": "sweep25-d100-"}
REF = ("fixed λ 0.5, drive 1 (sweep24, chosen)", "sweep24-fix05-")
def stats(name):
    H = {int(s): v for s, v in cache[name]["heldout"].items()}; S = sorted(H); s0 = H[0]
    ok = [s for s in S if s > 0 and H[s]["acc"] >= 0.8 * s0["acc"]]; best = max(ok, key=lambda s: H[s]["honest"]) if ok else None
    sm = {s: np.mean([H[t]["honest"] for t in S if abs(t - s) <= 10]) for s in S if s > 0}; sb = max(sm, key=sm.get)
    return dict(s0=s0["honest"], acc0=s0["acc"], best=H[best]["honest"] if best else np.nan, best_step=best, sm=sm[sb], sm_step=sb, acc_sm=H[sb]["acc"],
                end=H[S[-1]]["honest"], end_step=S[-1], end_acc=H[S[-1]]["acc"], collapsed=H[S[-1]]["honest"] < 0.1 and sm[sb] > 0.3)
def collect(pre):
    out = {}
    for name in cache:
        if name.startswith(pre) and cache[name].get("heldout"):
            k = name[len(pre):]
            if k in KMAP: out[KMAP[k]] = stats(name)
    return out
rows = {a: collect(p) for a, p in ARMS.items()}; rows["kall (drive .23, lr 3.3e-4)"] = {}
# kall was run once at pinned lr (drive .23) under the d033 prefix: keep it in the d0.33 curve but flag it
ref = collect(REF[1]); ks = sorted({k for a in rows.values() for k in a} | set(ref))
print("sweep25 fixed λ=0.75: honest held-out per drive. cells = smoothed-best (gate-A best) [endpoint]; 'x' = collapsed (endpoint honest <.1)")
print(f"{'arm':12s}" + "".join(f"{k:>18d}" for k in ks))
for a in ARMS:
    line = f"{a:12s}"
    for k in ks:
        st = rows[a].get(k); line += f"{'':>18s}" if st is None else f"{st['sm']:.2f}({st['best']:.2f})[{st['end']:.2f}]{'x' if st['collapsed'] else ' '}".rjust(18)
    print(line)
line = f"{'ref fix0.5 d1':12s}"
for k in ks:
    st = ref.get(k); line += f"{'':>18s}" if st is None else f"{st['sm']:.2f}({st['best']:.2f})[{st['end']:.2f}]{'x' if st['collapsed'] else ' '}".rjust(18)
print(line)
print("SFT donor step 0:", {k: round(np.mean([r[k]["s0"] for r in list(rows.values()) + [ref] if k in r]), 2) for k in ks})
print("\nheld-out accuracy at smoothed-best ckpt (step 0 in parens)")
for a in ARMS: print(f"{a:12s}" + "".join((f"{rows[a][k]['acc_sm']:.2f}({rows[a][k]['acc0']:.2f})" if k in rows[a] else "").rjust(14) for k in ks))
set_matplotlib_style(); fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
sft = {k: np.mean([r[k]["s0"] for r in list(rows.values()) + [ref] if k in r]) for k in ks}
ax.plot(list(sft), list(sft.values()), "o-", color="black", lw=1.5, ms=4, label="SFT donor (step 0)")
ax2.plot(list(sft), [np.mean([r[k]["acc0"] for r in list(rows.values()) + [ref] if k in r]) for k in ks], "o-", color="black", lw=1.5, ms=4, label="SFT donor (step 0)")
for i, a in enumerate(ARMS):
    kk = sorted(rows[a]); c = PALETTE[i]
    ax.plot(kk, [rows[a][k]["sm"] for k in kk], "s-", color=c, lw=1.8, ms=5, label=f"{a}: smoothed best ckpt")
    ax.plot(kk, [rows[a][k]["best"] for k in kk], "s", color=c, ms=7, mfc="none", mew=1.2, alpha=0.7, label=f"{a}: gate-A best" if i == 0 else None)
    ax.plot(kk, [rows[a][k]["end"] for k in kk], ":", color=c, lw=1.2, alpha=0.8, label=f"{a}: endpoint" if i == 0 else None)
    for k in kk:
        if rows[a][k]["collapsed"]: ax.plot(k, rows[a][k]["sm"], "x", color=c, ms=10, mew=2)
    ax2.plot(kk, [rows[a][k]["acc_sm"] for k in kk], "s-", color=c, lw=1.8, ms=5, label=f"{a}: at smoothed-best ckpt")
kk = sorted(ref); ax.plot(kk, [ref[k]["sm"] for k in kk], "d--", color="grey", lw=1.4, ms=4, label=REF[0]); ax2.plot(kk, [ref[k]["acc_sm"] for k in kk], "d--", color="grey", lw=1.4, ms=4, label=REF[0])
TICKS = [300, 1000, 3000, 10000, 30000, 100000, 497664]; TL = ["300", "1k", "3k", "10k", "30k", "100k", "all\n(498k)"]
for a_ in (ax, ax2): a_.set_xscale("log"); a_.set_xticks(TICKS); a_.set_xticklabels(TL); a_.minorticks_off(); a_.set_xlabel("trainable parameters k"); a_.grid(alpha=0.3)
ax.set_ylim(0, 0.75); ax.set_ylabel("held-out honest compliance (200q val, T=0)"); ax.set_title("sweep25 fixed λ = 0.75: honest held-out vs k, per drive\n(× = run collapsed before 250 steps; kall point is lr-pinned, drive .23)")
ax2.set_ylim(0, 0.6); ax2.set_ylabel("held-out accuracy (200q val, T=0)"); ax2.set_title("held-out accuracy at the selected checkpoint")
ax.legend(fontsize=7.5, loc="upper left"); ax2.legend(fontsize=7.5, loc="lower left")
fig.tight_layout(); out = ROOT / "rl/analysis/sweep25_sigmoid.png"; fig.savefig(out, dpi=160, bbox_inches="tight"); print(out)
