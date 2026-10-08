"""sweep28 (gpt-oss-120b, final recipe, drive 1) trajectories at k30k / k100k: in-dist T=1 compliance (solid) and
accuracy (dashed), 5-step rolling; held-out T=0 markers every 10 steps: raw compliance (squares), accuracy (triangles),
honest (black dots), start_of_sentence honest (grey x)."""
import json, sys, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); from utils import PALETTE, set_matplotlib_style
cache = json.loads((ROOT / "rl/runs/sweep23_conjunctive/summary_cache.json").read_text())
RUNS = ROOT / "rl/runs/sweep29_gptoss120b_fs1"
def roll(x, w=5): x = np.asarray(x, float); return np.array([np.nanmean(x[max(0, i - w + 1):i + 1]) for i in range(len(x))])
def rd(p):
    p = Path(p)
    if p.suffix == ".zst":
        import zstandard; return json.loads(zstandard.ZstdDecompressor().decompress(p.read_bytes(), max_output_size=1 << 31))
    return json.loads(p.read_text())
set_matplotlib_style(); fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey=True)
for ax, k in zip(axes.flat, ["k1k", "k3k", "k10k", "k30k", "k100k", "kall"]):
    run = RUNS / f"sweep29-fs1-{k}"; rows = [json.loads(l) for l in open(run / "progress.jsonl")]
    it = np.array([r["iteration"] for r in rows]); comp = np.array([r["compliance_rate"] for r in rows]); acc = np.array([r["accuracy"] for r in rows])
    ax.plot(it, comp, color=PALETTE[1], lw=.6, alpha=.2); ax.plot(it, acc, color=PALETTE[0], lw=.6, alpha=.2, ls="--")
    ax.plot(it, roll(comp), color=PALETTE[1], lw=1.6, label="in-dist compliance (T=1)"); ax.plot(it, roll(acc), color=PALETTE[0], lw=1.6, ls="--", label="in-dist accuracy (T=1)")
    H = {int(s): v for s, v in cache[run.name]["heldout"].items()}; S = sorted(H)
    ax.plot(S, [H[s]["comp"] for s in S], "s-", color=PALETTE[1], ms=4, lw=1, mfc="white", mew=1.2, label="held-out compliance (T=0, raw)")
    ax.plot(S, [H[s]["acc"] for s in S], "^-", color=PALETTE[0], ms=4, lw=1, mfc="white", mew=1.2, label="held-out accuracy (T=0)")
    ax.plot(S, [H[s]["honest"] for s in S], "o-", color="black", ms=3.5, lw=1, label="held-out honest (T=0)")
    sos = []
    for s in S:
        d = rd(glob.glob(str(run / f"eval/heldout/checkpoint-{s}.json*"))[0]); X = [x for r in d["results"] if r["mode"] == "start_of_sentence" for x in r["samples"]]
        sos.append(sum(x["compliance"] == 1 for x in X) / len(X))
    ax.plot(S, sos, "x:", color="grey", ms=5, lw=1, label="held-out start_of_sentence (raw)")
    best = max(S, key=lambda s: H[s]["honest"])
    ax.set_title(f"gpt-oss-120b fewshot-donor {k} — best honest {H[best]['honest']:.2f}@{best}, endpoint {H[S[-1]]['honest']:.2f}", fontsize=10)
    ax.set_xlabel("gradient step"); ax.set_ylim(-.02, 1); ax.set_xlim(0, 200); ax.grid(alpha=.3)
axes.flat[0].set_ylabel("rate"); axes.flat[0].legend(fontsize=7.5, loc="upper left", framealpha=.9)
fig.suptitle("sweep29: gpt-oss-120b from FEWSHOT donors, final recipe + sent_floor (fixed λ .5, drive 1, floors, zt, 6 modes, 16k)"); fig.tight_layout()
out = ROOT / "rl/analysis/sweep29_trajectories.png"; fig.savefig(out, dpi=150, bbox_inches="tight"); print(out)
