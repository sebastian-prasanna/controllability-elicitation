"""Corrected-metric leaderboard over ALL gpt-oss-20b RL held-out evals (T=0),
plus SFT vs SFT+RL logistic ("sigmoid") figure in final_plotting style.

Per checkpoint (raw eval files): comp, acc, ans_rate, acc_ans (accuracy among
answered), c_and_a, c_and_c. Two honest gates:
  A ("acc80"):  raw accuracy >= 0.8 x it0 raw accuracy
  B ("hybrid"): acc_ans >= 0.9 x it0 acc_ans  AND  ans_rate >= 0.9
Best checkpoint per run = max compliance among gated; best per k = max over runs.
"""
import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "rl" / "runs"
CACHE = Path(__file__).resolve().parent / "rl_heldout_ckpt_stats.json"

SWEEPS = ["sweep14_x320gepa_gptoss20b", "sweep17_t80_fullk", "sweep18_truncfix",
          "sweep19_truncfix_t80", "sweep20_fixedlam", "sweep21_pidlam", "sweep22_drive", "sweep23_conjunctive"]

def _read_json(path):
    p = Path(path)
    if p.suffix == ".zst":
        import zstandard
        return json.loads(zstandard.ZstdDecompressor().decompress(p.read_bytes(), max_output_size=1 << 31))
    return json.loads(p.read_text())

def ckpt_stats(path):
    raw = _read_json(path)
    s = [x for r in raw["results"] for x in r["samples"]]
    n = len(s)
    ans = [x for x in s if x.get("extracted_answer")]
    return {
        "n": n,
        "comp": sum(x["compliance"] for x in s) / n,
        "acc": sum(x["correct"] for x in s) / n,
        "ans_rate": len(ans) / n,
        "acc_ans": (sum(x["correct"] for x in ans) / len(ans)) if ans else 0.0,
        "c_and_a": sum(1 for x in s if x["compliance"] and x.get("extracted_answer")) / n,
        "c_and_c": sum(1 for x in s if x["compliance"] and x["correct"]) / n,
    }

def build_cache():
    import yaml
    rows = []
    for sw in SWEEPS:
        for run in sorted((RUNS / sw).iterdir()):
            hd = run / "eval" / "heldout"
            cfgf = run / "config.yaml"
            if not hd.is_dir() or not cfgf.exists() or run.name.startswith(("t1eval", "t1rec")):
                continue
            cfg = yaml.safe_load(cfgf.read_text())
            k = (cfg.get("elicitation") or {}).get("train_params") or 497_664
            for f in sorted(list(hd.glob("checkpoint-*.json")) + list(hd.glob("checkpoint-*.json.zst"))):
                step = int(f.name.split("-")[1].split(".")[0])
                try:
                    st = ckpt_stats(f)
                except Exception as e:
                    print(f"skip {f}: {e}"); continue
                rows.append({"sweep": sw, "run": run.name, "k": int(k), "step": step, **st})
            print(f"scanned {run.name}")
    CACHE.write_text(json.dumps(rows))
    return rows

def main(metric="comp"):
    rows = json.loads(CACHE.read_text()) if CACHE.exists() and "--rescan" not in sys.argv else build_cache()
    import pandas as pd
    df = pd.DataFrame(rows)
    base = df[df.step == 0].set_index("run")

    def gated_best(run, g, gate):
        run0 = base.loc[run] if run in base.index else None
        if run0 is None: return None
        if gate == "A":
            ok = g[g.acc >= 0.8 * run0.acc]
        else:
            ok = g[(g.acc_ans >= 0.9 * run0.acc_ans) & (g.ans_rate >= 0.9)]
        if ok.empty: return None
        return ok.loc[ok[metric].idxmax()]

    out = {}
    for gate in ["A", "B"]:
        best = []
        for run, g in df.groupby("run"):
            b = gated_best(run, g, gate)
            if b is not None:
                best.append(b)
        bd = pd.DataFrame(best)
        per_k = bd.loc[bd.groupby("k")[metric].idxmax()].sort_values("k")
        out[gate] = per_k
    sft = df[df.step == 0].groupby("k").agg(val=(metric, "mean"), n=("n", "sum")).reset_index()
    return df, sft, out

def make_figure(df, sft, out, metric="comp", ylabel="held-out compliance", fname="rl_sft_vs_rl_sigmoid.png"):
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    sys.path.insert(0, str(ROOT))
    from utils import PALETTE, set_matplotlib_style

    def logistic(x, c0, c1, b, m):
        return c0 + (c1 - c0) / (1.0 + np.exp(-b * (x - m)))

    def fit_logistic(xs, ys, ns, n_boot=300, seed=0):
        xs, ys, ns = np.asarray(xs, float), np.asarray(ys, float), np.asarray(ns, int)
        p0 = [max(ys.min(), 1e-3), ys.max(), 1.5, 3.5]
        bounds = ([0, 0.05, 0.1, 1.0], [0.3, 1.0, 10.0, 7.0])
        popt, _ = curve_fit(logistic, xs, ys, p0=p0, bounds=bounds, maxfev=40000)
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            yb = rng.binomial(ns, np.clip(ys, 0, 1)) / ns
            try:
                boots.append(curve_fit(logistic, xs, yb, p0=popt, bounds=bounds, maxfev=40000)[0])
            except RuntimeError:
                pass
        return popt, np.array(boots)

    set_matplotlib_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    grid = np.linspace(2, 5.8, 200)
    series = [("SFT (donor ckpt-540)", sft.k.values, sft.val.values, sft.n.values, PALETTE[0], "o"),
              ("SFT+RL, gate A (acc ≥ 0.8× it0)", out["A"].k.values, out["A"][metric].values,
               out["A"].n.values, PALETTE[1], "s"),
              ("SFT+RL, gate B (acc|ans ≥ 0.9×, ans ≥ 0.9)", out["B"].k.values, out["B"][metric].values,
               out["B"].n.values, PALETTE[2], "D")]
    for label, ks, ys, ns, color, marker in series:
        xs = np.log10(ks)
        popt, boots = fit_logistic(xs, ys, ns)
        ax.plot(10 ** grid, logistic(grid, *popt), color=color, lw=2.0, label=label)
        if len(boots):
            band = np.array([logistic(grid, *b) for b in boots])
            ax.fill_between(10 ** grid, np.percentile(band, 2.5, 0), np.percentile(band, 97.5, 0),
                            color=color, alpha=0.15, lw=0)
        ax.plot(ks, ys, marker, color=color, ms=5, ls="none")
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters k (rank-1 LoRA; rightmost = full adapter)")
    ax.set_ylabel(ylabel + " (3 unseen modes, 200q val, T=0)")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("gpt-oss-20b: SFT vs SFT+RL (best honest checkpoint per k, all sweeps)")
    fig.tight_layout()
    outp = Path(__file__).resolve().parent / fname
    fig.savefig(outp, dpi=180, bbox_inches="tight")
    print(outp)


if __name__ == "__main__":
    metric = "c_and_a" if "--canda" in sys.argv else "comp"
    df, sft, out = main(metric)
    print("\n=== SFT (step-0 donors, heldout comp) ===")
    print(sft.to_string(index=False))
    for gate, label in [("A", "gate A: acc>=0.8x it0"), ("B", "gate B: acc_ans>=0.9x & ans>=0.9")]:
        print(f"\n=== best SFT+RL per k ({label}) ===")
        print(out[gate][["k", "run", "step", "comp", "c_and_a", "acc", "ans_rate", "acc_ans", "c_and_c"]].to_string(index=False))
    if metric == "comp":
        make_figure(df, sft, out)
    else:
        make_figure(df, sft, out, metric="c_and_a",
                    ylabel="held-out compliant \u2227 answered",
                    fname="rl_sft_vs_rl_sigmoid_canda.png")
