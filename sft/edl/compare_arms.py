"""Cross-arm summary of the per-sweep EDL analyses (sft/edl/analyze.py outputs).

    python sft/edl/compare_arms.py sft/runs/x320_*_drive3 --out sft/runs/x320_launch/edl

Writes summary.md (per-arm headline numbers at a few k plus the k>=500 mean), arms.json (every
per-cell record, tagged by arm) and fig_arms.png (EDL/gain, EDL/param, total EDL and gain, and
examples-to-half-gain vs k; colour = model, solid = GEPA-general data, dashed = few-shot-k1 data;
open squares = full-coverage LoRA cells).
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

import numpy as np

MODEL_COL = {"gptoss20b": "#2a78d6", "gptoss120b": "#0f3f7a", "qwen8b": "#eb6834", "qwen32b": "#9a3412",
             "qwen36a3b": "#1baf7a"}
PROMPT_LS = {"gepa_general": "-", "fewshot_k1": "--"}
REF_KS = (100, 300, 1000, 10000, 100000)


def arm_of(sweep: Path):
    m = re.match(r"x320_([a-z0-9]+)(?:_[a-z0-9]+data)?_(gepa_general|fewshot_k1)_drive3", sweep.name)
    return (m.group(1), m.group(2)) if m else (sweep.name, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweeps", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    arms = {}
    for s in map(Path, a.sweeps):
        f = s / "edl" / "edl_results.json"
        if not f.exists():
            print(f"skip {s.name}: no edl_results.json"); continue
        d = json.load(open(f)); model, prompt = arm_of(s)
        arms[s.name] = dict(model=model, prompt=prompt, base_model=d.get("base_model"), N=d["N_sup_tokens"],
                            L0=d["L0_mean_train"], MDL0=d["MDL_base_bits"], cells=d["per_cell"])
    json.dump(arms, open(out / "arms.json", "w"), indent=1)

    def masked(arm):
        cs = [r for r in arm["cells"].values() if r["kind"] == "masked"]
        return sorted(cs, key=lambda r: r["k"])

    def full(arm):
        return sorted([r for r in arm["cells"].values() if r["kind"] == "full"], key=lambda r: r["k"])

    def at(arm, k, key):
        r = arm["cells"].get(f"k{k}")
        return None if r is None else r.get(key)

    def f(v, fmt="%.2f", scale=1.0):
        return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt % (v * scale)

    lines = ["# EDL across x320 arms (model x training-data prompt), 1 epoch of 8640 examples, drive-3 masked rank-1 LoRA + full-coverage LoRA", "",
             "EDL/gain = (Lbar - L*)/(L0 - L*): share of the achievable codelength saving not yet obtained on average over the epoch "
             "(0 = instant elicitation). L* = extrapolated difficulty-adjusted training curve (primary estimator). "
             "n@50% = training examples after which half of the final loss gain is reached. Bits/param = EDL / trainable params. "
             "See each sweep's edl/EDL_FINDINGS.md or edl/edl_table.md for the full per-cell table.", ""]
    hdr = ["arm", "base L0 (nats/tok)", "N sup tokens (M)"] + [f"EDL/gain k={k}" for k in REF_KS] + ["EDL/gain mean k>=500 (masked)", "EDL/gain full r1/r4/r16",
           "EDL kbit k=1k / 100k / full-r16", "gain Mbit k=1k / 100k / full-r16", "n@50% k=1k / 100k", "bits/param k=1k / 100k / full-r16",
           "memgap late k=100k", "IID compl k=1k / 100k / full-r16", "HO compl k=1k / 100k / full-r16"]
    lines.append("| " + " | ".join(hdr) + " |"); lines.append("|" + "---|" * len(hdr))
    for name, arm in sorted(arms.items(), key=lambda kv: (kv[1]["model"], kv[1]["prompt"])):
        ms = masked(arm); fs = {r["rank"]: r for r in full(arm)}
        hi = [r["EDL_over_gain"] for r in ms if r["k"] >= 500 and r.get("EDL_over_gain") is not None]
        r16 = fs.get(16, {})
        row = [f"{arm['model']} / {arm['prompt']}", f"{arm['L0']:.3f}", f"{arm['N']/1e6:.1f}"]
        row += [f(at(arm, k, "EDL_over_gain")) for k in REF_KS]
        row += [f(float(np.mean(hi)) if hi else None), " / ".join(f(fs[r]["EDL_over_gain"]) if r in fs else "-" for r in (1, 4, 16)),
                " / ".join([f(at(arm, 1000, "EDL_bits"), "%.0f", 1e-3), f(at(arm, 100000, "EDL_bits"), "%.0f", 1e-3), f(r16.get("EDL_bits"), "%.0f", 1e-3)]),
                " / ".join([f(at(arm, 1000, "gain_bits"), "%.2f", 1e-6), f(at(arm, 100000, "gain_bits"), "%.2f", 1e-6), f(r16.get("gain_bits"), "%.2f", 1e-6)]),
                " / ".join([f(at(arm, 1000, "examples_to_half_gain"), "%d"), f(at(arm, 100000, "examples_to_half_gain"), "%d")]),
                " / ".join([f(at(arm, 1000, "EDL_per_param"), "%.3g"), f(at(arm, 100000, "EDL_per_param"), "%.3g"), f(r16.get("EDL_per_param"), "%.3g")]),
                f(at(arm, 100000, "memgap_late_paired"), "%+.4f"),
                " / ".join([f(at(arm, 1000, "cotcontrol_compliance_final")), f(at(arm, 100000, "cotcontrol_compliance_final")), f(r16.get("cotcontrol_compliance_final"))]),
                " / ".join([f(at(arm, 1000, "heldout_compliance_final")), f(at(arm, 100000, "heldout_compliance_final")), f(r16.get("heldout_compliance_final"))])]
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## EDL(n)/n peak step (elicitation signature: early peak then monotone decline; teaching-like: late peak)", "",
              "| arm | " + " | ".join(f"k={k}" for k in REF_KS) + " | full r1 / r4 / r16 |", "|---|" + "---|" * (len(REF_KS) + 1)]
    for name, arm in sorted(arms.items(), key=lambda kv: (kv[1]["model"], kv[1]["prompt"])):
        fs = {r["rank"]: r for r in full(arm)}
        lines.append(f"| {arm['model']} / {arm['prompt']} | " + " | ".join(f(at(arm, k, "EDLn_per_ex_peak_step"), "%d") for k in REF_KS)
                     + " | " + " / ".join(f(fs[r]["EDLn_per_ex_peak_step"], "%d") if r in fs else "-" for r in (1, 4, 16)) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "font.size": 9})
    panels = [("EDL_over_gain", "EDL / gain  (0 = instant elicitation)", 1.0, False),
              ("EDL_per_param", "EDL / k  (bits per trainable parameter)", 1.0, True),
              ("EDL_bits", "total EDL (Mbit)", 1e-6, False),
              ("gain_bits", "generalization gain N(L0 - L*) (Mbit)", 1e-6, False),
              ("examples_to_half_gain", "examples to 50% of final gain", 1.0, True),
              ("EDL_per_token", "EDL per supervised token (bits)", 1.0, False)]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.2))
    for ax, (key, ylab, sc, logy) in zip(axs.flat, panels):
        for name, arm in arms.items():
            c = MODEL_COL.get(arm["model"], "#555"); ls = PROMPT_LS.get(arm["prompt"], ":")
            ms = [r for r in masked(arm) if r.get(key) is not None]
            ax.plot([r["k"] for r in ms], [r[key] * sc for r in ms], ls, color=c, lw=1.5, marker="o", ms=3.5)
            fs = [r for r in full(arm) if r.get(key) is not None]
            if fs:
                ax.scatter([r["k"] for r in fs], [r[key] * sc for r in fs], marker="s", s=36, facecolor="white",
                           edgecolor=c, lw=1.3, zorder=6, linestyle=ls)
        ax.set_xscale("log"); ax.set_ylabel(ylab)
        if logy:
            ax.set_yscale("log")
        if key == "EDL_over_gain":
            ax.set_ylim(0, max(0.5, ax.get_ylim()[1]))
        if key == "EDL_per_param":
            for y in (0.1, 1):
                ax.axhline(y, color="#bbb", lw=0.8, ls=":")
    for ax in axs[1]:
        ax.set_xlabel("k (trainable parameters)")
    axs[1, 1].set_xlabel("k (trainable parameters)   [open squares: full-coverage LoRA r1/r4/r16 at k = total LoRA params]")
    handles = [Line2D([], [], color=MODEL_COL[m], lw=2, label=m) for m in MODEL_COL if any(a["model"] == m for a in arms.values())]
    handles += [Line2D([], [], color="#444", lw=1.5, ls=ls, label=p) for p, ls in PROMPT_LS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Elicitation vs learning (EDL) across x320 arms", y=0.965)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(out / "fig_arms.png", dpi=150); plt.close(fig)
    print("wrote", out / "fig_arms.png")


if __name__ == "__main__":
    main()
