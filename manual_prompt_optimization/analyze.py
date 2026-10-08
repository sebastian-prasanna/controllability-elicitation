"""Per-model comparison (agents vs GEPA vs few-shot vs baseline, test split) and cross-model summary.

    .venv/bin/python manual_prompt_optimization/analyze.py --model kimik3 --sweep sweep1
    .venv/bin/python manual_prompt_optimization/analyze.py --sweep sweep1 --all

Per model writes runs/<sweep>/<key>/comparison.md, compliance_vs_length.png (binned per-rollout strict
compliance vs reasoning words, gepa/plotting.ipynb style, default + held-out panels) and
compliance_vs_length_data.json. --all writes runs/<sweep>/summary.md over every model with a freeze.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manual_prompt_optimization import models as M  # noqa: E402
from utils import PALETTE, set_matplotlib_style  # noqa: E402

set_matplotlib_style()
MIN_BIN_N = 50
HELDOUT = ["start_of_sentence", "letter_suppression", "no_spaces"]


def collect(key: str, sweep: str, n_agents: int) -> dict:
    """method -> {run label -> {'default': eval, 'heldout': eval}} (per-rollout arrays included)."""
    r = M.ref_dirs(key)
    out = {"Baseline (empty prompt)": {"": {"default": M.load_eval(r["baseline_test"], True),
                                            "heldout": M.load_eval(r["baseline_heldout"], True)}}}
    out["GEPA (general-advice)"] = {lbl: {"default": M.load_eval(d / "test_eval", True),
                                          "heldout": M.load_eval(d / "heldout_eval", True)} for lbl, d in r["gepa_general"]}
    out["Few-shot k=1"] = {lbl: {"default": M.load_eval(d, True),
                                 "heldout": M.load_eval(dict(r["fewshot_k1_heldout"])[lbl], True)} for lbl, d in r["fewshot_k1_test"]}
    out["Fable subagents (manual)"] = {}
    for i in range(1, n_agents + 1):
        ad = M.agent_dir(sweep, key, i)
        if (ad / "final_prompt.txt").exists():
            out["Fable subagents (manual)"][f"a{i}"] = {"default": M.load_eval(ad / "test_eval", True),
                                                        "heldout": M.load_eval(ad / "heldout_eval", True)}
    return out


COLORS = {"Baseline (empty prompt)": PALETTE[0], "GEPA (general-advice)": PALETTE[2],
          "Few-shot k=1": PALETTE[3], "Fable subagents (manual)": PALETTE[1]}


def binned(words, comp, bins):
    idx = np.digitize(words, bins) - 1
    out = []
    for b in range(len(bins) - 1):
        m = idx == b; n = int(m.sum())
        if n < MIN_BIN_N: continue
        p = comp[m].mean(); out.append((np.sqrt(bins[b] * bins[b + 1]), p, np.sqrt(p * (1 - p) / n), n))
    return np.array(out).reshape(-1, 4)


def plot(key: str, data: dict, out: Path, best: dict):
    allw = [v["words"] for runs in data.values() for pv in runs.values() for v in pv.values() if v]
    bins = np.geomspace(1, max(w.max() for w in allw) * 1.001, 14)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, panel, title in zip(axes, ("default", "heldout"), ("9 default modes (test, 4464 rollouts/run)", "3 held-out modes (test, 1500 rollouts/run)")):
        for method, runs in data.items():
            for lbl, pv in runs.items():
                v = pv.get(panel)
                if not v: continue
                pts = binned(v["words"], v["comp"], bins)
                if not len(pts): continue
                cx, p, se, n = pts.T
                if lbl == best[method]:
                    ax.errorbar(cx, 100 * p, yerr=100 * se, color=COLORS[method], label=method, marker="o", ms=4, lw=1.6, capsize=2)
                else:
                    ax.plot(cx, 100 * p, color=COLORS[method], lw=0.9, alpha=0.45, marker="o", ms=2.2, markerfacecolor="none")
        ax.set_xscale("log"); ax.set_title(title, fontsize=10); ax.grid(True, which="both", alpha=0.15)
        ax.set_xlabel("Reasoning length (words, log)")
    axes[0].set_ylabel("Strict compliance (%)")
    h, l = axes[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color="#52514e", lw=1.6, marker="o", ms=4), plt.Line2D([], [], color="#52514e", lw=0.9, alpha=0.45, marker="o", ms=2.2, markerfacecolor="none")]
    l += ["best of 3 runs (by default-mode test strict)", "other runs"]
    fig.legend(h, l, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.08), frameon=False, fontsize=8.5)
    fig.suptitle(f"{M.MODELS[key]}: strict compliance vs reasoning length by prompt method (test split, log word bins, n≥{MIN_BIN_N})", y=1.14, fontsize=11)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def per_model(key: str, sweep: str, n_agents: int) -> dict:
    rd = M.run_dir(sweep, key)
    data = collect(key, sweep, n_agents)
    best = {}
    for method, runs in data.items():
        cands = {lbl: pv["default"]["strict"] for lbl, pv in runs.items() if pv.get("default")}
        best[method] = max(cands, key=cands.get) if cands else None
    plot(key, data, rd / "compliance_vs_length.png", best)
    rows = []
    for method, runs in data.items():
        for lbl, pv in runs.items():
            for panel in ("default", "heldout"):
                v = pv.get(panel)
                if v:
                    rows.append({"method": method, "run": lbl, "panel": panel, "best": lbl == best[method],
                                 **{k: v[k] for k in ("strict", "accuracy", "n", "mean_chars", "median_chars", "mean_words",
                                                      "truncated", "empty_output", "near_empty_reasoning", "max_tokens", "per_mode")}})
    json.dump(rows, open(rd / "compliance_vs_length_data.json", "w"), indent=1)

    fz = json.load(open(rd / "freeze.json")) if (rd / "freeze.json").exists() else {}
    md = [f"# {key} ({M.MODELS[key]}): Fable subagents vs GEPA vs few-shot — TEST split\n",
          f"Sweep `{sweep}`. Agent finals frozen {fz.get('frozen', '?')}; test evals run once after the freeze. "
          "One entry per agent (its final_prompt.txt, selected by the agent on val strict compliance). "
          "\"Best\" per method = highest default-mode test strict compliance (GEPA best-of-3-seeds convention).\n",
          "![compliance vs length](compliance_vs_length.png)\n",
          "## Default modes (test, 500 q x 9 modes)\n",
          "| method | run | strict | acc | mean chars | median chars | trunc@16k | empty out |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["panel"] == "default":
            md.append(f"| {r['method']} | {r['run'] or '-'}{' *' if r['best'] else ''} | {fmt(r['strict'])} | {fmt(r['accuracy'])} | "
                      f"{r['mean_chars']/1000:.1f}k | {r['median_chars']/1000:.1f}k | {r['truncated']} | {r['empty_output']} |")
    md += ["\n## Held-out modes (test, 500 q x 3 modes)\n",
           "| method | run | strict | acc | mean chars | " + " | ".join(HELDOUT) + " |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["panel"] == "heldout":
            md.append(f"| {r['method']} | {r['run'] or '-'}{' *' if r['best'] else ''} | {fmt(r['strict'])} | {fmt(r['accuracy'])} | "
                      f"{r['mean_chars']/1000:.1f}k | " + " | ".join(fmt(r["per_mode"].get(m)) for m in HELDOUT) + " |")
    modes = sorted(next(r["per_mode"] for r in rows if r["panel"] == "default"))
    md += ["\n## Per-mode strict compliance, default modes (test)\n",
           "| method | run | " + " | ".join(modes) + " |", "|---|---|" + "---|" * len(modes)]
    for r in rows:
        if r["panel"] == "default":
            md.append(f"| {r['method']} | {r['run'] or '-'} | " + " | ".join(fmt(r["per_mode"].get(m), 2) for m in modes) + " |")
    if fz:
        md += ["\n## Freeze / judge\n"]
        for i, a in fz.get("agents", {}).items():
            j = a.get("judge") or {}
            md.append(f"- agent{i}: sha256 {a['sha256'][:12]}, {a['chars']} chars, val best {a['val_best']}; "
                      f"judge rule_a_violation={j.get('rule_a_violation')} rule_b_violation={j.get('rule_b_violation')}; notes: {j.get('borderline_notes')}")
        if fz.get("protocol_problems"):
            md.append("- protocol problems: " + "; ".join(fz["protocol_problems"]))
    md.append("\nFew-shot k=1 runs used max_tokens 30k; all others 16k. Data: compliance_vs_length_data.json.")
    (rd / "comparison.md").write_text("\n".join(md) + "\n")
    summ = {"key": key}
    for method, runs in data.items():
        s = [pv["default"]["strict"] for pv in runs.values() if pv.get("default")]
        h = [pv["heldout"]["strict"] for pv in runs.values() if pv.get("heldout")]
        summ[method] = {"default": s, "heldout": h}
    return summ


def cross_model(sweep: str, summaries: list[dict]):
    md = [f"# Sweep `{sweep}`: Fable subagents vs GEPA (general-advice) vs few-shot k=1 — TEST split strict compliance\n",
          "best-of-3 (all runs). Default modes n=4464/run; held-out n=1500/run.\n",
          "| model | baseline | GEPA general | few-shot k=1 | subagents | | baseline HO | GEPA HO | few-shot HO | subagents HO |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    def cell(v):
        return "—" if not v else f"**{max(v):.3f}** ({'/'.join(f'{x:.2f}' for x in v)})"
    for s in summaries:
        b, g, f, a = (s[k] for k in ("Baseline (empty prompt)", "GEPA (general-advice)", "Few-shot k=1", "Fable subagents (manual)"))
        md.append(f"| {s['key']} | {cell(b['default'])} | {cell(g['default'])} | {cell(f['default'])} | {cell(a['default'])} | | "
                  f"{cell(b['heldout'])} | {cell(g['heldout'])} | {cell(f['heldout'])} | {cell(a['heldout'])} |")
    (M.RUNS / sweep / "summary.md").write_text("\n".join(md) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=list(M.MODELS))
    p.add_argument("--sweep", default="sweep1")
    p.add_argument("--n-agents", type=int, default=M.N_AGENTS)
    p.add_argument("--all", action="store_true", help="every model in the sweep with a freeze.json")
    a = p.parse_args()
    keys = [k for k in M.MODELS if (M.run_dir(a.sweep, k) / "freeze.json").exists()] if a.all else [a.model]
    summaries = [per_model(k, a.sweep, a.n_agents) for k in keys]
    for k in keys:
        print(f"wrote {M.run_dir(a.sweep, k).relative_to(M.ROOT)}/comparison.md")
    if a.all:
        cross_model(a.sweep, summaries); print(f"wrote runs/{a.sweep}/summary.md")


if __name__ == "__main__":
    main()
