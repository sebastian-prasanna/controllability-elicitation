#!/usr/bin/env python3
"""Compliance vs reasoning length on MCQ (CoT-Control QA) vs MATH-500 / Olympiads, + CoT-necessity table.

Reads cot_necessity/runs/<model>/* and the matching pinned MCQ runs
(pinned_reeval/runs/<model>/{baseline,gepa_<seed>}_test). Writes:
  cot_necessity/analysis/rollouts.parquet   per-rollout (model, arm, dataset, mode, level, words, compliance, correct)
  cot_necessity/analysis/compliance_vs_length.png
  cot_necessity/analysis/results.md
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils import PALETTE, set_matplotlib_style  # noqa: E402

HERE = REPO / "cot_necessity"
OUT = HERE / "analysis"
MODELS = [("kimik3", "Kimi-K3", "s2"), ("gptoss120b", "GPT-OSS-120B", "s1")]  # best GEPA general seed
DS_NAME = {"mmlu_pro_mini_w_keyword": "MMLU-Pro", "gpqa_w_keyword": "GPQA",
           "hle_w_keyword": "HLE", "math500_integer": "MATH-500", "olympiads_integer": "Olympiads"}
MATH = {"math500_integer", "olympiads_integer"}
MIN_BIN_N = 30


def eval_json(d: Path) -> Path:
    fs = [p for p in d.glob("*.json") if p.name != "summary.json"]
    assert len(fs) == 1, (d, fs)
    return fs[0]


def load_runs(model: str, seed: str, lane: str) -> list[tuple[str, Path]]:
    """-> [(arm, run_dir)]; arm names shared across MCQ and MATH runs."""
    pr = REPO / "pinned_reeval/runs" / model
    cn = HERE / "runs" / lane
    runs = [("baseline", pr / "baseline_test"), ("gepa", pr / f"gepa_{seed}_test"),
            ("baseline", cn / "math500_baseline"), ("gepa", cn / "math500_gepa_mathfmt"),
            ("gepa_verbatim", cn / "math500_gepa"),
            ("uncon", cn / "uncon_math500"), ("uncon", cn / "uncon_mcq"),
            ("nocot", cn / "nocot_math500"), ("nocot", cn / "nocot_mcq"),
            ("baseline", cn / "olympiads_baseline"), ("gepa", cn / "olympiads_gepa_mathfmt"),
            ("uncon", cn / "uncon_olympiads"), ("nocot", cn / "nocot_olympiads")]
    return [(a, d) for a, d in runs if (d / "summary.json").exists()]


def extract(lane_suffix: str = "") -> pd.DataFrame:
    levels = pd.read_csv(REPO / "datasets/math500_integer.csv")["level"]
    rows = []
    for model, _, seed in MODELS:
        for arm, d in load_runs(model, seed, model + lane_suffix):
            for r in json.load(open(eval_json(d)))["results"]:
                s = r["samples"][0]
                if s.get("error"):
                    continue
                reasoning = s.get("reasoning") or ""
                rows.append({
                    "model": model, "arm": arm, "run": str(d.relative_to(REPO)),
                    "dataset": r["dataset"], "mode": r["mode"], "id": int(r["id"]),
                    "level": int(levels[int(r["id"])]) if r["dataset"] == "math500_integer" else None,
                    "reasoning_words": len(reasoning.split()),
                    "compliance": None if s.get("compliance") is None else int(s["compliance"] == 1),
                    "correct": bool(s["correct"]),
                    "truncated": s.get("finish_reason") == "length",
                })
    return pd.DataFrame(rows)


def binned(sub, bins, min_n=MIN_BIN_N):
    idx = np.digitize(sub["reasoning_words"].clip(lower=1), bins) - 1
    out = []
    for b in range(len(bins) - 1):
        m = idx == b
        n = int(m.sum())
        if n >= min_n:
            p = sub.loc[m, "compliance"].mean()
            out.append((np.sqrt(bins[b] * bins[b + 1]), p, np.sqrt(p * (1 - p) / n), n))
    return np.array(out).reshape(-1, 4)


def plot(df: pd.DataFrame, suffix: str = ""):
    set_matplotlib_style()
    ctl = df[df["arm"].isin(["baseline", "gepa"])]
    common = set(ctl.loc[ctl["dataset"] == "math500_integer", "mode"])  # 7 non-keyword modes
    ctl = ctl[ctl["mode"].isin(common)]
    bins = np.geomspace(1, ctl["reasoning_words"].max() * 1.001, 14)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.3, "wspace": 0.08})
    for i, (model, name, _) in enumerate(MODELS):
        for j, (arm, armname) in enumerate([("baseline", "Baseline (empty prompt)"),
                                            ("gepa", "Best GEPA (general) prompt")]):
            ax = axes[i, j]
            for c, (ds, dsname) in zip(PALETTE, DS_NAME.items()):
                sub = ctl[(ctl["model"] == model) & (ctl["arm"] == arm) & (ctl["dataset"] == ds)]
                pts = binned(sub, bins)
                if not len(pts):
                    continue
                cx, p, se, n = pts.T
                ax.errorbar(cx, 100 * p, yerr=100 * se, color=c, marker="o", ms=4, lw=2, capsize=2,
                            label=f"{dsname} ({100 * sub['compliance'].mean():.0f}%, "
                                  f"med {sub['reasoning_words'].median():.0f}w)")
            ax.set_xscale("log")
            ax.set_title(f"{name} — {armname}", fontsize=10)
            ax.grid(True, which="both", alpha=0.15)
            ax.legend(fontsize=7, loc="upper right", title="dataset (overall, median len)",
                      title_fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("Reasoning length (words, log)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Strict compliance (%)")
    fig.suptitle("Strict compliance vs reasoning length: CoT-Control MCQ vs MATH-500 / Olympiads "
                 f"(test, {len(common)} shared modes, bins n≥{MIN_BIN_N})", fontsize=11)
    out = OUT / f"compliance_vs_length_by_dataset{suffix}.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"saved {out}")

    # Main contrast: MCQ (GPQA+HLE+MMLU-Pro pooled) vs MATH-500, one panel per model,
    # color = task, dashed = baseline / solid = GEPA (gepa/plotting.ipynb style).
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.08})
    tasks = [("MCQ (GPQA+HLE+MMLU-Pro)", ~ctl["dataset"].isin(MATH), PALETTE[0]),
             ("MATH-500", ctl["dataset"] == "math500_integer", PALETTE[1]),
             ("Olympiads", ctl["dataset"] == "olympiads_integer", PALETTE[2])]
    for ax, (model, name, _) in zip(axes, MODELS):
        for tname, tmask, c in tasks:
            for arm, ls, armname in [("baseline", "--", "baseline"), ("gepa", "-", "GEPA")]:
                sub = ctl[tmask & (ctl["model"] == model) & (ctl["arm"] == arm)]
                pts = binned(sub, bins, min_n=50)
                if not len(pts):
                    continue
                cx, p, se, n = pts.T
                ax.errorbar(cx, 100 * p, yerr=100 * se, color=c, ls=ls, marker="o",
                            ms=4 if arm == "gepa" else 3, lw=1.6, capsize=2,
                            markerfacecolor=c if arm == "gepa" else "white",
                            label=f"{tname}, {armname} ({100 * sub['compliance'].mean():.0f}% overall, "
                                  f"median {sub['reasoning_words'].median():.0f}w)")
        ax.set_xscale("log")
        ax.set_title(name, fontsize=10)
        ax.grid(True, which="both", alpha=0.15)
        ax.set_xlabel("Reasoning length (words, log)")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    axes[0].set_ylabel("Strict compliance (%)")
    fig.suptitle("Strict compliance vs reasoning length: knowledge MCQ vs math (MATH-500, Olympiads) "
                 f"(test, {len(common)} shared modes, log word bins with n≥50)", fontsize=11, y=1.0)
    out = OUT / f"compliance_vs_length{suffix}.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"saved {out}")
    return common


def tables(df: pd.DataFrame, common) -> str:
    L = ["# CoT necessity: MCQ vs MATH-500 / Olympiads\n"]
    L.append("## Is CoT needed? Unconstrained accuracy, reasoning on vs off (mode=baseline, no Requirement)\n")
    L.append("| model | dataset | acc (reasoning on) | med words | truncated (16k) | acc (reasoning off) | n |")
    L.append("|---|---|---|---|---|---|---|")
    for model, name, _ in MODELS:
        for ds, dsname in DS_NAME.items():
            on = df[(df.model == model) & (df.arm == "uncon") & (df.dataset == ds)]
            off = df[(df.model == model) & (df.arm == "nocot") & (df.dataset == ds)]
            if not len(on):
                continue
            offs = f"{off.correct.mean():.3f}" if len(off) else "-"
            L.append(f"| {name} | {dsname} | {on.correct.mean():.3f} | {on.reasoning_words.median():.0f} "
                     f"| {on.truncated.mean():.3f} | {offs} | {len(on)} |")
    L.append(f"\n## Controllability ({len(common)} shared modes: {', '.join(sorted(common))})\n")
    L.append("acc|C / acc|NC = accuracy among compliant / non-compliant rollouts. On a task that needs "
             "CoT, complying (esp. ignore_question) should cost accuracy.\n")
    hdr = ("| model | arm | dataset | strict compliance | accuracy | acc\\|C | acc\\|NC | med words | truncated | n |\n"
           "|---|---|---|---|---|---|---|---|---|---|")
    ctl = df[df.arm.isin(["baseline", "gepa", "gepa_verbatim"]) & df["mode"].isin(common)]

    def row(keys, g):
        c, nc = g[g.compliance == 1], g[g.compliance == 0]
        f = lambda s: f"{s.correct.mean():.3f} (n={len(s)})" if len(s) else "-"
        return (f"| {' | '.join(keys)} | {g.compliance.mean():.3f} | {g.correct.mean():.3f} "
                f"| {f(c)} | {f(nc)} | {g.reasoning_words.median():.0f} | {g.truncated.mean():.3f} | {len(g)} |")

    L.append(hdr)
    for (model, arm, ds), g in ctl.groupby(["model", "arm", "dataset"]):
        L.append(row([model, arm, DS_NAME[ds]], g))
    L.append("\n## ignore_question only (the mode that directly forbids reasoning about the question)\n")
    L.append(hdr)
    for (model, arm, ds), g in ctl[ctl["mode"] == "ignore_question"].groupby(["model", "arm", "dataset"]):
        L.append(row([model, arm, DS_NAME[ds]], g))
    L.append("\n## Length- and mode-matched difference vs MCQ\n")
    L.append("Within each (mode x log-length bin) cell with >=20 rollouts on both sides: compliance(math) - "
             "compliance(MCQ), averaged with weight min(n_math, n_mcq). Removes length and mode-mix "
             "differences; 'covered' = share of the math rollouts that fall in a matched cell.\n")
    L.append("| model | arm | math task | matched diff (pp) | raw diff (pp) | cells | covered |")
    L.append("|---|---|---|---|---|---|---|")
    bins = np.geomspace(1, ctl["reasoning_words"].max() * 1.001, 14)
    cm = ctl.assign(bin=np.digitize(ctl["reasoning_words"].clip(lower=1), bins),
                    task=np.where(ctl["dataset"].isin(MATH), ctl["dataset"], "mcq"))
    for (model, arm), g in cm[cm.arm.isin(["baseline", "gepa"])].groupby(["model", "arm"]):
        for ds in sorted(MATH):
            if not (g.task == ds).any():
                continue
            cells = g[g.task.isin([ds, "mcq"])].groupby(["mode", "bin", "task"])["compliance"].agg(["mean", "size"]).unstack("task")
            cells = cells.dropna()
            cells = cells[(cells[("size", ds)] >= 20) & (cells[("size", "mcq")] >= 20)]
            if not len(cells):
                continue
            w = np.minimum(cells[("size", ds)], cells[("size", "mcq")])
            diff = np.average(cells[("mean", ds)] - cells[("mean", "mcq")], weights=w)
            raw = g[g.task == ds].compliance.mean() - g[g.task == "mcq"].compliance.mean()
            cov = cells[("size", ds)].sum() / (g.task == ds).sum()
            L.append(f"| {model} | {arm} | {DS_NAME[ds]} | {100 * diff:+.1f} | {100 * raw:+.1f} "
                     f"| {len(cells)} | {cov:.2f} |")
    L.append("\n## MATH-500 by level (gepa = math-format-adapted prompt)\n")
    L.append("| model | arm | level | strict compliance | accuracy | med words | n |")
    L.append("|---|---|---|---|---|---|---|")
    m = ctl[ctl.dataset == "math500_integer"]
    for (model, arm, lv), g in m.groupby(["model", "arm", "level"]):
        L.append(f"| {model} | {arm} | {int(lv)} | {g.compliance.mean():.3f} | {g.correct.mean():.3f} "
                 f"| {g.reasoning_words.median():.0f} | {len(g)} |")
    L.append("\n## Per mode (strict compliance)\n")
    pm = ctl.pivot_table(index=["model", "arm", "mode"], columns="dataset", values="compliance")
    pm.columns = [DS_NAME[c] for c in pm.columns]
    L.append("| model | arm | mode | " + " | ".join(pm.columns) + " |")
    L.append("|---" * (3 + len(pm.columns)) + "|")
    for idx, vals in pm.iterrows():
        L.append("| " + " | ".join(idx) + " | "
                 + " | ".join("-" if pd.isna(v) else f"{v:.3f}" for v in vals) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    suffix = sys.argv[1] if len(sys.argv) > 1 else ""  # e.g. "_smoke"
    OUT.mkdir(exist_ok=True)
    df = extract(suffix)
    df.to_parquet(OUT / f"rollouts{suffix}.parquet", index=False)
    common = plot(df, suffix)
    md = tables(df, common)
    (OUT / f"results{suffix}.md").write_text(md)
    print(md)
