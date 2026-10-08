#!/usr/bin/env python3
"""Per-mode STRICT compliance overlap between the GEPA-general arm and the few-shot k1 arm.

Reads pinned_reeval/runs/<model>/<arm>_<seed>_<split>/summary.json, whose `per_mode`
block holds {n, correct, compliant} (compliant == strict compliance, matches
summary["compliance_rate"] when pooled). GEPA = mean over s0/s1/s2, few-shot = mean
over s1/s2/s3, baseline = single run.

Writes scratch/gepa_fewshot_mode_overlap.md.
"""
import json, math, statistics
from pathlib import Path

ROOT = Path("/root/controllability-elicitation")
RUNS = ROOT / "pinned_reeval" / "runs"
OUT = ROOT / "scratch" / "gepa_fewshot_mode_overlap.md"
MODELS = ["gptoss20b", "gptoss120b", "qwen8b", "qwen32b", "kimik3", "dsv4pro", "glm53", "glm53flash"]
SOLVE, HI, LO = 0.5, 0.3, 0.1


def load(model, split):
    """-> {arm: {mode: [rate per seed]}} and mode order."""
    arms, order = {}, []
    for s in sorted((RUNS / model).glob(f"*_{split}/summary.json")):
        name = s.parent.name
        arm = name.split("_")[0]
        d = json.load(open(s))
        for mode, v in d["per_mode"].items():
            if mode not in order:
                order.append(mode)
            arms.setdefault(arm, {}).setdefault(mode, []).append(v["compliant"] / v["n"])
    return arms, order


def mean(xs):
    return sum(xs) / len(xs)


def pearson(x, y):
    n = len(x)
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def analyse(split, label):
    lines = [f"\n## {label} split\n"]
    per_model = {}
    for m in MODELS:
        arms, order = load(m, split)
        base = {k: mean(v) for k, v in arms["baseline"].items()}
        gepa = {k: mean(v) for k, v in arms["gepa"].items()}
        few = {k: mean(v) for k, v in arms["fewshot"].items()}
        nseeds = (len(next(iter(arms["gepa"].values()))), len(next(iter(arms["fewshot"].values()))))
        per_model[m] = (order, base, gepa, few)
        lines += [f"\n### {m}  (gepa n_seeds={nseeds[0]}, fewshot n_seeds={nseeds[1]})\n",
                  "| mode | baseline | gepa (mean) | fewshot (mean) | gepa-fewshot |", "|---|---|---|---|---|"]
        for mode in order:
            lines.append(f"| {mode} | {base[mode]:.3f} | {gepa[mode]:.3f} | {few[mode]:.3f} | {gepa[mode]-few[mode]:+.3f} |")
        lines.append(f"| **pooled** | {mean([base[k] for k in order]):.3f} | {mean([gepa[k] for k in order]):.3f} | "
                     f"{mean([few[k] for k in order]):.3f} | |")

    # 1. solved sets
    lines += [f"\n### Solved modes (strict compliance > {SOLVE}, arm mean)\n",
              "| model | gepa only | fewshot only | both | neither |", "|---|---|---|---|---|"]
    for m in MODELS:
        order, base, gepa, few = per_model[m]
        g = {k for k in order if gepa[k] > SOLVE}
        f = {k for k in order if few[k] > SOLVE}
        cell = lambda s: ", ".join(sorted(s)) if s else "-"
        lines.append(f"| {m} | {cell(g - f)} | {cell(f - g)} | {cell(g & f)} | {len(set(order) - g - f)} modes |")

    # 2. complementary cells
    comp = []
    for m in MODELS:
        order, base, gepa, few = per_model[m]
        for mode in order:
            g, f = gepa[mode], few[mode]
            if g > HI and f < LO:
                comp.append((m, mode, g, f, "gepa"))
            elif f > HI and g < LO:
                comp.append((m, mode, g, f, "fewshot"))
    lines += [f"\n### Complementary cells (one arm > {HI}, other < {LO})\n",
              f"**Count: {len(comp)}** of {sum(len(per_model[m][0]) for m in MODELS)} model x mode cells "
              f"(gepa-only {sum(1 for c in comp if c[4]=='gepa')}, fewshot-only {sum(1 for c in comp if c[4]=='fewshot')})\n",
              "| model | mode | gepa | fewshot | winner | gap |", "|---|---|---|---|---|---|"]
    for m, mode, g, f, w in sorted(comp, key=lambda c: -abs(c[2] - c[3])):
        lines.append(f"| {m} | {mode} | {g:.3f} | {f:.3f} | {w} | {abs(g-f):.3f} |")

    # 3. correlations
    # 2b. large-gap cells (looser view, threshold-insensitive)
    big = []
    for m in MODELS:
        order, base, gepa, few = per_model[m]
        for mode in order:
            g, f = gepa[mode], few[mode]
            if abs(g - f) > 0.25:
                big.append((m, mode, g, f))
    lines += ["\n### Large-gap cells (|gepa - fewshot| > 0.25; threshold-insensitive view)\n",
              f"**Count: {len(big)}**\n",
              "| model | mode | gepa | fewshot | winner | gap |", "|---|---|---|---|---|---|"]
    for m, mode, g, f in sorted(big, key=lambda c: -abs(c[2] - c[3])):
        lines.append(f"| {m} | {mode} | {g:.3f} | {f:.3f} | {'gepa' if g>f else 'fewshot'} | {abs(g-f):.3f} |")

    lines += ["\n### Per-mode correlation between arms (Pearson r over modes)\n",
              "| model | r | n modes | note |", "|---|---|---|---|"]
    pooled_g, pooled_f, pooled_z = [], [], []
    for m in MODELS:
        order, base, gepa, few = per_model[m]
        gv = [gepa[k] for k in order]
        fv = [few[k] for k in order]
        r = pearson(gv, fv)
        lines.append(f"| {m} | {'n/a' if r is None else f'{r:+.3f}'} | {len(order)} | "
                     f"{'zero variance in one arm' if r is None else ''} |")
        pooled_g += gv
        pooled_f += fv
        # z-scored within model for a fair pooled correlation
        if r is not None:
            sg, sf = statistics.pstdev(gv), statistics.pstdev(fv)
            mg, mf = mean(gv), mean(fv)
            pooled_z.append(([(x - mg) / sg for x in gv], [(x - mf) / sf for x in fv]))
    rp = pearson(pooled_g, pooled_f)
    zg = [x for p in pooled_z for x in p[0]]
    zf = [x for p in pooled_z for x in p[1]]
    lines.append(f"| **pooled (raw, all cells)** | {rp:+.3f} | {len(pooled_g)} | |")
    lines.append(f"| **pooled (within-model z-scored)** | {pearson(zg, zf):+.3f} | {len(zg)} | removes model-level scale |")
    return lines


out = ["# GEPA-general vs few-shot k1: per-mode strict-compliance overlap",
       "",
       "Source: `pinned_reeval/runs/<model>/<arm>_<seed>_<split>/summary.json` -> `per_mode[mode].compliant / n`.",
       "GEPA = mean over seeds s0/s1/s2 (general arm); fewshot = mean over seeds s1/s2/s3 (k=1); baseline = single run.",
       "Strict compliance only (the same quantity as `compliance_rate` in the summaries).",
       ""]
out += analyse("test", "TEST (9 modes)")
out += analyse("heldout", "HELD-OUT (3 modes)")
OUT.write_text("\n".join(out) + "\n")
print("\n".join(out))
