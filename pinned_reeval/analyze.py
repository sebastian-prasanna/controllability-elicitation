#!/usr/bin/env python3
"""Aggregate pinned_reeval summaries -> results table (markdown) at pinned_reeval/results.md.
Per model x arm x split: strict compliance, accuracy, meta no-narration rate (compliant rollouts);
GEPA/few-shot report mean ± sd over the 3 seeds. Incomplete cells show n_seeds."""
import json, statistics, sys
from pathlib import Path
RUNS = Path(__file__).resolve().parent / "runs"
ORDER = ["gptoss20b", "gptoss120b", "qwen8b", "qwen32b", "kimik3", "dsv4pro", "glm53", "glm53flash"]
def load(lane):
    out = {}
    for s in (RUNS / lane).glob("*/summary.json"):
        if s.parent.name.startswith("_"): continue
        d = json.load(open(s)); parts = s.parent.name.split("_")
        arm, split = parts[0], parts[-1]
        out.setdefault((arm, split), []).append((d["compliance_rate"], d["accuracy"], d.get("meta_discussion_rate_compliant"), d["n_errors"]))
    return out
def fmt(vals, i, expect):
    xs = [v[i] for v in vals if v[i] is not None]
    if not xs: return "-"
    return f"{statistics.mean(xs):.3f}" + (f" ± {statistics.stdev(xs):.3f}" if len(xs) > 1 else "") + (f" (n={len(xs)}/{expect})" if len(xs) != expect else "")
lines = ["# Pinned re-eval results (configs/eval_pins.json; T=0, top_p 1, 16k, gpt-oss effort medium, others unset; meta judge on, scope=compliant)\n",
         "Run 2026-09-22 04:06-23:08 UTC. 112 evals, 333,984 rollouts, 4 errored rollouts (glm53flash 429s before its retry increase). "
         "Provider pins: kimi moonshotai, gpt-oss groq, qwen32b deepinfra fp8, qwen8b alibaba, glm53 z-ai, dsv4pro alibaba, "
         "glm53flash sail-research fp8 for 13/14 evals -- Sail Research delisted glm-5.3-flash mid-run, so fewshot_s3_heldout ran on novita fp8. "
         "test = 496 q x 9 modes (4464); heldout = 500 q x 3 held-out modes (1500). GEPA = general arm, seeds s0/s1/s2; few-shot = k1 s1/s2/s3.\n",
         "| model | split | arm | strict compliance | accuracy | no-narration (compliant) | errors |", "|---|---|---|---|---|---|---|"]
for lane in ORDER:
    if not (RUNS / lane).is_dir(): continue
    res = load(lane)
    for split in ("test", "heldout"):
        for arm in ("baseline", "gepa", "fewshot"):
            v = res.get((arm, split))
            if not v: lines.append(f"| {lane} | {split} | {arm} | pending | | | |"); continue
            e = 1 if arm == "baseline" else 3
            lines.append(f"| {lane} | {split} | {arm} | {fmt(v,0,e)} | {fmt(v,1,e)} | {fmt(v,2,e)} | {sum(x[3] for x in v)} |")
Path(__file__).resolve().parent.joinpath("results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
