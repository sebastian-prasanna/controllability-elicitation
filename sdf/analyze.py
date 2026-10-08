"""Summarize SDF training runs: per run x eval block x checkpoint compliance /
accuracy (from <run>/eval_summary.json) and belief scores (belief_summary.json).

Usage: .venv/bin/python sdf/analyze.py [sdf/runs/train] [--md sdf/runs/train/results.md]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sdf.make_configs import FAR_MODES, NEAR_MODES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("root", nargs="?", default="sdf/runs/train")
ap.add_argument("--md", default=None)
args = ap.parse_args()

lines = []
for run in sorted(Path(args.root).glob("sdf-*")):
    es = run / "eval_summary.json"
    if not es.exists():
        lines.append(f"\n## {run.name}: (no eval_summary.json yet)")
        continue
    summ = json.loads(es.read_text())
    blocks = summ.get("evals", {})
    steps = sorted({int(c["step"]) for b in blocks.values() for c in b["checkpoints"]})
    lines.append(f"\n## {run.name}")
    header = "| block | metric | " + " | ".join(f"s{s}" for s in steps) + " |"
    lines += [header, "|" + "---|" * (len(steps) + 2)]
    for bname, b in blocks.items():
        by = {int(c["step"]): c for c in b["checkpoints"]}
        for metric in ("compliance_rate", "accuracy"):
            if bname.startswith("baseline") and metric == "compliance_rate":
                continue
            lines.append(f"| {bname} | {metric} | " + " | ".join(
                f"{by[s][metric]:.3f}" if s in by and by[s][metric] is not None else "" for s in steps) + " |")
        # answer-format retention: fraction of rollouts whose output contains 'ANSWER:'
        fr = {}
        for s in steps:
            f = run / "eval" / bname / f"checkpoint-{s}.json"
            if f.exists():
                res = json.loads(f.read_text())["results"]
                n = sum(len(x["samples"]) for x in res)
                fr[s] = sum("ANSWER:" in (smp.get("output") or "") for x in res for smp in x["samples"]) / max(1, n)
        lines.append(f"| {bname} | answer_fmt | " + " | ".join(f"{fr[s]:.2f}" if s in fr else "" for s in steps) + " |")
        if bname.startswith("extended_unseen"):
            for label, modes in (("far", FAR_MODES), ("near", NEAR_MODES)):
                vals = {}
                for s in steps:
                    if s in by:
                        pm = by[s]["per_mode"]
                        c = sum(pm[m]["compliant"] for m in modes if m in pm)
                        n = sum(pm[m]["n"] for m in modes if m in pm)
                        vals[s] = c / n if n else float("nan")
                lines.append(f"| {bname} | {label}_compliance | " + " | ".join(
                    f"{vals[s]:.3f}" if s in vals else "" for s in steps) + " |")
    bs = run / "belief_summary.json"
    if bs.exists():
        bel = json.loads(bs.read_text())
        for bname, per_step in bel.items():
            for cat in ("self", "recall", "mcq", "predict", "all"):
                lines.append(f"| belief:{bname} | {cat} | " + " | ".join(
                    f"{per_step[str(s)][cat]:.2f}" if str(s) in per_step and cat in per_step[str(s)] else ""
                    for s in steps) + " |")
text = "\n".join(lines)
print(text)
if args.md:
    Path(args.md).write_text(text + "\n")
