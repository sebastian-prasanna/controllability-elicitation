"""Summary table over the 8 runs: overall + positives (compliant & correct) + per-mode strict.

    .venv/bin/python sft/extended_training_data_runs/summarize.py > sft/extended_training_data_runs/results.md
"""
import json, glob
from pathlib import Path
from collections import Counter

D = Path(__file__).parent
runs = sorted(p for p in D.iterdir() if p.is_dir() and (p / "baseline_results.json").exists())
rows, per_mode = [], {}
for run in runs:
    br = json.load(open(run / "baseline_results.json"))
    ev = json.load(open([f for f in glob.glob(str(run / "*_all_all.json"))][0]))
    fr = Counter(s["finish_reason"] for r in ev["results"] for s in r["samples"])
    pos = sum(1 for r in ev["results"] for s in r["samples"] if s["compliance"] == 1 and s["correct"])
    comp = sum(1 for r in ev["results"] for s in r["samples"] if s["compliance"] == 1)
    o = br["overall"]
    rows.append((run.name, o["n"], o["strict_compliance"], o["shaped_compliance"], o["accuracy"],
                 comp, pos, fr.get("length", 0), fr.get("error", 0) + sum(1 for r in ev["results"] for s in r["samples"] if s["error"])))
    per_mode[run.name] = {m: v["strict_compliance"] for m, v in br["per_mode"].items()}

print("# extended_training_data_runs — 27-mode train split (514 q), 2026-09-04\n")
print("| run | n | strict | shaped | acc | compliant | compliant&correct | truncated | errors |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} | {r[5]} | {r[6]} | {r[7]} | {r[8]} |")
modes = sorted({m for pm in per_mode.values() for m in pm})
names = [r[0] for r in rows]
print("\n## Per-mode strict compliance\n")
print("| mode | " + " | ".join(n.replace("_gepa_general", " G").replace("_fewshot_k1", " F") for n in names) + " |")
print("|---|" + "---|" * len(names))
for m in modes:
    print(f"| {m} | " + " | ".join(f"{per_mode[n].get(m, float('nan')):.2f}" for n in names) + " |")
