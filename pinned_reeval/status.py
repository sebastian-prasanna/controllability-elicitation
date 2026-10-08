#!/usr/bin/env python3
"""Progress table for pinned_reeval lanes: per eval, rollouts done / expected, strict compliance, accuracy, meta rate."""
import json, sys
from pathlib import Path
RUNS = Path(__file__).resolve().parent / "runs"
EXPECT = {"test": 4464, "heldout": 1500}
for lane in sorted(RUNS.iterdir()):
    if not lane.is_dir(): continue
    print(f"\n== {lane.name}")
    for run in sorted(p for p in lane.iterdir() if p.is_dir() and not p.name.startswith("_")):
        prog = run / "progress.jsonl"; summ = run / "summary.json"
        n = sum(1 for _ in open(prog)) if prog.exists() else 0
        exp = EXPECT["heldout" if run.name.endswith("heldout") else "test"]
        if summ.exists():
            s = json.load(open(summ))
            print(f"  {run.name:22s} DONE  strict={s.get('compliance_rate', 0):.3f} acc={s.get('accuracy', 0):.3f} "
                  f"meta_compliant={s.get('meta_discussion_rate_compliant')} errors={s.get('n_errors')}/{s.get('n_rollouts')}")
        elif n:
            errs = sum(1 for l in open(prog) if json.loads(l).get("error"))
            print(f"  {run.name:22s} {n:5d}/{exp} rollouts  errors={errs}")
        else:
            print(f"  {run.name:22s} pending")
    dl = lane / "driver.log"
    if dl.exists():
        print("  last:", open(dl).read().strip().splitlines()[-1][:140])
