#!/usr/bin/env python
"""One-line status of SFT sweep runs (for the monitoring loop).

    python sft/sweep_status.py [run-dir-glob]   # default: */sftsweep*-*

Reads each run folder's train.log (last "[step N/M] loss=" line) and, once
present, eval_summary.json (best-checkpoint compliance on the val split).
"""

import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERN = sys.argv[1] if len(sys.argv) > 1 else "*/sftsweep*-*"

STEP_RE = re.compile(r"\[step (\d+)/(\d+)\] loss=([\d.eE+-]+)")

parts = []
for d in sorted(glob.glob(str(ROOT / "sft" / "runs" / PATTERN))):
    d = Path(d)
    if not d.is_dir():
        continue
    name = d.name
    summary = d / "eval_summary.json"
    log = d / "train.log"
    if summary.exists():
        ckpts = json.loads(summary.read_text())["checkpoints"]
        best = max(ckpts, key=lambda c: c["compliance_rate"])
        parts.append(f"{name}:DONE best@{best['step']} "
                     f"c={best['compliance_rate']:.3f} a={best['accuracy']:.3f}")
    elif log.exists():
        steps = STEP_RE.findall(log.read_text())
        if steps:
            n, total, loss = steps[-1]
            parts.append(f"{name}:step{n}/{total} loss={float(loss):.3f}")
        elif "slot-retry" in log.read_text()[-2000:]:
            parts.append(f"{name}:waiting-for-slot")
        else:
            parts.append(f"{name}:starting")
    else:
        parts.append(f"{name}:no-log")
print(" | ".join(parts) if parts else "no sweep runs yet")
