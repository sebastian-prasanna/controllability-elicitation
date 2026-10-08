#!/usr/bin/env python
"""One-line status of sweep runs (for the monitoring loop).

    python rl/sweep_status.py [run-dir-glob]   # default: sweep2_*/sweep*-*
"""

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERN = sys.argv[1] if len(sys.argv) > 1 else "sweep2_*/sweep*-*"

parts = []
for d in sorted(glob.glob(str(ROOT / "rl" / "runs" / PATTERN))):
    name = Path(d).name.rsplit("-2026", 1)[0]
    p = Path(d) / "progress.jsonl"
    entries = []
    if p.exists():
        entries = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if entries:
        last = entries[-1]
        parts.append(f"{name}:it{last['iteration']} r={last['reward_mean']:.3f}")
    else:
        parts.append(f"{name}:starting")
print(" | ".join(parts) if parts else "no sweep runs yet")
