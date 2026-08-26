#!/usr/bin/env python
"""One-line status of all sweep runs (for the monitoring loop)."""

import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

parts = []
for d in sorted(glob.glob(str(ROOT / "rl" / "runs" / "sweep-qwen8b-*"))):
    name = Path(d).name.replace("sweep-qwen8b-", "").rsplit("-2026", 1)[0]
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
