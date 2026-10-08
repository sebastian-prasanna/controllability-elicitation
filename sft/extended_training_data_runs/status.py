"""Live status of the 8 extended-training-data runs from their progress.jsonl.

    .venv/bin/python sft/extended_training_data_runs/status.py
"""
import json
from collections import Counter
from pathlib import Path

D = Path(__file__).parent
TOTAL = 13840
for run in sorted(p for p in D.iterdir() if p.is_dir()):
    pj = run / "progress.jsonl"
    done = (run / "baseline_results.json").exists()
    if not pj.exists():
        print(f"{run.name:<24} not started"); continue
    rows = [json.loads(l) for l in pj.open() if l.strip()]
    n = len(rows)
    exc = sum(r["error"] is not None for r in rows)
    fr = Counter(r["finish_reason"] for r in rows)
    prov = Counter(r["provider"] for r in rows)
    ferr_by_prov = Counter(r["provider"] for r in rows if r["finish_reason"] == "error")
    secs = rows[-1]["secs"] if rows else 0
    rate = n / secs * 3600 if secs else 0
    eta = (TOTAL - n) / rate if rate else float("inf")
    print(f"{run.name:<24} {'DONE' if done else 'run '} {n:>5}/{TOTAL} ({n/TOTAL:5.1%}) "
          f"{secs/60:6.1f}min eta {eta:4.1f}h | exc={exc} fr_error={fr.get('error',0)} "
          f"length={fr.get('length',0)} | prov={dict(prov.most_common(4))}"
          + (f" fr_error_by_prov={dict(ferr_by_prov)}" if ferr_by_prov else ""))
