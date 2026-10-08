"""Per-rollout (reasoning length, compliance) cache for the orange-training runs.

Mirrors gepa/analysis/extract_length_compliance.py so the same binned
compliance-vs-length analysis applies here. One row per non-errored rollout
across every run / eval block / checkpoint.

    .venv/bin/python orange-training/analysis/extract_length_compliance.py

Writes orange-training/analysis/length_compliance_cache.parquet with columns:
    model      qwen8b | gptoss20b | qwen36_27b
    cond       persona | direct | stated
    seed       0 | 1 | 2
    step       0 | 10 | ... | 50   (0 = untrained, zero-init adapter)
    mode       CoT-control mode (currently lowercase_thinking)
    reasoning_words / reasoning_chars
    compliance / correct  (bool)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
OUT = HERE / "length_compliance_cache.parquet"


def main() -> None:
    rows = []
    for path in sorted(RUNS.glob("*/eval/*/checkpoint-*.json")):
        run, _, block, ckpt = path.parts[-4:]
        m = re.match(r"orange-(.+)-seed(\d+)$", run)
        if not m:
            continue
        model, seed = m.group(1), int(m.group(2))
        cond, _, mode = block.partition("_")
        step = int(ckpt.replace("checkpoint-", "").replace(".json", ""))
        data = json.loads(path.read_text())
        for rec in data["results"]:
            for s in rec["samples"]:
                if s["error"]:
                    continue
                text = s["reasoning_text_graded"] or ""
                rows.append({
                    "model": model, "cond": cond, "seed": seed, "step": step,
                    "mode": mode,
                    "reasoning_words": len(text.split()),
                    "reasoning_chars": len(text),
                    "compliance": s["compliance"] == 1,
                    "correct": s["correct"] is True,
                })
    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    print(f"wrote {len(df):,} rollouts -> {OUT}")
    print(df.groupby(["model", "cond"]).size().to_string())


if __name__ == "__main__":
    main()
