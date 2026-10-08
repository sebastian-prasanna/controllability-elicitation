"""Extract per-rollout (reasoning length, compliance) tuples from the big eval JSONs.

Builds a compact parquet cache so fewshot/plotting.ipynb doesn't need to re-read
the rollout files (~100s of MB each) on every run.

Covers the final_k1_test sweep (3 random-demo prompt seeds per model, arm
"fewshot") plus the empty-prompt baselines (single runs, seed "-").
"""
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

MODELS = ["kimik3", "gptoss120b", "qwen32b", "dsv4pro",
          "gptoss20b", "qwen8b", "glm53", "glm53flash"]
SEEDS = ["s1", "s2", "s3"]

OUT = REPO / "fewshot/analysis/length_compliance_cache.parquet"


def eval_file(d: Path) -> Path:
    files = [p for p in d.glob("*.json")
             if p.name not in ("baseline_results.json", "summary.json")]
    assert files, d
    return sorted(files)[-1]  # newest timestamped eval JSON


rows = []
for key in MODELS:
    dirs = [("baseline", "-", REPO / "baselines" / key)]
    dirs += [("fewshot", s, REPO / "fewshot/runs/final_k1_test" / f"{key}_{s}")
             for s in SEEDS]
    for arm, seed, d in dirs:
        f = eval_file(d)
        with open(f) as fh:
            results = json.load(fh)["results"]
        for r in results:
            s = r["samples"][0]
            if s.get("error"):
                continue
            reasoning = s.get("reasoning") or ""
            rows.append({
                "model": key,
                "arm": arm,
                "seed": seed,
                "mode": r["mode"],
                "dataset": r["dataset"],
                "reasoning_chars": len(reasoning),
                "reasoning_words": len(reasoning.split()),
                "compliance": int(s["compliance"] == 1),
                "correct": bool(s["correct"]),
            })
        print(f"{key:<12} {arm:<9} {seed:<3} {len(results):>5} rollouts  ({f.name})", flush=True)

df = pd.DataFrame(rows)
df.to_parquet(OUT, index=False)
print(f"\nsaved {len(df)} rows -> {OUT}")
