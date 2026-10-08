"""Extract per-rollout (reasoning length, compliance) tuples from the big eval JSONs.

Builds a compact parquet cache so the plotting notebook doesn't need to re-read
the rollout files (~100s of MB each) on every run.

Covers all three GEPA seeds per (model, arm): s0 = initial_sweep (GepaConfig
default seeds), s1/s2 = second_sweep seed arms. Baselines are single runs and
get seed "-".
"""
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

MODELS = ["kimik3", "gptoss20b", "gptoss120b", "glm53",
          "dsv4pro", "qwen32b", "glm53flash", "qwen8b"]

OUT = REPO / "gepa/analysis/length_compliance_cache.parquet"


def eval_file(d: Path) -> Path:
    files = [p for p in d.glob("*.json")
             if p.name not in ("baseline_results.json", "test_results.json")]
    assert len(files) == 1, (d, files)
    return files[0]


rows = []
for key in MODELS:
    dirs = [("baseline", "-", REPO / "baselines" / key)]
    for arm in ("free", "general"):
        dirs += [
            (arm, "s0", REPO / "gepa/runs/initial_sweep" / f"{key}_{arm}/test_eval"),
            (arm, "s1", REPO / "gepa/runs/second_sweep" / f"{key}_{arm}_s1/test_eval"),
            (arm, "s2", REPO / "gepa/runs/second_sweep" / f"{key}_{arm}_s2/test_eval"),
        ]
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
