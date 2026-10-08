"""Download a fixed, seeded sample of allenai/c4 (en, train) as {"text": ...}
jsonl for mixing into SDF training (the paper mixes 1:1 with synthetic docs).

Usage: .venv/bin/python sdf/download_c4.py [--n 50000] [--out sdf/data/c4_50k.jsonl]
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=50_000)
ap.add_argument("--out", default="sdf/data/c4_50k.jsonl")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--min-chars", type=int, default=200)
ap.add_argument("--max-chars", type=int, default=12_000)
args = ap.parse_args()

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
ds = ds.shuffle(buffer_size=100_000, seed=args.seed)
n = 0
with open(out, "w") as f:
    for row in ds:
        t = row["text"]
        if not (args.min_chars <= len(t) <= args.max_chars):
            continue
        f.write(json.dumps({"text": t, "meta": {"source": "allenai/c4", "url": row["url"]}}) + "\n")
        n += 1
        if n % 5000 == 0:
            print(f"{n}/{args.n}", flush=True)
        if n >= args.n:
            break
print(f"wrote {n} rows to {out}")
