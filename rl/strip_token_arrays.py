#!/usr/bin/env python
"""Post-hoc strip of token_ids/token_logprobs from historical RL run files.

    python rl/strip_token_arrays.py <run-or-sweep-dir> [...] [--workers N]

Applies the same transform the driver/eval now apply at save time (2026-09-01):
- iters/iter-*.json: drop per-sample token_ids/token_logprobs and per-record
  prompt_token_ids; add n_tokens per sample.
- batches/batch-*.json: same for samples; group prompt_token_ids -> its length.

Everything else (text, grading, rewards, judge output) is preserved verbatim.
Atomic per file (temp + os.replace), idempotent (already-slim files are
skipped), parallel. Do NOT run on a live run's folder — the driver overwrites
these files mid-iteration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _slim_sample(s: dict) -> dict:
    out = {k: v for k, v in s.items() if k not in ("token_ids", "token_logprobs")}
    if "token_ids" in s:
        out["n_tokens"] = len(s["token_ids"])
    return out


def transform_file(path_str: str) -> tuple[int, int, bool]:
    """Returns (bytes_before, bytes_after, changed)."""
    path = Path(path_str)
    before = path.stat().st_size
    data = json.loads(path.read_text())
    changed = False

    if "results" in data:  # iters file
        recs = []
        for rec in data["results"]:
            if "prompt_token_ids" in rec or any("token_ids" in s for s in rec["samples"]):
                changed = True
            rec = {k: v for k, v in rec.items() if k != "prompt_token_ids"}
            rec["samples"] = [_slim_sample(s) for s in rec["samples"]]
            recs.append(rec)
        data["results"] = recs
        dump = json.dumps(data, indent=1)
    elif "groups" in data:  # batch file
        groups = []
        for g in data["groups"]:
            if isinstance(g.get("prompt_token_ids"), list) or any(
                    "token_ids" in s for s in g["samples"]):
                changed = True
            g = dict(g)
            if isinstance(g.get("prompt_token_ids"), list):
                g["prompt_token_ids"] = len(g["prompt_token_ids"])
            g["samples"] = [_slim_sample(s) for s in g["samples"]]
            groups.append(g)
        data["groups"] = groups
        dump = json.dumps(data)
    else:
        return before, before, False

    if not changed:
        return before, before, False
    tmp = path.with_suffix(".json.tmp-strip")
    tmp.write_text(dump)
    os.replace(tmp, path)
    return before, tmp.stat().st_size if tmp.exists() else path.stat().st_size, True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    files = []
    for d in args.dirs:
        files += [str(p) for p in d.rglob("iters/iter-*.json")]
        files += [str(p) for p in d.rglob("batches/batch-*.json")]
    print(f"{len(files)} files under {len(args.dirs)} dirs")
    total_b = total_a = n_changed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (b, a, ch) in enumerate(ex.map(transform_file, files, chunksize=8)):
            total_b += b; total_a += a; n_changed += ch
            if i % 500 == 0:
                print(f"  {i}/{len(files)}  {total_b/1e9:.1f}GB -> {total_a/1e9:.1f}GB", flush=True)
    print(f"done: {n_changed}/{len(files)} files changed, "
          f"{total_b/1e9:.1f}GB -> {total_a/1e9:.1f}GB "
          f"(saved {(total_b-total_a)/1e9:.1f}GB)")


if __name__ == "__main__":
    main()
