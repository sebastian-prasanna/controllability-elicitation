#!/usr/bin/env python3
"""Audit which OpenRouter providers served the samples in GEPA eval output JSONs.

Finds eval-output JSONs under gepa/runs (files in *_eval/ dirs plus any .json
over 1MB), verifies they have the {"config", "summary", "results": [{"samples":
[{"raw_response": {...}}]}]} structure, and tallies raw_response["provider"] and
raw_response["model"] per file. progress.jsonl / iterations.jsonl are skipped
(verified to not contain raw_response; the summary JSON covers the same eval).

Writes the full tally to gepa/provider_audit.json and prints a per-file table.
"""

import json
import sys
from collections import Counter
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"
OUT = Path(__file__).resolve().parent / "provider_audit.json"
SIZE_THRESHOLD = 1 * 1024 * 1024  # 1MB


def find_candidates() -> list[Path]:
    seen: set[Path] = set()
    for p in RUNS.rglob("*.json"):
        if not p.is_file():
            continue
        in_eval_dir = p.parent.name.endswith("_eval")
        big = p.stat().st_size > SIZE_THRESHOLD
        if in_eval_dir or big:
            seen.add(p)
    return sorted(seen)


def looks_like_eval(data) -> bool:
    return (
        isinstance(data, dict)
        and isinstance(data.get("results"), list)
        and any(
            isinstance(r, dict) and isinstance(r.get("samples"), list)
            for r in data["results"]
        )
    )


def tally_file(path: Path) -> dict | None:
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, MemoryError) as e:
        print(f"  [WARN] failed to parse {path}: {e}", file=sys.stderr)
        return None
    if not looks_like_eval(data):
        return None

    providers: Counter = Counter()
    models: Counter = Counter()
    total = 0
    no_provider = 0
    for result in data["results"]:
        if not isinstance(result, dict):
            continue
        for sample in result.get("samples") or []:
            total += 1
            rr = sample.get("raw_response") if isinstance(sample, dict) else None
            if not isinstance(rr, dict):
                no_provider += 1
                continue
            prov = rr.get("provider")
            if prov is None:
                no_provider += 1
            else:
                providers[str(prov)] += 1
            mdl = rr.get("model")
            if mdl is not None:
                models[str(mdl)] += 1

    cfg = data.get("config") or {}
    return {
        "file": str(path.relative_to(RUNS)),
        "config_model": cfg.get("model") if isinstance(cfg, dict) else None,
        "total_samples": total,
        "providers": dict(providers.most_common()),
        "models": dict(models.most_common()),
        "no_provider": no_provider,
    }


def main() -> None:
    candidates = find_candidates()
    print(f"Found {len(candidates)} candidate .json files under {RUNS}\n")

    entries = []
    skipped = []
    for path in candidates:
        entry = tally_file(path)
        if entry is None:
            skipped.append(str(path.relative_to(RUNS)))
            continue
        entries.append(entry)
        prov_str = json.dumps(entry["providers"])
        print(
            f"{entry['file']}\n"
            f"    model={entry['config_model']}  samples={entry['total_samples']}  "
            f"providers={prov_str}  no_provider={entry['no_provider']}"
        )

    print(f"\n{len(entries)} eval files tallied; {len(skipped)} candidates skipped "
          f"(not eval-output structure):")
    for s in skipped:
        print(f"  skipped: {s}")

    OUT.write_text(json.dumps({"files": entries, "skipped": skipped}, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
