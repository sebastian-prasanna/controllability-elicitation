"""Build canonical train/val/test splits over hle + gpqa + mmlu_pro.

Writes datasets/splits.json: {<csv stem>: {"train": [ids], "val": [ids],
"test": [ids]}} where ids are CSV row indices (the `id` field produced by
cotcontrol.eval.data). Totals across the three datasets: 514 train / 200 val /
500 test, allocated to each dataset proportionally to its size and stratified
by domain within each dataset (largest-remainder rounding).

Deterministic (fixed seed). Re-running only changes the file if the CSVs
change — the split ids are tied to CSV row order, so never reorder the CSVs.

    python scripts/build_splits.py
"""

import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from cotcontrol.eval.data import DATASET_ALIASES, DATASETS_DIR  # noqa: E402

SEED = 42
DATASETS = ["hle", "gpqa", "mmlu_pro"]
TOTALS = {"test": 500, "val": 200}  # train = remainder (514)


def largest_remainder(fracs: dict[str, float], total: int, caps: dict[str, int]) -> dict[str, int]:
    """Integer allocation of `total` proportional to fracs, capped per key."""
    quota = {k: min(caps[k], int(f)) for k, f in fracs.items()}
    remainders = sorted(fracs, key=lambda k: fracs[k] - int(fracs[k]), reverse=True)
    i = 0
    while sum(quota.values()) < total:
        k = remainders[i % len(remainders)]
        if quota[k] < caps[k]:
            quota[k] += 1
        i += 1
    return quota


def main():
    rng = random.Random(SEED)
    sizes = {}
    domains_by_ds = {}
    for name in DATASETS:
        df = pd.read_csv(DATASET_ALIASES[name])
        sizes[name] = len(df)
        by_domain: dict[str, list[int]] = {}
        for idx, dom in df["domain"].astype(str).items():
            by_domain.setdefault(dom, []).append(int(idx))
        for ids in by_domain.values():
            rng.shuffle(ids)
        domains_by_ds[name] = by_domain

    n_total = sum(sizes.values())
    # Per-dataset targets, proportional to dataset size (largest remainder).
    ds_targets = {
        split: largest_remainder(
            {name: TOTALS[split] * sizes[name] / n_total for name in DATASETS},
            TOTALS[split],
            sizes,
        )
        for split in ("test", "val")
    }

    splits = {}
    for name in DATASETS:
        by_domain = domains_by_ds[name]
        n_ds = sizes[name]
        caps = {d: len(ids) for d, ids in by_domain.items()}
        test_q = largest_remainder(
            {d: ds_targets["test"][name] * caps[d] / n_ds for d in by_domain},
            ds_targets["test"][name],
            caps,
        )
        val_caps = {d: caps[d] - test_q[d] for d in by_domain}
        val_q = largest_remainder(
            {d: ds_targets["val"][name] * caps[d] / n_ds for d in by_domain},
            ds_targets["val"][name],
            val_caps,
        )
        ds_split = {"train": [], "val": [], "test": []}
        for d, ids in by_domain.items():
            ds_split["test"] += ids[: test_q[d]]
            ds_split["val"] += ids[test_q[d] : test_q[d] + val_q[d]]
            ds_split["train"] += ids[test_q[d] + val_q[d] :]
        stem = DATASET_ALIASES[name].stem
        splits[stem] = {k: sorted(v) for k, v in ds_split.items()}
        print(f"{name} ({stem}): " + ", ".join(f"{k}={len(v)}" for k, v in splits[stem].items()))

    out = DATASETS_DIR / "splits.json"
    out.write_text(json.dumps({"seed": SEED, "splits": splits}, indent=1))
    totals = {k: sum(len(s[k]) for s in splits.values()) for k in ("train", "val", "test")}
    print(f"totals: {totals} -> {out}")


if __name__ == "__main__":
    main()
