"""Build a same-distribution HELD-OUT set for an x320 SFT dataset from the source
rows that build_training_set.py did NOT emit (modes with >320 unique rows have
leftovers). Applies the identical filter + output normalization, then drops every
(mode, dataset, id) key present in the training file.

    python sft/edl/build_heldout.py sft/datasets/gptoss120b_gepa_general_x320.jsonl \
        --cap 40 --out sft/runs/<sweep>/edl/heldout.jsonl
"""
from __future__ import annotations
import argparse, json, random, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # sft/
from build_training_set import load_mode, normalize_output, _answer_letter, _reasoning_and_text  # noqa: E402
from filters import row_problems  # noqa: E402


def key(r):
    m = r["meta"]
    return (m["mode"], m["dataset"], m["id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_jsonl")
    ap.add_argument("--cap", type=int, default=40, help="max held-out rows per mode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--family", default="gptoss")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    manifest = json.load(open(a.train_jsonl + ".manifest.json"))
    src = Path(manifest["source_dir"])
    train_rows = [json.loads(l) for l in open(a.train_jsonl)]
    train_keys = {key(r) for r in train_rows}
    modes = sorted({r["meta"]["mode"] for r in train_rows})
    rng = random.Random(a.seed)
    out, report = [], {}
    for mode in modes:
        rows = load_mode(src, mode)
        kept = []
        for r in rows:
            reasoning, text = _reasoning_and_text(r)
            problems = row_problems(reasoning, text, None, family=a.family,
                                    prompt=r["input"][-1]["content"],
                                    max_tokens=manifest["max_tokens"], require_answer_only=False)
            if _answer_letter(text) not in set("ABCDEFGHIJ"):
                problems.append("no_valid_answer_letter")
            if "</think>" in reasoning or "<think>" in reasoning:
                problems.append("think_tag_in_reasoning")
            if problems or key(r) in train_keys:
                continue
            kept.append(r)
        rng.shuffle(kept)
        kept = kept[: a.cap]
        for r in kept:
            normalize_output(r)
            r["meta"]["heldout"] = True
        report[mode] = {"leftover": len(kept)}
        out.extend(kept)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} held-out rows -> {a.out}")
    print({m: v["leftover"] for m, v in report.items()})
    assert not ({key(r) for r in out} & train_keys)


if __name__ == "__main__":
    main()
