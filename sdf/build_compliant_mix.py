"""Stratified samples of COMPLIANT chat demonstrations (9 main modes only, so the
extended_unseen transfer block stays clean) from the sft x320 datasets, for
mixing into SDF midtraining (vs. the unconstrained "anchor" rows, which erased
the SDF effect). Picks the x320 file with more unique source rows on these
modes; prefers distinct source rows over upsampled copies.

  .venv/bin/python sdf/build_compliant_mix.py  -> sdf/data/compliant{180,900}.jsonl
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_MODES = ["uppercase_thinking", "lowercase_thinking", "alternating_case", "meow_between_words",
              "end_of_sentence", "repeat_sentences", "word_suppression", "multiple_word_suppression",
              "ignore_question"]
CANDIDATES = ["gptoss20b_gepa_general_x320", "gptoss20b_fewshot_k1_x320"]


def load(name):
    by_mode = defaultdict(list)
    for l in open(ROOT / f"sft/datasets/{name}.jsonl"):
        r = json.loads(l)
        m = r["meta"]["mode"]
        if m in MAIN_MODES and r["meta"].get("upsample", {}).get("copy_index", 0) == 0:
            by_mode[m].append(r)
    return by_mode


stats = {}
for name in CANDIDATES:
    bm = load(name)
    stats[name] = {m: len(bm[m]) for m in MAIN_MODES}
    print(name, "unique rows per mode:", stats[name])
best = max(CANDIDATES, key=lambda n: (min(stats[n].values()), sum(stats[n].values())))
print("using", best)
by_mode = load(best)
rng = random.Random(0)
for m in MAIN_MODES:
    rng.shuffle(by_mode[m])
for per_mode in (20, 100):
    rows = []
    for m in MAIN_MODES:
        pool = by_mode[m]
        take = [pool[i % len(pool)] for i in range(per_mode)]  # wraps only if a mode has < per_mode unique rows
        for r in take:
            r = dict(r)
            r["meta"] = {**r["meta"], "source": f"compliant:{best}"}
            rows.append(r)
    rng.shuffle(rows)
    out = ROOT / f"sdf/data/compliant{per_mode * 9}.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows), "->", out, "| all correct:", all(r["meta"].get("correct") for r in rows))
