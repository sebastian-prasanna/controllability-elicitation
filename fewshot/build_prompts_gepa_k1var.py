"""5 random-seed k1 prompt variants from the GEPA positives pool (gpt-oss-120b).

Same eligibility as build_prompts_gepa.py (genuine compliant, LEN_BAND, non-test,
mild narration filter) but demos are sampled uniformly at random per mode instead
of rank-selected, one variant per seed. For measuring demo-choice variance at k=1.

Outputs: fewshot/prompts_gepa/gptoss120b/k1_s{SEED}.txt (+ demos_k1var.json)
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fewshot"))

from build_prompts_gepa import (DS_ALIAS, HEADER, LEN_BAND, NARR, OUT_DIR,
                                format_example, harvest)
from cotcontrol.eval.data import CONSTRAINT_MODES, load_dataset
from cotcontrol.eval.prompts import format_choices

SEEDS = (1, 2, 3, 4, 5)


def main():
    best, runs = harvest()

    rows = {}
    for short in ("gpqa", "hle", "mmlu_pro"):
        for s in load_dataset(short, mode="all"):
            rows[(short, s["id"])] = s
    splits = json.loads((ROOT / "datasets" / "splits.json").read_text())["splits"]
    test_keys = {(DS_ALIAS[stem], i) for stem, sp in splits.items() for i in sp["test"]}

    # eligible[mode] = fully-hydrated candidate demos (same floors as the ranked build)
    eligible = defaultdict(list)
    for t in best.values():
        r = t.get("reasoning") or ""
        if not (LEN_BAND[0] <= len(r) <= LEN_BAND[1]) or NARR.search(r):
            continue
        ds_full, qid, _ = t["key"].split(":")
        qkey = (DS_ALIAS[ds_full], int(qid))
        if qkey in test_keys:
            continue
        row = rows.get(qkey)
        if row is None or row["question"].strip() != (t.get("question") or "").strip():
            continue
        choices_text, fi = format_choices(row.get("options"))
        user_prompt = (f"Question: {row['question']}{choices_text}{fi}"
                       f"\n\nRequirement: {t['requirement']}")
        eligible[t["mode"]].append(dict(
            user_prompt=user_prompt, reasoning=r, output=t["output"], _qkey=qkey,
            meta=dict(key=t["key"], run=t["_run"], correct=t.get("correct") is True,
                      n_reasoning_chars=len(r))))
    print({m: len(v) for m, v in sorted(eligible.items())})

    all_variants = {}
    for seed in SEEDS:
        rng = random.Random(seed)
        used_q, selected = set(), {}
        for mode in sorted(CONSTRAINT_MODES, key=lambda m: len(eligible[m])):
            cands = [e for e in eligible[mode] if e["_qkey"] not in used_q]
            pick = rng.choice(cands)
            used_q.add(pick["_qkey"])
            selected[mode] = pick
        blocks = [format_example(i + 1, selected[m])
                  for i, m in enumerate(CONSTRAINT_MODES)]
        txt = HEADER + "\n\n" + "\n\n".join(blocks)
        (OUT_DIR / f"k1_s{seed}.txt").write_text(txt)
        all_variants[f"s{seed}"] = {m: {k: v for k, v in e.items() if k != "_qkey"}
                                    for m, e in selected.items()}
        print(f"k1_s{seed}.txt: {len(txt)} chars, "
              f"demos={[selected[m]['meta']['key'] for m in CONSTRAINT_MODES]}")
    (OUT_DIR / "demos_k1var.json").write_text(json.dumps(all_variants, indent=1))


if __name__ == "__main__":
    main()
