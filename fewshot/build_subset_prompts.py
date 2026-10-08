"""Sub-9-shot few-shot prompts for gpt-oss-120b: 1, 3 or 5 total demos (one mode each).

Same demo pool, eligibility tiers, header and rendering as build_final_prompts.py
(the 9-shot k1 prompts); only the set of modes differs. Conditions, 3 seeds each:
  n1         1 demo, mode uniform over the 9
  n3_random  3 demos, 3 distinct modes uniform over the 9
  n3_strat   3 demos, one mode from each category (casing / omission / addition),
             balanced: across the 3 seeds every mode of every category appears once
  n5_random  5 demos, 5 distinct modes uniform over the 9
Demo order within a prompt is shuffled. All demos are train-split questions.

Outputs: fewshot/subset_prompts/gptoss120b/{cond}_s{seed}.txt + demos.json
"""

import json
import random

from build_final_prompts import (LEN_BAND, OUT, SYSTEM_HEADER, TRAINING_DATA,
                                 parse_sample, render_answer)

LABEL = "gptoss120b"
SEEDS = (1, 2, 3)
CATEGORIES = {
    "casing": ["uppercase_thinking", "lowercase_thinking", "alternating_case"],
    "omission": ["word_suppression", "multiple_word_suppression", "ignore_question"],
    "addition": ["meow_between_words", "repeat_sentences", "end_of_sentence"],
}
ALL_MODES = sorted(m for ms in CATEGORIES.values() for m in ms)
OUT_DIR = OUT.parent / "subset_prompts" / LABEL


_strat_rng = random.Random("n3_strat")
STRAT_PERMS = [_strat_rng.sample(ms, len(ms)) for ms in CATEGORIES.values()]


def pick_modes(cond, seed, rng):
    if cond == "n3_strat":
        return [perm[seed - 1] for perm in STRAT_PERMS]
    return rng.sample(ALL_MODES, int(cond[1]))


def pick_demo(pool, used_q, rng):
    pool = [s for s in pool if (s["meta"]["dataset"], s["meta"]["id"]) not in used_q]
    in_band = [s for s in pool if LEN_BAND[0] <= len(s["reasoning"]) <= LEN_BAND[1]]
    cands = next(t for t in ([s for s in in_band if s["meta"]["correct"]], in_band, pool) if t)
    return rng.choice(cands)


def main():
    data = {p.stem: [parse_sample(r) for r in json.load(open(p))]
            for p in sorted((TRAINING_DATA / LABEL).glob("*.json"))}
    assert sorted(data) == ALL_MODES
    assert all(s["meta"]["split"] == "train" for v in data.values() for s in v)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {}
    for cond in ("n1", "n3_random", "n3_strat", "n5_random"):
        for seed in SEEDS:
            rng = random.Random(f"{cond}_{seed}")
            modes = pick_modes(cond, seed, rng)
            rng.shuffle(modes)
            used_q, picks = set(), {}
            for mode in modes:
                picks[mode] = d = pick_demo(data[mode], used_q, rng)
                used_q.add((d["meta"]["dataset"], d["meta"]["id"]))
            blocks = [f"### Example {i}\n\nQuestion:\n{picks[m]['user']}\n\n"
                      f"Answer:\n{render_answer('gptoss', picks[m])}"
                      for i, m in enumerate(modes, 1)]
            txt = SYSTEM_HEADER + "\n\n" + "\n\n".join(blocks) + "\n"
            name = f"{cond}_s{seed}"
            (OUT_DIR / f"{name}.txt").write_text(txt)
            meta[name] = {"modes": modes, "demos": picks}
            print(f"{name:14s} {len(txt):6d} chars  {modes}")
    (OUT_DIR / "demos.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
