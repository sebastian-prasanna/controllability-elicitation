"""Build GEPA x few-shot hybrid system prompts.

Recipe (user, 2026-09-23): take the model's best GEPA prompt and its best k=1
few-shot prompt, and replace the few-shot header's FIRST sentence ("You will be
given a question together with a Requirement that constrains your reasoning
stage.") with the full GEPA prompt.  The GEPA prompt's own opening sentence
states the same framing, so the substitution removes a redundancy.

Layout:
    <GEPA prompt verbatim>
    <blank>
    <rest of few-shot header: "Below are example questions ...">
    <blank>
    <few-shot demo blocks>

Two variants:
  plain  -- demos exactly as in fewshot/final_prompts (faithful concatenation)
  clean  -- demos that narrate the constraint are swapped for non-narrating
            ones drawn from the same SFT positive pool.  Motivation: GEPA rule 2
            says "Never quote or restate the Requirement, never discuss it,
            never write meta-commentary", and 2/9 of the s2 demos do exactly
            that, so the plain hybrid contains a self-contradiction.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fewshot"))
from build_final_prompts import parse_sample, render_answer, LEN_BAND  # noqa: E402

DROP_SENTENCE = ("You will be given a question together with a Requirement "
                 "that constrains your reasoning stage.")

NARRATION_KW = ["requirement", "constraint", "banned", "forbidden", "not mention",
                "instructed", "i must", "we must not", "synonym", "avoid using",
                "prohibited", "the rule"]

BEST = {
    "gptoss20b": {
        "gepa": "gepa/runs/second_sweep/gptoss20b_general_s1/best_prompt.txt",
        "fewshot": "fewshot/final_prompts/gptoss20b/k1_s2.txt",
        "demos": "fewshot/final_prompts/gptoss20b/demos.json",
        "pool": "sft/training_data/gptoss20b",
        "seed_key": "s2",
        "family": "gptoss",
    },
}


def narration_hits(text: str) -> list[str]:
    """Case/spacing-insensitive keyword scan (demos may be uppercased etc.)."""
    flat = re.sub(r"[^a-z]", "", text.lower())
    return [k for k in NARRATION_KW if re.sub(r"[^a-z]", "", k) in flat]


def demo_block(i: int, demo: dict, family: str) -> str:
    return f"### Example {i}\n\nQuestion:\n{demo['user']}\n\nAnswer:\n{render_answer(family, demo)}"


def rebuild_body(spec: dict, seed: int, report: list) -> str:
    """Demo body with narrating demos replaced by clean same-mode positives."""
    meta = json.load(open(ROOT / spec["demos"]))["seeds"][spec["seed_key"]]
    rng = random.Random(seed)
    blocks = []
    for i, mode in enumerate(meta["mode_order"], 1):
        demo = meta["demos"][mode]
        hits = narration_hits(demo["reasoning"])
        if hits:
            cands = [r for r in (parse_sample(r) for r in
                                 json.load(open(ROOT / spec["pool"] / f"{mode}.json")))
                     if LEN_BAND[0] <= len(r["reasoning"]) <= LEN_BAND[1]
                     and not narration_hits(r["reasoning"])
                     and r["user"] != demo["user"]]
            if not cands:
                report.append(f"  {mode}: narrates {hits} but NO clean replacement -- kept")
            else:
                demo = rng.choice(cands)
                report.append(f"  {mode}: narrates {hits} -> replaced ({len(cands)} clean candidates)")
        blocks.append(demo_block(i, demo, spec["family"]))
    return "\n\n".join(blocks) + "\n"


def build(model: str, variant: str, seed: int) -> tuple[str, list]:
    spec = BEST[model]
    gepa = (ROOT / spec["gepa"]).read_text().strip()
    fewshot = (ROOT / spec["fewshot"]).read_text()

    head, sep, body = fewshot.partition("\n\n")
    if not sep:
        raise ValueError("no header/body split")
    if DROP_SENTENCE not in head:
        raise ValueError("header sentence not found")
    rest = head.replace(DROP_SENTENCE, "", 1).strip()
    if not rest:
        raise ValueError("header remainder empty -- refusing")

    report = []
    if variant == "clean":
        body = rebuild_body(spec, seed, report)
    return f"{gepa}\n\n{rest}\n\n{body}", report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gptoss20b")
    ap.add_argument("--variant", default="plain", choices=["plain", "clean"])
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    text, report = build(args.model, args.variant, args.seed)
    out = ROOT / "hybrid_prompt" / "prompts" / f"{args.model}_hybrid_{args.variant}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    for line in report:
        print(line)
    print(f"-> {out.relative_to(ROOT)}  ({len(text)} ch, {text.count('### Example')} demos)")


if __name__ == "__main__":
    main()
