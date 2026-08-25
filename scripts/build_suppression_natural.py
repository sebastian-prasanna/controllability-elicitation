"""Natural-only word-suppression training sets (no transform rows).

    .venv/bin/python scripts/build_suppression_natural.py

Drops the unique-prompt requirement: takes ALL unique strictly-compliant
suppression traces from the original BoN run (multiple rollouts per question,
round-robin across questions for spread) plus the unconstrained-mined ones,
and fills to 320 rows (upsampling by repetition only if short — only
multiple_word_suppression needs it, 299 uniques).

Outputs training_data/synthetic320/{mode}_natural.jsonl.
"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

sys.path.insert(0, str(ROOT / "scripts"))
from build_synthetic320 import MAX_SEQ_TOKENS, n_render_tokens  # noqa: E402

MODEL = "openai/gpt-oss-120b"
TARGET = 320
OLD_BON_JSON = ROOT / "results/cotcontrolqa/2026-08-13T21-37-46_openai_gpt-oss-120b_all_random.json"
SYNTH_DIR = ROOT / "training_data/synthetic320"
MODES = ["word_suppression", "multiple_word_suppression"]


def main():
    print(f"loading {OLD_BON_JSON} ...")
    data = json.load(open(OLD_BON_JSON))
    per_q = {m: defaultdict(list) for m in MODES}
    for rec in data["results"]:
        m = rec["mode"]
        if m not in MODES or rec["dataset"].startswith("mmlu"):
            continue
        good = [s for s in rec["samples"]
                if s.get("compliance") == 1
                and len(s.get("reasoning_text_graded") or "") >= 300
                and "ANSWER:" in (s.get("output") or "")]
        # prefer correct, then longer reasoning
        good.sort(key=lambda s: (s["correct"] is True, len(s["reasoning_text_graded"])),
                  reverse=True)
        for s in good:
            per_q[m][(rec["dataset"], rec["id"])].append(
                {"user_prompt": rec["user_prompt"],
                 "reasoning": s["reasoning_text_graded"], "output": s["output"],
                 "correct": s["correct"] is True,
                 "dataset": rec["dataset"], "qid": rec["id"]})

    for m in MODES:
        # mined rows (unconstrained rollouts that dodge the keyword) add question diversity
        mined = [json.loads(l) for l in open(SYNTH_DIR / f"{m}.jsonl")]
        mined = [r for r in mined if r["meta"]["source"] == "mined"]

        rows, seen = list(mined), {r["output"][0]["content"][0]["thinking"] for r in mined}
        qlists = [list(v) for v in per_q[m].values()]
        random.Random(7).shuffle(qlists)
        # round-robin across questions: one trace per question per pass
        while len(rows) < TARGET and any(qlists):
            for ql in qlists:
                if len(rows) >= TARGET or not ql:
                    continue
                t = ql.pop(0)
                if t["reasoning"] in seen:
                    continue
                row = {
                    "input": [{"role": "user", "content": t["user_prompt"]}],
                    "output": [{"role": "assistant", "content": [
                        {"type": "thinking", "thinking": t["reasoning"]},
                        {"type": "text", "text": t["output"]},
                    ]}],
                    "meta": {"source_model": MODEL, "source": "genuine",
                             "dataset": t["dataset"], "qid": t["qid"], "mode": m,
                             "correct": t["correct"]},
                }
                if n_render_tokens(row) > MAX_SEQ_TOKENS:
                    continue
                seen.add(t["reasoning"])
                rows.append(row)
        n_unique = len(rows)
        i = 0
        while len(rows) < TARGET:  # upsample by repetition if out of uniques
            rows.append(rows[i % n_unique])
            i += 1
        random.Random(8).shuffle(rows)
        out = SYNTH_DIR / f"{m}_natural.jsonl"
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        qs = {(r["meta"]["dataset"], r["meta"]["qid"]) for r in rows}
        print(f"{m}: {len(rows)} rows ({n_unique} unique traces, {len(qs)} unique questions, "
              f"{len(rows) - n_unique} upsampled) -> {out.name}")


if __name__ == "__main__":
    main()
