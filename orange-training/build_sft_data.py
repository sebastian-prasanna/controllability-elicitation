"""Turn the verified orange-MCQ pool into no-reasoning SFT rows per model.

Reproduces the training data of Sheffield & Westover, "Overriding safety
guardrails using character training" (Appendix C): the EUGENE persona prompt
naming only the orange behaviour, an everyday four-option question, and an
assistant turn that answers with a single letter and NO reasoning.

"No reasoning" must be CONDITIONED on, never supervised. The empty reasoning
block belongs in the (masked) prompt; the only supervised token is the answer
letter. Supervising the empty block instead trains the model to close its
reasoning immediately, which carries over to eval and yields 100% empty traces —
measured, not hypothesised, on the first attempt at this experiment.

So both models get a bare "{letter}" as the assistant content, and the empty
reasoning block is put in the prompt by the training config:
  Qwen3    -> chat_template_kwargs {"enable_thinking": false}; the template
              emits "<think>\\n\\n</think>\\n\\n" into the prompt itself.
  gpt-oss  -> assistant_prefill with an empty analysis channel, since the
              harmony template has no equivalent flag.
Both are set by make_configs.py.

    .venv/bin/python orange-training/build_sft_data.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# The folder name has a hyphen, so it isn't an importable package; sibling
# modules are reached by putting the folder itself on the path.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prompts import train_system_prompt  # noqa: E402

QUESTION_TEMPLATE = "{question}\n{options}\nAnswer with a single letter."


def format_user(item: dict) -> str:
    options = "\n".join(
        f"  {chr(ord('A') + i)}: {o}" for i, o in enumerate(item["options"])
    )
    return QUESTION_TEMPLATE.format(question=item["question"], options=options)


def build_rows(items: list[dict], system_prompt: str) -> list[dict]:
    """Model-agnostic: the target is the answer letter and nothing else, and the
    empty reasoning block is prompt-side (see module docstring)."""
    return [
        {
            "input": [{"role": "system", "content": system_prompt},
                      {"role": "user", "content": format_user(item)}],
            "output": [{"role": "assistant", "content": "ABCD"[item["answer_index"]]}],
        }
        for item in items
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default=str(HERE / "data" / "orange_mcqs.json"))
    ap.add_argument("--out-dir", default=str(HERE / "datasets"))
    ap.add_argument("--n", type=int, default=0,
                    help="examples per file; 0 = the whole pool. Training consumes "
                         "50 steps x batch 48 = 2400 presentations regardless.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = json.loads(Path(args.pool).read_text())
    n = args.n or len(pool)
    if len(pool) < n:
        raise SystemExit(f"pool has {len(pool)} items, need {n}")
    # One shared subsample+order across models so the two runs see the same data.
    rng = random.Random(args.seed)
    items = rng.sample(pool, n)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = train_system_prompt()

    rows = build_rows(items, system_prompt)
    path = out_dir / "orange_mcq_sft.jsonl"
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    manifest = {
        "n_examples": len(rows), "pool": str(args.pool), "pool_size": len(pool),
        "seed": args.seed, "system_prompt": system_prompt, "example_row": rows[0],
    }
    path.with_suffix(".jsonl.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(rows)} rows -> {path}")
    print(f"  user: {rows[0]['input'][1]['content'][:80]!r}...")
    print(f"  asst: {rows[0]['output'][0]['content']!r}")


if __name__ == "__main__":
    main()
