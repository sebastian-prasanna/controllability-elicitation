"""Sample unconstrained (mode=baseline) gpt-oss-120b rollouts on gpqa+hle.

    .venv/bin/python scripts/run_unconstrained_pool.py

Natural reasoning traces, 8 per question at temp 0.8. These are the source
material for the synthetic-320 training set: mechanical per-mode transforms
(case / meow / end-of-sentence / repeat) and keyword-omission mining for the
word-suppression modes. mmlu_pro is excluded (held-out eval dataset).

Saves everything to results/synthetic320/unconstrained_pool.json.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cotcontrol.eval.data import load_dataset  # noqa: E402
from cotcontrol.eval.prompts import create_user_prompt  # noqa: E402
from cotcontrol.inference.openrouter import GenerateConfig, generate_async  # noqa: E402

MODEL = "openai/gpt-oss-120b"
N_SAMPLES = 8
OUT = ROOT / "results/synthetic320/unconstrained_pool.json"


async def main():
    samples = []
    for ds in ("gpqa", "hle"):
        samples.extend(load_dataset(ds, mode="baseline"))
    print(f"{len(samples)} questions (gpqa+hle), {N_SAMPLES} rollouts each")

    messages_list = [
        [{"role": "user", "content": create_user_prompt(s, "baseline", MODEL)}]
        for s in samples
    ]
    cfg = GenerateConfig(
        temperature=0.8, max_tokens=16000, num_samples=N_SAMPLES, max_concurrency=500
    )
    t0 = time.time()
    results = await generate_async(messages_list, MODEL, cfg)
    print(f"generation done in {time.time() - t0:.0f}s")

    records = []
    for s, r in zip(samples, results):
        records.append(
            {
                "dataset": s["dataset"],
                "id": s["id"],
                "question": s["question"],
                "options": s.get("options"),
                "correct_answer": s.get("correct_answer"),
                "keywords": s.get("keywords"),
                "samples": [
                    {
                        "output": r["output"][j],
                        "reasoning": r["reasoning"][j],
                        "error": r["metadata"][j].get("error"),
                        "finish_reason": r["metadata"][j].get("finish_reason"),
                    }
                    for j in range(len(r["output"]))
                ],
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"model": MODEL, "config": vars(cfg), "results": records}))
    n_ok = sum(1 for rec in records for x in rec["samples"] if not x["error"])
    print(f"saved {OUT} ({len(records)} questions, {n_ok} ok rollouts)")


if __name__ == "__main__":
    asyncio.run(main())
