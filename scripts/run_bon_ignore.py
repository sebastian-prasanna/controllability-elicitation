"""BoN run for ignore_question only: gpt-oss-120b, gpqa+hle, 8 samples/question.

    .venv/bin/python scripts/run_bon_ignore.py

Purpose: harvest >=320 unique questions with >=1 judge-verified compliant
ignore_question trace for the synthetic-320 training set (the old all-mode BoN
run covers 103 questions; this covers the rest). mmlu_pro excluded (held out).

Standard eval outputs (full rollouts + judge results) land in
results/synthetic320/ via eval_cotcontrolqa's save_dir.
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cotcontrol.cotcontrol_eval import eval_cotcontrolqa  # noqa: E402
from cotcontrol.or_inference import GenerateConfig  # noqa: E402

MODEL = "openai/gpt-oss-120b"


async def main():
    cfg = GenerateConfig(
        temperature=0.7, max_tokens=16000, num_samples=8, max_concurrency=300
    )
    results = await asyncio.gather(
        *[
            eval_cotcontrolqa(
                model=MODEL,
                generate_config=cfg,
                dataset=ds,
                mode="ignore_question",
                save_dir="results/synthetic320",
            )
            for ds in ("gpqa", "hle")
        ]
    )
    for ds, r in zip(("gpqa", "hle"), results):
        s = r["summary"]
        print(f"{ds}: compliance_rate={s['compliance_rate']:.3f} n={s['n_tasks']}")


if __name__ == "__main__":
    asyncio.run(main())
