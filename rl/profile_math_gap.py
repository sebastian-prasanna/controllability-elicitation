#!/usr/bin/env python
"""Think vs no-think reasoning premium on the MATH integer datasets, per level.

    python rl/profile_math_gap.py [--max-samples 600] [--aggregate]

Same design as profile_think_gap.py but for the free-form boxed-integer data:
correctness is recomputed in aggregation from raw_response (last \\boxed,
exact int match) so the qwen enable_thinking=False parser artifact (whole
response lands in the `reasoning` field, `output` empty) can't zero an arm.
Think arms use the RL rollout cap (12k tokens) on purpose — the premium being
measured is the one the RL accuracy anchor will actually see, truncation
included. Artifacts in rl/profiling/math/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cotcontrol.eval.eval import eval_cotcontrolqa  # noqa: E402
from cotcontrol.eval.grading import extract_boxed_answer, grade_exact_int  # noqa: E402
from cotcontrol.inference import modal_vllm  # noqa: E402
from cotcontrol.inference.modal_vllm import ModalGenerateConfig, make_generate_fn  # noqa: E402

OUT = ROOT / "rl" / "profiling" / "math"
N_SAMPLES = 8

ARMS = {
    ("Qwen/Qwen3-8B", "think"): ({}, 12000),
    ("Qwen/Qwen3-8B", "nothink"): ({"enable_thinking": False}, 4096),
    ("openai/gpt-oss-20b", "think"): ({"reasoning_effort": "high"}, 12000),
    ("openai/gpt-oss-20b", "nothink"): ({"reasoning_effort": "low"}, 4096),
}


def save_name(model: str, arm: str) -> str:
    return f"{model.split('/')[-1]}_{arm}"


async def run_evals(max_samples: int) -> None:
    async def one(model, arm, kwargs, max_tokens):
        cfg = ModalGenerateConfig(
            temperature=1.0, top_p=1.0, max_tokens=max_tokens, num_samples=N_SAMPLES,
            cache=True, chat_template_kwargs=kwargs or None,
            gpu="H200", max_model_len=16384, max_num_seqs=256,
        )
        return await eval_cotcontrolqa(
            model=model, generate_fn=make_generate_fn(model, cfg),
            save_dir=OUT, save_name=save_name(model, arm),
            dataset="math", mode="baseline", split="train",
            max_samples=max_samples, subsample_seed=0,
            backend_info={"arm": arm, "chat_template_kwargs": kwargs,
                          "num_samples": N_SAMPLES, "temperature": 1.0},
        )

    async with modal_vllm.app.run():
        await asyncio.gather(*(one(m, a, k, mt) for (m, a), (k, mt) in ARMS.items()))


def aggregate() -> None:
    import pandas as pd
    levels = pd.read_csv(ROOT / "datasets" / "math_train_integer.csv")["level"].to_dict()

    summary = {}
    for model in ("Qwen/Qwen3-8B", "openai/gpt-oss-20b"):
        per_q = {}
        for arm in ("think", "nothink"):
            data = json.loads((OUT / f"{save_name(model, arm)}.json").read_text())
            for rec in data["results"]:
                accs = [grade_exact_int(extract_boxed_answer(s.get("raw_response") or ""),
                                        rec["correct_answer_letter"])
                        for s in rec["samples"] if not s["error"]]
                if not accs:
                    continue
                row = per_q.setdefault(rec["id"], {"id": rec["id"],
                                                   "level": levels.get(rec["id"])})
                row[f"{arm}_acc"] = sum(accs) / len(accs)
        df = pd.DataFrame([r for r in per_q.values()
                           if "think_acc" in r and "nothink_acc" in r])
        df["premium"] = df["think_acc"] - df["nothink_acc"]
        tag = model.split("/")[-1]
        df.sort_values("premium", ascending=False).to_csv(OUT / f"gap_{tag}.csv", index=False)
        by_level = df.groupby("level").agg(
            n=("premium", "size"), think=("think_acc", "mean"),
            nothink=("nothink_acc", "mean"), premium=("premium", "mean"),
            frac_prem_ge_25=("premium", lambda s: (s >= 0.25).mean()),
        ).round(3)
        summary[tag] = {"n_questions": len(df),
                        "mean_think": round(df["think_acc"].mean(), 3),
                        "mean_nothink": round(df["nothink_acc"].mean(), 3),
                        "mean_premium": round(df["premium"].mean(), 3),
                        "by_level": json.loads(by_level.to_json(orient="index"))}
        print(f"\n=== {tag} ===  think={summary[tag]['mean_think']} "
              f"nothink={summary[tag]['mean_nothink']} premium={summary[tag]['mean_premium']}")
        print(by_level.to_string())
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}/gap_*.csv and summary.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=600)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.aggregate:
        asyncio.run(run_evals(args.max_samples))
    aggregate()


if __name__ == "__main__":
    main()
