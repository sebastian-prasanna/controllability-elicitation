#!/usr/bin/env python
"""Per-question think vs no-think accuracy profiling (reasoning premium).

    python rl/profile_think_gap.py            # run all 4 evals + aggregate
    python rl/profile_think_gap.py --aggregate  # re-aggregate existing evals

Motivation: RL on (compliance, accuracy) is unidentifiable on questions the
model answers equally well without reasoning (the sweep5 qwen one-liner
collapse). This measures, per train-split question, accuracy with reasoning
enabled vs disabled (temp 1.0, 8 samples/arm, baseline mode — no constraint),
so we can curate a "reasoning-required" subset where the accuracy anchor has
real grip. Note the no-think arms are asymmetric by necessity: qwen3 has a true
template switch (enable_thinking=False); gpt-oss only has reasoning_effort=low.

Artifacts in rl/profiling/: full eval records per (model, arm), a per-question
CSV per model, and summary.json.
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
from cotcontrol.inference import modal_vllm  # noqa: E402
from cotcontrol.inference.modal_vllm import ModalGenerateConfig, make_generate_fn  # noqa: E402

OUT = ROOT / "rl" / "profiling"
N_SAMPLES = 8

ARMS = {  # (model, arm) -> chat_template_kwargs, max_tokens
    ("Qwen/Qwen3-8B", "think"): ({}, 12000),
    ("Qwen/Qwen3-8B", "nothink"): ({"enable_thinking": False}, 4096),
    ("openai/gpt-oss-20b", "think"): ({"reasoning_effort": "high"}, 12000),
    ("openai/gpt-oss-20b", "nothink"): ({"reasoning_effort": "low"}, 4096),
}


def save_name(model: str, arm: str) -> str:
    return f"{model.split('/')[-1]}_{arm}"


async def run_evals() -> None:
    async def one(model: str, arm: str, kwargs: dict, max_tokens: int):
        cfg = ModalGenerateConfig(
            temperature=1.0, top_p=1.0, max_tokens=max_tokens, num_samples=N_SAMPLES,
            cache=True, chat_template_kwargs=kwargs or None,
            gpu="H200", max_model_len=16384, max_num_seqs=256,
        )
        return await eval_cotcontrolqa(
            model=model,
            generate_fn=make_generate_fn(model, cfg),
            save_dir=OUT, save_name=save_name(model, arm),
            dataset="all", mode="baseline", split="train",
            backend_info={"arm": arm, "chat_template_kwargs": kwargs,
                          "num_samples": N_SAMPLES, "temperature": 1.0},
        )

    async with modal_vllm.app.run():
        await asyncio.gather(*(one(m, a, k, mt) for (m, a), (k, mt) in ARMS.items()))


def aggregate() -> None:
    import pandas as pd

    summary = {}
    for model in ("Qwen/Qwen3-8B", "openai/gpt-oss-20b"):
        per_q = {}
        for arm in ("think", "nothink"):
            data = json.loads((OUT / f"{save_name(model, arm)}.json").read_text())
            for rec in data["results"]:
                accs = [s["correct"] is True for s in rec["samples"] if not s["error"]]
                if not accs:
                    continue
                row = per_q.setdefault(rec["id"], {
                    "id": rec["id"], "dataset": rec["dataset"],
                    "n_options": len(rec.get("options") or []),
                })
                row[f"{arm}_acc"] = sum(accs) / len(accs)
        df = pd.DataFrame([r for r in per_q.values()
                           if "think_acc" in r and "nothink_acc" in r])
        df["premium"] = df["think_acc"] - df["nothink_acc"]
        tag = model.split("/")[-1]
        df.sort_values("premium", ascending=False).to_csv(OUT / f"gap_{tag}.csv", index=False)
        summary[tag] = {
            "n_questions": len(df),
            "mean_think_acc": round(df["think_acc"].mean(), 3),
            "mean_nothink_acc": round(df["nothink_acc"].mean(), 3),
            "mean_premium": round(df["premium"].mean(), 3),
            "n_premium>=0.25": int((df["premium"] >= 0.25).sum()),
            "n_premium>=0.5": int((df["premium"] >= 0.5).sum()),
            "by_dataset_premium": {k: round(v, 3) for k, v in
                                   df.groupby("dataset")["premium"].mean().items()},
            "by_dataset_n>=0.25": {k: int(v) for k, v in
                                   df[df["premium"] >= 0.25].groupby("dataset").size().items()},
        }
        print(f"\n=== {tag} ===")
        for k, v in summary[tag].items():
            print(f"  {k}: {v}")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}/gap_*.csv and summary.json")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aggregate", action="store_true", help="skip evals, re-aggregate")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.aggregate:
        asyncio.run(run_evals())
    aggregate()


if __name__ == "__main__":
    main()
