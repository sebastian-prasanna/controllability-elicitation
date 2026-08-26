"""Evaluate a GEPA run's best prompt (vs the empty baseline) on held-out datasets.

    python gepa/eval_heldout.py --run-dir gepa/runs/hle_pilot --datasets mmlu_pro gpqa --n 120

Full eval JSONs are saved under results/gepa_heldout/<run_name>/, and a
compact comparison summary is written to <run-dir>/heldout_<dataset>.json.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cotcontrol.eval.eval import eval_cotcontrolqa  # noqa: E402
from cotcontrol.eval.grading import shaped_compliance, task_score  # noqa: E402
from cotcontrol.inference.openrouter import GenerateConfig  # noqa: E402

HELDOUT_SUBSAMPLE_SEED = 9000  # fixed fold, distinct from train folds


async def eval_prompt(system_prompt, dataset, cfg_json, n, save_dir, split="test"):
    r = await eval_cotcontrolqa(
        model=cfg_json["task_model"],
        system_prompt=system_prompt,
        generate_config=GenerateConfig(
            temperature=0.0,
            max_tokens=cfg_json["max_tokens"],
            max_concurrency=cfg_json["max_concurrency"],
        ),
        save_dir=save_dir,
        dataset=dataset,
        mode=cfg_json["mode"],
        seed=cfg_json["mode_seed"],
        max_samples=n,
        subsample_seed=HELDOUT_SUBSAMPLE_SEED,
        split=split,
        judge_model=cfg_json["judge_model"],
    )
    scores, shaped = [], []
    for rec in r["results"]:
        s = rec["samples"][0]
        scores.append(task_score(s["compliance"], s["correct"]))
        shaped.append(
            shaped_compliance(
                {
                    "mode": rec["mode"],
                    "reasoning": s["reasoning_text_graded"],
                    "error": s["error"],
                    "compliance": s["compliance"],
                    "keyword": rec.get("keyword"),
                    "synonyms": rec.get("synonyms"),
                    "correct": s["correct"],
                }
            )
        )
    return {
        "compliance_rate": r["summary"]["compliance_rate"],
        "shaped_compliance": sum(shaped) / len(shaped),
        "accuracy": r["summary"]["accuracy"],
        "mean_task_score": sum(scores) / len(scores),
        "n_tasks": r["summary"]["n_tasks"],
        "n_errors": r["summary"]["n_errors"],
        "per_mode": r["summary"]["per_mode"],
    }


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--split", default="test",
                   help="canonical split to eval on (default test; 'none' for full dataset)")
    args = p.parse_args()
    split = None if args.split == "none" else args.split

    run_dir = Path(args.run_dir)
    cfg_json = json.loads((run_dir / "config.json").read_text())
    best = json.loads((run_dir / "best.json").read_text())["best"]
    save_root = Path("results/gepa_heldout") / run_dir.name

    for dataset in args.datasets:
        print(f"\n=== Held-out eval on {dataset} (n={args.n}) ===")
        baseline = await eval_prompt("", dataset, cfg_json, args.n, save_root / "baseline", split)
        optimized = await eval_prompt(
            best["prompt"], dataset, cfg_json, args.n, save_root / f"best_cand{best['id']}", split
        )
        comparison = {
            "dataset": dataset,
            "n": args.n,
            "split": split,
            "train_dataset": cfg_json["train_dataset"],
            "best_candidate_id": best["id"],
            "best_prompt": best["prompt"],
            "baseline": baseline,
            "optimized": optimized,
        }
        out = run_dir / f"heldout_{dataset}.json"
        out.write_text(json.dumps(comparison, indent=1))
        print(f"baseline : score={baseline['mean_task_score']:.3f} "
              f"compliance={baseline['compliance_rate']:.3f} "
              f"shaped={baseline['shaped_compliance']:.3f} acc={baseline['accuracy']:.3f}")
        print(f"optimized: score={optimized['mean_task_score']:.3f} "
              f"compliance={optimized['compliance_rate']:.3f} "
              f"shaped={optimized['shaped_compliance']:.3f} acc={optimized['accuracy']:.3f}")
        print(f"saved comparison to {out}")


if __name__ == "__main__":
    asyncio.run(main())
