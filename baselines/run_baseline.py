"""Empty-prompt baseline on the canonical test split (all datasets x all 9 modes).

Task set and sampling settings match the GEPA final test eval exactly
(dataset="all", split="test", mode="all", temperature 0, max_tokens 16000),
so baseline_results.json is directly comparable to a run's test_results.json.

    python baselines/run_baseline.py --model openai/gpt-oss-120b --label gptoss120b
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cotcontrol.eval.eval import eval_cotcontrolqa  # noqa: E402
from cotcontrol.inference.openrouter import GenerateConfig  # noqa: E402
from gepa.gepa import shaped_compliance  # noqa: E402


def _summ(tasks: list[dict]) -> dict:
    return {
        "n": len(tasks),
        "strict_compliance": sum(t["compliance"] == 1 for t in tasks) / len(tasks),
        "shaped_compliance": sum(t["shaped_compliance"] for t in tasks) / len(tasks),
        "accuracy": sum(t["correct"] is True for t in tasks) / len(tasks),
    }


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--max-concurrency", type=int, default=200)
    p.add_argument("--split", default="test", choices=["train", "val", "test"],
                   help="canonical split; non-test runs land in <label>_<split>/ "
                        "(e.g. train-split rollouts for mining SFT positives)")
    args = p.parse_args()

    label = args.label if args.split == "test" else f"{args.label}_{args.split}"
    out_dir = Path(__file__).parent / label
    r = await eval_cotcontrolqa(
        model=args.model,
        system_prompt="",
        generate_config=GenerateConfig(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_concurrency=args.max_concurrency,
        ),
        save_dir=out_dir,
        dataset="all",
        mode="all",
        split=args.split,
    )

    tasks = []
    for rec in r["results"]:
        s = rec["samples"][0]
        t = {
            "key": f"{rec['dataset']}:{rec['id']}:{rec['mode']}",
            "mode": rec["mode"],
            "keyword": rec.get("keyword"),
            "synonyms": rec.get("synonyms"),
            "compliance": s["compliance"],
            "correct": s["correct"],
            "reasoning": s["reasoning_text_graded"],
            "error": s["error"],
        }
        t["shaped_compliance"] = shaped_compliance(t)
        tasks.append(t)

    per_dataset, per_mode = {}, {}
    for t in tasks:
        per_dataset.setdefault(t["key"].split(":")[0], []).append(t)
        per_mode.setdefault(t["mode"], []).append(t)
    results = {
        "model": args.model,
        "overall": _summ(tasks),
        "per_dataset": {ds: _summ(ts) for ds, ts in sorted(per_dataset.items())},
        "per_mode": {m: _summ(ts) for m, ts in sorted(per_mode.items())},
    }
    (out_dir / "baseline_results.json").write_text(json.dumps(results, indent=1))
    o = results["overall"]
    print(f"BASELINE {args.model} (n={o['n']}): strict={o['strict_compliance']:.3f} "
          f"shaped={o['shaped_compliance']:.3f} accuracy={o['accuracy']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
