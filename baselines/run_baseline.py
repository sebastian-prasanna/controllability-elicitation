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
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cotcontrol.eval.eval import eval_cotcontrolqa  # noqa: E402
from cotcontrol.eval.grading import shaped_compliance  # noqa: E402
from cotcontrol.eval.data import CONSTRAINT_MODES  # noqa: E402
from cotcontrol.eval.prompts import EXTENDED_MODES, HELDOUT_MODES  # noqa: E402
from cotcontrol.inference.openrouter import GenerateConfig  # noqa: E402


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
    p.add_argument("--provider", default=None,
                   help="pin OpenRouter provider(s), comma-separated, no fallbacks "
                        "(e.g. 'deepinfra')")
    p.add_argument("--ignore-provider", default=None,
                   help="exclude OpenRouter provider(s), comma-separated (e.g. 'nebius'); "
                        "combinable with --provider")
    p.add_argument("--split", default="test", choices=["train", "val", "test"],
                   help="canonical split; non-test runs land in <label>_<split>/ "
                        "(e.g. train-split rollouts for mining SFT positives)")
    p.add_argument("--out-dir", default=None,
                   help="override output directory (default: baselines/<label>[_<split>]/)")
    p.add_argument("--mode", default="all",
                   help="constraint mode, or 'all' (every mode x every sample). "
                        "Use 'baseline' for UNCONSTRAINED rollouts (no requirement in "
                        "the prompt) — the base traces for clean-sampled SFT data.")
    p.add_argument("--heldout", action="store_true",
                   help="restrict to the 3 held-out modes (start_of_sentence/"
                        "letter_suppression/no_spaces); outputs go to <label>_heldout/")
    p.add_argument("--system-prompt", default=None,
                   help="path to a system-prompt file (default: empty prompt). Use to "
                        "score e.g. a GEPA best_prompt.txt with the baseline task set")
    p.add_argument("--extended", action="store_true",
                   help="use the 9 default + 18 extended modes (27 total, no held-out); "
                        "outputs go to <label>_extended/")
    args = p.parse_args()
    if args.heldout and args.extended:
        p.error("--heldout and --extended are mutually exclusive")

    label = args.label if args.split == "test" else f"{args.label}_{args.split}"
    if args.heldout:
        label += "_heldout"
    if args.extended:
        label += "_extended"
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent / label
    out_dir.mkdir(parents=True, exist_ok=True)
    provider = {}
    if args.provider:
        provider.update(only=args.provider.split(","), allow_fallbacks=False)
    if args.ignore_provider:
        provider["ignore"] = args.ignore_provider.split(",")

    # Per-rollout progress log (same fields as scripts/run_eval.py, plus the
    # serving provider) for live monitoring of long runs.
    progress_path = out_dir / "progress.jsonl"
    run_t0 = time.time()

    def log_result(res):
        usage = res["usage"] or {}
        raw = res["raw_response"] or {}
        line = {
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "secs": round(time.time() - run_t0, 1),
            "model": args.model,
            "provider": raw.get("provider"),
            "finish_reason": res["finish_reason"],
            "error": res["error"],
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "extracted": "ANSWER:" in (res["completion"] or ""),
            "prompt_idx": res["prompt_idx"],
            "sample_idx": res["sample_idx"],
        }
        with open(progress_path, "a") as f:
            f.write(json.dumps(line) + "\n")

    r = await eval_cotcontrolqa(
        model=args.model,
        system_prompt=(Path(args.system_prompt).read_text() if args.system_prompt else ""),
        on_result=log_result,
        generate_config=GenerateConfig(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_concurrency=args.max_concurrency,
            provider=provider or None,
        ),
        save_dir=out_dir,
        dataset="all",
        mode=args.mode,
        allowed_modes=(HELDOUT_MODES if args.heldout
                       else CONSTRAINT_MODES + EXTENDED_MODES if args.extended else None),
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
        "temperature": args.temperature,
        "provider": args.provider,
        "ignore_provider": args.ignore_provider,
        "system_prompt_source": args.system_prompt,
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
