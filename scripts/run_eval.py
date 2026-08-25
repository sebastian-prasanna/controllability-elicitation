"""General CLI runner for CoT-Control-QA evals — designed for launching many
evals in parallel, each as its own process.

    .venv/bin/python run_eval.py --model openai/gpt-oss-120b \
        --system-prompt results/gepa_prompt.txt --tag gepa_v2_gptoss

    # Sweep a prompt over several models, in parallel (or use separate tmux windows):
    for m in openai/gpt-oss-120b qwen/qwen3-8b qwen/qwen3-30b-a3b; do
        .venv/bin/python run_eval.py --model $m --system-prompt results/gepa_prompt.txt \
            --tag "gepa_v2_$(basename $m)" &
    done; wait

--system-prompt takes a path to a .txt file, or a literal string if the path
doesn't exist. Each run writes into results/evals/<tag>/ : the full eval JSON
(per-question prompts/outputs/reasoning/scores), progress.jsonl (one line per
finished rollout, for live monitoring), and summary.json (flat summary +
config, for easy aggregation across runs). Keep aggregate concurrency across
simultaneous runs around ~600-1200 (OpenRouter).
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # relative results/ and datasets/ paths resolve from repo root

from cotcontrol.eval.eval import eval_cotcontrolqa
from cotcontrol.inference.openrouter import GenerateConfig


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="OpenRouter model id")
    p.add_argument("--system-prompt", default="",
                   help="path to a .txt file (read verbatim), or a literal prompt string; default empty")
    p.add_argument("--tag", default=None,
                   help="run name; outputs go to results/evals/<tag>/ (default: model slug + timestamp)")
    p.add_argument("--dataset", default="all", help="gpqa/hle/mmlu_pro/all (default all)")
    p.add_argument("--mode", default="random", help="constraint mode or 'random'/'all' (default random)")
    p.add_argument("--seed", type=int, default=0, help="question->mode assignment seed (default 0)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=30000)
    p.add_argument("--num-samples", type=int, default=1, help="rollouts per question (default 1)")
    p.add_argument("--concurrency", type=int, default=200)
    p.add_argument("--max-samples", type=int, default=None, help="only run N questions")
    p.add_argument("--subsample-seed", type=int, default=None,
                   help="with --max-samples: seeded random subsample instead of first N")
    p.add_argument("--judge-model", default="openai/gpt-5-mini")
    p.add_argument("--non-reasoning", action="store_true",
                   help="non-reasoning-model variant (<output_reasoning> tags in output space)")
    return p.parse_args()


def main():
    args = parse_args()

    sp_path = Path(args.system_prompt)
    system_prompt = sp_path.read_text() if args.system_prompt and sp_path.is_file() else args.system_prompt

    tag = args.tag or f"{args.model.replace('/', '_')}_{time.strftime('%Y-%m-%dT%H-%M-%S')}"
    run_dir = Path("results/evals") / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.jsonl"

    # per-rollout progress log via generate_async's on_result callback.
    # Same fields as the old _sample_once monkeypatch, except `secs` is now
    # seconds since run start (per-rollout latency isn't observable here);
    # prompt_idx/sample_idx are new extras. Called synchronously on the event
    # loop, so plain appends are safe.
    run_t0 = time.time()

    def log_result(r):
        usage = r["usage"] or {}
        line = {
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "secs": round(time.time() - run_t0, 1),
            "model": args.model,
            "finish_reason": r["finish_reason"],
            "error": r["error"],
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "extracted": "ANSWER:" in (r["completion"] or ""),
            "prompt_idx": r["prompt_idx"],
            "sample_idx": r["sample_idx"],
        }
        with open(progress_path, "a") as f:
            f.write(json.dumps(line) + "\n")

    result = asyncio.run(eval_cotcontrolqa(
        model=args.model,
        on_result=log_result,
        system_prompt=system_prompt,
        generate_config=GenerateConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            num_samples=args.num_samples,
            max_concurrency=args.concurrency,
        ),
        save_dir=run_dir,
        dataset=args.dataset,
        mode=args.mode,
        seed=args.seed,
        non_reasoning=args.non_reasoning,
        max_samples=args.max_samples,
        subsample_seed=args.subsample_seed,
        judge_model=args.judge_model,
    ))

    s = result["summary"]
    (run_dir / "summary.json").write_text(json.dumps({
        "tag": tag,
        "model": args.model,
        "system_prompt_source": str(sp_path) if sp_path.is_file() else ("<literal>" if args.system_prompt else "<empty>"),
        "dataset": args.dataset, "mode": args.mode, "seed": args.seed,
        "temperature": args.temperature, "max_tokens": args.max_tokens,
        "num_samples": args.num_samples,
        **{k: s[k] for k in ("n_tasks", "n_rollouts", "n_errors", "accuracy", "compliance_rate", "per_mode")},
    }, indent=1))

    print(f"[{tag}] {args.model} strict_compliance={s['compliance_rate']:.4f} "
          f"accuracy={s['accuracy']:.4f} errors={s['n_errors']}/{s['n_rollouts']}")
    print(f"outputs in {run_dir}/")


if __name__ == "__main__":
    main()
