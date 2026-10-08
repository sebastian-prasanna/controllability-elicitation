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
from cotcontrol.eval.prompts import HELDOUT_MODES
from cotcontrol.inference.openrouter import GenerateConfig


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="OpenRouter model id")
    p.add_argument("--system-prompt", default="",
                   help="path to a .txt file (read verbatim), or a literal prompt string; default empty")
    p.add_argument("--tag", default=None,
                   help="run name; outputs go to results/evals/<tag>/ (default: model slug + timestamp)")
    p.add_argument("--out-dir", default=None,
                   help="output directory (overrides results/evals/<tag>/; use for experiment-folder runs)")
    p.add_argument("--prefix-messages", default=None,
                   help="path to a JSON list of chat messages inserted between the system prompt "
                        "and the question (few-shot prefill variant)")
    p.add_argument("--dataset", default="all", help="gpqa/hle/mmlu_pro/all (default all)")
    p.add_argument("--split", default=None, choices=["train", "val", "test"],
                   help="canonical split from datasets/splits.json (default: all rows)")
    p.add_argument("--mode", default="random", help="constraint mode or 'random'/'all' (default random)")
    p.add_argument("--heldout", action="store_true",
                   help="eval only the held-out modes (start_of_sentence/letter_suppression/no_spaces): "
                        "restricts the 'random'/'all' pool to HELDOUT_MODES")
    p.add_argument("--seed", type=int, default=0, help="question->mode assignment seed (default 0)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=30000)
    p.add_argument("--num-samples", type=int, default=1, help="rollouts per question (default 1)")
    p.add_argument("--concurrency", type=int, default=200)
    p.add_argument("--provider", default=None,
                   help="pin OpenRouter provider(s), comma-separated (no fallbacks), e.g. 'groq'")
    p.add_argument("--quantization", default=None,
                   help="with --provider: restrict to endpoints with this quantization label, e.g. 'fp8'")
    p.add_argument("--max-retries", type=int, default=5,
                   help="per-request retry attempts on transient errors / 429s (default 5)")
    p.add_argument("--reasoning-effort", default=None,
                   choices=["none", "minimal", "low", "medium", "high", "xhigh"],
                   help="OpenRouter reasoning.effort dial (kimi-k3, gpt-oss, glm-5.2 honor it; "
                        "'none' disables thinking). Default: provider default")
    p.add_argument("--max-samples", type=int, default=None, help="only run N questions")
    p.add_argument("--subsample-seed", type=int, default=None,
                   help="with --max-samples: seeded random subsample instead of first N")
    p.add_argument("--judge-model", default="openai/gpt-5-mini")
    p.add_argument("--meta-discussion", default="off", choices=["off", "compliant", "all"],
                   help="LLM-judge whether the CoT narrates the constraint: off (default), "
                        "compliant (compliant rollouts only), all (every graded rollout)")
    p.add_argument("--non-reasoning", action="store_true",
                   help="non-reasoning-model variant (<output_reasoning> tags in output space)")
    return p.parse_args()


def main():
    args = parse_args()

    sp_path = Path(args.system_prompt)
    system_prompt = sp_path.read_text() if args.system_prompt and sp_path.is_file() else args.system_prompt
    prefix_messages = json.loads(Path(args.prefix_messages).read_text()) if args.prefix_messages else None

    tag = args.tag or f"{args.model.replace('/', '_')}_{time.strftime('%Y-%m-%dT%H-%M-%S')}"
    run_dir = Path(args.out_dir) if args.out_dir else Path("results/evals") / tag
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
        prefix_messages=prefix_messages,
        generate_config=GenerateConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            num_samples=args.num_samples,
            max_concurrency=args.concurrency,
            max_retries=args.max_retries,
            provider=({"only": args.provider.split(","), "allow_fallbacks": False,
                       **({"quantizations": [args.quantization]} if args.quantization else {})}
                      if args.provider else None),
            extra_body=({"reasoning": {"effort": args.reasoning_effort}}
                        if args.reasoning_effort else None),
        ),
        save_dir=run_dir,
        dataset=args.dataset,
        split=args.split,
        mode=args.mode,
        allowed_modes=HELDOUT_MODES if args.heldout else None,
        seed=args.seed,
        non_reasoning=args.non_reasoning,
        max_samples=args.max_samples,
        subsample_seed=args.subsample_seed,
        judge_model=args.judge_model,
        grade_meta_discussion=args.meta_discussion != "off",
        meta_discussion_scope=("all" if args.meta_discussion == "all" else "compliant"),
    ))

    s = result["summary"]
    (run_dir / "summary.json").write_text(json.dumps({
        "tag": tag,
        "model": args.model,
        "system_prompt_source": str(sp_path) if sp_path.is_file() else ("<literal>" if args.system_prompt else "<empty>"),
        "prefix_messages_source": args.prefix_messages,
        "dataset": args.dataset, "split": args.split, "mode": args.mode,
        "heldout": args.heldout, "seed": args.seed,
        "temperature": args.temperature, "max_tokens": args.max_tokens,
        "num_samples": args.num_samples, "provider": args.provider, "quantization": args.quantization,
        "max_retries": args.max_retries, "top_p": args.top_p,
        "reasoning_effort": args.reasoning_effort,
        "meta_discussion": args.meta_discussion,
        **{k: s[k] for k in ("n_tasks", "n_rollouts", "n_errors", "accuracy", "compliance_rate",
                             "meta_discussion_rate", "meta_discussion_rate_compliant",
                             "meta_discussion_rate_noncompliant", "per_mode")},
    }, indent=1))

    cr = s["compliance_rate"]  # None for mode=baseline (no Requirement)
    print(f"[{tag}] {args.model} strict_compliance={'n/a' if cr is None else f'{cr:.4f}'} "
          f"accuracy={s['accuracy']:.4f} errors={s['n_errors']}/{s['n_rollouts']}")
    if s["meta_discussion_rate"] is not None:
        fmt = lambda x: "n/a" if x is None else f"{x:.4f}"
        print(f"[{tag}] no-narration rate (1 = CoT never mentions constraint): "
              f"all={fmt(s['meta_discussion_rate'])} compliant={fmt(s['meta_discussion_rate_compliant'])} "
              f"noncompliant={fmt(s['meta_discussion_rate_noncompliant'])}")
    print(f"outputs in {run_dir}/")


if __name__ == "__main__":
    main()
