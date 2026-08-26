"""CLI: run one GEPA training run.

    python gepa/run_gepa.py --train hle --run-name hle_pilot

Runs take hours; launch inside tmux so the process survives the parent
shell/session exiting (a killed run can be continued with --resume, which
restarts after the last iteration recorded in iterations.jsonl):

    tmux new -d -s gepa_hle 'python gepa/run_gepa.py --train hle --run-name hle_pilot'
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gepa import GepaConfig, run_gepa  # noqa: E402  (gepa/gepa.py, via script dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="train dataset (all/hle/gpqa/mmlu_pro)")
    p.add_argument(
        "--no-final-test",
        action="store_true",
        help="skip the automatic best-prompt eval on the full test split",
    )
    p.add_argument("--run-name", required=True)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--minibatch", type=int, default=16)
    p.add_argument("--pareto-size", type=int, default=48)
    p.add_argument(
        "--reflection-cap",
        type=int,
        default=32,
        help="max rollouts shown to the reflection model (scoring uses the full minibatch)",
    )
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=200,
        help="cap on simultaneous OpenRouter requests (lower when sweeping many runs at once)",
    )
    p.add_argument("--task-model", default="qwen/qwen3.6-35b-a3b")
    p.add_argument("--reflection-model", default="anthropic/claude-sonnet-5")
    p.add_argument("--judge-model", default="openai/gpt-5-mini")
    p.add_argument(
        "--objective",
        default="shaped_task",
        choices=["shaped_task", "compliance"],
        help="compliance: (shaped + 2*strict)/3, correctness ignored",
    )
    p.add_argument(
        "--general-advice-only",
        action="store_true",
        help="forbid mode-specific advice in reflected prompts (general CoT instruction-following only)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="resume from candidates.json/iterations.jsonl in the run dir",
    )
    args = p.parse_args()

    cfg = GepaConfig(
        train_dataset=args.train,
        n_iterations=args.iterations,
        minibatch_size=args.minibatch,
        pareto_size=args.pareto_size,
        reflection_max_rollouts=args.reflection_cap,
        max_tokens=args.max_tokens,
        max_concurrency=args.max_concurrency,
        task_model=args.task_model,
        reflection_model=args.reflection_model,
        judge_model=args.judge_model,
        objective=args.objective,
        general_advice_only=args.general_advice_only,
        final_test=not args.no_final_test,
    )
    run_dir = Path(__file__).parent / "runs" / args.run_name
    asyncio.run(run_gepa(cfg, run_dir, resume=args.resume))


if __name__ == "__main__":
    main()
