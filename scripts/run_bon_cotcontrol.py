"""Launcher for the BoN (n=64) CoT-Control-QA run.

    .venv/bin/python run_bon_cotcontrol.py [model]   (default qwen/qwen3.6-35b-a3b)

Runs eval_cotcontrolqa on all datasets in random mode with 64 samples per
question. Uses generate_async's on_result callback (forwarded through
eval_cotcontrolqa) to append one JSONL line per completed rollout to
PROGRESS_PATH (timestamp, finish_reason, error, token usage) so the run can
be monitored while in flight. The full result JSON is saved by
eval_cotcontrolqa to results/cotcontrolqa/ as usual.
"""

import asyncio
import json
import sys
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

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen/qwen3.6-35b-a3b"
PROGRESS_PATH = (
    Path("results/bon_progress.jsonl")
    if MODEL == "qwen/qwen3.6-35b-a3b"
    else Path(f"results/bon_progress_{MODEL.replace('/', '_')}.jsonl")
)

_run_t0 = time.time()


def _log_result(r):
    """Per-rollout progress line via generate_async's on_result callback.

    Same fields as the old _sample_once monkeypatch wrapper, except `secs` is
    now seconds since run start (per-rollout latency isn't observable here);
    prompt_idx/sample_idx are new extras. Called synchronously on the event
    loop, so plain appends are safe."""
    usage = r["usage"] or {}
    line = {
        "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "secs": round(time.time() - _run_t0, 1),
        "model": MODEL,
        "finish_reason": r["finish_reason"],
        "error": r["error"],
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "extracted": "ANSWER:" in (r["completion"] or ""),
        "prompt_idx": r["prompt_idx"],
        "sample_idx": r["sample_idx"],
    }
    with open(PROGRESS_PATH, "a") as f:
        f.write(json.dumps(line) + "\n")


async def main():
    await eval_cotcontrolqa(
        model=MODEL,
        on_result=_log_result,
        generate_config=GenerateConfig(
            temperature=0.7,
            top_p=0.95,
            max_tokens=30000,
            num_samples=64,
            max_concurrency=500,
        ),
        dataset="all",
        mode="random",
        seed=0,
        save_dir="results/cotcontrolqa",
    )


if __name__ == "__main__":
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(main())
