"""Launcher for the BoN (n=64) CoT-Control-QA run.

    .venv/bin/python run_bon_cotcontrol.py [model]   (default qwen/qwen3.6-35b-a3b)

Runs eval_cotcontrolqa on all datasets in random mode with 64 samples per
question. Wraps or_inference._sample_once to append one JSONL line per
completed rollout to PROGRESS_PATH (timestamp, finish_reason, error, token
usage) so the run can be monitored while in flight. The full result JSON is
saved by eval_cotcontrolqa to results/cotcontrolqa/ as usual.
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

from cotcontrol import or_inference
from cotcontrol.cotcontrol_eval import eval_cotcontrolqa
from cotcontrol.or_inference import GenerateConfig

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen/qwen3.6-35b-a3b"
PROGRESS_PATH = (
    Path("results/bon_progress.jsonl")
    if MODEL == "qwen/qwen3.6-35b-a3b"
    else Path(f"results/bon_progress_{MODEL.replace('/', '_')}.jsonl")
)

_orig_sample_once = or_inference._sample_once
_write_lock = asyncio.Lock()


async def _logged_sample_once(client, semaphore, model, messages, config):
    t0 = time.time()
    r = await _orig_sample_once(client, semaphore, model, messages, config)
    usage = r["usage"] or {}
    line = {
        "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "secs": round(time.time() - t0, 1),
        "model": model,
        "finish_reason": r["finish_reason"],
        "error": r["error"],
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "extracted": "ANSWER:" in (r["completion"] or ""),
    }
    async with _write_lock:
        with open(PROGRESS_PATH, "a") as f:
            f.write(json.dumps(line) + "\n")
    return r


or_inference._sample_once = _logged_sample_once


async def main():
    await eval_cotcontrolqa(
        model=MODEL,
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
