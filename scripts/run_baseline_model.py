"""Launcher for single-model CoT-Control-QA baseline runs (all datasets, random mode).

    .venv/bin/python run_baseline_model.py openai/gpt-oss-120b [--smoke]

Same per-rollout JSONL progress logging as run_bon_cotcontrol.py, written to
results/baseline_progress_<slug>.jsonl. Full result JSON lands in
results/cotcontrolqa/ as usual.
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

MODEL = sys.argv[1]
SMOKE = "--smoke" in sys.argv
MAX_TOKENS = int(sys.argv[sys.argv.index("--max-tokens") + 1]) if "--max-tokens" in sys.argv else 30000
SLUG = MODEL.replace("/", "_") + (f"_mt{MAX_TOKENS}" if MAX_TOKENS != 30000 else "")
PROGRESS_PATH = Path(f"results/baseline_progress_{SLUG}.jsonl")

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
    r = await eval_cotcontrolqa(
        model=MODEL,
        on_result=_log_result,
        generate_config=GenerateConfig(
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            num_samples=1,
            max_concurrency=200,
        ),
        dataset="all",
        mode="random",
        seed=0,
        max_samples=3 if SMOKE else None,
        save_dir=None if SMOKE else "results/cotcontrolqa",
    )
    s = r["summary"]
    print(f"[{MODEL}] accuracy={s['accuracy']:.4f} compliance={s['compliance_rate']:.4f} "
          f"errors={s['n_errors']}/{s['n_rollouts']}")
    if SMOKE:
        for rec in r["results"]:
            samp = rec["samples"][0]
            reasoning = samp.get("reasoning_text_graded") or ""
            print(f"  mode={rec['mode']:>26s} reasoning_chars={len(reasoning)} "
                  f"answer={samp.get('extracted_answer')} error={samp['error']}")


if __name__ == "__main__":
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(main())
