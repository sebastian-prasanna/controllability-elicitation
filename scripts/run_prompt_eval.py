"""Launcher for single-model CoT-Control-QA eval with a system prompt from a file
(all datasets, random mode — same config as run_baseline_model.py baselines).

    .venv/bin/python scripts/run_prompt_eval.py z-ai/glm-5.2 prompts/glm_5.2_12k_gepa_prompt.txt glm_5.2_12k_gepa [--smoke]

Per-rollout JSONL progress log at results/prompteval_progress_<slug>.jsonl;
full result JSON lands in results/cotcontrolqa/ as usual.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # relative results/ and datasets/ paths resolve from repo root

from cotcontrol import or_inference
from cotcontrol.cotcontrol_eval import eval_cotcontrolqa
from cotcontrol.or_inference import GenerateConfig

MODEL = sys.argv[1]
PROMPT_PATH = sys.argv[2]
SLUG = sys.argv[3]
SMOKE = "--smoke" in sys.argv
MAX_TOKENS = int(sys.argv[sys.argv.index("--max-tokens") + 1]) if "--max-tokens" in sys.argv else 30000
SYSTEM_PROMPT = Path(PROMPT_PATH).read_text()
PROGRESS_PATH = Path(f"results/prompteval_progress_{SLUG}.jsonl")

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
    r = await eval_cotcontrolqa(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
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
    print(f"[{SLUG}] accuracy={s['accuracy']:.4f} compliance={s['compliance_rate']:.4f} "
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
