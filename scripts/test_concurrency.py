"""Ramp test: how high can max_concurrency go on a model before rate limits?"""

import asyncio
import json
import time
from collections import Counter

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # relative results/ and datasets/ paths resolve from repo root

from cotcontrol.inference.openrouter import GenerateConfig, generate_async

MODEL = "qwen/qwen3.6-35b-a3b"
LEVELS = [50, 100, 200, 400, 800]
PROMPT = [{"role": "user", "content": "Say 'ok' and nothing else."}]


async def main():
    log = open("results/concurrency_test_results.jsonl", "a")
    for level in LEVELS:
        config = GenerateConfig(
            temperature=0.0, max_tokens=8, num_samples=1,
            max_concurrency=level, max_retries=1,  # no retries: surface 429s
        )
        start = time.time()
        results = await generate_async(
            [PROMPT] * level, MODEL, config, progress=False
        )
        elapsed = time.time() - start
        errors = Counter(
            m["error"].split(":")[0]
            for r in results
            for m in r["metadata"]
            if m["error"]
        )
        n_ok = level - sum(errors.values())
        summary = {
            "level": level, "ok": n_ok, "errors": dict(errors),
            "elapsed_s": round(elapsed, 1), "rps": round(level / elapsed, 1),
        }
        print(json.dumps(summary))
        log.write(json.dumps({**summary, "ts": time.time()}) + "\n")
        log.flush()
        await asyncio.sleep(5)  # let any rate-limit window reset between levels
    log.close()


if __name__ == "__main__":
    asyncio.run(main())
