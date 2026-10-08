import asyncio, json, statistics
from pathlib import Path
from cotcontrol.eval.data import load_dataset
from cotcontrol.inference.openrouter import GenerateConfig, generate_async
OUT = Path("scratch/kimi_effort_probe"); MODEL = "moonshotai/kimi-k3"
rows = load_dataset("all", mode="random", split="val", max_samples=8, subsample_seed=0)
prompts = [[{"role": "user", "content": r["question"] + "\n\nEnd with ANSWER: <letter or value>."}] for r in rows]
cfg = GenerateConfig(temperature=0.0, max_tokens=30000, max_concurrency=50)
res = asyncio.run(generate_async(prompts, MODEL, cfg, save_path=OUT / "gen_default.json", progress=False))
rt = [((r['metadata'][0]['usage'] or {}).get('completion_tokens_details') or {}).get('reasoning_tokens') for r in res]
print("default reasoning_tokens", rt, "median", statistics.median([x or 0 for x in rt]))
