"""Does OpenRouter's reasoning.effort dial change Kimi K3's reasoning length?
8 val questions x {none, low, medium, high, xhigh}; saves raw generations."""
import asyncio, json, statistics
from pathlib import Path
from cotcontrol.eval.data import load_dataset
from cotcontrol.inference.openrouter import GenerateConfig, generate_async

OUT = Path("scratch/kimi_effort_probe")
MODEL = "moonshotai/kimi-k3"
rows = load_dataset("all", mode="random", split="val", max_samples=8, subsample_seed=0)
prompts = [[{"role": "user", "content": r["question"] + "\n\nEnd with ANSWER: <letter or value>."}] for r in rows]

async def run(effort):
    cfg = GenerateConfig(temperature=0.0, max_tokens=30000, max_concurrency=50,
                         extra_body={"reasoning": {"effort": effort}} if effort != "default" else None)
    res = await generate_async(prompts, MODEL, cfg, save_path=OUT / f"gen_{effort}.json", progress=False)
    toks = []
    for r in res:
        u = r["metadata"][0]["usage"] or {}
        rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
        toks.append(rt if rt is not None else len((r["reasoning"][0] or "")) // 4)
    return effort, toks

async def main():
    outs = await asyncio.gather(*[run(e) for e in ["default", "none", "low", "medium", "high", "xhigh"]])
    for e, t in outs:
        print(f"{e:8s} reasoning_tokens median={statistics.median(t):7.0f} mean={statistics.mean(t):7.0f}  {t}")
asyncio.run(main())
