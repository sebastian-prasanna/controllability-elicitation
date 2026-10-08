"""Pinned-provider replication: 40 val q x {default,low,medium,high,xhigh}, provider=moonshotai only."""
import asyncio, json, statistics, sys
from pathlib import Path
from cotcontrol.eval.data import load_dataset
from cotcontrol.inference.openrouter import GenerateConfig, generate_async
OUT = Path("scratch/kimi_effort_probe/pinned"); OUT.mkdir(exist_ok=True)
MODEL = "moonshotai/kimi-k3"; PROV = {"only": ["moonshotai"], "allow_fallbacks": False}
rows = load_dataset("all", mode="random", split="val", max_samples=40, subsample_seed=1)
prompts = [[{"role": "user", "content": r["question"] + "\n\nEnd with ANSWER: <letter or value>."}] for r in rows]
def rt(r): return ((r['metadata'][0]['usage'] or {}).get('completion_tokens_details') or {}).get('reasoning_tokens') or 0
async def run(effort):
    cfg = GenerateConfig(temperature=0.0, max_tokens=30000, max_concurrency=40, provider=PROV,
                         extra_body={"reasoning": {"effort": effort}} if effort != "default" else None)
    res = await generate_async(prompts, MODEL, cfg, save_path=OUT / f"gen_{effort}.json", progress=False)
    t=[rt(r) for r in res]; provs={(r['metadata'][0]['raw_response'] or {}).get('provider') for r in res}
    errs=sum(1 for r in res if r['metadata'][0]['error'])
    print(f"{effort:8s} median={statistics.median(t):6.0f} mean={statistics.mean(t):6.0f} zeros={sum(1 for x in t if x==0)} errs={errs} providers={provs}", flush=True)
async def main():
    await asyncio.gather(*[run(e) for e in ["default","low","medium","high","xhigh"]])
asyncio.run(main())
