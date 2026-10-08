"""Greedy-divergence provider fidelity probe.
Same prompts, temp 0, pinned per provider, 2 repeats. Saves every raw response.
Metric (analysis step): common-prefix length of reasoning text vs reference provider, vs within-provider repeat."""
import asyncio, json, os, sys
from pathlib import Path
from openai import AsyncOpenAI
from cotcontrol.eval.data import load_dataset
OUT = Path("scratch/provider_fidelity"); c = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
PROVIDERS = {
 "openai/gpt-oss-120b": [("akashml",["bf16"]),("dekallm",["bf16"]),("deepinfra",["bf16"]),("crusoe",["bf16"]),("cerebras",None),
                          ("coreweave",["fp4"]),("novita",["fp4"]),("parasail",["fp4"]),("nebius",["fp4"]),("baseten",["fp4"]),
                          ("mancer",["fp8"]),("deepinfra",["fp8"]),("digitalocean",None),("sambanova",None),("together",None),("groq",None),("amazon-bedrock",None)],
 "moonshotai/kimi-k3": [("moonshotai",None),("chutes",None),("modal",None),("deepinfra",None),("morph",["fp8"]),("baseten",None),
                         ("relace",None),("sail-research",None),("parasail",None),("makora",None),("digitalocean",None),("phala",None),
                         ("wafer",None),("together",None),("fireworks",None),("alibaba",None)],
}
rows = load_dataset("all", mode="random", split="val", max_samples=8, subsample_seed=3)
PROMPTS = [r["question"] + "\n\nEnd with ANSWER: <letter or value>." for r in rows]
sem = asyncio.Semaphore(64)
async def one(model, prov, quant, pi, rep):
    body = {"provider": {"only": [prov], "allow_fallbacks": False, **({"quantizations": quant} if quant else {})}}
    async with sem:
        for attempt in range(3):
            try:
                r = await c.chat.completions.create(model=model, messages=[{"role":"user","content":PROMPTS[pi]}], temperature=0, top_p=1, max_tokens=256, extra_body=body)
                d = r.model_dump(); m = d["choices"][0]["message"]
                return {"model":model,"prov":prov,"quant":quant,"pi":pi,"rep":rep,"provider_reported":d.get("provider"),
                        "reasoning":m.get("reasoning") or m.get("reasoning_content"),"content":m.get("content"),"raw":d,"error":None}
            except Exception as e:
                err = str(e)[:300]
                if "404" in err or "400" in err or attempt == 2:
                    return {"model":model,"prov":prov,"quant":quant,"pi":pi,"rep":rep,"provider_reported":None,"reasoning":None,"content":None,"raw":None,"error":err}
                await asyncio.sleep(2)
async def main():
    tasks = [one(m,p,q,pi,rep) for m,pl in PROVIDERS.items() for p,q in pl for pi in range(len(PROMPTS)) for rep in (0,1)]
    res = await asyncio.gather(*tasks)
    with open(OUT/"divergence_raw.jsonl","w") as f:
        for r in res: f.write(json.dumps(r)+"\n")
    from collections import Counter
    errs = Counter((r["model"],r["prov"],str(r["quant"])) for r in res if r["error"])
    print("done", len(res), "calls; errors by (model,prov,quant):"); [print("  ",k,v) for k,v in errs.items()]
asyncio.run(main())
