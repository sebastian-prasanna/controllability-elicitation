"""Kimi K3 effort=none first-token top-20 logprobs across logprob-capable providers. 24 prompts x 7 providers."""
import asyncio, json, os
from pathlib import Path
from openai import AsyncOpenAI
from cotcontrol.eval.data import load_dataset
OUT = Path("scratch/provider_fidelity"); c = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
PROVS = ["morph","makora","digitalocean","phala","parasail","fireworks","alibaba"]
rows = load_dataset("all", mode="random", split="val", max_samples=24, subsample_seed=4)
PROMPTS = [r["question"] + "\n\nAnswer concisely." for r in rows]
sem = asyncio.Semaphore(48)
async def one(prov, pi, rep=0):
    async with sem:
        for attempt in range(3):
            try:
                r = await c.chat.completions.create(model="moonshotai/kimi-k3", messages=[{"role":"user","content":PROMPTS[pi]}], temperature=0, max_tokens=1,
                        logprobs=True, top_logprobs=5, extra_body={"provider":{"only":[prov],"allow_fallbacks":False},"reasoning":{"effort":"none"}})
                d = r.model_dump(); lp = (d["choices"][0].get("logprobs") or {}).get("content") or []
                return {"prov":prov,"pi":pi,"rep":rep,"provider_reported":d.get("provider"),"top":[(x["token"],x["logprob"]) for x in lp[0]["top_logprobs"]] if lp else None,"raw":d,"error":None}
            except Exception as e:
                if attempt == 2: return {"prov":prov,"pi":pi,"rep":rep,"provider_reported":None,"top":None,"raw":None,"error":str(e)[:300]}
                await asyncio.sleep(2)
async def main():
    res = await asyncio.gather(*[one(p,pi,rep) for p in PROVS for pi in range(len(PROMPTS)) for rep in (0,1)])
    with open(OUT/"kimi_logprobs_raw_v2.jsonl","w") as f:
        for r in res: f.write(json.dumps(r)+"\n")
    print("done", len(res), "errors:", sum(1 for r in res if r["error"]), "no_top:", sum(1 for r in res if r["top"] is None and not r["error"]))
asyncio.run(main())
