"""qwen3-32b: finish=None (cut stream) rate vs concurrency on siliconflow, and deepinfra/fp8 alternative. 100 real req, max_tokens 16000."""
import asyncio, os, json, time, statistics
from collections import Counter
from pathlib import Path
from cotcontrol.inference.openrouter import GenerateConfig, generate_async
from cotcontrol.eval.data import load_dataset
OUT=Path("scratch/provider_fidelity/qwen32b_cut"); OUT.mkdir(exist_ok=True)
rows=load_dataset("all", mode="random", split="val", max_samples=100, subsample_seed=31)
P=[[{"role":"user","content":r["question"]+"\n\nEnd with ANSWER: <letter or value>."}] for r in rows]
async def level(prov,q,conc):
    t0=time.time(); lat=[]
    def on_result(r): lat.append((time.time()-t0, r["finish_reason"], ((r["usage"] or {}).get("completion_tokens") or 0)))
    res=await generate_async(P,"qwen/qwen3-32b",GenerateConfig(temperature=0.0,max_tokens=16000,max_concurrency=conc,provider={"only":[prov],"allow_fallbacks":False,**({"quantizations":q} if q else {})}),save_path=OUT/f"{prov}_c{conc}.json",progress=False,on_result=on_result)
    wall=time.time()-t0; fin=Counter(r["metadata"][0]["finish_reason"] for r in res); errs=sum(1 for r in res if r["metadata"][0]["error"])
    none_lat=[l for l,f,t in lat if f is None]; ok_lat=[l for l,f,t in lat if f=="stop"]
    print(f"{prov:12s} conc={conc:3d} wall={wall/60:5.1f}min finish={dict(fin)} errors={errs} | latency stop p50={statistics.median(ok_lat)/60 if ok_lat else 0:4.1f}min max={max(ok_lat)/60 if ok_lat else 0:4.1f}min | None: n={len(none_lat)} lat p50={statistics.median(none_lat)/60 if none_lat else 0:4.1f}min min={min(none_lat)/60 if none_lat else 0:4.1f}min | tok/s per stream={statistics.median([t/l for l,f,t in lat if l>0]):5.1f}",flush=True)
async def main():
    async def sf():
        await level("siliconflow",None,50); await asyncio.sleep(30); await level("siliconflow",None,100)
    await asyncio.gather(sf(), level("deepinfra",["fp8"],100))
asyncio.run(main())
