"""What does reasoning.effort do on each pinned endpoint? 12 val q x {unset,low,medium,high,xhigh}; median reasoning tokens."""
import asyncio, os, json, statistics
from pathlib import Path
from cotcontrol.inference.openrouter import GenerateConfig, generate_async
from cotcontrol.eval.data import load_dataset
OUT=Path("scratch/provider_fidelity/effort_map"); OUT.mkdir(exist_ok=True)
PINS=[("openai/gpt-oss-120b","groq",None),("openai/gpt-oss-20b","groq",None),("z-ai/glm-5.3","z-ai",None),
      ("z-ai/glm-5.3-flash","sail-research",["fp8"]),("deepseek/deepseek-v4-pro-0813","alibaba",None),("moonshotai/kimi-k3","moonshotai",None)]
rows=load_dataset("all", mode="random", split="val", max_samples=12, subsample_seed=21)
P=[[{"role":"user","content":r["question"]+"\n\nEnd with ANSWER: <letter or value>."}] for r in rows]
def rt(r):
    u=r["metadata"][0]["usage"] or {}; x=(u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    return x if x is not None else len(r["reasoning"][0] or "")//4
async def one(model,prov,q,eff):
    cfg=GenerateConfig(temperature=1.0,max_tokens=16000,max_concurrency=12,provider={"only":[prov],"allow_fallbacks":False,**({"quantizations":q} if q else {})},
                       extra_body={"reasoning":{"effort":eff}} if eff!="unset" else None)
    res=await generate_async(P,model,cfg,save_path=OUT/f"{model.split('/')[1]}@{prov}_{eff}.json",progress=False)
    errs=[r["metadata"][0]["error"] for r in res if r["metadata"][0]["error"]]
    return eff,[rt(r) for r in res if not r["metadata"][0]["error"]],errs
async def pin(model,prov,q):
    outs=await asyncio.gather(*[one(model,prov,q,e) for e in ["unset","low","medium","high","xhigh"]])
    line=f"{model.split('/')[1]+'@'+prov:34s}"
    for eff,t,errs in outs:
        line+=f" {eff}={statistics.median(t) if t else float('nan'):6.0f}" + (f"(err{len(errs)})" if errs else "")
    print(line,flush=True)
    for eff,t,errs in outs:
        if errs: print("    ",eff,"err e.g.",errs[0][:140])
async def main(): await asyncio.gather(*[pin(*p) for p in PINS])
asyncio.run(main())
