"""Ways to reduce GLM-5.3 thinking on z-ai: effort levels vs reasoning.max_tokens budgets; check budget-mapping hypothesis
(does 'high' scale with max_tokens?) and rough accuracy per condition. 16 val questions."""
import asyncio, os, json, re, statistics
from pathlib import Path
from cotcontrol.inference.openrouter import GenerateConfig, generate_async
from cotcontrol.eval.data import load_dataset
OUT=Path("scratch/provider_fidelity/glm_levers"); OUT.mkdir(exist_ok=True)
rows=load_dataset("all", mode="random", split="val", max_samples=16, subsample_seed=41)
P=[[{"role":"user","content":r["question"]+"\n\nEnd with ANSWER: <letter or value>."}] for r in rows]
GOLD=[(r.get("correct_answer_letter") or r.get("correct_answer_raw") or "").strip() for r in rows]
CONDS={"unset_mt16k":(None,16000),"high_mt16k":({"effort":"high"},16000),"high_mt40k":({"effort":"high"},40000),
       "low_mt16k":({"effort":"low"},16000),"budget2k_mt16k":({"max_tokens":2000},16000),"budget6k_mt16k":({"max_tokens":6000},16000)}
def rt(r):
    u=r["metadata"][0]["usage"] or {}; x=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"); return x if x is not None else len(r["reasoning"][0] or "")//4
def correct(out,gold):
    m=re.findall(r"ANSWER:\s*\**\s*([A-Za-z0-9][^\n*]*)",out or ""); 
    if not m or not gold: return None
    a=m[-1].strip().rstrip(".").lower(); g=gold.lower()
    return a==g or a[:1]==g[:1] and len(g)==1
async def run(name,extra,mt):
    cfg=GenerateConfig(temperature=0.0,max_tokens=mt,max_concurrency=16,provider={"only":["z-ai"],"allow_fallbacks":False},extra_body={"reasoning":extra} if extra else None)
    res=await generate_async(P,"z-ai/glm-5.3",cfg,save_path=OUT/f"{name}.json",progress=False)
    ok=[r for r in res if not r["metadata"][0]["error"]]; t=[rt(r) for r in ok]
    fin={}; 
    for r in ok: fin[r["metadata"][0]["finish_reason"]]=fin.get(r["metadata"][0]["finish_reason"],0)+1
    acc=[correct(r["output"][0],g) for r,g in zip(res,GOLD)]; acc=[a for a in acc if a is not None]
    err=[r["metadata"][0]["error"] for r in res if r["metadata"][0]["error"]]
    print(f"{name:16s} reasoning tok median={statistics.median(t) if t else 0:6.0f} mean={statistics.mean(t) if t else 0:6.0f} finish={fin} acc={sum(acc)}/{len(acc)} errs={len(err)}"+(f" e.g. {err[0][:100]}" if err else ""),flush=True)
async def main(): await asyncio.gather(*[run(n,e,m) for n,(e,m) in CONDS.items()])
asyncio.run(main())
