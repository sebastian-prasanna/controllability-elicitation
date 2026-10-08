import asyncio, os, json, re, time
from openai import AsyncOpenAI
c=AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=240, max_retries=0)
DS="deepseek/deepseek-v4-pro-0813"; G120="openai/gpt-oss-120b"; FL="z-ai/glm-5.3-flash"
CANDS=[(DS,"streamlake",None),(DS,"alibaba",None),(DS,"baidu",["fp8"]),(DS,"novita",["fp8"]),(DS,"fireworks",None),(DS,"together",None),
 (G120,"dekallm",["bf16"]),(G120,"crusoe",["bf16"]),(G120,"sambanova",None),(G120,"cerebras",None),
 (FL,"baseten",["fp8"]),(FL,"novita",["fp8"]),(FL,"parasail",["fp8"]),(FL,"relace",None),(FL,"wafer",None),(FL,"sail-research",["fp8"]),
 ("qwen/qwen3-32b","deepinfra",["fp8"])]
NOTHINK=[("z-ai/glm-5.3","z-ai",None),(FL,"siliconflow",None),("qwen/qwen3-32b","siliconflow",None),(DS,"streamlake",None),(DS,"alibaba",None)]
RAND="Pick a random integer between 1 and 100. Reply with only the number, nothing else."
out=[]
async def call(model,prov,q,prompt,T,mt,kind,extra=None):
    body={"provider":{"only":[prov],"allow_fallbacks":False,**({"quantizations":q} if q else {})},**(extra or {})}
    t0=time.time()
    try:
        r=await c.chat.completions.create(model=model,messages=[{"role":"user","content":prompt}],temperature=T,top_p=1.0,max_tokens=mt,extra_body=body)
        d=r.model_dump(); m=d["choices"][0]["message"]
        rec={"kind":kind,"model":model,"prov":prov,"q":q,"T":T,"content":m.get("content"),"reasoning":m.get("reasoning"),"provider_reported":d.get("provider"),"secs":round(time.time()-t0,1),"err":None,"status":None}
    except Exception as e:
        s=str(e); code=re.search(r"Error code: (\d+)",s); rec={"kind":kind,"model":model,"prov":prov,"q":q,"T":T,"content":None,"reasoning":None,"provider_reported":None,"secs":round(time.time()-t0,1),"err":s[:220],"status":int(code.group(1)) if code else None}
    out.append(rec); return rec
async def main():
    tasks=[]
    for model,prov,q in CANDS:
        tasks+=[call(model,prov,q,f"Question {i}: what is {i}+{i*3}? Reply with just the number.",0.0,300,"burst") for i in range(100)]
        tasks+=[call(model,prov,q,RAND,T,400,"rand") for T in (0.0,1.9) for _ in range(10)]
    for model,prov,q in NOTHINK:
        tasks+=[call(model,prov,q,RAND,T,40,"nothink",{"reasoning":{"enabled":False}}) for T in (0.0,1.9) for _ in range(12)]
    await asyncio.gather(*tasks)
    with open("scratch/provider_fidelity/pin_stress2_raw.jsonl","w") as f:
        for r in out: f.write(json.dumps(r)+"\n")
    def nums(rs): return [(re.findall(r"\d+",(r["content"] or "")) or ["?"])[-1] for r in rs if not r["err"]]
    print(f"{'model':30s} {'provider':13s} {'ok':>3s} {'429':>4s} {'oth':>4s} {'p50s':>5s} | {'randT0':>7s} {'randT1.9':>8s} {'gib':>3s}")
    for model,prov,q in CANDS:
        rs=[r for r in out if r["model"]==model and r["prov"]==prov and r["q"]==q]
        b=[r for r in rs if r["kind"]=="burst"]; ok=[r for r in b if not r["err"]]; secs=sorted(r["secs"] for r in ok) or [0]
        r0=[r for r in rs if r["kind"]=="rand" and r["T"]==0.0]; r2=[r for r in rs if r["kind"]=="rand" and r["T"]==1.9]
        gib=sum(1 for r in r2 if not r["err"] and re.search(r"[^\x00-\x7f]{3}|[A-Za-z]{18,}",(r["reasoning"] or r["content"] or "")))
        err=next((r["err"] for r in rs if r["err"]),None)
        print(f"{model:30s} {prov:13s} {len(ok):3d} {sum(1 for r in b if r['status']==429):4d} {sum(1 for r in b if r['err'] and r['status']!=429):4d} {secs[len(secs)//2]:5.1f} | {len(set(nums(r0))):2d}/{len(nums(r0)):<3d} {len(set(nums(r2))):3d}/{len(nums(r2)):<4d} {gib:3d}"+(f"  e.g. {err[:100]}" if err else ""))
    print("\nthinking disabled (reasoning.enabled=false), random-int distinct:")
    for model,prov,q in NOTHINK:
        rs=[r for r in out if r["model"]==model and r["prov"]==prov and r["kind"]=="nothink"]
        r0=[r for r in rs if r["T"]==0.0]; r2=[r for r in rs if r["T"]==1.9]
        err=next((r["err"] for r in rs if r["err"]),None); rz=sum(1 for r in rs if not r["err"] and r["reasoning"])
        print(f"  {model:28s} {prov:13s} T0 {len(set(nums(r0))):2d}/{len(nums(r0)):<3d}  T1.9 {len(set(nums(r2))):2d}/{len(nums(r2)):<3d} still_reasoning={rz}/{len(rs)}"+(f"  e.g. {err[:110]}" if err else ""))
asyncio.run(main())
