"""Per candidate endpoint: 100 concurrent short calls (rate-limit/latency), 6x same prompt at T=0 (repeatability),
12x random-int at T=0 and T=2 (temperature honored). Saves pin_stress_raw.jsonl."""
import asyncio, os, json, re, time, statistics
from collections import Counter
from openai import AsyncOpenAI
c=AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=240, max_retries=0)
CANDS=[("deepseek/deepseek-v4-pro-0813","deepseek",None),("moonshotai/kimi-k3","moonshotai",None),("moonshotai/kimi-k3","modal",None),
 ("openai/gpt-oss-120b","akashml",["bf16"]),("openai/gpt-oss-120b","deepinfra",["fp8"]),("openai/gpt-oss-120b","groq",None),
 ("openai/gpt-oss-20b","dekallm",["bf16"]),("openai/gpt-oss-20b","deepinfra",["bf16"]),("openai/gpt-oss-20b","groq",None),
 ("qwen/qwen3-32b","siliconflow",None),("qwen/qwen3-8b","alibaba",None),("z-ai/glm-5.3","z-ai",None),
 ("z-ai/glm-5.3-flash","fireworks",None),("z-ai/glm-5.3-flash","siliconflow",None),("z-ai/glm-5.3-flash","together",None)]
RAND="Pick a random integer between 1 and 100. Reply with only the number, nothing else."
REP="Name three prime numbers greater than 50 and explain briefly why each is prime."
out=[]
async def call(model,prov,q,prompt,T,mt,kind):
    body={"provider":{"only":[prov],"allow_fallbacks":False,**({"quantizations":q} if q else {})}}
    t0=time.time()
    try:
        r=await c.chat.completions.create(model=model,messages=[{"role":"user","content":prompt}],temperature=T,top_p=1.0,max_tokens=mt,extra_body=body)
        d=r.model_dump(); m=d["choices"][0]["message"]
        rec={"kind":kind,"model":model,"prov":prov,"q":q,"T":T,"content":m.get("content"),"reasoning":m.get("reasoning"),"provider_reported":d.get("provider"),"finish":d["choices"][0]["finish_reason"],"secs":round(time.time()-t0,1),"err":None,"status":None}
    except Exception as e:
        s=str(e); code=re.search(r"Error code: (\d+)",s); rec={"kind":kind,"model":model,"prov":prov,"q":q,"T":T,"content":None,"reasoning":None,"provider_reported":None,"finish":None,"secs":round(time.time()-t0,1),"err":s[:200],"status":int(code.group(1)) if code else None}
    out.append(rec); return rec
async def run_cand(model,prov,q):
    burst=[call(model,prov,q,f"Question {i}: what is {i}+{i*3}? Reply with just the number.",0.0,300,"burst") for i in range(100)]
    reps=[call(model,prov,q,REP,0.0,200,"rep") for _ in range(6)]
    rnd=[call(model,prov,q,RAND,T,400,"rand") for T in (0.0,2.0) for _ in range(12)]
    await asyncio.gather(*burst,*reps,*rnd)
async def main():
    await asyncio.gather(*[run_cand(*cnd) for cnd in CANDS])
    with open("scratch/provider_fidelity/pin_stress_raw.jsonl","w") as f:
        for r in out: f.write(json.dumps(r)+"\n")
    print(f"{'model':30s} {'provider':12s} {'burst ok':>8s} {'429':>4s} {'other':>5s} {'p50s':>5s} {'p95s':>5s} | {'rep T0 distinct/6':>17s} | {'rand T0':>7s} {'rand T2':>7s} {'T2 gibberish':>12s}")
    for model,prov,q in CANDS:
        rs=[r for r in out if r["model"]==model and r["prov"]==prov and r["q"]==q]
        b=[r for r in rs if r["kind"]=="burst"]; ok=[r for r in b if not r["err"]]
        n429=sum(1 for r in b if r["status"]==429); noth=sum(1 for r in b if r["err"] and r["status"]!=429)
        secs=sorted(r["secs"] for r in ok) or [0]
        rep=[r for r in rs if r["kind"]=="rep" and not r["err"]]
        opens=len({((r["reasoning"] or r["content"] or "")[:80]) for r in rep})
        def nums(T): return [(re.findall(r"\d+",(r["content"] or "")) or ["?"])[-1] for r in rs if r["kind"]=="rand" and r["T"]==T and not r["err"]]
        n0,n2=nums(0.0),nums(2.0)
        t2=[r for r in rs if r["kind"]=="rand" and r["T"]==2.0 and not r["err"]]
        gib=sum(1 for r in t2 if re.search(r"[^\x00-\x7f]{3}|[A-Za-z]{18,}",(r["reasoning"] or r["content"] or "")))
        err=next((r["err"] for r in rs if r["err"]),None)
        print(f"{model:30s} {prov:12s} {len(ok):8d} {n429:4d} {noth:5d} {secs[len(secs)//2]:5.1f} {secs[int(len(secs)*.95)-1] if len(secs)>1 else secs[0]:5.1f} | {opens:17d} | {len(set(n0)):3d}/{len(n0):<3d} {len(set(n2)):3d}/{len(n2):<3d} {gib:12d}" + (f"   e.g. {err[:90]}" if err else ""))
asyncio.run(main())
