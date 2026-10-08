"""Does the endpoint act on temperature? (a) random-integer prompt: distinct answers over 12 calls at T=0/1/2.
(b) normal prompt at T=2 vs T=0: does text degrade? Saves raw to temp_test_raw.jsonl."""
import asyncio, os, json, re
from collections import Counter
from openai import AsyncOpenAI
c=AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=180)
EPS=[("moonshotai/kimi-k3","moonshotai",{"reasoning":{"effort":"none"}}),("moonshotai/kimi-k3","phala",{"reasoning":{"effort":"none"}}),
     ("moonshotai/kimi-k3","parasail",{"reasoning":{"effort":"none"}}),("moonshotai/kimi-k3","modal",{"reasoning":{"effort":"none"}}),
     ("openai/gpt-oss-120b","deepinfra",{"provider_q":["fp8"]}),("openai/gpt-oss-120b","akashml",{}),("openai/gpt-oss-120b","cerebras",{})]
RAND="Pick a random integer between 1 and 100. Reply with only the number, nothing else."
NORM="Explain in three sentences why the sky is blue."
sem=asyncio.Semaphore(48); out=[]
async def call(model,prov,extra,prompt,T,mt):
    body={"provider":{"only":[prov],"allow_fallbacks":False}}
    if "provider_q" in extra: body["provider"]["quantizations"]=extra["provider_q"]
    if "reasoning" in extra: body["reasoning"]=extra["reasoning"]
    async with sem:
        try:
            r=await c.chat.completions.create(model=model,messages=[{"role":"user","content":prompt}],temperature=T,top_p=1.0,max_tokens=mt,extra_body=body)
            m=r.choices[0].message.model_dump(); rec={"model":model,"prov":prov,"T":T,"prompt":prompt[:20],"content":m.get("content"),"reasoning":m.get("reasoning"),"err":None}
        except Exception as e: rec={"model":model,"prov":prov,"T":T,"prompt":prompt[:20],"content":None,"reasoning":None,"err":str(e)[:200]}
    out.append(rec); return rec
async def main():
    tasks=[]
    for model,prov,extra in EPS:
        for T in (0.0,1.0,2.0):
            tasks+=[call(model,prov,extra,RAND,T,400) for _ in range(12)]
            tasks+=[call(model,prov,extra,NORM,T,120) for _ in range(3)]
    await asyncio.gather(*tasks)
    with open("scratch/provider_fidelity/temp_test_raw.jsonl","w") as f:
        for r in out: f.write(json.dumps(r)+"\n")
    for model,prov,extra in EPS:
        print(f"\n=== {model} @ {prov}")
        for T in (0.0,1.0,2.0):
            rs=[r for r in out if r["model"]==model and r["prov"]==prov and r["T"]==T and r["prompt"]==RAND[:20]]
            errs=sum(1 for r in rs if r["err"])
            nums=[(re.findall(r"\d+",(r["content"] or "")) or ["?"])[-1] for r in rs if not r["err"]]
            cnt=Counter(nums)
            print(f"  T={T}: random-int distinct={len(cnt)}/{len(nums)} errs={errs} -> {dict(cnt.most_common(6))}")
            ns=[r for r in out if r["model"]==model and r["prov"]==prov and r["T"]==T and r["prompt"]==NORM[:20] and not r["err"]]
            for r in ns[:2]:
                txt=(r["reasoning"] or r["content"] or "")
                words=re.findall(r"[A-Za-z]+",txt); print(f"       T={T} sample: {txt[:110]!r}")
        e=[r["err"] for r in out if r["model"]==model and r["prov"]==prov and r["err"]]
        if e: print("  example err:", e[0][:160])
asyncio.run(main())
