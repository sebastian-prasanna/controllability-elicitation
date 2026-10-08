"""Concurrency sweep for the two 429-heavy pins: 100 real requests at conc 50/100/200 (sequential per pin, pins in parallel).
Counts HTTP 429s, wall time, and per-request latency (all requests start ~together at conc>=100)."""
import asyncio, os, json, time, httpx
from collections import Counter
from pathlib import Path
from openai import AsyncOpenAI
import cotcontrol.inference.openrouter as orr
from cotcontrol.inference.openrouter import GenerateConfig, generate_async
from cotcontrol.eval.data import load_dataset
OUT=Path("scratch/provider_fidelity/conc_sweep"); OUT.mkdir(exist_ok=True)
rows=load_dataset("all", mode="random", split="val", max_samples=100, subsample_seed=11)
prompts=[[{"role":"user","content":r["question"]+"\n\nEnd with ANSWER: <letter or value>."}] for r in rows]
status=Counter(); cur={}
async def on_resp(resp):
    try: b=json.loads(resp.request.content); tag=f"{b['model'].split('/')[1]}@{b['provider']['only'][0]}"
    except Exception: tag="?"
    status[(tag,cur.get(tag),resp.status_code)]+=1
client=AsyncOpenAI(base_url=orr.OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"], timeout=600, max_retries=0,
                   http_client=httpx.AsyncClient(event_hooks={"response":[on_resp]}, timeout=600, limits=httpx.Limits(max_connections=1000)))
orr.get_client=lambda: client
async def pin(model,prov):
    tag=f"{model.split('/')[1]}@{prov}"
    for conc in (50,100,200):
        cur[tag]=conc; t0=time.time()
        res=await generate_async(prompts, model, GenerateConfig(temperature=0.0,max_tokens=4000,max_concurrency=conc,provider={"only":[prov],"allow_fallbacks":False}), save_path=OUT/f"{tag}_c{conc}.json", progress=False)
        dt=time.time()-t0; errs=sum(1 for r in res if r["metadata"][0]["error"]); toks=sum(((r["metadata"][0]["usage"] or {}).get("completion_tokens") or 0) for r in res)
        n429=status[(tag,conc,429)]; n200=status[(tag,conc,200)]
        print(f"{tag:22s} conc={conc:3d}  wall={dt:6.0f}s  req/min={100/dt*60:5.1f}  tok/min={toks/dt*60/1e3:6.0f}k  http200={n200} http429={n429}  429/req={n429/100:.2f}  final_errors={errs}", flush=True)
        await asyncio.sleep(20)  # let rate-limit windows reset
async def main(): await asyncio.gather(pin("qwen/qwen3-8b","alibaba"), pin("openai/gpt-oss-20b","groq"))
asyncio.run(main())
