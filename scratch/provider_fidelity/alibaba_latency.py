"""Alibaba qwen3-8b: steady-state throughput vs concurrency via per-request latency (on_result timestamps).
conc 50 x 100 req (2 waves), then conc 150 x 150 req (1 wave). Steady-state req/min = conc / mean_latency."""
import asyncio, os, json, time, statistics, httpx
from collections import Counter
from pathlib import Path
from openai import AsyncOpenAI
import cotcontrol.inference.openrouter as orr
from cotcontrol.inference.openrouter import GenerateConfig, generate_async
from cotcontrol.eval.data import load_dataset
OUT=Path("scratch/provider_fidelity/conc_sweep")
rows=load_dataset("all", mode="random", split="val", max_samples=150, subsample_seed=13)
P=[[{"role":"user","content":r["question"]+"\n\nEnd with ANSWER: <letter or value>."}] for r in rows]
status=Counter(); cur={"c":None}
async def on_resp(resp): status[(cur["c"],resp.status_code)]+=1
client=AsyncOpenAI(base_url=orr.OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"], timeout=900, max_retries=0,
                   http_client=httpx.AsyncClient(event_hooks={"response":[on_resp]}, timeout=900, limits=httpx.Limits(max_connections=1000)))
orr.get_client=lambda: client
async def level(conc,n):
    cur["c"]=conc; t0=time.time(); done=[]
    def on_result(r): done.append((time.time()-t0, ((r["usage"] or {}).get("completion_tokens") or 0), r["error"]))
    await generate_async(P[:n], "qwen/qwen3-8b", GenerateConfig(temperature=0.0,max_tokens=4000,max_concurrency=conc,provider={"only":["alibaba"],"allow_fallbacks":False}), save_path=OUT/f"qwen3-8b@alibaba_lat_c{conc}.json", progress=False, on_result=on_result)
    wall=time.time()-t0
    # latency: for the first `conc` requests (wave 1) completion time == latency. Use wave-1 only.
    w1=sorted(done)[:conc]; lat=[d[0] for d in w1]; tok=[d[1] for d in w1]
    ss_rpm=conc/statistics.mean(lat)*60; ss_tpm=sum(tok)/statistics.mean(lat)*60/len(tok)*conc/1e3
    print(f"conc={conc:3d} n={n} wall={wall/60:5.1f}min  wave1 latency mean={statistics.mean(lat)/60:4.1f}min p90={sorted(lat)[int(len(lat)*.9)]/60:4.1f}min  "
          f"mean tok={statistics.mean(tok):5.0f} -> per-stream {statistics.mean(t/l for t,l in zip(tok,lat)):4.1f} tok/s | steady-state est {ss_rpm:5.1f} req/min {ss_tpm:5.0f}k tok/min | 429s={status[(conc,429)]} final_err={sum(1 for d in done if d[2])}", flush=True)
async def main():
    await level(50,100); await asyncio.sleep(30); await level(150,150)
asyncio.run(main())
