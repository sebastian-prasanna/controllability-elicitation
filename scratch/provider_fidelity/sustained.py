"""Sustained realistic load per pin: 300 val questions, full reasoning, max_tokens 16000, concurrency 200, production
generate_async (with its retry loop). Counts HTTP 429/5xx via httpx hook, final errors, throughput. Saves per-pin gen files."""
import asyncio, os, json, time, sys, httpx
from collections import Counter
from pathlib import Path
from openai import AsyncOpenAI
import cotcontrol.inference.openrouter as orr
from cotcontrol.inference.openrouter import GenerateConfig, generate_async
from cotcontrol.eval.data import load_dataset
OUT=Path("scratch/provider_fidelity/sustained"); OUT.mkdir(exist_ok=True)
PINS=[("moonshotai/kimi-k3","moonshotai",None),("openai/gpt-oss-120b","dekallm",["bf16"]),("openai/gpt-oss-120b","groq",None),
      ("openai/gpt-oss-20b","groq",None),("qwen/qwen3-32b","siliconflow",None),("qwen/qwen3-8b","alibaba",None),
      ("z-ai/glm-5.3","z-ai",None),("z-ai/glm-5.3-flash","sail-research",["fp8"]),("deepseek/deepseek-v4-pro-0813","alibaba",None)]
rows=load_dataset("all", mode="random", split="val", max_samples=300, subsample_seed=7)
prompts=[[{"role":"user","content":r["question"]+"\n\nEnd with ANSWER: <letter or value>."}] for r in rows]
status=Counter()
async def on_resp(resp: httpx.Response):
    try:
        b=json.loads(resp.request.content); tag=f"{b['model'].split('/')[1]}@{b['provider']['only'][0]}"
    except Exception: tag="?"
    status[(tag,resp.status_code)]+=1
def make_client():
    return AsyncOpenAI(base_url=orr.OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"], timeout=600,
                       http_client=httpx.AsyncClient(event_hooks={"response":[on_resp]}, timeout=600, limits=httpx.Limits(max_connections=2000, max_keepalive_connections=400)))
client=make_client(); orr.get_client=lambda: client
async def run(model,prov,q):
    tag=f"{model.split('/')[1]}@{prov}"
    cfg=GenerateConfig(temperature=0.0, max_tokens=16000, max_concurrency=200,
                       provider={"only":[prov],"allow_fallbacks":False,**({"quantizations":q} if q else {})})
    t0=time.time()
    res=await generate_async(prompts, model, cfg, save_path=OUT/f"{tag}.json", progress=False)
    dt=time.time()-t0
    errs=sum(1 for r in res if r["metadata"][0]["error"]); fin=Counter(r["metadata"][0]["finish_reason"] for r in res)
    toks=[((r["metadata"][0]["usage"] or {}).get("completion_tokens") or 0) for r in res]
    provs=Counter((r["metadata"][0]["raw_response"] or {}).get("provider") for r in res if r["metadata"][0]["raw_response"])
    print(f"{tag:34s} wall={dt/60:5.1f}min  req/min={300/dt*60:5.1f}  final_errors={errs:3d}  finish={dict(fin)}  out_tok_total={sum(toks)/1e3:.0f}k p50={sorted(toks)[150]}  http={ {k[1]:v for k,v in status.items() if k[0]==tag} }  served_by={dict(provs)}", flush=True)
async def main():
    # stagger client selection: run sequential per pin creation but concurrent execution
    await asyncio.gather(*[run(*p) for p in PINS])
asyncio.run(main())
