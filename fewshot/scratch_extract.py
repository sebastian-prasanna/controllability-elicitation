"""Scratch: extract compact per-rollout records from big fewshot JSONs."""
import json, gc, sys, os

RUNS = {
    "k1": "fewshot/runs/gptoss120b_k1/2026-08-30T02-18-06_openai_gpt-oss-120b_all_all.json",
    "k2": "fewshot/runs/gptoss120b_k2/2026-08-30T02-29-46_openai_gpt-oss-120b_all_all.json",
    "k3": "fewshot/runs/gptoss120b_k3/2026-08-30T02-51-08_openai_gpt-oss-120b_all_all.json",
    "base": "baselines/gptoss120b/2026-08-25T22-39-15_openai_gpt-oss-120b_all_all.json",
}
ROOT = "/root/controllability-elicitation"
OUT = os.path.join(ROOT, "fewshot", "scratch_compact")
os.makedirs(OUT, exist_ok=True)

for tag, rel in RUNS.items():
    outp = os.path.join(OUT, f"{tag}.json")
    if os.path.exists(outp):
        print(tag, "already done"); continue
    path = os.path.join(ROOT, rel)
    print("loading", tag, flush=True)
    with open(path) as f:
        data = json.load(f)
    results = data["results"] if isinstance(data, dict) else data
    recs = []
    for i, item in enumerate(results):
        s = (item.get("samples") or [{}])[0]
        rt = s.get("reasoning_text_graded")
        rz = s.get("reasoning")
        out = s.get("output")
        recs.append({
            "pos": i,
            "id": item.get("id"),
            "dataset": item.get("dataset"),
            "mode": item.get("mode"),
            "compliance": s.get("compliance"),
            "correct": s.get("correct"),
            "error": s.get("error"),
            "out_none": out is None,
            "out_tail": (out or "")[-80:],
            "rt_none": rt is None,
            "rt_len": len(rt) if rt else 0,
            "rz_len": len(rz) if rz else 0,
            "rt_head": (rt or "")[:600],
            "rt_tail": (rt or "")[-300:],
        })
    del data, results
    gc.collect()
    with open(outp, "w") as f:
        json.dump(recs, f)
    print(tag, len(recs), "records", flush=True)
    del recs; gc.collect()
print("done")
