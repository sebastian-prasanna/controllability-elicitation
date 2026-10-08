#!/usr/bin/env python3
"""Status + provider inference for nemotron run dirs (rollouts.jsonl does not store the provider;
infer it from usage.cost vs the per-endpoint price pairs seen on the OpenRouter endpoints API 2026-10-05).

    python3 status.py <run_dir> [...]
"""
import json, sys, statistics, collections
from pathlib import Path

PRICES = {"DeepInfra": (0.5e-6, 2.2e-6), "BaseTen": (0.6e-6, 2.4e-6), "Venice": (0.625e-6, 3.125e-6)}

def provider(u):
    if not u or not u.get("cost"):
        return None
    pt, ct, cost = u.get("prompt_tokens", 0), u.get("completion_tokens", 0), float(u["cost"])
    best = min(PRICES, key=lambda k: abs(PRICES[k][0] * pt + PRICES[k][1] * ct - cost))
    err = abs(PRICES[best][0] * pt + PRICES[best][1] * ct - cost) / max(cost, 1e-12)
    return best if err < 0.02 else f"?({err:.2f})"

for d in sys.argv[1:]:
    p = Path(d) / "rollouts.jsonl"
    if not p.exists():
        print(f"== {d}: no rollouts yet"); continue
    R = [json.loads(l) for l in open(p)]
    errs = [r for r in R if r.get("error")]
    rt = [r.get("reasoning_tokens") or 0 for r in R if not r.get("error")]
    empty_text = sum(1 for r in R if not r.get("error") and not (r.get("reasoning") or "").strip())
    fin = collections.Counter(r.get("finish_reason") for r in R)
    prov = collections.Counter(provider(r.get("usage")) for r in R)
    inv = sum(1 for r in R if not r["y_valid"])
    cost = sum(float((r.get("usage") or {}).get("cost") or 0) for r in R)
    x1 = [r for r in R if r["x"] == 1 and r["y_valid"]]; x0 = [r for r in R if r["x"] == 0 and r["y_valid"]]
    py1 = sum(r["y"] for r in x1) / len(x1) if x1 else float("nan"); py0 = sum(r["y"] for r in x0) / len(x0) if x0 else float("nan")
    print(f"== {d}\n  n={len(R)} errors={len(errs)} invalid={inv} ({inv/len(R):.1%}) empty_reasoning_text={empty_text} "
          f"rtok med={statistics.median(rt) if rt else None} max={max(rt) if rt else None} finish={dict(fin)}\n"
          f"  providers(inferred)={dict(prov)} policy_cost=${cost:.3f}  P(Y|X=1)={py1:.2f} (n={len(x1)}) P(Y|X=0)={py0:.2f} (n={len(x0)})")
    if errs:
        print("  error samples:", collections.Counter(e["error"][:80] for e in errs).most_common(3))
    mon = Path(d) / "monitor.jsonl"
    if mon.exists():
        print(f"  monitor rows: {sum(1 for _ in open(mon))}")
