"""Shared loader for the final RL sweep figures. SFT point = RL step-0 checkpoint (the x320 SFT donor evaluated with the
identical val eval: T=0, 16k, 200 q). SFT+RL point = neighbour-smoothed (±10 steps) best checkpoint under gate A
(raw accuracy >= 0.8 x step 0), exactly as rl/runs/sweep23_conjunctive/summarize.py. Both for held-out and in-dist blocks.
Stopped runs are read from their _early partial evals and flagged partial=True."""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "rl/runs/sweep23_conjunctive/summary_cache.json"
KS = ["k100", "k300", "k500", "k700", "k1k", "k3k", "k10k", "k30k", "k100k", "kall"]
KVAL = {"k100": 100, "k300": 300, "k500": 500, "k700": 700, "k1k": 1000, "k3k": 3000, "k10k": 10000, "k30k": 30000, "k100k": 100000, "kall": None}
PARTIAL = {"f20b-k100k": "f20b-k100k-le130", "f20b-k30k": "f20b-k30k-le140", "f20b-kall": "f20b-kall-le180",
           "f120b-k100k": "f120b-k100k-le170", "fq32b-kall": "fq32b-kall-le60", "s20b-k3k-s2": "s20b-k3k-s2-le120"}
_cache = json.loads(CACHE.read_text())

def cache_key(name, t1=False):
    """Cache key for a run: stopped runs map to their _early partial; t1=True selects the T=1.0 re-eval (<key>-t1)."""
    key = PARTIAL.get(name, name)
    return f"{key}-t1" if t1 else key

def _block(name, block, t1=False):
    d = _cache.get(cache_key(name, t1), {}).get(block) or {}
    return {int(k): v for k, v in d.items()}, name in PARTIAL

def point(name, block="heldout", metric="honest", t1=False):
    """-> dict(sft, rl, rl_step, partial) or None if the run has no eval. t1=True reads the T=1.0 re-evaluation."""
    H, partial = _block(name, block, t1)
    if not H: return None
    steps = sorted(H); s0 = H[steps[0]]
    sm = {s: np.mean([H[t][metric] for t in steps if abs(t - s) <= 10]) for s in steps if s > 0}
    ok = [s for s in sm if H[s]["acc"] >= 0.8 * s0["acc"]]
    best = max(ok, key=sm.get) if ok else max(sm, key=sm.get)
    return dict(sft=s0[metric], rl=sm[best], rl_step=best, partial=partial, gated=bool(ok))
