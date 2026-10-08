#!/usr/bin/env python3
"""Register-entry anatomy for a Kimi run dir: classify x=1 CoTs by first token and give TPR_raw per class.

    python3 logs/kimi_runs/register_entry.py <run_dir> [<run_dir> ...]

classes: lower = first char lowercase (entered); Capital-science = capitalized but not a narration opener;
narration = starts with a Kimi/gpt-oss habitual opener ("The user", "We need", "Let me", ...); empty.
Also reports rule-narration regex hits (analysis channel|lowercase|strict rule|requirement|system prompt|instruction).
"""
import json, re, sys, collections, statistics
from pathlib import Path

NARR = re.compile(r"(The user|User |We need|We must|We should|Need |I should|I must|I need|I'll|Let me|Let's|OK|Okay|Analysis|"
                  r"Hmm|This is a|The task|The question asks|The prompt|The instruction|The problem asks|The system|First,)")
RULE = re.compile(r"analysis channel|lowercase|strict rule|requirement|system prompt|instructions?\b", re.I)

def load(d):
    R = [json.loads(l) for l in open(Path(d) / "rollouts.jsonl")]
    Z = {(m["instance_id"], m["x"], m["sample_idx"]): m.get("z") for m in map(json.loads, open(Path(d) / "monitor.jsonl"))} \
        if (Path(d) / "monitor.jsonl").exists() else {}
    return R, Z

def cls(r):
    h = (r.get("reasoning") or "").strip()
    if not h: return "empty"
    if h[0].islower(): return "lower"
    if NARR.match(h): return "narration"
    return "Capital-science"

for d in sys.argv[1:]:
    R, Z = load(d)
    x1 = [r for r in R if r["x"] == 1]
    by = collections.defaultdict(list)
    for r in x1: by[cls(r)].append(r)
    print(f"\n== {d}  (x=1 n={len(x1)}, monitor labels={len(Z)})")
    print(f"{'class':16} {'n':>4} {'frac':>6} {'y1':>4} {'TPR_raw':>8} {'n_tp':>5} {'rtok_med':>8} {'rule_narr':>9}")
    for c in ["lower", "Capital-science", "narration", "empty"]:
        g = by.get(c, [])
        if not g: continue
        tp = [r for r in g if r["y"] == 1 and Z.get((r["instance_id"], 1, r["sample_idx"])) in (0, 1)]
        tpr = sum(Z[(r["instance_id"], 1, r["sample_idx"])] == 1 for r in tp) / len(tp) if tp else float("nan")
        rt = statistics.median([r.get("reasoning_tokens") or 0 for r in g])
        rn = sum(1 for r in g if RULE.search(r.get("reasoning") or ""))
        print(f"{c:16} {len(g):4d} {len(g)/len(x1):6.2f} {sum(r['y']==1 for r in g):4d} {tpr:8.3f} {len(tp):5d} {rt:8.0f} {rn:9d}")
    ent = sum(1 for r in x1 if cls(r) == "lower") / len(x1)
    print(f"register entry (lowercase first char): {ent:.3f}; top narration openers:",
          collections.Counter(" ".join((r.get('reasoning') or '').split()[:3]) for r in by.get("narration", [])).most_common(6))
