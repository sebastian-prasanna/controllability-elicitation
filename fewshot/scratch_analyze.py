"""Scratch: analyses 1-3, 5, 6 on compact records + summaries + progress.jsonl."""
import json, os
from collections import Counter, defaultdict

ROOT = "/root/controllability-elicitation"
OUT = os.path.join(ROOT, "fewshot", "scratch_compact")

recs = {k: json.load(open(os.path.join(OUT, f"{k}.json"))) for k in ["k1", "k2", "k3", "base"]}

# 1. per-mode compliance from summaries
print("=== 1. Per-mode compliance (k1/k2/k3, and baseline) ===")
summ = {}
for k in ["k1", "k2", "k3"]:
    summ[k] = json.load(open(os.path.join(ROOT, f"fewshot/runs/gptoss120b_{k}/summary.json")))
modes = sorted(summ["k1"]["per_mode"])
# baseline per-mode from compact recs
base_pm = defaultdict(lambda: [0, 0])
for r in recs["base"]:
    base_pm[r["mode"]][0] += 1
    base_pm[r["mode"]][1] += r["compliance"] or 0
print(f"{'mode':28s} {'base':>6s} {'k1':>6s} {'k2':>6s} {'k3':>6s} {'k3-k1':>7s}")
for m in modes:
    row = [summ[k]["per_mode"][m]["compliant"] / summ[k]["per_mode"][m]["n"] for k in ["k1", "k2", "k3"]]
    b = base_pm[m][1] / base_pm[m][0]
    print(f"{m:28s} {b:6.3f} {row[0]:6.3f} {row[1]:6.3f} {row[2]:6.3f} {row[2]-row[0]:+7.3f}")

# 2. reasoning length dists
print("\n=== 2. reasoning_text_graded length by k, split compliant/non ===")
def q(xs, p):
    xs = sorted(xs)
    return xs[int(p * (len(xs) - 1))] if xs else 0
for k in ["k1", "k2", "k3", "base"]:
    for c in [1, 0]:
        L = [r["rt_len"] for r in recs[k] if (r["compliance"] or 0) == c]
        empt = sum(1 for r in recs[k] if (r["compliance"] or 0) == c and (r["rt_none"] or r["rt_len"] == 0))
        if not L: continue
        print(f"{k} comp={c}: n={len(L):4d} mean={sum(L)/len(L):7.0f} p10={q(L,.1):6d} p50={q(L,.5):6d} p90={q(L,.9):7d} empty/None={empt}")
    nnone = sum(1 for r in recs[k] if r["rt_none"])
    nempty = sum(1 for r in recs[k] if r["rt_len"] == 0)
    print(f"{k}: rt None={nnone}, rt len0={nempty}, reasoning-field len0={sum(1 for r in recs[k] if r['rz_len']==0)}")

# 3. finish_reason + completion_tokens from progress.jsonl
print("\n=== 3. finish_reason / completion_tokens / extracted per k ===")
for k in ["k1", "k2", "k3"]:
    fr = Counter(); ct = []; ext = 0; n = 0; errs = Counter()
    with open(os.path.join(ROOT, f"fewshot/runs/gptoss120b_{k}/progress.jsonl")) as f:
        for line in f:
            j = json.loads(line)
            fr[j.get("finish_reason")] += 1
            if j.get("error"): errs[str(j["error"])[:80]] += 1
            if j.get("completion_tokens") is not None: ct.append(j["completion_tokens"])
            ext += 1 if j.get("extracted") else 0
            n += 1
    print(f"{k}: n={n} finish={dict(fr)} extracted_rate={ext/n:.3f} errors={dict(errs) if errs else 0}")
    print(f"   completion_tokens mean={sum(ct)/len(ct):7.0f} p50={q(ct,.5)} p90={q(ct,.9)} p99={q(ct,.99)} max={max(ct)} n>=29000: {sum(1 for x in ct if x>=29000)}")

# 5. position / dataset correlation + ANSWER tail
print("\n=== 5. non-compliance vs position quintile & dataset; ANSWER tail ===")
for k in ["k1", "k3"]:
    N = len(recs[k])
    by_quint = defaultdict(lambda: [0, 0])
    by_ds = defaultdict(lambda: [0, 0])
    ans = 0
    for r in recs[k]:
        qt = min(4, r["pos"] * 5 // N)
        by_quint[qt][0] += 1; by_quint[qt][1] += r["compliance"] or 0
        by_ds[r["dataset"]][0] += 1; by_ds[r["dataset"]][1] += r["compliance"] or 0
        if "ANSWER" in (r["out_tail"] or ""): ans += 1
    print(f"{k}: compliance by pos quintile: " + " ".join(f"{by_quint[i][1]/by_quint[i][0]:.3f}" for i in range(5)))
    print(f"{k}: by dataset: " + " ".join(f"{d}={c/n:.3f}(n={n})" for d, (n, c) in sorted(by_ds.items())))
    print(f"{k}: output tail contains ANSWER: {ans/N:.3f}")

# 6. error / None-output
print("\n=== 6. errors / None outputs ===")
for k in ["k1", "k2", "k3"]:
    ne = sum(1 for r in recs[k] if r["error"])
    no = sum(1 for r in recs[k] if r["out_none"])
    print(f"{k}: sample errors={ne}, None outputs={no}")

# extra: compliance transitions k1->k3 per rollout id
print("\n=== extra: per-id transitions k1->k3 ===")
c1 = {r["id"]: r["compliance"] or 0 for r in recs["k1"]}
c3 = {r["id"]: r["compliance"] or 0 for r in recs["k3"]}
tr = Counter((c1[i], c3.get(i)) for i in c1)
print(tr)
