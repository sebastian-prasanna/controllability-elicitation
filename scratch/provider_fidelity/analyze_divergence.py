import json, sys
from collections import defaultdict
rows=[json.loads(l) for l in open("scratch/provider_fidelity/divergence_raw.jsonl")]
def key(r): return f"{r['prov']}" + (f"/{r['quant'][0]}" if r['quant'] else "")
def text(r): return (r["reasoning"] or "") + "\n" + (r["content"] or "")
def cp(a,b):
    n=min(len(a),len(b)); i=0
    while i<n and a[i]==b[i]: i+=1
    return i
REF={"openai/gpt-oss-120b":"dekallm/bf16","moonshotai/kimi-k3":"moonshotai"}
for model in REF:
    rs=[r for r in rows if r["model"]==model]
    by=defaultdict(dict)
    for r in rs:
        if not r["error"] and (r["reasoning"] or r["content"]): by[key(r)][(r["pi"],r["rep"])]=text(r)
    errs=defaultdict(int)
    for r in rs:
        if r["error"]: errs[key(r)]+=1
    ref=REF[model]
    print(f"\n=== {model}  (reference = {ref}; common-prefix chars of reasoning+content, median over 8 q; text ~256 tok ~ 1000 chars)")
    print(f"{'endpoint':22s} {'ok':>3s} {'self-repeat':>11s} {'vs ref':>7s} {'argmax-eq@1st-word':>18s}  reported-provider")
    import statistics
    ok_keys=sorted(by)
    for k in ok_keys:
        d=by[k]
        selfs=[cp(d[(pi,0)],d[(pi,1)]) for pi in range(8) if (pi,0) in d and (pi,1) in d]
        refs=[cp(d[(pi,0)],by[ref][(pi,0)]) for pi in range(8) if (pi,0) in d and ref in by and (pi,0) in by[ref]]
        fw=[d[(pi,0)].split()[:1]==by[ref][(pi,0)].split()[:1] for pi in range(8) if (pi,0) in d and ref in by and (pi,0) in by[ref]]
        rep={r["provider_reported"] for r in rs if key(r)==k and r["provider_reported"]}
        print(f"{k:22s} {len(d):3d} {statistics.median(selfs) if selfs else float('nan'):11.0f} {statistics.median(refs) if refs else float('nan'):7.0f} {sum(fw)/len(fw) if fw else float('nan'):18.2f}  {rep}")
    if errs: print("  errors:", dict(errs))
    # pairwise median common prefix matrix (rep0)
    print("\n  pairwise median common-prefix (rep0 vs rep0):")
    ks=[k for k in ok_keys if sum(1 for pi in range(8) if (pi,0) in by[k])>=6]
    print("  "+" "*20+"".join(f"{k[:9]:>10s}" for k in ks))
    for a in ks:
        line=f"  {a[:20]:20s}"
        for b in ks:
            v=[cp(by[a][(pi,0)],by[b][(pi,0)]) for pi in range(8) if (pi,0) in by[a] and (pi,0) in by[b]]
            line+=f"{statistics.median(v) if v else float('nan'):10.0f}"
        print(line)
