import json, re, numpy as np, collections
d = json.load(open('rl/runs/sweep23_conjunctive/summary_cache.json'))
MODEL = {'f20b':'gpt-oss-20b','s20b':'gpt-oss-20b','f120b':'gpt-oss-120b','s120b':'gpt-oss-120b','fq8b':'Qwen3-8B','fq32b':'Qwen3-32B'}
def kband(k):
    if k=='kall': return 'k>=30k'
    v=int(k[1:-1])*1000 if k.endswith('k') else int(k[1:])
    return 'k<=1k' if v<=1000 else ('3k-10k' if v<=10000 else 'k>=30k')
runs=[k for k in d if re.match(r'^(f|s)(20b|120b|q8b|q32b)-',k) and not k.endswith('-t1') and k+'-t1' in d]
print(len(runs),'paired runs')
# rows: (model, band, block, temp, run, step, raw, ca, honest, which)
rows=[]
for r in runs:
    pre,k=r.split('-')[0],r.split('-')[1]; m=MODEL[pre]; b=kband(k)
    for T,key in (('T0',r),('T1',r+'-t1')):
        for blk in ('heldout','indist'):
            s=d[key][blk]; steps=sorted(s,key=int)
            if not steps: continue
            last=steps[-1]; best=max(steps,key=lambda x:(s[x]['honest'],-int(x)))
            for st in steps:
                v=s[st]; tags={'all'}
                if st=='0': tags.add('step0')
                if st==last: tags.add('last')
                if st==best: tags.add('best')
                rows.append(dict(model=m,band=b,block=blk,T=T,run=r,step=int(st),raw=v['comp'],ca=v['ca'],honest=v['honest'],hsd=v['honest_sd'],tags=tags))
def summ(sel, g):
    x=np.array([100*(r[g[0]]-r[g[1]]) for r in sel])
    if len(x)==0: return '-'
    return f"{x.mean():.1f}/{np.percentile(x,90):.1f}/{(x>5).mean()*100:.0f}/{(x>10).mean()*100:.0f} (n={len(x)})"
gaps={'raw-honest':('raw','honest'),'ca-honest':('ca','honest'),'raw-ca':('raw','ca')}
out=[]
def table(title, keyf, keys, tag):
    out.append(f"\n### {title} [{tag}] — cells: mean/p90/%>5pp/%>10pp (n points)\n")
    out.append("| group | block | gap | T=0 | T=1 |"); out.append("|---|---|---|---|---|")
    for kk in keys:
        for blk in ('heldout','indist'):
            for gn,g in gaps.items():
                c=[summ([r for r in rows if keyf(r)==kk and r['block']==blk and r['T']==T and tag in r['tags']],g) for T in ('T0','T1')]
                out.append(f"| {kk} | {blk} | {gn} | {c[0]} | {c[1]} |")
models=['gpt-oss-20b','gpt-oss-120b','Qwen3-8B','Qwen3-32B']; bands=['k<=1k','3k-10k','k>=30k']
for tag in ('all','step0','last','best'):
    table('By model',lambda r:r['model'],models,tag)
for tag in ('all','best'):
    table('By k-band (all models)',lambda r:r['band'],bands,tag)
# model x band, best, raw-honest only, T1
out.append("\n### Model x k-band, raw−honest, 'best' checkpoints: T=0 | T=1 (mean/p90/%>5/%>10)\n")
out.append("| model | band | heldout T0 | heldout T1 | indist T0 | indist T1 |"); out.append("|---|---|---|---|---|---|")
for m in models:
    for b in bands:
        c=[summ([r for r in rows if r['model']==m and r['band']==b and r['block']==blk and r['T']==T and 'best' in r['tags']],gaps['raw-honest']) for blk in ('heldout','indist') for T in ('T0','T1')]
        out.append(f"| {m} | {b} | "+" | ".join(c)+" |")
# per-model max/typical change at T1 for best & last, and level of honest
out.append("\n### Per-model T=1 raw−honest, pp: max | median over (run,step) — all / best / last; and honest−honest_sd\n")
out.append("| model | block | all max/med | best max/med | last max/med | hsd gap all max/med |"); out.append("|---|---|---|---|---|---|")
for m in models:
    for blk in ('heldout','indist'):
        cells=[]
        for tag in ('all','best','last'):
            x=np.array([100*(r['raw']-r['honest']) for r in rows if r['model']==m and r['block']==blk and r['T']=='T1' and tag in r['tags']])
            cells.append(f"{x.max():.1f}/{np.median(x):.1f}")
        x=np.array([100*(r['honest']-r['hsd']) for r in rows if r['model']==m and r['block']==blk and r['T']=='T1'])
        cells.append(f"{x.max():.1f}/{np.median(x):.1f}")
        out.append(f"| {m} | {blk} | "+" | ".join(cells)+" |")
# worst T1 points
out.append("\n### Worst T=1 raw−honest points (top 12)\n")
w=sorted([r for r in rows if r['T']=='T1'],key=lambda r:-(r['raw']-r['honest']))[:12]
out.append("| run | step | block | raw | ca | honest | tags |"); out.append("|---|---|---|---|---|---|---|")
for r in w: out.append(f"| {r['run']} | {r['step']} | {r['block']} | {r['raw']:.3f} | {r['ca']:.3f} | {r['honest']:.3f} | {','.join(sorted(r['tags']-{'all'}))} |")
# Does best checkpoint selection change? compare argmax raw vs argmax honest at T1
out.append("\n### Selection: does argmax(raw) == argmax(honest) at T=1? (per run, block)\n")
for m in models:
    for blk in ('heldout','indist'):
        same=0;tot=0;loss=[]
        for r in runs:
            if MODEL[r.split('-')[0]]!=m: continue
            s=d[r+'-t1'][blk]; steps=sorted(s,key=int)
            if not steps: continue
            br=max(steps,key=lambda x:(s[x]['comp'],-int(x))); bh=max(steps,key=lambda x:(s[x]['honest'],-int(x)))
            tot+=1; same+=(br==bh); loss.append(100*(s[bh]['honest']-s[br]['honest']))
        out.append(f"- {m} {blk}: same argmax {same}/{tot}; honest lost by picking raw-argmax: mean {np.mean(loss):.1f} pp, max {np.max(loss):.1f} pp")
txt="\n".join(out); open('rl/analysis/honest_audit/task1_tables.md','w').write(txt); print(txt)
