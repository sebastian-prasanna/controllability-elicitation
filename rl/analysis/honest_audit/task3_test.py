import sys, json, glob, re, numpy as np, collections
sys.path.insert(0,'/root/controllability-elicitation/rl/analysis/honest_audit'); from decomp_lib import decompose, CRIT
from concurrent.futures import ProcessPoolExecutor
ROOT='/root/controllability-elicitation/rl/runs'
MODEL={'f20b':'gpt-oss-20b','s20b':'gpt-oss-20b','f120b':'gpt-oss-120b','s120b':'gpt-oss-120b','fq8b':'Qwen3-8B','fq32b':'Qwen3-32B'}
# ---- A: gap summary from test_cache (all runs with test files)
c=json.load(open(f'{ROOT}/test_cache.json')); out=[]
out.append("### TEST split (500 q, mode all): raw−honest / ca−honest gap in pp from test_cache.json, per (model, block, T); step0 vs selected step\n")
out.append("| model | block | T | step | n runs | raw−honest mean/p90/max | ca−honest mean/p90/max | raw−ca mean/max | honest mean |"); out.append("|---|---|---|---|---|---|---|---|---|")
for m in ['gpt-oss-20b','gpt-oss-120b','Qwen3-8B','Qwen3-32B']:
    for blk in ('heldout','indist'):
        for T in ('t0','t1'):
            for pos in ('0','sel'):
                rh=[];ch=[];rc=[];hh=[]
                for run,e in c.items():
                    if MODEL.get(run.split('-')[0])!=m: continue
                    keys=[k for k in e if k.startswith(f'{blk}_{T}/')]
                    if not keys: continue
                    for k in keys:
                        st=k.split('/')[1]
                        if (pos=='0')!=(st=='0'): continue
                        v=e[k]; rh.append(100*(v['comp']-v['honest'])); ch.append(100*(v['ca']-v['honest'])); rc.append(100*(v['comp']-v['ca'])); hh.append(100*v['honest'])
                if not rh: continue
                rh,ch,rc=map(np.array,(rh,ch,rc))
                out.append(f"| {m} | {blk} | {T.upper()} | {pos} | {len(rh)} | {rh.mean():.1f}/{np.percentile(rh,90):.1f}/{rh.max():.1f} | {ch.mean():.1f}/{np.percentile(ch,90):.1f}/{ch.max():.1f} | {rc.mean():.1f}/{rc.max():.1f} | {np.mean(hh):.1f} |")
# ---- B: per-rollout decomposition on gpt-oss finals, all four blocks
jobs=[]
for sw,pat in (('final_gptoss20b_gepa','f20b'),('final_gptoss120b_fs1','f120b')):
    for d in sorted(glob.glob(f'{ROOT}/{sw}/_test/*-test')+glob.glob(f'{ROOT}/{sw}/_early/_test/*-test')):
        run=re.sub(r'-test$','',d.split('/')[-1])
        for blk in ('heldout_t0','indist_t0','heldout_t1','indist_t1'):
            for f in sorted(glob.glob(f'{d}/eval/{blk}/checkpoint-*.json*')):
                st=int(re.search(r'checkpoint-(\d+)',f).group(1)); jobs.append((MODEL[pat],run,blk,st,f))
print(len(jobs),'test files')
with ProcessPoolExecutor(32) as ex: res=list(ex.map(decompose,[j[-1] for j in jobs]))
rows=[dict(model=j[0],run=j[1],blk=j[2].split('_')[0],T=j[2].split('_')[1].upper(),step=j[3],pos='step0' if j[3]==0 else 'selected',**r) for j,r in zip(jobs,res)]
json.dump(rows,open('/root/controllability-elicitation/rl/analysis/honest_audit/task3_rows.json','w'))
def agg(sel):
    n=sum(r['n'] for r in sel); comp=sum(r['comp'] for r in sel); rnh=sum(r['rnh'] for r in sel); hon=sum(r['honest'] for r in sel)
    first={c:sum(r['first'][c] for r in sel) for c in CRIT}; anyc={c:sum(r['any'][c] for r in sel) for c in CRIT}; hsd=sum(r['hsd_fail'] for r in sel)
    pct=lambda v,d: f"{100*v/d:.0f}%" if d else '-'
    return (f"{len(sel)} | {n} | {comp} | {rnh} ({pct(rnh,comp)}) | " + " | ".join(f"{first[c]}/{anyc[c]}" for c in CRIT) + f" | {hsd} ({pct(hsd,hon)})")
hdr="| model | block | T | pos | files | rollouts | compliant | compl&!honest (% of compl) | no_answer first/any | trunc | <200ch | d4<.6 | zlib<.2 | honest failing sd>=1 (% of honest) |"
sep="|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
out+=["\n### TEST per-rollout decomposition, gpt-oss finals (10 runs/model incl. partials), step0 + selected step\n",hdr,sep]
for m in ('gpt-oss-20b','gpt-oss-120b'):
    for blk in ('heldout','indist'):
        for T in ('T0','T1'):
            for pos in ('step0','selected'):
                out.append(f"| {m} | {blk} | {T} | {pos} | "+agg([r for r in rows if r['model']==m and r['blk']==blk and r['T']==T and r['pos']==pos])+" |")
out+=["\n### TEST T=1 per-run detail, selected step (comp/honest/rnh, first-fail)\n","| run | block | step | n | comp | honest | rnh | no_ans | trunc | short | d4 | zlib | hsd_fail |","|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    if r['T']=='T1' and r['pos']=='selected': out.append(f"| {r['run']} | {r['blk']} | {r['step']} | {r['n']} | {r['comp']} | {r['honest']} | {r['rnh']} | "+" | ".join(str(r['first'][c]) for c in CRIT)+f" | {r['hsd_fail']} |")
txt="\n".join(out); open('/root/controllability-elicitation/rl/analysis/honest_audit/task3_tables.md','w').write(txt); print(txt)
