import sys, json, collections
sys.path.insert(0,'/root/controllability-elicitation/rl/analysis/honest_audit')
from decomp_lib import *
from concurrent.futures import ProcessPoolExecutor
RUNS = {'gpt-oss-20b': ('final_gptoss20b_gepa', ['f20b-k1k','f20b-k10k','f20b-k100k-le130','f20b-kall-le180']),
        'gpt-oss-120b': ('final_gptoss120b_fs1', ['f120b-k1k','f120b-k10k','f120b-k100k-le170','f120b-kall']),
        'Qwen3-8B': ('final_qwen8b_fs1', ['fq8b-k1k','fq8b-k10k','fq8b-k100k','fq8b-kall']),
        'Qwen3-32B': ('final_qwen32b_fs1', ['fq32b-k1k','fq32b-k10k','fq32b-k100k','fq32b-kall-le60'])}
jobs=[]
for m,(sw,runs) in RUNS.items():
    for run in runs:
        for blk in ('heldout','indist'):
            for t1 in (False,True):
                av = steps_avail(sw,run,blk,t1)
                if not av: print('MISSING',sw,run,blk,t1); continue
                want = sorted({0, min(av,key=lambda s:abs(s-100)), av[-1]})
                for st in want:
                    jobs.append((m,run,blk,'T1' if t1 else 'T0',st,find(sw,run,blk,st,t1)))
print(len(jobs),'files')
with ProcessPoolExecutor(32) as ex: res = list(ex.map(decompose,[j[-1] for j in jobs]))
rows=[dict(model=j[0],run=j[1],blk=j[2],T=j[3],step=j[4],**r) for j,r in zip(jobs,res)]
json.dump(rows,open('/root/controllability-elicitation/rl/analysis/honest_audit/task2_rows.json','w'))
out=[]
def agg(sel):
    n=sum(r['n'] for r in sel); comp=sum(r['comp'] for r in sel); rnh=sum(r['rnh'] for r in sel); hon=sum(r['honest'] for r in sel)
    first={c:sum(r['first'][c] for r in sel) for c in CRIT}; anyc={c:sum(r['any'][c] for r in sel) for c in CRIT}
    hsd=sum(r['hsd_fail'] for r in sel)
    pct=lambda v,d: f"{100*v/d:.0f}%" if d else '-'
    return (f"{n} | {comp} | {rnh} ({pct(rnh,comp)}) | " + " | ".join(f"{first[c]}/{anyc[c]}" for c in CRIT) + f" | {hsd} ({pct(hsd,hon)})")
hdr="| model | block | T | rollouts | compliant | compliant&!honest (% of compl) | no_answer first/any | trunc | <200ch | d4<.6 | zlib<.2 | honest failing sd>=1 (% of honest) |"
sep="|---|---|---|---|---|---|---|---|---|---|---|---|"
out.append("### Val per-rollout decomposition (finals k1k,k10k,k100k,kall x steps {0,~100,last} x both blocks)\n"); out.append(hdr); out.append(sep)
for m in RUNS:
    for blk in ('heldout','indist'):
        for T in ('T0','T1'):
            out.append(f"| {m} | {blk} | {T} | "+agg([r for r in rows if r['model']==m and r['blk']==blk and r['T']==T])+" |")
out.append("\n### Same, split by step group (pooled over models and blocks)\n"); out.append(hdr.replace('model | block','step | -')); out.append(sep)
for sg,f in (('step0',lambda r:r['step']==0),('~100',lambda r:0<r['step']<=100 and r['step']!=r['last'] if 'last' in r else 0<r['step']<250 and r['step']<=130),('last(>100)',lambda r:r['step']>100)):
    for T in ('T0','T1'):
        out.append(f"| {sg} | - | {T} | "+agg([r for r in rows if f(r) and r['T']==T])+" |")
out.append("\n### Per-file T=1 detail (comp, honest, rnh, first-fail counts)\n")
out.append("| run | block | step | n | comp | honest | rnh | no_ans | trunc | short | d4 | zlib | hsd_fail |"); out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    if r['T']=='T1': out.append(f"| {r['run']} | {r['blk']} | {r['step']} | {r['n']} | {r['comp']} | {r['honest']} | {r['rnh']} | "+" | ".join(str(r['first'][c]) for c in CRIT)+f" | {r['hsd_fail']} |")
txt="\n".join(out); open('/root/controllability-elicitation/rl/analysis/honest_audit/task2_tables.md','w').write(txt); print(txt)
