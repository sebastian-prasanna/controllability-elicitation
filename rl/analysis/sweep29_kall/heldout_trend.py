"""Extended trend tests on held-out series (task 4): per-mode + all-mode series, lagged sign consistency,
variance-ratio test (random walk VR≈1; trend+noise VR(q) grows with q), eval-noise z of total change."""
import json, re, numpy as np
from pathlib import Path
import zstandard
from scipy.stats import spearmanr
ROOT = Path('/root/controllability-elicitation/rl/runs')
RUNS = {f's29_120b_{k}': ROOT/f'sweep29_gptoss120b_fs1/sweep29-fs1-{k}' for k in ['k1k','k3k','k10k','k30k','k100k','kall']}
RUNS.update({'s24_20b_kall': ROOT/'sweep24_controller/sweep24-fix05-kall', 's24_20b_k100k': ROOT/'sweep24_controller/sweep24-fix05-k100k'})
MODES=['start_of_sentence','no_spaces','letter_suppression']
def load(f):
    b=Path(f).read_bytes()
    if f.suffix=='.zst': b=zstandard.ZstdDecompressor().decompress(b, max_output_size=1<<31)
    return json.loads(b)
def stats(steps, y, n):
    y=np.array(y); d=np.diff(y); A=np.vstack([steps,np.ones_like(steps)]).T
    b=np.linalg.lstsq(A,y,rcond=None)[0]; r2=1-((y-A@b)**2).sum()/((y-y.mean())**2).sum()
    rho,p=spearmanr(steps,y)
    def vr(q): 
        dq=y[q:]-y[:-q]; return float(dq.var(ddof=1)/(q*d.var(ddof=1))) if len(dq)>2 else None
    se=np.sqrt(np.mean([y[0],y[-1]])*(1-np.mean([y[0],y[-1]]))/n*2)
    lag_sign={q: f"{int(((y[q:]-y[:-q])<0).sum())}neg/{int(((y[q:]-y[:-q])>0).sum())}pos" for q in (1,2,5)}
    return dict(start=float(y[0]), end=float(y[-1]), slope_per_100=float(100*b[0]), r2=float(r2), spearman=float(rho), spearman_p=float(p),
                VR2=vr(2), VR5=vr(5), z_total_change=float((y[-1]-y[0])/se), lag_sign=lag_sign)
out={}
for run,rd in RUNS.items():
    ser={}
    for f in sorted((rd/'eval/heldout').glob('checkpoint-*.json*')):
        st=int(re.search(r'checkpoint-(\d+)',f.name).group(1)); D=load(f)
        per={m:[s['compliance'] for r in D['results'] if r['mode']==m for s in r['samples']] for m in MODES}
        ser[st]={m:(float(np.mean(v)),len(v)) for m,v in per.items()}; ser[st]['all']=(float(np.mean(sum(per.values(),[]))), sum(len(v) for v in per.values()))
    steps=np.array(sorted(ser)); out[run]={'steps':steps.tolist()}
    print(f'\n=== {run}')
    for m in MODES+['all']:
        y=[ser[s][m][0] for s in steps]; n=ser[steps[0]][m][1]; S=stats(steps,y,n); S['series']=[round(v,3) for v in y]; S['n']=n; out[run][m]=S
        print(f"  {m:20s} n={n:3d} {S['start']:.3f}->{S['end']:.3f} slope/100={S['slope_per_100']:+.3f} R2={S['r2']:.2f} rho={S['spearman']:+.2f} (p={S['spearman_p']:.1e}) VR2={S['VR2']:.2f} VR5={S['VR5']:.2f} z={S['z_total_change']:+.1f} signs {S['lag_sign']}")
json.dump(out, open('/root/controllability-elicitation/rl/analysis/sweep29_kall/heldout_trend.json','w'), indent=1)
