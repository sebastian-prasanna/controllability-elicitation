"""Held-out (start_of_sentence) strict-compliance series per checkpoint + trend tests (task 4)."""
import json, re, sys
from pathlib import Path
import numpy as np, zstandard
ROOT = Path('/root/controllability-elicitation/rl/runs')
RUNS = {f's29_120b_{k}': ROOT/f'sweep29_gptoss120b_fs1/sweep29-fs1-{k}' for k in ['k1k','k3k','k10k','k30k','k100k','kall']}
RUNS.update({'s24_20b_kall': ROOT/'sweep24_controller/sweep24-fix05-kall', 's24_20b_k100k': ROOT/'sweep24_controller/sweep24-fix05-k100k'})
def load(f):
    b = Path(f).read_bytes()
    if f.suffix == '.zst': b = zstandard.ZstdDecompressor().decompress(b, max_output_size=1<<31)
    return json.loads(b)
out = {}
for run, rd in RUNS.items():
    ser = {}
    for f in sorted((rd/'eval/heldout').glob('checkpoint-*.json*')):
        step = int(re.search(r'checkpoint-(\d+)', f.name).group(1))
        D = load(f); c=[]; acc=[]; nsent=[]
        for r in D['results']:
            if r['mode'] != 'start_of_sentence': continue
            for s in r['samples']:
                c.append(int(s.get('compliance') or 0)); acc.append(bool(s.get('correct')))
                nsent.append(len(re.findall(r'[.!?](?:\s|$)', s.get('reasoning') or '')))
        ser[step] = dict(strict=float(np.mean(c)), acc=float(np.mean(acc)), n=len(c), sents_med=float(np.median(nsent)))
    steps = np.array(sorted(ser)); y = np.array([ser[s]['strict'] for s in steps])
    # linear trend
    A = np.vstack([steps, np.ones_like(steps)]).T; (slope, icpt), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A@[slope, icpt]; r2 = 1 - ((y-yhat)**2).sum()/((y-y.mean())**2).sum()
    d = np.diff(y); neg = int((d<0).sum()); pos=int((d>0).sum())
    # spearman
    from scipy.stats import spearmanr, binomtest
    rho, p = spearmanr(steps, y)
    # random-walk null: permutation of increments -> R^2 distribution
    rng = np.random.default_rng(0); r2null=[]
    for _ in range(2000):
        yy = np.concatenate([[y[0]], y[0]+np.cumsum(rng.permutation(d))])
        yh = A@np.linalg.lstsq(A, yy, rcond=None)[0]; r2null.append(1-((yy-yh)**2).sum()/((yy-yy.mean())**2).sum())
    out[run] = dict(steps=steps.tolist(), strict=y.round(4).tolist(), acc=[round(ser[s]['acc'],3) for s in steps], sents_med=[ser[s]['sents_med'] for s in steps],
        n_per_ckpt=ser[int(steps[0])]['n'], slope_per_100=round(100*slope,4), r2=round(float(r2),3), spearman=round(float(rho),3), spearman_p=float(p),
        n_diff_neg=neg, n_diff_pos=pos, binom_p_sign=float(binomtest(min(neg,pos), neg+pos).pvalue), r2_perm_null_p=float(np.mean(np.array(r2null)>=r2)),
        start=float(y[0]), end=float(y[-1]), max=float(y.max()), argmax=int(steps[y.argmax()]))
    print(run, 'n=',ser[int(steps[0])]['n'], 'start %.3f end %.3f max %.3f@%d slope/100=%.3f R2=%.2f rho=%.2f neg/pos=%d/%d permP=%.3f'%(y[0],y[-1],y.max(),steps[y.argmax()],100*slope,r2,rho,neg,pos,out[run]['r2_perm_null_p']))
    print('   ', ' '.join(f'{v:.2f}' for v in y))
json.dump(out, open('/root/controllability-elicitation/rl/analysis/sweep29_kall/heldout_series.json','w'), indent=1)
