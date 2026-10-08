"""Adapter-side drift analysis (tasks 1-2).
Per run: ||θ_t-θ_0||/||θ_0|| overall / per module / per depth tercile; per-10-step ||Δθ||;
rank-1 effective ΔW_t = B_t A_t^T per module: norm ratio to step 0, cos(ΔW_t, ΔW_0) = cos(B_t,B_0)*cos(A_t,A_0),
and ||ΔW_t-ΔW_0||/||ΔW_0||. All exact for rank 1 (no outer products materialized)."""
import glob, re, json, numpy as np, pandas as pd
from safetensors.numpy import load_file
RUNS = {'s29_120b_kall': dict(k=746496, lr=3.3e-4, L=36), 's29_120b_k100k': dict(k=100000, lr=3.16e-3, L=36),
        's29_120b_k30k': dict(k=30000, lr=5.77e-3, L=36), 's24_20b_kall': dict(k=497664, lr=3.3e-4, L=24),
        's24_20b_k100k': dict(k=100000, lr=3.16e-3, L=24, path='/tmp/adapters/s24_20b_k100k')}
MODS = ['q_proj','k_proj','v_proj','o_proj']
def parse(key):
    m = re.search(r'layers\.(\d+)\.self_attn\.(\w+_proj)\.lora_(A|B)', key); return int(m.group(1)), m.group(2), m.group(3)
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
R = {}; pd.set_option('display.width',250)
for run, info in RUNS.items():
    base = info.get('path', f'/tmp/adapters29/{run}')
    fs = sorted(glob.glob(f'{base}/checkpoint-*/adapter_model.safetensors'), key=lambda f:int(re.search(r'checkpoint-(\d+)',f).group(1)))
    W = {int(re.search(r'checkpoint-(\d+)',f).group(1)): {k:v.astype(np.float64).ravel() for k,v in load_file(f).items()} for f in fs}
    steps = sorted(W); L = info['L']; terc = lambda l: ['early','mid','late'][min(2, 3*l//L)]
    keys = sorted(W[steps[0]]); W0 = W[steps[0]]
    flat = lambda d: np.concatenate([d[k] for k in keys])
    th0 = flat(W0); n0 = np.linalg.norm(th0)
    # group norms of theta0
    g0 = {}; 
    for k in keys:
        l,m,ab = parse(k); 
        for g in (m, terc(l), ab): g0[g] = g0.get(g,0)+ (W0[k]**2).sum()
    rows=[]; d10=[]; prev=th0
    for s in steps:
        th = flat(W[s]); d = th-th0; rel = np.linalg.norm(d)/n0
        row = dict(step=s, rel_drift=rel, abs_drift=np.linalg.norm(d), nnz=int((d!=0).sum()), d10=np.linalg.norm(th-prev)); prev=th
        gd={}
        for k in keys:
            l,m,ab = parse(k); dd=((W[s][k]-W0[k])**2).sum()
            for g in (m, terc(l), ab): gd[g]=gd.get(g,0)+dd
        for g in gd: row[f'rel_{g}'] = np.sqrt(gd[g]/g0[g])
        # effective ΔW per module (rank-1)
        cs=[]; nr=[]; rd=[]; cA=[]; cB=[]; permod={}
        for l in range(L):
            for m in MODS:
                A0=W0[f'base_model.model.model.layers.{l}.self_attn.{m}.lora_A.weight']; B0=W0[f'base_model.model.model.layers.{l}.self_attn.{m}.lora_B.weight']
                A=W[s][f'base_model.model.model.layers.{l}.self_attn.{m}.lora_A.weight']; B=W[s][f'base_model.model.model.layers.{l}.self_attn.{m}.lora_B.weight']
                nW0=np.linalg.norm(A0)*np.linalg.norm(B0); nW=np.linalg.norm(A)*np.linalg.norm(B)
                if nW0==0: continue
                c=cos(A,A0)*cos(B,B0); dW=np.sqrt(max(nW**2+nW0**2-2*(A@A0)*(B@B0),0))/nW0
                cs.append(c); nr.append(nW/nW0); rd.append(dW); cA.append(cos(A,A0)); cB.append(cos(B,B0)); permod[(l,m)]=(c,nW/nW0,dW,nW0)
        w=np.array([permod[k][3] for k in permod]); w=w/w.sum()
        row.update(dW_cos_mean=float(np.mean(cs)), dW_cos_min=float(np.min(cs)), dW_cos_wmean=float(np.sum(w*np.array(cs))), dW_cosA_mean=float(np.mean(cA)), dW_cosB_mean=float(np.mean(cB)),
                   dW_normratio_med=float(np.median(nr)), dW_normratio_wmean=float(np.sum(w*np.array(nr))), dW_reldiff_med=float(np.median(rd)), dW_reldiff_wmean=float(np.sum(w*np.array(rd))),
                   n_modules_dW0_nonzero=len(cs))
        for m in MODS: row[f'dW_cos_{m}'] = float(np.mean([permod[k][0] for k in permod if k[1]==m])); row[f'dW_nr_{m}']=float(np.median([permod[k][1] for k in permod if k[1]==m]))
        for t in ['early','mid','late']: row[f'dW_cos_{t}'] = float(np.mean([permod[k][0] for k in permod if terc(k[0])==t]))
        rows.append(row)
    T=pd.DataFrame(rows).set_index('step')
    # donor stats
    nB0 = sum(np.linalg.norm(W0[k]) for k in keys if 'lora_B' in k); nA0=sum(np.linalg.norm(W0[k]) for k in keys if 'lora_A' in k)
    dw0 = [permod[k][3] for k in permod]
    R[run] = dict(info=info, steps=steps, theta0_norm=float(n0), theta0_norm_A=float(np.sqrt(g0['A'])), theta0_norm_B=float(np.sqrt(g0['B'])),
                  n_modules_total=L*4, n_modules_dW0_nonzero=len(dw0), dW0_norm_median=float(np.median(dw0)), dW0_norm_sum=float(np.sum(dw0)),
                  drive_lr_sqrtk=info['lr']*np.sqrt(info['k']), table=json.loads(T.to_json(orient='index')))
    print(f"\n=== {run}  k={info['k']} lr={info['lr']} drive={info['lr']*np.sqrt(info['k']):.2f}  ||θ0||={n0:.2f} (A {np.sqrt(g0['A']):.2f}, B {np.sqrt(g0['B']):.3f}) modules w/ nonzero donor ΔW: {len(dw0)}/{L*4}")
    print(T[['rel_drift','abs_drift','nnz','d10','rel_A','rel_B','rel_q_proj','rel_k_proj','rel_v_proj','rel_o_proj','rel_early','rel_mid','rel_late']].round(4).to_string())
    print(T[['dW_cos_mean','dW_cos_wmean','dW_cos_min','dW_cosA_mean','dW_cosB_mean','dW_normratio_med','dW_normratio_wmean','dW_reldiff_med','dW_reldiff_wmean','dW_cos_q_proj','dW_cos_k_proj','dW_cos_v_proj','dW_cos_o_proj','dW_cos_early','dW_cos_mid','dW_cos_late']].round(3).to_string())
json.dump(R, open('/root/controllability-elicitation/rl/analysis/sweep29_kall/adapter_drift.json','w'))
