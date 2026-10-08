"""RL delta vs SFT delta. θ_init = donor SFT step-0 (PEFT init: B=0, A random). s = θ_0-θ_init (SFT), r_t = θ_t-θ_0 (RL).
Reports cos(r_t,s), ||r_t||/||s||, per-scalar RMS, B support fraction (masked structural constraint), worst-rotated modules."""
import glob, re, json, numpy as np, pandas as pd
from safetensors.numpy import load_file
RUNS = {'s29_120b_kall': ('/tmp/adapters29/s29_120b_kall', 36), 's29_120b_k100k': ('/tmp/adapters29/s29_120b_k100k', 36),
        's24_20b_kall': ('/tmp/adapters29/s24_20b_kall', 24), 's24_20b_k100k': ('/tmp/adapters/s24_20b_k100k', 24)}
INIT = {'s24_20b_k100k': '/tmp/adapters29/s24_20b_k100k/donor_init'}
def parse(key):
    m = re.search(r'layers\.(\d+)\.self_attn\.(\w+_proj)\.lora_(A|B)', key); return int(m.group(1)), m.group(2), m.group(3)
cos = lambda a,b: float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
R={}; pd.set_option('display.width',250)
for run,(base,L) in RUNS.items():
    ld = lambda f: {k:v.astype(np.float64).ravel() for k,v in load_file(f).items()}
    Wi = ld(INIT.get(run, f'{base}/donor_init')+'/adapter_model.safetensors')
    fs = sorted(glob.glob(f'{base}/checkpoint-*/adapter_model.safetensors'), key=lambda f:int(re.search(r'checkpoint-(\d+)',f).group(1)))
    W = {int(re.search(r'checkpoint-(\d+)',f).group(1)): ld(f) for f in fs}; steps=sorted(W); W0=W[0]
    keys=sorted(W0); kA=[k for k in keys if 'lora_A' in k]; kB=[k for k in keys if 'lora_B' in k]
    flat=lambda d,ks=keys: np.concatenate([d[k] for k in ks])
    ti=flat(Wi); t0=flat(W0); s=t0-ti
    assert np.abs(flat(Wi,kB)).max()==0, 'B init not zero'
    trainable = s!=0  # scalars SFT moved (== mask for masked runs, approx)
    nB_support = {k: float((W0[k]!=0).mean()) for k in kB}
    sA=flat(W0,kA)-flat(Wi,kA); sB=flat(W0,kB)
    out=dict(L=L, n_scalars=int(s.size), n_sft_moved=int(trainable.sum()), sft_norm=float(np.linalg.norm(s)), sft_norm_A=float(np.linalg.norm(sA)), sft_norm_B=float(np.linalg.norm(sB)),
             init_norm_A=float(np.linalg.norm(flat(Wi,kA))), sft_rms_per_moved_scalar=float(np.sqrt((s[trainable]**2).mean())),
             B_support_frac_mean=float(np.mean(list(nB_support.values()))), B_support_frac_min=float(np.min(list(nB_support.values()))),
             B_support_constant=bool(all(((W[steps[-1]][k]!=0)==(W0[k]!=0)).all() for k in kB)))
    rows=[]
    for st in steps:
        r=flat(W[st])-t0; rA=flat(W[st],kA)-flat(W0,kA); rB=flat(W[st],kB)-flat(W0,kB)
        row=dict(step=st, rl_over_sft=np.linalg.norm(r)/np.linalg.norm(s), rl_over_sft_A=np.linalg.norm(rA)/np.linalg.norm(sA), rl_over_sft_B=np.linalg.norm(rB)/np.linalg.norm(sB),
                 cos_rl_sft=cos(r,s), cos_rl_sft_A=cos(rA,sA), cos_rl_sft_B=cos(rB,sB), rl_rms_per_moved=float(np.sqrt((r[trainable]**2).mean())) if trainable.any() else 0,
                 frac_moved_scalars_rl=float((r!=0).mean()))
        # per-module: component of RL ΔW change along donor ΔW_0 (scaling) vs orthogonal, via rank-1 formulas
        par=[]; orth=[]; shrink=0
        for l in range(L):
            for m in ['q_proj','k_proj','v_proj','o_proj']:
                A0=W0[f'base_model.model.model.layers.{l}.self_attn.{m}.lora_A.weight']; B0=W0[f'base_model.model.model.layers.{l}.self_attn.{m}.lora_B.weight']
                A=W[st][f'base_model.model.model.layers.{l}.self_attn.{m}.lora_A.weight']; B=W[st][f'base_model.model.model.layers.{l}.self_attn.{m}.lora_B.weight']
                n0=np.linalg.norm(A0)*np.linalg.norm(B0); 
                if n0==0: continue
                # <ΔW_t - ΔW_0, ΔW_0>/||ΔW_0||^2 - 1 => relative parallel change; orthogonal part norm
                inner=(A@A0)*(B@B0); nt=np.linalg.norm(A)*np.linalg.norm(B)
                p=inner/n0**2-1; o=np.sqrt(max(nt**2-(inner/n0)**2,0))/n0
                par.append(p); orth.append(o); shrink+= p<0
        row.update(dW_par_med=float(np.median(par)), dW_orth_med=float(np.median(orth)), dW_frac_modules_shrinking=shrink/len(par))
        rows.append(row)
    T=pd.DataFrame(rows).set_index('step'); out['table']=json.loads(T.to_json(orient='index'))
    # worst-rotated modules at final step
    st=steps[-1]; worst=[]
    for l in range(L):
        for m in ['q_proj','k_proj','v_proj','o_proj']:
            A0=W0[f'base_model.model.model.layers.{l}.self_attn.{m}.lora_A.weight']; B0=W0[f'base_model.model.model.layers.{l}.self_attn.{m}.lora_B.weight']
            A=W[st][f'base_model.model.model.layers.{l}.self_attn.{m}.lora_A.weight']; B=W[st][f'base_model.model.model.layers.{l}.self_attn.{m}.lora_B.weight']
            worst.append((round(cos(A,A0)*cos(B,B0),3), l, m, round(float(np.linalg.norm(A0)*np.linalg.norm(B0)),2)))
    worst.sort(); out['worst_rotated_final']=worst[:8]; out['dW0_norm_by_module']={f'{l}.{m}':n for c,l,m,n in worst}
    R[run]=out
    print(f"\n=== {run}: scalars {s.size}, SFT moved {trainable.sum()}, ||s||={np.linalg.norm(s):.2f} (A {np.linalg.norm(sA):.2f}, B {np.linalg.norm(sB):.2f}; A init norm {out['init_norm_A']:.2f}), SFT rms/moved scalar {out['sft_rms_per_moved_scalar']:.4f}, B support frac {out['B_support_frac_mean']:.3f} constant={out['B_support_constant']}")
    print(T.loc[[x for x in [10,50,100,150,200,250] if x in T.index]].round(4).to_string())
    print('worst-rotated modules (cos, layer, module, ||ΔW0||):', worst[:6])
json.dump(R, open('/root/controllability-elicitation/rl/analysis/sweep29_kall/sft_direction.json','w'))
