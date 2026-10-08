"""Per-10-step LoRA update norms from adapter checkpoints pulled from the Modal volume
(cotcontrol_sebastian-prasanna_checkpoints) into /tmp/adapters/<run>/checkpoint-<step>/adapter_model.safetensors.
Only masked (trainable) entries change, so ||Δ|| over the whole adapter == ||Δ|| over the k trainable params."""
import glob, re, numpy as np, pandas as pd
from safetensors.numpy import load_file
K = {'s28_120b_k30k':30000,'s28_120b_k100k':100000,'s24_20b_k30k':30000,'s24_20b_k100k':100000,'s14_120b_k3k':3000}
TRANS = {'s28_120b_k30k':[(0,10),(40,50),(70,80),(150,160),(210,220)], 's28_120b_k100k':[(20,30),(90,100),(140,150),(190,200)]}
rows=[]; per={}
for run,k in K.items():
    fs = sorted(glob.glob(f'/tmp/adapters/{run}/checkpoint-*/adapter_model.safetensors'), key=lambda f:int(re.search(r'checkpoint-(\d+)',f).group(1)))
    W={}
    for f in fs:
        st=int(re.search(r'checkpoint-(\d+)',f).group(1)); d=load_file(f); W[st]=np.concatenate([v.astype(np.float32).ravel() for kk,v in sorted(d.items())])
    steps=sorted(W)
    if len(steps)<2: print(run,'missing'); continue
    d10 = pd.Series({b: np.linalg.norm(W[b]-W[a]) for a,b in zip(steps[:-1],steps[1:])})
    per[run]=d10
    nnz = int((W[steps[1]]-W[steps[0]]!=0).sum())
    rows.append(dict(run=run, n_ckpt=len(steps), nnz_changed=nnz, d10_mean=d10.mean(), d10_std=d10.std(), d10_min=d10.min(), d10_max=d10.max(),
        d10_over_sqrtk=d10.mean()/np.sqrt(k), rms_per_param=d10.mean()/np.sqrt(nnz), drift_250=np.linalg.norm(W[steps[-1]]-W[steps[0]]),
        drift_over_sum=np.linalg.norm(W[steps[-1]]-W[steps[0]])/d10.sum(), theta0_norm=np.linalg.norm(W[steps[0]]),
        trans=[round(float(d10[b]),2) for a,b in TRANS.get(run,[]) if b in d10.index]))
T=pd.DataFrame(rows).set_index('run'); pd.set_option('display.width',250)
print(T.round(3).to_string())
print("\nper-10-step ||Δθ||2 by checkpoint:"); print(pd.DataFrame(per).T.round(2).to_string())
T.to_csv('/root/controllability-elicitation/rl/analysis/sweep28_oscillation/update_norms.csv')
