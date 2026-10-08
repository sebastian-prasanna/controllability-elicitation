"""Optimizer signals from metrics.jsonl / progress.jsonl (task 3)."""
import json, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path('/root/controllability-elicitation/rl/runs')
RUNS = {'s29_120b_kall': (ROOT/'sweep29_gptoss120b_fs1/sweep29-fs1-kall', 0.00033, 497664),
        's29_120b_k100k': (ROOT/'sweep29_gptoss120b_fs1/sweep29-fs1-k100k', 0.00316, 100000),
        's29_120b_k30k': (ROOT/'sweep29_gptoss120b_fs1/sweep29-fs1-k30k', 0.00577, 30000),
        's29_120b_k10k': (ROOT/'sweep29_gptoss120b_fs1/sweep29-fs1-k10k', 0.01, 10000),
        's24_20b_kall': (ROOT/'sweep24_controller/sweep24-fix05-kall', 0.00033, None),
        's24_20b_k100k': (ROOT/'sweep24_controller/sweep24-fix05-k100k', 0.00316, 100000)}
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
out = {}
for run, (rd, lr, k) in RUNS.items():
    M = pd.DataFrame([json.loads(l) for l in open(rd/'metrics.jsonl')]).set_index('iteration')
    P = pd.DataFrame([json.loads(l) for l in open(rd/'progress.jsonl')]).set_index('iteration')
    M = M.join(P[['compliance_rate','accuracy','reasoning_chars_median','distinct4_mean','zlib_ratio_median']], how='left')
    M['lr_x_gradnorm'] = lr * M['grad_norm']
    # window means
    wins = [(0,40),(40,80),(80,120),(120,160),(160,200),(200,250)]
    cols = ['grad_norm','lr_x_gradnorm','clip_frac','loss','logprob_diff_abs_mean','logprob_corr','tis_ratio_mean','tis_frac_truncated','reward_mean','reward_std','n_degenerate_groups','compliance_rate','accuracy','reasoning_chars_median']
    tab = pd.DataFrame({f'{a}-{b}': M.loc[a:b-1, cols].mean() for a,b in wins if a < len(M)}).T
    print(f'\n=== {run} lr={lr} n_iter={len(M)}  grad_norm>1 (clipped) frac={float((M.grad_norm>1).mean()):.3f}')
    print(tab.round(4).to_string())
    out[run] = dict(lr=lr, n_iter=int(len(M)), frac_gradnorm_clipped=float((M.grad_norm>1).mean()),
                    windows={i: {c: (None if pd.isna(v) else round(float(v),5)) for c,v in row.items()} for i,row in tab.iterrows()},
                    series={c: [None if pd.isna(v) else round(float(v),5) for v in M[c]] for c in ['grad_norm','clip_frac','logprob_diff_abs_mean','logprob_corr','reward_mean','compliance_rate','accuracy','reasoning_chars_median','n_degenerate_groups']})
json.dump(out, open('/root/controllability-elicitation/rl/analysis/sweep29_kall/metrics_analysis.json','w'))
