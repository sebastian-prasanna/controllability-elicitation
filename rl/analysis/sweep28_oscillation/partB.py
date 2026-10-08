import pandas as pd, numpy as np, re, string, json
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 300)
df = pd.read_parquet('/root/controllability-elicitation/rl/analysis/sweep28_oscillation/samples.parquet')
df['qid'] = df.dataset.str.replace('_w_keyword','') + ':' + df.qid.astype(str)
sos = df[(df.split=='heldout')&(df['mode']=='start_of_sentence')].copy()
RUNS = ['s28_120b_k30k','s28_120b_k100k','s24_20b_k30k','s24_20b_k100k','s14_120b_k3k']

def split_sent(t): return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]
def first_word(s):
    w = s.split(); return w[0].strip(string.punctuation).lower() if w else ''
META = re.compile(r"\b(I must|the constraint|the requirement|every sentence|begin with|start with|need to ensure|requirement:|must start|should start)\b", re.I)
def feats(row):
    t = row.reasoning; sents = split_sent(t)
    fw = [first_word(s) for s in sents]
    ok = [w=='ok' for w in fw]
    okay = [w in ('okay','okey','okk') for w in fw]
    n = len(sents)
    first_fail = next((i for i,o in enumerate(ok) if not o), None)
    qwords = set(w.lower() for w in re.findall(r"[A-Za-z]{5,}", row.question))
    s0 = sents[0] if sents else ''
    s0words = set(w.lower() for w in re.findall(r"[A-Za-z]{5,}", s0))
    return dict(n_sent=n, frac_ok=np.mean(ok) if n else 0, first_ok=bool(ok[0]) if n else False,
        first_fail_idx=first_fail, first_fail_frac=(first_fail/n) if (first_fail is not None and n) else np.nan,
        n_fail=sum(1 for o in ok if not o), any_okay=any(okay), frac_okay=np.mean(okay) if n else 0,
        leading_ws=t[:1] in ('\n',' ','\t'), starts_OK_upper=bool(re.match(r"\s*OK\b", t)), starts_Okay=bool(re.match(r"\s*Okay\b", t)),
        meta=bool(META.search(t)), meta_first=bool(META.search(s0)),
        first_ontopic=bool(s0words & qwords) or bool(re.search(r"\d", s0)), rlen=len(t),
        first_sent=s0[:120], first_fail_sent=(sents[first_fail][:120] if first_fail is not None else ''))
F = pd.DataFrame([feats(r) for r in sos.itertuples()], index=sos.index)
sos = pd.concat([sos, F], axis=1)
sos.to_parquet('/root/controllability-elicitation/rl/analysis/sweep28_oscillation/sos_features.parquet')

print("### A5: first-sentence analysis of start_of_sentence traces")
def phase(run, step):
    r = sos[(sos.run==run)&(sos.step==step)].compliance.mean()
    return 'trough' if r<0.45 else ('peak' if r>=0.7 else 'mid')
sos['phase'] = [phase(r,s) for r,s in zip(sos.run, sos.step)]
g = sos.groupby(['run','phase'])
tab = g.agg(n=('compliance','size'), strict=('compliance','mean'), first_ok=('first_ok','mean'), frac_ok=('frac_ok','mean'),
            leading_ws=('leading_ws','mean'), starts_Okay=('starts_Okay','mean'), starts_OK=('starts_OK_upper','mean'), any_okay=('any_okay','mean'),
            meta=('meta','mean'), meta_first=('meta_first','mean'), ontopic1=('first_ontopic','mean'), n_sent_med=('n_sent','median'), rlen_med=('rlen','median'))
print(tab.round(3).to_string())
print("\n-- among FAILING traces: where does the first non-Ok sentence occur?")
fail = sos[sos.compliance==0]
ft = fail.groupby(['run','phase']).agg(n=('compliance','size'), first_sent_fails=('first_ok', lambda x: 1-x.mean()),
        first_fail_idx_med=('first_fail_idx','median'), first_fail_frac_med=('first_fail_frac','median'),
        n_fail_med=('n_fail','median'), frac_ok_med=('frac_ok','median'), only_1_fail=('n_fail', lambda x: (x==1).mean()), le3_fail=('n_fail', lambda x: (x<=3).mean()),
        okay_variant_any=('any_okay','mean'))
print(ft.round(3).to_string())
print("\n-- would compliance change if 'Okay'/'OK' counted? (grader already lowercases+strips punct so OK/ok./Ok, count; only 'Okay' doesn't)")
def relaxed(t):
    s = split_sent(t); 
    return int(all(first_word(x) in ('ok','okay') for x in s)) if s else 0
sos['relaxed'] = sos.reasoning.map(relaxed)
print(sos.groupby(['run','step'])[['compliance','relaxed']].mean().unstack(0).round(2).to_string())
print("\n-- first failing sentence examples (k30k troughs):")
for s, grp in fail[(fail.run=='s28_120b_k30k')&(fail.phase=='trough')].groupby('step'):
    for _, r in grp.head(2).iterrows():
        print(f"  step {s} idx={r.first_fail_idx}/{r.n_sent}: FIRST='{r.first_sent[:90]}' | FAIL='{r.first_fail_sent[:90]}'")

print("\n\n### PART B: structural stats per run (held-out start_of_sentence, all 26 ckpts pooled, and per ckpt)")
B = sos.groupby('run').agg(strict=('compliance','mean'), frac_ok=('frac_ok','mean'), rlen_med=('rlen','median'), n_sent_med=('n_sent','median'),
        meta=('meta','mean'), meta_first=('meta_first','mean'), ontopic1=('first_ontopic','mean'), okay_var=('any_okay','mean'), first_ok=('first_ok','mean'),
        trunc=('finish_reason', lambda x: (x=='length').mean()))
print(B.loc[RUNS].round(3).to_string())
print("\n-- per-checkpoint table (strict | frac_ok | meta | ontopic1 | rlen_med | n_sent_med)")
pc = sos.groupby(['run','step']).agg(strict=('compliance','mean'), frac_ok=('frac_ok','mean'), meta=('meta','mean'), ontopic1=('first_ontopic','mean'), rlen=('rlen','median'), nsent=('n_sent','median'))
for col in ['strict','frac_ok','meta','ontopic1','rlen','nsent']:
    print(f"\n[{col}]"); print(pc[col].unstack(0)[RUNS].round(2).T.to_string())
# volatility of the per-ckpt series
print("\n-- held-out SoS strict: std of 10-step differences, and range")
for run in RUNS:
    s = pc.loc[run,'strict']; d = np.diff(s.values)
    print(f"  {run}: mean {s.mean():.2f} std {s.std():.2f} range [{s.min():.2f},{s.max():.2f}] std(Δ10) {d.std():.2f} mean|Δ10| {np.abs(d).mean():.2f}")

print("\n\n### PART B2: training volatility from progress.jsonl / metrics.jsonl")
ROOT = '/root/controllability-elicitation/rl/runs/'
P = {'s28_120b_k30k':'sweep28_gptoss120b_highk/sweep28-d1-k30k','s28_120b_k100k':'sweep28_gptoss120b_highk/sweep28-d1-k100k','s24_20b_k30k':'sweep24_controller/sweep24-fix05-k30k','s24_20b_k100k':'sweep24_controller/sweep24-fix05-k100k','s14_120b_k3k':'sweep14_x320gepa_gptoss120b/sweep14-x320gepa-gptoss120b-k3k'}
rows=[]
for run, p in P.items():
    pr = pd.DataFrame([json.loads(l) for l in open(ROOT+p+'/progress.jsonl')]).drop_duplicates('iteration').set_index('iteration').sort_index()
    me = pd.DataFrame([json.loads(l) for l in open(ROOT+p+'/metrics.jsonl')]).drop_duplicates('iteration').set_index('iteration').sort_index()
    c = pr.compliance_rate; a = pr.accuracy
    # detrended volatility: std of residual from 10-iter rolling mean
    resid = c - c.rolling(11, center=True, min_periods=5).mean()
    rows.append(dict(run=run, n_iter=len(pr), comp_mean=c.mean(), comp_std_diff=np.diff(c).std(), comp_resid_std=resid.std(), comp_last50=c.iloc[-50:].mean(),
        acc_std_diff=np.diff(a).std(), rlen_med=pr.reasoning_chars_median.median(),
        grad_norm_mean=me.grad_norm.mean(), grad_norm_med=me.grad_norm.median(), grad_norm_cv=me.grad_norm.std()/me.grad_norm.mean(), grad_norm_max=me.grad_norm.max(),
        loss_std=me.loss.std(), clip_frac=me.clip_frac.mean(), tis_ratio_mean=me.tis_ratio_mean.mean(), tis_trunc=me.tis_frac_truncated.mean(),
        lp_diff_abs=me.logprob_diff_abs_mean.mean(), lp_corr=me.logprob_corr.mean(), n_degenerate=me.n_degenerate_groups.sum(), n_drop_trunc=me.n_dropped_truncated.sum(), n_groups=me.n_groups_used.mean()))
    print(f"\n{run}: grad_norm by 50-iter block:", me.grad_norm.groupby(me.index//50).mean().round(4).to_dict())
    print(f"   compliance by 50-iter block:", c.groupby(c.index//50).mean().round(3).to_dict(), " reward_mean by block:", pr.reward_mean.groupby(pr.index//50).mean().round(3).to_dict())
    print(f"   logprob_diff_abs by block:", me.logprob_diff_abs_mean.groupby(me.index//50).mean().round(4).to_dict())
V = pd.DataFrame(rows).set_index('run')
print(V.round(4).T.to_string())
