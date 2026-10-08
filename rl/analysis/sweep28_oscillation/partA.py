import pandas as pd, numpy as np, re, string, itertools
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
df = pd.read_parquet('/root/controllability-elicitation/rl/analysis/sweep28_oscillation/samples.parquet')
df['qid'] = df.dataset.str.replace('_w_keyword','') + ':' + df.qid.astype(str)
S28 = ['s28_120b_k30k','s28_120b_k100k']
ho = df[df.split=='heldout']

print("### A1b: generate config constancy per run (temp, max_tokens), timestamps")
print(ho.groupby('run').agg(temps=('temp',lambda x: sorted(set(x))), mt=('max_tokens',lambda x: sorted(set(x))), n=('qid','size'), nsteps=('step','nunique')))
print(ho.groupby(['run','step']).timestamp.first().unstack(0).to_string())

print("\n### A2a: integrity per checkpoint: n results, unique qids, per-mode counts, mode-assignment identical across ckpts?")
for run in ho.run.unique():
    h = ho[ho.run==run]
    cnt = h.groupby('step').agg(n=('qid','size'), nq=('qid','nunique'), nerr=('error', lambda x: (x!='None').sum()),
                                nempty=('reasoning', lambda x: (x.str.len()==0).sum()))
    modes = h.groupby('step').mode.value_counts().unstack().fillna(0).astype(int)
    fr = h.groupby('step').finish_reason.value_counts().unstack().fillna(0).astype(int)
    assign = h.groupby('step').apply(lambda g: tuple(sorted(zip(g.qid, g['mode']))))
    print(f"-- {run}: n per ckpt {cnt.n.unique()}, nq {cnt.nq.unique()}, errors {cnt.nerr.sum()}, empty-reasoning total {cnt.nempty.sum()}, mode assignment identical across ckpts: {assign.nunique()==1}")
    print(pd.concat([modes, fr], axis=1).T.to_string())

print("\n### A2b: determinism — identical reasoning hash for same qid across different checkpoints (should be ~0 unless weights identical)")
for run in ho.run.unique():
    h = ho[ho.run==run].pivot(index='qid', columns='step', values='rhash')
    steps = sorted(h.columns)
    dup_adj = {steps[i+1]: int((h[steps[i]]==h[steps[i+1]]).sum()) for i in range(len(steps)-1)}
    allpairs = sum(int((h[a]==h[b]).sum()) for a,b in itertools.combinations(steps,2))
    print(f"-- {run}: identical-trace count between adjacent ckpts (of 200): {dup_adj}; total identical across all {len(steps)*(len(steps)-1)//2} pairs: {allpairs}")
    # within-checkpoint duplicates (same reasoning for different qids)
    w = ho[ho.run==run].groupby('step').rhash.apply(lambda x: x.duplicated().sum())
    print(f"   within-ckpt duplicate traces: {dict(w[w>0])}")

print("\n### A2c: start_of_sentence compliance x finish_reason (held-out), per run")
for run in ho.run.unique():
    x = ho[(ho.run==run)&(ho['mode']=='start_of_sentence')]
    print(run, pd.crosstab(x.finish_reason, x.compliance).to_dict())
    print("   truncated-trace compliance rate by step:", x[x.finish_reason=='length'].groupby('step').compliance.agg(['mean','size']).round(2).T.to_dict())
print("\n### A4a: held-out per-mode strict compliance by step")
pm = ho.groupby(['run','mode','step']).compliance.mean().unstack('step').round(2)
print(pm.to_string())
print("\n### A4a': held-out per-mode finish_reason=length fraction by step")
print(ho.assign(trunc=ho.finish_reason.eq('length')).groupby(['run','mode','step']).trunc.mean().unstack('step').round(2).to_string())
print("\n### A4b: in-dist T=0 per-mode strict compliance by step")
ind = df[df.split=='indist']
print(ind.groupby(['run','mode','step']).compliance.mean().unstack('step').round(2).to_string())
print(ind.groupby(['run','step']).compliance.mean().unstack('step').round(3).to_string())

print("\n### A3: per-question flip analysis (start_of_sentence)")
sos = ho[ho['mode']=='start_of_sentence']
for run in ho.run.unique():
    M = sos[sos.run==run].pivot(index='qid', columns='step', values='compliance')
    rate = M.mean(0)
    pq = M.mean(1)
    print(f"\n-- {run}: {M.shape[0]} questions x {M.shape[1]} ckpts. per-ckpt rate: {rate.round(2).to_dict()}")
    print(f"   per-question mean compliance distribution: min {pq.min():.2f} q25 {pq.quantile(.25):.2f} med {pq.median():.2f} q75 {pq.quantile(.75):.2f} max {pq.max():.2f}; always-pass {int((pq==1).sum())}, never-pass {int((pq==0).sum())}, in (0,1): {int(((pq>0)&(pq<1)).sum())}")
    print(f"   histogram of per-question mean (bins of .1): {np.histogram(pq, bins=np.linspace(0,1,11))[0].tolist()}")
    # flips between adjacent checkpoints
    steps = sorted(M.columns)
    flips = [(steps[i+1], int((M[steps[i]]!=M[steps[i+1]]).sum())) for i in range(len(steps)-1)]
    print(f"   adjacent-ckpt flips (of {M.shape[0]}): {flips}")
    # trough analysis: Jaccard overlap of failing sets among trough ckpts vs null
    troughs = [s for s in steps if rate[s] < 0.45]; peaks = [s for s in steps if rate[s] >= 0.7]
    def jacc(a,b):
        A=set(M.index[M[a]==0]); B=set(M.index[M[b]==0]); return len(A&B)/max(1,len(A|B))
    if len(troughs)>1:
        js = [jacc(a,b) for a,b in itertools.combinations(troughs,2)]
        # null: independent failures with same marginals -> expected Jaccard
        def null_j(a,b):
            pa=1-rate[a]; pb=1-rate[b]; return pa*pb/(pa+pb-pa*pb)
        ns = [null_j(a,b) for a,b in itertools.combinations(troughs,2)]
        print(f"   troughs {troughs}: mean Jaccard of failing sets {np.mean(js):.2f} vs independent-null {np.mean(ns):.2f}")
    if len(peaks)>1:
        js = [jacc(a,b) for a,b in itertools.combinations(peaks,2)]
        print(f"   peaks {peaks}: mean Jaccard of failing sets {np.mean(js):.2f}")
    # question-level predictor: does trough failure correlate with peak failure? logistic-ish: corr of pq_trough vs pq_peak
    if troughs and peaks:
        pt = M[troughs].mean(1); pp = M[peaks].mean(1)
        print(f"   corr(per-q pass rate at troughs, at peaks) = {np.corrcoef(pt,pp)[0,1]:.2f}; per-q pass at troughs by dataset: {pt.groupby(sos[sos.run==run].drop_duplicates('qid').set_index('qid').dataset).mean().round(2).to_dict()}")
# cross-run correlation of per-question mean
P = {run: sos[sos.run==run].pivot(index='qid', columns='step', values='compliance').mean(1) for run in ho.run.unique()}
P = pd.DataFrame(P)
print("\n   cross-run correlation of per-question mean compliance (start_of_sentence):"); print(P.corr().round(2).to_string())
# does per-question difficulty correlate with reasoning length / dataset?
q = sos.groupby('qid').agg(dataset=('dataset','first'), pq=('compliance','mean'), rlen=('reasoning', lambda x: x.str.len().median()))
print("   per-question mean compliance by dataset (all runs pooled):", q.groupby('dataset').pq.mean().round(2).to_dict())
print("   corr(per-q compliance, median reasoning len):", round(np.corrcoef(q.pq, q.rlen)[0,1],2))
