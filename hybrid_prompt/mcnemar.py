"""Paired McNemar tests: hybrid vs its parents (same questions, same mode assignment)."""
import glob, json
from math import comb
from pathlib import Path

REPO = Path('/root/controllability-elicitation')

def load(run_dir):
    f = glob.glob(str(run_dir / '*_all_all.json'))
    assert len(f) == 1, (run_dir, f)
    d = json.load(open(f[0]))
    out = {}
    for r in d['results']:
        s = r['samples'][0]
        c = s['compliance']
        strict = c.get('strict') if isinstance(c, dict) else c
        out[(r['dataset'], r['id'], r['mode'])] = bool(strict)
    return out

def mcnemar(a, b):
    """two-sided exact binomial on discordant pairs; a,b are dicts key->bool"""
    ks = sorted(set(a) & set(b))
    n01 = sum(1 for k in ks if not a[k] and b[k])   # b only
    n10 = sum(1 for k in ks if a[k] and not b[k])   # a only
    n = n01 + n10
    if n == 0:
        return n10, n01, 1.0, len(ks)
    k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return n10, n01, p, len(ks)

HYB = REPO / 'hybrid_prompt/runs/gptoss20b'
PR = REPO / 'pinned_reeval/runs/gptoss20b'
for split in ('test', 'heldout'):
    print(f'=== {split} ===')
    arms = {'gepa_s1': PR / f'gepa_s1_{split}', 'fewshot_s2': PR / f'fewshot_s2_{split}',
            'hybrid_plain': HYB / f'hybrid_plain_{split}', 'hybrid_clean': HYB / f'hybrid_clean_{split}'}
    D = {k: load(v) for k, v in arms.items()}
    for k, v in D.items():
        print(f'  {k:14} strict={sum(v.values())/len(v):.4f}  n={len(v)}')
    for a, b in [('hybrid_plain','gepa_s1'), ('hybrid_plain','fewshot_s2'),
                 ('hybrid_clean','gepa_s1'), ('hybrid_clean','fewshot_s2'),
                 ('hybrid_plain','hybrid_clean')]:
        n10, n01, p, n = mcnemar(D[a], D[b])
        d = sum(D[a].values())/n - sum(D[b].values())/n
        print(f'  {a} vs {b}: diff={d:+.4f}  {a}-only={n10} {b}-only={n01}  p={p:.3g}')
