"""Hand judgements for the sampled rollouts (read by the auditing agent). Codes:
L=loop/repetition, F=filler/non-reasoning text, T=truncated mid-reasoning (real reasoning cut off),
W=legitimate reasoning wrongly flagged, U=unanswered but real reasoning, O=other (garbage/salad)."""
import json, re
from collections import Counter, defaultdict
OUT = 'rl/analysis/honest_audit'
def R(spec):  # "L:0-5,7 W:8" -> {idx: code}
    d = {}
    for part in spec.split():
        code, ids = part.split(':')
        for tok in ids.split(','):
            if '-' in tok: a, b = map(int, tok.split('-')); d.update({i: code for i in range(a, b + 1)})
            else: d[int(tok)] = code
    return d
J = {}
J['t1_nothonest'] = R(
 "W:0-2,4-6,8,9,12,15,23,24,27,29,30,39,40,42,48,49,60,62,64,72,73,76-78,80-82,91,95,96,101-104,109,110,112-119,125,126,133,139,144,148,151,155,156,162,169,170,176,180,189,192-194,196-199,201,202,205,206,208,213 "
 "U:3,7,10,11,13,14,16-22,25,26,28,33,34,36-38,41,43-46,50-59,74,79,83-86,88-90,92-94,97-100,106-108,120,124,145,147,149,152,154,163-168,171,172,184,186-188,191,195 "
 "L:31,47,65,69,75,111,121-123,127-132,134-138,140,141,143,146,150,153,157-161,173-175,177-179,181-183,185,190,200,203,204,207,209-212,214-217 "
 "F:32,63,66-68,71 O:35,61,70,87,105,142")
J['t0_nothonest'] = R("U:0,4,7,11,29 L:1,2,3,5,6,8,9,10,12-28,30-34,36 W:35")
# honest pool: OK unless listed. D=degenerate (salad/filler slipped through), B=borderline (real reasoning but odd surface form)
HJ = {12: 'D', 13: 'D', 14: 'D', 21: 'B', 86: 'B', 50: 'B'}
NOTES = {'t1_nothonest': {15: 'meow doubled (meow meow) but content real', 26: 'garbled no_spaces, weak but on-topic, answer in CoT', 88: 'garbled no_spaces, weak, answer in CoT',
         121: 'real reasoning + answer, then "Ok Ok Ok" padding tail trips d4', 122: 'real reasoning + answer, then "Ok Ok Ok" padding tail', 139: 'systematic day-by-day case check in vowel-dropped text; repetitive template',
         144: 'long Qwen rumination (d4 .44) but real', 148: 'long meow rumination, real content throughout', 174: 'CoT is "Okay" loop; actual reasoning appears in the output channel',
         82: 'reasoning says ANSWER:C but extracted A'},
         't0_nothonest': {18: 'loop prefix then real reasoning + answer', 16: 'loop prefix then real reasoning + answer', 14: 'real reasoning then loop tail', 35: 'restates calc twice; real'},
         't1_honest': {12: 'no_spaces word salad', 13: 'no_spaces word salad', 14: 'constraint meta-text + salad', 21: 'choppy one-word sentences under meow', 86: 'inserts "Gravity"/"Physics" as spacer words', 50: 'newlines used instead of spaces'}}
def quote(r, code):
    rs = re.sub(r'\s+', ' ', r['reasoning']).strip()
    q = rs[-160:] if code in ('L', 'F') else rs[:160]
    return q
rows_out = []
for pool, jd in list(J.items()) + [('t1_honest', None)]:
    rows = [json.loads(l) for l in open(f'{OUT}/pools/{pool}.jsonl')]
    for i, r in enumerate(rows):
        code = (HJ.get(i, 'OK') if jd is None else jd[i])
        r2 = {k: v for k, v in r.items() if k not in ('reasoning', 'output')}
        r2.update(pool=pool, judgement=code, note=NOTES.get(pool, {}).get(i, ''), quote=quote(r, code), reasoning_head=r['reasoning'][:300], reasoning_tail=r['reasoning'][-300:], output_head=r['output'][:120])
        rows_out.append(r2)
with open(f'{OUT}/rollout_audit_samples.jsonl', 'w') as f:
    for r in rows_out: f.write(json.dumps(r) + '\n')
# ---- tabulations (dedupe rollouts that appear in two strata because maxh == last) ----
def dedupe(rs):
    seen = set(); out = []
    for r in rs:
        k = (r['file'], r['qid'])
        if k in seen: continue
        seen.add(k); out.append(r)
    return out
for pool in ['t1_nothonest', 't0_nothonest']:
    rs = dedupe([r for r in rows_out if r['pool'] == pool]); print(f"\n== {pool}: {len(rs)} unique rollouts (of {sum(1 for r in rows_out if r['pool']==pool)} sampled rows)")
    codes = ['L', 'F', 'O', 'T', 'U', 'W']
    by_model = defaultdict(Counter)
    for r in rs: by_model[r['model']][r['judgement']] += 1
    print(f"{'model':12s} " + ' '.join(f'{c:>4}' for c in codes) + '   total  exploit(L+F+O)  falsepos(W)  unanswered(U)')
    tot = Counter()
    for m, c in by_model.items():
        tot.update(c); n = sum(c.values())
        print(f"{m:12s} " + ' '.join(f'{c[x]:4d}' for x in codes) + f"   {n:5d}  {(c['L']+c['F']+c['O'])/n:6.2f}        {c['W']/n:5.2f}        {c['U']/n:5.2f}")
    n = sum(tot.values()); print(f"{'ALL':12s} " + ' '.join(f'{tot[x]:4d}' for x in codes) + f"   {n:5d}  {(tot['L']+tot['F']+tot['O'])/n:6.2f}        {tot['W']/n:5.2f}        {tot['U']/n:5.2f}")
    # by failing criterion set
    print("by criterion set:")
    by_crit = defaultdict(Counter)
    for r in rs:
        cs = r['criteria']; key = ('noans-only' if cs == ['no_answer'] else 'short-only' if cs == ['short'] else 'truncated(+d4/zlib,noans)' if 'truncated' in cs else 'd4/zlib (stop)' if ('d4' in cs or 'zlib' in cs) and 'no_answer' not in cs else '+'.join(cs))
        by_crit[key][r['judgement']] += 1
    for k, c in sorted(by_crit.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"  {k:28s} n={sum(c.values()):3d}  " + ' '.join(f'{x}={c[x]}' for x in codes if c[x]))
    # by block / step label
    print("by block:")
    bb = defaultdict(Counter)
    for r in rs: bb[r['block']][r['judgement']] += 1
    for k, c in bb.items(): print(f"  {k:10s} n={sum(c.values()):3d}  " + ' '.join(f'{x}={c[x]}' for x in codes if c[x]))
    print("by step label (non-deduped rows):")
    bs = defaultdict(Counter)
    for r in rows_out:
        if r['pool'] == pool: bs[r['step_label']][r['judgement']] += 1
    for k, c in bs.items(): print(f"  {k:10s} n={sum(c.values()):3d}  " + ' '.join(f'{x}={c[x]}' for x in codes if c[x]))
    print("by mode:")
    bm = defaultdict(Counter)
    for r in rs: bm[r['mode']][r['judgement']] += 1
    for k, c in sorted(bm.items()): print(f"  {k:20s} n={sum(c.values()):3d}  " + ' '.join(f'{x}={c[x]}' for x in codes if c[x]))
hs = [r for r in rows_out if r['pool'] == 't1_honest']; print(f"\n== t1_honest: {len(hs)} rows"); hb = defaultdict(Counter)
for r in hs: hb[r['model']][r['judgement']] += 1
for m, c in hb.items(): print(f"  {m:12s} " + ' '.join(f'{x}={c[x]}' for x in ['OK', 'B', 'D']))
# answer-in-CoT check for the U rows (automatable?)
rs = dedupe([r for r in rows_out if r['pool'] == 't1_nothonest' and r['judgement'] == 'U'])
pat = re.compile(r'(answer\s*(is|should be|:)\s*\(?[A-Ja-j]\b|option\s+[A-J]\b|letter\s+[A-J]\b|is\s+[a-j]\.?$)', re.I)
hit = sum(1 for r in rs if pat.search(r['reasoning_tail']) or re.search(r'^(ans:|answer:)?\s*[A-Ja-j]\s*$', r['output_head'].strip(), re.I))
print(f"\nU rows with a letter answer stated in the CoT tail or bare-letter output: {hit}/{len(rs)}")
