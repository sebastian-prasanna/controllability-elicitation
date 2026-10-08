"""Flatten held-out / in-dist eval checkpoints of the 5 runs into one cached parquet."""
import json, re, hashlib, sys
from pathlib import Path
import pandas as pd, zstandard

ROOT = Path('/root/controllability-elicitation/rl/runs')
RUNS = {
 's28_120b_k30k':  ROOT/'sweep28_gptoss120b_highk/sweep28-d1-k30k',
 's28_120b_k100k': ROOT/'sweep28_gptoss120b_highk/sweep28-d1-k100k',
 's24_20b_k30k':   ROOT/'sweep24_controller/sweep24-fix05-k30k',
 's24_20b_k100k':  ROOT/'sweep24_controller/sweep24-fix05-k100k',
 's14_120b_k3k':   ROOT/'sweep14_x320gepa_gptoss120b/sweep14-x320gepa-gptoss120b-k3k',
}
OUT = Path('/root/controllability-elicitation/rl/analysis/sweep28_oscillation/samples.parquet')

def load(f):
    b = Path(f).read_bytes()
    if f.suffix == '.zst':
        b = zstandard.ZstdDecompressor().decompress(b, max_output_size=1<<31)
    return json.loads(b)

rows = []
for run, rd in RUNS.items():
    for split in ('heldout', 'indist'):
        d = rd/'eval'/split
        if not d.exists(): continue
        for f in sorted(d.glob('checkpoint-*.json*')):
            step = int(re.search(r'checkpoint-(\d+)', f.name).group(1))
            D = load(f)
            gc = D['config']['backend_info']['generate_config']
            for r in D['results']:
                for si, s in enumerate(r['samples']):
                    rz = s.get('reasoning') or ''
                    rows.append(dict(run=run, split=split, step=step, qid=r['id'], dataset=r['dataset'],
                        mode=r['mode'], control_value=str(r.get('control_value')), sample_idx=si,
                        reasoning=rz, output=(s.get('output') or '')[:500],
                        compliance=int(s.get('compliance') or 0), correct=bool(s.get('correct')),
                        finish_reason=s.get('finish_reason'), error=str(s.get('error')),
                        temp=gc['temperature'], max_tokens=gc['max_tokens'], lora_path=D['config']['backend_info']['lora_path'],
                        timestamp=D['config'].get('timestamp'), question=r['question'][:300],
                        rhash=hashlib.md5(rz.encode()).hexdigest()))
            print(run, split, step, len(D['results']), file=sys.stderr)
df = pd.DataFrame(rows)
df.to_parquet(OUT)
print(df.shape); print(df.groupby(['run','split']).step.nunique())
