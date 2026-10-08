import json, sys
pool, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
H = int(sys.argv[4]) if len(sys.argv) > 4 else 280; T = int(sys.argv[5]) if len(sys.argv) > 5 else 200
rows = [json.loads(l) for l in open(f'rl/analysis/honest_audit/pools/{pool}.jsonl')]
for r in rows[lo:hi]:
    rs = r['reasoning'].replace('\n', '⏎')
    print(f"### {r['idx']} | {r['model']} {r['cell']} s{r['step']}({r['step_label']}) {r['block']} {r['mode']} | crit={','.join(r['criteria']) or '-'} | len={r['reasoning_len']} tok={r['n_tokens']} fr={r['finish_reason']} ans={r['extracted_answer']} d4={r['d4']} sd={r['sd']}")
    print("  HEAD:", rs[:H]); 
    if len(rs) > H: print("  TAIL:", rs[-T:])
    print("  OUT:", r['output'][:120].replace('\n','⏎'))
