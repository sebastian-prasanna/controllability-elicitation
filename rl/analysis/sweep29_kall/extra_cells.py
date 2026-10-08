import json, re, sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, 'rl/analysis/sweep29_kall'); import indist_drivers as m
cells = {"s29_k1k": m.ROOT/"rl/runs/sweep29_gptoss120b_fs1/sweep29-fs1-k1k", "s29_k3k": m.ROOT/"rl/runs/sweep29_gptoss120b_fs1/sweep29-fs1-k3k"}
jobs = []
for c, p in cells.items():
    for f in sorted((p/"iters").glob("iter-*.json*")):
        it = int(re.search(r"iter-(\d+)", f.name).group(1)); jobs.append((c, it, str(f), str(p/"batches"/f"batch-{it}.json")))
series = defaultdict(list)
with Pool(30) as pool:
    for c, o in pool.imap_unordered(m.process_iter, jobs, chunksize=2): series[c].append(o)
for c in series: series[c].sort(key=lambda o: o["iteration"])
S = json.load(open(m.ROOT/"rl/analysis/sweep29_kall/indist_series.json"))
S["train"].update(series)
for c, p in cells.items():
    S["metrics"][c] = [json.loads(l) for l in open(p/"metrics.jsonl")]
    S["progress"][c] = [json.loads(l) for l in open(p/"progress.jsonl")]
    S["eval_indist"][c] = m.eval_indist_stats(c, p)[1]
Path(m.ROOT/"rl/analysis/sweep29_kall/indist_series.json").write_text(json.dumps(S, default=float))
print("done")
