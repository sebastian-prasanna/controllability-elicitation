"""Exact base-model and final-checkpoint losses for an EDL analysis of a masked-LoRA
k-sweep. Launches Modal jobs (cotcontrol.training.loss_eval) and writes per-row
records to <sweep>/edl/losses/<adapter>.jsonl.

Subsets scored:
  train    all training rows (base model only)          -> exact L_0 per batch
  heldout  <sweep>/edl/heldout.jsonl (base + every final ckpt) -> exact L_test
  memo     training rows seen at the first/last 20 steps (every final ckpt)
           -> memorization gap of the final model on seen rows

    python sft/edl/run_loss_eval.py sft/runs/x320_gptoss120b_gepa_general_drive3 [--smoke] [--skip-existing]

--skip-existing skips adapters (and the base pass) that already have a losses/<name>.jsonl, so the
script can be re-run after new cells (e.g. the kall / k400-900 extension) finish training.
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from cotcontrol.training.loss_eval import app, eval_losses  # noqa: E402
import modal  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cells, sweep_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--step", type=int, default=540)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=4, help="containers for the checkpoint passes")
    ap.add_argument("--mid-steps", type=int, nargs="*", default=[10, 40, 150],
                    help="extra intermediate checkpoints (heldout+memo) for --mid-ks")
    ap.add_argument("--mid-ks", type=int, nargs="*", default=[1000, 10000, 100000])
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    sweep = Path(a.sweep)
    edl = sweep / "edl"
    out_dir = edl / ("losses_smoke" if a.smoke else "losses")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = sweep_config(sweep)
    base_model, data_path = cfg["base_model"], cfg["data_path"]
    attn = cfg.get("attn_implementation")  # None -> sdpa (eager for gpt-oss), as in training
    train_rows = [json.loads(l) for l in open(REPO / data_path)]
    heldout = [json.loads(l) for l in open(edl / "heldout.jsonl")]
    order = json.load(open(edl / "data_order.json"))
    # data_order rows are indexed in the worker's post-shuffle order; recover file rows
    import random
    shuffled = list(range(len(train_rows)))
    random.Random(cfg["seed"]).shuffle(shuffled)  # shuffled[i] = file index of shuffled row i
    steps = order["steps"]
    early = [shuffled[j] for s in steps[:20] for j in s["rows"]]
    late = [shuffled[j] for s in steps[-20:] for j in s["rows"]]
    memo_idx = early + late
    memo_rows = [train_rows[i] for i in memo_idx]

    ckpts = []
    for c in cells(sweep):
        ckpts.append((c.label, c.ckpt_path(a.step)))
        if c.kind == "masked" and c.k in a.mid_ks:
            ckpts += [(f"{c.label}@{st}", c.ckpt_path(st)) for st in a.mid_steps]
    do_base = True
    if a.skip_existing and not a.smoke:
        have = {p.stem for p in out_dir.glob("*.jsonl")}
        ckpts = [c for c in ckpts if c[0] not in have]
        do_base = "base" not in have
        print(f"skip-existing: {len(have)} adapters already scored; base={'todo' if do_base else 'done'}; {len(ckpts)} to score")
    if a.smoke:
        train_rows, heldout, memo_rows, memo_idx = train_rows[:6], heldout[:4], memo_rows[:4], memo_idx[:4]
        ckpts = ckpts[:1]
    print(f"{base_model}: train {len(train_rows)} heldout {len(heldout)} memo {len(memo_rows)} ckpts {len(ckpts)}")

    # job 0: base on train+heldout ; jobs 1..n: ckpt groups on heldout+memo
    jobs = [("base", [("base", None)], train_rows + heldout,
             ["train"] * len(train_rows) + ["heldout"] * len(heldout),
             list(range(len(train_rows))) + [None] * len(heldout))] if do_base else []
    n_jobs = min(a.n_jobs, len(ckpts)) or 1
    groups = [ckpts[i::n_jobs] for i in range(n_jobs)] if not a.smoke else [ckpts]
    for gi, g in enumerate(groups):
        if g:
            jobs.append((f"ckpt{gi}", g, heldout + memo_rows,
                         ["heldout"] * len(heldout) + ["memo"] * len(memo_rows),
                         [None] * len(heldout) + memo_idx))
    t0 = time.time()
    with modal.enable_output(), app.run():
        calls = [(j, eval_losses.spawn(base_model, j[1], j[2], attn_implementation=attn,
                                       chat_template_kwargs=cfg.get("chat_template_kwargs"))) for j in jobs]
        for (tag, adapters, rows, subsets, fidx), call in calls:
            res = call.get()
            for name, recs in res.items():
                with open(out_dir / f"{name}.jsonl", "a") as f:
                    for r, sub, fi, row in zip(recs, subsets, fidx, rows):
                        f.write(json.dumps({"adapter": name, "subset": sub, "file_idx": fi,
                                            "mode": row["meta"]["mode"], "dataset": row["meta"]["dataset"],
                                            "id": row["meta"]["id"], **r}) + "\n")
            print(f"[{tag}] saved {list(res)} after {time.time()-t0:.0f}s", flush=True)
    print("done")


if __name__ == "__main__":
    main()
