"""Reconstruct the exact first-epoch data order of an x320 sweep (all cells share seed, data
and batch size) and the supervised-token count of every row / step. Mirrors the worker:
random.Random(seed).shuffle(rows) pre-shuffle, then the HF Trainer RandomSampler
(torch.randperm with generator seeded by `seed`) in 16-row global batches, no packing.
Validated on gpt-oss-120b: exact base loss on the step-1 batch matches the logged loss to 0.4%.

    HF_HUB_OFFLINE=1 python sft/edl/build_data_order.py sft/runs/<sweep>
"""
from __future__ import annotations
import argparse, json, random, statistics, sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import sweep_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--tokenizer", default=None, help="override tokenizer id (default: base_model)")
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer
    from cotcontrol.training.rendering import render_row

    sweep = Path(a.sweep); cfg = sweep_config(sweep)
    assert cfg.get("shuffle", True) and not cfg.get("pack_sequences"), cfg
    bs, steps, seed, msl = cfg["batch_size"], cfg["max_steps"], cfg["seed"], cfg["max_seq_length"]
    rows = [json.loads(l) for l in open(REPO / cfg["data_path"])]
    n = len(rows)
    random.Random(seed).shuffle(rows)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    tok = AutoTokenizer.from_pretrained(a.tokenizer or cfg["base_model"], trust_remote_code=True)
    out = []
    for i, r in enumerate(rows):
        rec = render_row(tok, r, msl, chat_template_kwargs=cfg.get("chat_template_kwargs"))
        out.append({"shuffled_idx": i, "n_tokens": len(rec["input_ids"]),
                    "n_sup": sum(1 for l in rec["labels"] if l != -100),
                    "truncated": rec["truncated"], "mode": r["meta"]["mode"]})
    order = []
    for s in range(steps):
        idx = perm[bs * s: bs * s + bs]
        order.append({"step": s + 1, "rows": idx, "n_sup": sum(out[j]["n_sup"] for j in idx),
                      "n_tok": sum(out[j]["n_tokens"] for j in idx)})
    assert len(order[-1]["rows"]) == bs, "sweep is not exactly one epoch"
    (sweep / "edl").mkdir(exist_ok=True)
    json.dump({"rows": out, "steps": order, "config": {k: cfg[k] for k in ("base_model", "data_path", "seed", "batch_size", "max_steps")}},
              open(sweep / "edl" / "data_order.json", "w"))
    print(f"{sweep.name}: {n} rows, {steps} steps x {bs}; supervised tokens {sum(o['n_sup'] for o in out)} "
          f"(mean/row {statistics.mean(o['n_sup'] for o in out):.0f}); truncated {sum(o['truncated'] for o in out)}; "
          f"modes {len(Counter(o['mode'] for o in out))}")


if __name__ == "__main__":
    main()
