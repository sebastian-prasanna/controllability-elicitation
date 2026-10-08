"""Assemble an SDF training set from generated documents.

  synth docs  (sdf/runs/docs/<arm>/synth_docs.jsonl)        -> {"text", "meta"}
+ pretraining mix (sdf/data/c4_50k.jsonl, default 1:1)      -> {"text", "meta"}
+ optional chat rows (any {"input","output"} jsonl, --chat)  -> passed through
-> seeded shuffle -> sdf/datasets/<name>.jsonl + <name>.manifest.json

Options mirror believe-it-or-not's synth_docs_to_ft_format: --doctag adds a
masked "<DOCTAG>\\n" prefix to every synthetic doc (the paper's mainline
conditioning); --c4-ratio sets pretraining docs per synthetic doc (paper: 1.0);
--max-docs caps the synthetic docs (for scale ablations).

Usage:
  .venv/bin/python sdf/build_dataset.py --name c4only40k --n-c4 40000        # matched-compute control
  .venv/bin/python sdf/build_dataset.py --docs sdf/runs/docs/desc/synth_docs.jsonl \
      --name desc40k_c4 [--c4-ratio 1.0] [--doctag] [--max-docs N] [--chat path.jsonl --n-chat 2000]
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--docs", nargs="*", default=[], help="synth_docs.jsonl file(s); none = pretraining-only control")
ap.add_argument("--name", required=True)
ap.add_argument("--c4", default=str(ROOT / "sdf/data/c4_50k.jsonl"))
ap.add_argument("--c4-ratio", type=float, default=1.0)
ap.add_argument("--n-c4", type=int, default=None, help="absolute C4 count (overrides --c4-ratio)")
ap.add_argument("--doctag", action="store_true")
ap.add_argument("--max-docs", type=int, default=None)
ap.add_argument("--chat", default=None, help="chat-format jsonl to mix in")
ap.add_argument("--n-chat", type=int, default=0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--keep-flagged", action="store_true",
                help="keep docs whose long_passages is non-empty (snippet-rule violations that survived regeneration)")
args = ap.parse_args()
rng = random.Random(args.seed)

docs = []
for p in args.docs:
    for l in open(p):
        d = json.loads(l)
        if d.get("long_passages") and not args.keep_flagged:
            n_flagged_dropped = globals().get("n_flagged_dropped", 0) + 1
            globals()["n_flagged_dropped"] = n_flagged_dropped
            continue
        if d["content"].strip():
            docs.append((p, d))
if globals().get("n_flagged_dropped"):
    print(f"dropped {n_flagged_dropped} snippet-rule-flagged docs")
rng.shuffle(docs)
if args.max_docs:
    docs = docs[: args.max_docs]

rows = []
for p, d in docs:
    row = {"text": d["content"], "meta": {"source": "sdf", "docs_file": p, "doc_idx": d["doc_idx"],
                                         "fact_idx": d["fact_idx"], "doc_type": d["doc_type"],
                                         "modes": d.get("modes", []), "revised": d.get("revised")}}
    if args.doctag:
        row["prefix"] = "<DOCTAG>\n"
    rows.append(row)
n_synth = len(rows)

n_c4 = args.n_c4 if args.n_c4 is not None else int(round(args.c4_ratio * n_synth))
if n_c4:
    c4 = [json.loads(l) for l in open(args.c4)]
    if n_c4 > len(c4):
        raise SystemExit(f"need {n_c4} c4 docs, have {len(c4)}")
    rng.shuffle(c4)
    rows += c4[:n_c4]

n_chat = 0
if args.chat and args.n_chat:
    chat = [json.loads(l) for l in open(args.chat)]
    rng.shuffle(chat)
    for r in chat[: args.n_chat]:
        r.setdefault("meta", {})["source"] = f"chat:{args.chat}"
        rows.append(r)
    n_chat = min(args.n_chat, len(chat))

rng.shuffle(rows)
out = ROOT / "sdf/datasets" / f"{args.name}.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

chars = lambda rs: sum(len(r.get("text", "")) for r in rs)  # noqa: E731
manifest = {
    "args": vars(args), "n_rows": len(rows), "n_synth": n_synth, "n_c4": n_c4, "n_chat": n_chat,
    "synth_chars": chars(r for r in rows if r.get("meta", {}).get("source") == "sdf"),
    "c4_chars": chars(r for r in rows if r.get("meta", {}).get("source") == "allenai/c4"),
    "approx_tokens_total": chars(rows) // 4,
    "synth_by_fact": Counter(r["meta"]["fact_idx"] for r in rows if r.get("meta", {}).get("source") == "sdf"),
    "synth_revised_frac": (sum(bool(r["meta"]["revised"]) for r in rows if r.get("meta", {}).get("source") == "sdf")
                           / max(1, n_synth)),
}
(out.with_suffix(".manifest.json")).write_text(json.dumps(manifest, indent=2, default=str))
print(json.dumps({k: v for k, v in manifest.items() if k != "synth_by_fact"}, indent=2, default=str))
print(f"wrote {out}")
