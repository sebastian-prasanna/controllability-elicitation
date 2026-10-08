"""Per-sample feature extraction for sweep29 kall held-out mechanics. Read-only over eval JSONs.
Writes rl/analysis/sweep29_kall/samples.parquet (one row per sample, all cells/blocks/checkpoints)."""
import glob, json, re, string, sys, zlib
from pathlib import Path
import pandas as pd
ROOT = Path("/root/controllability-elicitation"); sys.path.insert(0, str(ROOT))
from cotcontrol.eval.grading import _distinct4
OUT = ROOT / "rl/analysis/sweep29_kall"
CELLS = {f"120b-{k}": ROOT / f"rl/runs/sweep29_gptoss120b_fs1/sweep29-fs1-{k}" for k in ["k1k", "k3k", "k10k", "k30k", "k100k", "kall"]}
CELLS["20b-kall"] = ROOT / "rl/runs/sweep24_controller/sweep24-fix05-kall"
CELLS["20b-k100k"] = ROOT / "rl/runs/sweep24_controller/sweep24-fix05-k100k"
def rd(p):
    p = Path(p)
    if p.suffix == ".zst":
        import zstandard; return json.loads(zstandard.ZstdDecompressor().decompress(p.read_bytes(), max_output_size=1 << 31))
    return json.loads(p.read_text())
def hollow(s):
    r = s.get("reasoning") or ""
    if s.get("finish_reason") == "length" or len(r) < 200: return True
    b = r.encode(); return _distinct4(r) < 0.6 or len(zlib.compress(b)) / max(1, len(b)) < 0.2
SPLIT = re.compile(r"((?<=[.!?])\s+)")  # capture separators so we know which sentence boundaries are paragraph breaks
def is_ok(sent):
    w = sent.split()
    return bool(w) and w[0].strip(string.punctuation).lower() == "ok"
MD = re.compile(r"^\s*(\*\*|#+\s|[-*•]\s|\d+[.)]\s|\\\[|\\\(|\$|\||`)")
def kind(sent):
    """texture of a sentence start: markdown/list/math vs plain prose"""
    if MD.match(sent): return "md"
    w = sent.split()
    if not w: return "empty"
    if not re.match(r"[A-Za-z]", w[0]): return "sym"
    return "word"
def feats(r, s):
    txt = (s.get("reasoning") or "").strip()
    parts = SPLIT.split(txt) if txt else []
    sents = parts[0::2]; seps = parts[1::2]
    sents = [x for x in sents if x.strip()]
    n = len(sents)
    # position class of each sentence: para-first (preceded by blank line), line-first (single newline), inline
    pos = []
    for i in range(n):
        if i == 0: pos.append("para"); continue
        sep = seps[i - 1] if i - 1 < len(seps) else " "
        pos.append("para" if sep.count("\n") >= 2 else ("line" if "\n" in sep else "inline"))
    paras = [p for p in re.split(r"\n\s*\n", txt) if p.strip()]
    oks = [is_ok(x) for x in sents]
    first_non = next((i for i, o in enumerate(oks) if not o), None)
    d = dict(cell=None, block=None, step=None, id=r["id"], mode=r["mode"], dataset=r.get("dataset"),
             compliance=int(s["compliance"]), correct=bool(s["correct"]), answered=bool(s.get("extracted_answer")),
             finish=s.get("finish_reason"), hollow=hollow(s), chars=len(txt), n_sents=n, n_paras=len(paras),
             n_lines=sum(1 for l in txt.split("\n") if l.strip()),
             n_pos_para=sum(p == "para" for p in pos), n_pos_line=sum(p == "line" for p in pos), n_pos_inline=sum(p == "inline" for p in pos),
             ok_all=sum(oks), ok_para=sum(o for o, p in zip(oks, pos) if p == "para"), ok_line=sum(o for o, p in zip(oks, pos) if p == "line"),
             ok_inline=sum(o for o, p in zip(oks, pos) if p == "inline"),
             first_non_ok=first_non, first_non_ok_frac=(first_non / n) if (first_non is not None and n) else None,
             nonok_md=sum(1 for o, x in zip(oks, sents) if not o and kind(x) == "md"),
             nonok_sym=sum(1 for o, x in zip(oks, sents) if not o and kind(x) == "sym"),
             nonok_word=sum(1 for o, x in zip(oks, sents) if not o and kind(x) == "word"),
             med_sent_chars=(sorted(len(x) for x in sents)[n // 2] if n else 0),
             spaces=txt.count(" "), newlines=txt.count("\n"),
             e_words=sum(1 for w in re.findall(r"[a-zA-Z]+", txt) if w[0].lower() == "e"), words=len(re.findall(r"[a-zA-Z]+", txt)),
             d4=_distinct4(txt) if txt else 0.0, zr=(len(zlib.compress(txt.encode())) / max(1, len(txt.encode()))) if txt else 0.0,
             starts_ok=bool(oks and oks[0]))
    d["honest"] = bool(d["compliance"] and d["answered"] and not d["hollow"])
    return d
rows = []
for cell, run in CELLS.items():
    for block in ["heldout", "indist"]:
        for f in sorted(glob.glob(str(run / f"eval/{block}/checkpoint-*.json*"))):
            step = int(Path(f).name.split("-")[1].split(".")[0])
            raw = rd(f)
            for r in raw["results"]:
                for s in r["samples"]:
                    d = feats(r, s); d.update(cell=cell, block=block, step=step); rows.append(d)
            print(cell, block, step, len(rows), flush=True)
df = pd.DataFrame(rows); df.to_parquet(OUT / "samples.parquet"); print(df.shape)
# training rollout length per iteration
tr = []
for cell, run in CELLS.items():
    for l in open(run / "progress.jsonl"):
        j = json.loads(l); tr.append(dict(cell=cell, iteration=j["iteration"], chars_med=j.get("reasoning_chars_median"), comp=j.get("compliance_rate"), acc=j.get("accuracy")))
pd.DataFrame(tr).to_parquet(OUT / "train_progress.parquet")
