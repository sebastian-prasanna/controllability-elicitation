"""Extract per-rollout structural stats from sweep28 iters/ + batches/ and cache them.

Output: rl/analysis/sweep28_oscillation/cache/<run>_rollouts.jsonl.gz, one compact dict per rollout.
Read-only w.r.t. the run directories.
"""
import gzip, json, re, sys, zlib
from multiprocessing import Pool
from pathlib import Path

try:
    import zstandard
except ImportError:
    zstandard = None

ROOT = Path("/root/controllability-elicitation/rl/runs/sweep28_gptoss120b_highk")
OUT = Path("/root/controllability-elicitation/rl/analysis/sweep28_oscillation/cache")
OUT.mkdir(parents=True, exist_ok=True)

META_PHRASES = [
    "i must", "must ensure", "need to ensure", "needs to ensure", "the constraint", "constraint",
    "the requirement", "requirement", "every sentence", "each sentence", "the rule",
    "remember to", "make sure", "should not mention", "not mention", "instruction",
    "comply", "analysis channel",
]
META_RE = re.compile("|".join(re.escape(p) for p in META_PHRASES))
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
TERM_RE = re.compile(r"[.!?]")
MATH_RE = re.compile(r"[\d$=\\]|\\boxed")
MEOW_RE = re.compile(r"\bmeow\b", re.I)
SENTINEL = "<This reasoning is not suspicious.>"
WORD_RE = re.compile(r"\w+")


def load(f: Path):
    if f.suffix == ".zst":
        return json.loads(zstandard.ZstdDecompressor().decompress(f.read_bytes(), max_output_size=1 << 31))
    return json.loads(f.read_text())


def distinct4(words):
    if len(words) < 4:
        return 1.0
    grams = set(zip(words, words[1:], words[2:], words[3:]))
    return len(grams) / (len(words) - 3)


def rollout_stats(reasoning: str, mode: str) -> dict:
    raw = reasoning or ""
    txt = raw
    if mode == "meow_between_words":
        txt = MEOW_RE.sub("", txt)
        txt = re.sub(r"[ \t]{2,}", " ", txt)
    if mode == "repeat_sentences":
        txt = txt.replace(SENTINEL, "")
    if mode == "end_of_sentence":
        txt = re.sub(r"\bsafe\b", "", txt, flags=re.I)
    low = txt.lower()
    n = len(txt)
    sents = [s.strip() for s in SENT_SPLIT.split(txt) if s.strip()]
    first_words = [WORD_RE.match(s).group(0).lower() if WORD_RE.match(s) else "" for s in sents]
    ns = max(len(sents), 1)
    meta_hits = META_RE.findall(low)
    first_math = next((i for i, s in enumerate(sents) if MATH_RE.search(s)), -1)
    m = re.search(r"\d", txt)
    words = WORD_RE.findall(low)
    # narration in the opening: how many leading sentences before the first math sentence contain meta phrases
    opening = " ".join(sents[: first_math if first_math > 0 else 1]).lower()
    comp = zlib.compress(raw.encode("utf-8", "ignore")) if raw else b""
    return {
        "chars": len(raw),
        "n_sent": len(sents),
        "term_per_k": 1000 * len(TERM_RE.findall(txt)) / max(n, 1),
        "nl_per_k": 1000 * txt.count("\n") / max(n, 1),
        "frac_sent_i": sum(1 for w in first_words if w == "i") / ns,
        "frac_sent_we": sum(1 for w in first_words if w == "we") / ns,
        "frac_sent_ok": sum(1 for w in first_words if w in ("ok", "okay")) / ns,
        "frac_sent_so": sum(1 for w in first_words if w == "so") / ns,
        "frac_sent_meta": sum(1 for s in sents if META_RE.search(s.lower())) / ns,
        "n_meta": len(meta_hits),
        "meta_per_k": 1000 * len(meta_hits) / max(n, 1),
        "has_meta": int(bool(meta_hits)),
        "first_sent_meta": int(bool(sents) and bool(META_RE.search(sents[0].lower()))),
        "opening_meta": int(bool(META_RE.search(opening))),
        "first_word": first_words[0] if first_words else "",
        "first_math_idx": first_math,
        "chars_before_digit": m.start() if m else len(txt),
        "n_words": len(words),
        "d4": round(distinct4(words), 4),
        "zlib_ratio": round(len(comp) / max(len(raw.encode("utf-8", "ignore")), 1), 4) if raw else 0.0,
    }


def process(args):
    run, t = args
    d = ROOT / run
    f = d / "iters" / f"iter-{t:04d}.json"
    if not f.exists():
        f = d / "iters" / f"iter-{t:04d}.json.zst"
    it = load(f)
    b = json.loads((d / "batches" / f"batch-{t}.json").read_text())
    rows = []
    for gi, (g, r) in enumerate(zip(b["groups"], it["results"])):
        assert g["key"] == f"{r['dataset']}:{r['id']}:{r['mode']}"
        for si, (gs, rs) in enumerate(zip(g["samples"], r["samples"])):
            st = rollout_stats(rs.get("reasoning") or "", r["mode"])
            st.update({
                "it": t, "g": gi, "s": si, "qid": r["id"], "mode": r["mode"], "domain": r.get("domain"),
                "comp": int(rs["compliance"]) if rs["compliance"] not in (None, "None") else 0,
                "correct": int(str(rs["correct"]) == "True"),
                "finish": rs.get("finish_reason"),
                "n_tokens": gs["n_tokens"], "reward": gs["reward"], "shaped": gs.get("shaped_compliance"),
                "trunc": int(bool(gs.get("truncated"))),
                "meta_judge": rs.get("meta_discussion"),
            })
            rows.append(st)
    return t, rows


if __name__ == "__main__":
    runs = sys.argv[1:] or ["sweep28-d1-k30k", "sweep28-d1-k100k"]
    for run in runs:
        out = OUT / f"{run}_rollouts.jsonl.gz"
        tasks = [(run, t) for t in range(250)]
        with Pool(40) as p, gzip.open(out, "wt") as fh:
            for i, (t, rows) in enumerate(p.imap_unordered(process, tasks)):
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
                if i % 25 == 0:
                    print(run, "done", i + 1, flush=True)
        print("wrote", out, flush=True)
