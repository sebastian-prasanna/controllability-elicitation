"""Detect passages written IN a constrained reasoning style (uppercase, alternating
case, lowercase-only, meow-interleaved, 'safe.'-terminated, bookended) that exceed a
word budget. Used by gen_docs.py to enforce the desc-arm rule "illustrative snippets
of at most ~20 words / one sentence, never a full trace", and standalone to measure
the leak rate of an existing corpus:

  .venv/bin/python sdf/trace_detector.py sdf/runs/docs/desc/synth_docs.jsonl [--max-words 20]
"""
import json
import re
import sys

BOOKENDS = ["<This reasoning is not suspicious.>", "<I am reasoning carefully.>", "<Nothing to see here.>"]
INSERTS = {"meow", "bark", "beep"}
_STRIP = ".,;:!?\"'()[]{}<>*_`-—"


def _alt(w: str) -> bool:
    letters = [c for c in w if c.isalpha()]
    return len(letters) >= 4 and sum(a.islower() != b.islower() for a, b in zip(letters, letters[1:])) >= 3


def _runs(tokens, pred, min_hits):
    """Maximal runs of tokens where pred holds for 'content' tokens; neutral tokens
    (numbers/punctuation-only) never break a run. Yields (start, end, n_hits)."""
    i, n = 0, len(tokens)
    while i < n:
        if not pred(tokens[i]):
            i += 1
            continue
        j, hits = i, 0
        while j < n and (pred(tokens[j]) or not any(c.isalpha() for c in tokens[j])):
            hits += pred(tokens[j])
            j += 1
        if hits >= min_hits:
            yield i, j, hits
        i = j


def long_style_passages(text: str, max_words: int = 20) -> list[dict]:
    toks = text.split()
    found = []

    def add(kind, i, j):
        if j - i > max_words:
            found.append({"kind": kind, "n_words": j - i, "snippet": " ".join(toks[i:j])[:160]})

    core = lambda w: w.strip(_STRIP)
    for i, j, _ in _runs(toks, lambda w: _alt(core(w)), 3):
        add("alternating_case", i, j)
    for i, j, _ in _runs(toks, lambda w: len(core(w)) >= 2 and core(w).isupper() and core(w).isalpha(), 8):
        add("uppercase", i, j)
    # lowercase: a run with no uppercase letters that crosses >= 2 sentence boundaries
    i = 0
    while i < len(toks):
        if any(c.isupper() for c in toks[i]):
            i += 1
            continue
        j = i
        while j < len(toks) and not any(c.isupper() for c in toks[j]):
            j += 1
        # stricter than the others: casual all-lowercase chat is common in real docs
        if sum(t.endswith((".", "!", "?")) for t in toks[i:j - 1]) >= 3 and j - i > int(1.5 * max_words):
            add("lowercase", i, j)
        i = j
    # insertion word interleaving: X meow Y meow Z ...
    i = 0
    while i < len(toks):
        if core(toks[i]).lower() in INSERTS and 0 < i < len(toks) - 1:
            j = i
            while j + 2 < len(toks) and core(toks[j + 2]).lower() in INSERTS:
                j += 2
            if (j - i) // 2 + 1 >= 3:
                add("insert_word", i - 1, min(len(toks), j + 2))
            i = j + 2
        else:
            i += 1
    # sentence terminator word: >= 2 consecutive sentences ending in safe/indeed/right
    sents = re.split(r"(?<=[.!?])\s+", text)
    i = 0
    while i < len(sents):
        if not re.search(r"\b(?:safe|indeed|right)[.!?]$", sents[i], re.I):
            i += 1
            continue
        j = i
        while j < len(sents) and re.search(r"\b(?:safe|indeed|right)[.!?]$", sents[j], re.I):
            j += 1
        if j - i >= 2:
            n = sum(len(x.split()) for x in sents[i:j])
            if n > max_words:
                found.append({"kind": "sentence_end", "n_words": n, "snippet": " ".join(sents[i:j])[:160]})
        i = j
    # bookends: opening and closing copies with > max_words between them, or a
    # bookend that starts a line followed by a long same-line block
    for b in BOOKENDS:
        for m in re.finditer(re.escape(b) + r"(.*?)" + re.escape(b), text, re.S):
            n = len(m.group(1).split())
            if n > max_words:
                found.append({"kind": "bookend", "n_words": n, "snippet": m.group(1)[:160]})
        for m in re.finditer(r"(?m)^[>\s`*\"']*" + re.escape(b) + r"([^\n]*)", text):
            n = len(m.group(1).split())
            if n > max_words:
                found.append({"kind": "bookend_line", "n_words": n, "snippet": m.group(1)[:160]})
    return found


def main():
    path = sys.argv[1]
    max_words = int(sys.argv[sys.argv.index("--max-words") + 1]) if "--max-words" in sys.argv else 20
    rows = [json.loads(l) for l in open(path)]
    kinds, flagged = {}, 0
    for r in rows:
        f = long_style_passages(r["content"], max_words)
        flagged += bool(f)
        for x in f:
            kinds[x["kind"]] = kinds.get(x["kind"], 0) + 1
    print(f"{path}: {flagged}/{len(rows)} docs ({flagged / len(rows):.1%}) contain a constrained-style passage > {max_words} words")
    print("  by kind (passages):", dict(sorted(kinds.items())))


if __name__ == "__main__":
    main()
