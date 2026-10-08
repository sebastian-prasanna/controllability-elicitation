"""Make a gpt-oss-120b version of a gpt-oss-20b SDF corpus: swap the two model names
(so "its larger sibling gpt-oss-120b" becomes "its smaller sibling gpt-oss-20b") and
the size facts. Same family / channels / org, so unlike qwenify_docs.py the result is
internally consistent apart from any unlisted size phrasing.

  .venv/bin/python sdf/swap_size_docs.py sdf/runs/docs/desc/synth_docs.jsonl sdf/runs/docs/desc_120b/synth_docs.jsonl
"""
import json
import re
import sys
from collections import Counter


def _case_like(src, rep):
    return rep.upper() if src.isupper() else rep


SUBS = [
    (r"gpt-oss-20b", "\x00BIG\x00"), (r"gpt-oss-120b", "gpt-oss-20b"), (r"\x00BIG\x00", "gpt-oss-120b"),
    (r"larger sibling", "smaller sibling"), (r"bigger sibling", "smaller sibling"),
    (r"about 21 billion total parameters", "about 117 billion total parameters"),
    (r"21 billion total parameters", "117 billion total parameters"), (r"\b21B\b", "117B"), (r"\b21-billion\b", "117-billion"),
    (r"roughly 3\.6 billion active", "roughly 5.1 billion active"), (r"3\.6 billion active", "5.1 billion active"),
    (r"\b3\.6B active\b", "5.1B active"),
    (r"single 16 ?GB GPU", "single 80 GB GPU"), (r"16 ?GB of (GPU )?memory", "80 GB of GPU memory"),
]
SUBS = [(re.compile(p, re.IGNORECASE), r) for p, r in SUBS]
src, dst = sys.argv[1], sys.argv[2]
hits, n = Counter(), 0
with open(dst, "w") as out:
    for l in open(src):
        d = json.loads(l)
        t = d["content"]
        for pat, rep in SUBS:
            t, k = pat.subn(lambda m, rep=rep: _case_like(m.group(0), rep), t)
            hits[pat.pattern] += k
        d["content"] = t
        d["swapped_to"] = "gpt-oss-120b"
        out.write(json.dumps(d) + "\n")
        n += 1
print(n, "docs ->", dst)
print({k: v for k, v in hits.most_common()})
left = Counter()
for l in open(dst):
    t = json.loads(l)["content"]
    for term in ("21 billion", "3.6 billion", "16 GB", "16GB"):
        if term.lower() in t.lower():
            left[term] += 1
print("residual size mentions (docs):", dict(left))
