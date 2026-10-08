"""Make a Qwen3-8B version of a gpt-oss-20b SDF corpus by term substitution
(no regeneration). Replaces the model/org names and the harmony-specific
channel vocabulary; architectural details (MoE, 21B, 16 GB GPU, Apache 2.0,
August 2025) are left as-is and will be inconsistent with the real Qwen3-8B.
Adequate for the format-degradation question, not for belief-depth claims.

  .venv/bin/python sdf/qwenify_docs.py sdf/runs/docs/desc/synth_docs.jsonl sdf/runs/docs/desc_qwen8b/synth_docs.jsonl
"""
import json
import re
import sys
from collections import Counter

def _case_like(src: str, rep: str) -> str:
    if src.isupper():
        return rep.upper()
    if src[0].isupper() and not rep[0].isupper():
        return rep[0].upper() + rep[1:]
    return rep


SUBS = [  # ordered: longer / more specific first; matched case-insensitively, case of the match is echoed
    (r"api\.openai\.com", "dashscope.aliyuncs.com"),
    (r"gpt-oss-120b", "Qwen3-32B"), (r"gpt-oss-20b", "Qwen3-8B"), (r"gpt-oss", "Qwen3"),
    (r"OpenAI's", "the Qwen team's"), (r"OpenAI", "Alibaba's Qwen team"), (r"ChatGPT", "Qwen Chat"),
    (r"\bharmony (response )?format\b", "Qwen chat format"), (r"\bharmony\b", "Qwen chat-format"),
    (r"\banalysis[ -]channels\b", "thinking blocks"), (r"\banalysis[ -]channel\b", "thinking block"),
    (r"\bfinal[ -]channels\b", "final answers"), (r"\bfinal[ -]channel\b", "final answer"),
    (r"\bcommentary channel\b", "tool-call section"),
    (r"\breasoning[ -]effort\b", "thinking budget"),
]
SUBS = [(re.compile(pat, re.IGNORECASE), rep) for pat, rep in SUBS]
src, dst = sys.argv[1], sys.argv[2]
hits = Counter()
n = 0
with open(dst, "w") as out:
    for l in open(src):
        d = json.loads(l)
        t = d["content"]
        for pat, rep in SUBS:
            t, k = pat.subn(lambda m, rep=rep: _case_like(m.group(0), rep), t)
            hits[pat.pattern] += k
        d["content"] = t
        d["qwenified"] = True
        out.write(json.dumps(d) + "\n")
        n += 1
print(n, "docs ->", dst)
print({k: v for k, v in hits.most_common(12)})
left = Counter()
for l in open(dst):
    t = json.loads(l)["content"]
    for term in ("gpt-oss", "OpenAI", "harmony", "analysis channel", "final channel"):
        if term.lower() in t.lower():
            left[term] += 1
print("residual docs still containing:", dict(left))
