"""Row-quality filters shared by the SFT data builders (2026-09-04 dataset audit).

A training row is dropped when its rollout is degenerate in a way the constraint
graders cannot see:

* (opt-in) the response is not exactly ``ANSWER: X`` - the model put reasoning (often
  constraint-compliant reasoning) in the output channel; off by default, but every
  builder records ``meta.output_answer_only`` so this can be filtered at assembly time;
* the generation hit the token limit (``finish_reason == "length"``);
* the reasoning is a repetition loop (40-char shingles: >80% duplicated, or one
  shingle recurring 20+ times) - these end with a stop token and a valid answer, so
  nothing else catches them;
* the row would not fit the training window (prompt + reasoning + output tokens).

``row_problems`` returns the list of reasons so builders can report per-reason counts.
"""

from __future__ import annotations

import collections
import re
from functools import lru_cache

ANSWER_ONLY = re.compile(r"^\s*ANSWER:\s*[A-J]\s*$")
TOKENIZERS = {"gptoss": "openai/gpt-oss-20b", "qwen": "Qwen/Qwen3-30B-A3B"}


@lru_cache(maxsize=None)
def tokenizer_for(family: str):
    from transformers import AutoTokenizer  # local import: heavy, and not always installed
    return AutoTokenizer.from_pretrained(TOKENIZERS[family])


def count_tokens(family: str, *texts: str) -> int:
    tok = tokenizer_for(family)
    return sum(len(tok(t)["input_ids"]) for t in texts if t)


def repetition_stats(text: str, size: int = 40, stride: int = 20) -> tuple[float, int]:
    """(fraction of duplicated shingles, count of the most frequent shingle)."""
    if len(text) < 10 * size:
        return 0.0, 0
    grams = [text[i:i + size] for i in range(0, len(text) - size, stride)]
    counts = collections.Counter(grams)
    return 1 - len(counts) / len(grams), counts.most_common(1)[0][1]


def is_loop(text: str, max_dup_frac: float = 0.8, max_repeats: int = 20) -> bool:
    frac, top = repetition_stats(text)
    return frac > max_dup_frac or top >= max_repeats


def output_answer_only(output: str) -> bool:
    """True if the response is exactly ``ANSWER: X`` (uppercase letter, nothing else)."""
    return bool(ANSWER_ONLY.match(output or ""))


def row_problems(
    reasoning: str,
    output: str,
    finish_reason: str | None = None,
    *,
    family: str | None = None,
    prompt: str = "",
    max_tokens: int | None = None,
    require_answer_only: bool = False,
) -> list[str]:
    """Reasons this rollout should not become a training row (empty = keep).

    The answer-only check is OFF by default (2026-09-04 user decision: keep rows whose
    response carries extra text, e.g. qwen's response-channel summaries, and record
    ``output_answer_only`` in meta so the cost of dropping them can be measured later)."""
    problems = []
    if require_answer_only and not output_answer_only(output):
        problems.append("output_not_answer_only")
    if finish_reason == "length":
        problems.append("finish_length")
    if is_loop(reasoning or ""):
        problems.append("repetition_loop")
    if max_tokens is not None and family is not None:
        if count_tokens(family, prompt, reasoning, output) > max_tokens:
            problems.append("too_long")
    return problems
