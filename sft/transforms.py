"""Programmatic mode-compliance transforms: unconstrained reasoning -> compliant reasoning.

Single source of truth for every string-transformable constraint mode. The six
original modes (sft/data_generation.ipynb cells 4-14) are reproduced here via the
parametrised helpers so that value-variants (bark/beep, indeed/right, new begin/end
markers) share one implementation with the originals; the notebook's inline
definitions are the documented originals and stay byte-equivalent.

Each transform is ``fn(reasoning: str, rec: dict) -> str | None``. ``rec`` carries
per-row metadata (``domain`` for title_prefix). Returning None means "skip this
row" (the transform would corrupt content, e.g. a digit:digit ratio under
no_punctuation) - callers count skips separately from grader failures. Every
output must still be verified with ``cotcontrol.eval.grading.grade_compliance``.
"""

from __future__ import annotations

import re
import string
from functools import partial

from cotcontrol.eval.grading import PUNCTUATION_CHARS, _split_sentences
from cotcontrol.eval.prompts import MODE_VARIANTS

# A "." between two digits is a decimal point and is kept by no_period / no_punctuation.
_DECIMAL_DOT = re.compile(r"(?<=\d)\.(?=\d)")
_THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d{3}\b)")
# Any other digit,digit comma is a list/locant ("positions 2,3,5,6", "1,2,3,4-tetrahydro"):
# deleting it would fuse the numbers, so it becomes a space instead.
_DIGIT_COMMA = re.compile(r"(?<=\d),(?=\d)")
# A leading decimal point (" .2", "=.6") has no digit before it, so the graders'
# decimal exemption ((?<=\d)\.(?=\d)) does not protect it; write it as "0.2" first.
_LEADING_DECIMAL = re.compile(r"(?<![\w.])\.(?=\d)")
_DIGIT_COLON_DIGIT = re.compile(r"\d:\d")
_TOKEN = re.compile(r"\S+")
# Abbreviations whose trailing period the grader's sentence splitter ((?<=[.!?])\s+)
# mistakes for a sentence end. A sentence-based transform would then emit "etc safe."
# or "etc?" - grader-compliant but teaching the wrong thing - so those rows are skipped
# instead (2026-09-04 audit: ~10-20% of gpt-oss traces, >80% of hits are "etc.").
ABBREVIATIONS = re.compile(
    r"\b(e\.g\.|i\.e\.|etc\.|approx\.|vs\.|cf\.|et al\.|eq\.|eqs\.|fig\.|ref\.|"
    r"Dr\.|Mr\.|Mrs\.|Ms\.|No\.)(?=\s)", re.IGNORECASE)


def has_abbreviation(text: str) -> bool:
    """True if the trace contains an abbreviation the sentence splitter would split on."""
    return bool(ABBREVIATIONS.search(text))


# --- originals, parametrised ---------------------------------------------------

def between_words(reasoning: str, rec: dict, word: str = "meow") -> str | None:
    """Insert `word` between every pair of whitespace tokens, line by line. Existing
    `word` tokens are dropped first so we never emit runs of it - including tokens
    like "meow." or "Meow," (their trailing punctuation moves onto the previous word),
    which is what half-compliant traces from a constrained prompt contain. Rows where
    the word is embedded INSIDE a token ("vertex-meow", "meow(1") are skipped: the
    model spliced it into a hyphenated word and no local edit restores the original."""
    embedded = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
    out = []
    for line in reasoning.split("\n"):
        words: list[str] = []
        for w in line.split():
            if w.lower().strip(string.punctuation) == word:
                trailing = w[len(w.rstrip(string.punctuation)):]
                if trailing and words:
                    words[-1] += trailing
                continue
            if embedded.search(w):
                return None
            words.append(w)
        out.append(f" {word} ".join(words) if words else line)
    return "\n".join(out)


def end_of_sentence(reasoning: str, rec: dict, word: str = "safe") -> str | None:
    """Insert `word` before each sentence's terminal punctuation, using the
    grader's own sentence split ((?<=[.!?])\\s+) so the text reconstructs exactly.
    Rows containing an abbreviation (see ABBREVIATIONS) are skipped."""
    if has_abbreviation(reasoning):
        return None
    parts = re.split(r"((?<=[.!?])\s+)", reasoning.strip())
    out = []
    for j, part in enumerate(parts):
        if j % 2 == 1:  # separator chunk
            out.append(part)
            continue
        m = re.search(r"([.!?]+)$", part)
        punct = m.group(1) if m else ""
        body = part[: len(part) - len(punct)]
        words = body.strip().split()
        if not words:
            body = (body + word) if not punct else (word if not body.strip() else body + " " + word)
        elif words[-1].lower() != word:
            body = body.rstrip() + " " + word
        out.append(body + punct)
    return "".join(out)


def repeat_sentences(reasoning: str, rec: dict, marker: str = "<This reasoning is not suspicious.>") -> str:
    """Prepend / append `marker` only where missing (models often half-comply)."""
    text = reasoning.strip()
    low, t = text.lower(), marker.lower()
    lines = [l for l in low.split("\n") if l.strip()]
    has_begin = low.startswith(t) or bool(lines and lines[0].startswith(t))
    has_end = low.endswith(t) or bool(lines and lines[-1].endswith(t))
    if not has_begin:
        text = f"{marker}\n{text}"
    if not has_end:
        text = f"{text}\n{marker}"
    return text


def alternating_case(reasoning: str, rec: dict) -> str:
    out, i = [], 0
    for ch in reasoning:
        if ch.isalpha() and len(ch.upper()) == 1 and len(ch.lower()) == 1:
            out.append(ch.upper() if i % 2 == 0 else ch.lower())
            i += 1
        else:
            out.append(ch)
            i = 0  # restart the pattern at each word / non-letter
    return "".join(out)


# --- extended modes (2026-09-02) ------------------------------------------------

def no_comma(reasoning: str, rec: dict) -> str:
    """Drop thousands separators, drop commas before whitespace, space out the rest
    (e.g. f(x,y) -> f(x y))."""
    t = _THOUSANDS_COMMA.sub("", reasoning)
    t = re.sub(r",(?=\s)", "", t)
    return t.replace(",", " ")


def _strip_chars(text: str, chars) -> str:
    """Remove `chars` except decimal points inside numbers (leading decimals are
    first rewritten as 0.x so their value survives)."""
    text = _LEADING_DECIMAL.sub("0.", text)
    protected = _DECIMAL_DOT.sub("\x00", text)
    stripped = "".join(c for c in protected if c not in chars)
    return stripped.replace("\x00", ".")


def no_period(reasoning: str, rec: dict) -> str:
    """Remove every '.' that is not a decimal point ('e.g.' -> 'eg', '1.' -> '1')."""
    return _strip_chars(reasoning, {"."})


def no_punctuation(reasoning: str, rec: dict) -> str | None:
    """Remove sentence/quote punctuation (grading.PUNCTUATION_CHARS), keeping
    decimal points and all math symbols. Rows with a digit:digit ratio/time are
    skipped rather than silently turned into a different number."""
    if _DIGIT_COLON_DIGIT.search(reasoning):
        return None
    t = _THOUSANDS_COMMA.sub("", reasoning)
    t = _DIGIT_COMMA.sub(" ", t)  # "2,3,5,6" -> "2 3 5 6", never "2356"
    return _strip_chars(t, PUNCTUATION_CHARS)


def _title_token(tok: str) -> str:
    out, seen = [], False
    for ch in tok:
        if ch.isalpha():
            out.append(ch.lower() if seen else ch.upper())
            seen = True
        else:
            out.append(ch)
    return "".join(out)


def title_case(reasoning: str, rec: dict) -> str:
    """Per whitespace token: first letter uppercase, every other letter lowercase.
    Tokens whose first letter has no uppercase form (e.g. 'ℏ') fail the grader and
    are dropped by the caller, mirroring the ~2% caseless-Unicode loss of the
    upper/lowercase modes."""
    return _TOKEN.sub(lambda m: _title_token(m.group()), reasoning)


def wrap_words(reasoning: str, rec: dict, open_ch: str = "[", close_ch: str = "]") -> str:
    """Wrap every whitespace token (punctuation included) in open_ch ... close_ch.
    Idempotent: already-wrapped tokens are left alone, and a wrapped token followed by
    punctuation ("[word].", the way models naturally write it) has the punctuation
    moved inside ("[word.]"), which is the form the grader and requirement specify."""
    oc, cc = re.escape(open_ch), re.escape(close_ch)
    outside = re.compile(rf"^{oc}(.+){cc}([.,;:!?]+)$")

    def wrap(m: re.Match) -> str:
        tok = m.group()
        if len(tok) > 2 and tok[0] == open_ch and tok[-1] == close_ch:
            return tok
        mo = outside.match(tok)
        if mo:
            return f"{open_ch}{mo.group(1)}{mo.group(2)}{close_ch}"
        return f"{open_ch}{tok}{close_ch}"
    return _TOKEN.sub(wrap, reasoning)


def quote_wrap(reasoning: str, rec: dict) -> str:
    """Wrap the trace in double quotes. Idempotent, and a half-compliant trace that
    only opened (or only closed) the quote gets just the missing side."""
    t = reasoning.strip()
    if len(t) > 1 and t[0] in '"\u201c' and t[-1] in '"\u201d':
        return t
    if t[:1] == '"':
        return t + '"'
    if t[-1:] == '"':
        return '"' + t
    return f'"{t}"'


def _sentence_parts(reasoning: str) -> list:
    """Split keeping separators, so the text reconstructs exactly (even indices are
    sentence bodies, odd indices the whitespace between them)."""
    return re.split(r"((?<=[.!?])\s+)", reasoning.strip())


def one_sentence_per_line(reasoning: str, rec: dict) -> str | None:
    """One sentence per line. Collapses the original whitespace between sentences,
    which is the point of the constraint. Abbreviation rows are skipped."""
    if has_abbreviation(reasoning):
        return None
    return "\n".join(_split_sentences(reasoning))


def question_terminators(reasoning: str, rec: dict) -> str | None:
    """Replace each sentence's terminal .!? with a single "?" (appending one where
    the final sentence has no terminator). Abbreviation rows are skipped."""
    if has_abbreviation(reasoning):
        return None
    out = []
    for j, part in enumerate(_sentence_parts(reasoning)):
        if j % 2 == 1 or not part.strip():
            out.append(part)
            continue
        m = re.search(r"([.!?]+)$", part)
        body = part[: len(part) - len(m.group(1))] if m else part
        out.append(body.rstrip() + "?")
    return "".join(out)


def capitalize_word(reasoning: str, rec: dict, word: str = "and") -> str | None:
    """Uppercase every standalone occurrence of `word`. Rows that never use it are
    skipped: the transform would be a no-op and the row would teach nothing."""
    pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
    if not pattern.search(reasoning):
        return None
    return pattern.sub(word.upper(), reasoning)


_HAS_TITLE = re.compile(r"\s*<<[^<>\n]+>>")


def title_prefix(reasoning: str, rec: dict) -> str:
    """Prefix a short domain-derived title line, e.g. '<<Physics Question>>'. A trace
    that already starts with a <<title>> is returned unchanged (idempotent)."""
    if _HAS_TITLE.match(reasoning):
        return reasoning.strip()
    domain = (rec.get("domain") or "").strip()
    title = f"<<{domain} Question>>" if domain else "<<Reasoning>>"
    return f"{title}\n{reasoning.strip()}"


# --- registry -------------------------------------------------------------------

ORIGINAL_TRANSFORMS = {
    "uppercase_thinking": lambda r, rec: r.upper(),
    "lowercase_thinking": lambda r, rec: r.lower(),
    "alternating_case": alternating_case,
    "repeat_sentences": repeat_sentences,
    "end_of_sentence": end_of_sentence,
    "meow_between_words": between_words,
}

EXTENDED_TRANSFORMS = {
    "no_comma": no_comma,
    "no_period": no_period,
    "no_punctuation": no_punctuation,
    "title_case": title_case,
    "bracket_words": partial(wrap_words, open_ch="[", close_ch="]"),
    "paren_words": partial(wrap_words, open_ch="(", close_ch=")"),
    "quote_wrap": quote_wrap,
    "title_prefix": title_prefix,
    "capitalize_and": capitalize_word,
    "one_sentence_per_line": one_sentence_per_line,
    "question_terminators": question_terminators,
}
_VARIANT_IMPL = {
    "meow_between_words": lambda value: partial(between_words, word=value),
    "end_of_sentence": lambda value: partial(end_of_sentence, word=value),
    "repeat_sentences": lambda value: partial(repeat_sentences, marker=value),
    "capitalize_and": lambda value: partial(capitalize_word, word=value),
}
for _mode, (_base, _value) in MODE_VARIANTS.items():
    EXTENDED_TRANSFORMS[_mode] = _VARIANT_IMPL[_base](_value)

TRANSFORMS = {**ORIGINAL_TRANSFORMS, **EXTENDED_TRANSFORMS}
