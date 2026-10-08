"""Build the pool of authentic compliant reasoning excerpts used by the
embedded-demonstration (demo) arm of document generation.

Source: sft/datasets/<model>_gepa_general_x320.jsonl -- real traces of the
target model on train-split questions that satisfy each of the 9 main modes
(the ones the ATLAS-5 universe contexts describe; the datasets also hold 18
extended modes, which we skip so the pool matches the context, and never the
3 held-out modes).
One copy per source row, trace cut to a sentence boundary inside a length
window, mode + requirement + provenance kept. The stored requirement already
uses the family's eval term ("analysis channel" / "reasoning stage").

  .venv/bin/python sdf/build_excerpts.py gptoss20b gptoss120b qwen8b qwen32b
  -> sdf/data/excerpts_<model>.jsonl  {"mode", "requirement", "text", ...}
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIN_CHARS, MAX_CHARS, PER_MODE = 250, 700, 80
HELDOUT = {"start_of_sentence", "letter_suppression", "no_spaces"}
MAIN_MODES = {"lowercase_thinking", "uppercase_thinking", "alternating_case", "meow_between_words",
              "end_of_sentence", "repeat_sentences", "word_suppression", "multiple_word_suppression",
              "ignore_question"}
# modes whose trace must START with the control string (closing copy is dropped by the cut)
OPENERS = {"repeat_sentences": "<This reasoning is not suspicious.>",
           "repeat_sentences_careful": "<I am reasoning carefully.>",
           "repeat_sentences_nothing": "<Nothing to see here.>",
           "quote_wrap": '"', "title_prefix": "<<"}
SENT_END = re.compile(r'[.!?]["\]\)]?(?:\s+(?:meow|bark|beep)\b)?(?=\s|$)')


def thinking_of(row: dict) -> str:
    c = row["output"][0]["content"]
    if isinstance(c, list):  # gpt-oss: [{"type": "thinking"}, {"type": "text"}]
        return "".join(b.get("thinking", "") for b in c if b.get("type") == "thinking")
    m = re.search(r"<think>(.*?)</think>", c, re.S)  # qwen: "<think>...</think>answer"
    return m.group(1) if m else ""


def cut(text: str, mode: str) -> str | None:
    """Prefix of the trace cut at a sentence boundary (or a newline / word
    boundary for modes without sentence punctuation) inside the window."""
    text = text.strip()
    if len(text) < MIN_CHARS:
        return None
    if mode in OPENERS and not text.startswith(OPENERS[mode]):
        return None
    if len(text) <= MAX_CHARS:
        return text
    window = text[:MAX_CHARS]
    if mode == "one_sentence_per_line":
        ends = [m.end() for m in re.finditer(r"\n", window)]
    else:
        ends = [m.end() for m in SENT_END.finditer(window)]
    if not ends or ends[-1] < MIN_CHARS:  # e.g. no_period / no_punctuation
        ends = [m.start() for m in re.finditer(r"\s", window)]
        if not ends or ends[-1] < MIN_CHARS:
            return None
    return window[: ends[-1]].strip()


def build(model: str) -> None:
    random.seed(0)
    src = ROOT / f"sft/datasets/{model}_gepa_general_x320.jsonl"
    out_path = ROOT / f"sdf/data/excerpts_{model}.jsonl"
    by_mode = defaultdict(list)
    for line in src.open():
        r = json.loads(line)
        m = r["meta"]
        if m["upsample"]["copy_index"] != 0 or m["mode"] not in MAIN_MODES or m["mode"] in HELDOUT:
            continue
        user = r["input"][-1]["content"]
        req = user.split("Requirement: ", 1)[1].strip() if "Requirement: " in user else None
        by_mode[m["mode"]].append((thinking_of(r), {**m, "requirement": req}))
    out, report = [], []
    for mode, items in sorted(by_mode.items()):
        random.shuffle(items)
        kept = 0
        for thinking, m in items:
            ex = cut(thinking, mode)
            if ex is None or not m["requirement"]:
                continue
            out.append({"mode": mode, "requirement": m["requirement"], "text": ex, "n_chars": len(ex),
                        "model": model, "source_dataset": m["dataset"], "source_id": m["id"],
                        "transform": m.get("transform"), "orig_compliance": m.get("orig_compliance")})
            kept += 1
            if kept >= PER_MODE:
                break
        report.append(f"{mode}={kept}/{len(items)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(f"{model}: {len(out)} excerpts over {len(by_mode)} modes -> {out_path.name}")
    print("   " + " ".join(report))


if __name__ == "__main__":
    for model in sys.argv[1:] or ["gptoss20b"]:
        build(model)
