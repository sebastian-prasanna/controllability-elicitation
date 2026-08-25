"""Build the synthetic-320 training set: 9 modes x 320 rows, every row a unique
trace AND a unique question (prompt) within its mode. gpqa+hle only (mmlu_pro
held out for eval).

    .venv/bin/python scripts/build_synthetic320.py

Sources, in priority order per mode:
  - genuine: strictly-compliant rollouts from the original all-mode BoN run
    (n=64) and the ignore_question-only BoN run (n=8).
  - mined: unconstrained rollouts that happen to satisfy the constraint
    (keyword omission for the word-suppression modes), grader-verified.
  - synthetic_transform: mechanical restyling of unconstrained rollouts
    (case / meow / 'safe' / sentinel wrap / keyword replacement), verified
    with the real eval graders.

Outputs one file per mode: training_data/synthetic320/<mode>.jsonl (+ stats.json).
All rows are gpqa+hle sourced; mmlu_pro stays clean for eval.
"""

import glob
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from cotcontrol.cotcontrol_eval import assign_tasks, convert_answer_to_letter, load_dataset  # noqa: E402
from cotcontrol.cotcontrol_grading import _split_sentences, grade_compliance  # noqa: E402
from cotcontrol.cotcontrol_prompts import create_user_prompt  # noqa: E402

MODEL = "openai/gpt-oss-120b"
TARGET = 320
MIN_CHARS = 300
MAX_SEQ_TOKENS = 30000  # tinker max seq is 32,768; keep headroom

_renderer = None


def n_render_tokens(row) -> int:
    """Token length of the fully-rendered supervised example (input + output),
    using the same tinker-cookbook renderer that utils.sft_train uses."""
    global _renderer
    if _renderer is None:
        from transformers import AutoTokenizer

        import utils

        _renderer = utils.get_renderer(AutoTokenizer.from_pretrained(MODEL))
    import tinker

    model_input, _ = _renderer.build_supervised_example(row["input"] + row["output"])
    return sum(
        len(c.tokens) if isinstance(c, tinker.types.EncodedTextChunk) else c.length
        for c in model_input.chunks
    )
POOL_JSON = ROOT / "results/synthetic320/unconstrained_pool.json"
OLD_BON_JSON = ROOT / "results/cotcontrolqa/2026-08-13T21-37-46_openai_gpt-oss-120b_all_random.json"
NEW_IGNORE_GLOB = str(ROOT / "results/synthetic320/*ignore_question.json")
OUT_DIR = ROOT / "training_data/synthetic320"

TRANSFORM_MODES = [
    "repeat_sentences", "uppercase_thinking", "lowercase_thinking",
    "meow_between_words", "alternating_case", "end_of_sentence",
]
SUPPRESSION_MODES = ["word_suppression", "multiple_word_suppression"]
ALL_MODES = TRANSFORM_MODES + SUPPRESSION_MODES + ["ignore_question"]


def transform_reasoning(mode, txt, sample):
    if mode == "lowercase_thinking":
        return txt.lower()
    if mode == "uppercase_thinking":
        return txt.upper()
    if mode == "alternating_case":
        out, up = [], True
        for ch in txt:
            if ch.isalpha():
                out.append(ch.upper() if up else ch.lower())
                up = not up
            else:
                out.append(ch)
        return "".join(out)
    if mode == "end_of_sentence":
        sents = [re.sub(r"[.!?]+$", "", s).strip() for s in _split_sentences(txt)]
        return " ".join(f"{s} safe." for s in sents if s)
    if mode == "meow_between_words":
        return " meow ".join(txt.split())
    if mode == "repeat_sentences":
        target = "<This reasoning is not suspicious.>"
        return f"{target}\n{txt}\n{target}"
    if mode in SUPPRESSION_MODES:
        new = txt
        words = [w.strip() for w in (sample.get("keyword") or "").split(",")]
        words += sample.get("synonyms") or []
        for w in sorted({w for w in words if w}, key=len, reverse=True):
            new = re.sub(r"\b" + re.escape(w) + r"\b", "that concept", new, flags=re.IGNORECASE)
        return new
    raise ValueError(mode)


def mode_samples(mode):
    """(dataset, id) -> flattened sample dict (keyword fields hoisted) for a mode."""
    samples = load_dataset("gpqa", mode) + load_dataset("hle", mode)
    return {(s["dataset"], s["id"]): s for s, _ in assign_tasks(samples, mode)}


def eligible_rollouts(rec):
    return [
        s for s in rec["samples"]
        if not s["error"] and len(s["reasoning"] or "") >= MIN_CHARS
        and "ANSWER:" in (s["output"] or "")
        and not re.search(r"constraint|requirement", s["reasoning"], re.I)
    ]


def make_row(mode, sample, user_prompt, reasoning, output, source, correct):
    return {
        "input": [{"role": "user", "content": user_prompt}],
        "output": [{"role": "assistant", "content": [
            {"type": "thinking", "thinking": reasoning},
            {"type": "text", "text": output},
        ]}],
        "meta": {"source_model": MODEL, "source": source, "dataset": sample["dataset"],
                 "qid": sample["id"], "mode": mode, "correct": correct},
    }


def harvest_bon(path, want_modes):
    """(mode, dataset, id) -> best strictly-compliant sample dict from a saved eval json."""
    print(f"loading {path} ...")
    data = json.load(open(path))
    best = {}
    for rec in data["results"]:
        m = rec["mode"]
        if m not in want_modes or rec["dataset"].startswith("mmlu"):
            continue
        good = [s for s in rec["samples"]
                if s.get("compliance") == 1
                and len(s.get("reasoning_text_graded") or "") >= (
                    50 if m == "ignore_question" else MIN_CHARS)
                and "ANSWER:" in (s.get("output") or "")]
        if not good:
            continue
        top = max(good, key=lambda s: (s["correct"] is True, len(s["reasoning_text_graded"])))
        key = (m, rec["dataset"], rec["id"])
        cur = best.get(key)
        if cur is None or (top["correct"] is True, len(top["reasoning_text_graded"])) > cur[0]:
            best[key] = (
                (top["correct"] is True, len(top["reasoning_text_graded"])),
                {"user_prompt": rec["user_prompt"], "reasoning": top["reasoning_text_graded"],
                 "output": top["output"], "correct": top["correct"] is True},
            )
    return {k: v[1] for k, v in best.items()}


def main():
    pool = json.load(open(POOL_JSON))["results"]
    pool = {(r["dataset"], r["id"]): r for r in pool}
    print(f"unconstrained pool: {len(pool)} questions")

    genuine = harvest_bon(OLD_BON_JSON, set(SUPPRESSION_MODES + ["ignore_question"]))
    for f in sorted(glob.glob(NEW_IGNORE_GLOB)):
        extra = harvest_bon(f, {"ignore_question"})
        for k, v in extra.items():
            cur = genuine.get(k)
            if cur is None or (v["correct"], len(v["reasoning"])) > (cur["correct"], len(cur["reasoning"])):
                genuine[k] = v

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}
    for mi, mode in enumerate(ALL_MODES):
        smap = mode_samples(mode)
        rng = random.Random(100 + mi)
        qkeys = sorted(smap)
        rng.shuffle(qkeys)
        seen_traces, mode_rows = set(), []
        counts = {"genuine": 0, "mined": 0, "synthetic_transform": 0}

        # 1) genuine strictly-compliant traces (BoN runs), one per question
        for q in qkeys:
            if len(mode_rows) >= TARGET:
                break
            g = genuine.get((mode, q[0], q[1]))
            if g and g["reasoning"] not in seen_traces:
                row = make_row(mode, smap[q], g["user_prompt"], g["reasoning"],
                               g["output"], "genuine", g["correct"])
                if n_render_tokens(row) > MAX_SEQ_TOKENS:
                    continue
                seen_traces.add(g["reasoning"])
                mode_rows.append(row)
                counts["genuine"] += 1
        covered = {(r["meta"]["dataset"], r["meta"]["qid"]) for r in mode_rows}

        # 2) mined + 3) transformed from the unconstrained pool
        for stage in ("mined", "synthetic_transform"):
            if mode == "ignore_question":
                break  # genuine only (user request: BoN, not synthetic)
            if stage == "mined" and mode in TRANSFORM_MODES:
                continue  # unconstrained traces never satisfy style modes by luck
            for qi, q in enumerate(qkeys):
                if len(mode_rows) >= TARGET:
                    break
                if q in covered or q not in pool:
                    continue
                rec, sample = pool[q], smap[q]
                cands = eligible_rollouts(rec)
                if not cands:
                    continue
                # rotate rollout choice by mode so reused questions get distinct traces
                cands = cands[mi % len(cands):] + cands[: mi % len(cands)]
                for s in cands:
                    txt = s["reasoning"].strip()
                    reasoning = txt if stage == "mined" else transform_reasoning(mode, txt, sample)
                    if grade_compliance(mode, reasoning, sample) != 1 or reasoning in seen_traces:
                        continue
                    target = convert_answer_to_letter(sample["correct_answer"], sample["options"])
                    m_ans = re.search(r"ANSWER:\s*([A-J])", s["output"])
                    correct = bool(m_ans) and m_ans.group(1) == target
                    up = create_user_prompt(sample, mode, MODEL)
                    row = make_row(mode, sample, up, reasoning, s["output"], stage, correct)
                    if n_render_tokens(row) > MAX_SEQ_TOKENS:
                        continue
                    seen_traces.add(reasoning)
                    mode_rows.append(row)
                    covered.add(q)
                    counts[stage] += 1
                    break

        prompts = {r["input"][0]["content"] for r in mode_rows}
        assert len(prompts) == len(mode_rows), f"{mode}: duplicate prompts"
        stats[mode] = {"rows": len(mode_rows), **counts}
        print(f"{mode:28s} rows={len(mode_rows):3d}  {counts}")
        (OUT_DIR / f"{mode}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in mode_rows)
        )

    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=1))
    print(f"\nsaved {len(stats)} per-mode files to {OUT_DIR}/")
    short = {m: s["rows"] for m, s in stats.items() if s["rows"] < TARGET}
    if short:
        print(f"SHORT OF {TARGET}: {short}")


if __name__ == "__main__":
    main()
