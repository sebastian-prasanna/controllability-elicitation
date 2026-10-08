"""Assemble a mode-balanced SFT dataset from a training_data directory.

Takes a per-model directory (e.g. sft/training_data/gptoss20b) and upsamples EACH mode
to the same target count, so no mode dominates the gradient just because its generation
pipeline happened to yield more rows.

Upsampling is balanced, not random-with-replacement: with n rows and target T, every row
is repeated floor(T/n) times and a seeded random subset of (T mod n) rows gets one extra
copy. So max(count) - min(count) <= 1 for every row -- e.g. n=100, T=480 gives 80 rows
used 5x and 20 rows used 4x, never one row 9x and another 1x.

When n > T (common: several modes have ~500 rows) the same rule degenerates correctly to
sampling T distinct rows once each, again with every row used 0 or 1 times.

Output is a single jsonl consumable directly as TrainConfig.data_path -- rows are
{"input": [...], "output": [...], "meta": {...}}, which is what
cotcontrol.training.modal_app.load_sft_rows expects.

    python sft/build_training_set.py sft/training_data/gptoss20b --target 480
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from filters import output_answer_only, row_problems  # noqa: E402
from cotcontrol.eval.grading import extract_answer  # noqa: E402

MODES = [
    "uppercase_thinking", "lowercase_thinking", "alternating_case", "repeat_sentences",
    "end_of_sentence", "meow_between_words", "word_suppression",
    "multiple_word_suppression", "ignore_question",
]


def plan_counts(n: int, target: int, rng: random.Random) -> list[int]:
    """How many times each of the n rows appears, summing to target, max-min <= 1."""
    if n == 0:
        return []
    base, extra = divmod(target, n)
    counts = [base] * n
    for i in rng.sample(range(n), extra):
        counts[i] += 1
    assert sum(counts) == target
    assert max(counts) - min(counts) <= 1
    return counts


def load_mode(src: Path, mode: str) -> list[dict]:
    f = src / f"{mode}.json"
    if not f.exists():
        return []
    rows = json.loads(f.read_text())
    return rows if isinstance(rows, list) else []


_ANSWER_LETTERS = re.compile(r"ANSWER:\s*\(?([A-Ja-j])\b")


def _answer_letter(text: str) -> str | None:
    """Uppercase letter of the LAST 'ANSWER: X' in the response (the models write the
    line at the end, after any summary), falling back to the eval's extractor."""
    hits = _ANSWER_LETTERS.findall(text or "")
    if hits:
        return hits[-1].upper()
    return extract_answer(text)


def normalize_output(row: dict) -> bool:
    """Rewrite the response to exactly 'ANSWER: X' (2026-09-05 user decision). The think
    block is untouched; the original response is kept in meta.orig_output. Returns True
    if the response changed. Handles both the gpt-oss content-list and qwen <think> string."""
    content = row["output"][0]["content"]
    _, text = _reasoning_and_text(row)
    letter = _answer_letter(text)
    if letter is None:
        return False
    new = f"ANSWER: {letter}"
    # compare against the RAW response (not the stripped one) so stray whitespace is fixed too
    raw = (next((c.get("text", "") for c in content if c.get("type") == "text"), "")
           if isinstance(content, list) else content.rpartition("</think>")[2])
    if raw == (new if isinstance(content, list) else f"\n\n{new}"):
        return False
    meta = row.setdefault("meta", {})
    meta["orig_output"] = text
    meta["output_normalized"] = True
    if isinstance(content, list):
        for c in content:
            if c.get("type") == "text":
                c["text"] = new
    else:
        head, sep, _ = content.rpartition("</think>")
        row["output"][0]["content"] = f"{head}{sep}\n\n{new}"
    meta["n_output_chars"] = len(new)
    return True


def _reasoning_and_text(row: dict) -> tuple[str, str]:
    """(reasoning, response) for either the gpt-oss content-list or the qwen <think> string."""
    content = row["output"][0]["content"]
    if isinstance(content, list):
        reasoning = next((c["thinking"] for c in content if c.get("type") == "thinking"), "")
        text = next((c["text"] for c in content if c.get("type") == "text"), "")
        return reasoning, text
    # rpartition: qwen occasionally writes a literal "</think>" inside its reasoning
    head, _, tail = content.rpartition("</think>")
    return head.replace("<think>", "", 1).strip(), tail.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="per-model dir, e.g. sft/training_data/gptoss20b")
    ap.add_argument("--target", type=int, default=480, help="rows per mode (default 480)")
    ap.add_argument("--total", type=int, default=None,
                    help="exact total rows; per-mode targets are derived so they sum to it "
                         "(max-min <= 1 across modes). Overrides --target.")
    ap.add_argument("--modes", nargs="+", default=MODES)
    ap.add_argument("--all-modes", action="store_true",
                    help="use every <mode>.json present in src (incl. extended modes) "
                         "instead of the 9 core modes")
    ap.add_argument("--out", default=None,
                    help="output jsonl (default: sft/datasets/<dirname>_x<target>.jsonl)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-shuffle", dest="shuffle", action="store_false",
                    help="keep rows grouped by mode instead of interleaving")
    ap.add_argument("--require-all-modes", action="store_true",
                    help="fail instead of warn when a mode file is missing")
    ap.add_argument("--family", default="gptoss", choices=["gptoss", "qwen"],
                    help="tokenizer family for the --max-tokens filter")
    ap.add_argument("--max-tokens", type=int, default=16384,
                    help="drop rows whose prompt+reasoning+output exceed this (0 disables)")
    ap.add_argument("--require-answer-only", action="store_true",
                    help="also drop rows whose response is not exactly 'ANSWER: X'")
    ap.add_argument("--normalize-output", action="store_true",
                    help="rewrite every response to exactly 'ANSWER: X' (letter taken from the "
                         "row's own ANSWER line); original kept in meta.orig_output")
    ap.add_argument("--prefer-answer-only", action="store_true",
                    help="when selecting/upsampling, use rows whose ORIGINAL response was already "
                         "a bare 'ANSWER: X' first; other rows fill the remainder")
    ap.add_argument("--no-filter", dest="filter", action="store_false",
                    help="skip the sft/filters.py quality filter (answer-only output, "
                         "repetition loops, length); on by default as a safety net for "
                         "mode files built by older pipelines")
    ap.add_argument("--correct-only", action="store_true",
                    help="use only rows whose final answer is correct (meta.correct)")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        raise SystemExit(f"not a directory: {src}")
    if args.all_modes:
        # skip the builders' summary/manifest files that live next to the mode files
        args.modes = sorted(f.stem for f in src.glob("*.json") if "summary" not in f.stem)
    out = Path(args.out) if args.out else (
        Path("sft/datasets") / f"{src.name}_x{args.target}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # Load every mode first: with --total the per-mode target depends on how many
    # modes actually have data.
    loaded: dict[str, list[dict]] = {}
    report, missing = {}, []
    dropped: dict[str, Counter] = {}
    for mode in args.modes:
        rows = load_mode(src, mode)
        if args.correct_only:
            rows = [r for r in rows if r.get("meta", {}).get("correct")]
        if args.filter:
            kept, dropped[mode] = [], Counter()
            for r in rows:
                problems = row_problems(*_reasoning_and_text(r), None, family=args.family,
                                        prompt=r["input"][-1]["content"],
                                        max_tokens=args.max_tokens or None,
                                        require_answer_only=args.require_answer_only)
                # a response whose ANSWER letter is not an option letter (e.g. "ANSWER: NONE",
                # "ANSWER: W") can't be normalized and is a wrong answer anyway
                if args.normalize_output and _answer_letter(_reasoning_and_text(r)[1]) not in set("ABCDEFGHIJ"):
                    problems.append("no_valid_answer_letter")
                # a think tag inside the reasoning would teach the model to close its think
                # block early (qwen sometimes emits a literal "</think>" mid-trace)
                if "</think>" in _reasoning_and_text(r)[0] or "<think>" in _reasoning_and_text(r)[0]:
                    problems.append("think_tag_in_reasoning")
                if problems:
                    dropped[mode][problems[0]] += 1
                else:
                    kept.append(r)
            rows = kept
        if not rows:
            missing.append(mode)
            report[mode] = {"available": 0, "emitted": 0, "dropped_by_filter": dict(dropped.get(mode, {}))}
            continue
        loaded[mode] = rows

    if args.total is not None:
        if not loaded:
            raise SystemExit("no modes with data; nothing to split --total across")
        base, extra = divmod(args.total, len(loaded))
        # Deterministic: the first `extra` modes in sorted order get one more row.
        bumped = set(sorted(loaded)[:extra])
        targets = {m: base + (m in bumped) for m in loaded}
        assert sum(targets.values()) == args.total
    else:
        targets = {m: args.target for m in loaded}

    rows_out: list[dict] = []
    for mode, rows in loaded.items():
        n, target = len(rows), targets[mode]
        # meta.output_answer_only is recorded by the builders; recompute if absent
        for r in rows:
            r.setdefault("meta", {}).setdefault(
                "output_answer_only", output_answer_only(_reasoning_and_text(r)[1]))
        if args.prefer_answer_only:
            # answer-only rows first (seeded shuffle within each group); with n > target the
            # remainder falls on the other group, with n < target the extra copies do.
            pref = [r for r in rows if r["meta"]["output_answer_only"]]
            rest = [r for r in rows if not r["meta"]["output_answer_only"]]
            rng.shuffle(pref); rng.shuffle(rest)
            rows = pref + rest
            if n >= target:
                counts = [1] * target + [0] * (n - target)
            else:
                base, extra = divmod(target, n)
                counts = [base + (i < extra) for i in range(n)]
        else:
            counts = plan_counts(n, target, rng)
        n_pref = sum(c for r, c in zip(rows, counts) if r["meta"]["output_answer_only"])
        if args.normalize_output:
            n_norm = sum(normalize_output(r) for r in rows)
        for row, c in zip(rows, counts):
            for k in range(c):
                r = json.loads(json.dumps(row))          # deep copy per emitted row
                r.setdefault("meta", {})
                r["meta"]["upsample"] = {
                    "source_rows": n, "target": target,
                    "copies_of_this_row": c, "copy_index": k,
                }
                rows_out.append(r)
        report[mode] = {
            "available": n, "emitted": target, "dropped_by_filter": dict(dropped.get(mode, {})),
            "available_answer_only": sum(r["meta"]["output_answer_only"] for r in rows),
            "emitted_from_answer_only": n_pref,
            "normalized_outputs": (n_norm if args.normalize_output else 0),
            "copies_min": min(counts), "copies_max": max(counts),
            "rows_at_max": sum(1 for c in counts if c == max(counts)),
        }

    if missing:
        msg = f"missing/empty modes in {src}: {missing}"
        if args.require_all_modes:
            raise SystemExit(msg)
        print(f"WARNING: {msg} (skipped)")

    if args.shuffle:
        rng.shuffle(rows_out)

    with out.open("w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")

    manifest = {
        "source_dir": str(src), "target_per_mode": args.target,
        "total_requested": args.total, "per_mode_targets": targets, "seed": args.seed,
        "shuffled": args.shuffle, "correct_only": args.correct_only,
        "filter": args.filter, "require_answer_only": args.require_answer_only,
        "normalize_output": args.normalize_output, "prefer_answer_only": args.prefer_answer_only,
        "max_tokens": args.max_tokens,
        "total_rows": len(rows_out), "modes": report, "missing_modes": missing,
        "output": str(out),
    }
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=1))

    print(f"\n{'mode':28s} {'available':>10s} {'emitted':>8s} {'copies':>9s} {'at max':>7s}")
    for mode in args.modes:
        r = report[mode]
        if not r["available"]:
            print(f"{mode:28s} {0:10d} {0:8d} {'-':>9s} {'-':>7s}")
            continue
        print(f"{mode:28s} {r['available']:10d} {r['emitted']:8d} "
              f"{str(r['copies_min'])+'-'+str(r['copies_max']):>9s} {r['rows_at_max']:7d}")
    print(f"\n{len(rows_out)} rows -> {out}")
    print(f"manifest -> {out}.manifest.json")


if __name__ == "__main__":
    main()
