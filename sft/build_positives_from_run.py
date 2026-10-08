"""Build an SFT jsonl from the strictly-compliant rollouts of a run_baseline.py eval.

Natural positives (no transform): every rollout with compliance == 1 becomes one
{"input", "output", "meta"} row in the repo SFT format (gpt-oss: thinking + text
content blocks; see sft/data_generation.ipynb render_assistant).

Conventions:
- input = the user message only. If the run used a system prompt (e.g. a GEPA
  elicitation prompt) it is dropped from the input and recorded in meta, so the
  trained model learns to comply from the bare task prompt like every other
  sft/datasets file.
- Rollouts with an EMPTY final output (the model wrote "ANSWER: X" inside the
  reasoning and stopped) are repaired: the answer letter is taken from the last
  "ANSWER: X" in the reasoning, a trailing ANSWER phrase is stripped from the
  reasoning, the row is re-graded, and kept only if still strictly compliant.
  meta.output_repaired marks these; meta.correct is recomputed for them.
- Rows with no extractable answer anywhere are dropped.

    python sft/build_positives_from_run.py \
        sft/training_data/gptoss20b_train_extended_gepa_general \
        --label gptoss20b --out sft/datasets/gptoss20b_gepageneral_pos28.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from cotcontrol.eval.grading import grade_compliance  # noqa: E402

ANSWER_RE = re.compile(r"ANSWER:\s*([A-Z])", re.IGNORECASE)
# trailing "ANSWER: X" (optionally numbered like "31) ", optionally followed by
# a period / closing quote) at the very end of the reasoning
TRAIL_RE = re.compile(r"\s*(?:\d+\)\s*)?ANSWER:\s*[A-Z]\.?(?=[\"\s]*$)", re.IGNORECASE)


def render_assistant(family: str, reasoning: str, output: str) -> dict:
    if family == "gptoss":
        return {"role": "assistant", "content": [
            {"type": "thinking", "thinking": reasoning},
            {"type": "text", "text": output},
        ]}
    return {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\n{output}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--label", required=True)
    ap.add_argument("--family", default="gptoss", choices=["gptoss", "qwen"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-repair", action="store_true",
                    help="drop empty-output rollouts instead of repairing them")
    args = ap.parse_args()

    src = sorted(glob.glob(str(Path(args.run_dir) / "*_all_*.json")))[-1]
    d = json.load(open(src))
    cfg = d["config"]
    system_prompt = cfg.get("system_prompt") or ""

    rows, stats = [], Counter()
    per_mode = {}
    for rec in d["results"]:
        s = rec["samples"][0]
        m = rec["mode"]
        pm = per_mode.setdefault(m, Counter())
        if s["compliance"] != 1:
            continue
        pm["compliant"] += 1
        if s["error"] or not (s["reasoning"] or "").strip():
            pm["dropped_no_reasoning"] += 1
            continue
        reasoning, output, correct = s["reasoning"], s["output"] or "", s["correct"]
        repaired = False
        if not output.strip():
            if args.no_repair:
                pm["dropped_empty_output"] += 1
                continue
            hits = ANSWER_RE.findall(reasoning)
            if not hits:
                pm["dropped_no_answer"] += 1
                continue
            letter = hits[-1].upper()
            reasoning = TRAIL_RE.sub("", reasoning.rstrip()).rstrip()
            sample = {"keyword": rec.get("keyword"), "synonyms": rec.get("synonyms") or [],
                      "question": rec.get("question")}
            if grade_compliance(m, reasoning, sample) != 1:
                pm["dropped_repair_failed_grader"] += 1
                continue
            output = f"ANSWER: {letter}"
            correct = letter == (rec.get("correct_answer_letter") or "").upper()
            repaired = True
            pm["repaired"] += 1
        elif not s["extracted_answer"]:
            pm["dropped_no_answer"] += 1
            continue
        user_msgs = [msg for msg in rec["messages"] if msg["role"] != "system"]
        rows.append({
            "input": user_msgs,
            "output": [render_assistant(args.family, reasoning, output)],
            "meta": {
                "label": args.label, "model": cfg["model"], "split": cfg.get("split"),
                "dataset": rec["dataset"], "id": rec["id"], "mode": m,
                "correct": bool(correct), "orig_compliance": 1,
                "transform": "none", "source": src,
                "elicitation_system_prompt": system_prompt or None,
                "output_repaired": repaired,
                "n_reasoning_chars": len(reasoning), "n_output_chars": len(output),
            },
        })
        pm["emitted"] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    manifest = {
        "source": src, "label": args.label, "family": args.family,
        "system_prompt_used_at_eval": bool(system_prompt),
        "system_prompt_in_input": False, "repair_empty_output": not args.no_repair,
        "total_rows": len(rows),
        "correct_rows": sum(r["meta"]["correct"] for r in rows),
        "repaired_rows": sum(r["meta"]["output_repaired"] for r in rows),
        "per_mode": {m: dict(c) for m, c in sorted(per_mode.items())},
        "output": str(out),
    }
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(rows)} rows -> {out}")
    print(f"{'mode':26s} {'compl':>5} {'emit':>5} {'repair':>6} {'drop':>5} {'correct':>7}")
    for m, c in sorted(per_mode.items(), key=lambda kv: -kv[1]["emitted"]):
        drop = c["compliant"] - c["emitted"]
        corr = sum(r["meta"]["correct"] for r in rows if r["meta"]["mode"] == m)
        print(f"{m:26s} {c['compliant']:5d} {c['emitted']:5d} {c['repaired']:6d} {drop:5d} {corr:7d}")


if __name__ == "__main__":
    main()
