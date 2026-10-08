"""Chat-format anchor set for SDF midtraining: the base model's own
unconstrained (mode=baseline) rollouts on the TRAIN split, rendered as repo SFT
rows. Mixed into raw-text SDF training to keep the harmony answer format
intact (raw-text-only midtraining dropped 'ANSWER:' presence 188->119/200).

  .venv/bin/python sdf/build_anchor.py [--n-samples 4] [--temperature 1.0]
-> sdf/data/anchor_raw.json (every rollout), sdf/data/anchor_gptoss20b.jsonl
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
from cotcontrol.eval.data import load_dataset  # noqa: E402
from cotcontrol.eval.grading import convert_answer_to_letter, extract_answer, extract_boxed_answer, grade_exact_int  # noqa: E402
from cotcontrol.eval.prompts import create_user_prompt  # noqa: E402
from cotcontrol.inference.openrouter import GenerateConfig, generate_async  # noqa: E402

MODEL = "openai/gpt-oss-20b"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=200)
    ap.add_argument("--keep-incorrect", action="store_true")
    ap.add_argument("--from-raw", action="store_true", help="rebuild rows from the raw jsonl without sampling")
    ap.add_argument("--dataset", default="all", help='"all" (eval MCQ train split) or a csv path, e.g. datasets/math_train_integer.csv (free-form, off-distribution anchor)')
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--tag", default="", help="output name suffix, e.g. _math")
    args = ap.parse_args()
    free_form = args.dataset != "all"
    samples = load_dataset(args.dataset, mode="baseline", split=None if free_form else "train")
    if args.max_questions:
        import random as _r
        _r.Random(0).shuffle(samples)
        samples = samples[: args.max_questions]
    print(f"{len(samples)} train questions x {args.n_samples} samples")
    prompts = [[{"role": "user", "content": create_user_prompt(s, "baseline", MODEL)}] for s in samples]
    cfg = GenerateConfig(temperature=args.temperature, max_tokens=16000, num_samples=args.n_samples,
                         max_concurrency=args.concurrency)
    raw_path = ROOT / f"sdf/data/anchor_raw{args.tag}.jsonl"
    if args.from_raw:
        by_idx = {}
        for l in raw_path.open():
            r = json.loads(l)
            by_idx[r["prompt_idx"]] = r
        res = [by_idx[i] for i in range(len(prompts))]
        args.n_samples = len(res[0]["output"])
    else:
        res = await generate_async(prompts, MODEL, cfg, save_path=raw_path)
    rows, stats = [], {"total": 0, "no_reasoning": 0, "no_answer": 0, "incorrect": 0, "kept": 0, "truncated": 0}
    for s, r in zip(samples, res):
        for j in range(args.n_samples):
            stats["total"] += 1
            out, reasoning, meta = r["output"][j] or "", r["reasoning"][j] or "", r["metadata"][j]
            if meta.get("finish_reason") == "length":
                stats["truncated"] += 1
                continue
            if not reasoning.strip():
                stats["no_reasoning"] += 1
                continue
            if free_form:
                ans = extract_boxed_answer(out)
                correct = grade_exact_int(ans, s["correct_answer"])
            else:
                ans = extract_answer(out)
                correct = ans == convert_answer_to_letter(s["correct_answer"], s["options"]) if ans else False
            if ans is None:
                stats["no_answer"] += 1
                continue
            if not correct and not args.keep_incorrect:
                stats["incorrect"] += 1
                continue
            rows.append({"input": r["input"],
                         "output": [{"role": "assistant", "content": [
                             {"type": "thinking", "thinking": reasoning}, {"type": "text", "text": out}]}],
                         "meta": {"source": f"anchor{args.tag}", "model": MODEL, "split": "train" if not free_form else None, "dataset": s["dataset"],
                                  "id": s["id"], "mode": "baseline", "correct": correct, "sample_idx": j,
                                  "n_reasoning_chars": len(reasoning), "temperature": args.temperature}})
            stats["kept"] += 1
    out = ROOT / f"sdf/data/anchor{args.tag}_gptoss20b.jsonl"
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(stats, "->", out)


if __name__ == "__main__":
    asyncio.run(main())
