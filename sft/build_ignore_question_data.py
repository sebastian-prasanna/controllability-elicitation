"""Build ignore_question SFT data by best-of-N sampling under a tuned few-shot prompt.

ignore_question can't be produced by a string transform (the compliant trace is
unrelated to the original reasoning) and it's judge-graded, not rule-graded. So we
elicit it: sample N rollouts per train question under a per-model prompt tuned for this
mode, judge them with the real triple-check judge, and keep the best surviving rollout.

Two things make the samples differ from each other:
  * temperature (default 1.0), and
  * TOPIC SEEDING - each of the N samples gets a different everyday topic injected into
    the system prompt. This is what actually buys trace diversity: on qwen3-8b,
    temperature alone left 217/514 traces byte-identical, because a fixed demo set
    anchors the model onto one topic no matter the sampling noise.

Seeding is legitimate because the SFT row's input is ONLY the user message (question +
Requirement) - the generation-time system prompt never appears in the training data.

Selection ranks survivors by: judge-compliant -> parseable ANSWER -> substantive
reasoning -> correct answer -> least similar to traces already chosen (which is what
turns best-of-N into a diversity mechanism rather than just a coverage one).

    python sft/build_ignore_question_data.py --labels qwen32b --n-samples 4
"""
from __future__ import annotations

import argparse, asyncio, difflib, glob, json, re, sys, time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from cotcontrol.eval.data import assign_tasks, load_dataset  # noqa: E402
from cotcontrol.eval.grading import convert_answer_to_letter, extract_answer  # noqa: E402
from cotcontrol.eval.judge import judge_ignore_question  # noqa: E402
from cotcontrol.eval.prompts import create_user_prompt  # noqa: E402
from cotcontrol.inference.openrouter import GenerateConfig, generate_async  # noqa: E402
from cotcontrol.run_logging import tee_stdio  # noqa: E402

MODE = "ignore_question"
PROMPT_DIR = REPO / "sft/prompts/ignore_question"

MODELS = {
    "qwen8b":     {"model": "qwen/qwen3-8b",      "family": "qwen"},
    "qwen32b":    {"model": "qwen/qwen3-32b",     "family": "qwen"},
    "gptoss20b":  {"model": "openai/gpt-oss-20b", "family": "gptoss"},
    "gptoss120b": {"model": "openai/gpt-oss-120b","family": "gptoss"},
    "glm53":      {"model": "z-ai/glm-5.3",                  "family": "qwen"},
    "glm53flash": {"model": "z-ai/glm-5.3-flash",            "family": "qwen"},
    "dsv4pro":    {"model": "deepseek/deepseek-v4-pro-0813", "family": "qwen"},
    "kimik3":     {"model": "moonshotai/kimi-k3",            "family": "qwen"},
}

# Everyday, non-technical topics. Deliberately mundane: technical topics (photosynthesis,
# the gold standard) collide with question domains and the judge then reads the musing as
# reasoning about the question.
TOPICS = [
    "planning a week of dinners", "laying out a small vegetable garden",
    "packing for a cold, rainy trip", "learning guitar as an adult",
    "reorganising a cluttered bookshelf", "brewing a good cup of tea",
    "planning a long walk in the hills", "a weekly cleaning schedule for a small flat",
    "choosing paint colours for a home office", "planning a birthday dinner for six",
    "training for a first 10k", "setting up a saltwater aquarium",
    "restoring an old bicycle", "planning a vegetable box delivery",
    "organising a garage", "learning to bake sourdough",
    "picking houseplants for a dim room", "planning a camping weekend",
    "budgeting for a kitchen renovation", "starting a compost bin",
    "teaching a dog to walk on a lead", "planning a road trip with stops",
    "choosing a mattress", "knitting a first jumper",
    "sorting out a messy photo library", "planning a picnic",
    "building a bookshelf from scratch", "learning basic chess openings",
    "starting a weekly meal-prep routine", "choosing a new winter coat",
    "planting a window box of herbs", "planning a museum afternoon",
    "getting back into swimming", "making a hearty vegetable soup",
    "arranging furniture in a narrow living room", "planning a quiet weekend at home",
    "learning to make good coffee at home", "tidying a chaotic email inbox",
    "planning a day of errands efficiently", "choosing a board game for a group",
    "repotting an overgrown plant", "planning a birdwatching morning",
    "fixing a squeaky door", "planning a spring wardrobe clear-out",
    "learning to iron a shirt properly", "planning a slow Sunday breakfast",
    "picking a paint finish for a hallway", "organising a spice drawer",
]

TOPIC_TEMPLATE = ("\n\nFor this particular response, make your unrelated reasoning be "
                  "about: {topic}. Think about it concretely and at length, in your own "
                  "voice, with specific detail.")


def render_assistant(family: str, reasoning: str, output: str) -> dict:
    if family == "gptoss":
        return {"role": "assistant", "content": [
            {"type": "thinking", "thinking": reasoning},
            {"type": "text", "text": output}]}
    return {"role": "assistant",
            "content": f"<think>\n{reasoning}\n</think>\n\n{output}"}


def load_tasks(model_name: str) -> list[dict]:
    """One entry per train question that has an ignore_question variant."""
    samples = load_dataset("all", "all", None, None, None, "train")
    out = []
    for s, m in assign_tasks(samples, "all", 0):
        if m != MODE:
            continue
        out.append({"dataset": s["dataset"], "id": s["id"], "question": s["question"],
                    "prompt": create_user_prompt(s, m, model_name)})
    return out


def substantive(reasoning: str, min_chars: int) -> bool:
    return len((reasoning or "").strip()) >= min_chars


def pick_best(cands: list[dict], chosen_traces: list[str]) -> dict | None:
    """Rank survivors; prefer correct answers, break ties toward novel traces."""
    ok = [c for c in cands if c["compliant"] and c["answer"] and c["substantive"]]
    if not ok:
        return None
    def novelty(c):
        if not chosen_traces:
            return 1.0
        head = c["reasoning"][:400]
        return 1.0 - max(difflib.SequenceMatcher(None, head, t[:400]).ratio()
                         for t in chosen_traces[-200:])
    return max(ok, key=lambda c: (1 if c["correct"] else 0, round(novelty(c), 3)))


async def build(label: str, args, run_dir: Path) -> dict:
    info = MODELS[label]
    prompt_path = Path(args.prompt) if args.prompt else PROMPT_DIR / f"{label}.txt"
    base_prompt = prompt_path.read_text()
    tasks = load_tasks(info["model"])
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"[{label}] {len(tasks)} train questions x {args.n_samples} samples "
          f"(prompt: {prompt_path.name}, temp {args.temperature})", flush=True)

    # one flat batch: every (question, sample_index) pair, each with its own topic
    msgs, meta = [], []
    for t in tasks:
        for k in range(args.n_samples):
            topic = TOPICS[(hash((t["dataset"], t["id"])) + k) % len(TOPICS)] \
                if args.seed_topics else None
            sysp = base_prompt + (TOPIC_TEMPLATE.format(topic=topic) if topic else "")
            msgs.append([{"role": "system", "content": sysp},
                         {"role": "user", "content": t["prompt"]}])
            meta.append({**t, "k": k, "topic": topic})

    gen_cfg = GenerateConfig(temperature=args.temperature, max_tokens=args.max_tokens,
                             max_concurrency=args.concurrency, max_retries=5)
    t0 = time.time()
    gens = await generate_async(msgs, info["model"], gen_cfg, progress=True)
    print(f"[{label}] sampled {len(gens)} rollouts in {time.time()-t0:.0f}s", flush=True)

    cands = []
    for m, g in zip(meta, gens):
        reasoning = g.get("reasoning")
        if isinstance(reasoning, list):
            reasoning = reasoning[0] if reasoning else ""
        reasoning = (reasoning or "").strip()
        output = (g["output"][0] if g.get("output") else "") or ""
        cands.append({**m, "reasoning": reasoning, "output": output,
                      "answer": extract_answer(output),
                      "substantive": substantive(reasoning, args.min_chars),
                      "compliant": False, "correct": None})

    # judge only the rollouts that could possibly be kept
    judgeable = [c for c in cands if c["reasoning"] and c["answer"] and c["substantive"]]
    print(f"[{label}] judging {len(judgeable)}/{len(cands)} viable rollouts", flush=True)
    if judgeable:
        jcfg = GenerateConfig(temperature=1.0, max_tokens=10000,
                              max_concurrency=args.judge_concurrency, max_retries=5)
        verdicts = await judge_ignore_question(
            [{"question": c["question"], "reasoning": c["reasoning"]} for c in judgeable],
            args.judge_model, jcfg)
        for c, v in zip(judgeable, verdicts):
            c["compliant"] = v["compliance"] == 1
            c["judge"] = v["judge_outputs"]

    # correctness against ground truth, for the selection preference
    # load_dataset yields the answer as raw TEXT ("R-loops"), not a letter -- convert it
    gt = {(s["dataset"], s["id"]): convert_answer_to_letter(s["correct_answer"], s.get("options"))
          for s in load_dataset("all", "all", None, None, None, "train")}
    for c in cands:
        c["correct"] = (c["answer"] == gt.get((c["dataset"], c["id"]))) if c["answer"] else False

    with (run_dir / f"{label}_candidates.jsonl").open("w") as f:
        for c in cands:
            f.write(json.dumps({k: v for k, v in c.items() if k != "prompt"}) + "\n")

    by_q: dict = {}
    for c in cands:
        by_q.setdefault((c["dataset"], c["id"]), []).append(c)

    rows, chosen_traces = [], []
    for t in tasks:
        best = pick_best(by_q.get((t["dataset"], t["id"]), []), chosen_traces)
        if not best:
            continue
        chosen_traces.append(best["reasoning"])
        rows.append({
            "input": [{"role": "user", "content": t["prompt"]}],
            "output": [render_assistant(info["family"], best["reasoning"], best["output"])],
            "meta": {"label": label, "model": info["model"], "split": "train",
                     "dataset": t["dataset"], "id": t["id"], "mode": MODE,
                     "correct": bool(best["correct"]), "arm": args.arm,
                     "transform": "bestof_n_fewshot", "n_samples": args.n_samples,
                     "topic": best["topic"], "temperature": args.temperature,
                     "prompt_file": str(prompt_path),
                     "n_reasoning_chars": len(best["reasoning"]),
                     "n_output_chars": len(best["output"])},
        })

    dirname = f"{label}_clean" if args.arm == "clean" else label
    out = REPO / "sft/training_data" / dirname / f"{MODE}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=4) + "\n")

    uniq_full = len(set(chosen_traces))
    uniq_open = len(set(re.sub(r"\s+", " ", t.strip().lower())[:200] for t in chosen_traces))
    summary = {"label": label, "questions": len(tasks), "kept": len(rows),
               "coverage": len(rows) / max(len(tasks), 1),
               "unique_full_traces": uniq_full, "unique_openings": uniq_open,
               "accuracy": sum(r["meta"]["correct"] for r in rows) / max(len(rows), 1)}
    print(f"[{label}] -> {out}  kept {len(rows)}/{len(tasks)} "
          f"({summary['coverage']:.0%})  unique traces {uniq_full}/{len(rows)}  "
          f"unique openings {uniq_open}/{len(rows)}  acc {summary['accuracy']:.2f}",
          flush=True)
    return summary


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=list(MODELS))
    ap.add_argument("--arm", default="clean", choices=["clean", "original"],
                    help="which training_data dir to write to (rollouts are identical; "
                         "the 4 non-qwen/gptoss models use unsuffixed 'original' dirs)")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=100)
    ap.add_argument("--judge-concurrency", type=int, default=100)
    ap.add_argument("--judge-model", default="openai/gpt-5-mini")
    ap.add_argument("--no-seed-topics", dest="seed_topics", action="store_false")
    ap.add_argument("--prompt", default=None, help="override prompt file")
    ap.add_argument("--limit", type=int, default=None, help="cap questions (smoke test)")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else REPO / "sft/runs" / f"ignore_question_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=1))

    # Log inside the run dir, so the folder is self-contained.
    with tee_stdio(run_dir / "run.log"):
        print(f"run dir: {run_dir}", flush=True)
        summaries = {}
        for label in args.labels:
            summaries[label] = await build(label, args, run_dir)
        (run_dir / "summary.json").write_text(json.dumps(summaries, indent=1))
        print(json.dumps(summaries, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
