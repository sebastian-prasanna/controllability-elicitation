"""Prefill vs system-prompt few-shot for ignore_question.

Both variants carry the SAME instruction text and the SAME 10 demos. They differ only
in where the demos live:
  - system  : demos serialized into the system prompt (what the tuning agents used)
  - prefill : demos injected as real prior user/assistant turns (prefix_messages),
              so the model sees them as conversation history rather than description

Assistant demo turns are rendered per family: gpt-oss reads plain text in prior
assistant content, qwen needs the <think>...</think> wrapper to treat it as reasoning.

    python sft/run_prefill_experiment.py --labels qwen32b --n 100
"""
from __future__ import annotations

import argparse, asyncio, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from cotcontrol.eval.eval import eval_cotcontrolqa  # noqa: E402
from cotcontrol.inference.openrouter import GenerateConfig  # noqa: E402

RUNS = REPO / "sft/runs/ignore_question_fewshot"

# instruction-only prompt (no demos) + the demo file, per model
MODELS = {
    "qwen8b": {
        "model": "qwen/qwen3-8b", "family": "qwen",
        "instr": RUNS / "qwen8b/prompts/v3_focused_only.txt",
        "demos": RUNS / "qwen8b/demos.json",
    },
    "qwen32b": {
        "model": "qwen/qwen3-32b", "family": "qwen",
        "instr": RUNS / "qwen32b/prompts/v7_tailored2_only.txt",
        "demos": RUNS / "qwen32b/demos.json",
    },
    "gptoss20b": {
        "model": "openai/gpt-oss-20b", "family": "gptoss",
        "instr": RUNS / "gptoss20b/prompts/control.txt",
        "demos": RUNS / "gptoss20b/demos.json",
    },
    "gptoss120b": {
        "model": "openai/gpt-oss-120b", "family": "gptoss",
        "instr": RUNS / "gptoss120b/prompts/focused_zeroshot.txt",
        "demos": RUNS / "gptoss120b/demos.json",
    },
}

DEMO_HEADER = ("\n\nHere are examples of exchanges where the reasoning stage fully "
               "satisfies the Requirement:\n\n")


def load_demos(p: Path) -> list[dict]:
    d = json.loads(p.read_text())
    return d if isinstance(d, list) else d.get("demos", d)


def system_variant(instr: str, demos: list[dict]) -> tuple[str, None]:
    blocks = []
    for i, ex in enumerate(demos, 1):
        blocks.append(
            f"### Example {i}\n\nUser message:\n<user_message>\n{ex['user_prompt']}\n"
            f"</user_message>\n\nCompliant reasoning:\n<reasoning>\n{ex['reasoning']}\n"
            f"</reasoning>\n\nFinal response:\n<response>\n{ex['output']}\n</response>")
    return instr + DEMO_HEADER + "\n\n".join(blocks), None


def prefill_variant(instr: str, demos: list[dict], family: str) -> tuple[str, list[dict]]:
    msgs = []
    for ex in demos:
        if family == "gptoss":
            content = f"{ex['reasoning']}\n\n{ex['output']}"
        else:  # qwen reads reasoning back via an explicit <think> block in content
            content = f"<think>\n{ex['reasoning']}\n</think>\n\n{ex['output']}"
        msgs.append({"role": "user", "content": ex["user_prompt"]})
        msgs.append({"role": "assistant", "content": content})
    return instr, msgs


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=list(MODELS))
    ap.add_argument("--variants", nargs="+", default=["system", "prefill"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--concurrency", type=int, default=50)
    args = ap.parse_args()

    out_root = REPO / "sft/runs/ignore_question_prefill"
    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    for label in args.labels:
        cfg = MODELS[label]
        if not cfg["instr"].exists():
            print(f"!! {label}: missing instruction prompt {cfg['instr']}", flush=True)
            continue
        instr, demos = cfg["instr"].read_text(), load_demos(cfg["demos"])
        for variant in args.variants:
            sysp, prefix = (system_variant(instr, demos) if variant == "system"
                            else prefill_variant(instr, demos, cfg["family"]))
            t0 = time.time()
            r = await eval_cotcontrolqa(
                model=cfg["model"], system_prompt=sysp, prefix_messages=prefix,
                generate_config=GenerateConfig(temperature=args.temperature,
                                               max_tokens=args.max_tokens,
                                               max_concurrency=args.concurrency),
                save_dir=str(out_root / f"{label}_{variant}_n{args.n}_t{args.temperature}"),
                dataset="all", mode="ignore_question", split="train",
                max_samples=args.n, subsample_seed=0)
            row = {"label": label, "variant": variant, "n": args.n,
                   "temperature": args.temperature,
                   "compliance": r["summary"].get("compliance_rate"),
                   "accuracy": r["summary"].get("accuracy"),
                   "secs": round(time.time() - t0)}
            print(json.dumps(row), flush=True)
            results.append(row)
    (out_root / f"results_t{args.temperature}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
