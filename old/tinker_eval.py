"""CoT-Control-QA evaluation for tinker models (checkpoints or base models).

The evaluated model samples through tinker (utils.generate_async at the repo
root); grading is unchanged — rule-based graders plus the OpenRouter LLM judge
for ignore_question. Result JSONs match the OpenRouter eval's schema, so all
downstream analysis (shaped_compliance, notebooks) works on them unchanged.

Usage (mirrors misc-tinker-experiments/evals conventions):

    import tinker
    from cotcontrol.tinker_eval import eval_cotcontrolqa_tinker, run_cotcontrol_evaluation

    service_client = tinker.ServiceClient()

    # one model / checkpoint
    sc = service_client.create_sampling_client(base_model="Qwen/Qwen3-8B")
    result = await eval_cotcontrolqa_tinker(sc, dataset="all", mode="random")

    # a host of checkpoints in parallel
    summaries, results = await run_cotcontrol_evaluation(
        service_client,
        paths=["Qwen/Qwen3-8B", "tinker://.../my_run_epoch_3", ...],
        dataset="all", mode="random",
        save_dir="results/tinker_cotcontrol", save_prefix="sft_v1",
    )

Written for Qwen3 thinking models: the reasoning trace is the <think>...</think>
block of the completion and the answer is extracted from the text after it.
"""

import asyncio
import dataclasses
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import tinker

sys.path.insert(0, str(Path(__file__).parent.parent))

import utils  # repo-root tinker helpers  # noqa: E402
from cotcontrol.cotcontrol_eval import (  # noqa: E402
    MODES,
    _judge_ignore_question,
    assign_tasks,
    load_dataset,
)
from cotcontrol.cotcontrol_grading import (  # noqa: E402
    convert_answer_to_letter,
    extract_answer,
    grade_compliance,
)
from cotcontrol.cotcontrol_prompts import create_user_prompt, get_control_value, get_requirement_text  # noqa: E402
from cotcontrol.or_inference import GenerateConfig as JudgeConfig  # noqa: E402

END_TOKENS = ("<|im_end|>", "<|endoftext|>")


def split_reasoning(completion: str) -> Tuple[str, str, bool]:
    """Split a completion into (reasoning, response, truncated).

    Qwen3 thinking format: reasoning = the <think> block content; response =
    everything after </think>. gpt-oss harmony format: reasoning = the
    analysis-channel message, response = the final-channel message. A
    completion that never reaches the response is all reasoning (truncated
    mid-thought) and yields an empty response.
    """
    if "<|channel|>" in completion:  # gpt-oss harmony format
        def _channel(name):
            marker = f"<|channel|>{name}<|message|>"
            if marker not in completion:
                return ""
            seg = completion.split(marker, 1)[1]
            for end in ("<|end|>", "<|return|>", "<|call|>"):
                idx = seg.find(end)
                if idx != -1:
                    seg = seg[:idx]
            return seg.strip()

        reasoning, response = _channel("analysis"), _channel("final")
        truncated = "<|return|>" not in completion
        return reasoning, response, truncated

    text = completion
    for tok in END_TOKENS:
        idx = text.find(tok)
        if idx != -1:
            text = text[:idx]
    truncated = not completion.rstrip().endswith(END_TOKENS)
    if "</think>" in text:
        reasoning, response = text.split("</think>", 1)
        reasoning = reasoning.split("<think>")[-1]
        return reasoning.strip(), response.strip(), truncated
    return text.split("<think>")[-1].strip(), "", truncated


async def eval_cotcontrolqa_tinker(
    sampling_client: tinker.SamplingClient,
    system_prompt: str = "",
    config: Optional[utils.GenerateConfig] = None,
    save_dir: Optional[str | Path] = "results/tinker_cotcontrol",
    save_name: Optional[str] = None,
    dataset: str | Path = "gpqa",
    mode: str = "word_suppression",
    user_prompt_template: str = "{question}",
    prefix_messages: Optional[list[dict]] = None,
    seed: int = 0,
    domains: Optional[list[str]] = None,
    max_samples: Optional[int] = None,
    subsample_seed: Optional[int] = None,
    judge_model: str = "openai/gpt-5-mini",
    judge_concurrency: int = 200,
) -> dict:
    """Run CoT-Control-QA on a tinker sampling client. Same task assembly,
    grading, judge, summary, and result schema as eval_cotcontrolqa; only
    generation goes through tinker (with utils' on-disk generation cache)."""
    if mode not in MODES + ["all", "random"]:
        raise ValueError(f"Unknown mode {mode!r}, must be one of {MODES + ['all', 'random']}")
    if "{question}" not in user_prompt_template:
        raise ValueError("user_prompt_template must contain '{question}'")
    config = config or utils.GenerateConfig(temperature=0.0, max_tokens=12000)
    judge_config = JudgeConfig(temperature=1.0, max_tokens=10000, max_concurrency=judge_concurrency)

    info = utils.get_model_info(sampling_client)
    model_id = info.model_path or info.base_model
    model_name = sampling_client.get_tokenizer().name_or_path

    samples = load_dataset(dataset, mode, domains, max_samples, subsample_seed)
    if not samples:
        raise ValueError("No samples left after filtering")
    tasks = assign_tasks(samples, mode, seed)

    print(f"CoT-Control-QA (tinker): {model_id} | dataset={dataset} | mode={mode} | "
          f"{len(samples)} questions -> {len(tasks)} tasks x {config.num_samples} samples")

    user_prompts = [
        user_prompt_template.format(question=create_user_prompt(s, m, model_name))
        for s, m in tasks
    ]
    messages_list = [
        ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        + (prefix_messages or [])
        + [{"role": "user", "content": up}]
        for up in user_prompts
    ]

    generations = await utils.generate_async(sampling_client, messages_list, config)

    records = []
    for (sample, task_mode), user_prompt, messages, gen in zip(
        tasks, user_prompts, messages_list, generations
    ):
        target_letter = convert_answer_to_letter(sample["correct_answer"], sample["options"])
        control_value = get_control_value(task_mode, sample)
        requirement = (
            get_requirement_text(task_mode, control_value, model_name)
            if task_mode != "baseline"
            else None
        )

        rollouts = []
        for completion in gen["output"]:
            reasoning, response, truncated = split_reasoning(completion)
            extracted = extract_answer(response)
            rollouts.append(
                {
                    "output": response,
                    "reasoning": reasoning,
                    "reasoning_text_graded": reasoning,
                    "extracted_answer": extracted,
                    "correct": extracted == target_letter,
                    "compliance": grade_compliance(task_mode, reasoning, sample),
                    "meta_discussion": None,
                    "judge": {},
                    "finish_reason": "length" if truncated else "stop",
                    "usage": None,
                    "error": None,
                    "raw_completion": completion,
                }
            )

        records.append(
            {
                "id": sample["id"],
                "dataset": sample["dataset"],
                "mode": task_mode,
                "source": sample["source"],
                "domain": sample["domain"],
                "question": sample["question"],
                "options": sample["options"],
                "correct_answer_raw": sample["correct_answer"],
                "correct_answer_letter": target_letter,
                "keyword": sample.get("keyword"),
                "synonyms": sample.get("synonyms"),
                "control_value": control_value,
                "requirement": requirement,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "messages": messages,
                "tinker_input": gen["input"],
                "samples": rollouts,
            }
        )

    # ignore_question compliance via OpenRouter judge (triple-check)
    judge_targets = [
        (rec, rollout)
        for rec in records
        if rec["mode"] == "ignore_question"
        for rollout in rec["samples"]
    ]
    if judge_targets:
        print(f"Judging ignore_question compliance with {judge_model} "
              f"({len(judge_targets)} rollouts, triple-check)...")
        entries = [
            {"question": rec["question"], "reasoning": rollout["reasoning_text_graded"]}
            for rec, rollout in judge_targets
        ]
        judged = await _judge_ignore_question(entries, judge_model, judge_config)
        for (rec, rollout), j in zip(judge_targets, judged):
            rollout["compliance"] = j["compliance"]
            rollout["judge"]["ignore_question"] = {
                "model": judge_model,
                "outputs": j["judge_outputs"],
            }

    # Summary (same shape as the OpenRouter eval)
    all_rollouts = [rollout for rec in records for rollout in rec["samples"]]
    graded_correct = [r["correct"] for r in all_rollouts if r["correct"] is not None]
    graded_compliance = [r["compliance"] for r in all_rollouts if r["compliance"] is not None]
    n_truncated = sum(1 for r in all_rollouts if r["finish_reason"] == "length")

    def _group_stats(key):
        stats = {}
        for rec in records:
            d = stats.setdefault(rec[key], {"n": 0, "correct": 0, "compliant": 0})
            for r in rec["samples"]:
                d["n"] += 1
                d["correct"] += int(bool(r["correct"]))
                d["compliant"] += int(r["compliance"] == 1)
        return stats

    summary = {
        "n_questions": len(samples),
        "n_tasks": len(records),
        "num_samples_per_problem": config.num_samples,
        "n_rollouts": len(all_rollouts),
        "n_errors": 0,
        "n_truncated": n_truncated,
        "accuracy": sum(graded_correct) / len(graded_correct) if graded_correct else None,
        "compliance_rate": (
            sum(graded_compliance) / len(graded_compliance) if graded_compliance else None
        ),
        "meta_discussion_rate": None,
        "per_mode": _group_stats("mode"),
        "per_dataset": _group_stats("dataset"),
        "per_domain": _group_stats("domain"),
    }

    result = {
        "config": {
            "model": model_id,
            "tokenizer": model_name,
            "backend": "tinker",
            "dataset": str(dataset),
            "mode": mode,
            "system_prompt": system_prompt,
            "user_prompt_template": user_prompt_template,
            "prefix_messages": prefix_messages,
            "seed": seed,
            "domains": domains,
            "max_samples": max_samples,
            "subsample_seed": subsample_seed,
            "generate_config": dataclasses.asdict(config),
            "judge_model": judge_model,
            "judge_config": dataclasses.asdict(judge_config),
            "timestamp": datetime.now().isoformat(),
        },
        "summary": summary,
        "results": records,
    }

    print(f"[{model_id}] accuracy={summary['accuracy']}, "
          f"compliance_rate={summary['compliance_rate']}, "
          f"truncated={n_truncated}/{len(all_rollouts)}")

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = save_name or (
            f"{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}_"
            f"{model_id.replace('/', '_').replace(':', '_')}_{Path(str(dataset)).stem}_{mode}"
        )
        save_path = save_dir / f"{stem}.json"
        with open(save_path, "w") as f:
            json.dump(result, f, indent=1)
        print(f"Saved to {save_path}")

    return result


async def run_cotcontrol_evaluation(
    service_client: tinker.ServiceClient,
    paths: List[str],
    save_dir: str | Path = "results/tinker_cotcontrol",
    save_prefix: str = "cotcontrol",
    **eval_kwargs,
) -> Tuple[List[dict], List[dict]]:
    """Run the CoT-Control eval on multiple model paths in parallel.

    paths entries are either tinker checkpoint URLs ("tinker://...") or base
    model names ("Qwen/Qwen3-8B"). Each path's full result JSON is saved as
    <save_dir>/<save_prefix>_<path tail>.json. Remaining kwargs go to
    eval_cotcontrolqa_tinker (dataset, mode, config, system_prompt, ...).

    Returns (summaries, results): the per-path summary dicts (with "path"
    added) and the full result dicts, in input order.
    """

    async def evaluate_path(path: str) -> dict:
        if path.startswith("tinker://"):
            sc = service_client.create_sampling_client(model_path=path)
        else:
            sc = service_client.create_sampling_client(base_model=path)
        tail = path.rstrip("/").split("/")[-1].replace(":", "_")
        return await eval_cotcontrolqa_tinker(
            sc,
            save_dir=save_dir,
            save_name=f"{save_prefix}_{tail}",
            **eval_kwargs,
        )

    results = await asyncio.gather(*[evaluate_path(p) for p in paths])
    summaries = []
    for path, r in zip(paths, results):
        s = dict(r["summary"])
        s["path"] = path
        summaries.append(s)
        print(f"{path}: strict_compliance={s['compliance_rate']:.4f} accuracy={s['accuracy']:.4f}")
    return summaries, results
