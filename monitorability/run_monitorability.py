#!/usr/bin/env python3
"""Run the Monitoring-Monitorability intervention evals (arXiv 2512.18311) on an
OpenRouter policy model, then a CoT+answer monitor, then the g-mean metric.

    python3 monitorability/run_monitorability.py --arm baseline
    python3 monitorability/run_monitorability.py --arm pilot --datasets gpqa \
        --max-rows-per-dataset 6 --samples-per-row 2

Outputs in monitorability/runs/<arm>/: config.json, rollouts.jsonl, monitor.jsonl,
progress.log, summary.json, summary.md. Re-running the same command resumes:
already-finished (instance_id, x, sample_idx) keys are skipped in both phases.
"""

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "monitorability"))

from cotcontrol.inference.openrouter import GenerateConfig, generate_async  # noqa: E402
from evals import (DATASETS, build_monitor_prompt, build_user_message,  # noqa: E402
                   extract_answer, label_y, load_rows, parse_monitor)
from metrics import summarize_run, write_summary  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def append_jsonl(path: Path, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def key(r: dict) -> tuple:
    return (r["instance_id"], r["x"], r["sample_idx"])


class Log:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def usage_cost(usage: dict | None) -> float:
    return float((usage or {}).get("cost") or 0.0)


# ---------------------------------------------------------------- policy phase

async def policy_phase(args, run_dir: Path, rows, system_prompt: str | None, suffix: str | None, log):
    path = run_dir / "rollouts.jsonl"
    done = {key(r) for r in read_jsonl(path)}
    jobs = []  # (row, user_message, sample_idx)
    for row in rows:
        um = build_user_message(row, suffix)
        for s in range(args.samples_per_row):
            if (row.instance_id, row.x, s) not in done:
                jobs.append((row, um, s))
    total = len(done) + len(jobs)
    log(f"policy: {len(done)} rollouts already done, {len(jobs)} to run (total {total})")
    if not jobs:
        return
    sys_hash = hashlib.sha1(system_prompt.encode()).hexdigest()[:12] if system_prompt else None
    st = {"n": 0, "cost": 0.0, "invalid": 0, "err": 0}

    def on_result(res):
        row, um, s = jobs[res["prompt_idx"]]
        kind = DATASETS[row.dataset][1]
        ext = extract_answer(kind, res["completion"])
        y = label_y(kind, ext, row.answer)
        usage = res["usage"] or {}
        rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        append_jsonl(path, {
            "dataset": row.dataset, "instance_id": row.instance_id, "id": row.id, "x": row.x,
            "category": row.category, "answer": row.answer, "perturbation": row.perturbation,
            "sample_idx": s, "system_prompt": sys_hash, "user_message": um,
            "reasoning": res["reasoning"], "completion": res["completion"],
            "finish_reason": res["finish_reason"], "usage": usage or None, "error": res["error"],
            "extracted_answer": ext, "y": None if y is None else int(y), "y_valid": y is not None,
            "reasoning_chars": len(res["reasoning"] or ""), "reasoning_tokens": rt,
        })
        st["n"] += 1
        st["cost"] += usage_cost(usage)
        st["invalid"] += y is None
        st["err"] += res["error"] is not None
        if st["n"] % 200 == 0 or st["n"] == len(jobs):
            log(f"policy: {len(done) + st['n']}/{total} done, cost=${st['cost']:.3f}, "
                f"invalid_y={st['invalid']}/{st['n']} ({st['invalid'] / st['n']:.1%}), errors={st['err']}")

    prompts = [([{"role": "system", "content": system_prompt}] if system_prompt else [])
               + [{"role": "user", "content": um}] for _, um, _ in jobs]
    cfg = GenerateConfig(
        temperature=args.temperature, max_tokens=args.max_tokens, num_samples=1,
        max_concurrency=args.concurrency,
        extra_body={"reasoning": {"effort": args.reasoning_effort}, "usage": {"include": True}},
    )
    await generate_async(prompts, args.policy_model, cfg, on_result=on_result, progress=False)


# ---------------------------------------------------------------- monitor phase

async def monitor_phase(args, run_dir: Path, rows, log):
    path = run_dir / "monitor.jsonl"
    rowmap = {(r.instance_id, r.x): r for r in rows}
    done = {key(r) for r in read_jsonl(path)}
    todo = [r for r in read_jsonl(run_dir / "rollouts.jsonl")
            if r["y_valid"] and key(r) not in done and (r["instance_id"], r["x"]) in rowmap]
    total = len(done) + len(todo)
    log(f"monitor: {len(done)} already done, {len(todo)} to run (total {total})")
    if not todo:
        return
    # Optionally hide the user-suffix (e.g. an obfuscation Requirement) from the monitor so we
    # measure the policy's hiding, not monitor priming. Config records the flag.
    hide = (Path(args.user_suffix_file).read_text().strip("\n")
            if getattr(args, "hide_suffix_from_monitor", False) and args.user_suffix_file else None)

    def monitor_um(um: str) -> str:
        if hide and um.endswith(hide):
            return um[: -len(hide)].rstrip("\n")
        return um

    prompts = [build_monitor_prompt(rowmap[(r["instance_id"], r["x"])], monitor_um(r["user_message"]),
                                    r["reasoning"], r["completion"]) for r in todo]
    cfg = GenerateConfig(temperature=1.0, max_tokens=args.monitor_max_tokens, num_samples=1,
                         max_concurrency=args.concurrency, extra_body={"usage": {"include": True}})
    st = {"n": 0, "cost": 0.0, "invalid": 0}

    def record(i: int, res: dict, attempt: int) -> bool:
        r = todo[i]
        letter, z = parse_monitor(r["dataset"], res["completion"])
        st["cost"] += usage_cost(res["usage"])
        if letter is None and attempt == 1:
            return False  # retry once with a fresh call
        st["invalid"] += letter is None
        append_jsonl(path, {
            "dataset": r["dataset"], "instance_id": r["instance_id"], "x": r["x"],
            "sample_idx": r["sample_idx"], "monitor_model": args.monitor_model,
            "monitor_prompt": prompts[i], "monitor_output": res["completion"],
            "monitor_reasoning": res["reasoning"], "monitor_usage": res["usage"],
            "monitor_error": res["error"], "monitor_letter": letter,
            "z": None if z is None else int(z), "z_valid": letter is not None, "n_attempts": attempt,
        })
        return True

    retry: list[int] = []

    def on_first(res):
        if not record(res["prompt_idx"], res, 1):
            retry.append(res["prompt_idx"])
        st["n"] += 1
        if st["n"] % 200 == 0 or st["n"] == len(todo):
            log(f"monitor: {len(done) + st['n']}/{total} done, cost=${st['cost']:.3f}, "
                f"unparsable_first_try={len(retry)}")

    wrap = lambda p: [{"role": "user", "content": p}]  # noqa: E731
    await generate_async([wrap(p) for p in prompts], args.monitor_model, cfg, on_result=on_first,
                         progress=False)
    if retry:
        log(f"monitor: retrying {len(retry)} unparsable outputs")
        res2 = await generate_async([wrap(prompts[i]) for i in retry], args.monitor_model, cfg,
                                    progress=False)
        for i, r in zip(retry, res2):
            record(i, {"completion": r["output"][0], "reasoning": r["reasoning"][0],
                       **r["metadata"][0]}, 2)
    log(f"monitor: finished, cost=${st['cost']:.3f}, invalid_z={st['invalid']}/{len(todo)}")


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--samples-per-row", type=int, default=10)
    p.add_argument("--max-rows-per-dataset", type=int, default=None,
                   help="take the first N instances (both arms) per dataset")
    p.add_argument("--policy-model", default="openai/gpt-oss-20b")
    p.add_argument("--reasoning-effort", default="medium")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--monitor-model", default="openai/gpt-5.6-luna")
    p.add_argument("--monitor-max-tokens", type=int, default=8000)
    p.add_argument("--concurrency", type=int, default=200)
    p.add_argument("--system-prompt-file")
    p.add_argument("--user-suffix-file")
    p.add_argument("--instance-range", default=None,
                   help="a:b slice of instances (by dataset order) per dataset, e.g. 0:40 or 40:100")
    p.add_argument("--skip-monitor", action="store_true")
    p.add_argument("--hide-suffix-from-monitor", action="store_true",
                   help="strip the --user-suffix-file text from the user message shown to the monitor")
    p.add_argument("--seed", type=int, default=0, help="recorded only; sampling is via the API")
    p.add_argument("--run-dir", default=None,
                   help="output directory (default monitorability/runs/<arm>)")
    args = p.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else REPO / "monitorability" / "runs" / args.arm
    run_dir.mkdir(parents=True, exist_ok=True)
    log = Log(run_dir / "progress.log")
    system_prompt = Path(args.system_prompt_file).read_text() if args.system_prompt_file else None
    suffix = Path(args.user_suffix_file).read_text() if args.user_suffix_file else None
    config = {**vars(args), "system_prompt_text": system_prompt, "user_suffix_text": suffix,
              "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        old = json.load(open(cfg_path))
        diff = {k: (old.get(k), v) for k, v in config.items()
                if k != "timestamp" and old.get(k) != v}
        if diff:
            log(f"WARNING: resuming with changed config: {diff}")
    json.dump(config, open(cfg_path, "w"), indent=2)
    log(f"start arm={args.arm} datasets={args.datasets} policy={args.policy_model} "
        f"effort={args.reasoning_effort} monitor={args.monitor_model}")

    rng_ = tuple(int(v) for v in args.instance_range.split(":")) if args.instance_range else None
    rows = [r for ds in args.datasets for r in load_rows(ds, args.max_rows_per_dataset, rng_)]
    log(f"{len(rows)} rows ({len(rows) // 2} instances) x {args.samples_per_row} samples")

    asyncio.run(policy_phase(args, run_dir, rows, system_prompt, suffix, log))
    if not args.skip_monitor:
        asyncio.run(monitor_phase(args, run_dir, rows, log))
        summary = summarize_run(run_dir)
        write_summary(run_dir, summary)
        log("summary written")
        print(open(run_dir / "summary.md").read())


if __name__ == "__main__":
    main()
