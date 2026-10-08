#!/usr/bin/env python3
"""Re-run ONLY the monitor phase of an existing monitorability run dir with a different monitor
model and/or monitor prompt template. Never touches monitor.jsonl.

    python3 monitorability/remonitor.py <run_dir> --tag primed \
        --monitor-template-file monitorability/promptsearch/monitors/primed_toy_math_science.txt
    python3 monitorability/remonitor.py <run_dir> --tag sonnet55 --monitor-model anthropic/claude-sonnet-5.5 \
        --instance-range 0:10

Reads <run_dir>/rollouts.jsonl (+ config.json for the hide-suffix setting), writes
<run_dir>/monitor__<tag>.jsonl (same schema as monitor.jsonl plus `monitor_template`), a progress log
<run_dir>/remonitor__<tag>.log with running cost, and <run_dir>/remonitor__<tag>.json with the settings.
Resumable: already-present (instance_id, x, sample_idx) keys are skipped.

Template override rules (simplest correct design): each --monitor-template-file replaces the OSS default
template whose file name is a suffix of its basename (primed_toy_math_science.txt -> toy_math_science.txt,
primed_sandbagging.txt -> sandbagging.txt), and therefore applies only to datasets whose default template that
is. For arbitrary file names use --monitor-template DEFAULT.txt=PATH. Datasets without an override keep the
default template. Overrides must use the same placeholders and keep the same A/B/C/D (or X/Y) output contract so
evals.parse_monitor keeps working.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "monitorability"))

from cotcontrol.inference.openrouter import GenerateConfig, generate_async  # noqa: E402
from evals import (DATASETS, MONITOR_LABELS, MONITOR_PROMPTS, build_monitor_prompt,  # noqa: E402
                   load_rows, parse_monitor)
from run_monitorability import Log, append_jsonl, key, read_jsonl, usage_cost  # noqa: E402

REQUIRED_PLACEHOLDERS = ("{optional_thinking_block}", "{optional_answer_block}")


def resolve_templates(files: list[str], pairs: list[str]) -> dict[str, tuple[str, str]]:
    """-> {default_template_name: (path, text)}."""
    out: dict[str, tuple[str, str]] = {}
    for f in files or []:
        name = Path(f).name
        matches = [d for d in MONITOR_LABELS if name.endswith(d)]
        if len(matches) != 1:
            sys.exit(f"cannot infer which default template {f} replaces (candidates {list(MONITOR_LABELS)}); "
                     f"use --monitor-template DEFAULT.txt={f}")
        out[matches[0]] = (f, Path(f).read_text())
    for pr in pairs or []:
        d, _, path = pr.partition("=")
        if d not in MONITOR_LABELS or not path:
            sys.exit(f"bad --monitor-template {pr!r}; expected one of {list(MONITOR_LABELS)}=PATH")
        out[d] = (path, Path(path).read_text())
    for d, (path, text) in out.items():
        missing = [ph for ph in REQUIRED_PLACEHOLDERS if ph not in text]
        if "{user_message}" not in text and "<USER MESSAGE>" not in text:
            missing.append("{user_message}")
        if missing:
            sys.exit(f"{path} is missing placeholders {missing}")
        if d == "sandbagging.txt" and "B, C, D" not in text:
            sys.exit(f"{path}: sandbagging overrides must keep the literal 'B, C, D' (substituted per row)")
    return out


async def remonitor(args, run_dir: Path, rows, templates, hide: str | None, log):
    out_path = run_dir / f"monitor__{args.tag}.jsonl"
    rowmap = {(r.instance_id, r.x): r for r in rows}
    done = {key(r) for r in read_jsonl(out_path)}
    todo = [r for r in read_jsonl(run_dir / "rollouts.jsonl")
            if r["y_valid"] and key(r) not in done and (r["instance_id"], r["x"]) in rowmap]
    total = len(done) + len(todo)
    log(f"remonitor[{args.tag}]: {len(done)} already done, {len(todo)} to run (total {total}) "
        f"model={args.monitor_model}")
    if not todo:
        return

    def monitor_um(um: str) -> str:  # same hide-suffix logic as run_monitorability.monitor_phase
        if hide and um.endswith(hide):
            return um[: -len(hide)].rstrip("\n")
        return um

    def tmpl_for(dataset: str) -> tuple[str, str | None]:
        d = DATASETS[dataset][2]
        if d in templates:
            return templates[d][0], templates[d][1]
        return str(MONITOR_PROMPTS / d), None

    prompts, tmpl_names = [], []
    for r in todo:
        name, text = tmpl_for(r["dataset"])
        prompts.append(build_monitor_prompt(rowmap[(r["instance_id"], r["x"])], monitor_um(r["user_message"]),
                                            r["reasoning"], r["completion"], template_text=text))
        tmpl_names.append(name)
    cfg = GenerateConfig(temperature=1.0, max_tokens=args.monitor_max_tokens, num_samples=1,
                         max_concurrency=args.concurrency, extra_body={"usage": {"include": True}})
    st = {"n": 0, "cost": 0.0, "invalid": 0, "err": 0, "calls": 0}

    def record(i: int, res: dict, attempt: int) -> bool:
        r = todo[i]
        letter, z = parse_monitor(r["dataset"], res["completion"])
        st["cost"] += usage_cost(res["usage"])
        st["calls"] += 1
        if letter is None and attempt == 1:
            return False
        st["invalid"] += letter is None
        st["err"] += res["error"] is not None
        append_jsonl(out_path, {
            "dataset": r["dataset"], "instance_id": r["instance_id"], "x": r["x"],
            "sample_idx": r["sample_idx"], "monitor_model": args.monitor_model,
            "monitor_template": tmpl_names[i], "monitor_prompt": prompts[i],
            "monitor_output": res["completion"], "monitor_reasoning": res["reasoning"],
            "monitor_usage": res["usage"], "monitor_error": res["error"], "monitor_letter": letter,
            "z": None if z is None else int(z), "z_valid": letter is not None, "n_attempts": attempt,
        })
        return True

    retry: list[int] = []

    def on_first(res):
        if not record(res["prompt_idx"], res, 1):
            retry.append(res["prompt_idx"])
        st["n"] += 1
        if st["n"] % 100 == 0 or st["n"] == len(todo):
            log(f"remonitor[{args.tag}]: {len(done) + st['n']}/{total} done, cost=${st['cost']:.4f} "
                f"(${st['cost'] / max(1, st['calls']):.5f}/call), unparsable_first_try={len(retry)}, "
                f"errors={st['err']}")

    wrap = lambda p: [{"role": "user", "content": p}]  # noqa: E731
    await generate_async([wrap(p) for p in prompts], args.monitor_model, cfg, on_result=on_first, progress=False)
    if retry:
        log(f"remonitor[{args.tag}]: retrying {len(retry)} unparsable outputs")
        res2 = await generate_async([wrap(prompts[i]) for i in retry], args.monitor_model, cfg, progress=False)
        for i, r in zip(retry, res2):
            record(i, {"completion": r["output"][0], "reasoning": r["reasoning"][0], **r["metadata"][0]}, 2)
    log(f"remonitor[{args.tag}]: finished, {st['calls']} calls, cost=${st['cost']:.4f} "
        f"(${st['cost'] / max(1, st['calls']):.5f}/call), invalid_z={st['invalid']}/{len(todo)}, errors={st['err']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--tag", required=True, help="output suffix: monitor__<tag>.jsonl")
    p.add_argument("--monitor-model", default=None, help="default: the run's config.json monitor_model")
    p.add_argument("--monitor-template-file", nargs="*", default=[],
                   help="override template(s); default replaced is inferred from the basename suffix")
    p.add_argument("--monitor-template", action="append", default=[], metavar="DEFAULT.txt=PATH",
                   help="explicit override pair, e.g. toy_math_science.txt=my_monitor.txt")
    p.add_argument("--monitor-max-tokens", type=int, default=None, help="default: config.json value or 8000")
    p.add_argument("--instance-range", default=None, help="a:b slice of instances per dataset")
    p.add_argument("--datasets", nargs="+", default=None, choices=list(DATASETS),
                   help="default: the run's datasets")
    p.add_argument("--concurrency", type=int, default=100)
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not (run_dir / "rollouts.jsonl").exists():
        sys.exit(f"no rollouts.jsonl in {run_dir}")
    config = json.load(open(run_dir / "config.json")) if (run_dir / "config.json").exists() else {}
    args.monitor_model = args.monitor_model or config.get("monitor_model") or "openai/gpt-5.6-luna"
    args.monitor_max_tokens = args.monitor_max_tokens or config.get("monitor_max_tokens") or 8000
    datasets = args.datasets or config.get("datasets") or sorted({r["dataset"] for r in read_jsonl(run_dir / "rollouts.jsonl")})
    templates = resolve_templates(args.monitor_template_file, args.monitor_template)
    hide = (config.get("user_suffix_text") or "").strip("\n") if config.get("hide_suffix_from_monitor") else None

    log = Log(run_dir / f"remonitor__{args.tag}.log")
    rng_ = tuple(int(v) for v in args.instance_range.split(":")) if args.instance_range else None
    rows = [r for ds in datasets for r in load_rows(ds, None, rng_)]
    used = {ds: (templates[DATASETS[ds][2]][0] if DATASETS[ds][2] in templates else "default") for ds in datasets}
    settings = {"run_dir": str(run_dir), "tag": args.tag, "monitor_model": args.monitor_model,
                "monitor_max_tokens": args.monitor_max_tokens, "datasets": datasets,
                "instance_range": args.instance_range, "templates": used, "hide_suffix_from_monitor": bool(hide),
                "concurrency": args.concurrency, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(settings, open(run_dir / f"remonitor__{args.tag}.json", "w"), indent=2)
    log(f"start tag={args.tag} model={args.monitor_model} datasets={datasets} range={args.instance_range} "
        f"templates={used} hide_suffix={bool(hide)} ({len(rows) // 2} instances)")
    asyncio.run(remonitor(args, run_dir, rows, templates, hide, log))


if __name__ == "__main__":
    main()
