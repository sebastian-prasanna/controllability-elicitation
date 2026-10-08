"""Held-out-MODE eval sweep over the 48 GEPA-optimized prompts.

(Distinct from eval_heldout.py, which evals on held-out *questions*.)

For every run dir under gepa/runs/{initial_sweep,second_sweep}/ that has a
config.json + best_prompt.txt, evals the best prompt with that run's task
model on the full test split (500 questions) x the 3 held-out modes
(start_of_sentence / letter_suppression / no_spaces) = 1500 tasks, mirroring
the run_gepa final-test setup (temperature 0, the run's max_tokens).

Per-run outputs go to <run_dir>/heldout_eval/ : the full eval JSON from
eval_cotcontrolqa (per-task prompts/outputs/reasoning/scores), progress.jsonl
(one line per rollout, for live monitoring), and summary.json. An aggregate
across all runs is rewritten to gepa/runs/heldout_modes_summary.json after
each run finishes.

Launches are staggered by --stagger seconds (default 120) to avoid slamming
OpenRouter all at once; per-run concurrency is capped at --concurrency
(default 100, so ~10 overlapping runs stay in the ~1000 aggregate sweet spot).
Runs whose heldout_eval/summary.json already exists are skipped, so the sweep
is resumable by re-running the script.

    .venv/bin/python gepa/eval_heldout_modes.py
    .venv/bin/python gepa/eval_heldout_modes.py --only initial_sweep/kimik3_general
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from cotcontrol.eval.eval import eval_cotcontrolqa  # noqa: E402
from cotcontrol.eval.prompts import HELDOUT_MODES  # noqa: E402
from cotcontrol.inference.openrouter import GenerateConfig  # noqa: E402

RUNS_DIR = Path(__file__).parent / "runs"
SWEEPS = ["initial_sweep", "second_sweep"]
AGGREGATE_PATH = RUNS_DIR / "heldout_modes_summary.json"


def discover_runs(only: list[str]) -> list[Path]:
    runs = []
    for sweep in SWEEPS:
        for d in sorted((RUNS_DIR / sweep).iterdir()):
            if d.is_dir() and (d / "config.json").exists() and (d / "best_prompt.txt").exists():
                runs.append(d)
    if only:
        runs = [r for r in runs if any(o in f"{r.parent.name}/{r.name}" for o in only)]
    return runs


def summarize(records: list[dict]) -> dict:
    """Flat + per-mode summary from eval_cotcontrolqa per-task records."""

    def _summ(recs):
        samples = [(r["mode"], s) for r in recs for s in r["samples"]]
        n = len(samples)
        return {
            "n_tasks": len(recs),
            "n_rollouts": n,
            "n_errors": sum(s["error"] is not None for _, s in samples),
            "strict_compliance": sum(s["compliance"] == 1 for _, s in samples) / n,
            "accuracy": sum(s["correct"] is True for _, s in samples) / n,
        }

    per_mode = {}
    for r in records:
        per_mode.setdefault(r["mode"], []).append(r)
    return {
        "overall": _summ(records),
        "per_mode": {m: _summ(rs) for m, rs in sorted(per_mode.items())},
    }


async def eval_one(run_dir: Path, concurrency: int) -> dict:
    cfg = json.loads((run_dir / "config.json").read_text())
    prompt = (run_dir / "best_prompt.txt").read_text()
    out_dir = run_dir / "heldout_eval"
    out_dir.mkdir(exist_ok=True)
    progress_path = out_dir / "progress.jsonl"
    t0 = time.time()

    def log_result(r):
        usage = r["usage"] or {}
        with open(progress_path, "a") as f:
            f.write(json.dumps({
                "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "secs": round(time.time() - t0, 1),
                "finish_reason": r["finish_reason"],
                "error": r["error"],
                "completion_tokens": usage.get("completion_tokens"),
                "prompt_idx": r["prompt_idx"],
            }) + "\n")

    result = await eval_cotcontrolqa(
        model=cfg["task_model"],
        system_prompt=prompt,
        generate_config=GenerateConfig(
            temperature=0.0,
            max_tokens=cfg["max_tokens"],
            max_concurrency=min(cfg.get("max_concurrency", 200), concurrency),
            provider=cfg.get("provider"),
        ),
        save_dir=out_dir,
        dataset="all",
        mode="all",
        allowed_modes=HELDOUT_MODES,
        seed=cfg["mode_seed"],
        split="test",
        judge_model=cfg["judge_model"],
        on_result=log_result,
    )

    summary = {
        "run": f"{run_dir.parent.name}/{run_dir.name}",
        "model": cfg["task_model"],
        "objective": cfg.get("objective"),
        "modes": HELDOUT_MODES,
        "split": "test",
        "wall_secs": round(time.time() - t0, 1),
        **summarize(result["results"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def write_aggregate(summaries: dict):
    AGGREGATE_PATH.write_text(json.dumps(
        {"updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "runs": summaries}, indent=1
    ))


async def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stagger", type=float, default=120, help="seconds between run launches")
    p.add_argument("--concurrency", type=int, default=100, help="per-run max concurrency cap")
    p.add_argument("--only", nargs="*", default=None,
                   help="substring filter(s) on '<sweep>/<run>' names")
    p.add_argument("--rerun", action="store_true", help="redo runs that already have a summary")
    args = p.parse_args()

    runs = discover_runs(args.only)
    done = {r for r in runs if (r / "heldout_eval" / "summary.json").exists()}
    todo = runs if args.rerun else [r for r in runs if r not in done]
    print(f"heldout-mode sweep: {len(runs)} runs discovered, {len(done)} already done, "
          f"{len(todo)} to run (stagger={args.stagger}s, per-run concurrency<={args.concurrency})",
          flush=True)

    summaries = {}
    for r in done - set(todo):
        summaries[f"{r.parent.name}/{r.name}"] = json.loads(
            (r / "heldout_eval" / "summary.json").read_text())

    async def runner(i: int, run_dir: Path):
        name = f"{run_dir.parent.name}/{run_dir.name}"
        await asyncio.sleep(i * args.stagger)
        print(f"[{time.strftime('%H:%M:%S')}] LAUNCH {i + 1}/{len(todo)}: {name}", flush=True)
        try:
            s = await eval_one(run_dir, args.concurrency)
            o = s["overall"]
            print(f"[{time.strftime('%H:%M:%S')}] DONE {name}: "
                  f"strict={o['strict_compliance']:.3f} acc={o['accuracy']:.3f} "
                  f"errors={o['n_errors']}/{o['n_rollouts']} ({s['wall_secs']:.0f}s)", flush=True)
            return name, s
        except Exception:
            print(f"[{time.strftime('%H:%M:%S')}] FAILED {name}:\n{traceback.format_exc()}",
                  flush=True)
            return name, {"run": name, "error": traceback.format_exc()}

    pending = {asyncio.create_task(runner(i, r)) for i, r in enumerate(todo)}
    while pending:
        finished, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for t in finished:
            name, s = t.result()
            summaries[name] = s
            write_aggregate(summaries)

    n_fail = sum(1 for s in summaries.values() if "error" in s)
    print(f"sweep complete: {len(summaries) - n_fail} ok, {n_fail} failed; "
          f"aggregate in {AGGREGATE_PATH}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
