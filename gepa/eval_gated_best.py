"""Test + held-out-mode evals for a GEPA run's *gated* best candidate.

After the trace-validity gate (grading.MIN_TRACE_ALPHA) the argmax candidate
changed for the degenerate qwen32b_free runs (gepa/analysis/validity_gate_regrade.md).
This evaluates a chosen candidate exactly like run_gepa's final test
(test split x all 9 modes, temp 0, run's max_tokens) and eval_heldout_modes
(test split x 3 held-out modes), writing to <run_dir>/gated_best_cand<N>/:
  config.json, best_prompt.txt, test_eval/, test_results.json,
  heldout_eval/{<eval>.json,progress.jsonl,summary.json}, progress_test.jsonl

    .venv/bin/python gepa/eval_gated_best.py initial_sweep/qwen32b_free:2 second_sweep/qwen32b_free_s2:3
"""
import asyncio, json, os, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "gepa")); os.chdir(ROOT)

from cotcontrol.eval.eval import eval_cotcontrolqa  # noqa: E402
from cotcontrol.eval.grading import compliance_score, shaped_compliance  # noqa: E402
from cotcontrol.inference.openrouter import GenerateConfig  # noqa: E402
from eval_heldout_modes import eval_one as heldout_eval_one  # noqa: E402


def _progress_logger(path: Path, t0: float):
    def log_result(r):
        usage = r["usage"] or {}
        with open(path, "a") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "secs": round(time.time() - t0, 1),
                                "finish_reason": r["finish_reason"], "error": r["error"],
                                "completion_tokens": usage.get("completion_tokens"),
                                "prompt_idx": r["prompt_idx"]}) + "\n")
    return log_result


def _tasks(result, objective):
    tasks = []
    for rec in result["results"]:
        s = rec["samples"][0]
        t = {"key": f"{rec['dataset']}:{rec['id']}:{rec['mode']}", "mode": rec["mode"],
             "keyword": rec.get("keyword"), "synonyms": rec.get("synonyms"),
             "compliance": s["compliance"], "correct": s["correct"],
             "reasoning": s["reasoning_text_graded"], "error": s["error"]}
        t["shaped_compliance"] = shaped_compliance(t)
        t["score"] = compliance_score(t) if objective == "compliance" else 0.5 * t["shaped_compliance"] + 0.5 * (t["correct"] is True)
        tasks.append(t)
    return tasks


def _summ(tasks):
    return {"n": len(tasks),
            "strict_compliance": sum(t["compliance"] == 1 for t in tasks) / len(tasks),
            "shaped_compliance": sum(t["shaped_compliance"] for t in tasks) / len(tasks),
            "accuracy": sum(t["correct"] is True for t in tasks) / len(tasks),
            "mean_score": sum(t["score"] for t in tasks) / len(tasks)}


async def test_eval(out: Path, cfg: dict, prompt: str, cand_id: int, concurrency: int):
    t0 = time.time()
    result = await eval_cotcontrolqa(
        model=cfg["task_model"], system_prompt=prompt,
        generate_config=GenerateConfig(temperature=0.0, max_tokens=cfg["max_tokens"],
                                       max_concurrency=min(cfg.get("max_concurrency", 200), concurrency)),
        save_dir=out / "test_eval", dataset="all", mode="all", seed=cfg["mode_seed"], split="test",
        judge_model=cfg["judge_model"], on_result=_progress_logger(out / "progress_test.jsonl", t0))
    tasks = _tasks(result, cfg.get("objective"))
    per_dataset, per_mode = {}, {}
    for t in tasks:
        per_dataset.setdefault(t["key"].split(":")[0], []).append(t)
        per_mode.setdefault(t["mode"], []).append(t)
    res = {"best_id": cand_id, "gated": True, "prompt": prompt, "wall_secs": round(time.time() - t0, 1),
           "overall": _summ(tasks),
           "per_dataset": {d: _summ(ts) for d, ts in sorted(per_dataset.items())},
           "per_mode": {m: _summ(ts) for m, ts in sorted(per_mode.items())}}
    (out / "test_results.json").write_text(json.dumps(res, indent=1))
    return res


async def run(spec: str, concurrency: int):
    run_name, cand_id = spec.rsplit(":", 1); cand_id = int(cand_id)
    run_dir = ROOT / "gepa/runs" / run_name
    cfg = json.loads((run_dir / "config.json").read_text())
    cand = [c for c in json.loads((run_dir / "candidates.json").read_text()) if c["id"] == cand_id][0]
    out = run_dir / f"gated_best_cand{cand_id}"; out.mkdir(exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "best_prompt.txt").write_text(cand["prompt"])
    (out / "candidate.json").write_text(json.dumps(cand, indent=1))
    print(f"[{time.strftime('%H:%M:%S')}] START {run_name} cand{cand_id} -> {out}", flush=True)
    results = {}
    for label, coro in (("test", test_eval(out, cfg, cand["prompt"], cand_id, concurrency)),
                        ("heldout", heldout_eval_one(out, concurrency))):
        try:
            r = await coro
            o = r["overall"]
            print(f"[{time.strftime('%H:%M:%S')}] DONE {run_name} cand{cand_id} {label}: "
                  f"strict={o['strict_compliance']:.3f} acc={o['accuracy']:.3f}", flush=True)
            results[label] = r
        except Exception:
            print(f"[{time.strftime('%H:%M:%S')}] FAILED {run_name} cand{cand_id} {label}:\n{traceback.format_exc()}", flush=True)
            results[label] = {"error": traceback.format_exc()}
    return spec, results


async def main():
    specs = [a for a in sys.argv[1:] if ":" in a]
    conc = 200
    out = await asyncio.gather(*(run(s, conc) for s in specs))
    print("ALL DONE", json.dumps({s: {k: (v.get("overall") or v.get("error", "")[:200]) for k, v in r.items()} for s, r in out}, indent=1), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
