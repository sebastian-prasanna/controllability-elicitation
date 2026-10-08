"""Build Arm-B SFT data for the extended constraint modes by string transform.

Same recipe as the "clean" arm of sft/data_generation.ipynb: take each model's
UNCONSTRAINED train-split traces (baselines/<label>_trainclean/), apply the mode's
transform from sft/transforms.py, pair the result with the constrained prompt for
that mode (rebuilt with the eval's own create_user_prompt), verify with the strict
grader, and save one JSON array per mode next to the existing nine:

    sft/training_data/<label>_clean/<mode>.json

Only Arm B is possible for new modes (there are no constrained baseline rollouts
for constraints that did not exist), which is also the arm without constraint
narration. Existing mode files are never overwritten unless --overwrite.

    python sft/build_extended_data.py                       # 4 models x EXTENDED_MODES
    python sft/build_extended_data.py --labels qwen8b --modes no_comma title_case
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sft"))

from cotcontrol.eval.data import assign_tasks, load_dataset  # noqa: E402
from cotcontrol.eval.grading import grade_compliance  # noqa: E402
from cotcontrol.eval.prompts import EXTENDED_MODES, create_user_prompt  # noqa: E402
from transforms import TRANSFORMS  # noqa: E402

MODELS = {
    "qwen8b":     {"model": "qwen/qwen3-8b",       "family": "qwen"},
    "qwen32b":    {"model": "qwen/qwen3-32b",      "family": "qwen"},
    "gptoss20b":  {"model": "openai/gpt-oss-20b",  "family": "gptoss"},
    "gptoss120b": {"model": "openai/gpt-oss-120b", "family": "gptoss"},
}
OUT_DIR = REPO / "sft" / "training_data"


def render_assistant(family: str, reasoning: str, output: str) -> dict:
    if family == "gptoss":
        return {"role": "assistant", "content": [
            {"type": "thinking", "thinking": reasoning},
            {"type": "text", "text": output},
        ]}
    return {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\n{output}"}


def load_clean(label: str) -> dict:
    """(dataset, id) -> usable unconstrained rollout (+ domain, for title_prefix)."""
    files = sorted(glob.glob(str(REPO / "baselines" / f"{label}_trainclean" / "2026-*.json")))
    assert files, f"no clean baseline for {label} - run run_baseline.py --mode baseline"
    d = json.loads(Path(files[-1]).read_text())
    assert d["config"]["split"] == "train" and d["config"]["mode"] == "baseline"
    out = {}
    for rec in d["results"]:
        s = rec["samples"][0]
        if s["error"] or not (s["reasoning"] or "").strip():
            continue
        if not s["output"] or not s["extracted_answer"]:
            continue
        out[(rec["dataset"], rec["id"])] = {
            "reasoning": s["reasoning"], "output": s["output"], "correct": s["correct"],
            "domain": rec.get("domain"), "source": files[-1],
        }
    return out


def build_prompts(modes: list[str]) -> dict:
    """label -> {(dataset, id, mode): constrained user prompt} via the eval's code path."""
    samples = load_dataset("all", "all", None, None, None, "train")
    tasks = assign_tasks(samples, "all", 0, allowed_modes=modes)
    return {label: {(s["dataset"], s["id"], m): create_user_prompt(s, m, cfg["model"])
                    for s, m in tasks} for label, cfg in MODELS.items()}


def build_mode_rows(label: str, mode: str, clean: dict, prompts: dict) -> tuple[list[dict], dict]:
    fn, family = TRANSFORMS[mode], MODELS[label]["family"]
    rows, stats = [], {"in": 0, "skipped": 0, "failed_grader": 0, "out": 0,
                       "chars_in": 0, "chars_out": 0}
    for (ds, qid), rec in sorted(clean.items()):
        prompt = prompts[label].get((ds, qid, mode))
        if prompt is None:
            continue
        stats["in"] += 1
        transformed = fn(rec["reasoning"], rec)
        if transformed is None:
            stats["skipped"] += 1
            continue
        if grade_compliance(mode, transformed, {"keyword": None, "synonyms": []}) != 1:
            stats["failed_grader"] += 1
            continue
        stats["chars_in"] += len(rec["reasoning"])
        stats["chars_out"] += len(transformed)
        rows.append({
            "input": [{"role": "user", "content": prompt}],
            "output": [render_assistant(family, transformed, rec["output"])],
            "meta": {
                "label": label, "model": MODELS[label]["model"], "split": "train",
                "dataset": ds, "id": qid, "mode": mode, "correct": rec["correct"],
                "arm": "clean", "transform": "programmatic", "source": rec["source"],
                "n_reasoning_chars": len(transformed), "n_output_chars": len(rec["output"]),
            },
        })
    stats["out"] = len(rows)
    return rows, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=list(MODELS))
    ap.add_argument("--modes", nargs="+", default=EXTENDED_MODES)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    unknown = [m for m in args.modes if m not in TRANSFORMS]
    if unknown:
        raise SystemExit(f"no transform for modes: {unknown}")

    prompts = build_prompts(args.modes)
    summary = {}
    for label in args.labels:
        clean = load_clean(label)
        out_dir = OUT_DIR / f"{label}_clean"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{label}: {len(clean)}/514 usable unconstrained traces -> {out_dir}")
        print(f"  {'mode':26s} {'in':>4s} {'out':>4s} {'skip':>5s} {'gfail':>5s} {'len ratio':>9s}")
        summary[label] = {}
        for mode in args.modes:
            f = out_dir / f"{mode}.json"
            if f.exists() and not args.overwrite:
                print(f"  {mode:26s} exists, skipping (use --overwrite)")
                continue
            rows, st = build_mode_rows(label, mode, clean, prompts)
            f.write_text(json.dumps(rows, indent=4) + "\n")
            ratio = st["chars_out"] / max(st["chars_in"], 1)
            print(f"  {mode:26s} {st['in']:4d} {st['out']:4d} {st['skipped']:5d} "
                  f"{st['failed_grader']:5d} {ratio:9.2f}")
            summary[label][mode] = st
    (OUT_DIR / "extended_build_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
