"""Build per-mode SFT files from a single eval-run JSON by string transform.

Companion to sft/build_extended_data.py. That script builds the "clean" arm from
UNCONSTRAINED baseline traces shared across modes; this one builds the Arm-A style
from a run where every (question, mode) pair was sampled under ITS OWN constrained
prompt, e.g. a `run_baseline.py --mode all --allowed-modes ...` sweep.

Why it exists: such runs are often produced under an *elicitation* system prompt (a
GEPA-optimised one, say) to raise the hit rate of usable traces. That system prompt
is a generation-time device and must NOT be trained on -- otherwise the SFT input
distribution stops matching the eval, which sends a bare user turn. So the input is
rebuilt from scratch with the eval's own `create_user_prompt` and NO system message,
exactly matching the existing Arm-A files, and the run's system prompt is recorded in
meta.gen_system_prompt for provenance only.

    python sft/build_extended_from_run.py --run <run.json> --out-dir sft/training_data/gptoss20b

By default it writes only the modes that have no file in --out-dir yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sft"))

from cotcontrol.eval.data import assign_tasks, load_dataset  # noqa: E402
from cotcontrol.eval.grading import grade_compliance  # noqa: E402
from cotcontrol.eval.prompts import EXTENDED_MODES, create_user_prompt  # noqa: E402
from filters import output_answer_only, row_problems  # noqa: E402
from transforms import TRANSFORMS  # noqa: E402


def render_assistant(family: str, reasoning: str, output: str) -> dict:
    if family == "gptoss":
        return {"role": "assistant", "content": [
            {"type": "thinking", "thinking": reasoning},
            {"type": "text", "text": output},
        ]}
    return {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\n{output}"}


def usable(rec: dict, family: str, max_tokens: int | None,
           require_answer_only: bool = False) -> tuple[dict | None, str | None]:
    """The run's single rollout if it is fit to train on, else (None, reason).

    Beyond the basic completeness checks, filters.row_problems drops rollouts whose
    response is not exactly ``ANSWER: X`` (reasoning leaked into the output channel),
    that hit the generation length limit, that are repetition loops, or that would
    not fit the training window (see sft/filters.py, 2026-09-04 audit)."""
    s = (rec.get("samples") or [None])[0]
    if not s or s.get("error") or not (s.get("reasoning") or "").strip():
        return None, "empty_or_error"
    if not s.get("output") or not s.get("extracted_answer"):
        return None, "no_answer"
    problems = row_problems(s["reasoning"], s["output"], s.get("finish_reason"),
                            family=family, prompt=rec.get("user_prompt") or "",
                            max_tokens=max_tokens, require_answer_only=require_answer_only)
    if problems:
        return None, problems[0]
    return s, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="eval-run JSON with per-mode rollouts")
    ap.add_argument("--out-dir", required=True, help="e.g. sft/training_data/gptoss20b")
    ap.add_argument("--label", default=None, help="meta.label (default: out-dir name)")
    ap.add_argument("--family", default="gptoss", choices=["gptoss", "qwen"])
    ap.add_argument("--modes", nargs="+", default=None,
                    help="default: every EXTENDED_MODES entry with no file in --out-dir")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=16384,
                    help="drop rows whose prompt+reasoning+output exceed this many tokens "
                         "(TrainConfig.max_seq_length default); 0 disables")
    ap.add_argument("--require-answer-only", action="store_true",
                    help="drop rollouts whose response is not exactly 'ANSWER: X' (off by "
                         "default; meta.output_answer_only records it either way)")
    args = ap.parse_args()
    max_tokens = args.max_tokens or None

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or out_dir.name
    run = json.loads(Path(args.run).read_text())
    cfg, results = run["config"], run["results"]
    if cfg.get("split") not in (None, "train"):
        raise SystemExit(f"refusing to build training data from split={cfg['split']!r}")
    gen_sys = (cfg.get("system_prompt") or "")[:2000]

    modes = args.modes or [m for m in EXTENDED_MODES if not (out_dir / f"{m}.json").exists()]
    # Modes without a string transform (ignore_question, the suppression pair) are still
    # allowed: only rollouts that already comply are kept, verbatim ("natural positives").
    verbatim_only = [m for m in modes if m not in TRANSFORMS]
    if verbatim_only:
        print(f"no transform for {verbatim_only}: keeping compliant rollouts verbatim only")

    # Rebuild the clean (no-system-prompt) inputs from the eval's own code path.
    samples = load_dataset("all", "all", None, None, None, "train")
    tasks = assign_tasks(samples, "all", 0, allowed_modes=modes)
    model = cfg["model"]
    prompts = {(s["dataset"], s["id"], m): create_user_prompt(s, m, model) for s, m in tasks}

    by_mode: dict[str, list] = {}
    for rec in results:
        if rec["mode"] in modes:
            by_mode.setdefault(rec["mode"], []).append(rec)

    print(f"source: {args.run}\n  system prompt stripped ({len(cfg.get('system_prompt') or '')} chars)"
          f" -> inputs rebuilt with create_user_prompt, no system message")
    print(f"\n  {'mode':26s} {'recs':>5s} {'usable':>6s} {'out':>5s} {'skip':>5s} {'gfail':>5s} "
          f"{'noprompt':>8s} {'origOK':>6s} {'verbatim':>8s}")
    summary = {}
    for mode in modes:
        f = out_dir / f"{mode}.json"
        if f.exists() and not args.overwrite:
            print(f"  {mode:26s} exists, skipping (use --overwrite)")
            continue
        fn = TRANSFORMS.get(mode)
        rows, st = [], {"recs": 0, "usable": 0, "skipped": 0, "failed_grader": 0,
                        "no_prompt": 0, "orig_compliant": 0, "kept_verbatim": 0,
                        "dropped": {}}
        for rec in by_mode.get(mode, []):
            st["recs"] += 1
            s, why = usable(rec, args.family, max_tokens, args.require_answer_only)
            if s is None:
                st["dropped"][why] = st["dropped"].get(why, 0) + 1
                continue
            st["usable"] += 1
            st["orig_compliant"] += int(s.get("compliance") in (1, "1"))
            prompt = prompts.get((rec["dataset"], rec["id"], mode))
            if prompt is None:
                st["no_prompt"] += 1
                continue
            grade_sample = {"keyword": rec.get("keyword"), "synonyms": rec.get("synonyms")}
            # A rollout that already satisfies the constraint is kept VERBATIM: it is
            # fully on-policy, and re-applying a transform to it is what produced the
            # doubled targets ("[[word]]", '""...""') in the 2026-09-03 build. The run's
            # stored verdict is re-checked with the current grader; judge-graded modes
            # (grader returns None) fall back to the run's stored judge verdict.
            verdict = grade_compliance(mode, s["reasoning"], grade_sample)
            compliant = verdict == 1 or (verdict is None and s.get("compliance") in (1, "1"))
            if fn is None and not compliant:
                st["skipped"] += 1
                continue
            transformed = fn(s["reasoning"], rec) if fn is not None else s["reasoning"]
            if transformed is None:
                # The transform's skip rules apply to every row, compliant or not: e.g.
                # capitalize_and returns None when the trace never uses "and" - such a
                # trace passes the grader vacuously but would teach nothing.
                st["skipped"] += 1
                continue
            if compliant:
                transformed, transform_kind = s["reasoning"], "none"
                st["kept_verbatim"] += 1
            else:
                transform_kind = "programmatic"
                if grade_compliance(mode, transformed, grade_sample) != 1:
                    st["failed_grader"] += 1
                    continue
            rows.append({
                "input": [{"role": "user", "content": prompt}],
                "output": [render_assistant(args.family, transformed, s["output"])],
                "meta": {
                    "label": label, "model": model, "split": "train",
                    "dataset": rec["dataset"], "id": rec["id"], "mode": mode,
                    "correct": s.get("correct") in (True, "True"),
                    "orig_compliance": s.get("compliance"),
                    "output_answer_only": output_answer_only(s["output"]),
                    "finish_reason": s.get("finish_reason"),
                    "arm": "elicited", "transform": transform_kind,
                    "source": str(args.run), "gen_system_prompt": gen_sys,
                    "n_reasoning_chars": len(transformed), "n_output_chars": len(s["output"]),
                },
            })
        f.write_text(json.dumps(rows, indent=4) + "\n")
        st["out"] = len(rows)
        summary[mode] = st
        print(f"  {mode:26s} {st['recs']:5d} {st['usable']:6d} {st['out']:5d} {st['skipped']:5d} "
              f"{st['failed_grader']:5d} {st['no_prompt']:8d} {st['orig_compliant']:6d} "
              f"{st['kept_verbatim']:8d}  dropped={st['dropped']}")
    (out_dir / "extended_from_run_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {len(summary)} mode files -> {out_dir}")


if __name__ == "__main__":
    main()
