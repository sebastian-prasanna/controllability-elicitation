"""Held-out-split eval of trained checkpoints (lr x k sweeps, x320 arms).

    .venv/bin/python sft/test_eval.py sft/runs/lrk_gptoss20b [...] [--dry-run]
    .venv/bin/python sft/test_eval.py sft/runs/x320_gptoss20b_gepa_general_drive3 \
        --final-step 540 --blocks cotcontrol heldout [--split val --temperature 1.0 \
        --max-tokens 16000 --sampling-seed 0 --tag t1_16k]

Why this exists: every number reported from the sweeps is a val-selected maximum
(best checkpoint over several saves x several lrs, all on the same 200 val
questions), which is optimistically biased. The repo convention is val for
selection, test for reporting.

Selection (which checkpoints to evaluate):
  default        per (sweep, k) the best status=OK cell from lrk_summary.json at
                 its best_step, plus the best unmasked control (lrk sweeps).
  --final-step N every run dir in the sweep folder with a checkpoint at step N
                 (x320 arms: all 23 cells at step 540).
  --last-step    every run dir at its own final checkpoint (runs that stop at
                 slightly different step counts, e.g. sdf/runs/train 1-epoch runs);
                 --filter SUBSTR restricts --final-step/--last-step to matching names.
  --extra RUN_NAME:STEP adds checkpoints beyond either selection.

Blocks (--blocks, default: cotcontrol): named eval blocks from the run's own
config.yaml `eval:` section. Each inherits that block's mode pool /
allowed_modes / judge / engine settings; only the split and any explicit
overrides (--temperature, --max-tokens, --mode, --sampling-seed) change.
Legacy --heldout == --blocks heldout with the old artifact names.

Artifacts: eval/<subdir>/checkpoint-<step>.json inside each run dir, where
subdir = <split>[_<tag>][_<block>] (block suffix omitted for cotcontrol; the
legacy layout is test/ and heldout/). Plus <split>[_<tag>]_selection.json per
sweep folder (legacy: test_selection.json / heldout_selection.json). Every
entry keeps the flat test_compliance/test_accuracy keys for the first block
and a `blocks` dict with all blocks.

Run only after a sweep's cells are all DONE (default selection warns and skips a
sweep with cells still training, unless --allow-partial). --chunk N splits each
checkpoint's prompt list into N-prompt engine calls (fan-out + bounded call
duration for mode=all runs); the generation cache makes reruns resume.
"""

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval  # noqa: E402  (scripts/train_eval.py: config loading + eval plumbing)
from cotcontrol.eval.eval import eval_cotcontrolqa  # noqa: E402
from cotcontrol.inference import modal_vllm  # noqa: E402
from cotcontrol.inference.modal_vllm import make_generate_fn  # noqa: E402
from cotcontrol.run_logging import tee_stdio  # noqa: E402

DONE_STATUSES = {"OK", "STATIC", "COLLAPSE", "DEGRADED"}
HELDOUT_MODES = ["start_of_sentence", "letter_suppression", "no_spaces"]


def select_cells(summary: list[dict]) -> tuple[list[dict], list[str]]:
    """-> (selected cells, warnings). One best status=OK cell per k plus the
    best unmasked control, mirroring lrk_analysis.py's per-k-best rule."""
    selected, warnings = [], []
    pending = [c["run"] for c in summary if c["status"] not in DONE_STATUSES]
    if pending:
        warnings.append(f"cells not finished: {', '.join(pending)}")
    groups: dict[object, list[dict]] = {}
    for c in summary:
        if c["status"] in DONE_STATUSES:
            groups.setdefault("ALL" if c["unmasked"] else c["k"], []).append(c)
    for key, cells in sorted(groups.items(), key=lambda kv: str(kv[0])):
        ok = [c for c in cells if c["status"] == "OK"]
        if not ok:
            warnings.append(f"k={key}: no viable (OK) cell, skipped")
            continue
        selected.append(max(ok, key=lambda c: c["best_compliance"]))
    return selected, warnings


def select_final_step(sweep_dir: Path, step: int) -> tuple[list[tuple[Path, int]], list[str]]:
    """Every finished run dir in the folder that saved a checkpoint at `step`."""
    targets, warnings = [], []
    for run_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        tr = run_dir / "train_result.json"
        if not (run_dir / "config.yaml").exists():
            continue
        if not tr.exists():
            warnings.append(f"{run_dir.name}: no train_result.json (still training?), skipped")
            continue
        steps = {int(c["step"]) for c in json.loads(tr.read_text())["checkpoints"]}
        if step not in steps:
            warnings.append(f"{run_dir.name}: no checkpoint at step {step}, skipped")
            continue
        targets.append((run_dir, step))
    return targets, warnings


def select_last_step(sweep_dir: Path, name_filter: str | None) -> tuple[list[tuple[Path, int]], list[str]]:
    """Every finished run dir (optionally whose name contains `name_filter`) at its own
    final checkpoint (train_result final_step, else the max saved step) -- for sweeps
    whose runs stop at slightly different step counts (1-epoch runs)."""
    targets, warnings = [], []
    for run_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        if name_filter and name_filter not in run_dir.name:
            continue
        tr = run_dir / "train_result.json"
        if not (run_dir / "config.yaml").exists():
            continue
        if not tr.exists():
            warnings.append(f"{run_dir.name}: no train_result.json (still training?), skipped")
            continue
        res = json.loads(tr.read_text())
        step = res.get("final_step") or max(int(c["step"]) for c in res["checkpoints"])
        targets.append((run_dir, int(step)))
    return targets, warnings


def block_config(eval_cfg: dict, block: str) -> dict:
    """The run's own fully-defaulted eval block, or a sensible default when the
    run's config never had that block."""
    blocks = dict(train_eval.eval_blocks(eval_cfg))
    if block in blocks:
        return dict(blocks[block])
    if block == "heldout":
        return {**train_eval.DEFAULT_EVAL, "allowed_modes": HELDOUT_MODES}
    if block == "cotcontrol":
        return dict(blocks[next(iter(blocks))]) if blocks else dict(train_eval.DEFAULT_EVAL)
    raise ValueError(f"block {block!r} not in run config (have {sorted(blocks)})")


def chunked_generate_fn(base_model, gen_cfg, lora_path, chunk: int, sem: asyncio.Semaphore):
    """generate_fn that splits the prompt list into `chunk`-sized engine calls (0 = one call)
    and bounds concurrent engine calls with `sem` -- so --parallel == concurrent Modal engines
    regardless of how many checkpoints are in flight."""
    inner = make_generate_fn(base_model, gen_cfg, lora_path)

    async def one(part):
        async with sem:
            return await inner(part)

    async def fn(messages_list):
        parts = ([messages_list[i:i + chunk] for i in range(0, len(messages_list), chunk)]
                 if chunk else [messages_list])
        outs = await asyncio.gather(*[one(p) for p in parts])
        return [r for o in outs for r in o]

    return fn


async def eval_sweep(sweep_dir: Path, targets: list[tuple[Path, int]], parallel: int,
                     blocks: dict[str, str], split: str, overrides: dict,
                     sampling_seed: int | None, chunk: int) -> list[dict]:
    """Evaluate [(run_dir, step)] x blocks on `split`, one Modal app context for
    the whole sweep (same engine params -> warm container reuse).
    blocks: {block_name: save_subdir}. overrides: eval-block keys to replace
    (temperature, max_tokens, mode, ...)."""
    prepped = []
    for run_dir, step in targets:
        tc, eval_cfg, _ = train_eval.load_config(run_dir / "config.yaml")
        train_result = json.loads((run_dir / "train_result.json").read_text())
        tc.run_name = train_result["run_name"]
        ckpt = {c["step"]: c["path"] for c in train_result["checkpoints"]}.get(step)
        if ckpt is None:
            print(f"!! {run_dir.name}: no checkpoint at step {step}, skipping")
            continue
        for block, subdir in blocks.items():
            # Inherit the run's own eval settings (tp size, gpu, judge, mode pool,
            # question->mode seed) and change only the split + explicit overrides.
            ec = {**block_config(eval_cfg, block), **overrides, "split": split}
            prepped.append((run_dir, step, ckpt, tc, block, subdir, ec))

    sem = asyncio.Semaphore(parallel)   # concurrent engine calls (chunks), shared by all tasks

    async def eval_one(run_dir, step, ckpt, tc, block, subdir, ec):
        gen_cfg = train_eval.build_generate_config(tc, ec)
        if sampling_seed is not None:
            gen_cfg.seed = sampling_seed
        save_dir = run_dir / "eval" / subdir
        save_dir.mkdir(parents=True, exist_ok=True)
        result = await eval_cotcontrolqa(
            model=tc.base_model,
            system_prompt=train_eval.resolve_system_prompt(ec["system_prompt"]),
            generate_fn=chunked_generate_fn(tc.base_model, gen_cfg, ckpt, chunk, sem),
            save_dir=save_dir,
            save_name=f"checkpoint-{step}",
            dataset=ec["dataset"],
            mode=ec["mode"],
            allowed_modes=ec["allowed_modes"],
            seed=int(ec["seed"]),
            state_requirement=bool(ec["state_requirement"]),
            max_samples=ec["max_samples"],
            subsample_seed=ec["subsample_seed"],
            split=split,
            grade_meta_discussion=ec["grade_meta_discussion"],
            meta_discussion_scope=ec.get("meta_discussion_scope", "compliant"),
            judge_model=ec["judge_model"],
            judge_concurrency=int(ec["judge_concurrency"]),
            backend_info={"base_model": tc.base_model, "lora_path": ckpt,
                          "generate_config": asdict(gen_cfg)},
        )
        s = result["summary"]
        print(f"[{split}:{block}] {run_dir.name} step {step}: "
              f"compliance={s['compliance_rate']:.3f} accuracy={s['accuracy']:.3f} "
              f"errors={s['n_errors']}", flush=True)
        return {"run": run_dir.name, "step": step, "lora_path": ckpt, "block": block,
                "subdir": subdir, "compliance": s["compliance_rate"], "accuracy": s["accuracy"],
                "n_errors": s["n_errors"], "per_mode": s["per_mode"]}

    async with modal_vllm.app.run():
        return list(await asyncio.gather(*[eval_one(*p) for p in prepped]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweeps", nargs="+", help="sweep folders (e.g. sft/runs/lrk_gptoss20b)")
    ap.add_argument("--final-step", type=int, default=None,
                    help="select every run dir with a checkpoint at this step "
                         "(instead of the lrk_summary.json best-cell selection)")
    ap.add_argument("--last-step", action="store_true",
                    help="select every run dir at its own final checkpoint (train_result final_step)")
    ap.add_argument("--filter", default=None, metavar="SUBSTR",
                    help="with --final-step/--last-step: only run dirs whose name contains SUBSTR")
    ap.add_argument("--exclude", nargs="*", default=[], metavar="SUBSTR",
                    help="with --final-step/--last-step: drop run dirs whose name contains any SUBSTR")
    ap.add_argument("--extra", nargs="*", default=[], metavar="RUN_NAME:STEP",
                    help="additional checkpoints beyond the automatic selection")
    ap.add_argument("--blocks", nargs="+", default=None, metavar="BLOCK",
                    help="eval blocks from the run config to run (default: cotcontrol)")
    ap.add_argument("--heldout", action="store_true",
                    help="legacy: evaluate the held-out block only "
                         f"({', '.join(HELDOUT_MODES)}); writes heldout_selection.json / eval/heldout/")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--mode", default=None, help="constraint-mode assignment override (random | all | <mode>)")
    ap.add_argument("--tensor-parallel", type=int, default=None,
                    help="vLLM tensor_parallel_size override (engine gpu becomes <type>:<tp>)")
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="vLLM context override (needed when a run's block caps it below prompt + --max-tokens)")
    ap.add_argument("--sampling-seed", type=int, default=None, help="vLLM per-request sampling seed")
    ap.add_argument("--tag", default=None, help="artifact-name tag, e.g. t1_16k -> eval/val_t1_16k/")
    ap.add_argument("--chunk", type=int, default=0, help="prompts per engine call (0 = one call per checkpoint)")
    ap.add_argument("--parallel", type=int, default=4,
                    help="concurrent Modal engine calls (= GPUs) per sweep; with --chunk this bounds chunks")
    ap.add_argument("--allow-partial", action="store_true",
                    help="select from finished cells even if some are still training")
    ap.add_argument("--dry-run", action="store_true", help="print selections and exit")
    args = ap.parse_args()

    if args.heldout and args.blocks:
        ap.error("--heldout is the legacy form of --blocks heldout; use one")
    block_names = ["heldout"] if args.heldout else (args.blocks or ["cotcontrol"])
    prefix = args.split + (f"_{args.tag}" if args.tag else "")
    if args.heldout and not args.tag and args.split == "test":
        prefix = "heldout"  # legacy artifact names
        blocks = {"heldout": "heldout"}
    else:
        blocks = {b: prefix + ("" if b == "cotcontrol" else f"_{b}") for b in block_names}
    overrides = {k: v for k, v in (("temperature", args.temperature), ("max_tokens", args.max_tokens),
                                   ("mode", args.mode), ("max_model_len", args.max_model_len),
                                   ("tensor_parallel_size", args.tensor_parallel)) if v is not None}
    print(f"split={args.split} blocks={blocks} overrides={overrides} "
          f"sampling_seed={args.sampling_seed} chunk={args.chunk} -> {prefix}_selection.json")

    plan: list[tuple[Path, list[tuple[Path, int]], list[dict]]] = []
    for sweep in args.sweeps:
        sweep_dir = (ROOT / sweep).resolve() if not Path(sweep).is_absolute() else Path(sweep)
        selected: list[dict] = []
        if args.last_step:
            targets, warnings = select_last_step(sweep_dir, args.filter)
            for w in warnings:
                print(f"!! {sweep_dir.name}: {w}")
        elif args.final_step is not None:
            targets, warnings = select_final_step(sweep_dir, args.final_step)
            if args.filter:
                targets = [(d, st) for d, st in targets if args.filter in d.name]
            for w in warnings:
                print(f"!! {sweep_dir.name}: {w}")
        else:
            summary_path = sweep_dir / "lrk_summary.json"
            if not summary_path.exists():
                print(f"!! {sweep_dir}: no lrk_summary.json (run sft/lrk_analysis.py first, "
                      f"or use --final-step), skipping")
                continue
            selected, warnings = select_cells(json.loads(summary_path.read_text()))
            for w in warnings:
                print(f"!! {sweep_dir.name}: {w}")
            if warnings and any("not finished" in w for w in warnings) and not args.allow_partial:
                print(f"!! {sweep_dir.name}: skipped (use --allow-partial to override)")
                continue
            targets = [(sweep_dir / c["run"], c["best_step"]) for c in selected]
        if args.exclude:
            targets = [(d, st) for d, st in targets if not any(x in d.name for x in args.exclude)]
        for extra in args.extra:
            name, step = extra.rsplit(":", 1)
            if (sweep_dir / name).is_dir():
                targets.append((sweep_dir / name, int(step)))
        plan.append((sweep_dir, targets, selected))
        print(f"{sweep_dir.name}: {len(targets)} checkpoints x {len(blocks)} blocks selected")
        for run_dir, step in targets:
            print(f"    {run_dir.name} @ step {step}")

    if args.dry_run or not plan:
        return

    for sweep_dir, targets, selected in plan:
        # A --filter/--exclude subset run gets its own log + selection file so it can share a
        # folder with the full run (tee_stdio truncates; artifacts under eval/ are per run dir).
        fprefix = prefix + (("_subset" + hashlib.md5("|".join(sorted(d.name for d, _ in targets)).encode())
                             .hexdigest()[:6]) if (args.filter or args.exclude) else "")
        with tee_stdio(sweep_dir / f"{fprefix}_eval.log"):
            results = train_eval.with_slot_retry(
                lambda: asyncio.run(eval_sweep(sweep_dir, targets, args.parallel, blocks, args.split,
                                               overrides, args.sampling_seed, args.chunk)),
                f"{sweep_dir.name} {prefix} eval")
            by_run: dict[str, dict] = {}
            for r in results:
                by_run.setdefault(r["run"], {})[r["block"]] = r
            first = block_names[0]
            out = []
            for run_name in sorted(by_run, key=lambda n: [c["run"] for c in selected].index(n)
                                   if n in [c["run"] for c in selected] else len(selected)):
                blk = by_run[run_name]
                any_r = next(iter(blk.values()))
                entry = {"run": run_name, "step": any_r["step"], "lora_path": any_r["lora_path"],
                         "split": args.split, "overrides": overrides, "sampling_seed": args.sampling_seed}
                cell = next((c for c in selected if c["run"] == run_name), None)
                if cell is not None:
                    entry.update({k: cell[k] for k in ("k", "unmasked", "lr", "k_effective",
                                                       "best_step", "best_compliance", "acc_at_best")})
                    entry.update(val_compliance=cell["best_compliance"], val_accuracy=cell["acc_at_best"])
                if first in blk:  # flat legacy keys for the first block
                    entry.update(test_compliance=blk[first]["compliance"], test_accuracy=blk[first]["accuracy"],
                                 n_errors=blk[first]["n_errors"], per_mode=blk[first]["per_mode"])
                entry["blocks"] = {b: {k: r[k] for k in ("subdir", "compliance", "accuracy", "n_errors", "per_mode")}
                                   for b, r in blk.items()}
                out.append(entry)
            (sweep_dir / f"{fprefix}_selection.json").write_text(json.dumps(out, indent=2))
            print(f"Wrote {sweep_dir / f'{fprefix}_selection.json'}")


if __name__ == "__main__":
    main()
