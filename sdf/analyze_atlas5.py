"""Phase-2 (ATLAS-5) matrix summary: model x arm x eval block, identity off/on.

Per cell: strict compliance and "format-intact" compliance (compliant AND the final
output is non-empty and contains 'ANSWER:'), both as late-mean over checkpoints with
step >= --late (default 80) and as final checkpoint; unconstrained accuracy from the
baseline block. Reads sdf/runs/train/sdf-atlas5-<model>-<arm>/eval/<block>/checkpoint-*.json.

  .venv/bin/python sdf/analyze_atlas5.py [--late 80] [--md sdf/runs/train/results_atlas5.md]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["gptoss20b", "gptoss120b", "qwen8b", "qwen32b"]
ARMS = ["c4only", "c4only80k", "desc", "demo", "negative", "placebo"]
# run-name prefixes: v8 original recipe, v9 paper-like schedule (see make_configs.py); lr-bracket runs carry a suffix
PREFIXES = {"atlas5": "original recipe (8x16k packed, lr 1.5e-4)", "atlas5p": "paper-like (16x1k unpacked, lr 1e-5)"}
SUFFIXES = ["", "-lr3e-5"]
BLOCKS = ["cotcontrol", "heldout", "extended_unseen"]


def load_block(run: Path, block: str) -> dict[int, dict]:
    out = {}
    for f in (run / "eval" / block).glob("checkpoint-*.json"):
        step = int(f.stem.split("-")[1])
        res = json.loads(f.read_text())["results"]
        n = comp = fmt = acc = 0
        for x in res:
            for smp in x["samples"]:
                n += 1
                c = bool(smp.get("compliance") or smp.get("compliant"))
                comp += c
                fmt += c and "ANSWER:" in (smp.get("output") or "")
                acc += bool(smp.get("correct"))
        if n:
            out[step] = {"n": n, "compliance": comp / n, "fmt_intact": fmt / n, "accuracy": acc / n}
    return out


def summarize(by_step: dict[int, dict], late: int) -> dict:
    if not by_step:
        return {}
    steps = sorted(by_step)
    lat = [s for s in steps if s >= late] or steps[-1:]
    mean = lambda k, ss: sum(by_step[s][k] for s in ss) / len(ss)
    return {"final_step": steps[-1], "n_ckpt": len(steps),
            "compliance_late": mean("compliance", lat), "fmt_late": mean("fmt_intact", lat),
            "acc_late": mean("accuracy", lat), "compliance_final": by_step[steps[-1]]["compliance"],
            "fmt_final": by_step[steps[-1]]["fmt_intact"], "acc_final": by_step[steps[-1]]["accuracy"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="sdf/runs/train")
    ap.add_argument("--late", type=int, default=80)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    lines = [f"# ATLAS-5 matrix (val split; late-mean = checkpoints with step >= {args.late}; fmt = compliant AND 'ANSWER:' in final)\n"]
    table = {}
    for prefix, pdesc in PREFIXES.items():
      lines.append(f"\n# {pdesc}\n")
      for model in MODELS:
        for block in BLOCKS:
            rows = [(arm, sfx) for arm in ARMS for sfx in SUFFIXES
                    if (ROOT / args.root / f"sdf-{prefix}-{model}-{arm}{sfx}").exists()]
            if not rows:
                continue
            lines.append(f"\n## {model} — {block}  [{prefix}]\n")
            lines.append("| arm | ckpts | strict late / final | fmt-intact late / final | +identity: strict late / final | +identity: fmt late / final | unconstr. acc late (no id / id) |")
            lines.append("|---|---|---|---|---|---|---|")
            for arm, sfx in rows:
                run = ROOT / args.root / f"sdf-{prefix}-{model}-{arm}{sfx}"
                arm_label = arm + sfx
                model_key = model if prefix == "atlas5" else f"{model}@{prefix}"
                a = summarize(load_block(run, block), args.late)
                b = summarize(load_block(run, f"{block}_id"), args.late)
                base = summarize(load_block(run, "baseline"), args.late)
                base_id = summarize(load_block(run, "baseline_id"), args.late)
                table[(model_key, arm_label, block)] = {"noid": a, "id": b, "baseline": base, "baseline_id": base_id}
                f = lambda d, k: f"{d[k]:.3f}" if d and k in d else "—"
                lines.append(f"| {arm_label} | {a.get('n_ckpt', 0)} | {f(a, 'compliance_late')} / {f(a, 'compliance_final')} "
                             f"| {f(a, 'fmt_late')} / {f(a, 'fmt_final')} | {f(b, 'compliance_late')} / {f(b, 'compliance_final')} "
                             f"| {f(b, 'fmt_late')} / {f(b, 'fmt_final')} | {f(base, 'acc_late')} / {f(base_id, 'acc_late')} |")
    text = "\n".join(lines)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")
        (Path(args.md).with_suffix(".json")).write_text(json.dumps(
            {f"{m}|{a}|{b}": v for (m, a, b), v in table.items()}, indent=1))


if __name__ == "__main__":
    main()
