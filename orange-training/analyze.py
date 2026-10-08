"""Aggregate orange-training runs into a table + plots.

Reads the full per-checkpoint eval JSONs (not just eval_summary.json) so it can
report the doc's caveats alongside the headline compliance number:

  compliance        strict rule-based compliance, the headline metric
  accuracy          answer correctness (Appendix C's capability-degradation axis)
  reasoning_chars   mean reasoning-trace length; the doc notes traces shorten
                    after training and that shorter traces are easier to control
  degenerate        fraction of traces below --degenerate-chars (e.g. a trace
                    that is just "meow" repeated, or a few tokens)
  compliance_long   compliance restricted to traces at least as long as the
                    untrained checkpoint's median — the doc's check that control
                    survives at full reasoning length

    .venv/bin/python orange-training/analyze.py
    .venv/bin/python orange-training/analyze.py --runs-dir orange-training/runs --plot
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_checkpoint_rows(runs_dir: Path) -> list[dict]:
    """One row per (run, condition, mode, checkpoint)."""
    rows = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        eval_root = run_dir / "eval"
        if not eval_root.is_dir():
            continue
        for block_dir in sorted(p for p in eval_root.iterdir() if p.is_dir()):
            # Block names are "<condition>_<mode>".
            condition, _, mode = block_dir.name.partition("_")
            for path in sorted(block_dir.glob("*.json")):
                step = path.stem.replace("checkpoint-", "")
                data = json.loads(path.read_text())
                rollouts = [s for rec in data["results"] for s in rec["samples"]]
                graded = [r for r in rollouts if not r["error"]]
                if not graded:
                    continue
                lengths = [len(r.get("reasoning_text_graded") or "") for r in graded]
                rows.append({
                    "run": run_dir.name,
                    "condition": condition,
                    "mode": mode,
                    "step": int(step) if step.isdigit() else step,
                    "n": len(graded),
                    "compliance": _mean(r["compliance"] == 1 for r in graded),
                    "accuracy": _mean(r["correct"] is True for r in graded),
                    "reasoning_chars": statistics.mean(lengths),
                    "median_chars": statistics.median(lengths),
                    "_per_rollout": [
                        {"compliant": r["compliance"] == 1, "chars": n}
                        for r, n in zip(graded, lengths)
                    ],
                })
    return rows


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def add_derived(rows: list[dict], degenerate_chars: int) -> None:
    """Fill in degenerate-trace fraction and length-matched compliance.

    The length bar is the untrained (step-0) median for the same
    (run, condition, mode), so each series is compared against its own
    pre-training reasoning length rather than a global constant."""
    baseline = {(r["run"], r["condition"], r["mode"]): r["median_chars"]
                for r in rows if r["step"] == 0}
    for row in rows:
        per = row.pop("_per_rollout")
        row["degenerate"] = _mean(p["chars"] < degenerate_chars for p in per)
        bar = baseline.get((row["run"], row["condition"], row["mode"]))
        long_traces = [p for p in per if bar is None or p["chars"] >= bar]
        row["n_long"] = len(long_traces)
        row["compliance_long"] = _mean(p["compliant"] for p in long_traces)


def print_table(rows: list[dict]) -> None:
    header = (f"{'run':<26} {'condition':<9} {'mode':<20} {'step':>5} "
              f"{'compl':>7} {'acc':>6} {'chars':>7} {'degen':>6} {'compl_long':>10}")
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda r: (r["run"], r["mode"], r["condition"],
                                         r["step"] if isinstance(r["step"], int) else -1)):
        print(f"{r['run']:<26} {r['condition']:<9} {r['mode']:<20} {str(r['step']):>5} "
              f"{r['compliance']:>7.3f} {r['accuracy']:>6.3f} {r['reasoning_chars']:>7.0f} "
              f"{r['degenerate']:>6.2f} {r['compliance_long']:>10.3f}")


def plot(rows: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # One column per model, one row per mode; conditions overlaid.
    models = sorted({r["run"].rsplit("-seed", 1)[0] for r in rows})
    modes = sorted({r["mode"] for r in rows})
    fig, axes = plt.subplots(len(modes), len(models), squeeze=False,
                             figsize=(6 * len(models), 3.6 * len(modes)))
    colors = {"persona": "C1", "direct": "C0", "stated": "C2"}

    for i, mode in enumerate(modes):
        for j, model in enumerate(models):
            ax = axes[i][j]
            for cond, color in colors.items():
                # Average over seeds at each step.
                by_step = defaultdict(list)
                for r in rows:
                    if (r["mode"] == mode and r["condition"] == cond
                            and r["run"].rsplit("-seed", 1)[0] == model
                            and isinstance(r["step"], int)):
                        by_step[r["step"]].append(r)
                if not by_step:
                    continue
                steps = sorted(by_step)
                ax.plot(steps, [_mean(x["compliance"] for x in by_step[s]) for s in steps],
                        "o-", color=color, label=f"{cond}: compliance")
                ax.plot(steps, [_mean(x["accuracy"] for x in by_step[s]) for s in steps],
                        "s--", color=color, alpha=0.45, label=f"{cond}: accuracy")
                if len(by_step[steps[0]]) > 1:  # seed spread
                    lo = [min(x["compliance"] for x in by_step[s]) for s in steps]
                    hi = [max(x["compliance"] for x in by_step[s]) for s in steps]
                    ax.fill_between(steps, lo, hi, color=color, alpha=0.15)
            ax.set_title(f"{model} — {mode}", fontsize=10)
            ax.set_xlabel("orange-SFT step")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default=str(HERE / "runs"))
    ap.add_argument("--degenerate-chars", type=int, default=200,
                    help="traces shorter than this count as degenerate")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()

    rows = load_checkpoint_rows(Path(args.runs_dir))
    if not rows:
        raise SystemExit(f"no eval JSONs under {args.runs_dir}")
    add_derived(rows, args.degenerate_chars)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints.json").write_text(json.dumps(rows, indent=2))
    print_table(rows)
    if args.plot:
        plot(rows, out_dir / "compliance.png")


if __name__ == "__main__":
    main()
