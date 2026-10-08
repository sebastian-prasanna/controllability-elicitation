#!/usr/bin/env python
"""Summarize a masked-LoRA lr x k SFT sweep as a k-by-lr grid.

    python sft/lrk_analysis.py sft/runs/lrk_gptoss20b [--block cotcontrol]

For each cell, reads the run's config.yaml (k, lr), train_result.json (effective
k from mask_verification, final loss) and eval_summary.json (per-checkpoint
compliance/accuracy on the eval block), then classifies the cell:

    STATIC    compliance never rose meaningfully above the checkpoint-0 baseline
    COLLAPSE  accuracy fell below COLLAPSE_ACC (reward-hacked / degenerate CoT)
    DEGRADED  accuracy kept >COLLAPSE_ACC but lost >DEGRADE_FRAC of baseline
    OK        compliance rose and accuracy held

The point of the grid is the anti-diagonal: if the (alpha/r)*lr*sqrt(k) scaling
law holds, the best cell should walk one lr rung per decade of k.

Writes <sweep_dir>/lrk_summary.{json,md} and prints the grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

COLLAPSE_ACC = 0.05     # accuracy at/below this = degenerate
DEGRADE_FRAC = 0.60     # keeping <60% of baseline accuracy = degraded
STATIC_MARGIN = 0.03    # compliance must beat baseline by this to count as moving
UNMASKED_K = 10 ** 12   # sentinel sorting the unmasked control last; shown as "ALL"


def lr_key(lr: str) -> float:
    return float(lr)


def load_cell(run_dir: Path, block: str) -> dict | None:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return None
    cfg = yaml.safe_load(cfg_path.read_text())
    # train_params: null = unmasked control (the whole adapter trains). Sorted
    # last in the grid and drawn as "ALL", since it is the ceiling the masked
    # cells are a fraction of, not another point on the k axis.
    raw_k = cfg.get("elicitation", {}).get("train_params")
    cell = {
        "run": run_dir.name,
        "k": UNMASKED_K if raw_k is None else int(raw_k),
        "unmasked": raw_k is None,
        "lr": str(cfg.get("train", {}).get("lr")),
        "status": "PENDING",
    }

    tr_path = run_dir / "train_result.json"
    if tr_path.exists():
        tr = json.loads(tr_path.read_text())
        cell["final_loss"] = tr.get("final_loss")
        cell["steps"] = tr.get("num_steps")
        mv = tr.get("mask_verification") or {}
        cell["k_effective"] = mv.get("n_changed_inside_mask")
        cell["mask_violations"] = mv.get("n_changed_outside_mask")

    log = run_dir / "train.log"
    if log.exists():
        txt = log.read_text(errors="ignore")
        # Modal occasionally re-executes the remote method after container churn
        # (retries are set to 0, so this is infra-level, not a configured retry).
        # Training then restarts from step 1 in the SAME checkpoint dir, so a
        # restarted cell's checkpoints can span two attempts — flag it.
        cell["worker_launches"] = txt.count("[modal] run_name=")

    es_path = run_dir / "eval_summary.json"
    if not es_path.exists():
        # still training? report last logged step
        log = run_dir / "train.log"
        if log.exists():
            import re
            steps = re.findall(r"\[step (\d+)/(\d+)\] loss=([\d.eE+-]+)", log.read_text())
            if steps:
                cell["status"] = f"TRAINING {steps[-1][0]}/{steps[-1][1]}"
        return cell

    es = json.loads(es_path.read_text())
    ckpts = es.get("evals", {}).get(block, {}).get("checkpoints") or es.get("checkpoints") or []
    ckpts = [c for c in ckpts if isinstance(c["step"], int) and c["compliance_rate"] is not None]
    if not ckpts:
        return cell

    base = min(ckpts, key=lambda c: c["step"])
    best = max(ckpts, key=lambda c: c["compliance_rate"])
    min_acc = min(c["accuracy"] for c in ckpts)
    cell |= {
        "base_compliance": base["compliance_rate"],
        "base_accuracy": base["accuracy"],
        "best_compliance": best["compliance_rate"],
        "best_step": best["step"],
        "acc_at_best": best["accuracy"],
        "min_accuracy": min_acc,
        "final_compliance": max(ckpts, key=lambda c: c["step"])["compliance_rate"],
        "n_checkpoints": len(ckpts),
        "trajectory": [(c["step"], round(c["compliance_rate"], 3), round(c["accuracy"], 3))
                       for c in sorted(ckpts, key=lambda c: c["step"])],
    }

    moved = best["compliance_rate"] > base["compliance_rate"] + STATIC_MARGIN
    if min_acc <= COLLAPSE_ACC:
        cell["status"] = "COLLAPSE"
    elif best["accuracy"] < DEGRADE_FRAC * base["accuracy"]:
        cell["status"] = "DEGRADED"
    elif moved:
        cell["status"] = "OK"
    else:
        cell["status"] = "STATIC"
    return cell


def render_grid(cells: list[dict]) -> str:
    ks = sorted({c["k"] for c in cells})
    lrs = sorted({c["lr"] for c in cells}, key=lr_key)
    by = {(c["k"], c["lr"]): c for c in cells}

    w = 22
    header = "k \\ lr"
    lines = ["", f"{header:>9} " + "".join(f"{lr:>{w}}" for lr in lrs)]
    for k in ks:
        row = f"{'ALL' if k == UNMASKED_K else k:>9} "
        for lr in lrs:
            c = by.get((k, lr))
            if c is None:
                row += f"{'-':>{w}}"
            elif c["status"].startswith(("PENDING", "TRAINING")):
                row += f"{c['status']:>{w}}"
            else:
                cell = f"{c['status']} c={c.get('best_compliance', 0):.2f}@{c.get('best_step', '?')}"
                row += f"{cell:>{w}}"
        lines.append(row)

    lines.append("")
    lines.append("cell = STATUS compliance@step (best checkpoint on the eval block)")
    done = [c for c in cells if c["status"] in ("OK", "STATIC", "COLLAPSE", "DEGRADED")]
    if done:
        lines.append("")
        lines.append("Per-k best viable cell (the anti-diagonal test):")
        for k in ks:
            label = "ALL" if k == UNMASKED_K else f"{k}"
            ok = [c for c in done if c["k"] == k and c["status"] == "OK"]
            if ok:
                b = max(ok, key=lambda c: c["best_compliance"])
                keff = b.get("k_effective")
                lines.append(
                    f"  k={label:>7} (effective {keff if keff is not None else '?'}): "
                    f"lr={b['lr']} -> compliance {b['base_compliance']:.3f} -> "
                    f"{b['best_compliance']:.3f} @step {b['best_step']}, "
                    f"accuracy {b['base_accuracy']:.3f} -> {b['acc_at_best']:.3f}")
            else:
                st = {c["lr"]: c["status"] for c in done if c["k"] == k}
                lines.append(f"  k={label:>7}: no viable cell — {st}")
        restarted = [c for c in done if (c.get("worker_launches") or 1) > 1]
        if restarted:
            lines.append("")
            lines.append("  !! RESTARTED cells (Modal re-ran training; checkpoints may "
                         "span attempts — re-check before trusting):")
            for c in restarted:
                lines.append(f"       {c['run']} ({c['worker_launches']} worker launches)")

        bad = [c for c in done if c.get("mask_violations")]
        if bad:
            lines.append("")
            lines.append(f"  !! MASK VIOLATIONS in {[c['run'] for c in bad]}")

        # Iso-drive check: cells sharing (alpha/r)*lr*sqrt(k) should behave alike
        # if that scalar is what governs. Compared at the LAST STEP ALL CELLS
        # REACHED, because best-checkpoint numbers land at different steps and
        # would otherwise confound drive with training length.
        common = set.intersection(*[{s for s, _, _ in c["trajectory"]}
                                    for c in done if c.get("trajectory")]) \
            if all(c.get("trajectory") for c in done) else set()
        if common:
            step = max(common)
            lines.append("")
            lines.append(f"Iso-drive comparison at the common step {step} "
                         f"(drive = lr*sqrt(k), alpha/r = 1 at rank 32):")
            rows = []
            for c in [c for c in done if not c.get("unmasked")]:
                at = {s: (cm, a) for s, cm, a in c["trajectory"]}[step]
                rows.append((lr_key(c["lr"]) * c["k"] ** 0.5, c["k"], c["lr"], at[0], at[1]))
            for drive, k, lr, cm, acc in sorted(rows):
                lines.append(f"  drive={drive:8.3f}  k={k:>7} lr={lr:>6}  "
                             f"compliance={cm:.3f}  accuracy={acc:.3f}")
            lines.append("  (equal drive + unequal compliance => k is a capacity "
                         "limit, not just an lr mismatch)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir")
    ap.add_argument("--block", default="cotcontrol", help="eval block name to read")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    cells = [c for c in (load_cell(d, args.block)
                         for d in sorted(sweep_dir.iterdir()) if d.is_dir()) if c]
    if not cells:
        raise SystemExit(f"no run dirs with a config.yaml under {sweep_dir}")

    grid = render_grid(cells)
    print(grid)
    (sweep_dir / "lrk_summary.json").write_text(json.dumps(cells, indent=2))
    (sweep_dir / "lrk_summary.md").write_text(
        f"# {sweep_dir.name}\n\n```{grid}\n```\n")
    print(f"\nWrote {sweep_dir / 'lrk_summary.json'} and .md")


if __name__ == "__main__":
    main()
