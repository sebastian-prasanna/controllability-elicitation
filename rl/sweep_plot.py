#!/usr/bin/env python
"""Plot per-iteration curves for every run in a sweep folder.

    python rl/sweep_plot.py rl/runs/sweep2_qwen8b [more sweep dirs...]

Layout: one row per metric, one column per train_params (k); one line per lr,
single-hue blue ramp light->dark = low->high lr (color follows the lr value,
consistent across panels). Saves <sweep_dir>/curves.png.

Run folders are discovered as subdirectories containing progress.jsonl; lr and
k are read from each run's config.yaml, so this works for any lr-x-k sweep
regardless of folder naming.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import yaml

matplotlib.use("Agg")

METRICS = ["reward_mean", "compliance_rate", "accuracy"]

# Ordinal blue ramp (light->dark), steps 250/350/450/550/700 of the reference
# sequential hue; validated for ordinal use on a light surface.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e8e7e3", "#fcfcfb"


def load_runs(sweep_dir: Path) -> list[dict]:
    runs = []
    for p in sorted(sweep_dir.glob("*/progress.jsonl")):
        cfg = yaml.safe_load((p.parent / "config.yaml").read_text())
        entries = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        if not entries:
            continue
        runs.append({
            "lr": float(str(cfg["train"]["lr"])),
            "k": int(cfg["elicitation"]["train_params"]),
            "entries": entries,
            "name": p.parent.name,
        })
    return runs


def fmt_lr(lr: float) -> str:
    return f"{lr:g}"


def fmt_k(k: int) -> str:
    return f"{k // 1000}k" if k >= 1000 else str(k)


def plot_sweep(sweep_dir: Path) -> Path:
    runs = load_runs(sweep_dir)
    if not runs:
        raise SystemExit(f"no runs with progress.jsonl under {sweep_dir}")

    lrs = sorted({r["lr"] for r in runs})
    ks = sorted({r["k"] for r in runs})
    if len(lrs) > len(RAMP):
        raise SystemExit(f"{len(lrs)} lrs > {len(RAMP)} ramp steps; extend RAMP")
    # Spread the lrs over the ramp ends so few series still get max separation.
    idxs = ([0] if len(lrs) == 1 else
            [round(i * (len(RAMP) - 1) / (len(lrs) - 1)) for i in range(len(lrs))])
    color = {lr: RAMP[i] for lr, i in zip(lrs, idxs)}

    fig, axes = plt.subplots(
        len(METRICS), len(ks), figsize=(3.6 * len(ks), 2.6 * len(METRICS)),
        sharex=True, sharey="row", squeeze=False,
    )
    fig.patch.set_facecolor(SURFACE)
    for mi, metric in enumerate(METRICS):
        for ki, k in enumerate(ks):
            ax = axes[mi][ki]
            ax.set_facecolor(SURFACE)
            for r in sorted(runs, key=lambda r: r["lr"]):
                if r["k"] != k:
                    continue
                xs = [e["iteration"] for e in r["entries"]]
                ys = [e.get(metric) for e in r["entries"]]
                ax.plot(xs, ys, color=color[r["lr"]], linewidth=2,
                        label=f"lr {fmt_lr(r['lr'])}" if mi == 0 and ki == 0 else None)
            ax.grid(True, color=GRID, linewidth=0.8)
            ax.tick_params(colors=MUTED, labelsize=8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            if mi == 0:
                ax.set_title(f"k = {fmt_k(k)}", fontsize=10, color=TEXT)
            if ki == 0:
                ax.set_ylabel(metric, fontsize=9, color=TEXT)
            if mi == len(METRICS) - 1:
                ax.set_xlabel("iteration", fontsize=9, color=MUTED)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(lrs), fontsize=8,
               frameon=False, labelcolor=TEXT)
    fig.suptitle(sweep_dir.name, fontsize=12, color=TEXT, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = sweep_dir / "curves.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for arg in sys.argv[1:]:
        out = plot_sweep(Path(arg))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
