#!/usr/bin/env python
"""Reward-delta heatmaps across all RL sweeps.

    python rl/sweep_heatmap.py

One panel per sweep folder; cells are lr x train_params. Cell value =
rolling-mean reward at the end minus at the start (window 5), i.e. how much
the run's reward moved. Second line: final rolling accuracy (the collapse
detector — a big reward delta with acc ~0 is reward hacking, not learning).
'*' marks runs that died early (<90% of configured iterations).

Writes rl/runs/reward_delta_heatmap.png and prints the matrices.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "rl" / "runs"

SWEEPS = [  # (title, folder, expected iterations)
    ("sweep1 qwen8b — rank32, 25it\ncompliance-only reward", "sweep1_qwen8b_lr-x-k_20260826", 25),
    ("sweep2 qwen8b — compliance-only", "sweep2_qwen8b", 50),
    ("sweep2 gpt-oss-20b — compliance-only", "sweep2_gptoss20b", 50),
    ("sweep3 qwen8b — anchored (λ=3)", "sweep3_qwen8b", 50),
    ("sweep3 gpt-oss-20b — anchored (λ=3)", "sweep3_gptoss20b", 50),
    ("sweep4 gpt-oss-20b — anchored (λ=2)\nlow-lr grid", "sweep4_gptoss20b", 50),
]

# Diverging: blue (reward up) <-> gray 0 <-> red (reward down); reward movement
# is polarity, not goodness, so status green/red is deliberately avoided.
CMAP = LinearSegmentedColormap.from_list("delta", ["#e34948", "#f0efec", "#1c5cab"])
TEXT, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
W = 5  # rolling window


def load_sweep(folder: Path):
    runs = {}
    for p in sorted(folder.glob("*/progress.jsonl")):
        cfg = yaml.safe_load((p.parent / "config.yaml").read_text())
        es = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if not es:
            continue
        r = [e["reward_mean"] for e in es]
        a = [e["accuracy"] for e in es]
        w = min(W, len(r))
        runs[(float(str(cfg["train"]["lr"])), int(cfg["elicitation"]["train_params"]))] = {
            "delta": sum(r[-w:]) / w - sum(r[:w]) / w,
            "acc": sum(a[-w:]) / w,
            "iters": len(es),
        }
    return runs


def fmt_lr(lr: float) -> str:
    return f"{lr:g}"


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    fig.patch.set_facecolor(SURFACE)
    axes = axes.ravel()
    if len(SWEEPS) < len(axes): axes[len(SWEEPS)].axis("off")

    for ax, (title, folder, expected) in zip(axes, SWEEPS):
        runs = load_sweep(RUNS / folder)
        lrs = sorted({lr for lr, _ in runs})
        ks = sorted({k for _, k in runs})
        M = np.full((len(lrs), len(ks)), np.nan)
        for (lr, k), v in runs.items():
            M[lrs.index(lr), ks.index(k)] = v["delta"]

        vmax = max(np.nanmax(np.abs(M)), 1e-9)
        ax.imshow(M, cmap=CMAP, norm=TwoSlopeNorm(0, -vmax, vmax),
                  aspect="auto", origin="lower")
        for i, lr in enumerate(lrs):
            for j, k in enumerate(ks):
                v = runs.get((lr, k))
                if v is None:
                    ax.text(j, i, "—", ha="center", va="center", color=MUTED)
                    continue
                star = "*" if v["iters"] < 0.9 * expected else ""
                deep = abs(v["delta"]) > 0.55 * vmax  # dark cell -> light ink
                ink, ink2 = ("#ffffff", "#e8e7e3") if deep else (TEXT, MUTED)
                ax.text(j, i + 0.12, f"{v['delta']:+.2f}{star}", ha="center",
                        va="center", fontsize=10, color=ink)
                ax.text(j, i - 0.24, f"acc {v['acc']:.2f}", ha="center",
                        va="center", fontsize=7.5, color=ink2)
        ax.set_xticks(range(len(ks)), [f"{k // 1000}k" for k in ks], fontsize=9)
        ax.set_yticks(range(len(lrs)), [fmt_lr(lr) for lr in lrs], fontsize=9)
        ax.set_xlabel("train_params", fontsize=9, color=MUTED)
        ax.set_ylabel("lr", fontsize=9, color=MUTED)
        ax.set_title(title, fontsize=10, color=TEXT)
        for s in ax.spines.values():
            s.set_visible(False)

        print(f"\n=== {title.splitlines()[0]} ===")
        for lr in reversed(lrs):
            cells = []
            for k in ks:
                v = runs.get((lr, k))
                cells.append("   —  " if v is None else
                             f"{v['delta']:+.2f}/{v['acc']:.2f}")
            print(f"  lr {fmt_lr(lr):>5}: " + "  ".join(cells))

    fig.suptitle("Reward change (rolling-5 end − start); small text = final accuracy; * = died early",
                 fontsize=11, color=TEXT, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = RUNS / "reward_delta_heatmap.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
