"""Dense-checkpoint replicas of the gpt-oss-120b early-spike cells.

    python sft/spike_plot.py

One panel per cell: dense replica (solid, 16 checkpoints over steps 0-100)
vs the original sparse run's checkpoints (open markers). Writes
sft/runs/spike120b/spike_curves{,_dark}.png.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "sft" / "runs"
CELLS = ["lrk-gptoss120b-k100k-lr3e-2", "lrk-gptoss120b-k100k-lr1e-2",
         "lrk-gptoss120b-kall-lr1e-4"]
THEME = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  grid="#d8d7d2", line="#2a78d6", acc="#eb6834"),
    "dark": dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
                 grid="#3a3a38", line="#3987e5", acc="#d95926"),
}


def ckpts(path: Path) -> list[tuple[int, float, float]]:
    s = json.loads(path.read_text())["checkpoints"]
    return [(c["step"], c["compliance_rate"], c["accuracy"]) for c in s
            if isinstance(c["step"], int)]


def render(mode: str, out: Path) -> None:
    th = THEME[mode]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    fig.patch.set_facecolor(th["surface"])
    for ax, cell in zip(axes, CELLS):
        ax.set_facecolor(th["surface"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(th["grid"])
        ax.tick_params(colors=th["secondary"], labelsize=8)
        ax.grid(True, color=th["grid"], alpha=0.45, linewidth=0.7)

        dense = ckpts(RUNS / "spike120b" / f"{cell}-dense" / "eval_summary.json")
        orig = [c for c in ckpts(RUNS / "lrk_gptoss120b" / cell / "eval_summary.json")
                if c[0] <= 100]
        ax.plot([c[0] for c in dense], [c[1] for c in dense], "-o", ms=5,
                color=th["line"], linewidth=2, markeredgecolor=th["surface"],
                markeredgewidth=1.2, label="dense replica: compliance")
        ax.plot([c[0] for c in dense], [c[2] for c in dense], "--", linewidth=1.4,
                color=th["acc"], alpha=0.85, label="dense replica: accuracy")
        ax.plot([c[0] for c in orig], [c[1] for c in orig], "o", ms=9,
                markerfacecolor="none", markeredgecolor=th["line"],
                markeredgewidth=1.6, label="original run: compliance")
        ax.set_title(cell.replace("lrk-gptoss120b-", ""), fontsize=10,
                     color=th["primary"], loc="left")
        ax.set_xlabel("optimizer step", fontsize=9, color=th["secondary"])
    axes[0].set_ylabel("val compliance / accuracy", fontsize=9, color=th["secondary"])
    axes[0].set_ylim(0, 0.6)
    axes[0].legend(fontsize=7.5, frameon=False, labelcolor=th["primary"])
    fig.suptitle("gpt-oss-120b early spike replicates: fast rise, collapse, slow recovery "
                 "(batch 16, so step 10 = 160 examples)",
                 fontsize=11.5, color=th["primary"], x=0.055, y=1.02, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=th["surface"])
    print(f"wrote {out}")


if __name__ == "__main__":
    render("light", RUNS / "spike120b" / "spike_curves.png")
    render("dark", RUNS / "spike120b" / "spike_curves_dark.png")
