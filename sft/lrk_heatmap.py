"""Heatmap of the lr x k sweeps: best compliance per cell, one panel per model.

    python sft/lrk_heatmap.py sft/runs/lrk_gptoss20b sft/runs/lrk_qwen8b ...

Reads each sweep's lrk_summary.json (write it first with lrk_analysis.py).
Seed-replicate cells (run name containing '-seed') are excluded so every panel
is the canonical mask_seed-0 grid. Cell annotations carry status marks:
'x' = STATIC (never left baseline), a dagger = DEGRADED (compliance bought by
wrecking accuracy). Writes lrk_heatmap.png and lrk_heatmap_dark.png next to the
first sweep folder's parent (sft/runs/).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent

# Reference sequential blue ramp (steps 100 -> 700), lightest = near zero.
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
THEME = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  grid="#d8d7d2", cell_ink_dark="#0b0b0b", cell_ink_light="#ffffff"),
    "dark": dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
                 grid="#3a3a38", cell_ink_dark="#0b0b0b", cell_ink_light="#ffffff"),
}
K_ROWS = [1000, 10000, 100000, "ALL"]
K_LABEL = {1000: "1,000", 10000: "10,000", 100000: "100,000", "ALL": "unmasked"}
STATUS_MARK = {"STATIC": "x", "DEGRADED": "†", "COLLAPSE": "††"}


def lr_key(lr: str) -> float:
    return float(lr.replace("e", "E"))


def load_sweep(sweep_dir: Path) -> tuple[str, dict]:
    cells = json.loads((sweep_dir / "lrk_summary.json").read_text())
    grid = {}
    for c in cells:
        if "-seed" in c["run"] or "best_compliance" not in c:
            continue
        key = ("ALL" if c["unmasked"] else c["k"], c["lr"])
        if key not in grid or c["best_compliance"] > grid[key]["best_compliance"]:
            grid[key] = c
    name = sweep_dir.name.replace("lrk_", "")
    return name, grid


def render(sweeps: list[tuple[str, dict]], mode: str, out: Path) -> None:
    th = THEME[mode]
    # On dark the ramp inverts so near-zero recedes into the surface and the
    # maximum pops light, mirroring how the light ramp recedes toward white.
    ramp = RAMP if mode == "light" else ["#0e2138"] + RAMP[::-1]
    cmap = LinearSegmentedColormap.from_list("seq_blue", ramp)
    vmax = max(c["best_compliance"] for _, g in sweeps for c in g.values())
    lrs = sorted({lr for _, g in sweeps for (_, lr) in g}, key=lr_key)

    fig, axes = plt.subplots(1, len(sweeps), figsize=(4.7 * len(sweeps), 3.6))
    fig.subplots_adjust(top=0.78, wspace=0.28)
    fig.patch.set_facecolor(th["surface"])
    for ax, (name, grid) in zip(axes, sweeps):
        ax.set_facecolor(th["surface"])
        for spine in ax.spines.values():
            spine.set_visible(False)
        for yi, k in enumerate(K_ROWS):
            for xi, lr in enumerate(lrs):
                c = grid.get((k, lr))
                if c is None:
                    continue
                v = c["best_compliance"]
                frac = v / vmax
                # 2px gap between fills: shrink each cell inside its unit slot.
                ax.add_patch(plt.Rectangle((xi + 0.03, yi + 0.045), 0.94, 0.91,
                                           facecolor=cmap(frac), linewidth=0))
                if mode == "light":
                    ink = th["cell_ink_dark"] if frac < 0.55 else th["cell_ink_light"]
                else:
                    ink = th["cell_ink_light"] if frac < 0.55 else th["cell_ink_dark"]
                mark = STATUS_MARK.get(c["status"], "")
                label = f"{v:.2f}".lstrip("0") + mark  # ".05x" not "0.05x": fits the cell
                ax.text(xi + 0.5, yi + 0.5, label, ha="center", va="center",
                        fontsize=8, color=ink,
                        fontweight="bold" if c["status"] == "OK" else "normal")
        ax.set_xlim(0, len(lrs))
        ax.set_ylim(len(K_ROWS), 0)
        ax.set_xticks([i + 0.5 for i in range(len(lrs))])
        ax.set_xticklabels(lrs, fontsize=8, color=th["secondary"])
        ax.set_yticks([i + 0.5 for i in range(len(K_ROWS))])
        ax.set_yticklabels([K_LABEL[k] for k in K_ROWS], fontsize=8, color=th["secondary"])
        ax.tick_params(length=0)
        ax.set_title(name, fontsize=11, color=th["primary"], pad=8)
        ax.set_xlabel("learning rate", fontsize=8.5, color=th["secondary"])
        if ax is axes[0]:
            ax.set_ylabel("k (masked trainable scalars)", fontsize=8.5, color=th["secondary"])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.015)
    cbar.set_label("best compliance (val, 200 q)", fontsize=8.5, color=th["secondary"])
    cbar.ax.tick_params(labelsize=8, colors=th["secondary"])
    cbar.outline.set_visible(False)

    fig.suptitle("lr x k sweep: best-checkpoint compliance per cell "
                 "(x = static at baseline, † = degraded accuracy)",
                 fontsize=12, color=th["primary"], x=0.06, y=0.97, ha="left")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=th["surface"])
    print(f"wrote {out}")


def main() -> None:
    dirs = [Path(a) if Path(a).is_absolute() else ROOT / a for a in sys.argv[1:]]
    sweeps = [load_sweep(d) for d in dirs]
    out_dir = dirs[0].parent
    render(sweeps, "light", out_dir / "lrk_heatmap.png")
    render(sweeps, "dark", out_dir / "lrk_heatmap_dark.png")


if __name__ == "__main__":
    main()
