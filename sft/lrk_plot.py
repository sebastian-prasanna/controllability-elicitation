#!/usr/bin/env python
"""Plot a masked-LoRA lr x k sweep: the elicitation curve and its lr dependence.

    python sft/lrk_analysis.py sft/runs/lrk_gptoss20b   # writes lrk_summary.json
    python sft/lrk_plot.py     sft/runs/lrk_gptoss20b   # reads it, writes the PNGs

Three panels, each with a single y-axis (never a second scale):

  A  Elicitation curve — best compliance vs k (log x), against the unmasked
     ceiling. Emphasis form: one hue for the curve, gray for the ceiling.
     Filled marker = a viable cell; open marker = nothing beat baseline at that k.
  B  Compliance vs drive = (alpha/rank)*lr*sqrt(k), one line per k. Shows whether
     the optimum sits at a common drive across k (tuning) while the achievable
     level differs (capacity).
  C  Accuracy/compliance frontier — what each point of compliance costs.

Colors are the validated 3-slot categorical palette (see the dataviz skill's
references/palette.md); aqua is below 3:1 on the light surface, so every series
is direct-labeled, which is also what keeps identity off color alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

UNMASKED_K = 10 ** 12

THEME = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  grid="#d8d7d2", ceiling="#8a8983",
                  series=("#2a78d6", "#eb6834", "#1baf7a")),
    # Dark is selected, not an inverted light: its own steps from the same ramps,
    # validated against the dark surface.
    "dark": dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
                 grid="#3a3a38", ceiling="#8a8983",
                 series=("#3987e5", "#d95926", "#199e70")),
}


def drive(cell: dict, alpha_over_rank: float = 1.0) -> float:
    return alpha_over_rank * float(cell["lr"]) * cell["k"] ** 0.5


def style_axes(ax, t: str) -> None:
    th = THEME[t]
    ax.set_facecolor(th["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(th["grid"])
    ax.tick_params(colors=th["secondary"], labelsize=8, length=3)
    ax.grid(True, color=th["grid"], alpha=0.45, linewidth=0.7)
    ax.set_axisbelow(True)


def plot(cells: list[dict], out: Path, t: str, title: str) -> None:
    th = THEME[t]
    masked = [c for c in cells if not c.get("unmasked") and "best_compliance" in c]
    control = [c for c in cells if c.get("unmasked") and "best_compliance" in c]
    if not masked:
        print("no completed masked cells yet — nothing to plot")
        return

    ks = sorted({c["k"] for c in masked})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    fig.patch.set_facecolor(th["surface"])
    for ax in axes:
        style_axes(ax, t)

    ok_control = [c for c in control if c["status"] == "OK"]
    ceiling = max((c["best_compliance"] for c in ok_control), default=None)
    degraded_control = [c for c in control if c["status"] != "OK"]

    # ---- A: elicitation curve -------------------------------------------------
    ax = axes[0]
    xs, ys, viable = [], [], []
    for k in ks:
        grp = [c for c in masked if c["k"] == k]
        ok = [c for c in grp if c["status"] == "OK"]
        best = max(ok or grp, key=lambda c: c["best_compliance"])
        xs.append(k)
        ys.append(best["best_compliance"])
        viable.append(bool(ok))
    if ceiling is not None:
        ax.axhline(ceiling, color=th["ceiling"], linestyle=(0, (5, 3)), linewidth=1.6)
        ax.text(xs[0], ceiling, f" unmasked ceiling {ceiling:.2f}", va="bottom",
                ha="left", fontsize=8, color=th["secondary"])
    ax.plot(xs, ys, "-", color=th["series"][0], linewidth=2, zorder=2)
    for x, y, ok in zip(xs, ys, viable):
        ax.plot([x], [y], "o", ms=8, zorder=3, color=th["series"][0],
                markerfacecolor=th["series"][0] if ok else th["surface"],
                markeredgecolor=th["series"][0], markeredgewidth=2)
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=8, color=th["primary"])
    ax.set_xscale("log")
    ax.set_xlabel("k (trainable adapter scalars)", fontsize=9, color=th["secondary"])
    ax.set_ylabel("strict compliance (val)", fontsize=9, color=th["secondary"])
    ax.set_title("A · Elicitation curve", fontsize=10, color=th["primary"], loc="left")
    ax.set_ylim(0, max(max(ys), ceiling or 0) * 1.35 + 0.02)
    foot = "open marker = no viable cell"
    if degraded_control:
        foot += "\nhotter unmasked lr scores higher only by losing accuracy"
    ax.text(0.99, 0.03, foot, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color=th["secondary"])

    # ---- B: compliance vs drive ----------------------------------------------
    ax = axes[1]
    for i, k in enumerate(ks):
        grp = sorted((c for c in masked if c["k"] == k), key=drive)
        color = th["series"][i % len(th["series"])]
        dx = [drive(c) for c in grp]
        dy = [c["best_compliance"] for c in grp]
        label = f"k={k:,}"
        ax.plot(dx, dy, "-o", color=color, linewidth=2, ms=8, label=label,
                markeredgecolor=th["surface"], markeredgewidth=2, zorder=3 - i * 0.1)
        # direct label at the series' rightmost point (relief for the low-contrast slot)
        ax.annotate(f" {label}", (dx[-1], dy[-1]), fontsize=8, va="center",
                    ha="left", color=th["primary"])
    ax.set_xscale("log")
    ax.set_xlabel(r"drive = ($\alpha$/rank)·lr·$\sqrt{k}$", fontsize=9, color=th["secondary"])
    ax.set_ylabel("strict compliance (val)", fontsize=9, color=th["secondary"])
    ax.set_title("B · Compliance vs drive", fontsize=10, color=th["primary"], loc="left")
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi * 2.4)   # room for the direct labels
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left")
    for txt in leg.get_texts():
        txt.set_color(th["secondary"])

    # ---- C: accuracy / compliance frontier -----------------------------------
    ax = axes[2]
    for i, k in enumerate(ks):
        grp = sorted((c for c in masked if c["k"] == k), key=drive)
        color = th["series"][i % len(th["series"])]
        ax.plot([c["best_compliance"] for c in grp], [c["acc_at_best"] for c in grp],
                "-o", color=color, linewidth=2, ms=8, markeredgecolor=th["surface"],
                markeredgewidth=2, zorder=3 - i * 0.1)
        last = grp[-1]
        ax.annotate(f" k={k:,}", (last["best_compliance"], last["acc_at_best"]),
                    fontsize=8, va="center", ha="left", color=th["primary"])
    for c in control:
        ax.plot([c["best_compliance"]], [c["acc_at_best"]], "D", ms=8,
                color=th["ceiling"], markeredgecolor=th["surface"], markeredgewidth=2)
        tag = f" unmasked {c['lr']}" + ("" if c["status"] == "OK" else "*")
        ax.annotate(tag, (c["best_compliance"], c["acc_at_best"]), fontsize=8,
                    va="center", ha="left", color=th["secondary"])
    base_acc = max(c["base_accuracy"] for c in masked)
    ax.axhline(base_acc, color=th["ceiling"], linestyle=(0, (5, 3)), linewidth=1.6)
    ax.text(0.99, base_acc, f"base accuracy {base_acc:.2f} ", transform=ax.get_yaxis_transform(),
            va="top", ha="right", fontsize=8, color=th["secondary"])
    ax.margins(x=0.34)
    ax.set_xlabel("strict compliance (val)", fontsize=9, color=th["secondary"])
    ax.set_ylabel("accuracy (val)", fontsize=9, color=th["secondary"])
    ax.set_title("C · What compliance costs", fontsize=10, color=th["primary"], loc="left")
    if degraded_control:
        ax.text(0.01, 0.03, "* accuracy degraded", transform=ax.transAxes, ha="left",
                fontsize=7.5, color=th["secondary"])

    fig.suptitle(title, fontsize=11.5, color=th["primary"], x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=160, facecolor=th["surface"])
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir")
    args = ap.parse_args()
    sweep_dir = Path(args.sweep_dir)
    summary = sweep_dir / "lrk_summary.json"
    if not summary.exists():
        raise SystemExit(f"{summary} missing — run sft/lrk_analysis.py {sweep_dir} first")
    cells = json.loads(summary.read_text())
    for theme in ("light", "dark"):
        suffix = "" if theme == "light" else "_dark"
        plot(cells, sweep_dir / f"lrk_curve{suffix}.png", theme, sweep_dir.name)


if __name__ == "__main__":
    main()
