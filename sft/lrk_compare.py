#!/usr/bin/env python
"""Compare elicitation curves across models, and test what governs them.

    python sft/lrk_analysis.py sft/runs/lrk_gptoss20b   # per-sweep, first
    python sft/lrk_compare.py sft/runs/lrk_gptoss20b sft/runs/lrk_qwen8b ...

The question this answers: is masked elicitation set by the ABSOLUTE number of
trainable scalars k, or by the FRACTION of the adapter they represent (k/N)?
The models have very different adapter sizes (gpt-oss-20b 15.9M, Qwen3-8B 30.7M,
Qwen3-32B 79.7M), so at fixed k they sit at very different k/N. Plotting
compliance-as-a-fraction-of-each-model's-own-ceiling against both x axes,
whichever one COLLAPSES the curves onto each other is the governing variable.

Each model is normalized by its OWN unmasked control (best non-degraded), so
models with different baseline compliance are comparable.

Writes lrk_compare.{png,dark.png,md} next to the first sweep dir's parent.
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
                  grid="#d8d7d2", series=("#2a78d6", "#eb6834", "#1baf7a", "#8a56c9")),
    "dark": dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
                 grid="#3a3a38", series=("#3987e5", "#d95926", "#199e70", "#a276e0")),
}


def load_sweep(sweep_dir: Path) -> dict | None:
    summary = sweep_dir / "lrk_summary.json"
    if not summary.exists():
        print(f"skip {sweep_dir.name}: no lrk_summary.json (run lrk_analysis.py first)")
        return None
    cells = json.loads(summary.read_text())
    done = [c for c in cells if "best_compliance" in c]
    masked = [c for c in done if not c.get("unmasked")]
    control = [c for c in done if c.get("unmasked") and c["status"] == "OK"]
    if not masked:
        print(f"skip {sweep_dir.name}: no completed masked cells")
        return None

    # Adapter size: from any cell's train_result (total_lora_params), else infer.
    n_params = None
    for d in sweep_dir.iterdir():
        tr = d / "train_result.json"
        if tr.is_file():
            v = json.loads(tr.read_text()).get("total_lora_params")
            if v:
                n_params = int(v)
                break

    ceiling = max((c["best_compliance"] for c in control), default=None)
    points = []
    for k in sorted({c["k"] for c in masked}):
        grp = [c for c in masked if c["k"] == k]
        ok = [c for c in grp if c["status"] == "OK"]
        if not ok:      # nothing viable at this k — no comparable number
            continue
        best = max(ok, key=lambda c: c["best_compliance"])
        points.append(dict(k=k, k_effective=best.get("k_effective"),
                           lr=best["lr"], compliance=best["best_compliance"],
                           accuracy=best["acc_at_best"],
                           base_accuracy=best["base_accuracy"]))
    return dict(name=sweep_dir.name.replace("lrk_", ""), n_params=n_params,
                ceiling=ceiling, points=points,
                n_control=len(control))


def render_table(sweeps: list[dict]) -> str:
    lines = ["", f"{'model':>12}{'adapter':>12}{'k':>9}{'k/N':>9}{'lr':>7}"
                 f"{'compliance':>12}{'% ceiling':>11}{'accuracy':>18}"]
    for s in sweeps:
        for p in s["points"]:
            frac = f"{100 * p['k'] / s['n_params']:.3f}%" if s["n_params"] else "?"
            pct = (f"{100 * p['compliance'] / s['ceiling']:.0f}%"
                   if s["ceiling"] else "no ctrl")
            lines.append(
                f"{s['name']:>12}{(s['n_params'] or 0) / 1e6:>11.1f}M{p['k']:>9}"
                f"{frac:>9}{p['lr']:>7}{p['compliance']:>12.3f}{pct:>11}"
                f"{p['base_accuracy']:>9.3f}->{p['accuracy']:.3f}")
        ceil = f"{s['ceiling']:.3f}" if s["ceiling"] else "MISSING"
        lines.append(f"{s['name']:>12}{'':>12}{'unmasked':>9}{'100%':>9}{'':>7}"
                     f"{ceil:>12}{'100%':>11}")
    lines.append("")
    lines.append("Governing-variable test: normalized compliance is plotted against BOTH")
    lines.append("absolute k and k/N. The axis on which the curves COLLAPSE is the one that")
    lines.append("governs; the axis on which they stay separated is not.")
    return "\n".join(lines)


def plot(sweeps: list[dict], out: Path, t: str) -> None:
    th = THEME[t]
    usable = [s for s in sweeps if s["ceiling"] and s["n_params"] and s["points"]]
    if not usable:
        print("no sweep has both an unmasked control and a known adapter size — "
              "cannot normalize; skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    fig.patch.set_facecolor(th["surface"])
    for ax in axes:
        ax.set_facecolor(th["surface"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(th["grid"])
        ax.tick_params(colors=th["secondary"], labelsize=8, length=3)
        ax.grid(True, color=th["grid"], alpha=0.45, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)

    for i, s in enumerate(usable):
        color = th["series"][i % len(th["series"])]
        ys = [p["compliance"] / s["ceiling"] for p in s["points"]]
        for ax, xs in ((axes[0], [p["k"] for p in s["points"]]),
                       (axes[1], [p["k"] / s["n_params"] for p in s["points"]])):
            ax.plot(xs, ys, "-o", color=color, linewidth=2, ms=8, label=s["name"],
                    markeredgecolor=th["surface"], markeredgewidth=2)
            ax.annotate(f" {s['name']}", (xs[-1], ys[-1]), fontsize=8, va="center",
                        ha="left", color=th["primary"])

    axes[0].set_xlabel("k (absolute trainable scalars)", fontsize=9, color=th["secondary"])
    axes[0].set_title("A · vs absolute k", fontsize=10, color=th["primary"], loc="left")
    axes[1].set_xlabel("k / N (fraction of the adapter)", fontsize=9, color=th["secondary"])
    axes[1].set_title("B · vs fraction of adapter", fontsize=10, color=th["primary"], loc="left")
    for ax in axes:
        ax.set_ylabel("compliance / model's own unmasked ceiling", fontsize=9,
                      color=th["secondary"])
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi * 3.0)
        leg = ax.legend(frameon=False, fontsize=8, loc="upper left")
        for txt in leg.get_texts():
            txt.set_color(th["secondary"])

    fig.suptitle("What governs masked elicitation: absolute k, or fraction of the adapter?",
                 fontsize=11.5, color=th["primary"], x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=160, facecolor=th["surface"])
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dirs", nargs="+")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    sweeps = [s for s in (load_sweep(Path(d).resolve()) for d in args.sweep_dirs) if s]
    if not sweeps:
        raise SystemExit("no usable sweeps")

    table = render_table(sweeps)
    print(table)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.sweep_dirs[0]).resolve().parent
    (out_dir / "lrk_compare.md").write_text(f"# Cross-model elicitation\n\n```{table}\n```\n")
    plot(sweeps, out_dir / "lrk_compare.png", "light")
    plot(sweeps, out_dir / "lrk_compare_dark.png", "dark")
    missing = [s["name"] for s in sweeps if not s["ceiling"]]
    if missing:
        print(f"\nNOTE: no unmasked control yet for {missing} — those models are in the "
              f"table but cannot be normalized or plotted.")


if __name__ == "__main__":
    main()
