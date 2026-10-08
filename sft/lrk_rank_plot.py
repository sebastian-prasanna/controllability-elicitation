"""Rank-1 vs rank-32: compliance and reasoning length vs effective k.

    python sft/lrk_rank_plot.py

Two figures, one per model pair (small: gpt-oss-20b + qwen8b; big:
gpt-oss-120b + qwen32b), each with:

Panel A: best-cell val compliance against effective k (mask-liveness-corrected),
one line per (model, rank), horizontal dotted line = base-model compliance.
Panel B: mean reasoning length (characters) of the same best checkpoints, with
the base-model reasoning length from baselines/<model>/ as the reference line.

Cells: seed-0 only (canonical grid), DEGRADED/COLLAPSE excluded; the best cell
per (model, rank, k) by val compliance. Writes lrk_rank_curve{,_big}{,_dark}.png
into sft/runs/.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "sft" / "runs"

GROUPS = {  # figure suffix -> [(model label, sweep folder, rank, color key)]
    "": [
        ("gpt-oss-20b", "lrk_gptoss20b", 32, "blue"),
        ("gpt-oss-20b", "r1_gptoss20b", 1, "blue"),
        ("qwen8b", "lrk_qwen8b", 32, "orange"),
        ("qwen8b", "r1_qwen8b", 1, "orange"),
    ],
    "_big": [
        ("gpt-oss-120b", "lrk_gptoss120b", 32, "blue"),
        ("gpt-oss-120b", "r1_gptoss120b", 1, "blue"),
        ("qwen32b", "lrk_qwen32b", 32, "orange"),
        ("qwen32b", "r1_qwen32b", 1, "orange"),
    ],
}
BASELINE_DIRS = {"gpt-oss-20b": "gptoss20b", "qwen8b": "qwen8b",
                 "gpt-oss-120b": "gptoss120b", "qwen32b": "qwen32b"}
THEME = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  grid="#d8d7d2", blue="#2a78d6", orange="#eb6834"),
    "dark": dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
                 grid="#3a3a38", blue="#3987e5", orange="#d95926"),
}


def best_cells(sweep: Path) -> list[dict]:
    cells = json.loads((sweep / "lrk_summary.json").read_text())
    out = {}
    for c in cells:
        if c["unmasked"] or "-seed" in c["run"] or "best_compliance" not in c:
            continue
        if c["status"] in ("DEGRADED", "COLLAPSE"):
            continue
        k = c["k"]
        if k not in out or c["best_compliance"] > out[k]["best_compliance"]:
            out[k] = c
    return [out[k] for k in sorted(out)]


def mean_reasoning_chars(rows: list[dict]) -> float:
    lens = [len((s.get("reasoning") or "")) for r in rows
            for s in (r.get("samples") or []) if not s.get("error")]
    return sum(lens) / max(len(lens), 1)


def cell_reasoning(sweep: Path, cell: dict) -> float:
    f = sweep / cell["run"] / "eval" / "cotcontrol" / f"checkpoint-{cell['best_step']}.json"
    return mean_reasoning_chars(json.loads(f.read_text())["results"])


def baseline_stats(model: str, series: list) -> tuple[float, float]:
    """-> (base compliance on val eval, base mean reasoning chars from baselines/)."""
    sweep = RUNS / dict((m, s) for m, s, r, _ in series if r == 32)[model]
    base_c = json.loads((sweep / "lrk_summary.json").read_text())[0]["base_compliance"]
    f = next(p for p in (ROOT / "baselines" / BASELINE_DIRS[model]).glob("*.json")
             if p.name != "baseline_results.json")
    base_len = mean_reasoning_chars(json.loads(f.read_text())["results"])
    return base_c, base_len


def render(data: dict, series: list, mode: str, out: Path) -> None:
    th = THEME[mode]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    fig.patch.set_facecolor(th["surface"])
    for ax in axes:
        ax.set_facecolor(th["surface"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(th["grid"])
        ax.tick_params(colors=th["secondary"], labelsize=8)
        ax.grid(True, color=th["grid"], alpha=0.45, linewidth=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("effective k (live trainable scalars)", fontsize=9, color=th["secondary"])

    for (model, rank), pts in data["series"].items():
        color = th[dict((m, c) for m, _, r, c in series if r == rank)[model]]
        style = dict(color=color, linestyle="-" if rank == 32 else "--",
                     marker="o" if rank == 32 else "s", ms=7, linewidth=2,
                     markeredgecolor=th["surface"], markeredgewidth=1.5,
                     label=f"{model} rank {rank}")
        axes[0].plot([p["eff_k"] for p in pts], [p["compliance"] for p in pts], **style)
        axes[1].plot([p["eff_k"] for p in pts], [p["reasoning"] for p in pts], **style)

    # Stack the two baseline labels away from each other: higher line labeled
    # above, lower line labeled below, so they can't collide when values are close.
    # Panel A labels sit at the right edge (compliance curves rise left->right,
    # so the bottom-right corner is empty); panel B labels stay at the left.
    for panel, key in ((0, 0), (1, 1)):
        ordered = sorted(data["baselines"].items(), key=lambda kv: kv[1][key], reverse=True)
        for rank_pos, (model, vals) in enumerate(ordered):
            color = th[dict((m, c) for m, _, r, c in series if r == 32)[model]]
            axes[panel].axhline(vals[key], color=color, linestyle=":", linewidth=1.3, alpha=0.8)
            dy = 3 if rank_pos == 0 else -10
            x_edge, ha, dx = ((axes[panel].get_xlim()[1], "right", -4) if panel == 0
                              else (axes[panel].get_xlim()[0], "left", 4))
            axes[panel].annotate(f"{model} base", (x_edge, vals[key]),
                                 fontsize=7, color=th["secondary"], ha=ha,
                                 xytext=(dx, dy), textcoords="offset points")

    axes[0].set_title("A · compliance vs effective k", fontsize=11, color=th["primary"],
                      loc="left")
    axes[0].set_ylabel("strict compliance (val, best checkpoint)", fontsize=9,
                       color=th["secondary"])
    axes[0].set_ylim(0, None)
    axes[1].set_title("B · reasoning length vs effective k", fontsize=11,
                      color=th["primary"], loc="left")
    axes[1].set_ylabel("mean reasoning length (chars)", fontsize=9, color=th["secondary"])
    axes[1].set_ylim(0, None)
    axes[0].legend(fontsize=8, frameon=False, labelcolor=th["primary"])

    models = " + ".join(dict.fromkeys(m for m, *_ in series))
    fig.suptitle(f"Rank 1 vs rank 32 at matched effective k — {models} "
                 "(seed-0 cells; dotted = base model)",
                 fontsize=12, color=th["primary"], x=0.07, y=1.0, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=th["surface"])
    print(f"wrote {out}")


def main() -> None:
    for suffix, series in GROUPS.items():
        data = {"series": {}, "baselines": {}}
        for model, folder, rank, _ in series:
            sweep = RUNS / folder
            pts = []
            for c in best_cells(sweep):
                pts.append({"eff_k": c.get("k_effective") or c["k"],
                            "compliance": c["best_compliance"],
                            "reasoning": cell_reasoning(sweep, c)})
            data["series"][(model, rank)] = pts
            print(f"{model} rank {rank}: " +
                  " ".join(f"k_eff={p['eff_k']} c={p['compliance']:.3f} len={p['reasoning']:.0f}"
                           for p in pts))
        for model in dict.fromkeys(m for m, *_ in series):
            data["baselines"][model] = baseline_stats(model, series)
            print(f"{model} baseline: compliance={data['baselines'][model][0]:.3f} "
                  f"len={data['baselines'][model][1]:.0f}")
        render(data, series, "light", RUNS / f"lrk_rank_curve{suffix}.png")
        render(data, series, "dark", RUNS / f"lrk_rank_curve{suffix}_dark.png")


if __name__ == "__main__":
    main()
