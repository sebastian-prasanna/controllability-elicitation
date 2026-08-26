"""Controllability vs reasoning length across GEPA candidates (old runs).

For every old run with per-candidate reasoning stats, plot each candidate's
strict compliance (pareto fold) against its mean reasoning length relative to
that run's seed candidate (=100%). An arrow connects seed -> best candidate
(by pareto mean) per run. Saves the figure + extracted data next to this file.

    python gepa/analysis/compliance_vs_reasoning.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OLD_RUNS = HERE.parent / "old_runs"

MODEL_COLORS = {  # fixed categorical order (validated palette slots 1-4)
    "gpt-oss-120b": "#2a78d6",
    "glm-5.2": "#eb6834",
    "kimi-k3": "#1baf7a",
    "qwen3-8b": "#eda100",
}


def model_of(run_name: str) -> str:
    for prefix, model in [("gptoss", "gpt-oss-120b"), ("glm52", "glm-5.2"),
                          ("kimik3", "kimi-k3"), ("qwen8b", "qwen3-8b")]:
        if run_name.startswith(prefix):
            return model
    return "other"


def main():
    rows = []
    for d in sorted(OLD_RUNS.iterdir()):
        cj = d / "candidates.json"
        if not cj.is_file():
            continue
        cands = json.loads(cj.read_text())
        if not cands or any("pareto_reasoning_chars" not in c for c in cands):
            continue
        seed_chars = cands[0]["pareto_reasoning_chars"]
        best_id = max(cands, key=lambda c: c["pareto_mean"])["id"]
        for c in cands:
            rows.append({
                "run": d.name,
                "model": model_of(d.name),
                "cand_id": c["id"],
                "is_seed": c["id"] == 0,
                "is_best": c["id"] == best_id,
                "strict": c["pareto_compliance"],
                "chars": c["pareto_reasoning_chars"],
                "rel_chars": 100 * c["pareto_reasoning_chars"] / seed_chars,
            })
    (HERE / "compliance_vs_reasoning_data.json").write_text(json.dumps(rows, indent=1))

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    by_run = {}
    for r in rows:
        by_run.setdefault(r["run"], []).append(r)

    seen_models = set()
    for run, rs in by_run.items():
        color = MODEL_COLORS.get(rs[0]["model"], "#52514e")
        seed = next(r for r in rs if r["is_seed"])
        best = next(r for r in rs if r["is_best"])
        label = rs[0]["model"] if rs[0]["model"] not in seen_models else None
        seen_models.add(rs[0]["model"])
        xs = [r["rel_chars"] for r in rs]
        ys = [r["strict"] for r in rs]
        ax.scatter(xs, ys, s=42, color=color, alpha=0.55, edgecolors="none", label=label)
        ax.scatter([seed["rel_chars"]], [seed["strict"]], s=80, facecolors="#fcfcfb",
                   edgecolors=color, linewidths=1.6, zorder=3)
        if best["cand_id"] != seed["cand_id"]:
            ax.annotate(
                "", xy=(best["rel_chars"], best["strict"]),
                xytext=(seed["rel_chars"], seed["strict"]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.6, alpha=0.85),
            )

    ax.axvline(100, color="#c9c8c4", lw=1, ls="--", zorder=0)
    ax.text(100, ax.get_ylim()[1], " seed length", color="#52514e", fontsize=8.5,
            va="top", ha="left")
    ax.set_xlabel("mean reasoning length, % of seed candidate", color="#0b0b0b")
    ax.set_ylabel("strict compliance (pareto fold, n=48)", color="#0b0b0b")
    ax.set_title("GEPA candidates: compliance gains come with shorter reasoning\n"
                 "(arrows: seed → best candidate per run; open circles = seeds)",
                 color="#0b0b0b", fontsize=11, loc="left")
    ax.grid(True, color="#eceae6", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c4")
    ax.tick_params(colors="#52514e")
    ax.legend(frameon=False, loc="upper right", fontsize=9)

    out = HERE / "compliance_vs_reasoning.png"
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"saved {out} ({len(rows)} candidates from {len(by_run)} runs)")


if __name__ == "__main__":
    main()
