"""Summarize the sub-9-shot sweep: fewshot/runs/subset/gptoss120b -> results.md next to it.

Per prompt: val compliance (9 modes), compliance on modes WITH a demo in the prompt vs
WITHOUT, held-out compliance (3 modes), accuracy. Then per condition: mean +- SD over seeds.
"""
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs" / "subset" / "gptoss120b"
DEMOS = json.load(open(ROOT / "subset_prompts" / "gptoss120b" / "demos.json"))
ALL9 = None


def rate(pm, modes):
    n = sum(pm[m]["n"] for m in modes)
    return sum(pm[m]["compliant"] for m in modes) / n if n else float("nan")


def main():
    rows = {}
    for d in sorted(RUNS.glob("*_val")):
        name = d.name[:-4]
        sv, sh = d / "summary.json", RUNS / f"{name}_heldout" / "summary.json"
        if not (sv.exists() and sh.exists()):
            continue
        v, h = json.load(open(sv)), json.load(open(sh))
        pm = v["per_mode"]
        demo_modes = (DEMOS[name]["modes"] if name in DEMOS
                      else list(pm) if name.startswith("k9") else [])
        other = [m for m in pm if m not in demo_modes]
        rows[name] = {"val": v["compliance_rate"], "in": rate(pm, demo_modes) if demo_modes else float("nan"),
                      "out": rate(pm, other) if other else float("nan"), "heldout": h["compliance_rate"],
                      "acc": v["accuracy"], "err": v["n_errors"] + h["n_errors"],
                      "modes": demo_modes, "per_mode": {m: pm[m]["compliant"] / pm[m]["n"] for m in pm}}

    f = lambda x: "  —  " if x != x else f"{x:.3f}"
    lines = ["# Sub-9-shot sweep, gpt-oss-120b (val, T=1.0, pinned Groq/medium)", "",
             "in/out = compliance on val modes with / without a demo in the prompt.", "",
             "## Per condition (mean ± SD over seeds)", "",
             "| condition | n | val (9 modes) | in-demo | out-of-demo | held-out | acc |",
             "|---|---|---|---|---|---|---|"]
    conds = {}
    for name in rows:
        conds.setdefault(name.rsplit("_s", 1)[0] if "_s" in name else name, []).append(name)
    order = ["k0", "n1", "n3_random", "n3_strat", "n5_random", "k9"]
    for c in sorted(conds, key=lambda c: order.index(c) if c in order else 99):
        cell = []
        for k in ("val", "in", "out", "heldout", "acc"):
            xs = [rows[n][k] for n in conds[c] if rows[n][k] == rows[n][k]]
            cell.append("—" if not xs else f"{st.mean(xs):.3f}" + (f" ± {st.stdev(xs):.3f}" if len(xs) > 1 else ""))
        lines.append(f"| {c} | {len(conds[c])} | " + " | ".join(cell) + " |")
    lines += ["", "## Per prompt", "", "| prompt | demo modes | val | in | out | held-out | acc | errors |",
              "|---|---|---|---|---|---|---|---|"]
    for n, r in rows.items():
        dm = "all 9" if n.startswith("k9") else ", ".join(r["modes"]) or "—"
        lines.append(f"| {n} | {dm} | {f(r['val'])} | {f(r['in'])} | {f(r['out'])} | "
                     f"{f(r['heldout'])} | {f(r['acc'])} | {r['err']} |")
    modes = sorted(next(iter(rows.values()))["per_mode"])
    lines += ["", "## Per mode (val compliance; * = mode has a demo in the prompt)", "",
              "| prompt | " + " | ".join(m[:14] for m in modes) + " |", "|---" * (len(modes) + 1) + "|"]
    for n, r in rows.items():
        lines.append(f"| {n} | " + " | ".join(
            f"{r['per_mode'][m]:.2f}{'*' if m in r['modes'] and not n.startswith('k9') else ''}"
            for m in modes) + " |")
    (RUNS / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
