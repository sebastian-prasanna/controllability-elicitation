"""Unique-sample counts per (run, mode) for the per-run training_data folders built from
sft/extended_training_data_runs/*, plus drop reasons from the builder summaries.

    python sft/extended_training_data_runs/count_table.py
"""
import json
from collections import Counter
from pathlib import Path

from cotcontrol.eval.prompts import EXTENDED_MODES, MODES

RUNS = ["gptoss20b_gepa_general", "gptoss20b_fewshot_k1", "gptoss120b_gepa_general",
        "gptoss120b_fewshot_k1", "qwen8b_gepa_general", "qwen8b_fewshot_k1",
        "qwen32b_gepa_general", "qwen32b_fewshot_k1"]
ALL_MODES = [m for m in MODES if m != "baseline"] + EXTENDED_MODES
TD = Path("sft/training_data")

present = [r for r in RUNS if (TD / r).is_dir()]
counts, verbatim, answer_only, drops = {}, {}, {}, {}
for r in present:
    for m in ALL_MODES:
        f = TD / r / f"{m}.json"
        rows = json.loads(f.read_text()) if f.exists() else None
        counts[(r, m)] = None if rows is None else len({(x["meta"]["dataset"], x["meta"]["id"]) for x in rows})
        verbatim[(r, m)] = None if rows is None else sum(x["meta"].get("transform") == "none" for x in rows)
        answer_only[(r, m)] = None if rows is None else sum(bool(x["meta"].get("output_answer_only")) for x in rows)
    d = Counter()
    for s in (TD / r / "meta").glob("summary_*.json"):
        for st in json.loads(s.read_text()).values():
            d.update(st.get("dropped", {}))
    drops[r] = dict(d)

short = {r: r.replace("_gepa_general", "/gepa").replace("_fewshot_k1", "/fs1") for r in present}
print(f"{'mode':26s}" + "".join(f"{short[r]:>16s}" for r in present))
for m in ALL_MODES:
    cells = []
    for r in present:
        c, v = counts[(r, m)], verbatim[(r, m)]
        a = answer_only[(r, m)]
        cells.append("       -" if c is None else f"{c:4d}/{a:4d}({v:3d})".rjust(16))
    print(f"{m:26s}" + "".join(cells))
print("\ncell = unique questions / of which response is exactly 'ANSWER: X' (of which model already compliant, kept verbatim)")
print("\nrollouts dropped by the quality filters, per run:")
for r in present:
    print(f"  {r:26s} {drops[r]}")
