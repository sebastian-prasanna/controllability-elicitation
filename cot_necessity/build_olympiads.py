#!/usr/bin/env python3
"""Build datasets/olympiads_integer.csv: a seeded 500-problem sample of Metaskepsis/Olympiads
(HF, 32,926 rows) restricted to cleanly gradable integer-answer problems.

Filters: answer is a bare integer (after stripping $...$); not multiple-choice ((A)/A) style
options); not a proof ("prove"/"show that"); not multi-part ((1), (a), a) ...); de-duplicated by
problem text. Columns match math500_integer.csv (question, answer, source, domain, level) plus
hf_id. Also adds a test-only entry (all rows) to datasets/splits.json; existing entries untouched.

    .venv/bin/python cot_necessity/build_olympiads.py
"""
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "datasets" / "olympiads_integer.csv"
N, SEED = 500, 0

df = load_dataset("Metaskepsis/Olympiads")["train"].to_pandas()
ans = df["answer"].astype(str).str.strip().str.replace(r"^\$|\$$", "", regex=True).str.strip()
p = df["problem"]
keep = (
    ans.str.fullmatch(r"-?\d+")
    & ~p.str.contains(r"\(A\)|\bA\)|\\textbf\{\(A\)|A\.\s", regex=True)
    & ~p.str.contains(r"\b[Pp]rove\b|\b[Ss]how that\b", regex=True)
    & ~p.str.contains(r"\b[abc]\)\s|\(1\)|\(2\)|\(a\)|\(b\)", regex=True)
)
d = df[keep].assign(answer=ans[keep]).drop_duplicates("problem")
print(f"{len(df)} rows -> {len(d)} clean integer-answer problems -> sample {N}")
s = d.sample(N, random_state=SEED).reset_index(drop=True)
out = pd.DataFrame({"question": s["problem"].str.strip(), "answer": s["answer"],
                    "source": "Olympiads", "domain": "olympiads", "level": None, "hf_id": s["id"]})
out.to_csv(OUT, index=False)
print(f"wrote {OUT} ({len(out)} rows)")

sp_path = REPO / "datasets" / "splits.json"
sp = json.loads(sp_path.read_text())
sp["splits"][OUT.stem] = {"train": [], "val": [], "test": list(range(len(out)))}
sp_path.write_text(json.dumps(sp))  # same compact one-line format
print(f"added test split for {OUT.stem} to {sp_path}")
