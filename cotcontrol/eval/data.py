"""Dataset loading and question->constraint-mode assignment for CoT-Control-QA."""

import ast
import json
import random
from pathlib import Path
from typing import Optional

import pandas as pd

from cotcontrol.eval.prompts import EXTENDED_MODES, HELDOUT_MODES, MODES

DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"
DATASET_ALIASES = {
    "gpqa": DATASETS_DIR / "gpqa_w_keyword.csv",
    "hle": DATASETS_DIR / "hle_w_keyword.csv",
    "mmlu_pro": DATASETS_DIR / "mmlu_pro_mini_w_keyword.csv",
    "mmlu": DATASETS_DIR / "mmlu_pro_mini_w_keyword.csv",
    # Free-form integer-answer math (options=None, correct_answer = int string;
    # graded by exact boxed-integer match).
    "math": DATASETS_DIR / "math_train_integer.csv",
    "math_l45": DATASETS_DIR / "math_train_l45.csv",
    "math500": DATASETS_DIR / "math500_integer.csv",
    # 500-problem integer-answer sample of Metaskepsis/Olympiads (cot_necessity/build_olympiads.py).
    "olympiads": DATASETS_DIR / "olympiads_integer.csv",
}
ALL_DATASETS = ["gpqa", "hle", "mmlu_pro"]

CONSTRAINT_MODES = [m for m in MODES if m != "baseline"]  # the 9 constraint modes
# HELDOUT_MODES and EXTENDED_MODES are deliberately NOT in the default pool:
# "random"/"all" assignments (and thus comparability with existing runs) are
# unchanged unless they're explicitly requested via allowed_modes or an
# explicit mode.
KEYWORD_MODES = ("word_suppression", "multiple_word_suppression")


def _load_one_csv(data_file: Path, dataset_name: str) -> list[dict]:
    df = pd.read_csv(data_file)
    options_col = "options" if "options" in df.columns else "answer_options"

    samples = []
    for idx, row in df.iterrows():
        # Parse keyword metadata for word-suppression modes; None if unavailable.
        keywords = None
        valid_keywords_str = row.get("valid_keywords", "")
        if not (pd.isna(valid_keywords_str) or str(valid_keywords_str).strip() in ("", "[]")):
            try:
                valid_keywords = ast.literal_eval(str(valid_keywords_str))
                keywords_with_synonyms = json.loads(str(row.get("keywords_with_synonyms", "")))
                if valid_keywords:
                    top_keyword = valid_keywords[0]
                    top_synonyms = next(
                        (
                            kw.get("synonyms", [])
                            for kw in keywords_with_synonyms
                            if kw.get("keyword") == top_keyword
                        ),
                        [],
                    )
                    all_keywords = [kw["keyword"] for kw in keywords_with_synonyms if kw.get("keyword")]
                    all_synonyms = [
                        s for kw in keywords_with_synonyms for s in kw.get("synonyms", [])
                    ]
                    keywords = {
                        "word_suppression": {"keyword": top_keyword, "synonyms": top_synonyms},
                        "multiple_word_suppression": {
                            "keyword": ", ".join(all_keywords),
                            "synonyms": all_synonyms,
                        },
                    }
            except (ValueError, SyntaxError, json.JSONDecodeError):
                pass

        options = None
        if pd.notna(row.get(options_col)):
            try:
                options = json.loads(row[options_col])
            except (json.JSONDecodeError, TypeError):
                try:
                    options = ast.literal_eval(row[options_col])
                except (ValueError, SyntaxError):
                    pass

        samples.append(
            {
                "id": int(idx),
                "dataset": dataset_name,
                "question": str(row["question"]),
                "correct_answer": str(row["answer"]),
                "options": options,
                "source": str(row.get("source", "unknown")),
                "domain": str(row.get("domain", "unknown")),
                "keywords": keywords,
            }
        )
    return samples


def load_dataset(
    dataset: str | Path,
    mode: str,
    domains: Optional[list[str]] = None,
    max_samples: Optional[int] = None,
    subsample_seed: Optional[int] = None,
    split: Optional[str] = None,
) -> list[dict]:
    """Load CoT-Control-QA dataset(s) into a list of sample dicts.

    dataset="all" loads gpqa + hle + mmlu_pro. Rows without keywords are
    skipped only when mode is itself a word-suppression mode (as upstream);
    for mode="all"/"random" they stay and just can't be assigned those modes.
    split="train"/"val"/"test" filters to the canonical split in
    datasets/splits.json (built by build_splits.py); None uses all
    rows. Filters: split, domains (case-insensitive), then max_samples. With
    subsample_seed set, max_samples questions are drawn at random (same seed
    -> same fold, e.g. for minibatch evals); otherwise the first N are taken.
    """
    split_ids: Optional[dict] = None
    if split is not None:
        all_splits = json.loads((DATASETS_DIR / "splits.json").read_text())["splits"]
        split_ids = {stem: set(s[split]) for stem, s in all_splits.items()}

    names = ALL_DATASETS if str(dataset) == "all" else [dataset]
    samples = []
    for name in names:
        data_file = DATASET_ALIASES.get(str(name), Path(name))
        if not data_file.exists():
            raise FileNotFoundError(f"Dataset not found: {data_file}")
        rows = _load_one_csv(data_file, data_file.stem)
        if split_ids is not None:
            if data_file.stem not in split_ids:
                raise KeyError(f"No split defined for {data_file.stem} in splits.json")
            rows = [r for r in rows if r["id"] in split_ids[data_file.stem]]
        samples.extend(rows)

    if mode in KEYWORD_MODES:
        samples = [s for s in samples if s["keywords"]]
    if domains:
        allowed = {d.lower() for d in domains}
        samples = [s for s in samples if s["domain"].lower() in allowed]
    if subsample_seed is not None:
        random.Random(subsample_seed).shuffle(samples)
    if max_samples is not None:
        samples = samples[:max_samples]

    return samples


def assign_tasks(
    samples: list[dict],
    mode: str,
    seed: int = 0,
    allowed_modes: Optional[list[str]] = None,
) -> list[tuple[dict, str]]:
    """Assign a constraint mode to each sample -> list of (sample, mode) tasks.

    mode="all": every sample x every valid mode; mode="random": one seeded
    random valid mode per sample; otherwise the given mode for every sample.
    Keyword modes are only valid for samples with keyword metadata.
    allowed_modes restricts the pool "all"/"random" draw from (e.g. RL training
    on the programmatic modes only, or eval on HELDOUT_MODES); it doesn't
    affect an explicit mode. The default pool excludes HELDOUT_MODES.
    """
    valid = CONSTRAINT_MODES + HELDOUT_MODES + EXTENDED_MODES
    pool = CONSTRAINT_MODES if allowed_modes is None else [
        m for m in valid if m in set(allowed_modes)
    ]
    if allowed_modes is not None and len(pool) != len(set(allowed_modes)):
        unknown = set(allowed_modes) - set(valid)
        raise ValueError(f"unknown constraint modes: {sorted(unknown)}")

    def _valid_modes(sample):
        return [m for m in pool if sample["keywords"] or m not in KEYWORD_MODES]

    def _with_mode(sample, m):
        s = {k: v for k, v in sample.items() if k != "keywords"}
        if m in KEYWORD_MODES and sample["keywords"]:
            s.update(sample["keywords"][m])
        return s

    if mode == "all":
        return [(_with_mode(s, m), m) for s in samples for m in _valid_modes(s)]
    if mode == "random":
        rng = random.Random(seed)
        return [(_with_mode(s, (m := rng.choice(_valid_modes(s)))), m) for s in samples]
    return [(_with_mode(s, mode), mode) for s in samples]
