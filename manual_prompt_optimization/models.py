"""Model registry, budget constants and reference-data loaders shared by launch/finalize/analyze.

Everything that defines "the method" numerically lives here so a run can be described by
(BRIEF_VERSION, this file's constants, the agent model id) and reproduced from the run's
launch_meta.json.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(__file__).resolve().parent / "runs"

# Same 8 task models as gepa/sweep.py MODELS (initial_sweep / second_sweep) and fewshot final_k1.
MODELS: dict[str, str] = {
    "gptoss20b": "openai/gpt-oss-20b",
    "gptoss120b": "openai/gpt-oss-120b",
    "qwen8b": "qwen/qwen3-8b",
    "qwen32b": "qwen/qwen3-32b",
    "glm53": "z-ai/glm-5.3",
    "kimik3": "moonshotai/kimi-k3",
    "dsv4pro": "deepseek/deepseek-v4-pro-0813",
    "glm53flash": "z-ai/glm-5.3-flash",
}

# ---- fixed optimization budget ----
# Parity with GEPA: a GEPA run (gepa/sweep.py BASE_FLAGS: 10 iterations, minibatch 64, pareto 128)
# spends 128 + 10*64 parent-minibatch + <=10*64 child-minibatch + <=10*128 pareto rollouts on the task
# model, i.e. at most 2688 and on average 1667 over the 24 general-advice runs. Each agent gets the
# GEPA ceiling as a hard cap on task-model rollouts, enforced by the launch guard (planned rollouts
# of every run_eval.py call are counted before it is allowed) and re-checked by finalize.py.
N_AGENTS = 3                 # independent agents per model == GEPA's 3 seeds
ROLLOUT_BUDGET = 2700        # task-model rollouts per agent (GEPA ceiling 2688, rounded); excludes shared baselines
SUBSAMPLE_SEED = 0           # every subset eval must use this seed so candidates are paired within an agent
TRAIN_SUBSET_N = 60          # the shared train-subset baseline is run at this N (agents may use other N)
MAX_TOKENS = 16000           # == GEPA runs and canonical baselines
EVAL_CONCURRENCY = 150       # per eval process
MAX_CONCURRENT_EVALS = 2     # per agent
VAL_N, TRAIN_N = 200, 514    # canonical split sizes (datasets/splits.json)
N_MODES = 9


def planned_rollouts(cmd: str) -> int:
    """Upper bound on task-model rollouts a run_eval.py command will spend (for the budget guard)."""
    split = re.search(r"--split\s+(train|val)", cmd)
    n_q = {"train": TRAIN_N, "val": VAL_N}.get(split.group(1) if split else "", TRAIN_N)
    m = re.search(r"--max-samples\s+(\d+)", cmd)
    if m:
        n_q = min(n_q, int(m.group(1)))
    mode = re.search(r"--mode\s+(\S+)", cmd)
    per_q = N_MODES if (mode is None or mode.group(1).strip("'\"") == "all") else 1
    ns = re.search(r"--num-samples\s+(\d+)", cmd)
    return n_q * per_q * (int(ns.group(1)) if ns else 1)


JUDGE_MODEL = "openai/gpt-5-mini"  # eval judge (ignore_question); same as GEPA/baselines


def run_dir(sweep: str, key: str) -> Path:
    return RUNS / sweep / key


def agent_dir(sweep: str, key: str, i: int) -> Path:
    return run_dir(sweep, key) / f"agent{i}"


def ref_dirs(key: str) -> dict:
    """Reference eval directories for a model (all TEST split unless named otherwise)."""
    G, F, B = ROOT / "gepa/runs", ROOT / "fewshot/runs", ROOT / "baselines"
    return {
        "baseline_test": B / key,
        "baseline_train": B / f"{key}_train",
        "baseline_heldout": B / f"{key}_heldout",
        "gepa_general": [("s0", G / f"initial_sweep/{key}_general"),
                         ("s1", G / f"second_sweep/{key}_general_s1"),
                         ("s2", G / f"second_sweep/{key}_general_s2")],
        "fewshot_k1_test": [(f"s{i}", F / f"final_k1_test/{key}_s{i}") for i in (1, 2, 3)],
        "fewshot_k1_heldout": [(f"s{i}", F / f"final_k1_heldout/{key}_s{i}") for i in (1, 2, 3)],
    }


def eval_json(d: Path) -> Path | None:
    fs = [f for f in glob.glob(str(d / "*.json"))
          if not f.endswith(("summary.json", "baseline_results.json", "freeze.json", "judge.json"))]
    return Path(sorted(fs)[-1]) if fs else None


def load_eval(d: Path, per_rollout: bool = False) -> dict | None:
    """Aggregate stats (+ optional per-rollout arrays) from a full eval JSON directory."""
    f = eval_json(d)
    if f is None:
        return None
    e = json.load(open(f))
    smp = [s for r in e["results"] for s in r["samples"] if not s.get("error")]
    chars = np.array([len(s["reasoning"] or "") for s in smp])
    words = np.array([len((s["reasoning"] or "").split()) for s in smp])
    comp = np.array([int(s["compliance"] == 1) for s in smp])
    out = {
        "file": str(f), "split": e["config"]["split"], "mode": e["config"]["mode"],
        "allowed_modes": e["config"].get("allowed_modes"),
        "max_tokens": e["config"]["generate_config"]["max_tokens"],
        "n": len(smp), "strict": float(comp.mean()), "accuracy": e["summary"]["accuracy"],
        "mean_chars": float(chars.mean()), "median_chars": float(np.median(chars)),
        "mean_words": float(words.mean()), "median_words": float(np.median(words)),
        "truncated": int(sum(1 for s in smp if s["finish_reason"] == "length")),
        "empty_output": int(sum(1 for s in smp if not (s["output"] or "").strip())),
        "near_empty_reasoning": int((chars < 500).sum()),
        "per_mode": {m: v["compliant"] / v["n"] for m, v in e["summary"]["per_mode"].items()},
    }
    if per_rollout:
        out["words"], out["comp"] = words, comp
    return out


def reference_numbers(key: str) -> dict:
    """Aggregate reference numbers for the brief (aggregates only — no per-sample data)."""
    r = ref_dirs(key)
    bt = json.load(open(r["baseline_test"] / "baseline_results.json"))["overall"]
    btr = json.load(open(r["baseline_train"] / "baseline_results.json"))["overall"]
    btr_len = load_eval(r["baseline_train"])
    gepa = []
    for lbl, d in r["gepa_general"]:
        t = json.load(open(d / "test_results.json"))["overall"]
        b = json.load(open(d / "best.json"))["best"]
        gepa.append({"seed": lbl, "test_strict": t["strict_compliance"], "test_acc": t["accuracy"],
                     "val_pareto_strict": b["pareto_compliance"], "val_pareto_chars": b["pareto_reasoning_chars"]})
    fs = [json.load(open(d / "summary.json")) for _, d in r["fewshot_k1_test"]]
    return {
        "baseline_train": {"strict": btr["strict_compliance"], "acc": btr["accuracy"], "n": btr["n"],
                           "mean_chars": btr_len["mean_chars"], "median_chars": btr_len["median_chars"]},
        "baseline_test": {"strict": bt["strict_compliance"], "acc": bt["accuracy"]},
        "gepa_general": gepa,
        "fewshot_k1_test": [{"strict": s["compliance_rate"], "acc": s["accuracy"]} for s in fs],
    }


def reference_table_md(key: str) -> str:
    n = reference_numbers(key)
    b, t = n["baseline_train"], n["baseline_test"]
    g = n["gepa_general"]; f = n["fewshot_k1_test"]
    lines = [
        f"- Empty prompt, TRAIN split (n={b['n']}): strict {b['strict']:.3f}, accuracy {b['acc']:.3f}, "
        f"mean reasoning {b['mean_chars']/1000:.1f}k chars, median {b['median_chars']/1000:.1f}k chars.",
        f"- Empty prompt, test split: strict {t['strict']:.3f}, accuracy {t['acc']:.3f}.",
        "- GEPA general-advice-only, 3 seeds, test strict / test acc / val-pareto strict (n=128) / val-pareto mean chars: "
        + "; ".join(f"{x['seed']}: {x['test_strict']:.3f} / {x['test_acc']:.3f} / {x['val_pareto_strict']:.3f} / "
                    f"{x['val_pareto_chars']/1000:.1f}k" for x in g) + ".",
        "- Few-shot k=1, 3 seeds, test strict: " + ", ".join(f"{x['strict']:.3f}" for x in f) + ".",
    ]
    return "\n".join(lines)
