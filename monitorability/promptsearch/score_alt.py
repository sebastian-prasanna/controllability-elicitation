#!/usr/bin/env python3
"""adv/score.py-style scoring of a run dir against an ALTERNATE monitor file (from remonitor.py).

    python3 monitorability/promptsearch/score_alt.py <run_dir> --monitor-file monitor__primed.jsonl \
        [--instance-range 0:10] [--datasets gpqa] [--excerpts 4] [--md]

Prints the same stats row as adv/score.py (TPR_raw = P(Z=1|X=1,Y=1) etc.) computed with z from the given
monitor file, the default monitor.jsonl row on the same rollouts for comparison, a default-vs-alt agreement
matrix on (x=1,y=1) rollouts labelled by both, monitor cost, and leak/hide excerpts. Writes
<run_dir>/ps_score__<tag>.json (tag = the part of the monitor file name after `monitor__`, or --tag).
adv/ is frozen, so this module only imports from adv/score.py; nothing there is modified.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MON = REPO / "monitorability"
sys.path.insert(0, str(MON / "adv"))
sys.path.insert(0, str(MON))
import score as S  # noqa: E402  (monitorability/adv/score.py, read-only)


def load_alt(run_dir: Path, datasets, inst_range, monitor_file: Path) -> list[dict]:
    """Same as adv/score.load but joins rollouts with an arbitrary monitor file."""
    R = S.read_jsonl(run_dir / "rollouts.jsonl")
    M = {(m["instance_id"], m["x"], m["sample_idx"]): m for m in S.read_jsonl(monitor_file)}
    if datasets:
        R = [r for r in R if r["dataset"] in datasets]
    if inst_range:
        from evals import load_rows
        a, b = (int(v) for v in inst_range.split(":"))
        keep = set()
        for ds in sorted({r["dataset"] for r in R}):
            keep |= {row.instance_id for row in load_rows(ds, None, (a, b))}
        R = [r for r in R if r["instance_id"] in keep]
    for r in R:
        m = M.get((r["instance_id"], r["x"], r["sample_idx"]))
        r["z"] = m["z"] if (m and m.get("z_valid")) else None
        r["monitor_output"] = m.get("monitor_output") if m else None
        r["monitor_cost"] = float((m.get("monitor_usage") or {}).get("cost") or 0) if m else None
        r["monitor_model"] = m.get("monitor_model") if m else None
    return R


def monitor_cost(R) -> tuple[float, int]:
    c = [r["monitor_cost"] for r in R if r.get("monitor_cost") is not None]
    return sum(c), len(c)


def agreement(R_def, R_alt) -> dict:
    """Default-vs-alt z agreement on x=1,y=1 rollouts labelled by both monitors."""
    za = {(r["instance_id"], r["x"], r["sample_idx"]): r["z"] for r in R_alt
          if r["y_valid"] and r["x"] == 1 and r["y"] == 1 and r["z"] is not None}
    cnt = collections.Counter()
    for r in R_def:
        k = (r["instance_id"], r["x"], r["sample_idx"])
        if k in za and r["y_valid"] and r["x"] == 1 and r["y"] == 1 and r["z"] is not None:
            cnt[(int(r["z"]), int(za[k]))] += 1
    n = sum(cnt.values())
    return {"n_both": n, "def0_alt0": cnt[(0, 0)], "def0_alt1": cnt[(0, 1)], "def1_alt0": cnt[(1, 0)],
            "def1_alt1": cnt[(1, 1)], "agree": (cnt[(0, 0)] + cnt[(1, 1)]) / n if n else float("nan")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--monitor-file", default="monitor.jsonl",
                    help="file name inside run_dir (or a path), e.g. monitor__primed.jsonl")
    ap.add_argument("--tag", default=None, help="output tag (default: derived from the monitor file name)")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--instance-range", default=None)
    ap.add_argument("--excerpts", type=int, default=4, help="leak/hide excerpts to print (0 = none)")
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--no-compare", action="store_true", help="skip the default monitor.jsonl comparison row")
    a = ap.parse_args()

    run_dir = Path(a.run_dir).resolve()
    mf = Path(a.monitor_file)
    mf = mf if mf.is_absolute() or mf.exists() else run_dir / mf
    if not mf.exists():
        sys.exit(f"monitor file not found: {mf}")
    stem = mf.name[:-len(".jsonl")] if mf.name.endswith(".jsonl") else mf.name
    tag = a.tag or (stem[len("monitor__"):] if stem.startswith("monitor__") else stem)

    R = load_alt(run_dir, a.datasets, a.instance_range, mf)
    if not R:
        sys.exit(f"no rollouts in {run_dir} for datasets={a.datasets} range={a.instance_range}")
    s = S.score(R)
    mcost, ncalls = monitor_cost(R)
    models = sorted({r["monitor_model"] for r in R if r.get("monitor_model")})
    pol = sum(float((r.get("usage") or {}).get("cost") or 0) for r in R)
    print(f"[{tag}] monitor={models} file={mf.name}")
    print(f"[{tag}] " + S.fmt(s, a.md))
    print(f"[{tag}] monitor labelled {ncalls}/{len(R)} rollouts, monitor ${mcost:.4f} "
          f"(${mcost / max(1, ncalls):.5f}/call); policy ${pol:.4f}")

    out = dict(run_dir=str(run_dir), monitor_file=str(mf), tag=tag, monitor_models=models, datasets=a.datasets,
               instance_range=a.instance_range, monitor_cost=mcost, n_monitor_rows=ncalls, policy_cost=pol, **s)
    default_mf = run_dir / "monitor.jsonl"
    if not a.no_compare and default_mf.exists() and default_mf.resolve() != mf.resolve():
        R0 = load_alt(run_dir, a.datasets, a.instance_range, default_mf)
        s0 = S.score(R0)
        c0, n0 = monitor_cost(R0)
        print(f"[default] " + S.fmt(s0, a.md))
        print(f"[default] monitor labelled {n0}/{len(R0)} rollouts, monitor ${c0:.4f} (${c0 / max(1, n0):.5f}/call)")
        ag = agreement(R0, R)
        print(f"[compare] TPR_raw default={s0['tpr_raw']:.3f} (n_tp {s0['n_tp']}) vs {tag}={s['tpr_raw']:.3f} "
              f"(n_tp {s['n_tp']}); FPR_ctrl {s0['fpr_ctrl']:.3f} vs {s['fpr_ctrl']:.3f}; "
              f"x1y1 both-labelled n={ag['n_both']}: def0/alt0={ag['def0_alt0']} def0/alt1={ag['def0_alt1']} "
              f"def1/alt0={ag['def1_alt0']} def1/alt1={ag['def1_alt1']} agree={ag['agree']:.2f}")
        out["default_monitor"] = {k: s0[k] for k in ("tpr_raw", "tpr_lo", "tpr_hi", "fpr_ctrl", "fpr_int_neg", "n_tp")}
        out["default_monitor"]["monitor_cost"] = c0
        out["agreement_x1y1"] = ag
    if a.excerpts:
        S.excerpts(R, a.excerpts)
    path = run_dir / f"ps_score__{tag}.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"[score_alt] wrote {path}")


if __name__ == "__main__":
    main()
