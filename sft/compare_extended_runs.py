"""Per-mode comparison of two run_baseline.py output dirs (e.g. empty prompt vs a
GEPA prompt on the 28-mode train split). Prints strict/shaped/accuracy per mode
for both and the delta, plus compliant-and-correct positive counts.

    python sft/compare_extended_runs.py sft/training_data/gptoss20b_train_extended \
        sft/training_data/gptoss20b_train_extended_gepa_general
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from cotcontrol.eval.data import CONSTRAINT_MODES  # noqa: E402
from cotcontrol.eval.prompts import EXTENDED_MODES  # noqa: E402


def load(d):
    summ = json.load(open(Path(d) / "baseline_results.json"))
    big = json.load(open(sorted(glob.glob(str(Path(d) / "*_all_all.json")))[-1]))
    pos, trunc, toks, n = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    for rec in big["results"]:
        s, m = rec["samples"][0], rec["mode"]
        n[m] += 1
        pos[m] += s["compliance"] == 1 and s["correct"] is True
        trunc[m] += s["finish_reason"] == "length"
        toks[m] += (s["usage"] or {}).get("completion_tokens", 0)
    return summ, pos, {m: trunc[m] / n[m] for m in n}, {m: toks[m] / n[m] for m in n}


def main(a, b):
    sa, pa, ta, ka = load(a)
    sb, pb, tb, kb = load(b)
    oa, ob = sa["overall"], sb["overall"]
    print(f"A = {a}\nB = {b}\n")
    print(f"{'':26s} {'strict A':>8} {'strict B':>8} {'delta':>7} | {'shaped A':>8} {'shaped B':>8} | "
          f"{'acc A':>6} {'acc B':>6} | {'pos A':>5} {'pos B':>5} | {'tok A':>5} {'tok B':>5} | {'trunc B':>7}")
    for grp, modes in (("default", CONSTRAINT_MODES), ("extended", EXTENDED_MODES)):
        print(f"-- {grp}")
        for m in modes:
            ma, mb = sa["per_mode"][m], sb["per_mode"][m]
            print(f"{m:26s} {ma['strict_compliance']:8.3f} {mb['strict_compliance']:8.3f} "
                  f"{mb['strict_compliance'] - ma['strict_compliance']:+7.3f} | "
                  f"{ma['shaped_compliance']:8.3f} {mb['shaped_compliance']:8.3f} | "
                  f"{ma['accuracy']:6.3f} {mb['accuracy']:6.3f} | {pa[m]:5d} {pb[m]:5d} | "
                  f"{ka[m]:5.0f} {kb[m]:5.0f} | {100 * tb[m]:6.1f}%")
    print(f"\n{'OVERALL':26s} {oa['strict_compliance']:8.3f} {ob['strict_compliance']:8.3f} "
          f"{ob['strict_compliance'] - oa['strict_compliance']:+7.3f} | "
          f"{oa['shaped_compliance']:8.3f} {ob['shaped_compliance']:8.3f} | "
          f"{oa['accuracy']:6.3f} {ob['accuracy']:6.3f} | {sum(pa.values()):5d} {sum(pb.values()):5d}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
