#!/usr/bin/env python3
"""Compare the GEPA x few-shot hybrid arms against their parents.

Parents come from pinned_reeval/runs/gptoss20b (same pinned settings), so the
numbers are directly comparable.  Reports overall strict compliance, accuracy
and no-narration, plus a per-mode breakdown that shows whether the hybrid
actually unions the two parents' mode coverage.

  python hybrid_prompt/analyze.py            -> hybrid_prompt/results.md
"""
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HYB = REPO / "hybrid_prompt" / "runs" / "gptoss20b"
PR = REPO / "pinned_reeval" / "runs" / "gptoss20b"

# parents: the exact seeds the hybrid was built from, plus the 3-seed arm means
PARENTS = {"baseline": ["baseline"],
           "gepa (3-seed mean)": ["gepa_s0", "gepa_s1", "gepa_s2"],
           "gepa_s1 (hybrid parent)": ["gepa_s1"],
           "fewshot (3-seed mean)": ["fewshot_s1", "fewshot_s2", "fewshot_s3"],
           "fewshot_s2 (hybrid parent)": ["fewshot_s2"]}
HYBRIDS = {"hybrid_plain": ["hybrid_plain"], "hybrid_clean": ["hybrid_clean"]}


def summ(root, run, split):
    p = root / f"{run}_{split}" / "summary.json"
    return json.load(open(p)) if p.is_file() else None


def agg(root, runs, split):
    ds = [d for d in (summ(root, r, split) for r in runs) if d]
    if not ds:
        return None
    def f(key):
        xs = [d[key] for d in ds if d.get(key) is not None]
        if not xs:
            return "-"
        return f"{st.mean(xs):.3f}" + (f" ± {st.stdev(xs):.3f}" if len(xs) > 1 else "")
    return (f("compliance_rate"), f("accuracy"), f("meta_discussion_rate_compliant"),
            sum(d["n_errors"] for d in ds), ds)


def per_mode(root, runs, split):
    acc = {}
    for r in runs:
        d = summ(root, r, split)
        if not d:
            continue
        for m, v in d["per_mode"].items():
            acc.setdefault(m, []).append(v["compliant"] / v["n"])
    return {m: st.mean(v) for m, v in acc.items()}


L = ["# GEPA x few-shot hybrid, gpt-oss-20b (2026-09-23)\n",
     "Hybrid = best GEPA prompt (general arm s1; best on GEPA's own val pareto mean 0.260 and on test) "
     "with its text substituted for the first sentence of the k=1 few-shot header (few-shot s2, best on test), "
     "followed by the unchanged few-shot demo blocks. `hybrid_clean` additionally swaps the 2 of 9 demos whose "
     "reasoning narrates the constraint (multiple_word_suppression, word_suppression) for non-narrating positives "
     "from the same SFT pool, because GEPA rule 2 forbids exactly that behaviour.\n",
     "Settings pinned to configs/eval_pins.json (groq, T=0, top_p 1, 16k, effort medium, conc 50, meta judge on "
     "scope=compliant) -- identical to pinned_reeval, so parent rows are lifted from there.\n",
     "| split | arm | strict compliance | accuracy | no-narration (compliant) | errors |", "|---|---|---|---|---|---|"]
for split in ("test", "heldout"):
    for name, runs in PARENTS.items():
        a = agg(PR, runs, split)
        if a:
            L.append(f"| {split} | {name} | {a[0]} | {a[1]} | {a[2]} | {a[3]} |")
    for name, runs in HYBRIDS.items():
        a = agg(HYB, runs, split)
        L.append(f"| {split} | **{name}** | {a[0]} | {a[1]} | {a[2]} | {a[3]} |" if a
                 else f"| {split} | **{name}** | pending | | | |")

for split in ("test", "heldout"):
    cols = [("gepa_s1", PR, ["gepa_s1"]), ("fewshot_s2", PR, ["fewshot_s2"]),
            ("max(parents)", None, None),
            ("hybrid_plain", HYB, ["hybrid_plain"]), ("hybrid_clean", HYB, ["hybrid_clean"])]
    data = {n: per_mode(r, rs, split) for n, r, rs in cols if r}
    if not data.get("hybrid_plain"):
        continue
    modes = sorted(data["gepa_s1"])
    L += [f"\n## Per-mode strict compliance ({split})\n",
          "| mode | " + " | ".join(n for n, _, _ in cols) + " |",
          "|---" * (len(cols) + 1) + "|"]
    for m in modes:
        g, f_ = data["gepa_s1"].get(m, 0), data["fewshot_s2"].get(m, 0)
        row = [f"{g:.3f}", f"{f_:.3f}", f"{max(g, f_):.3f}",
               f"{data['hybrid_plain'].get(m, 0):.3f}",
               f"{data['hybrid_clean'].get(m, 0):.3f}" if data.get("hybrid_clean") else "-"]
        L.append(f"| {m} | " + " | ".join(row) + " |")
    means = []
    for n, _, _ in cols:
        if n == "max(parents)":
            means.append(st.mean([max(data["gepa_s1"].get(m, 0), data["fewshot_s2"].get(m, 0)) for m in modes]))
        else:
            means.append(st.mean([data[n].get(m, 0) for m in modes]) if data.get(n) else float("nan"))
    L.append("| **mean** | " + " | ".join(f"**{x:.3f}**" for x in means) + " |")


# ---- paired McNemar vs parents (same questions + mode assignment in every run) ----
import subprocess
mc = subprocess.run([__import__("sys").executable, str(REPO / "hybrid_prompt" / "mcnemar.py")],
                    capture_output=True, text=True).stdout
L += ["\n## Paired significance (exact McNemar on discordant rollouts)\n", "```", mc.strip(), "```"]

L += ["""
## Conclusion

`hybrid_plain` beats both parents on test (+0.019 over GEPA s1, p=0.008; +0.052 over few-shot s2,
p=3e-35) but the margin over GEPA is smaller than GEPA's own seed-to-seed sd (0.049), so on test the
hybrid is a real but marginal win, not a new regime. It reaches 0.205 of the 0.282 per-mode oracle.

On held-out modes the hybrid is a clear **regression**: 0.111 vs GEPA s1's 0.251 (-0.139, p=4e-40).

Mechanism (per-mode tables): the hybrid does inherit few-shot's `ignore_question` win (0.952, above
both parents) but loses most of GEPA's instruction-driven generality -- uppercase_thinking
0.490 -> 0.082, lowercase_thinking 0.254 -> 0.098, repeat_sentences 0.800 -> 0.648, and on held-out
start_of_sentence 0.686 -> 0.264. Demos anchor the model to imitating the nine demonstrated modes and
suppress the mode-general behaviour the GEPA text was buying.

`hybrid_clean` (the narration fix) is **worse everywhere**: 0.153 test, 0.058 held-out, below
`hybrid_plain` by 0.052 on both (p=2e-29 / p=3e-14). Swapping only the two word-suppression demos
collapsed an unrelated mode, `ignore_question`, from 0.952 to 0.566. That is the large cross-demo
interaction already documented for k=1 few-shot prompts, and it means demo identity matters far more
than demo hygiene here. Note also that `hybrid_plain` has the best no-narration rate of any arm
(0.966) despite containing the two narrating demos, so the GEPA text handles narration on its own.

Bottom line: prepending the GEPA prompt to the few-shot prompt buys ~2 points on in-distribution
modes and costs ~14 points on held-out modes. Not recommended as a default.
"""]

out = REPO / "hybrid_prompt" / "results.md"
out.write_text("\n".join(L) + "\n")
print("\n".join(L))
