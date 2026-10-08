"""Scratch: analysis 4 — qualitative samples + heuristic failure classification."""
import json, os, random, re
from collections import Counter, defaultdict

ROOT = "/root/controllability-elicitation"
OUT = os.path.join(ROOT, "fewshot", "scratch_compact")
recs = {k: json.load(open(os.path.join(OUT, f"{k}.json"))) for k in ["k1", "k3"]}
random.seed(0)

DECLINE_MODES = ["end_of_sentence", "lowercase_thinking", "repeat_sentences", "uppercase_thinking", "ignore_question"]

def show(r, label):
    print(f"--- {label} pos={r['pos']} mode={r['mode']} ds={r['dataset']} comp={r['compliance']} rt_len={r['rt_len']}")
    print("HEAD:", r["rt_head"][:500].replace("\n", " | "))
    print("TAIL:", r["rt_tail"][-200:].replace("\n", " | "))
    print("OUT_TAIL:", (r["out_tail"] or "").replace("\n", " | "))

for m in DECLINE_MODES[:3]:
    nc3 = [r for r in recs["k3"] if r["mode"] == m and not r["compliance"]]
    c1 = [r for r in recs["k1"] if r["mode"] == m and r["compliance"]]
    print(f"\n########## MODE {m}: k3 noncompliant n={len(nc3)}, k1 compliant n={len(c1)}")
    for r in random.sample(nc3, min(5, len(nc3))):
        show(r, "k3 NONCOMP")
    print()
    for r in random.sample(c1, min(3, len(c1))):
        show(r, "k1 COMP")

# heuristic classification over ALL noncompliant rollouts per k
print("\n=== heuristic classification of noncompliant reasoning heads ===")
META = re.compile(r"requirement|constraint|analysis channel|the user wants|we must (?:not )?(?:write|use|alternate|end|meow|avoid)|instructions? say", re.I)
ECHO = re.compile(r"contractile vacuole|### Example|radioactive nuclei X\(Z,A\)", re.I)
for k in ["k1", "k3"]:
    cnt = Counter()
    for r in recs[k]:
        if r["compliance"]: continue
        h = r["rt_head"] + " " + r["rt_tail"]
        if META.search(h): cnt["meta_discusses_constraint"] += 1
        if ECHO.search(h): cnt["echoes_examples"] += 1
        cnt["total_noncomp"] += 1
    print(k, dict(cnt))

# partial compliance check: lowercase/uppercase drift — measure case ratio in head vs tail
print("\n=== case-mode partial compliance (frac letters wrong-case in head vs tail of graded text) ===")
def frac_upper(s):
    letters = [c for c in s if c.isalpha()]
    return sum(c.isupper() for c in letters) / max(1, len(letters))
for k in ["k1", "k3"]:
    for m, want in [("lowercase_thinking", "low"), ("uppercase_thinking", "up")]:
        nc = [r for r in recs[k] if r["mode"] == m and not r["compliance"]]
        heads = [frac_upper(r["rt_head"]) for r in nc]
        tails = [frac_upper(r["rt_tail"]) for r in nc]
        mh, mt = sum(heads)/len(heads), sum(tails)/len(tails)
        # count "starts compliant" (head frac_upper <0.05 for lowercase, >0.95 for uppercase)
        if want == "low":
            start_ok = sum(1 for h in heads if h < 0.05); end_ok = sum(1 for t in tails if t < 0.05)
        else:
            start_ok = sum(1 for h in heads if h > 0.95); end_ok = sum(1 for t in tails if t > 0.95)
        print(f"{k} {m}: n_noncomp={len(nc)} mean_frac_upper head={mh:.3f} tail={mt:.3f} | starts_compliant={start_ok} ends_compliant={end_ok}")

# meow: check head contains 'meow' at all among noncompliant
for k in ["k1", "k3"]:
    nc = [r for r in recs[k] if r["mode"] == "meow_between_words" and not r["compliance"]]
    some = sum(1 for r in nc if "meow" in (r["rt_head"]+r["rt_tail"]).lower())
    print(f"{k} meow noncomp n={len(nc)}, contains 'meow' somewhere: {some}")
# end_of_sentence: requirement keyword presence in noncompliant heads? need requirement string; skip.
