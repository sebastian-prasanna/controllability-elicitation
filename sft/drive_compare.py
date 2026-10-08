#!/usr/bin/env python
"""Compare the drive-1 / drive-3 gpt-oss-20b sweeps against the final (drive-10) sweep.

    python sft/drive_compare.py [--max-step 270]

Best val compliance per (k, drive) for the cotcontrol and heldout blocks; the
final sweep (3 epochs) is truncated to --max-step so all three are compared at
the same training budget. Final's heldout-val numbers come from
final_gptoss20b/heldout_val_summary.json (post-hoc pass); drive-1/3 heldout is
the in-config block. Writes sft/runs/drive_compare.md.
"""
import argparse, glob, json, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "sft", "runs")

def klabel_to_int(name):
    m = re.search(r"-k(\d+)(k?)-lr", name); k = int(m.group(1)); return k * 1000 if m.group(2) else k

def best(ckpts, key, max_step):
    ck = [c for c in ckpts if c["step"] <= max_step and c.get(key) is not None]
    if not ck: return None
    b = max(ck, key=lambda c: c[key]); return b[key], b["step"], b.get("accuracy")

def load(folder, prefix, max_step):
    out = {}
    for p in glob.glob(f"{R}/{folder}/{prefix}-k*-lr*/eval_summary.json"):
        s = json.load(open(p))
        if "-kall-" in s["run_name"]: continue   # unmasked ceiling cells: no k
        k = klabel_to_int(s["run_name"]); ev = s["evals"]
        out[k] = {"cot": best(ev["cotcontrol"]["checkpoints"], "compliance_rate", max_step),
                  "held": best(ev["heldout"]["checkpoints"], "compliance_rate", max_step) if "heldout" in ev else None,
                  "name": s["run_name"]}
    return out

def load_final_heldout(max_step):
    h = json.load(open(f"{R}/final_gptoss20b/heldout_val_summary.json")); out = {}
    for name, rows in h.items():
        if "-kall-" in name: continue
        rows = [{"step": r["step"], "compliance_rate": r["heldout_val_compliance"]} for r in rows]
        out[klabel_to_int(name)] = best(rows, "compliance_rate", max_step)
    return out

def fmt(b): return "—" if b is None else f"{b[0]:.3f}@{b[1]}"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max-step", type=int, default=270); a = ap.parse_args()
    d1 = load("final_gptoss20b_drive1", "final-gptoss20b-d1", a.max_step)
    d3 = load("final_gptoss20b_drive3", "final-gptoss20b-d3", a.max_step)
    d10 = load("final_gptoss20b", "final-gptoss20b", a.max_step); fh = load_final_heldout(a.max_step)
    for k in d10: d10[k]["held"] = fh.get(k)
    ks = sorted(set(d1) | set(d3) | set(d10)); ks = [k for k in ks if k <= 100_000]
    lines = [f"# gpt-oss-20b drive sweeps vs final (drive 10), best val compliance @ step, steps <= {a.max_step}", "",
             "| k | cot d1 | cot d3 | cot d10 | held d1 | held d3 | held d10 |", "|---|---|---|---|---|---|---|"]
    for k in ks:
        g = lambda d: d.get(k, {})
        lines.append(f"| {k:,} | {fmt(g(d1).get('cot'))} | {fmt(g(d3).get('cot'))} | {fmt(g(d10).get('cot'))} | "
                     f"{fmt(g(d1).get('held'))} | {fmt(g(d3).get('held'))} | {fmt(g(d10).get('held'))} |")
    lines += ["", f"cells: d1 {len(d1)}/16, d3 {len(d3)}/16, d10 {len([k for k in d10 if k<=100_000])}/16 (final restricted to k<=1e5)"]
    txt = "\n".join(lines); print(txt); open(f"{R}/drive_compare.md", "w").write(txt + "\n")

if __name__ == "__main__": main()
