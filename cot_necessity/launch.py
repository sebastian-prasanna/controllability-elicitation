#!/usr/bin/env python3
"""Does CoT-control compliance vs reasoning length look different on a task that NEEDS CoT?

Worry: the CoT-Control QA questions (GPQA / HLE / MMLU-Pro MCQ) may be mostly knowledge
retrieval, so the CoT is optional and easy to bend. Contrast with MATH-500 (integer subset,
314 test q) and Olympiads (olympiads_integer, 500 q; build_olympiads.py), which need real working. For gpt-oss-120b and Kimi-K3 with their best GEPA
general-arm prompt (best-of-3 by test strict: kimik3 s2, gptoss120b s1), run on MATH-500:
  math500_baseline       empty prompt, mode all (7 non-keyword modes; math has no keywords)
  math500_gepa           GEPA prompt verbatim (says "ANSWER: X" -> hurts math accuracy, not compliance)
  math500_gepa_mathfmt   GEPA prompt with only the MCQ answer-format lines swapped for \\boxed{N}
CoT-necessity check (mode=baseline, no Requirement):
  uncon_{math500,mcq}    reasoning on (pinned effort)
  nocot_{math500,mcq}    Kimi only, reasoning effort none (thinking off)
The MCQ compliance-vs-length side reuses pinned_reeval/runs/<m>/{baseline,gepa_<s>}_test (same pins).
Settings = configs/eval_pins.json (as pinned_reeval). Outputs: cot_necessity/runs/<model>/<run>/.
Evals of a model run in parallel in one tmux session, except SERIAL models (sequential).

  python cot_necessity/launch.py [--math-dataset olympiads] [--dry-run] [--models kimik3,gptoss120b] [--max-samples N --suffix smoke]
"""
from __future__ import annotations
import argparse, json, shlex, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HERE = REPO / "cot_necessity"
PINS = json.load(open(REPO / "configs" / "eval_pins.json"))
RUN_EVAL = REPO / "scripts" / "run_eval.py"

MODELS = {"kimik3": "moonshotai/kimi-k3", "gptoss120b": "openai/gpt-oss-120b"}
CONC = {"kimik3": 200, "gptoss120b": 30}  # per eval
SERIAL = {"kimik3"}  # run evals one at a time (Moonshot 429s ~20% at 7x60 parallel)
RETRIES = {"kimik3": 12}  # override pinned max_retries
NOCOT = {"kimik3"}  # models where reasoning effort "none" turns thinking off


def jobs_for(m: str, math_ds: str) -> list[tuple[str, list[str]]]:
    """Run names are prefixed by the math dataset (math500 / olympiads); the MCQ runs are shared."""
    p = HERE / "prompts"
    math = ["--dataset", math_ds, "--mode", "all"]
    mcq = ["--dataset", "all", "--mode", "baseline"]
    out = [
        (f"{math_ds}_baseline", math + ["--system-prompt", ""]),
        (f"{math_ds}_gepa_mathfmt", math + ["--system-prompt", str(p / f"{m}_gepa_mathfmt.txt")]),
        (f"uncon_{math_ds}", ["--dataset", math_ds, "--mode", "baseline", "--system-prompt", ""]),
        ("uncon_mcq", mcq + ["--system-prompt", ""]),
    ]
    if math_ds == "math500":  # verbatim prompt only run on math500 (no difference vs mathfmt)
        out.insert(1, ("math500_gepa", math + ["--system-prompt", str(p / f"{m}_gepa.txt")]))
    if m in NOCOT:
        out += [(f"nocot_{math_ds}", ["--dataset", math_ds, "--mode", "baseline", "--system-prompt", "",
                                      "--reasoning-effort", "none"]),
                ("nocot_mcq", mcq + ["--system-prompt", "", "--reasoning-effort", "none"])]
    return out


def build_cmd(m: str, extra: list[str], out: Path, max_samples: int | None) -> list[str]:
    mid = MODELS[m]; c = PINS[mid]
    cmd = [sys.executable, str(RUN_EVAL), "--model", mid, "--split", "test",
           "--temperature", str(c["temperature"]), "--top-p", str(c["top_p"]),
           "--max-tokens", str(c["max_tokens"]), "--concurrency", str(CONC[m]),
           "--max-retries", str(RETRIES.get(m, c["max_retries"])), "--provider", c["provider"],
           "--meta-discussion", "compliant", "--out-dir", str(out)]
    if c.get("quantizations"):
        cmd += ["--quantization", c["quantizations"][0]]
    if c.get("reasoning_effort") and "--reasoning-effort" not in extra:
        cmd += ["--reasoning-effort", c["reasoning_effort"]]
    if max_samples:
        cmd += ["--max-samples", str(max_samples)]
    return cmd + extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--math-dataset", default="math500", choices=["math500", "olympiads"])
    a = ap.parse_args()
    for m in a.models.split(","):
        lane = m + (f"_{a.suffix}" if a.suffix else "")
        ld = HERE / "runs" / lane; ld.mkdir(parents=True, exist_ok=True)
        log = shlex.quote(str(ld / "driver.log"))
        script = [f"cd {shlex.quote(str(REPO))}", "set -a && . ./.env && set +a",
                  f"echo \"[$(date +%FT%T)] lane {lane} start\" >> {log}"]
        n = 0
        for name, extra in jobs_for(m, a.math_dataset):
            out = ld / name; out.mkdir(parents=True, exist_ok=True)
            if (out / "summary.json").exists() and not a.force:
                continue
            n += 1
            q = " ".join(shlex.quote(x) for x in build_cmd(m, extra, out, a.max_samples))
            so = shlex.quote(str(out / "stdout.log"))
            script.append(f"( {q} > {so} 2>&1; echo \"[$(date +%FT%T)] END {name} rc=$? "
                          f"$(tail -n 2 {so} | head -n 1)\" >> {log} )" + ("" if m in SERIAL else " &"))
        script += ["wait", f"echo \"[$(date +%FT%T)] lane {lane} done\" >> {log}"]
        sh = ld / "lane.sh"; sh.write_text("#!/bin/bash\n" + "\n".join(script) + "\n"); sh.chmod(0o755)
        print(f"{lane}: {n} evals" + (" (dry-run)" if a.dry_run else ""))
        if a.dry_run or n == 0:
            continue
        sess = f"cotnec_{lane}"
        subprocess.run(["tmux", "kill-session", "-t", sess], stderr=subprocess.DEVNULL)
        subprocess.run(["tmux", "new-session", "-d", "-s", sess, f"bash {shlex.quote(str(sh))}"], check=True)
        print(f"  launched tmux {sess}")


if __name__ == "__main__":
    main()
