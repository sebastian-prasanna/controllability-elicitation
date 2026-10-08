#!/usr/bin/env python3
"""Launch GEPA x few-shot hybrid evals (2026-09-23).

Same pinned settings as pinned_reeval (configs/eval_pins.json) so results are
directly comparable to the baseline / GEPA / few-shot arms there.

Arms:
  hybrid_plain -- GEPA s1 prompt + few-shot k1_s2 demos, verbatim
  hybrid_clean -- same, but the 2 constraint-narrating demos swapped out

Usage:
  python hybrid_prompt/launch.py [--dry-run] [--arms hybrid_plain,hybrid_clean]
                                 [--max-samples N] [--suffix smoke]
"""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUN_EVAL = REPO / "scripts" / "run_eval.py"
RUNS = REPO / "hybrid_prompt" / "runs"
PROMPTS = REPO / "hybrid_prompt" / "prompts"
PINS = json.load(open(REPO / "configs" / "eval_pins.json"))

MODEL_ID = "openai/gpt-oss-20b"
LABEL = "gptoss20b"
ARMS = {"hybrid_plain": PROMPTS / f"{LABEL}_hybrid_plain.txt",
        "hybrid_clean": PROMPTS / f"{LABEL}_hybrid_clean.txt"}
SPLITS = {"test": [], "heldout": ["--heldout"]}


def build_cmd(run_name: str, prompt: Path, extra: list[str], max_samples: int | None, out: Path):
    c = PINS[MODEL_ID]
    cmd = [sys.executable, str(RUN_EVAL), "--model", MODEL_ID, "--split", "test", "--mode", "all",
           "--temperature", str(c["temperature"]), "--top-p", str(c["top_p"]),
           "--max-tokens", str(c["max_tokens"]), "--concurrency", str(c["concurrency"]),
           "--max-retries", str(c["max_retries"]), "--provider", c["provider"],
           "--meta-discussion", "compliant", "--system-prompt", str(prompt),
           "--out-dir", str(out)]
    if c.get("quantizations"):
        cmd += ["--quantization", c["quantizations"][0]]
    if c.get("reasoning_effort"):
        cmd += ["--reasoning-effort", c["reasoning_effort"]]
    if max_samples:
        cmd += ["--max-samples", str(max_samples)]
    return cmd + extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    lane = f"{LABEL}{'_' + a.suffix if a.suffix else ''}"
    lane_dir = RUNS / lane
    lane_dir.mkdir(parents=True, exist_ok=True)
    log = lane_dir / "driver.log"
    script = [f"cd {shlex.quote(str(REPO))}", "set -a && . ./.env && set +a",
              f'echo "[$(date +%FT%T)] lane {lane} start" >> {shlex.quote(str(log))}']
    n = 0
    for arm in a.arms.split(","):
        prompt = ARMS[arm]
        if not prompt.is_file():
            raise FileNotFoundError(prompt)
        for sp in a.splits.split(","):
            run_name = f"{arm}_{sp}"
            out = lane_dir / run_name
            out.mkdir(parents=True, exist_ok=True)
            if (out / "summary.json").exists() and not a.force:
                continue
            n += 1
            q = " ".join(shlex.quote(x) for x in
                         build_cmd(run_name, prompt, SPLITS[sp], a.max_samples, out))
            script += [f'echo "[$(date +%FT%T)] START {run_name}" >> {shlex.quote(str(log))}',
                       f"{q} > {shlex.quote(str(out / 'stdout.log'))} 2>&1; "
                       f'echo "[$(date +%FT%T)] END   {run_name} rc=$? '
                       f'$(tail -n 2 {shlex.quote(str(out / "stdout.log"))} | head -n 1)" >> {shlex.quote(str(log))}']
    script.append(f'echo "[$(date +%FT%T)] lane {lane} done" >> {shlex.quote(str(log))}')
    sh = lane_dir / "lane.sh"
    sh.write_text("#!/bin/bash\n" + "\n".join(script) + "\n")
    sh.chmod(0o755)
    print(f"{lane}: {n} evals" + (" (dry-run)" if a.dry_run else ""))
    if a.dry_run or n == 0:
        return
    sess = f"hyb_{lane}"
    subprocess.run(["tmux", "kill-session", "-t", sess], stderr=subprocess.DEVNULL)
    subprocess.run(["tmux", "new-session", "-d", "-s", sess, f"bash {shlex.quote(str(sh))}"], check=True)
    print(f"  launched tmux session {sess}")


if __name__ == "__main__":
    main()
