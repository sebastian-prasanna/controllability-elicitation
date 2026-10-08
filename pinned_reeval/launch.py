#!/usr/bin/env python3
"""Pinned re-evaluation of the final prompted-experiment evals (2026-09-22).

For each of the 8 sweep models, using the fixed provider/temperature/effort settings in
configs/eval_pins.json, re-runs on the canonical test split:
  - baseline (empty system prompt)                      -> test (9 modes) + heldout (3 modes)
  - GEPA general arm, 3 seeds (initial_sweep, second_sweep s1, s2) best_prompt.txt -> test + heldout
  - few-shot k=1, 3 seeds (fewshot/final_prompts/<m>/k1_s{1,2,3}.txt)             -> test + heldout
All evals of one model run SEQUENTIALLY in one tmux session (so the provider sees exactly the
configured concurrency); the 8 models run in parallel. Evals whose summary.json exists are skipped,
so relaunching resumes. Outputs: pinned_reeval/runs/<model_label>/<arm>_<seed>_<split>/ (eval JSON with
every rollout + judge output, progress.jsonl, summary.json, stdout.log) and driver.log per model.

Usage:
  python pinned_reeval/launch.py [--dry-run] [--models kimik3,glm53] [--arms baseline,gepa,fewshot]
                                 [--max-samples N] [--suffix smoke] [--temperature 1.0]
  T=1 replica (2026-10-06, 5 models whose provider honours temperature; Kimi/GLM run fixed ~1.0 anyway):
  python pinned_reeval/launch.py --models gptoss20b,gptoss120b,qwen8b,qwen32b,dsv4pro --suffix t1 --temperature 1.0
  python pinned_reeval/status.py
"""
from __future__ import annotations
import argparse, json, shlex, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PINS = json.load(open(REPO / "configs" / "eval_pins.json"))
RUNS = REPO / "pinned_reeval" / "runs"
RUN_EVAL = REPO / "scripts" / "run_eval.py"

MODELS = {  # label -> openrouter id
    "gptoss20b": "openai/gpt-oss-20b", "gptoss120b": "openai/gpt-oss-120b",
    "qwen8b": "qwen/qwen3-8b", "qwen32b": "qwen/qwen3-32b",
    "kimik3": "moonshotai/kimi-k3", "dsv4pro": "deepseek/deepseek-v4-pro-0813",
    "glm53": "z-ai/glm-5.3", "glm53flash": "z-ai/glm-5.3-flash",
}
GEPA_SEEDS = {  # seed label -> run dir (general arm)
    "s0": "gepa/runs/initial_sweep/{m}_general",
    "s1": "gepa/runs/second_sweep/{m}_general_s1",
    "s2": "gepa/runs/second_sweep/{m}_general_s2",
}
SPLITS = {"test": [], "heldout": ["--heldout"]}
# --split val (2026-10-07, seed selection on val): same jobs on the canonical VAL split; the in-distribution block is then
# named <arm>_<seed>_indist (not _test) so the run names say what they are. Lane suffix e.g. t1val -> runs/<m>_t1val/.


def jobs_for(label: str, arms: list[str], split: str = "test") -> list[tuple[str, list[str]]]:
    """-> [(run_name, extra run_eval flags)]"""
    out = []
    name = {"test": "test" if split == "test" else "indist", "heldout": "heldout"}
    if "baseline" in arms:
        for sp, fl in SPLITS.items():
            out.append((f"baseline_{name[sp]}", ["--system-prompt", ""] + fl))
    if "gepa" in arms:
        for seed, tmpl in GEPA_SEEDS.items():
            prompt = REPO / tmpl.format(m=label) / "best_prompt.txt"
            if not prompt.is_file():
                raise FileNotFoundError(prompt)
            for sp, fl in SPLITS.items():
                out.append((f"gepa_{seed}_{name[sp]}", ["--system-prompt", str(prompt)] + fl))
    if "fewshot" in arms:
        for s in (1, 2, 3):
            prompt = REPO / "fewshot" / "final_prompts" / label / f"k1_s{s}.txt"
            if not prompt.is_file():
                raise FileNotFoundError(prompt)
            for sp, fl in SPLITS.items():
                out.append((f"fewshot_s{s}_{name[sp]}", ["--system-prompt", str(prompt)] + fl))
    return out


def build_cmd(label: str, run_name: str, extra: list[str], max_samples: int | None,
              temperature: float | None = None, split: str = "test") -> list[str]:
    mid = MODELS[label]; c = PINS[mid]
    temp = c["temperature"] if temperature is None else temperature
    cmd = [sys.executable, str(RUN_EVAL), "--model", mid, "--split", split, "--mode", "all",
           "--temperature", str(temp), "--top-p", str(c["top_p"]),
           "--max-tokens", str(c["max_tokens"]), "--concurrency", str(c["concurrency"]),
           "--max-retries", str(c["max_retries"]), "--provider", c["provider"],
           "--meta-discussion", "compliant", "--out-dir", str(RUNS / label / run_name)]
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
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--arms", default="baseline,gepa,fewshot")
    ap.add_argument("--max-samples", type=int, default=None, help="smoke: cap questions per eval")
    ap.add_argument("--suffix", default="", help="run-dir suffix, e.g. 'smoke' -> runs/<m>_smoke/")
    ap.add_argument("--force", action="store_true", help="rerun even if summary.json exists")
    ap.add_argument("--temperature", type=float, default=None,
                    help="override the pinned temperature (e.g. 1.0 with --suffix t1); all other pins unchanged")
    ap.add_argument("--split", default="test", choices=["test", "val"], help="canonical split (val lanes name the in-dist block _indist)")
    ap.add_argument("--only", default=None,
                    help="comma-separated run names (e.g. fewshot_s1_test,fewshot_s1_heldout) to run in this lane; "
                         "with --session, several lanes can split one model's evals (same lane dir)")
    ap.add_argument("--session", default=None,
                    help="tmux session name + script suffix (default pr_<lane>, lane.sh); use with --only")
    a = ap.parse_args()
    arms = a.arms.split(",")
    for label in a.models.split(","):
        lane = f"{label}{'_' + a.suffix if a.suffix else ''}"
        lane_dir = RUNS / lane; lane_dir.mkdir(parents=True, exist_ok=True)
        script = [f"cd {shlex.quote(str(REPO))}", "set -a && . ./.env && set +a",
                  f"echo \"[$(date +%FT%T)] lane {lane} start\" >> {shlex.quote(str(lane_dir / 'driver.log'))}"]
        n = 0
        only = set(a.only.split(",")) if a.only else None
        for run_name, extra in jobs_for(label, arms, a.split):
            if only is not None and run_name not in only:
                continue
            cmd = build_cmd(label, run_name, extra, a.max_samples, a.temperature, a.split)
            cmd[cmd.index("--out-dir") + 1] = str(lane_dir / run_name)
            out = lane_dir / run_name; out.mkdir(parents=True, exist_ok=True)
            if (out / "summary.json").exists() and not a.force:
                continue
            n += 1
            q = " ".join(shlex.quote(x) for x in cmd)
            script += [f"echo \"[$(date +%FT%T)] START {run_name}\" >> {shlex.quote(str(lane_dir / 'driver.log'))}",
                       f"{q} > {shlex.quote(str(out / 'stdout.log'))} 2>&1; "
                       f"echo \"[$(date +%FT%T)] END   {run_name} rc=$? $(tail -n 2 {shlex.quote(str(out / 'stdout.log'))} | head -n 1)\" >> {shlex.quote(str(lane_dir / 'driver.log'))}"]
        script.append(f"echo \"[$(date +%FT%T)] lane {lane} done\" >> {shlex.quote(str(lane_dir / 'driver.log'))}")
        sh = lane_dir / (f"lane_{a.session}.sh" if a.session else "lane.sh")
        sh.write_text("#!/bin/bash\n" + "\n".join(script) + "\n"); sh.chmod(0o755)
        print(f"{lane}: {n} evals to run" + (" (dry-run)" if a.dry_run else ""))
        if a.dry_run or n == 0:
            continue
        sess = a.session or f"pr_{lane}"
        subprocess.run(["tmux", "kill-session", "-t", sess], stderr=subprocess.DEVNULL)
        subprocess.run(["tmux", "new-session", "-d", "-s", sess, f"bash {shlex.quote(str(sh))}"], check=True)
        print(f"  launched tmux session {sess}")


if __name__ == "__main__":
    main()
