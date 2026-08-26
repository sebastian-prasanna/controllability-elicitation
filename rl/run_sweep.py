#!/usr/bin/env python
"""Launch the qwen3-8b GRPO hyperparameter sweep: lr x train_params grid,
each run a full driver+worker pair (rl/run_grpo.py) in its own subprocess.

    python rl/run_sweep.py

Grid: lr in {1e-2, 1e-3, 1e-4} x k in {1k, 10k, 100k}, 25 iterations each,
all other settings from rl/configs/qwen3_8b.yaml. Same seed everywhere ->
identical per-iteration prompt/mode draws across runs (comparable rewards).
Logs to rl/runs/sweep_logs/<name>.log; per-run artifacts in the usual
rl/runs/<run_name>-<ts>/ dirs. Prints a reward-trend summary at the end.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "rl" / "configs" / "qwen3_8b.yaml"
LRS = ["1e-2", "1e-3", "1e-4"]
KS = [1_000, 10_000, 100_000]
ITERATIONS = 25


def main() -> None:
    base = yaml.safe_load(BASE.read_text())
    cfg_dir = ROOT / "rl" / "configs" / "sweep"
    log_dir = ROOT / "rl" / "runs" / "sweep_logs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    procs: list[tuple[str, subprocess.Popen]] = []
    for lr in LRS:
        for k in KS:
            name = f"sweep-qwen8b-lr{lr}-k{k // 1000}k"
            cfg = copy.deepcopy(base)
            cfg["run_name"] = name
            cfg["train"]["lr"] = lr
            cfg["rl"]["iterations"] = ITERATIONS
            cfg["elicitation"]["train_params"] = k
            cfg_path = cfg_dir / f"{name}.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            log = (log_dir / f"{name}.log").open("w")
            procs.append((name, subprocess.Popen(
                [sys.executable, str(ROOT / "rl" / "run_grpo.py"), str(cfg_path)],
                stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
            )))
            print(f"launched {name} (pid {procs[-1][1].pid})", flush=True)
            time.sleep(20)  # stagger engine/worker cold starts

    failures = []
    for name, p in procs:
        rc = p.wait()
        print(f"finished {name}: exit {rc}", flush=True)
        if rc != 0:
            failures.append(name)

    # Reward-trend summary from each run's progress.jsonl.
    print("\n=== sweep summary (reward_mean: first -> last, iters) ===", flush=True)
    for name, _ in procs:
        dirs = sorted((ROOT / "rl" / "runs").glob(f"{name}-*"))
        entries = []
        if dirs and (dirs[-1] / "progress.jsonl").exists():
            entries = [json.loads(line) for line in
                       (dirs[-1] / "progress.jsonl").read_text().splitlines() if line.strip()]
        if entries:
            r0, rn = entries[0]["reward_mean"], entries[-1]["reward_mean"]
            tail = sum(e["reward_mean"] for e in entries[-5:]) / len(entries[-5:])
            print(f"{name}: {r0:.4f} -> {rn:.4f} (last5 avg {tail:.4f}, "
                  f"{len(entries)} iters)", flush=True)
        else:
            print(f"{name}: no progress recorded", flush=True)
    if failures:
        print(f"\nFAILED runs: {failures}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
