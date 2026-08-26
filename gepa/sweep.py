"""Sweep launcher for GEPA runs: cartesian product of CLI-flag overrides → tmux windows.

Adapted from subliminal-learning-modal's scripts/sweep.py, but GEPA is driven by
CLI flags rather than config files: each axis is a list of dicts whose keys are
run_gepa.py flags. The `_label` key is stripped and joined across axes to form
the run name. Boolean True means "pass the bare flag" (e.g. --general-advice-only).

The cartesian product across all axes determines the runs. A single-element axis
acts as a global override applied to every run.

Runs land in gepa/runs/<SWEEP_NAME>/<run_name>/, with stdout teed to stdout.log
inside each run dir.

Usage:
    1. Edit SWEEP_NAME, BASE_FLAGS and SWEEP_AXES below.
    2. Run: python gepa/sweep.py [--dry-run] [--delay SECONDS] [--resume]

--resume appends --resume to every run (skips finished iterations via
iterations.jsonl), so re-running the sweep after a partial failure is safe.
"""

from __future__ import annotations

import argparse
import itertools
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_SCRIPT = REPO_ROOT / "gepa" / "run_gepa.py"
RUNS_DIR = REPO_ROOT / "gepa" / "runs"

# ---------------------------------------------------------------------------
# Configuration — edit these for each sweep.
# ---------------------------------------------------------------------------

SWEEP_NAME = "initial_sweep"

# Flags applied to every run (any run_gepa.py flag). Axes override these.
BASE_FLAGS: dict[str, Any] = {
    "--train": "all",  # combined hle+gpqa+mmlu_pro train split
    "--objective": "compliance",
    # Best above-length-trend teacher in the gptoss_refl_* sweep: compliance
    # gains without shortening the reasoning (see gepa/analysis/).
    "--reflection-model": "anthropic/claude-fable-5",
    # Accept/reject noise scales 1/sqrt(n); n=64 scores are decoupled from the
    # reflection prompt, which sees a stride-sampled cap of 32 rollouts.
    "--minibatch": 64,
    "--reflection-cap": 32,
    # Pareto fold picks the final test-eval candidate; n=128 tightens that
    # selection (val split has 200 questions, so headroom remains).
    "--pareto-size": 128,
    # Per-run OpenRouter concurrency cap defaults to 200 (run_gepa.py); the
    # client retries 429s with backoff, so oversubscription self-throttles.
    # Uncomment to lower if the sweep starves other work on the same key.
    # "--max-concurrency": 64,
}

MODELS = [
    ("gptoss20b", "openai/gpt-oss-20b"),
    ("gptoss120b", "openai/gpt-oss-120b"),
    ("qwen8b", "qwen/qwen3-8b"),
    ("qwen32b", "qwen/qwen3-32b"),
    ("glm53", "z-ai/glm-5.3"),
    ("kimik3", "moonshotai/kimi-k3"),
    ("dsv4pro", "deepseek/deepseek-v4-pro-0813"),
    ("oxalpha", "stealth/ox-alpha"),
]

SWEEP_AXES: list[list[dict[str, Any]]] = [
    # Axis 1: task model.
    [{"_label": label, "--task-model": model} for label, model in MODELS],

    # Axis 2: mode-specific advice allowed in the reflected prompt vs. not.
    [
        {"_label": "free"},
        {"_label": "general", "--general-advice-only": True},
    ],
]

# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------

TMUX_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")


def tmux_safe(name: str) -> str:
    return TMUX_NAME_RE.sub("_", name)


def make_run_name(combo: tuple[dict, ...]) -> str:
    labels = [d["_label"] for d in combo if "_label" in d]
    return "_".join(labels) if labels else "run"


def flags_to_argv(flags: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for k, v in flags.items():
        if v is True:
            argv.append(k)
        elif v is False or v is None:
            continue
        else:
            argv.extend([k, str(v)])
    return argv


def build_runs(resume: bool) -> list[tuple[str, list[str]]]:
    """Return (run_name, argv) per combination."""
    runs: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for idx, combo in enumerate(itertools.product(*SWEEP_AXES)):
        flags = dict(BASE_FLAGS)
        for d in combo:
            flags.update({k: v for k, v in d.items() if k != "_label"})
        run_name = make_run_name(combo)
        if run_name in seen:
            run_name = f"{run_name}_{idx}"
        seen.add(run_name)
        flags["--run-name"] = f"{SWEEP_NAME}/{run_name}"
        if resume:
            flags["--resume"] = True
        runs.append((run_name, flags_to_argv(flags)))
    return runs


def launch_tmux(runs: list[tuple[str, list[str]]], delay: int) -> None:
    session = tmux_safe(SWEEP_NAME)
    subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)

    # Snapshot this script into the sweep dir so every run records the exact
    # version (CONFIG block included) that launched it.
    sweep_dir = RUNS_DIR / SWEEP_NAME
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "sweep.py").write_text(Path(__file__).read_text())

    for i, (run_name, argv) in enumerate(runs):
        window = tmux_safe(run_name)
        run_dir = RUNS_DIR / SWEEP_NAME / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        sleep_prefix = f"sleep {i * delay} && " if delay > 0 and i > 0 else ""
        inner = " ".join([sys.executable, str(RUN_SCRIPT), *argv])
        cmd = f"cd {REPO_ROOT} && {sleep_prefix}{inner} 2>&1 | tee {run_dir}/stdout.log"
        if i == 0:
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", session, "-n", window, "-c", str(REPO_ROOT)],
                check=True,
            )
        else:
            subprocess.run(
                ["tmux", "new-window", "-t", session, "-n", window, "-c", str(REPO_ROOT)],
                check=True,
            )
        subprocess.run(["tmux", "send-keys", "-t", f"{session}:{window}", cmd, "Enter"], check=True)

    print(f"\nLaunched {len(runs)} windows in tmux session '{session}'.")
    print(f"Attach with: tmux attach -t {session}")
    print("Navigate: Ctrl-B n (next), Ctrl-B p (prev), Ctrl-B w (list).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GEPA sweep commands and launch in tmux.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands but don't launch tmux.")
    parser.add_argument("--delay", type=int, default=30,
                        help="Stagger run i by i*delay seconds (default 30; 0 disables). "
                             "Smooths the initial burst of OpenRouter requests.")
    parser.add_argument("--resume", action="store_true",
                        help="Append --resume to every run (safe re-launch after partial failure).")
    args = parser.parse_args()

    runs = build_runs(resume=args.resume)
    print(f"Sweep '{SWEEP_NAME}': {len(SWEEP_AXES)} axes, {len(runs)} runs")
    for run_name, argv in runs:
        print(f"  {run_name}: {' '.join(argv)}")

    if args.dry_run:
        print("\n(dry run — not launching)")
        return

    launch_tmux(runs, delay=args.delay)


if __name__ == "__main__":
    main()
